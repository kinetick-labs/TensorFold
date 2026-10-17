// Cooperative prompt experts (bit-identical to prompt_body): a CTA of WARPS warps takes one item's 16*T-pair row
// tile and WARPS column blocks; the tile's inputs go through shared memory once (cp.async, S stages of one 32-input
// group, zero rows past the item), each warp reads its A fragments there. Weights, mma, fmaf and epilogue as before.
namespace tf_fn_experts_coop {
using namespace tf_fn_experts_prompt;

__device__ __forceinline__ void cp16(void* smem, const void* g, bool valid) {
  const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(g), "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

constexpr int ROWB = 80;  // bytes a row of one group in shared memory (64 + 16 pad)

template <int M, int EPI, int T, int WARPS, int S>
__device__ __forceinline__ void coop_body(const __nv_bfloat16* __restrict__ X, int x_stride, int slots,
                                          const uint4* __restrict__ W, const float* __restrict__ scale, int KG, int NB,
                                          const int* __restrict__ items, const int* __restrict__ counts,
                                          const int* __restrict__ members, void* __restrict__ out, int N, float limit,
                                          int skip) {
  constexpr int ROWS = 16 * T;
  constexpr int MS = M;
  extern __shared__ __align__(16) unsigned char sm[];   // [S][ROWS][ROWB]
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, gq = lane >> 2, t = lane & 3;
  const int CBG = (NB + WARPS - 1) / WARPS;
  const int units = __ldg(counts) * CBG;
  // loader: thread -> (row, 16-byte chunk); ROWS * 4 chunks a group
  constexpr int CHUNKS = ROWS * 4;
  for (int unit = blockIdx.x; unit < units; unit += gridDim.x) {
    const int it = unit / CBG, cbg = unit - it * CBG;
    const int e = __ldg(items + 3 * it), first = __ldg(items + 3 * it + 1), cnt = __ldg(items + 3 * it + 2);
    if (e == skip) continue;
    const int cb = cbg * WARPS + warp;
    const bool active = cb < NB;
    const uint4* blk = W + ((size_t)e * NB + (active ? cb : 0)) * (size_t)KG * (MS * BLOCK4);
    for (int p0 = 0; p0 < cnt; p0 += ROWS) {
      // this thread's loader rows
      const __nv_bfloat16* lsrc[(CHUNKS + 32 * WARPS - 1) / (32 * WARPS)];
      bool lval[(CHUNKS + 32 * WARPS - 1) / (32 * WARPS)];
#pragma unroll
      for (int c = 0; c < (CHUNKS + 32 * WARPS - 1) / (32 * WARPS); ++c) {
        const int ch = threadIdx.x + c * 32 * WARPS;
        const int row = ch >> 2;
        lval[c] = ch < CHUNKS && p0 + row < cnt;
        int pr = lval[c] ? __ldg(members + first + p0 + row) : 0;
        const int xr = slots ? pr / slots : pr;
        lsrc[c] = X + (size_t)xr * x_stride + (ch & 3) * 8;
      }
      auto load_x = [&](int g, int slot) {
#pragma unroll
        for (int c = 0; c < (CHUNKS + 32 * WARPS - 1) / (32 * WARPS); ++c) {
          const int ch = threadIdx.x + c * 32 * WARPS;
          if (ch < CHUNKS) cp16(sm + (slot * ROWS + (ch >> 2)) * ROWB + (ch & 3) * 16, lsrc[c] + g * 32, lval[c]);
        }
      };
      bool v0[T], v1[T];
      int pr0[T], pr1[T];
#pragma unroll
      for (int r = 0; r < T; ++r) {
        const int r0 = p0 + 16 * r;
        v0[r] = r0 + gq < cnt;
        v1[r] = r0 + gq + 8 < cnt;
        pr0[r] = v0[r] ? __ldg(members + first + r0 + gq) : 0;
        pr1[r] = v1[r] ? __ldg(members + first + r0 + gq + 8) : 0;
      }
      float acc[T][M][1][NTW][4];
#pragma unroll
      for (int r = 0; r < T; ++r)
#pragma unroll
        for (int m = 0; m < M; ++m)
#pragma unroll
          for (int j = 0; j < NTW; ++j) acc[r][m][0][j][0] = acc[r][m][0][j][1] = acc[r][m][0][j][2] = acc[r][m][0][j][3] = 0.f;
      __syncthreads();   // the previous pass's readers are done with the slots
#pragma unroll
      for (int d = 0; d < S - 1; ++d) {
        if (d < KG) load_x(d, d);
        cp_commit();
      }
      uint4 w[M], s[M], wn[M], sn[M];
      if (active) {
#pragma unroll
        for (int m = 0; m < M; ++m) { w[m] = ld_nc(blk + m * BLOCK4 + lane); s[m] = ld_nc(blk + m * BLOCK4 + 32 + t); }
      }
      for (int g = 0; g < KG; ++g) {
        if (g + S - 1 < KG) load_x(g + S - 1, (g + S - 1) % S);
        cp_commit();
        if (active && g + 1 < KG) {
          const uint4* b = blk + (size_t)(g + 1) * (MS * BLOCK4);
#pragma unroll
          for (int m = 0; m < M; ++m) { wn[m] = ld_nc(b + m * BLOCK4 + lane); sn[m] = ld_nc(b + m * BLOCK4 + 32 + t); }
        }
        cp_wait<S - 1>();
        __syncthreads();
        if (active) {
          const unsigned char* xs = sm + (g % S) * ROWS * ROWB;
#pragma unroll
          for (int m = 0; m < M; ++m)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
              uint4 xq[T];
#pragma unroll
              for (int r = 0; r < T; ++r) {
                const uint2 xa = *reinterpret_cast<const uint2*>(xs + (16 * r + gq) * ROWB + (4 * t + 16 * h) * 2);
                const uint2 xb = *reinterpret_cast<const uint2*>(xs + (16 * r + gq + 8) * ROWB + (4 * t + 16 * h) * 2);
                xq[r] = make_uint4(xa.x, xb.x, xa.y, xb.y);
              }
#pragma unroll
              for (int j = 0; j < NTW; ++j) {
                const uint32_t word = comp(w[m], j);
                const uint32_t b0 = fp4pair(word, 8 * h), b1 = fp4pair(word, 8 * h + 4);
                const uint32_t sw = comp(s[m], 2 * h + (j >> 1));
                const int sh = (j & 1) * 16;
                const float s0 = e4m3f((sw >> sh) & 0xFFu), s1 = e4m3f((sw >> (sh + 8)) & 0xFFu);
#pragma unroll
                for (int r = 0; r < T; ++r) {
                  float p[4];
                  mma0(p, xq[r].x, xq[r].y, xq[r].z, xq[r].w, b0, b1);
                  float(&a)[4] = acc[r][m][0][j];
                  a[0] = fmaf(p[0], s0, a[0]);
                  a[1] = fmaf(p[1], s1, a[1]);
                  if (v1[r]) {
                    a[2] = fmaf(p[2], s0, a[2]);
                    a[3] = fmaf(p[3], s1, a[3]);
                  }
                }
              }
            }
#pragma unroll
          for (int m = 0; m < M; ++m) { w[m] = wn[m]; s[m] = sn[m]; }
        }
        __syncthreads();   // slot g % S free for the load at g + S
      }
      cp_wait<0>();
      if (active) {
#pragma unroll
        for (int r = 0; r < T; ++r) {
#pragma unroll
          for (int m = 0; m < M; ++m) {
            const float gs = __ldg(scale + e * MS + m);
#pragma unroll
            for (int j = 0; j < NTW; ++j)
#pragma unroll
              for (int q = 0; q < 4; ++q) acc[r][m][0][j][q] *= gs;
          }
          epilogue<EPI, M, 1>(acc[r], 0, out, N, cb * COLS + 2 * t, pr0[r], pr1[r], v0[r], v1[r], limit);
        }
      }
    }
  }
}
}  // namespace tf_fn_experts_coop

#define TF_COOP(NAME, M, EPI, T, WARPS, S, MINB) \
  extern "C" __global__ void __launch_bounds__(32 * WARPS MINB) \
      NAME(const __nv_bfloat16* __restrict__ X, int x_stride, int slots, const uint4* __restrict__ W, \
           const float* __restrict__ scale, int KG, int NB, const int* __restrict__ items, \
           const int* __restrict__ counts, const int* __restrict__ members, void* __restrict__ out, int N, \
           float limit, int skip) { \
    tf_fn_experts_coop::coop_body<M, EPI, T, WARPS, S>(X, x_stride, slots, W, scale, KG, NB, items, counts, members, \
                                                       out, N, limit, skip); \
  }
TF_COOP(fn_coop_gu_t2_w4, 2, 2, 2, 4, 4, )
TF_COOP(fn_coop_gu_t1_w4, 2, 2, 1, 4, 4, )
TF_COOP(fn_coop_gu_t2_w5, 2, 2, 2, 5, 4, )
TF_COOP(fn_coop_f32_t2_w8, 1, 0, 2, 8, 4, )
TF_COOP(fn_coop_f32_t1_w8, 1, 0, 1, 8, 4, )
