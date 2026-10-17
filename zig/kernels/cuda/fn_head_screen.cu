// Flash Next's exact two-stage greedy lm_head (ours; TF_FLASHNEXT_HEAD_SCREEN): an MXFP8 copy's logits A [n, V] f32
// (fn_qmmf, launched by cuda_head_screen.zig), a rigorous bound B on |out - A| per (row, id), where out is the bf16 head's
// logit (Triton _b16mm BM16 BN64 BK64 SK=1: one in-order m16n8k16 chain over K per element, then RNE to bf16):
//   B0 = sum_b ||x_b||2 ||W_vb - Q_vb||2  (32-input blocks, Cauchy-Schwarz per block; dnb is stored rounded up)
//      + ||x||2 * wq_v                    (wq_v = g (||W_v|| + ||Q_v||): both kernels' fp32 accumulation, g = 1e-3)
//   B  = 1.01 (B0 + 2^-9 (|A| + B0))      (the bf16 rounding's half ulp, and this kernel's own f32 rounding)
// Each row's T = max_v (A - B) (an id whose out >= T exists); an id whose A + B < T on every row has out < T on every
// row, so it is no row's argmax (nor tied): its logits are set to -inf, every other id's recomputed exactly
// (fn_head_exact: the same chain as _b16mm, bit-identical, mb7/hx.cu) for all rows. The argmax over the result is the
// argmax over the full head. Non-finite A forces a candidate.
#include <cuda_bf16.h>
#include <stdint.h>

namespace {
constexpr int KB_MAX = 128;  // 32-input blocks (K <= 4096)
constexpr int R_MAX = 16;
__device__ __forceinline__ int enc(float f) { const int i = __float_as_int(f); return i >= 0 ? i : i ^ 0x7FFFFFFF; }
__device__ __forceinline__ float dec(int i) { return __int_as_float(i >= 0 ? i : i ^ 0x7FFFFFFF); }
__device__ __forceinline__ float bf(uint16_t v) { return __uint_as_float(static_cast<uint32_t>(v) << 16); }
__device__ __forceinline__ void mma16816(float* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
}  // namespace

// tmax[r] = enc(-inf), count = 0
extern "C" __global__ void fn_head_init(int* tmax, int* count, int n) {
  if (threadIdx.x < n) tmax[threadIdx.x] = enc(-__int_as_float(0x7f800000));
  if (threadIdx.x == 0) *count = 0;
}

// A [n, V] f32 -> U = A + B in place; tmax[r] = max enc(A - B). dnb [KB, V] f32 (block-major), wq [V] f32, x [n, ldx] bf16.
extern "C" __global__ void __launch_bounds__(256) fn_head_bound(float* __restrict__ a, const uint16_t* __restrict__ x, int ldx,
                                                              const float* __restrict__ dnb, const float* __restrict__ wq,
                                                              int* __restrict__ tmax, int n, int V, int K) {
  __shared__ float xb[R_MAX][KB_MAX];
  __shared__ float xn[R_MAX];
  __shared__ float red[R_MAX][8];
  const int KB = K / 32;
  for (int i = threadIdx.x; i < n * KB; i += blockDim.x) {
    const int r = i / KB, b = i % KB;
    const uint16_t* p = x + (size_t)r * ldx + b * 32;
    double s = 0;
    for (int j = 0; j < 32; ++j) { const double v = bf(p[j]); s += v * v; }
    xb[r][b] = (float)(sqrt(s) * (1.0 + 1e-6));
  }
  __syncthreads();
  if (threadIdx.x < n) {
    double s = 0;
    for (int b = 0; b < KB; ++b) s += (double)xb[threadIdx.x][b] * xb[threadIdx.x][b];
    xn[threadIdx.x] = (float)(sqrt(s) * (1.0 + 1e-6));
  }
  __syncthreads();
  float lmax[R_MAX];
  for (int r = 0; r < R_MAX; ++r) lmax[r] = -__int_as_float(0x7f800000);
  for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < V; v += gridDim.x * blockDim.x) {
    float b0[R_MAX];
    for (int r = 0; r < R_MAX; ++r) b0[r] = 0.f;
    for (int b = 0; b < KB; ++b) {
      const float d = __ldg(dnb + (size_t)b * V + v);
#pragma unroll
      for (int r = 0; r < R_MAX; ++r) if (r < n) b0[r] = fmaf(xb[r][b], d, b0[r]);
    }
    const float w = __ldg(wq + v);
#pragma unroll
    for (int r = 0; r < R_MAX; ++r) if (r < n) {
      float* ap = a + (size_t)r * V + v;
      const float av = *ap;
      float bb = fmaf(xn[r], w, b0[r]);
      bb = 1.01f * (bb + ldexpf(fabsf(av) + bb, -9));
      if (!isfinite(av) || !isfinite(bb)) { *ap = __int_as_float(0x7f800000); continue; }
      *ap = av + bb;
      lmax[r] = fmaxf(lmax[r], av - bb);
    }
  }
  // block max per row, then one atomic
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  for (int r = 0; r < n; ++r) {
    float m = lmax[r];
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if (lane == 0) red[r][warp] = m;
  }
  __syncthreads();
  if (threadIdx.x < n) {
    float m = red[threadIdx.x][0];
    for (int w8 = 1; w8 < (int)(blockDim.x >> 5); ++w8) m = fmaxf(m, red[threadIdx.x][w8]);
    atomicMax(tmax + threadIdx.x, enc(m));
  }
}

// per id: candidate when some row's U >= T (or U is NaN); else every row's logit -inf; candidates listed (any order).
extern "C" __global__ void __launch_bounds__(256) fn_head_select(const float* __restrict__ u, const int* __restrict__ tmax,
                                                               uint16_t* __restrict__ logits, int* __restrict__ list,
                                                               int* __restrict__ count, int n, int V) {
  __shared__ float t[R_MAX];
  if (threadIdx.x < n) t[threadIdx.x] = dec(tmax[threadIdx.x]);
  __syncthreads();
  for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < V; v += gridDim.x * blockDim.x) {
    bool c = false;
    for (int r = 0; r < n; ++r) c |= !(u[(size_t)r * V + v] < t[r]);
    if (c) list[atomicAdd(count, 1)] = v;
    else for (int r = 0; r < n; ++r) logits[(size_t)r * V + v] = 0xFF80;  // bf16 -inf
  }
}

// the listed ids' logits for rows < n (<= 16): _b16mm's chain (K in k16 steps, in order), RNE to bf16. A warp takes 8
// listed ids a step, grid-stride over the list.
extern "C" __global__ void __launch_bounds__(128) fn_head_exact(const uint16_t* __restrict__ xh, int ldx, const uint16_t* __restrict__ wh,
                                                              const int* __restrict__ list, const int* __restrict__ count,
                                                              uint16_t* __restrict__ out, int V, int K, int n) {
  const int lane = threadIdx.x & 31, g = lane >> 2, c = lane & 3;
  const int nids = *count;
  const int warps = gridDim.x * (blockDim.x >> 5);
  const uint32_t* x0 = reinterpret_cast<const uint32_t*>(xh + (size_t)(g < n ? g : 0) * ldx);
  const uint32_t* x1 = reinterpret_cast<const uint32_t*>(xh + (size_t)(g + 8 < n ? g + 8 : 0) * ldx);
  for (int n0 = (blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) * 8; n0 < nids; n0 += warps * 8) {
    const int my = n0 + g < nids ? list[n0 + g] : -1;
    const uint32_t* wr = reinterpret_cast<const uint32_t*>(wh + (size_t)(my >= 0 ? my : 0) * K);
    float d[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 1
    for (int s0 = 0; s0 < K / 16; s0 += 8) {
      uint32_t b[8][2], a[8][4];
#pragma unroll
      for (int q = 0; q < 8; ++q) {
        const int s = s0 + q;
        b[q][0] = my >= 0 ? __ldg(wr + s * 8 + c) : 0u;
        b[q][1] = my >= 0 ? __ldg(wr + s * 8 + 4 + c) : 0u;
        a[q][0] = g < n ? __ldg(x0 + s * 8 + c) : 0u;
        a[q][1] = g + 8 < n ? __ldg(x1 + s * 8 + c) : 0u;
        a[q][2] = g < n ? __ldg(x0 + s * 8 + 4 + c) : 0u;
        a[q][3] = g + 8 < n ? __ldg(x1 + s * 8 + 4 + c) : 0u;
      }
#pragma unroll
      for (int q = 0; q < 8; ++q) mma16816(d, a[q], b[q][0], b[q][1]);
    }
    for (int e = 0; e < 2; ++e) {
      const int j = n0 + 2 * c + e;
      if (j >= nids) continue;
      const int v = list[j];
      const __nv_bfloat16 lo = __float2bfloat16_rn(d[e]), hi = __float2bfloat16_rn(d[2 + e]);
      if (g < n) out[(size_t)g * V + v] = *reinterpret_cast<const uint16_t*>(&lo);
      if (g + 8 < n) out[(size_t)(g + 8) * V + v] = *reinterpret_cast<const uint16_t*>(&hi);
    }
  }
}
