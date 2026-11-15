// GLM prompt chunks' MLA absorb / expand on tensor cores: per head, C [rows][N] = A [rows][K] . B, bf16 in, fp32 sums,
// bf16 out. A rows (r * heads + h) * K, C rows (r * heads + h) * N; B [heads][K][N] (absorb's wk) or, BNK, [heads][N][K]
// (expand's wv). Block: 64 rows x 64 columns of one head, 4 warps of 32 x 32, K in 32-wide steps.
#pragma once
#include <cstdint>
#include <cuda_bf16.h>

namespace glm_pf_mla {

__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], const void* p) {
    const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void ldsm4t(uint32_t (&r)[4], const void* p) {
    const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void mma_bf16(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

template <int K, int N, bool BNK>
__device__ __forceinline__ void head_gemm(const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B,
                                          __nv_bfloat16* __restrict__ C, int64_t rows, int64_t heads) {
    constexpr int BM = 64, BN = 64, BK = 32, AS = BK + 8, BS_NK = BK + 8, BS_KN = BN + 8;
    __shared__ __align__(16) __nv_bfloat16 As[2][BM * AS];
    __shared__ __align__(16) __nv_bfloat16 Bs[2][BNK ? BN * BS_NK : BK * BS_KN];
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31, wm = warp >> 1, wn = warp & 1;
    const int64_t h = blockIdx.z, n0 = (int64_t)blockIdx.x * BN, m0 = (int64_t)blockIdx.y * BM;
    const __nv_bfloat16* Bh = B + h * (int64_t)K * N;
    uint4 ra[2], rb[2];
    auto gload = [&](int k0) {
#pragma unroll
        for (int i = 0; i < 2; ++i) {   // A: 64 rows x 4 chunks of 8
            const int c = t + i * 128, r = c >> 2, kc = (c & 3) * 8;
            const int64_t row = m0 + r;
            ra[i] = row < rows ? *reinterpret_cast<const uint4*>(A + (row * heads + h) * K + k0 + kc) : make_uint4(0, 0, 0, 0);
        }
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int c = t + i * 128;
            if (BNK) { const int n = c >> 2, kc = (c & 3) * 8; rb[i] = *reinterpret_cast<const uint4*>(Bh + (n0 + n) * K + k0 + kc); }
            else { const int k = c >> 3, nc = (c & 7) * 8; rb[i] = *reinterpret_cast<const uint4*>(Bh + (int64_t)(k0 + k) * N + n0 + nc); }
        }
    };
    auto sstore = [&](int buf) {
#pragma unroll
        for (int i = 0; i < 2; ++i) { const int c = t + i * 128, r = c >> 2, kc = (c & 3) * 8; *reinterpret_cast<uint4*>(&As[buf][r * AS + kc]) = ra[i]; }
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int c = t + i * 128;
            if (BNK) { const int n = c >> 2, kc = (c & 3) * 8; *reinterpret_cast<uint4*>(&Bs[buf][n * BS_NK + kc]) = rb[i]; }
            else { const int k = c >> 3, nc = (c & 7) * 8; *reinterpret_cast<uint4*>(&Bs[buf][k * BS_KN + nc]) = rb[i]; }
        }
    };
    float acc[2][4][4];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[i][j][c] = 0.f;
    gload(0);
    sstore(0);
    __syncthreads();
    constexpr int steps = K / BK;
    for (int s = 0; s < steps; ++s) {
        const int buf = s & 1;
        if (s + 1 < steps) gload((s + 1) * BK);
#pragma unroll
        for (int kk = 0; kk < BK / 16; ++kk) {
            uint32_t a[2][4];
#pragma unroll
            for (int i = 0; i < 2; ++i) ldsm4(a[i], &As[buf][(wm * 32 + i * 16 + (lane & 15)) * AS + kk * 16 + (lane >> 4) * 8]);
#pragma unroll
            for (int jp = 0; jp < 2; ++jp) {   // n16 pairs of n8 tiles
                uint32_t r[4];
                const int nb = wn * 32 + jp * 16;
                if (BNK) ldsm4(r, &Bs[buf][(nb + (lane & 7) + ((lane >> 4) & 1) * 8) * BS_NK + kk * 16 + ((lane >> 3) & 1) * 8]);
                else ldsm4t(r, &Bs[buf][(kk * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * BS_KN + nb + ((lane >> 4) & 1) * 8]);
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    mma_bf16(acc[i][2 * jp], a[i], r[0], r[1]);
                    mma_bf16(acc[i][2 * jp + 1], a[i], r[2], r[3]);
                }
            }
        }
        if (s + 1 < steps) { sstore(buf ^ 1); }
        __syncthreads();
    }
    const int g = lane >> 2, tq = lane & 3;
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
                const int64_t row = m0 + wm * 32 + i * 16 + g + 8 * hh;
                if (row >= rows) continue;
                const int64_t col = n0 + wn * 32 + j * 8 + 2 * tq;
                *reinterpret_cast<__nv_bfloat162*>(C + (row * heads + h) * N + col) = __floats2bfloat162_rn(acc[i][j][2 * hh], acc[i][j][2 * hh + 1]);
            }
}

}  // namespace glm_pf_mla

// latent.absorb_q at 256-wide q_nope, 512-wide latent: qa [rows][heads][512] = q [rows][heads][256] . wk [heads][256][512]
extern "C" __global__ void __launch_bounds__(128) tf_glm_pf_absorb(const __nv_bfloat16* q, const __nv_bfloat16* wk,
                                                                   __nv_bfloat16* qa, int64_t rows, int64_t heads) {
    glm_pf_mla::head_gemm<256, 512, false>(q, wk, qa, rows, heads);
}
// latent.expand_v at 256-wide v heads: out [rows][heads][256] = o_lat [rows][heads][512] . wv [heads][256][512]^T
extern "C" __global__ void __launch_bounds__(128) tf_glm_pf_expand(const __nv_bfloat16* ol, const __nv_bfloat16* wv,
                                                                   __nv_bfloat16* out, int64_t rows, int64_t heads) {
    glm_pf_mla::head_gemm<512, 256, true>(ol, wv, out, rows, heads);
}
