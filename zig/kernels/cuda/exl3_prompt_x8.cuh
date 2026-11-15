// The e4m3 prompt gate/up (TF_GLM_X3_FP8): its input rotation to e4m3 rows and the gate/up kernel.
#pragma once
#include <cuda_fp8.h>
// FP8 (e4m3) gate/up of the prompt experts: EXL3 tiles decoded to f16 as before, converted to e4m3 pairs, m16n8k32 MMA
// against e4m3 rows of Xg / Xu (per-row scales, K permuted within each 32-chunk to the A fragment's logical order).
namespace tf_x8 {
using namespace tf_exl3_prompt;
constexpr int XS8 = TBK + 16;                         // bytes a staged row (conflict-free ldmatrix at 48 B)
constexpr int XB8 = 2 * TPM * XS8;                    // gate rows then up rows
constexpr int SB8 = XB8 + TWORDS * 4;                 // a stage's bytes
constexpr int ep8 = 2 * TPH * GU_ES * 4;
constexpr int smem8 = (TSTAGES * SB8 > ep8) ? TSTAGES * SB8 : ep8;
static_assert(TKT == 2, "one e4m3 k32 step a stage");

__device__ __forceinline__ uint32_t pk8(uint32_t lo, uint32_t hi) {
    uint16_t a, b;
    asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(a) : "r"(lo));
    asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(b) : "r"(hi));
    return (uint32_t)a | ((uint32_t)b << 16);
}
__device__ __forceinline__ void mma8(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void ldm4(uint32_t (&r)[4], const void* p) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}

template <int K2, int CB>
__device__ __forceinline__ void tmain8(const uint8_t* __restrict__ X0, const uint8_t* __restrict__ X1, const uint64_t* __restrict__ tp0,
        const uint64_t* __restrict__ tp1, const int* rows_sh, int cnt, int e, int K, int NTILES, int cb, unsigned char* stage,
        float (&acc)[TMT][TNB][4]) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int mat, nt_unused;
    ttile<2>(warp * TMT, cb, mat, nt_unused);
    const int chunks = K / TBK;
    const uint32_t* T0 = reinterpret_cast<const uint32_t*>(tp0[e]);
    const uint32_t* T1 = reinterpret_cast<const uint32_t*>(tp1[e]);
    const int nblocks = (cnt + 15) >> 4;
#pragma unroll
    for (int i = 0; i < TMT; ++i)
#pragma unroll
        for (int j = 0; j < TNB; ++j)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[i][j][c] = 0.f;
    auto load = [&](int chunk) {
        unsigned char* dst = stage + (size_t)(chunk % TSTAGES) * SB8;
        const int k0 = chunk * TBK;
        for (int idx = threadIdx.x; idx < 2 * TPM * (TBK / 16); idx += TT) {
            const int m = idx / (TPM * (TBK / 16)), rem = idx % (TPM * (TBK / 16)), i = rem / (TBK / 16), j = rem % (TBK / 16);
            if (i >= nblocks * 16) continue;
            const int r = rows_sh[i];
            const uint8_t* src = (m ? X1 : X0) + (size_t)(r < 0 ? 0 : r) * K + k0 + j * 16;
            cp_async16(dst + ((size_t)m * TPM + i) * XS8 + j * 16, src, r >= 0);
        }
        uint32_t* wdst = reinterpret_cast<uint32_t*>(dst + XB8);
        for (int idx = threadIdx.x; idx < 16 * TKT * K2; idx += TT) {
            const int q = idx / (TKT * K2), kk = (idx / K2) % TKT, part = idx % K2;
            int wm, nt;
            ttile<2>(q, cb, wm, nt);
            const uint32_t* src = (wm ? T1 : T0) + (((size_t)chunk * TKT + kk) * NTILES + nt) * tile_words<K2>() + part * 4;
            cp_async16(wdst + ((size_t)q * TKT + kk) * 32 + part * 4, src, true);
        }
    };
#pragma unroll
    for (int c = 0; c < TSTAGES - 1; ++c) {
        if (c < chunks) load(c);
        cp_commit();
    }
    for (int c = 0; c < chunks; ++c) {
        cp_wait<TSTAGES - 2>();
        __syncthreads();
        if (c + TSTAGES - 1 < chunks) load(c + TSTAGES - 1);
        cp_commit();
        const unsigned char* base = stage + (size_t)(c % TSTAGES) * SB8;
        const unsigned char* xs = base + (size_t)mat * TPM * XS8;
        const uint32_t* ws = reinterpret_cast<const uint32_t*>(base + XB8);
        uint32_t a[TMT][4];
#pragma unroll
        for (int i = 0; i < TMT; ++i) {
            uint32_t b0[2], b1[2], c0[2], c1[2];
            decode_tile_p<K2, CB>(ws + ((size_t)(warp * TMT + i) * TKT + 0) * 32, lane, b0, b1);
            decode_tile_p<K2, CB>(ws + ((size_t)(warp * TMT + i) * TKT + 1) * 32, lane, c0, c1);
            a[i][0] = pk8(b0[0], b0[1]); a[i][1] = pk8(b1[0], b1[1]); a[i][2] = pk8(c0[0], c0[1]); a[i][3] = pk8(c1[0], c1[1]);
        }
#pragma unroll
        for (int jb = 0; jb < TPM / 16; ++jb) {
            if (jb < nblocks) {
                uint32_t r[4];
                const int row = jb * 16 + (lane & 7) + ((lane >> 4) & 1) * 8;
                ldm4(r, xs + (size_t)row * XS8 + ((lane >> 3) & 1) * 16);
                const uint32_t bl[2] = {r[0], r[1]}, bh[2] = {r[2], r[3]};
#pragma unroll
                for (int i = 0; i < TMT; ++i) {
                    mma8(acc[i][2 * jb], a[i], bl);
                    mma8(acc[i][2 * jb + 1], a[i], bh);
                }
            }
        }
    }
    cp_wait<0>();
    __syncthreads();
}
}  // namespace tf_x8

// As tf_exl3_prompt_gu (xdiv 1, mul1 widths), with Xg / Xu e4m3 [P][K] (32-chunk logical order) and their row scales.
extern "C" __global__ void __launch_bounds__(256, 2) tf_exl3_prompt_gu8(const uint8_t* __restrict__ Xg, const uint8_t* __restrict__ Xu,
        const float* __restrict__ sg, const float* __restrict__ su,
        const uint64_t* __restrict__ Tg, const uint64_t* __restrict__ Tu, const int* __restrict__ items, const int* __restrict__ counts,
        const int* __restrict__ members, const half* __restrict__ svh_g, const half* __restrict__ svh_u, const half* __restrict__ suh_d,
        half* __restrict__ xd, int K, int N, float limit, int act_mode, int skip, const int* __restrict__ k2s) {
    using namespace tf_exl3_prompt;
    const int cb = blockIdx.x, item = blockIdx.y;
    if (item >= counts[0]) return;
    const int e = items[3 * item];
    if (e == skip) return;
    const int first = items[3 * item + 1], cnt = items[3 * item + 2];
    __shared__ __align__(16) unsigned char smem_raw[tf_x8::smem8];
    __shared__ int rows_sh[TPM];
    if (threadIdx.x < TPM) rows_sh[threadIdx.x] = (int)threadIdx.x < cnt ? members[first + threadIdx.x] : -1;
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float acc[TMT][TNB][4];
    const int k2 = k2s[e];
    if (k2 == 4) tf_x8::tmain8<4, CB_MUL1>(Xg, Xu, Tg, Tu, rows_sh, cnt, e, K, N >> 4, cb, smem_raw, acc);
    else if (k2 == 6) tf_x8::tmain8<6, CB_MUL1>(Xg, Xu, Tg, Tu, rows_sh, cnt, e, K, N >> 4, cb, smem_raw, acc);
    else if (k2 == 8) tf_x8::tmain8<8, CB_MUL1>(Xg, Xu, Tg, Tu, rows_sh, cnt, e, K, N >> 4, cb, smem_raw, acc);
    else __trap();
    float* Eg = reinterpret_cast<float*>(smem_raw);
    int mat, nt;
    ttile<2>(warp * TMT, cb, mat, nt);
    const int col0 = (nt - cb * 8) * 16;
    const int n = cb * 128 + 4 * lane;
    for (int hf = 0; hf < 2 && hf * TPH < cnt; ++hf) {
        if (hf) __syncthreads();
        tspill(Eg + (size_t)mat * TPH * GU_ES, GU_ES, col0, acc, hf);
        __syncthreads();
        for (int i = warp; i < TPH && hf * TPH + i < cnt; i += TNW) {
            const int p = rows_sh[hf * TPH + i];
            const float gs = sg[p], us = su[p];
            float gv[4], uv[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                gv[j] = Eg[(size_t)i * GU_ES + 4 * lane + j] * gs;
                uv[j] = Eg[(size_t)(TPH + i) * GU_ES + 4 * lane + j] * us;
            }
            fwht128(gv, lane);
            fwht128(uv, lane);
            float v[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float gg = gv[j] * HAD_SCALE * __half2float(svh_g[(size_t)e * N + n + j]);
                float uu = uv[j] * HAD_SCALE * __half2float(svh_u[(size_t)e * N + n + j]);
                if (act_mode == 0) { gg = bf16r(gg); uu = bf16r(uu); }
                if (limit > 0) { gg = upTo(gg, limit); uu = within(uu, limit); }
                float act = gg / (1.f + expf(-gg));
                if (act_mode == 0) act = bf16r(act);
                act *= uu;
                if (act_mode == 0) act = bf16r(act);
                v[j] = act * __half2float(suh_d[(size_t)e * N + n + j]);
            }
            fwht128(v, lane);
            half* o = xd + (size_t)p * N + n;
#pragma unroll
            for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * HAD_SCALE);
        }
    }
}

// Program (slot, matrix), 1024 threads (K <= 4096, a warp a 128-block): as tf_exl3_experts_rot_in (bf16 rows), the
// slot's rotated row (f16-rounded as there) as e4m3 of v / (absmax / 448), in tf_exl3_prompt_gu8's K order (each
// 32-chunk in the m16n8k32 A fragment's logical order), its scale in s0 / s1.
extern "C" __global__ void __launch_bounds__(1024) tf_exl3_experts_rot_in8(const __nv_bfloat16* __restrict__ x, int x_stride,
        const int* __restrict__ pick, const half* __restrict__ suh0, const half* __restrict__ suh1, uint8_t* __restrict__ out0,
        uint8_t* __restrict__ out1, float* __restrict__ s0, float* __restrict__ s1, int k, int slots, int experts) {
    using namespace tf_exl3;
    const int p = blockIdx.x, mat = blockIdx.y;
    const int e = pick[p];
    if (e < 0 || e >= experts) return;
    if (k > 4096) __trap();
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = p / slots;
    __shared__ float wmax[32];
    float v[4] = {0.f, 0.f, 0.f, 0.f};
    const bool on = w * 128 < k;
    if (on) {
        const half* suh = (mat ? suh1 : suh0) + (size_t)e * k + w * 128 + 4 * lane;
        const __nv_bfloat16* xr = x + (size_t)row * x_stride + w * 128 + 4 * lane;
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = __bfloat162float(xr[j]) * __half2float(suh[j]);
        fwht128(v, lane);
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = __half2float(__float2half_rn(v[j] * HAD_SCALE));
    }
    float m = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if (lane == 0) wmax[w] = m;
    __syncthreads();
    if (w == 0) {
        float q = wmax[lane];
        for (int o = 16; o; o >>= 1) q = fmaxf(q, __shfl_xor_sync(0xffffffffu, q, o));
        if (lane == 0) wmax[0] = q;
    }
    __syncthreads();
    const float mx = wmax[0], sc = mx > 0.f ? mx / 448.f : 1.f, inv = 1.f / sc;
    if (on) {
        uint8_t* o = (mat ? out1 : out0) + (size_t)p * k + w * 128;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int P = 4 * lane + j, c = P & ~31, q = P & 15, hi = (P & 16) ? 16 : 0;
            const int L = c + hi + (q < 8 ? 4 * (q >> 1) + (q & 1) : 4 * ((q - 8) >> 1) + 2 + (q & 1));
            o[L] = (uint8_t)__nv_cvt_float_to_fp8(v[j] * inv, __NV_SATFINITE, __NV_E4M3);
        }
    }
    if (threadIdx.x == 0) (mat ? s1 : s0)[p] = sc;
}
