// #952's hyper-connection read-out (fn_hc_down / fn_hc_upmix bits) on prompt tiles: factored groups, norm in smem.
#include "qmm_frag.cuh"
#include "glue_math.cuh"

namespace hcf {
using namespace qmm_frag;
using tf_glue::bf;
using tf_glue::rbf;

template <int BM, int BN, int WM, int WN, int STAGES, int GPS>
struct Tile {
    static constexpr int GS = 32;
    static constexpr int THREADS = WM * WN * 32;
    static constexpr int MT = BM / WM / 16;
    static constexpr int NT = BN / WN / 8;
    static constexpr int ROW = GS * 2;
    static constexpr int CHUNKS = ROW / 16;
    static constexpr int XG = BM * ROW;
    static constexpr int WG = BN * 16;
    static constexpr int SG = BN * 2;
    static constexpr int X = GPS * XG;
    static constexpr int W = GPS * WG;
    static constexpr int S = GPS * SG;
    static constexpr int XS = GPS * BM * 4;
    static constexpr int SC = GPS * GS * 4;            // the norm's scale for the stage's groups
    static constexpr int STAGE = X + W + 2 * S + XS + SC;
    static constexpr int RINV = BM * 4 * 4;
    static constexpr int SMEM = STAGES * STAGE + RINV;
};

// fn_hc's nibbles(w, 2t): codes 2t (low half) and 2t + 1 (high half) as exact bf16 integers.
__device__ __forceinline__ uint32_t nib2(uint32_t w, int t) {
    const uint32_t v = w >> (8 * t);
    const uint32_t q = (v & 0xFu) | ((v << 12) & 0xF0000u) | 0x43004300u;
    uint32_t r;
#if __CUDA_ARCH__ >= 900
    asm("sub.rn.bf16x2 %0, %1, %2;\n" : "=r"(r) : "r"(q), "r"(0x43004300u));
#else
    asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(r) : "r"(q), "r"(0x3F803F80u), "r"(0xC300C300u));  // qmm_frag::pair's form
#endif
    return r;
}

template <int BM, int BN, int WM, int WN, int STAGES, int GPS, bool NORM, bool FOLD, bool OUT16, int MINB = 1>
__global__ void __launch_bounds__(WM * WN * 32, MINB) hcf_kernel(
        const __nv_bfloat16* __restrict__ x, const float* __restrict__ xs, const uint4* __restrict__ w,
        const __nv_bfloat16* __restrict__ sc, const __nv_bfloat16* __restrict__ bi, void* __restrict__ out,
        int M, int N, int K, int SK, int ldx, int group, const float* __restrict__ pss, const float* __restrict__ scale,
        float eps, int D, int NC, int SS) {
    using T = Tile<BM, BN, WM, WN, STAGES, GPS>;
    constexpr int GS = T::GS;
    extern __shared__ __align__(128) unsigned char buf[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g8 = lane >> 2, t4 = lane & 3;
    const int wm = warp / WN, wn = warp % WN;
    const int KG = K / GS, cut = KG / SK, npad = (N + 63) / 64 * 64;
    const int2 at = tile_of(blockIdx.x, M, N, BM, BN, group);
    const int m0 = at.x, n0 = at.y;
    const int stages_n = KG / GPS;
    float* rinv_s = reinterpret_cast<float*>(buf + STAGES * T::STAGE);

    if constexpr (NORM) {
        for (int i = tid; i < BM * SS; i += T::THREADS) {
            const int r = min(m0 + i / SS, M - 1), st = i % SS;
            float total = 0.0f;
            for (int c = 0; c < NC; ++c) total = __fadd_rn(total, pss[(static_cast<long long>(r) * NC + c) * SS + st]);
            rinv_s[i] = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(total, static_cast<float>(D)), eps)));
        }
    }

    auto stage = [&](int s) { return buf + s * T::STAGE; };
    auto load = [&](int s, int gfirst) {
        unsigned char* p = stage(s);
#pragma unroll
        for (int gi = 0; gi < GPS; ++gi) {
            const int g = gfirst + gi;
            unsigned char* px = p + gi * T::XG;
            for (int c = tid; c < BM * T::CHUNKS; c += T::THREADS) {
                const int r = c / T::CHUNKS, ch = c % T::CHUNKS;
                const int row = min(m0 + r, M - 1);
                cp16z(px + r * T::ROW + swz<T::CHUNKS>(r, ch) * 16, x + static_cast<size_t>(row) * ldx + g * GS + ch * 8, m0 + r < M);
            }
            unsigned char* pw = p + T::X + gi * T::WG;
            for (int c = tid; c < BN; c += T::THREADS) {
                const int col = n0 + c;
                if (col < npad) cp16(pw + c * 16, w + (static_cast<size_t>(col / 64) * KG + g) * 64 + col % 64);
            }
            unsigned char* ps = p + T::X + T::W + gi * T::SG;
            for (int c = tid; c < BN; c += T::THREADS) {   // c < BN / 2: scale pairs, the rest bias pairs
                const int which = c / (BN / 2), off = (c % (BN / 2)) * 2, col = n0 + off;
                uint32_t* dst = reinterpret_cast<uint32_t*>(ps + which * T::S + off * 2);
                if (col < N) cp4(dst, (which ? bi : sc) + static_cast<size_t>(g) * N + col);
                else *dst = 0u;
            }
            if constexpr (NORM) {
                float* psc = reinterpret_cast<float*>(p + T::X + T::W + 2 * T::S + T::XS) + gi * GS;
                for (int c = tid; c < GS / 4; c += T::THREADS) cp16(psc + c * 4, scale + g * GS + c * 4);
            } else {
                float* pxs = reinterpret_cast<float*>(p + T::X + T::W + 2 * T::S) + gi * BM;
                for (int r = tid; r < BM; r += T::THREADS) cp4(pxs + r, xs + static_cast<size_t>(min(m0 + r, M - 1)) * KG + g);
            }
        }
    };

    float acc[T::MT][T::NT][4];
    float tot[FOLD ? T::MT : 1][FOLD ? T::NT : 1][4];
#pragma unroll
    for (int i = 0; i < T::MT; ++i)
#pragma unroll
        for (int j = 0; j < T::NT; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;

#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < stages_n) load(s, s * GPS);
        commit();
    }
    for (int it = 0; it < stages_n; ++it) {
        wait<STAGES - 2>();
        __syncthreads();
        const int next = it + STAGES - 1;
        if (next < stages_n) load(next % STAGES, next * GPS);
        commit();
        unsigned char* p = stage(it % STAGES);
        if constexpr (NORM) {
            // x = bf16(h * rinv[row, stream] * scale[k]) as fn_hc_down rounds it, each row's 32-group sum in order
            float* pxs = reinterpret_cast<float*>(p + T::X + T::W + 2 * T::S);
            for (int c = tid; c < GPS * BM; c += T::THREADS) {
                const int gi = c / BM, r = c % BM, k0 = (it * GPS + gi) * GS;
                unsigned char* px = p + gi * T::XG + r * T::ROW;
                const float rv = rinv_s[r * SS + k0 / D];
                float sum = 0.0f;
#pragma unroll
                for (int ch = 0; ch < T::CHUNKS; ++ch) {
                    uint4* cell = reinterpret_cast<uint4*>(px + swz<T::CHUNKS>(r, ch) * 16);
                    const uint4 raw = *cell;
                    const __nv_bfloat16* hv = reinterpret_cast<const __nv_bfloat16*>(&raw);
                    const float* psc = reinterpret_cast<const float*>(p + T::X + T::W + 2 * T::S + T::XS) + gi * GS + ch * 8;
                    const float4 sa = *reinterpret_cast<const float4*>(psc);
                    const float4 sb = *reinterpret_cast<const float4*>(psc + 4);
                    const float sv[8] = {sa.x, sa.y, sa.z, sa.w, sb.x, sb.y, sb.z, sb.w};
                    __nv_bfloat16 o8[8];
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        o8[i] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(bf(hv[i]), rv), sv[i]));
                        sum = __fadd_rn(sum, bf(o8[i]));
                    }
                    *cell = *reinterpret_cast<const uint4*>(o8);
                }
                pxs[gi * BM + r] = sum;
            }
            __syncthreads();
        }
#pragma unroll
        for (int gi = 0; gi < GPS; ++gi) {
            const unsigned char* px = p + gi * T::XG;
            const uint4* pw = reinterpret_cast<const uint4*>(p + T::X + gi * T::WG);
            const __nv_bfloat16* ps = reinterpret_cast<const __nv_bfloat16*>(p + T::X + T::W + gi * T::SG);
            const float* pxs = reinterpret_cast<const float*>(p + T::X + T::W + 2 * T::S) + gi * BM;
            uint32_t a[2][T::MT][4];
#pragma unroll
            for (int kt = 0; kt < 2; ++kt)
#pragma unroll
                for (int i = 0; i < T::MT; ++i) {
                    const int r = wm * (BM / WM) + i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                    const int ch = kt * 2 + (lane >> 4);
                    ldmatrix4(a[kt][i], px + r * T::ROW + swz<T::CHUNKS>(r, ch) * 16);
                }
            float xi[T::MT][2];
#pragma unroll
            for (int i = 0; i < T::MT; ++i) {
                const int row = wm * (BM / WM) + i * 16 + g8;
                xi[i][0] = pxs[row];
                xi[i][1] = pxs[row + 8];
            }
#pragma unroll
            for (int j = 0; j < T::NT; ++j) {
                const int cw = wn * (BN / WN) + j * 8;
                const uint4 words = pw[cw + g8];
                const uint32_t q0 = nib2(words.x, t4), q1 = nib2(words.y, t4), q2 = nib2(words.z, t4), q3 = nib2(words.w, t4);
                const __nv_bfloat162 s2 = *reinterpret_cast<const __nv_bfloat162*>(ps + cw + 2 * t4);
                const __nv_bfloat162 b2 = *reinterpret_cast<const __nv_bfloat162*>(ps + T::S / 2 + cw + 2 * t4);
                const float s0 = __low2float(s2), s1 = __high2float(s2), b0 = __low2float(b2), b1 = __high2float(b2);
#pragma unroll
                for (int i = 0; i < T::MT; ++i) {
                    float dd[4];
                    mma0(dd, a[0][i], q0, q1);
                    mma(dd, a[1][i], q2, q3);
                    acc[i][j][0] = __fmaf_rn(xi[i][0], b0, __fmaf_rn(dd[0], s0, acc[i][j][0]));
                    acc[i][j][1] = __fmaf_rn(xi[i][0], b1, __fmaf_rn(dd[1], s1, acc[i][j][1]));
                    acc[i][j][2] = __fmaf_rn(xi[i][1], b0, __fmaf_rn(dd[2], s0, acc[i][j][2]));
                    acc[i][j][3] = __fmaf_rn(xi[i][1], b1, __fmaf_rn(dd[3], s1, acc[i][j][3]));
                }
            }
            if constexpr (FOLD) {
                const int gl = it * GPS + gi;
                if ((gl + 1) % cut == 0) {
#pragma unroll
                    for (int i = 0; i < T::MT; ++i)
#pragma unroll
                        for (int j = 0; j < T::NT; ++j)
#pragma unroll
                            for (int e = 0; e < 4; ++e) {
                                tot[i][j][e] = gl + 1 == cut ? acc[i][j][e] : __fadd_rn(tot[i][j][e], acc[i][j][e]);
                                acc[i][j][e] = 0.0f;
                            }
                }
            }
        }
    }
    wait<0>();
#pragma unroll
    for (int i = 0; i < T::MT; ++i)
#pragma unroll
        for (int j = 0; j < T::NT; ++j) {
            const int col = n0 + wn * (BN / WN) + j * 8 + 2 * t4;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = m0 + wm * (BM / WM) + i * 16 + g8 + h * 8;
                if (row >= M) continue;
                const float v0 = FOLD ? tot[i][j][2 * h] : acc[i][j][2 * h];
                const float v1 = FOLD ? tot[i][j][2 * h + 1] : acc[i][j][2 * h + 1];
                if constexpr (OUT16) {
                    __nv_bfloat16* dst = reinterpret_cast<__nv_bfloat16*>(out) + static_cast<size_t>(row) * N + col;
                    if (col + 1 < N) *reinterpret_cast<__nv_bfloat162*>(dst) = __floats2bfloat162_rn(v0, v1);
                    else if (col < N) dst[0] = __float2bfloat16_rn(v0);
                } else {
                    float* dst = reinterpret_cast<float*>(out) + static_cast<size_t>(row) * N + col;
                    if (col < N) dst[0] = v0;
                    if (col + 1 < N) dst[1] = v1;
                }
            }
        }
}

}  // namespace hcf

// HC down: 32 x 192 tiles, the streams normed in shared memory, the 32 K slices folded in slice order (fp32 out).
template __global__ void hcf::hcf_kernel<32, 192, 2, 4, 4, 2, true, true, false, 2>(const __nv_bfloat16* __restrict__,
    const float* __restrict__, const uint4* __restrict__, const __nv_bfloat16* __restrict__,
    const __nv_bfloat16* __restrict__, void* __restrict__, int, int, int, int, int, int, const float* __restrict__,
    const float* __restrict__, float, int, int, int);
// HC up: 64 x 128 tiles, bf16 out (the up rows fn_hc_upmix mixes).
template __global__ void hcf::hcf_kernel<64, 128, 2, 4, 3, 2, false, false, true, 2>(const __nv_bfloat16* __restrict__,
    const float* __restrict__, const uint4* __restrict__, const __nv_bfloat16* __restrict__,
    const __nv_bfloat16* __restrict__, void* __restrict__, int, int, int, int, int, int, const float* __restrict__,
    const float* __restrict__, float, int, int, int);

// ---- the native (bf16) hyper-connection read-out on prompt tiles: fn_hc_normalize2 folded into the two projections ----
//
// The exact chain is fn_hc_normalize2 -> qmmf down (f32, K slices folded in order) -> fn_hc_reduce_act -> qmmf up (bf16) ->
// fn_hc_dense_mix2. These two kernels run the SAME mma chains (one m16n8k16 per k16 step from a zero accumulator, the
// shape-driven K slices added in slice order) and the same rounding points, but read the streams `h` directly: the normed
// rows are formed in shared memory (down) or in the epilogue (up) and never travel through memory, and the up rows are mixed
// in the epilogue. Every output is byte for byte the exact chain's (tests/cuda/flashnext_hc_fuse_tests.zig).
namespace hcn {
using namespace qmm_frag;
using tf_glue::bf;
using tf_glue::rbf;

__device__ __forceinline__ float bsig(float x) { return rbf(tf_glue::tf_sigmoid(x)); }

__device__ __forceinline__ float rinv_of(const float* __restrict__ pss, long long r, int st, int NC, int SS, int D, float eps) {
    float total = 0.0f;
    for (int c = 0; c < NC; ++c) total = __fadd_rn(total, pss[(r * NC + c) * SS + st]);
    return __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(total, static_cast<float>(D)), eps)));
}

// ---- down: [rows, K] normed streams x [N <= 336, K] bf16 weights -> fp32 [rows, N], 16 MT rows by 168 columns a CTA ----
constexpr int DN_THREADS = 224;  // 7 warps, three n8 tiles each
// (two column tiles: columns [0, 160) and [160, 336); the warps hold 21 n8 tiles)
constexpr int DN_SPLIT = 20;     // n8 tiles of the first column tile (five groups of 32 columns)
constexpr int DN_SS = 8;         // streams at most

template <int MT, int STAGES>
struct DnShape {
    static constexpr int BM = 16 * MT;
    static constexpr int STAGE = BM * 128;
    static constexpr int SMEM = STAGES * STAGE + BM * DN_SS * 4;
};

template <int MT, int STAGES>
__device__ __forceinline__ void down_body(const __nv_bfloat16* __restrict__ H, const float* __restrict__ PSS,
                                          const float* __restrict__ SCALE, const uint2* __restrict__ W,
                                          float* __restrict__ PART, int M, int N, int K, int SK, int D, int NC, int SS,
                                          float eps) {
    using S = DnShape<MT, STAGES>;
    constexpr int BM = S::BM;
    extern __shared__ __align__(128) unsigned char buf[];
    float* rinv_s = reinterpret_cast<float*>(buf + STAGES * S::STAGE);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g8 = lane >> 2, t4 = lane & 3;
    const int KG = K / 64, per = KG / SK;
    const int ct = blockIdx.x & 1, m0 = (blockIdx.x >> 1) * BM;

    for (int i = tid; i < BM * SS; i += DN_THREADS)
        rinv_s[(i / SS) * DN_SS + i % SS] = rinv_of(PSS, min(m0 + i / SS, M - 1), i % SS, NC, SS, D, eps);

    const uint2* wp[3];
    bool valid[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int n8 = ct * DN_SPLIT + warp * 3 + j;
        valid[j] = n8 * 8 < N && (ct == 1 || n8 < DN_SPLIT);
        wp[j] = W + static_cast<size_t>(n8 >> 3) * KG * 1024 + (n8 & 7) * 128 + lane;
    }

    auto stage = [&](int s) { return buf + s * S::STAGE; };
    auto load = [&](int s, int g) {
        unsigned char* p = stage(s);
        for (int c = tid; c < BM * 8; c += DN_THREADS) {
            const int r = c >> 3, ch = c & 7, row = min(m0 + r, M - 1);
            cp16z(p + r * 128 + swz<8>(r, ch) * 16, H + static_cast<size_t>(row) * K + g * 64 + ch * 8, m0 + r < M);
        }
    };
    auto loadb = [&](int g, uint2 (&b)[3][4]) {
#pragma unroll
        for (int j = 0; j < 3; ++j)
#pragma unroll
            for (int kt = 0; kt < 4; ++kt) b[j][kt] = valid[j] ? __ldg(wp[j] + static_cast<size_t>(g) * 1024 + kt * 32) : make_uint2(0u, 0u);
    };

    float acc[MT][3][4], tot[MT][3][4];
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < 3; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;

#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < KG) load(s, s);
        commit();
    }
    uint2 bufa[3][4], bufb[3][4];
    loadb(0, bufa);

    auto body = [&](int it, uint2 (&cur)[3][4], uint2 (&nxt)[3][4]) {
        wait<STAGES - 2>();
        __syncthreads();
        const int next = it + STAGES - 1;
        if (next < KG) load(next % STAGES, next);
        commit();
        if (it + 1 < KG) loadb(it + 1, nxt);
        unsigned char* p = stage(it % STAGES);
        {
            // x = bf16(h * rinv[row, stream] * scale[k]) as fn_hc_normalize2 rounds it, in place
            const int s_id = (it * 64) / D;
            for (int c = tid; c < BM * 8; c += DN_THREADS) {
                const int r = c >> 3, ch = c & 7;
                uint4* cell = reinterpret_cast<uint4*>(p + r * 128 + swz<8>(r, ch) * 16);
                const uint4 raw = *cell;
                const __nv_bfloat16* hv = reinterpret_cast<const __nv_bfloat16*>(&raw);
                const float rv = rinv_s[r * DN_SS + s_id];
                const float4 sa = __ldg(reinterpret_cast<const float4*>(SCALE + it * 64 + ch * 8));
                const float4 sb = __ldg(reinterpret_cast<const float4*>(SCALE + it * 64 + ch * 8 + 4));
                const float sv[8] = {sa.x, sa.y, sa.z, sa.w, sb.x, sb.y, sb.z, sb.w};
                __nv_bfloat16 o8[8];
#pragma unroll
                for (int i = 0; i < 8; ++i) o8[i] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(bf(hv[i]), rv), sv[i]));
                *cell = *reinterpret_cast<const uint4*>(o8);
            }
        }
        __syncthreads();
#pragma unroll
        for (int kt = 0; kt < 4; ++kt) {
            uint32_t a[MT][4];
#pragma unroll
            for (int i = 0; i < MT; ++i) {
                const int r = i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                const int ch = kt * 2 + (lane >> 4);
                ldmatrix4(a[i], p + r * 128 + swz<8>(r, ch) * 16);
            }
#pragma unroll
            for (int j = 0; j < 3; ++j)
                if (valid[j]) {
#pragma unroll
                    for (int i = 0; i < MT; ++i) mma(acc[i][j], a[i], cur[j][kt].x, cur[j][kt].y);
                }
        }
        if ((it + 1) % per == 0) {
#pragma unroll
            for (int i = 0; i < MT; ++i)
#pragma unroll
                for (int j = 0; j < 3; ++j)
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        tot[i][j][e] = it + 1 == per ? acc[i][j][e] : __fadd_rn(tot[i][j][e], acc[i][j][e]);
                        acc[i][j][e] = 0.0f;
                    }
        }
    };
    for (int it = 0; it < KG; it += 2) {
        body(it, bufa, bufb);
        if (it + 1 < KG) body(it + 1, bufb, bufa);
    }
    wait<0>();
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            if (!valid[j]) continue;
            const int col = (ct * DN_SPLIT + warp * 3 + j) * 8 + 2 * t4;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = m0 + i * 16 + g8 + h * 8;
                if (row >= M) continue;
                float* dst = PART + static_cast<size_t>(row) * N + col;
                if (col + 1 < N) {
                    if ((N & 1) == 0) *reinterpret_cast<float2*>(dst) = make_float2(tot[i][j][2 * h], tot[i][j][2 * h + 1]);
                    else {
                        dst[0] = tot[i][j][2 * h];
                        dst[1] = tot[i][j][2 * h + 1];
                    }
                } else if (col < N) dst[0] = tot[i][j][2 * h];
            }
        }
}

// ---- down, second form: the next group's norm is formed in shared memory while the current group's mma run ----
// Three row buffers rotate: the one the mma read, the one being normed (raw bytes landed a group ago), the one the next
// group's copy fills. One barrier a group; the norm's work is spread between the four k16 steps so the tensor pipe and the
// ALUs run together. The arithmetic of every output is down_body's.
template <int MT, bool XFORM>
__device__ __forceinline__ void down_body2(const __nv_bfloat16* __restrict__ H, const float* __restrict__ PSS,
                                           const float* __restrict__ SCALE, const uint2* __restrict__ W,
                                           float* __restrict__ PART, int M, int N, int K, int SK, int D, int NC, int SS,
                                           float eps) {
    constexpr int BM = 16 * MT, CHUNKS = BM * 8, SLOTS = (CHUNKS + DN_THREADS - 1) / DN_THREADS;
    static_assert(SLOTS <= 4, "one norm slot between each pair of k16 steps");
    extern __shared__ __align__(128) unsigned char buf[];
    float* rinv_s = reinterpret_cast<float*>(buf + 3 * BM * 128);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g8 = lane >> 2, t4 = lane & 3;
    const int KG = K / 64, per = KG / SK;
    const int ct = blockIdx.x & 1, m0 = (blockIdx.x >> 1) * BM;

    for (int i = tid; i < BM * SS; i += DN_THREADS)
        rinv_s[(i / SS) * DN_SS + i % SS] = rinv_of(PSS, min(m0 + i / SS, M - 1), i % SS, NC, SS, D, eps);

    const uint2* wp[3];
    bool valid[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int n8 = ct * DN_SPLIT + warp * 3 + j;
        valid[j] = n8 * 8 < N && (ct == 1 || n8 < DN_SPLIT);
        wp[j] = W + static_cast<size_t>(n8 >> 3) * KG * 1024 + (n8 & 7) * 128 + lane;
    }
    auto stage = [&](int s) { return buf + (s % 3) * (BM * 128); };
    auto load = [&](int g) {
        unsigned char* p = stage(g);
        for (int c = tid; c < CHUNKS; c += DN_THREADS) {
            const int r = c >> 3, ch = c & 7, row = min(m0 + r, M - 1);
            cp16z(p + r * 128 + swz<8>(r, ch) * 16, H + static_cast<size_t>(row) * K + g * 64 + ch * 8, m0 + r < M);
        }
    };
    auto loadb = [&](int g, uint2 (&b)[3][4]) {
#pragma unroll
        for (int j = 0; j < 3; ++j)
#pragma unroll
            for (int kt = 0; kt < 4; ++kt) b[j][kt] = valid[j] ? __ldg(wp[j] + static_cast<size_t>(g) * 1024 + kt * 32) : make_uint2(0u, 0u);
    };
    // x = bf16(h * rinv[row, stream] * scale[k]) as fn_hc_normalize2 rounds it, in place, chunk slot `j` of group `g`
    auto norm_slot = [&](int g, int j) {
        const int c = tid + j * DN_THREADS;
        if (c >= CHUNKS) return;
        const int r = c >> 3, ch = c & 7;
        uint4* cell = reinterpret_cast<uint4*>(stage(g) + r * 128 + swz<8>(r, ch) * 16);
        const uint4 raw = *cell;
        const __nv_bfloat16* hv = reinterpret_cast<const __nv_bfloat16*>(&raw);
        const float rv = rinv_s[r * DN_SS + (g * 64) / D];
        const float4 sa = __ldg(reinterpret_cast<const float4*>(SCALE + g * 64 + ch * 8));
        const float4 sb = __ldg(reinterpret_cast<const float4*>(SCALE + g * 64 + ch * 8 + 4));
        const float sv[8] = {sa.x, sa.y, sa.z, sa.w, sb.x, sb.y, sb.z, sb.w};
        __nv_bfloat16 o8[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) o8[i] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(bf(hv[i]), rv), sv[i]));
        *cell = *reinterpret_cast<const uint4*>(o8);
    };

    float acc[MT][3][4], tot[MT][3][4];
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < 3; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;

    load(0);
    commit();
    if (KG > 1) load(1);
    commit();
    uint2 bufa[3][4], bufb[3][4];
    loadb(0, bufa);
    wait<1>();
    __syncthreads();  // group 0's bytes and the row sums are visible
    if constexpr (XFORM) {
#pragma unroll
        for (int j = 0; j < SLOTS; ++j) norm_slot(0, j);
    }
    wait<0>();
    __syncthreads();

    auto body = [&](int it, uint2 (&cur)[3][4], uint2 (&nxt)[3][4]) {
        if (it + 2 < KG) load(it + 2);
        commit();
        if (it + 1 < KG) loadb(it + 1, nxt);
        const unsigned char* p = stage(it);
#pragma unroll
        for (int kt = 0; kt < 4; ++kt) {
            uint32_t a[MT][4];
#pragma unroll
            for (int i = 0; i < MT; ++i) {
                const int r = i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                const int ch = kt * 2 + (lane >> 4);
                ldmatrix4(a[i], p + r * 128 + swz<8>(r, ch) * 16);
            }
#pragma unroll
            for (int j = 0; j < 3; ++j)
                if (valid[j]) {
#pragma unroll
                    for (int i = 0; i < MT; ++i) mma(acc[i][j], a[i], cur[j][kt].x, cur[j][kt].y);
                }
            if constexpr (XFORM) {
                if (kt < SLOTS && it + 1 < KG) norm_slot(it + 1, kt);
            }
        }
        if ((it + 1) % per == 0) {
#pragma unroll
            for (int i = 0; i < MT; ++i)
#pragma unroll
                for (int j = 0; j < 3; ++j)
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        tot[i][j][e] = it + 1 == per ? acc[i][j][e] : __fadd_rn(tot[i][j][e], acc[i][j][e]);
                        acc[i][j][e] = 0.0f;
                    }
        }
        wait<0>();
        __syncthreads();
    };
    for (int it = 0; it < KG; it += 2) {
        body(it, bufa, bufb);
        if (it + 1 < KG) body(it + 1, bufb, bufa);
    }
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            if (!valid[j]) continue;
            const int col = (ct * DN_SPLIT + warp * 3 + j) * 8 + 2 * t4;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = m0 + i * 16 + g8 + h * 8;
                if (row >= M) continue;
                float* dst = PART + static_cast<size_t>(row) * N + col;
                if (col + 1 < N) {
                    if ((N & 1) == 0) *reinterpret_cast<float2*>(dst) = make_float2(tot[i][j][2 * h], tot[i][j][2 * h + 1]);
                    else {
                        dst[0] = tot[i][j][2 * h];
                        dst[1] = tot[i][j][2 * h + 1];
                    }
                } else if (col < N) dst[0] = tot[i][j][2 * h];
            }
        }
}

// The tensor pipe's ceiling for this instruction: independent m16n8k16 chains, nothing else (a benchmark, not an engine kernel).
__device__ __forceinline__ void peak_body(float* __restrict__ OUT, int iters) {
    uint32_t a[4], b0, b1;
    const uint32_t t = threadIdx.x * 2654435761u;
    a[0] = t & 0x3f003f00u;
    a[1] = (t >> 3) & 0x3f003f00u;
    a[2] = (t >> 5) & 0x3f003f00u;
    a[3] = (t >> 7) & 0x3f003f00u;
    b0 = (t >> 2) & 0x3f003f00u;
    b1 = (t >> 4) & 0x3f003f00u;
    float acc[12][4] = {};
    for (int it = 0; it < iters; ++it) {
#pragma unroll
        for (int i = 0; i < 12; ++i) mma(acc[i], a, b0, b1);
    }
    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < 12; ++i) s += acc[i][0] + acc[i][1] + acc[i][2] + acc[i][3];
    if (s == 123.456f) OUT[threadIdx.x] = s;
}

// ---- up + mix: act [rows, K] x [4 D, K] bf16 weights, rounded to bf16, gated onto the normed streams, mixed in order ----
// A CTA: 16 MT rows by 32 dimensions of all four streams (8 warps, two to a stream, two n8 tiles each).
template <int MT, int KG>
struct UpShape {
    static constexpr int BM = 16 * MT;
    static constexpr int K = KG * 64;
    static constexpr int A = BM * K * 2;            // act rows, swizzled by 128-byte groups
    static constexpr int PARTS = BM * 4 * 32 * 4 + BM * 32 * 4;  // each stream's terms, then the mixed rows (aliased on A)
    static constexpr int HS = BM * 4 * 32 * 2;      // the streams' 32 dimensions of the CTA's rows
    static constexpr int RV = BM * 4 * 4;
    static constexpr int SMEM = (A > PARTS ? A : PARTS) + HS + RV;
};

template <int MT, int KG, int DBG = 0>
__device__ __forceinline__ void upmix_body(const __nv_bfloat16* __restrict__ ACT, const uint2* __restrict__ W,
                                           const __nv_bfloat16* __restrict__ H, const float* __restrict__ PSS,
                                           const float* __restrict__ SCALE, __nv_bfloat16* __restrict__ MIXED,
                                           float* __restrict__ XSM, int M, int D, int NC, float eps) {
    using S = UpShape<MT, KG>;
    constexpr int BM = S::BM, K = S::K, SS = 4;
    extern __shared__ __align__(128) unsigned char buf[];
    unsigned char* As = buf;
    constexpr int ABYTES = S::A > S::PARTS ? S::A : S::PARTS;
    __nv_bfloat16* Hs = reinterpret_cast<__nv_bfloat16*>(buf + ABYTES);
    float* rinv_s = reinterpret_cast<float*>(buf + ABYTES + S::HS);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g8 = lane >> 2, t4 = lane & 3;
    const int dt = D / 32;
    const int m0 = (blockIdx.x / dt) * BM, d0 = (blockIdx.x % dt) * 32;
    const int sw = warp >> 1, q0 = (warp & 1) * 2;

    // this warp's weights: the stream's two n8 tiles, every k16 step, straight from memory
    uint2 b[KG][4][2];
    const uint2* wp[2];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const int c8 = sw * D + d0 + (q0 + j) * 8;
        wp[j] = W + static_cast<size_t>(c8 >> 6) * KG * 1024 + ((c8 >> 3) & 7) * 128 + lane;
    }
#pragma unroll
    for (int g = 0; g < KG; ++g)
#pragma unroll
        for (int kt = 0; kt < 4; ++kt)
#pragma unroll
            for (int j = 0; j < 2; ++j) b[g][kt][j] = __ldg(wp[j] + g * 1024 + kt * 32);

    for (int c = tid; c < BM * (K / 8); c += 256) {
        const int r = c / (K / 8), cc = c % (K / 8), g = cc >> 3, ch = cc & 7, row = min(m0 + r, M - 1);
        cp16z(As + r * (K * 2) + g * 128 + swz<8>(r, ch) * 16, ACT + static_cast<size_t>(row) * K + cc * 8, m0 + r < M);
    }
    for (int c = tid; c < BM * SS * 4; c += 256) {
        const int r = c / (SS * 4), s = (c >> 2) % SS, q = c & 3, row = min(m0 + r, M - 1);
        cp16z(reinterpret_cast<unsigned char*>(Hs) + ((r * SS + s) * 32 + q * 8) * 2,
              H + static_cast<size_t>(row) * (SS * D) + s * D + d0 + q * 8, m0 + r < M);
    }
    commit();
    for (int i = tid; i < BM * SS; i += 256) rinv_s[i] = rinv_of(PSS, min(m0 + i / SS, M - 1), i % SS, NC, SS, D, eps);
    wait<0>();
    __syncthreads();

    float acc[MT][2][4];
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;
    if constexpr (DBG != 2) {
#pragma unroll
        for (int g = 0; g < KG; ++g)
#pragma unroll
            for (int kt = 0; kt < 4; ++kt) {
                uint32_t a[MT][4];
#pragma unroll
                for (int i = 0; i < MT; ++i) {
                    const int r = i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                    const int ch = kt * 2 + (lane >> 4);
                    ldmatrix4(a[i], As + r * (K * 2) + g * 128 + swz<8>(r, ch) * 16);
                }
#pragma unroll
                for (int j = 0; j < 2; ++j)
#pragma unroll
                    for (int i = 0; i < MT; ++i) mma(acc[i][j], a[i], b[g][kt][j].x, b[g][kt][j].y);
            }
    }
    if constexpr (DBG == 1) {
        // benchmark form: the projection alone (nothing downstream, a sum keeps it alive)
        float sum = 0.0f;
#pragma unroll
        for (int i = 0; i < MT; ++i)
#pragma unroll
            for (int j = 0; j < 2; ++j)
#pragma unroll
                for (int e = 0; e < 4; ++e) sum += acc[i][j][e] + __uint_as_float(b[KG - 1][3][j].x);
        if (sum == 123.456f) XSM[tid] = sum;
        return;
    }
    __syncthreads();  // A is dead: its bytes now hold the terms
    float* part = reinterpret_cast<float*>(As);
    float* mixed_s = part + BM * 4 * 32;
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int rr = i * 16 + g8 + 8 * (e >> 1), dd = (q0 + j) * 8 + 2 * t4 + (e & 1);
                const float nv = rbf(__fmul_rn(__fmul_rn(bf(Hs[(rr * SS + sw) * 32 + dd]), rinv_s[rr * SS + sw]), SCALE[sw * D + d0 + dd]));
                part[(sw * BM + rr) * 32 + dd] = rbf(__fmul_rn(bsig(rbf(acc[i][j][e])), nv));
            }
    __syncthreads();
    for (int e = tid; e < BM * 32; e += 256) {
        const int rr = e >> 5, dd = e & 31;
        float total = 0.0f;
#pragma unroll
        for (int s = 0; s < SS; ++s) total = __fadd_rn(total, part[(s * BM + rr) * 32 + dd]);
        const float m = rbf(__fdiv_rn(total, static_cast<float>(SS)));
        mixed_s[e] = m;
        if (m0 + rr < M) MIXED[static_cast<size_t>(m0 + rr) * D + d0 + dd] = __float2bfloat16_rn(m);
    }
    __syncthreads();
    if (tid < BM && m0 + tid < M) {
        float sum = 0.0f;
        for (int k = 0; k < 32; ++k) sum = __fadd_rn(sum, mixed_s[tid * 32 + k]);
        XSM[static_cast<size_t>(m0 + tid) * dt + d0 / 32] = sum;
    }
}

}  // namespace hcn

#define HCN_DOWN(NAME, MT, STAGES)                                                                                         \
    extern "C" __global__ void __launch_bounds__(hcn::DN_THREADS) NAME(                                                   \
        const __nv_bfloat16* __restrict__ H, const float* __restrict__ PSS, const float* __restrict__ SCALE,              \
        const uint2* __restrict__ W, float* __restrict__ PART, int M, int N, int K, int SK, int D, int NC, int SS,        \
        float eps) {                                                                                                       \
        hcn::down_body<MT, STAGES>(H, PSS, SCALE, W, PART, M, N, K, SK, D, NC, SS, eps);                                  \
    }
#define HCN_UP(NAME, MT, KG)                                                                                               \
    extern "C" __global__ void __launch_bounds__(256) NAME(                                                               \
        const __nv_bfloat16* __restrict__ ACT, const uint2* __restrict__ W, const __nv_bfloat16* __restrict__ H,         \
        const float* __restrict__ PSS, const float* __restrict__ SCALE, __nv_bfloat16* __restrict__ MIXED,               \
        float* __restrict__ XSM, int M, int D, int NC, float eps) {                                                       \
        hcn::upmix_body<MT, KG, 0>(ACT, W, H, PSS, SCALE, MIXED, XSM, M, D, NC, eps);                                     \
    }

HCN_DOWN(fn_hcn_down2, 2, 3)
HCN_DOWN(fn_hcn_down3, 3, 3)
HCN_DOWN(fn_hcn_down4, 4, 3)
HCN_DOWN(fn_hcn_down8, 8, 3)
HCN_UP(fn_hcn_up2_k5, 2, 5)
HCN_UP(fn_hcn_up3_k5, 3, 5)
HCN_UP(fn_hcn_up4_k5, 4, 5)

#define HCN_DOWN2(NAME, MT, XF)                                                                                            \
    extern "C" __global__ void __launch_bounds__(hcn::DN_THREADS) NAME(                                                   \
        const __nv_bfloat16* __restrict__ H, const float* __restrict__ PSS, const float* __restrict__ SCALE,              \
        const uint2* __restrict__ W, float* __restrict__ PART, int M, int N, int K, int SK, int D, int NC, int SS,        \
        float eps) {                                                                                                       \
        hcn::down_body2<MT, XF>(H, PSS, SCALE, W, PART, M, N, K, SK, D, NC, SS, eps);                                     \
    }
HCN_DOWN2(fn_hcn_dv2_2, 2, true)
HCN_DOWN2(fn_hcn_dv2_3, 3, true)
HCN_DOWN2(fn_hcn_dv2_4, 4, true)
HCN_DOWN2(fn_hcn_dv2_4_nx, 4, false)
extern "C" __global__ void __launch_bounds__(256) fn_hcn_peak(float* OUT, int iters) { hcn::peak_body(OUT, iters); }

#define HCN_UP_DBG(NAME, MT, KG, DBG)                                                                                      \
    extern "C" __global__ void __launch_bounds__(256) NAME(                                                               \
        const __nv_bfloat16* __restrict__ ACT, const uint2* __restrict__ W, const __nv_bfloat16* __restrict__ H,         \
        const float* __restrict__ PSS, const float* __restrict__ SCALE, __nv_bfloat16* __restrict__ MIXED,               \
        float* __restrict__ XSM, int M, int D, int NC, float eps) {                                                       \
        hcn::upmix_body<MT, KG, DBG>(ACT, W, H, PSS, SCALE, MIXED, XSM, M, D, NC, eps);                                   \
    }
HCN_UP_DBG(fn_hcn_up3_k5_gemm, 3, 5, 1)
HCN_UP_DBG(fn_hcn_up3_k5_epi, 3, 5, 2)
