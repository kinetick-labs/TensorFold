// Flash Next's attention glue and its sparse indexer's pooling and scores: each row's own fixed-order sums.

// glue.py and attention.py's arithmetic and bf16 rounding points; the attention dots on mma tiles, K ascending.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <math.h>
#include <stdint.h>

#include "glue_math.cuh"
#include "qmm_frag.cuh"

using namespace tf_glue;

namespace fn_attn {

constexpr int TK = 32;   // keys an attention tile holds
constexpr int HD = 256;  // head width the attention tiles are built for
constexpr int QR = 16;   // query heads a KV head's tile holds (G <= 16, the rest zero)

// A block's fixed-order sum: each warp's butterfly, then the warps' sums in warp order.
__device__ __forceinline__ float block_sum(float v, float* scratch) {
    v = warp_sum(v);
    const int warps = blockDim.x >> 5;
    if ((threadIdx.x & 31) == 0) scratch[threadIdx.x >> 5] = v;
    __syncthreads();
    float total = 0.0f;
    for (int w = 0; w < warps; ++w) total = __fadd_rn(total, scratch[w]);
    __syncthreads();
    return total;
}

__device__ __forceinline__ float inv_rms(float ss, int width, float eps) {
    return __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(ss, static_cast<float>(width)), eps)));
}

// Rotate-half RoPE over the first 2 * half features at `pos`: cos and sin of the fp32 angle (full range reduction).
__device__ __forceinline__ float rope(float xn, float xpn, int d, int half, int pos, const float* __restrict__ inv) {
    if (d >= 2 * half) return xn;
    const int i = d < half ? d : d - half;
    float s, c;
    sincosf(__fmul_rn(static_cast<float>(pos), inv[i]), &s, &c);
    return d < half ? __fsub_rn(__fmul_rn(xn, c), __fmul_rn(xpn, s)) : __fadd_rn(__fmul_rn(xpn, s), __fmul_rn(xn, c));
}

// ldmatrix .x4 .trans: four 8x8 bf16 tiles of shared memory, each thread's pair down a column (the B layout of rows).
__device__ __forceinline__ void ldmatrix4t(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(qmm_frag::smem(p)));
}

__device__ __forceinline__ int partner(int d, int half) {
    return d < half ? d + half : (d < 2 * half ? d - half : d);
}

// --kv-cache fp8 (cuda_kv.zig): a position's KV head keeps its keys as a row of KS bytes, HD e4m3 codes then s_k and s_v
// (fp32) and 8 zero bytes, and its values as a row of HD codes. A row's scale is 2^(ceil(log2 amax) - 8) from the bits of
// its largest magnitude, so its largest code lies in (128, 256] and nothing saturates (e4m3 holds 448).
constexpr int KS = HD + 16;

// A block's largest value, whatever the order (fmaxf drops a NaN: the codes carry it).
__device__ __forceinline__ float block_max(float v, float* scratch) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    const int warps = blockDim.x >> 5;
    if ((threadIdx.x & 31) == 0) scratch[threadIdx.x >> 5] = v;
    __syncthreads();
    float m = scratch[0];
    for (int w = 1; w < warps; ++w) m = fmaxf(m, scratch[w]);
    __syncthreads();
    return m;
}

// The scale's biased exponent, 1 to 254 (a zero row gets 2^-126): cuda_kv.zig scaleExponent. The scale is 2^(e - 127), its
// reciprocal 2^(127 - e), the float with biased exponent 254 - e; both exact.
__device__ __forceinline__ int kv8_exponent(float amax) {
    const unsigned bits = __float_as_uint(amax);
    const int e = static_cast<int>((bits >> 23) & 0xFFu) + ((bits & 0x7FFFFFu) != 0u ? 1 : 0);
    return min(max(e - 8, 1), 254);
}

// x (already times the reciprocal scale) to its e4m3 code: nearest, ties to even, saturating at 448 (cuda_kv.zig encode).
__device__ __forceinline__ unsigned char kv8_code(float x) {
    return static_cast<unsigned char>(__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3));
}

// A position's KV head: key k and value v (bf16 values as floats) of thread d of HD to codes, and the two scales (thread 0).
__device__ __forceinline__ void kv8_store(unsigned char* __restrict__ KC, unsigned char* __restrict__ VC, long long slot, float k,
                                          float v, float* scratch) {
    const int d = threadIdx.x;
    const int ek = kv8_exponent(block_max(fabsf(k), scratch));
    const int ev = kv8_exponent(block_max(fabsf(v), scratch));
    KC[slot * KS + d] = kv8_code(__fmul_rn(k, __uint_as_float(static_cast<unsigned>(254 - ek) << 23)));
    VC[slot * HD + d] = kv8_code(__fmul_rn(v, __uint_as_float(static_cast<unsigned>(254 - ev) << 23)));
    if (d == 0) *reinterpret_cast<uint4*>(KC + slot * KS + HD) = make_uint4(static_cast<unsigned>(ek) << 23, static_cast<unsigned>(ev) << 23, 0u, 0u);
}

// Eight e4m3 codes (two words of a row) times s as eight bf16 (one 16-byte store): every product is a bf16 value, so the
// widening through half and fp32 is exact.
__device__ __forceinline__ uint4 kv8_widen8(uint32_t w0, uint32_t w1, float s) {
    const __nv_fp8x2_storage_t* p0 = reinterpret_cast<const __nv_fp8x2_storage_t*>(&w0);
    const __nv_fp8x2_storage_t* p1 = reinterpret_cast<const __nv_fp8x2_storage_t*>(&w1);
    uint4 r;
    uint32_t* out = reinterpret_cast<uint32_t*>(&r);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 f = __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2(i < 2 ? p0[i] : p1[i - 2], __NV_E4M3)));
        const __nv_bfloat162 b = __floats2bfloat162_rn(__fmul_rn(f.x, s), __fmul_rn(f.y, s));
        out[i] = *reinterpret_cast<const uint32_t*>(&b);
    }
    return r;
}

}  // namespace fn_attn

using namespace fn_attn;

__device__ __forceinline__ int media_pos(int pos, int d, int half, const int* rope3, int n, int delta) {
    if (!rope3) return pos;
    const int axis = (d < half ? d : d - half) % 3;
    return pos < n ? rope3[axis * n + pos] : pos + delta;
}

// glue._attn_prep (text positions, bf16 cache), program (row, head): heads normed, rounded, roped, rounded, stored.
// KV8: the cache is cuda_kv.zig's fp8 rows (KC and VC are bytes there); the KV head's program quantizes its key and value.
template <bool KV8 = false>
__device__ __forceinline__ void fn_attn_prep_body(const __nv_bfloat16* __restrict__ P, const int* __restrict__ POS0,
                                                               const float* __restrict__ QW, const float* __restrict__ KW,
                                                               const float* __restrict__ IW, const float* __restrict__ INV,
                                                               __nv_bfloat16* __restrict__ Q, __nv_bfloat16* __restrict__ KC,
                                                               __nv_bfloat16* __restrict__ VC, __nv_bfloat16* __restrict__ IQ,
                                                               __nv_bfloat16* __restrict__ IKC, float eps, int PW, int NQ, int NKV,
                                                               int NI, int IHD, int HALF, const int* rope3, int n, int delta) {
    __shared__ float scratch[32];
    const int r = blockIdx.x, head = blockIdx.y, d = threadIdx.x;
    const int pos = POS0[0] + r;
    const long long row = static_cast<long long>(r) * PW;
    if (head >= NQ + NKV + NI) {
        if (d < IHD) IKC[static_cast<long long>(pos) * IHD + d] = P[row + NQ * 2 * HD + 2 * NKV * HD + NI * IHD + d];
        return;
    }
    const bool index = head >= NQ + NKV;
    const int width = index ? IHD : HD;
    const bool live = d < width;
    const long long src = row + (head < NQ ? head * 2 * HD : (head < NQ + NKV ? NQ * 2 * HD + (head - NQ) * HD : NQ * 2 * HD + 2 * NKV * HD + (head - NQ - NKV) * IHD));
    const float* w = head < NQ ? QW : (head < NQ + NKV ? KW : IW);
    const float x = live ? bf(P[src + d]) : 0.0f;
    const float rinv = inv_rms(block_sum(__fmul_rn(x, x), scratch), width, eps);
    if (!live) return;
    const int p = partner(d, HALF);
    const float xn = rbf(__fmul_rn(__fmul_rn(x, rinv), w[d]));
    const float xpn = rbf(__fmul_rn(__fmul_rn(bf(P[src + p]), rinv), w[p]));
    const __nv_bfloat16 out = __float2bfloat16_rn(rope(xn, xpn, d, HALF, media_pos(pos, d, HALF, rope3, n, delta), INV));
    if (head < NQ) {
        Q[(static_cast<long long>(r) * NQ + head) * HD + d] = out;
    } else if (!index) {
        const long long slot = static_cast<long long>(pos) * NKV + head - NQ;
        const __nv_bfloat16 v = P[row + NQ * 2 * HD + NKV * HD + (head - NQ) * HD + d];
        if constexpr (KV8) {
            kv8_store(reinterpret_cast<unsigned char*>(KC), reinterpret_cast<unsigned char*>(VC), slot, bf(out), bf(v), scratch);
        } else {
            KC[slot * HD + d] = out;
            VC[slot * HD + d] = v;
        }
    } else {
        IQ[(static_cast<long long>(r) * NI + head - NQ - NKV) * IHD + d] = out;
    }
}

extern "C" __global__ void __launch_bounds__(256) fn_attn_prep(const __nv_bfloat16* __restrict__ P, const int* __restrict__ POS0,
                                                               const float* __restrict__ QW, const float* __restrict__ KW,
                                                               const float* __restrict__ IW, const float* __restrict__ INV,
                                                               __nv_bfloat16* __restrict__ Q, __nv_bfloat16* __restrict__ KC,
                                                               __nv_bfloat16* __restrict__ VC, __nv_bfloat16* __restrict__ IQ,
                                                               __nv_bfloat16* __restrict__ IKC, float eps, int PW, int NQ, int NKV,
                                                               int NI, int IHD, int HALF) {
    fn_attn_prep_body(P, POS0, QW, KW, IW, INV, Q, KC, VC, IQ, IKC, eps, PW, NQ, NKV, NI, IHD, HALF, nullptr, 0, 0);
}

extern "C" __global__ void __launch_bounds__(256) fn_attn_prep_media(const __nv_bfloat16* __restrict__ P, const int* __restrict__ POS0,
                                                               const float* __restrict__ QW, const float* __restrict__ KW,
                                                               const float* __restrict__ IW, const float* __restrict__ INV,
                                                               __nv_bfloat16* __restrict__ Q, __nv_bfloat16* __restrict__ KC,
                                                               __nv_bfloat16* __restrict__ VC, __nv_bfloat16* __restrict__ IQ,
                                                               __nv_bfloat16* __restrict__ IKC, float eps, int PW, int NQ, int NKV,
                                                               int NI, int IHD, int HALF, const int* rope3, int n, int delta) {
    fn_attn_prep_body(P, POS0, QW, KW, IW, INV, Q, KC, VC, IQ, IKC, eps, PW, NQ, NKV, NI, IHD, HALF, rope3, n, delta);
}

// fn_attn_prep into fp8 caches: KC [cap, NKV, KS] and VC [cap, NKV, HD] bytes; the grid, block and other arguments as fn_attn_prep's.
extern "C" __global__ void __launch_bounds__(256) fn_attn_prep_kv8(const __nv_bfloat16* __restrict__ P, const int* __restrict__ POS0,
                                                               const float* __restrict__ QW, const float* __restrict__ KW,
                                                               const float* __restrict__ IW, const float* __restrict__ INV,
                                                               __nv_bfloat16* __restrict__ Q, unsigned char* __restrict__ KC,
                                                               unsigned char* __restrict__ VC, __nv_bfloat16* __restrict__ IQ,
                                                               __nv_bfloat16* __restrict__ IKC, float eps, int PW, int NQ, int NKV,
                                                               int NI, int IHD, int HALF) {
    fn_attn_prep_body<true>(P, POS0, QW, KW, IW, INV, Q, reinterpret_cast<__nv_bfloat16*>(KC), reinterpret_cast<__nv_bfloat16*>(VC), IQ, IKC, eps, PW, NQ, NKV, NI, IHD, HALF, nullptr, 0, 0);
}

extern "C" __global__ void __launch_bounds__(256) fn_attn_prep_media_kv8(const __nv_bfloat16* __restrict__ P, const int* __restrict__ POS0,
                                                               const float* __restrict__ QW, const float* __restrict__ KW,
                                                               const float* __restrict__ IW, const float* __restrict__ INV,
                                                               __nv_bfloat16* __restrict__ Q, unsigned char* __restrict__ KC,
                                                               unsigned char* __restrict__ VC, __nv_bfloat16* __restrict__ IQ,
                                                               __nv_bfloat16* __restrict__ IKC, float eps, int PW, int NQ, int NKV,
                                                               int NI, int IHD, int HALF, const int* rope3, int n, int delta) {
    fn_attn_prep_body<true>(P, POS0, QW, KW, IW, INV, Q, reinterpret_cast<__nv_bfloat16*>(KC), reinterpret_cast<__nv_bfloat16*>(VC), IQ, IKC, eps, PW, NQ, NKV, NI, IHD, HALF, rope3, n, delta);
}

// glue._attn_gate, program (row, head) of HD threads: bf16(o / (1 + e^-gate)) and its 32-group sums.
extern "C" __global__ void __launch_bounds__(256) fn_attn_gate(const __nv_bfloat16* __restrict__ O, const __nv_bfloat16* __restrict__ P,
                                                               __nv_bfloat16* __restrict__ OUT, float* __restrict__ XS, int PW, int NQ) {
    const int r = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    const float o = bf(O[(static_cast<long long>(r) * NQ + h) * HD + d]);
    const float g = bf(P[static_cast<long long>(r) * PW + h * 2 * HD + HD + d]);
    const float out = rbf(__fdiv_rn(o, __fadd_rn(1.0f, tf_exp(-g))));
    OUT[static_cast<long long>(r) * NQ * HD + h * HD + d] = __float2bfloat16_rn(out);
    const float sum = warp_sum(out);
    if ((d & 31) == 0) XS[static_cast<long long>(r) * (NQ * HD / 32) + h * (HD / 32) + d / 32] = sum;
}

// attention._pool, a block a completed block of RATIO indexer keys: their mean, rounded, normed and roped at its start.
__device__ __forceinline__ void fn_qsa_pool_body(const __nv_bfloat16* __restrict__ IKC, __nv_bfloat16* __restrict__ POOLED,
                                                              const int* __restrict__ POS0, const float* __restrict__ W,
                                                              const float* __restrict__ INV, float eps, int R, int DI, int HALF,
                                                              int RATIO, const int* rope3, int n, int delta) {
    __shared__ float scratch[32];
    __shared__ float xs[256];
    const int p0 = POS0[0];
    const int b = p0 / RATIO + blockIdx.x, d = threadIdx.x;
    if (RATIO * b + RATIO > p0 + R) return;
    float acc = bf(IKC[static_cast<long long>(RATIO) * b * DI + d]);
    for (int k = 1; k < RATIO; ++k) acc = __fadd_rn(acc, bf(IKC[(static_cast<long long>(RATIO) * b + k) * DI + d]));
    const float x = rbf(__fdiv_rn(acc, static_cast<float>(RATIO)));
    xs[d] = x;
    const float rinv = inv_rms(block_sum(__fmul_rn(x, x), scratch), DI, eps);
    const int p = partner(d, HALF);
    const float xn = rbf(__fmul_rn(__fmul_rn(x, rinv), W[d]));
    const float xpn = rbf(__fmul_rn(__fmul_rn(xs[p], rinv), W[p]));
    POOLED[static_cast<long long>(b) * DI + d] = __float2bfloat16_rn(rope(xn, xpn, d, HALF, media_pos(RATIO * b, d, HALF, rope3, n, delta), INV));
}

extern "C" __global__ void __launch_bounds__(128) fn_qsa_pool(const __nv_bfloat16* __restrict__ IKC, __nv_bfloat16* __restrict__ POOLED,
                                                              const int* __restrict__ POS0, const float* __restrict__ W,
                                                              const float* __restrict__ INV, float eps, int R, int DI, int HALF,
                                                              int RATIO) {
    fn_qsa_pool_body(IKC, POOLED, POS0, W, INV, eps, R, DI, HALF, RATIO, nullptr, 0, 0);
}

extern "C" __global__ void __launch_bounds__(128) fn_qsa_pool_media(const __nv_bfloat16* __restrict__ IKC, __nv_bfloat16* __restrict__ POOLED,
                                                              const int* __restrict__ POS0, const float* __restrict__ W,
                                                              const float* __restrict__ INV, float eps, int R, int DI, int HALF,
                                                              int RATIO, const int* rope3, int n, int delta) {
    fn_qsa_pool_body(IKC, POOLED, POS0, W, INV, eps, R, DI, HALF, RATIO, rope3, n, delta);
}

// attention._scores, program (row, 64 blocks): the heads' sum of relu(q . k) / sqrt(DI), rows past TOP (DI 128).
extern "C" __global__ void __launch_bounds__(256) fn_qsa_scores(const __nv_bfloat16* __restrict__ IQ, const __nv_bfloat16* __restrict__ POOLED,
                                                                const int* __restrict__ POS0, float* __restrict__ SC, int NB, int HI,
                                                                int RATIO, int TOP) {
    constexpr int DI = 128;
    __shared__ float qs[8][DI];
    const int r = blockIdx.x, j = blockIdx.y;
    const int complete = (POS0[0] + r + 1) / RATIO;
    if (complete <= TOP || j * 64 >= complete) return;
    for (int e = threadIdx.x; e < HI * DI; e += blockDim.x) qs[e / DI][e % DI] = bf(IQ[static_cast<long long>(r) * HI * DI + e]);
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const float scale = __fsqrt_rn(static_cast<float>(DI));
    uint2 raw[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int b = j * 64 + warp * 8 + i;
        raw[i] = b < complete ? *reinterpret_cast<const uint2*>(POOLED + static_cast<long long>(b) * DI + 4 * lane) : make_uint2(0, 0);
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int b = j * 64 + warp * 8 + i;
        if (b >= complete) break;
        const __nv_bfloat16* kb = reinterpret_cast<const __nv_bfloat16*>(&raw[i]);
        float total = 0.0f;
        for (int h = 0; h < HI; ++h) {
            float part = 0.0f;
#pragma unroll
            for (int e = 0; e < 4; ++e) part = __fmaf_rn(bf(kb[e]), qs[h][4 * lane + e], part);
            total = __fadd_rn(total, fmaxf(warp_sum(part), 0.0f));
        }
        if (lane == 0) SC[static_cast<long long>(r) * NB + b] = __fdiv_rn(total, scale);
    }
}

// attention._chunks, program (row, KV head, chunk): the chunk's keys TK at a time, online softmax, partial o, m, l.
// KV8 (ONE only): KC and VC are cuda_kv.zig's fp8 rows; a tile's codes are widened to bf16 (times each key's scale, exact)
// into the same shared rows, so everything after the loads is the bf16 kernel's, on the dequantized keys and values.
template <int NB, bool ONE = false, bool FUSE = false, bool KV8 = false>
__device__ __forceinline__ void attn_chunks(const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ KC,
                                            const __nv_bfloat16* __restrict__ VC, const int* __restrict__ POS0,
                                            float* __restrict__ PO, float* __restrict__ PM, float* __restrict__ PL,
                                            const int* __restrict__ IDS, const int* __restrict__ NKR, const int* __restrict__ SPR,
                                            int H, int HK, int G, int CH, int NCH, float SCALE, int IDW, int QSA) {
    static_assert(!KV8 || ONE, "fp8 rows are staged one buffer at a time");
    extern __shared__ __align__(16) unsigned char smem_raw[];
    typedef __nv_bfloat16 Row[HD + 8];
    Row* qs = reinterpret_cast<Row*>(smem_raw);                  // [QR] the row's query heads
    Row* tiles_kv = qs + QR;                                     // [NB][2][TK] each buffer's keys, then values
    // ONE: one [TK] buffer, the tile's keys and then, once the scores are taken, its values (fewer bytes, more blocks)
    __nv_bfloat16 (*ps)[TK + 8] = reinterpret_cast<__nv_bfloat16 (*)[TK + 8]>(tiles_kv + (ONE ? TK : NB * 2 * TK));
    __shared__ float red_m[4][QR], red_l[4][QR];
    __shared__ int key_ids[KV8 ? 2 : NB][TK];
    const int r = blockIdx.x, hk = blockIdx.y, c = blockIdx.z;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    int n = POS0[0] + r + 1;
    bool sparse = false;
    if (QSA) {
        sparse = SPR[r] != 0;
        if (sparse) n = NKR[r];
    }
    // FUSE: one program a row takes every chunk in order and merges them as fn_attn_merge does (no partials)
    const int c_end = FUSE ? (n + CH - 1) / CH : c + 1;
    if (c * CH >= n) return;
    int start = c * CH;
    for (int e = threadIdx.x; e < QR * HD / 8; e += blockDim.x) {
        const int i = e / (HD / 8), v = e % (HD / 8);
        qmm_frag::cp16z(&qs[i][8 * v], Q + (static_cast<long long>(r) * H + hk * G + min(i, G - 1)) * HD + 8 * v, i < G);
    }
    // a tile's key ids, then its keys and values staged as two cp.async groups
    auto load = [&](int t0, int buf) {
        if (threadIdx.x < TK) {
            const int ki = start + t0 + threadIdx.x;
            key_ids[buf][threadIdx.x] = ki < n ? (sparse ? IDS[static_cast<long long>(r) * IDW + ki] : ki) : -1;
        }
        __syncthreads();
        for (int pass = 0; pass < 2; ++pass) {
            const __nv_bfloat16* src = pass == 0 ? KC : VC;
            Row* dst = tiles_kv + (buf * 2 + pass) * TK;
            for (int e = threadIdx.x; e < TK * HD / 8; e += blockDim.x) {
                const int k = e / (HD / 8), v = e % (HD / 8), key = key_ids[buf][k];
                qmm_frag::cp16z(&dst[k][8 * v], src + (static_cast<long long>(max(key, 0)) * HK + hk) * HD + 8 * v, key >= 0);
            }
            qmm_frag::commit();
        }
    };
    // KV8: a thread's four 16-code items a tile and pass (blockDim 128: a tile is TK x HD / 16 items), their codes and the key's
    // scale held in registers: the next tile's loads are issued before this tile's math, so their latency hides behind it. An
    // absent key is zero codes, scale 0.
    uint4 raw_k[4], raw_v[4];
    float raw_sk[4], raw_sv[4];
    auto fetch_kv8 = [&](int ids_buf, int pass) {
        const unsigned char* kc = reinterpret_cast<const unsigned char*>(KC);
        const unsigned char* vc = reinterpret_cast<const unsigned char*>(VC);
#pragma unroll
        for (int i = 0; i < TK * (HD / 16) / 128; ++i) {
            const int e = threadIdx.x + i * 128, k = e / (HD / 16), cc16 = e % (HD / 16), key = key_ids[ids_buf][k];
            uint4 w = make_uint4(0u, 0u, 0u, 0u);
            float sc = 0.0f;
            if (key >= 0) {
                const long long slot = static_cast<long long>(key) * HK + hk;
                const unsigned char* krow = kc + slot * KS;
                w = *reinterpret_cast<const uint4*>((pass == 0 ? krow : vc + slot * HD) + 16 * cc16);
                sc = *reinterpret_cast<const float*>(krow + HD + 4 * pass);
            }
            if (pass == 0) { raw_k[i] = w; raw_sk[i] = sc; } else { raw_v[i] = w; raw_sv[i] = sc; }
        }
    };
    // the registers' tile widened into the one shared buffer (keys, pass 0, or values, pass 1)
    auto widen_kv8 = [&](int pass) {
#pragma unroll
        for (int i = 0; i < TK * (HD / 16) / 128; ++i) {
            const int e = threadIdx.x + i * 128, k = e / (HD / 16), cc16 = e % (HD / 16);
            const uint4 w = pass == 0 ? raw_k[i] : raw_v[i];
            const float sc = pass == 0 ? raw_sk[i] : raw_sv[i];
            *reinterpret_cast<uint4*>(&tiles_kv[k][16 * cc16]) = kv8_widen8(w.x, w.y, sc);
            *reinterpret_cast<uint4*>(&tiles_kv[k][16 * cc16 + 8]) = kv8_widen8(w.z, w.w, sc);
        }
    };
    // a tile's key ids (-1 past the context)
    auto id_of = [&](int tile_t0, int k) {
        const int ki = start + tile_t0 + k;
        return ki < n ? (sparse ? IDS[static_cast<long long>(r) * IDW + ki] : ki) : -1;
    };
    // ONE: the keys (pass 0) or the values (pass 1) of the tile whose ids are in key_ids[0], into the one buffer
    auto load_one = [&](int pass) {
        const __nv_bfloat16* src = pass == 0 ? KC : VC;
        for (int e = threadIdx.x; e < TK * HD / 8; e += blockDim.x) {
            const int k = e / (HD / 8), v = e % (HD / 8), key = key_ids[0][k];
            qmm_frag::cp16z(&tiles_kv[k][8 * v], src + (static_cast<long long>(max(key, 0)) * HK + hk) * HD + 8 * v, key >= 0);
        }
        qmm_frag::commit();
    };
    float mm[2] = {-INFINITY, -INFINITY}, ll[2] = {0.0f, 0.0f}, oo[FUSE ? 8 : 1][4] = {};
    for (int cc = c; cc < c_end; ++cc) {
    start = cc * CH;
    float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.0f, 0.0f};
    float o[8][4] = {};
    const int tiles = min(n - start, CH);
    if (NB == 2) load(0, 0);
    if constexpr (KV8) {                                   // the first tile's ids, then its codes into registers
        if (threadIdx.x < TK) key_ids[0][threadIdx.x] = id_of(0, threadIdx.x);
        __syncthreads();
        fetch_kv8(0, 0);
        fetch_kv8(0, 1);
        qmm_frag::commit();                                // the group holding the queries' copies
    }
    for (int t0 = 0, it = 0; t0 < tiles; t0 += TK, ++it) {
        const int buf = NB == 2 ? it & 1 : 0;
        const bool ahead = NB == 2 && t0 + TK < tiles;  // the next tile's loads go out before this tile's math
        const bool more = t0 + TK < tiles;
        if constexpr (KV8) {
            // this tile's codes are in registers (issued an iteration ago): its keys widen into the buffer, the next tile's ids beside
            const int next_id = more && threadIdx.x < TK ? id_of(t0 + TK, threadIdx.x) : -1;
            widen_kv8(0);
            if (more && threadIdx.x < TK) key_ids[(it & 1) ^ 1][threadIdx.x] = next_id;
        } else if constexpr (ONE) {
            if (threadIdx.x < TK) {
                const int ki = start + t0 + threadIdx.x;
                key_ids[0][threadIdx.x] = ki < n ? (sparse ? IDS[static_cast<long long>(r) * IDW + ki] : ki) : -1;
            }
            __syncthreads();
            load_one(0);
        } else {
            if (NB == 1) load(t0, 0);
            if (ahead) load(t0 + TK, buf ^ 1);
        }
        Row* kt = tiles_kv + (ONE ? 0 : buf * 2 * TK);
        Row* vt = ONE ? kt : kt + TK;
        const int* keys = key_ids[KV8 ? (it & 1) : buf];
        if constexpr (ONE) qmm_frag::wait<0>();
        else if (ahead) qmm_frag::wait<3>(); else qmm_frag::wait<1>();
        __syncthreads();
        if constexpr (KV8) {
            if (more) fetch_kv8((it & 1) ^ 1, 0);           // the next tile's keys load during this tile's scores
        }
        constexpr int JT = TK / 32;  // a warp's 8-key tiles
        float s[JT][4] = {};
#pragma unroll 4
        for (int ks = 0; ks < HD / 16; ks += 2) {
            uint32_t a0[4], a1[4];
            qmm_frag::ldmatrix4(a0, &qs[lane % 16][ks * 16 + (lane / 16) * 8]);
            qmm_frag::ldmatrix4(a1, &qs[lane % 16][(ks + 1) * 16 + (lane / 16) * 8]);
#pragma unroll
            for (int jt = 0; jt < JT; ++jt) {
                uint32_t b[4];  // two k16 steps of the tile's 8 keys
                qmm_frag::ldmatrix4(b, &kt[warp * 8 * JT + jt * 8 + lane % 8][ks * 16 + (lane / 8) * 8]);
                qmm_frag::mma(s[jt], a0, b[0], b[1]);
                qmm_frag::mma(s[jt], a1, b[2], b[3]);
            }
        }
        if constexpr (ONE) {                              // every warp has read the keys: the values take their place
            __syncthreads();
            if constexpr (KV8) {
                widen_kv8(1);
                if (more) fetch_kv8((it & 1) ^ 1, 1);
            } else {
                load_one(1);
            }
        }
        float tile_m[2] = {-INFINITY, -INFINITY};
#pragma unroll
        for (int jt = 0; jt < JT; ++jt)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const bool valid = keys[warp * 8 * JT + jt * 8 + 2 * t + (e & 1)] >= 0;
                s[jt][e] = valid ? __fmul_rn(s[jt][e], SCALE) : -INFINITY;
                tile_m[e >> 1] = fmaxf(tile_m[e >> 1], s[jt][e]);
            }
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            tile_m[h] = fmaxf(tile_m[h], __shfl_xor_sync(0xffffffffu, tile_m[h], 1));
            tile_m[h] = fmaxf(tile_m[h], __shfl_xor_sync(0xffffffffu, tile_m[h], 2));
            if (t == 0) red_m[warp][g + 8 * h] = tile_m[h];
        }
        __syncthreads();
        float next_m[2], alpha[2], rows[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int row = g + 8 * h;
            const float tm = fmaxf(fmaxf(red_m[0][row], red_m[1][row]), fmaxf(red_m[2][row], red_m[3][row]));
            const bool active = tm != -INFINITY;
            next_m[h] = active ? fmaxf(m[h], tm) : m[h];
            alpha[h] = active ? (m[h] == -INFINITY ? 0.0f : tf_exp(__fsub_rn(m[h], next_m[h]))) : 1.0f;
            rows[h] = 0.0f;
        }
#pragma unroll
        for (int jt = 0; jt < JT; ++jt)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int h = e >> 1;
                const float p = s[jt][e] == -INFINITY ? 0.0f : tf_exp(__fsub_rn(s[jt][e], next_m[h]));
                rows[h] = __fadd_rn(rows[h], p);
                ps[g + 8 * h][warp * 8 * JT + jt * 8 + 2 * t + (e & 1)] = __float2bfloat16_rn(p);
            }
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            rows[h] = __fadd_rn(rows[h], __shfl_xor_sync(0xffffffffu, rows[h], 1));
            rows[h] = __fadd_rn(rows[h], __shfl_xor_sync(0xffffffffu, rows[h], 2));
            if (t == 0) red_l[warp][g + 8 * h] = rows[h];
        }
        if (ONE) qmm_frag::wait<0>(); else if (ahead) qmm_frag::wait<2>(); else qmm_frag::wait<0>();
        __syncthreads();
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int row = g + 8 * h;
            const float sum = __fadd_rn(__fadd_rn(__fadd_rn(red_l[0][row], red_l[1][row]), red_l[2][row]), red_l[3][row]);
            l[h] = __fmaf_rn(l[h], alpha[h], sum);
            m[h] = next_m[h];
        }
#pragma unroll
        for (int q = 0; q < 8; ++q)
#pragma unroll
            for (int e = 0; e < 4; ++e) o[q][e] = __fmul_rn(o[q][e], alpha[e >> 1]);
        for (int ks = 0; ks < TK / 16; ++ks) {
            uint32_t a[4];
            qmm_frag::ldmatrix4(a, &ps[lane % 16][ks * 16 + (lane / 16) * 8]);
#pragma unroll
            for (int q = 0; q < 8; q += 2) {
                uint32_t b[4];  // two 8-column tiles of the values, transposed into the B layout
                ldmatrix4t(b, &vt[ks * 16 + lane % 16][warp * 64 + q * 8 + (lane / 16) * 8]);
                qmm_frag::mma(o[q], a, b[0], b[1]);
                qmm_frag::mma(o[q + 1], a, b[2], b[3]);
            }
        }
        __syncthreads();
    }
    if constexpr (FUSE) {                                  // fn_attn_merge's step for this chunk, element by element
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const bool active = l[h] > 0.0f;
            const float next = active ? fmaxf(mm[h], m[h]) : mm[h];
            const float a = active ? (mm[h] == -INFINITY ? 0.0f : tf_exp(__fsub_rn(mm[h], next))) : 1.0f;
            const float bb = active ? tf_exp(__fsub_rn(m[h], next)) : 0.0f;
#pragma unroll
            for (int q = 0; q < 8; ++q)
#pragma unroll
                for (int e = 0; e < 2; ++e) oo[FUSE ? q : 0][2 * h + e] = __fmaf_rn(o[q][2 * h + e], bb, __fmul_rn(oo[FUSE ? q : 0][2 * h + e], a));
            ll[h] = __fmaf_rn(l[h], bb, __fmul_rn(ll[h], a));
            mm[h] = next;
        }
        continue;
    }
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int i = g + 8 * h;
        if (i >= G) continue;
        const long long base = (static_cast<long long>(r) * NCH + cc) * H + hk * G + i;
#pragma unroll
        for (int q = 0; q < 8; ++q) {
            float2 v = make_float2(o[q][2 * h], o[q][2 * h + 1]);
            *reinterpret_cast<float2*>(PO + base * HD + warp * 64 + q * 8 + 2 * t) = v;
        }
        if (warp == 0 && t == 0) {
            PM[base] = m[h];
            PL[base] = l[h];
        }
    }
    }
    if constexpr (FUSE) {                                  // o / l, rounded once, as fn_attn_merge writes OUT
        __nv_bfloat16* OUT = reinterpret_cast<__nv_bfloat16*>(PO);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int i = g + 8 * h;
            if (i >= G) continue;
            const long long row = (static_cast<long long>(r) * H + hk * G + i) * HD;
#pragma unroll
            for (int q = 0; q < 8; ++q)
#pragma unroll
                for (int e = 0; e < 2; ++e)
                    OUT[row + warp * 64 + q * 8 + 2 * t + e] = __float2bfloat16_rn(__fdiv_rn(oo[q][2 * h + e], ll[h]));
        }
    }
}

// attn_chunks one tile at a time (prompt chunks: two blocks a multiprocessor).
extern "C" __global__ void __launch_bounds__(128) fn_attn_chunks(const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ KC,
                                                                 const __nv_bfloat16* __restrict__ VC, const int* __restrict__ POS0,
                                                                 float* __restrict__ PO, float* __restrict__ PM, float* __restrict__ PL,
                                                                 const int* __restrict__ IDS, const int* __restrict__ NKR,
                                                                 const int* __restrict__ SPR, int H, int HK, int G, int CH, int NCH,
                                                                 float SCALE, int IDW, int QSA) {
    attn_chunks<1>(Q, KC, VC, POS0, PO, PM, PL, IDS, NKR, SPR, H, HK, G, CH, NCH, SCALE, IDW, QSA);
}

// attn_chunks on one buffer, a row's chunks merged in place (prompt rows): OUT bf16 in PO's place, no partials.
extern "C" __global__ void __launch_bounds__(128) fn_attn_fused(const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ KC,
                                                                const __nv_bfloat16* __restrict__ VC, const int* __restrict__ POS0,
                                                                float* __restrict__ OUT, const int* __restrict__ IDS, const int* __restrict__ NKR,
                                                                const int* __restrict__ SPR, int H, int HK, int G, int CH,
                                                                float SCALE, int IDW, int QSA) {
    attn_chunks<1, true, true>(Q, KC, VC, POS0, OUT, nullptr, nullptr, IDS, NKR, SPR, H, HK, G, CH, 1, SCALE, IDW, QSA);
}

// attn_chunks with one buffer, keys then values (17 to 256 rows: prompt chunks and wide decode rounds), the same sums.
extern "C" __global__ void __launch_bounds__(128) fn_attn_chunks_one(const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ KC,
                                                                     const __nv_bfloat16* __restrict__ VC, const int* __restrict__ POS0,
                                                                     float* __restrict__ PO, float* __restrict__ PM, float* __restrict__ PL,
                                                                     const int* __restrict__ IDS, const int* __restrict__ NKR,
                                                                     const int* __restrict__ SPR, int H, int HK, int G, int CH, int NCH,
                                                                     float SCALE, int IDW, int QSA) {
    attn_chunks<1, true>(Q, KC, VC, POS0, PO, PM, PL, IDS, NKR, SPR, H, HK, G, CH, NCH, SCALE, IDW, QSA);
}

// fn_attn_chunks_one over fp8 caches (cuda_kv.zig): KC [cap, HK, KS] and VC [cap, HK, HD] bytes; every other argument the same.
extern "C" __global__ void __launch_bounds__(128) fn_attn_chunks_kv8(const __nv_bfloat16* __restrict__ Q, const unsigned char* __restrict__ KC,
                                                                     const unsigned char* __restrict__ VC, const int* __restrict__ POS0,
                                                                     float* __restrict__ PO, float* __restrict__ PM, float* __restrict__ PL,
                                                                     const int* __restrict__ IDS, const int* __restrict__ NKR,
                                                                     const int* __restrict__ SPR, int H, int HK, int G, int CH, int NCH,
                                                                     float SCALE, int IDW, int QSA) {
    attn_chunks<1, true, false, true>(Q, reinterpret_cast<const __nv_bfloat16*>(KC), reinterpret_cast<const __nv_bfloat16*>(VC), POS0, PO, PM, PL,
                                      IDS, NKR, SPR, H, HK, G, CH, NCH, SCALE, IDW, QSA);
}

// fn_attn_fused over fp8 caches.
extern "C" __global__ void __launch_bounds__(128) fn_attn_fused_kv8(const __nv_bfloat16* __restrict__ Q, const unsigned char* __restrict__ KC,
                                                                    const unsigned char* __restrict__ VC, const int* __restrict__ POS0,
                                                                    float* __restrict__ OUT, const int* __restrict__ IDS, const int* __restrict__ NKR,
                                                                    const int* __restrict__ SPR, int H, int HK, int G, int CH,
                                                                    float SCALE, int IDW, int QSA) {
    attn_chunks<1, true, true, true>(Q, reinterpret_cast<const __nv_bfloat16*>(KC), reinterpret_cast<const __nv_bfloat16*>(VC), POS0, OUT, nullptr,
                                     nullptr, IDS, NKR, SPR, H, HK, G, CH, 1, SCALE, IDW, QSA);
}

// attn_chunks with the next tile's loads in flight (a decode round's few rows): the same sums.
extern "C" __global__ void __launch_bounds__(128) fn_attn_chunks_ahead(const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ KC,
                                                                       const __nv_bfloat16* __restrict__ VC, const int* __restrict__ POS0,
                                                                       float* __restrict__ PO, float* __restrict__ PM, float* __restrict__ PL,
                                                                       const int* __restrict__ IDS, const int* __restrict__ NKR,
                                                                       const int* __restrict__ SPR, int H, int HK, int G, int CH, int NCH,
                                                                       float SCALE, int IDW, int QSA) {
    attn_chunks<2>(Q, KC, VC, POS0, PO, PM, PL, IDS, NKR, SPR, H, HK, G, CH, NCH, SCALE, IDW, QSA);
}

// attention._merge, program (row, head) of HD threads: the head's chunk partials merged in chunk order, o / l.
extern "C" __global__ void __launch_bounds__(256) fn_attn_merge(const float* __restrict__ PO, const float* __restrict__ PM,
                                                                const float* __restrict__ PL, const int* __restrict__ POS0,
                                                                __nv_bfloat16* __restrict__ OUT, const int* __restrict__ NKR,
                                                                const int* __restrict__ SPR, int H, int HK, int G, int CH, int NCH,
                                                                int QSA) {
    const int r = blockIdx.x, head = blockIdx.y, d = threadIdx.x;
    int n = POS0[0] + r + 1;
    if (QSA && SPR[r] != 0) n = NKR[r];
    const int chunks = (n + CH - 1) / CH;
    {
        float m = -INFINITY, l = 0.0f, o = 0.0f;
        for (int c = 0; c < chunks; ++c) {
            const long long base = (static_cast<long long>(r) * NCH + c) * H + head;
            const float cm = PM[base], cl = PL[base], co = PO[base * HD + d];
            const bool active = cl > 0.0f;
            const float next = active ? fmaxf(m, cm) : m;
            const float a = active ? (m == -INFINITY ? 0.0f : tf_exp(__fsub_rn(m, next))) : 1.0f;
            const float b = active ? tf_exp(__fsub_rn(cm, next)) : 0.0f;
            o = __fmaf_rn(co, b, __fmul_rn(o, a));
            l = __fmaf_rn(cl, b, __fmul_rn(l, a));
            m = next;
        }
        OUT[(static_cast<long long>(r) * H + head) * HD + d] = __float2bfloat16_rn(__fdiv_rn(o, l));
    }
}

#include "flashnext_qsa_tile.cuh"

// The indexer scores 8 rows a block, every head, the sums in fn_qsa_scores' order (its bits).
template __global__ void qsa_tile::scores<8, 4>(const __nv_bfloat16* __restrict__, const __nv_bfloat16* __restrict__,
    const int* __restrict__, float* __restrict__, int, int, int, int);
