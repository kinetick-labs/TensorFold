// EXL3 (ExLlamaV3) device code for the native Flash Next CUDA backend, vendored from the Python line so the
// engine can read a turboderp EXL3 pack (quant_method "exl3") without a Python interpreter at serve time.
//
// Source: upstream/python-0.6 src/tensorfold/cuda/exl3/{decode.cuh, linear.cu, experts_grouped.cuh, experts.cu}
// (MIT, Copyright (c) 2025 Turboderp). The device code is verbatim; the only edits are
//   - the ATen/c10 includes and the <algorithm> include, dropped (nothing here is torch);
//   - the at::Tensor host launchers, dropped, and their launch contract kept in the drivers below;
//   - two TORCH_CHECK(false, ...) tails, which cannot compile without torch, replaced by (void) expressions;
//   - the reference's anonymous namespaces given names (tf_exl3_lin, tf_exl3_exp): an anonymous namespace
//     mangles with a hash of the source path (cuobjdump shows _GLOBAL__N__b46678aa_10_fn_exl3_cu_...), which
//     would make the names the Zig launcher resolves depend on the build directory;
//   - the reference's per-codebook instantiation TUs (experts_cb{0,1,2}.cu) folded in, and instance drivers
//     added so one fatbin carries every kernel the family can ask for;
//   - group_kernel replaced by a multi-block histogram, exclusive scan and stable scatter (same member order).
// The reference's own namespaces are kept (tf_exl3 for the tile decoder, tf_exl3x for the expert GEMM); the whole
// translation unit sits in tf_fn_exl3, this tree's one-namespace-a-fatbin convention, and the Zig launcher
// resolves the kernels by the mangled names recorded in cuda_kernels.zig (cuobjdump -symbols of this fatbin).

#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace tf_fn_exl3 {

// ===============================================================================================================
// decode.cuh: the shared 16x16 tile decoder (namespace tf_exl3).
// ===============================================================================================================

// EXL3 tiles (format.py, after ExLlamaV3, MIT, Copyright (c) 2025 Turboderp): lane L decodes a tile's values 8L..8L+7 straight into its two mma.m16n8k16 B fragments, bit for bit.
namespace tf_exl3 {

enum Codebook : int { CB_3INST = 0, CB_MCG = 1, CB_MUL1 = 2 };

// 32-bit words of a tile: 256 values of K2 / 2 bits.
template <int K2>
__host__ __device__ constexpr int tile_words() {
    return 4 * K2;
}

// E(p): one past the last stream bit of value p's 16-bit window (format.py's stream_ends).
template <int K2>
__host__ __device__ constexpr int stream_end(int p) {
    return (K2 & 1) ? ((p + 1) * K2 - ((p + 1) & 1)) / 2 : (p + 1) * (K2 / 2);
}

// Words a lane reads for its 8 values: the most any lane's windows span (2, or 3 for 3.5 bits and 5 to 8 bits).
template <int K2>
__host__ __device__ constexpr int lane_words() {
    int most = 0;
    for (int l = 0; l < 32; ++l) {
        const int first = (stream_end<K2>(8 * l) - 16 + 128 * K2) % 32;
        const int need = first + stream_end<K2>(8 * l + 7) - stream_end<K2>(8 * l) + 16;
        most = need > most ? need : most;
    }
    return (most + 31) / 32;
}

// The index of the first word lane `lane` reads, and the bit offset of its first window in that word.
template <int K2>
__device__ __forceinline__ void lane_start(int lane, int& word, int& offset) {
    const int first = 4 * lane * K2 + K2 / 2 - 16 + 128 * K2;   // E(8 lane) - 16, made non-negative
    word = (first >> 5) % tile_words<K2>();
    offset = first & 31;
}

// This lane's words of a tile (tile: the tile's first 32-bit word; global or shared memory).
template <int K2>
__device__ __forceinline__ void load_lane_words(const uint32_t* tile, int lane, uint32_t (&w)[lane_words<K2>()]) {
    int word, offset;
    lane_start<K2>(lane, word, offset);
#pragma unroll
    for (int i = 0; i < lane_words<K2>(); ++i) w[i] = tile[(word + i) % tile_words<K2>()];
}

// The same from global memory through the read-only cache.
template <int K2>
__device__ __forceinline__ void ldg_lane_words(const uint32_t* tile, int lane, uint32_t (&w)[lane_words<K2>()]) {
    int word, offset;
    lane_start<K2>(lane, word, offset);
#pragma unroll
    for (int i = 0; i < lane_words<K2>(); ++i) w[i] = __ldg(tile + (word + i) % tile_words<K2>());
}

// Bits [d, d + 16) of the 96-bit big-endian stream hi:mid:lo (d a compile-time constant after unrolling).
__device__ __forceinline__ uint32_t window16(uint32_t hi, uint32_t mid, uint32_t lo, int d) {
    if (d <= 16) return (hi >> (16 - d)) & 0xffffu;
    if (d < 32) return __funnelshift_l(mid, hi, d) >> 16;
    if (d <= 48) return (mid >> (48 - d)) & 0xffffu;
    return __funnelshift_l(lo, mid, d - 32) >> 16;
}

// The 16-bit states of this lane's 8 values (8 lane + j, j = 0..7) from its words.
template <int K2>
__device__ __forceinline__ void lane_states(const uint32_t (&w)[lane_words<K2>()], int lane, uint32_t (&s)[8]) {
    int word, offset;
    lane_start<K2>(lane, word, offset);
    const uint32_t c = lane_words<K2>() > 2 ? w[lane_words<K2>() > 2 ? 2 : 0] : 0u;
    const uint32_t hi = __funnelshift_l(w[1], w[0], offset);   // the stream from the lane's first window on
    const uint32_t mid = __funnelshift_l(c, w[1], offset);
    const uint32_t lo = c << offset;
#pragma unroll
    for (int j = 0; j < 8; ++j) s[j] = window16(hi, mid, lo, stream_end<K2>(j) - stream_end<K2>(0));
}

// Two codebook values from two 16-bit states, as a half2 in a uint32 (the first state in the low half).
template <int CB>
__device__ __forceinline__ uint32_t decode2(uint32_t s0, uint32_t s1) {
    if constexpr (CB == CB_MUL1) {
        const uint32_t x0 = s0 * 0x83DCD12Du, x1 = s1 * 0x83DCD12Du;
        const uint32_t h0 = __dp4a(x0, 0x01010101u, 0x6400u);    // fp16 bits of 1024 + the byte sum
        const uint32_t h1 = __dp4a(x1, 0x01010101u, 0x6400u);
        const uint32_t hh = (h0 & 0xffffu) | (h1 << 16);
        const half2 r = __hfma2(*reinterpret_cast<const half2*>(&hh), __half2half2(__ushort_as_half(0x1eee)),
                                __half2half2(__ushort_as_half(0xc931)));
        return *reinterpret_cast<const uint32_t*>(&r);
    } else {
        uint32_t x0, x1;
        if constexpr (CB == CB_MCG) {
            x0 = s0 * 0xCBAC1FEDu;
            x1 = s1 * 0xCBAC1FEDu;
        } else {
            x0 = s0 * 89226354u + 64248484u;
            x1 = s1 * 89226354u + 64248484u;
        }
        x0 = (x0 & 0x8FFF8FFFu) ^ 0x3B603B60u;
        x1 = (x1 & 0x8FFF8FFFu) ^ 0x3B603B60u;
        const uint32_t lo = __byte_perm(x0, x1, 0x5410);
        const uint32_t hi = __byte_perm(x0, x1, 0x7632);
        const half2 r = __hadd2(*reinterpret_cast<const half2*>(&lo), *reinterpret_cast<const half2*>(&hi));
        return *reinterpret_cast<const uint32_t*>(&r);
    }
}

// One codebook value.
template <int CB>
__device__ __forceinline__ half decode1(uint32_t s) {
    const uint32_t v = decode2<CB>(s, s);
    return __ushort_as_half(static_cast<unsigned short>(v & 0xffffu));
}

// This lane's 8 values of a tile as the B fragments of the tile's two n8 halves (columns 0-7 in b0, 8-15 in b1).
template <int K2, int CB>
__device__ __forceinline__ void decode_lane(const uint32_t (&w)[lane_words<K2>()], int lane, uint32_t (&b0)[2],
                                            uint32_t (&b1)[2]) {
    uint32_t s[8];
    lane_states<K2>(w, lane, s);
    b0[0] = decode2<CB>(s[0], s[1]);
    b0[1] = decode2<CB>(s[2], s[3]);
    b1[0] = decode2<CB>(s[4], s[5]);
    b1[1] = decode2<CB>(s[6], s[7]);
}

// Row and column in the 16x16 tile (row = k, column = n) of this lane's value j (0..7).
__device__ __forceinline__ int value_row(int lane, int j) { return 2 * (lane & 3) + (j & 1) + 8 * ((j >> 1) & 1); }
__device__ __forceinline__ int value_col(int lane, int j) { return (lane >> 2) + 8 * (j >> 2); }

// mma.m16n8k16, fp16 inputs, fp32 accumulators: d += a @ b.
__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Unscaled Walsh-Hadamard transform of 128 fp32 values, 4 a lane, in a fixed butterfly order (multiply by 1/sqrt(128)).
__device__ __forceinline__ void fwht128(float (&v)[4], int lane) {
    const float a = v[0] + v[1], b = v[0] - v[1], c = v[2] + v[3], d = v[2] - v[3];
    v[0] = a + c;
    v[1] = b + d;
    v[2] = a - c;
    v[3] = b - d;
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float o = __shfl_xor_sync(0xffffffffu, v[j], m);
            v[j] = (lane & m) ? o - v[j] : v[j] + o;
        }
    }
}

constexpr float HAD_SCALE = 0.08838834764831845f;   // 1 / sqrt(128)

}  // namespace tf_exl3




// ===============================================================================================================
// linear.cu: the dense GEMMs. The reference wrapped this in an anonymous namespace; see the header for why it is
// named here. It uses decode.cuh's helpers directly, as the reference's `using namespace tf_exl3` did.
// ===============================================================================================================

namespace tf_exl3_lin {

using namespace tf_exl3;



// EXL3 linear, any codebook and width, 1-128 rows: a row's bits depend only on it (mma keeps rows apart, K ranges fixed by (K, N), fixed-order sums).


enum DType : int { F16 = 0, BF16 = 1, F32 = 2 };

__device__ __forceinline__ void load4(const void* p, int dtype, size_t i, float (&v)[4]) {
    if (dtype == F32) {
        const float4 u = *reinterpret_cast<const float4*>(static_cast<const float*>(p) + i);
        v[0] = u.x; v[1] = u.y; v[2] = u.z; v[3] = u.w;
    } else if (dtype == BF16) {
        const uint2 u = *reinterpret_cast<const uint2*>(static_cast<const __nv_bfloat16*>(p) + i);
        const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.x));
        const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.y));
        v[0] = a.x; v[1] = a.y; v[2] = b.x; v[3] = b.y;
    } else {
        const uint2 u = *reinterpret_cast<const uint2*>(static_cast<const half*>(p) + i);
        const float2 a = __half22float2(*reinterpret_cast<const half2*>(&u.x));
        const float2 b = __half22float2(*reinterpret_cast<const half2*>(&u.y));
        v[0] = a.x; v[1] = a.y; v[2] = b.x; v[3] = b.y;
    }
}

__device__ __forceinline__ void store4(void* p, int dtype, size_t i, const float (&v)[4]) {
    if (dtype == F32) {
        *reinterpret_cast<float4*>(static_cast<float*>(p) + i) = make_float4(v[0], v[1], v[2], v[3]);
    } else if (dtype == BF16) {
        __nv_bfloat162 a = __floats2bfloat162_rn(v[0], v[1]), b = __floats2bfloat162_rn(v[2], v[3]);
        uint2 u;
        u.x = *reinterpret_cast<uint32_t*>(&a);
        u.y = *reinterpret_cast<uint32_t*>(&b);
        *reinterpret_cast<uint2*>(static_cast<__nv_bfloat16*>(p) + i) = u;
    } else {
        half2 a = __floats2half2_rn(v[0], v[1]), b = __floats2half2_rn(v[2], v[3]);
        uint2 u;
        u.x = *reinterpret_cast<uint32_t*>(&a);
        u.y = *reinterpret_cast<uint32_t*>(&b);
        *reinterpret_cast<uint2*>(static_cast<half*>(p) + i) = u;
    }
}

// The finished outputs of one row's 128 columns from their fp32 sums (4 a lane): H / sqrt(128), * svh, + bias.
__device__ __forceinline__ void finish(float (&v)[4], int lane, const half* svh, const half* bias, int col) {
    fwht128(v, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        v[j] = v[j] * HAD_SCALE * __half2float(__ldg(svh + col + j));
        // Speculative loads must have a valid address even when the split-K result has no bias.
        const half bv = __ldg((bias ? bias : svh) + col + j);
        if (bias) v[j] += __half2float(bv);
    }
}

// A lane's words of one k step: at 1 and 2 bits the warp loads the step together and each lane takes its words by shuffle.
template <int K2>
__host__ __device__ constexpr bool step_shuffled() {
    return K2 == 2 || K2 == 4;
}

template <int K2>
__host__ __device__ constexpr int step_regs() {
    return step_shuffled<K2>() ? tile_words<K2>() / 4 : 8 * lane_words<K2>();
}

template <int K2>
__device__ __forceinline__ void load_step(const uint32_t* step, int lane, uint32_t (&raw)[step_regs<K2>()]) {
    constexpr int TW = tile_words<K2>(), LW = lane_words<K2>();
    if constexpr (step_shuffled<K2>()) {
#pragma unroll
        for (int c = 0; c < step_regs<K2>(); ++c) raw[c] = __ldg(step + c * 32 + lane);
    } else {
        int word, offset;
        lane_start<K2>(lane, word, offset);
#pragma unroll
        for (int j = 0; j < 8; ++j)
#pragma unroll
            for (int q = 0; q < LW; ++q) raw[j * LW + q] = __ldg(step + j * TW + (word + q) % TW);
    }
}

// Tile j's lane words from a loaded step; prev: the lane's first window starts in the word before its own.
template <int K2>
__device__ __forceinline__ void step_lane_words(const uint32_t (&raw)[step_regs<K2>()], int j, int lane, bool prev,
                                                uint32_t (&w)[lane_words<K2>()]) {
    constexpr int TW = tile_words<K2>(), LW = lane_words<K2>();
    if constexpr (step_shuffled<K2>()) {
        static_assert(LW == 2, "a lane's windows span two words at 1 and 2 bits");
        constexpr int LPW = 8 / K2;                  // lanes whose windows end in the same word
        const uint32_t r = raw[j * TW / 32];
        const int base = (j * TW) % 32;
        const uint32_t own = __shfl_sync(0xffffffffu, r, base + lane / LPW);
        const uint32_t before = __shfl_sync(0xffffffffu, r, base + (lane / LPW + TW - 1) % TW);
        w[0] = prev ? before : own;
        w[1] = prev ? own : 0u;                      // a window inside one word never reads w[1]
    } else {
#pragma unroll
        for (int q = 0; q < LW; ++q) w[q] = raw[j * LW + q];
    }
}

__global__ void __launch_bounds__(128) rot_in_kernel(const void* __restrict__ x, int x_dtype,
                                                     const half* __restrict__ suh, half* __restrict__ xh, int K) {
    const int blk = blockIdx.x * 4 + (threadIdx.x >> 5), row = blockIdx.y, lane = threadIdx.x & 31;
    if (blk * 128 >= K) return;
    const int k = blk * 128 + 4 * lane;
    float v[4], s[4];
    load4(x, x_dtype, (size_t)row * K + k, v);
    load4(suh, F16, k, s);
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] *= s[j];
    fwht128(v, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] *= HAD_SCALE;
    store4(xh, F16, (size_t)row * K + k, v);
}

template <int K2, int CB, int WK>
__global__ void __launch_bounds__(WK * 32) linear_kernel(
    const half* __restrict__ xh, const uint32_t* __restrict__ T, long long stride_k, long long stride_nb,
    const half* __restrict__ svh, const half* __restrict__ bias, void* __restrict__ y, int y_dtype,
    float* __restrict__ Z, int* __restrict__ counters, int M, int K, int N, int SK, int ldy) {
    constexpr int TW = tile_words<K2>();
    constexpr int LW = lane_words<K2>();
    extern __shared__ __align__(16) float red[];              // WK * RH * 128 floats
    __shared__ int last;
    const int RH = min(M, 8);                                 // rows of red a warp

    const int nb = blockIdx.x, split = blockIdx.y, NB = gridDim.x;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int per_warp = (K >> 4) / SK / WK;
    const int kt0 = split * (per_warp * WK) + warp * per_warp;
    const uint32_t* tiles = T + nb * stride_nb;
    const int col0 = nb * 128;
    bool prev = false;
    if constexpr (step_shuffled<K2>()) {
        int word, offset;
        lane_start<K2>(lane, word, offset);
        prev = word != lane / (8 / K2);
    }

    // MT 16-row m tiles share each decoded weight tile (a row's mma sums do not depend on the other rows: same bits)
    constexpr int MT = 2;
    for (int mb = 0; mb < M; mb += 16 * MT) {
        float acc[MT][8][2][4];
#pragma unroll
        for (int u = 0; u < MT; ++u)
#pragma unroll
            for (int i = 0; i < 8; ++i)
#pragma unroll
                for (int h = 0; h < 2; ++h)
#pragma unroll
                    for (int c = 0; c < 4; ++c) acc[u][i][h][c] = 0.f;

        // the walk's rows for the mma, clamped so rows past the pass read inside the buffer (their outputs are dropped)
        const half* x0[MT];
        const half* x1[MT];
#pragma unroll
        for (int u = 0; u < MT; ++u) {
            const int m0 = mb + 16 * u, Rm = M - m0;
            x0[u] = xh + (size_t)(m0 + (g < Rm ? g : Rm - 1)) * K;
            x1[u] = xh + (size_t)(m0 + (g + 8 < Rm ? g + 8 : Rm - 1)) * K;
        }
        const uint32_t* tile = tiles + (size_t)kt0 * stride_k;
        // up to 6 bits the next k step's words are loaded while this one is decoded; 7 and 8 bits load per tile
        constexpr bool PF = K2 <= 12;
        constexpr int SR = step_regs<K2>();
        uint32_t cur[PF ? SR : 1], nxt[PF ? SR : 1];
        if constexpr (PF) load_step<K2>(tile, lane, cur);
#pragma unroll 1
        for (int i = 0; i < per_warp; ++i) {
            const int kt = kt0 + i;
            uint32_t a[MT][4];
#pragma unroll
            for (int u = 0; u < MT; ++u) {
                if (mb + 16 * u < M) {
                    a[u][0] = __ldg(reinterpret_cast<const uint32_t*>(x0[u] + kt * 16 + 2 * t));
                    a[u][1] = __ldg(reinterpret_cast<const uint32_t*>(x1[u] + kt * 16 + 2 * t));
                    a[u][2] = __ldg(reinterpret_cast<const uint32_t*>(x0[u] + kt * 16 + 2 * t + 8));
                    a[u][3] = __ldg(reinterpret_cast<const uint32_t*>(x1[u] + kt * 16 + 2 * t + 8));
                }
            }
            if constexpr (PF) {
                if (i + 1 < per_warp) load_step<K2>(tile + (size_t)(i + 1) * stride_k, lane, nxt);
            }
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                uint32_t w[LW];
                if constexpr (PF) step_lane_words<K2>(cur, j, lane, prev, w);
                else ldg_lane_words<K2>(tile + (size_t)i * stride_k + j * TW, lane, w);
                uint32_t b0[2], b1[2];
                decode_lane<K2, CB>(w, lane, b0, b1);
#pragma unroll
                for (int u = 0; u < MT; ++u) {
                    if (mb + 16 * u < M) {
                        mma16816(acc[u][j][0], a[u], b0);
                        mma16816(acc[u][j][1], a[u], b1);
                    }
                }
            }
            if constexpr (PF) {
#pragma unroll
                for (int q = 0; q < SR; ++q) cur[q] = nxt[q];
            }
        }

#pragma unroll
      for (int u = 0; u < MT; ++u) {
        const int m0 = mb + 16 * u, pass = m0 >> 4;
        if (m0 >= M) break;
        const int R = min(16, M - m0);
        // the warps' sums, added in warp order, rows 0-7 of the pass and then rows 8-15
        for (int rlo = 0; rlo < R; rlo += 8) {
            const int rn = min(R - rlo, 8);
            __syncthreads();                         // red is reused by every half and pass
            if (g < RH) {
#pragma unroll
                for (int i = 0; i < 8; ++i)
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const int col = i * 16 + h * 8 + 2 * t;
                        *reinterpret_cast<float2*>(red + (warp * RH + g) * 128 + col) =
                            rlo ? make_float2(acc[u][i][h][2], acc[u][i][h][3]) : make_float2(acc[u][i][h][0], acc[u][i][h][1]);
                    }
            }
            __syncthreads();

            if (SK == 1) {
                for (int r = warp; r < rn; r += WK) {
                    float v[4];
                    const float4 u = *reinterpret_cast<const float4*>(red + r * 128 + 4 * lane);
                    v[0] = u.x; v[1] = u.y; v[2] = u.z; v[3] = u.w;
#pragma unroll
                    for (int w = 1; w < WK; ++w) {
                        const float4 q = *reinterpret_cast<const float4*>(red + (w * RH + r) * 128 + 4 * lane);
                        v[0] += q.x; v[1] += q.y; v[2] += q.z; v[3] += q.w;
                    }
                    finish(v, lane, svh, bias, col0 + 4 * lane);
                    store4(y, y_dtype, (size_t)(m0 + rlo + r) * ldy + col0 + 4 * lane, v);
                }
            } else {
                for (int idx = threadIdx.x; idx < rn * 32; idx += WK * 32) {
                    const int r = idx >> 5, c = 4 * (idx & 31);
                    float4 s = *reinterpret_cast<const float4*>(red + r * 128 + c);
#pragma unroll
                    for (int w = 1; w < WK; ++w) {
                        const float4 q = *reinterpret_cast<const float4*>(red + (w * RH + r) * 128 + c);
                        s.x += q.x; s.y += q.y; s.z += q.z; s.w += q.w;
                    }
                    *reinterpret_cast<float4*>(Z + ((size_t)split * M + m0 + rlo + r) * N + col0 + c) = s;
                }
            }
        }
        if (SK > 1) {
            __threadfence();
            __syncthreads();
            if (threadIdx.x == 0) last = atomicAdd(counters + pass * NB + nb, 1) == SK - 1;
            __syncthreads();
            if (last) {
                __threadfence();
                for (int r = warp; r < R; r += WK) {
                    const size_t at = ((size_t)m0 + r) * N + col0 + 4 * lane;
                    float4 s = __ldcg(reinterpret_cast<const float4*>(Z + at));
                    for (int q = 1; q < SK; ++q) {
                        const float4 u = __ldcg(reinterpret_cast<const float4*>(Z + (size_t)q * M * N + at));
                        s.x += u.x; s.y += u.y; s.z += u.z; s.w += u.w;
                    }
                    float v[4] = {s.x, s.y, s.z, s.w};
                    finish(v, lane, svh, bias, col0 + 4 * lane);
                    store4(y, y_dtype, (size_t)(m0 + r) * ldy + col0 + 4 * lane, v);
                }
                if (threadIdx.x == 0) counters[pass * NB + nb] = 0;   // every program of the block has arrived
            }
        }
        __syncthreads();                             // red is reused in the next pass
      }
    }
}

// W_q [K, N] fp16 from the trellis words (tile (kt, nt) at kt * stride_k + (nt / 8) * stride_nb): one warp per tile.
template <int K2, int CB>
__global__ void __launch_bounds__(32) unpack_kernel(const uint32_t* __restrict__ T, half* __restrict__ W, int N,
                                                    int64_t stride_k, int64_t stride_nb) {
    const int kt = blockIdx.y, nt = blockIdx.x, lane = threadIdx.x;
    uint32_t w[lane_words<K2>()];
    ldg_lane_words<K2>(T + kt * stride_k + (nt >> 3) * stride_nb + (nt & 7) * tile_words<K2>(), lane, w);
    uint32_t b[2][2];
    decode_lane<K2, CB>(w, lane, b[0], b[1]);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const uint32_t v = b[j >> 2][(j >> 1) & 1];
        const unsigned short h = (j & 1) ? (unsigned short)(v >> 16) : (unsigned short)(v & 0xffffu);
        W[(size_t)(kt * 16 + value_row(lane, j)) * N + nt * 16 + value_col(lane, j)] = __ushort_as_half(h);
    }
}



#define TF_EXL3_WIDTHS(X, CB) X(2, CB) X(4, CB) X(6, CB) X(8, CB) X(10, CB) X(12, CB) X(14, CB) X(16, CB)
#define TF_EXL3_ALL(X) TF_EXL3_WIDTHS(X, 0) TF_EXL3_WIDTHS(X, 1) TF_EXL3_WIDTHS(X, 2) X(3, 2) X(5, 2) X(7, 2)




// ---------------------------------------------------------------------------------------------------------------
// Instance drivers. The reference's host launchers marshalled at::Tensor; these keep their launch contract -- the
// grid, block and shared-memory shapes the Zig launcher reproduces -- and, by compiling the (K2, CB) switch, make
// nvcc emit every kernel instance the family can ask for. Never called, never on the forward path.
// ---------------------------------------------------------------------------------------------------------------

// exl3_linear_cuda (linear.cu:303): grid (N/128, SK), block WK*32, dynamic smem WK*min(M,8)*128*4 lifted past
// 48 KiB through cudaFuncSetAttribute; WK is 2, 4 or 8 and (K/16) must divide over SK*WK.
void exl3_linear_instances(const half* xh, const uint32_t* T, long long stride_k, long long stride_nb,
                           const half* svh, const half* bias, void* y, int y_dtype, float* Z, int* counters, int M,
                           int K, int N, int K2, int cb, int SK, int WK, cudaStream_t stream) {
    dim3 grid((unsigned)(N / 128), (unsigned)SK);
#define TF_LAUNCH(K2_, CB_)                                                                                            \
    if (K2 == K2_ && cb == CB_) {                                                                                      \
        auto kernel = WK == 2   ? linear_kernel<K2_, CB_, 2>                                                           \
                      : WK == 4 ? linear_kernel<K2_, CB_, 4>                                                           \
                                : linear_kernel<K2_, CB_, 8>;                                                          \
        const int smem = (int)(WK * (M < 8 ? M : 8) * 128 * sizeof(float));                                             \
        if (smem > 48 * 1024) cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);           \
        kernel<<<grid, (unsigned)(WK * 32), smem, stream>>>(xh, T, stride_k, stride_nb, svh, bias, y, y_dtype, Z,       \
                                                            counters, M, K, N, SK, N);                                  \
        return;                                                                                                        \
    }
    TF_EXL3_ALL(TF_LAUNCH)
#undef TF_LAUNCH
}

// exl3_unpack_cuda (linear.cu:333): grid (N/16, K/16), block 32, no smem; one warp a tile, W fp16 [K, N].
void exl3_unpack_instances(const uint32_t* T, half* W, int K, int N, int64_t stride_k, int64_t stride_nb, int K2,
                           int cb, cudaStream_t stream) {
    dim3 grid((unsigned)(N / 16), (unsigned)(K / 16));
#define TF_LAUNCH(K2_, CB_)                                                                                            \
    if (K2 == K2_ && cb == CB_) {                                                                                      \
        unpack_kernel<K2_, CB_><<<grid, 32, 0, stream>>>(T, W, N, stride_k, stride_nb);                                \
        return;                                                                                                        \
    }
    TF_EXL3_ALL(TF_LAUNCH)
#undef TF_LAUNCH
}




}  // namespace tf_exl3_lin



// ===============================================================================================================
// experts.cu: the routed experts' grouping, rotation and epilogues. It carries its own fwht128 and HAD_SCALE
// (bit-identical to decode.cuh's), so it cannot share tf_exl3's namespace.
// ===============================================================================================================

namespace tf_exl3_exp {



// EXL3 routed experts, any codebook and a width per expert: fixed-order splits, slots and butterflies, no atomics; 4-bit mcg matches GLM's kernel bit for bit.


constexpr float HAD_SCALE = 0.08838834764831845f;   // 1 / sqrt(128)

// Grouping: distinct experts (< E) in id order, members (row * 32 + slot) in pick order, -1 after the last.
// Three kernels, one stream: a tile histogram, an exclusive scan that compacts uids, then a stable scatter.
// Tile size matches cuda_exl3.zig group_tile. Counts may use atomics; write order does not.
constexpr int GROUP_TILE = 1024;

__device__ __forceinline__ int group_tiles(int n) { return (n + GROUP_TILE - 1) / GROUP_TILE; }

// work: hist[tiles * E], then totals[E], then place[E]. hist is zero on entry.
__global__ void group_hist_kernel(const int* __restrict__ pick, int* __restrict__ work, int n, int E) {
    const int tile = (int)blockIdx.x;
    const int begin = tile * GROUP_TILE;
    int end = begin + GROUP_TILE;
    if (end > n) end = n;
    int* hist = work + (size_t)tile * E;
    for (int i = begin + (int)threadIdx.x; i < end; i += (int)blockDim.x) {
        const int e = pick[i];
        if ((unsigned)e >= (unsigned)E) continue;
        atomicAdd(hist + e, 1);
    }
}

// Exclusive scan of each expert across tiles (in place), uids in id order, members filled with -1.
__global__ void group_scan_kernel(int* __restrict__ work, int* __restrict__ uids, int* __restrict__ ucount,
                                  int* __restrict__ members, int n, int E, int maxm) {
    const int tiles = group_tiles(n);
    int* totals = work + (size_t)tiles * E;
    int* place = totals + E;
    for (int e = (int)threadIdx.x; e < E; e += (int)blockDim.x) {
        int sum = 0;
        for (int t = 0; t < tiles; ++t) {
            int* bin = work + (size_t)t * E + e;
            const int c = *bin;
            *bin = sum;
            sum += c;
        }
        totals[e] = sum;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int u = 0;
        for (int e = 0; e < E; ++e) {
            if (totals[e] == 0) {
                place[e] = -1;
                continue;
            }
            uids[u] = e;
            place[e] = u;
            ++u;
        }
        ucount[0] = u;
    }
    __syncthreads();
    const size_t nfill = (size_t)ucount[0] * (size_t)maxm;
    for (size_t i = threadIdx.x; i < nfill; i += blockDim.x) members[i] = -1;
}

// One block a tile, picks in index order: rank is earlier tiles' count plus this tile's running count.
__global__ void group_scatter_kernel(const int* __restrict__ pick, const int* __restrict__ work,
                                     int* __restrict__ members, int n, int slots, int E, int maxm) {
    extern __shared__ int tf_group_local[];
    for (int e = (int)threadIdx.x; e < E; e += (int)blockDim.x) tf_group_local[e] = 0;
    __syncthreads();
    if (threadIdx.x != 0) return;
    const int tile = (int)blockIdx.x;
    const int begin = tile * GROUP_TILE;
    int end = begin + GROUP_TILE;
    if (end > n) end = n;
    const int* off = work + (size_t)tile * E;
    const int tiles = group_tiles(n);
    const int* place = work + (size_t)tiles * E + E;
    for (int i = begin; i < end; ++i) {
        const int e = pick[i];
        if ((unsigned)e >= (unsigned)E) continue;
        const int rank = off[e] + tf_group_local[e];
        tf_group_local[e] += 1;
        if (rank >= maxm) continue;
        members[(size_t)place[e] * maxm + rank] = (i / slots) * 32 + (i % slots);
    }
}

// n <= GROUP_TILE: one block does hist, compaction and the stable scatter in shared memory (no workspace, no memset).
// Same uids (id order), same member ids in pick order, same -1 pad as the three-kernel path.
__global__ void group_small_kernel(const int* __restrict__ pick, int* __restrict__ uids, int* __restrict__ ucount,
                                   int* __restrict__ members, int n, int slots, int E, int maxm) {
    extern __shared__ int tf_group_small[];
    int* cnt = tf_group_small;
    int* place = tf_group_small + E;
    __shared__ int su;
    for (int e = (int)threadIdx.x; e < E; e += (int)blockDim.x) cnt[e] = 0;
    __syncthreads();
    for (int i = (int)threadIdx.x; i < n; i += (int)blockDim.x) {
        const int e = pick[i];
        if ((unsigned)e < (unsigned)E) atomicAdd(cnt + e, 1);
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        const int lane = (int)threadIdx.x;
        int u = 0;
        for (int base = 0; base < E; base += 32) {
            const int e = base + lane;
            const bool hit = e < E && cnt[e] > 0;
            const unsigned m = __ballot_sync(0xffffffffu, hit);
            if (e < E) {
                if (hit) {
                    const int r = u + __popc(m & ((1u << lane) - 1u));
                    uids[r] = e;
                    place[e] = r;
                } else {
                    place[e] = -1;
                }
            }
            u += __popc(m);
        }
        if (lane == 0) { ucount[0] = u; su = u; }
    }
    __syncthreads();
    const int nfill = su * maxm;
    for (int i = (int)threadIdx.x; i < nfill; i += (int)blockDim.x) members[i] = -1;
    for (int e = (int)threadIdx.x; e < E; e += (int)blockDim.x) cnt[e] = 0;
    __syncthreads();
    if (threadIdx.x != 0) return;
    for (int i = 0; i < n; ++i) {
        const int e = pick[i];
        if ((unsigned)e >= (unsigned)E) continue;
        const int rank = cnt[e];
        cnt[e] = rank + 1;
        if (rank >= maxm) continue;
        members[(size_t)place[e] * maxm + rank] = (i / slots) * 32 + (i % slots);
    }
}

// Walsh-Hadamard transform of 128 values, 4 a lane, fixed butterfly order (strides 1, 2 in registers, 4..64 across lanes).
__device__ __forceinline__ void fwht128(float (&v)[4], int lane) {
    float a = v[0] + v[1], b = v[0] - v[1], c = v[2] + v[3], d = v[2] - v[3];
    v[0] = a + c; v[1] = b + d; v[2] = a - c; v[3] = b - d;
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float o = __shfl_xor_sync(0xffffffffu, v[j], m);
            v[j] = (lane & m) ? o - v[j] : v[j] + o;
        }
    }
}

template <typename T> __device__ __forceinline__ float to_f(T v);
template <> __device__ __forceinline__ float to_f<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }
template <> __device__ __forceinline__ float to_f<half>(half v) { return __half2float(v); }

// Program (member row, 128-block of K, matrix): Xh = fp16((x * suh) @ H) for gate and up of every routed slot (pick < E).
template <typename TIN>
__global__ void rot_in_kernel(const TIN* __restrict__ x, int x_stride, const int* __restrict__ pick,
                              const half* __restrict__ suh0, const half* __restrict__ suh1, half* __restrict__ out0,
                              half* __restrict__ out1, int K, int slots, int E) {
    const int p = blockIdx.x, blk = blockIdx.y, mat = blockIdx.z;
    const int row = p / slots;
    const int e = pick[p];
    if (e < 0 || e >= E) return;
    const int lane = threadIdx.x;
    const half* suh = (mat ? suh1 : suh0) + (size_t)e * K + blk * 128 + 4 * lane;
    const TIN* xr = x + (size_t)row * x_stride + blk * 128 + 4 * lane;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = to_f<TIN>(xr[j]) * __half2float(suh[j]);
    fwht128(v, lane);
    half* o = (mat ? out1 : out0) + (size_t)p * K + blk * 128 + 4 * lane;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * HAD_SCALE);
}

__device__ __forceinline__ float bf16r(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

// Program (member row, 128-block of the width): splits summed in order, rotated, * svh, SwiGLU (0: GLM's bf16 roundings, 1: fp32), then Xd = fp16((act * suh_d) @ H).
__global__ void gateup_epilogue_kernel(const float* __restrict__ Z, const int* __restrict__ pick,
                                       const half* __restrict__ svh_g, const half* __restrict__ svh_u,
                                       const half* __restrict__ suh_d, half* __restrict__ xd, int P, int N, int SK,
                                       int E, float limit, int act_mode) {
    const int warps = (int)blockDim.x >> 5, warp = (int)threadIdx.x >> 5, lane = (int)threadIdx.x & 31;
    const int p = (int)blockIdx.x * warps + warp, blk = (int)blockIdx.y;
    if (p >= P) return;
    const int e = pick[p];
    if (e < 0 || e >= E) return;
    const int n = blk * 128 + 4 * lane;
    float gv[4], uv[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float sg = 0.f, su = 0.f;
        for (int s = 0; s < SK; ++s) {
            sg += Z[((size_t)(0 * SK + s) * P + p) * N + n + j];
            su += Z[((size_t)(1 * SK + s) * P + p) * N + n + j];
        }
        gv[j] = sg;
        uv[j] = su;
    }
    fwht128(gv, lane);
    fwht128(uv, lane);
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float act;
        if (act_mode == 0) {
            float gg = fminf(bf16r(gv[j] * HAD_SCALE * __half2float(svh_g[(size_t)e * N + n + j])), limit);
            float uu = fminf(fmaxf(bf16r(uv[j] * HAD_SCALE * __half2float(svh_u[(size_t)e * N + n + j])), -limit),
                             limit);
            act = bf16r(bf16r(gg / (1.f + expf(-gg))) * uu);
        } else {
            float gg = fminf(gv[j] * HAD_SCALE * __half2float(svh_g[(size_t)e * N + n + j]), limit);
            float uu = fminf(fmaxf(uv[j] * HAD_SCALE * __half2float(svh_u[(size_t)e * N + n + j]), -limit), limit);
            act = gg / (1.f + expf(-gg)) * uu;
        }
        v[j] = act * __half2float(suh_d[(size_t)e * N + n + j]);
    }
    fwht128(v, lane);
    half* o = xd + (size_t)p * N + n;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * HAD_SCALE);
}

// Program (member row, 128-block of the model width): Y = (splits summed in order) @ H * svh_d, fp32.
__global__ void down_epilogue_kernel(const float* __restrict__ Z, const int* __restrict__ pick,
                                     const half* __restrict__ svh_d, float* __restrict__ y, int P, int D, int SK,
                                     int E) {
    const int warps = (int)blockDim.x >> 5, warp = (int)threadIdx.x >> 5, lane = (int)threadIdx.x & 31;
    const int p = (int)blockIdx.x * warps + warp, blk = (int)blockIdx.y;
    if (p >= P) return;
    const int e = pick[p];
    if (e < 0 || e >= E) return;
    const int n = blk * 128 + 4 * lane;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float s = 0.f;
        for (int k = 0; k < SK; ++k) s += Z[((size_t)k * P + p) * D + n + j];
        v[j] = s;
    }
    fwht128(v, lane);
    float* o = y + (size_t)p * D + n;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = v[j] * HAD_SCALE * __half2float(svh_d[(size_t)e * D + n + j]);
}

// out[r][d] = sum over slots in order of wts[r][k] * y[r * slots + k][d] (fp32, fma chain from 0).
__global__ void combine_kernel(const float* __restrict__ y, const float* __restrict__ wts, float* __restrict__ out,
                               int D, int slots) {
    const int r = blockIdx.x;
    const int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (d >= D) return;
    float acc = 0.f;
    for (int k = 0; k < slots; ++k) acc = fmaf(wts[r * slots + k], y[((size_t)r * slots + k) * D + d], acc);
    out[(size_t)r * D + d] = acc;
}

// down_epilogue_kernel then combine_kernel in one launch, the same arithmetic in the same order (the same bits).
__global__ void down_combine_kernel(const float* __restrict__ Z, const int* __restrict__ pick,
                                    const half* __restrict__ svh_d, float* __restrict__ y,
                                    const float* __restrict__ wts, float* __restrict__ out, int P, int D, int SK,
                                    int E, int slots) {
    __shared__ float4 part[32][32];                 // [slot][lane]: the slot's 4 outputs of the lane
    const int r = blockIdx.x, blk = blockIdx.y;
    const int k = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int n = blk * 128 + 4 * lane;
    const int p = r * slots + k;
    const int e = pick[p];
    float o[4];
    if (e >= 0 && e < E) {
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float s = 0.f;
            for (int q = 0; q < SK; ++q) s += Z[((size_t)q * P + p) * D + n + j];
            v[j] = s;
        }
        fwht128(v, lane);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            o[j] = v[j] * HAD_SCALE * __half2float(svh_d[(size_t)e * D + n + j]);
            y[(size_t)p * D + n + j] = o[j];
        }
    } else {
#pragma unroll
        for (int j = 0; j < 4; ++j) o[j] = y[(size_t)p * D + n + j];
    }
    part[k][lane] = make_float4(o[0], o[1], o[2], o[3]);
    __syncthreads();
    if (k != 0) return;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int q = 0; q < slots; ++q) {
        const float w = wts[r * slots + q];
        const float4 u = part[q][lane];
        acc[0] = fmaf(w, u.x, acc[0]);
        acc[1] = fmaf(w, u.y, acc[1]);
        acc[2] = fmaf(w, u.z, acc[2]);
        acc[3] = fmaf(w, u.w, acc[3]);
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) out[(size_t)r * D + n + j] = acc[j];
}




// exl3x_rot_in_cuda (experts.cu:312): grid (rows*slots, K/128, 2), block 32; the input's dtype picks the instance.
void exl3x_rot_in_instances(const __nv_bfloat16* xb, const half* xh, int x_stride, const int* pick,
                            const half* suh0, const half* suh1, half* out0, half* out1, int rows, int K, int slots,
                            int E, cudaStream_t stream) {
    dim3 grid((unsigned)(rows * slots), (unsigned)(K / 128), 2);
    rot_in_kernel<__nv_bfloat16><<<grid, 32, 0, stream>>>(xb, x_stride, pick, suh0, suh1, out0, out1, K, slots, E);
    rot_in_kernel<half><<<grid, 32, 0, stream>>>(xh, x_stride, pick, suh0, suh1, out0, out1, K, slots, E);
}

// The rest of the expert kernels' launch contract, for the Zig side:
//   group_hist_kernel              grid (tiles), block 256; tiles of GROUP_TILE picks, histogram in work
//   group_scan_kernel              1 block of 256: exclusive scan, uids in id order, members set to -1
//   group_scatter_kernel           grid (tiles), block 128, E*4 shared: stable scatter, same member ids
//   gateup_epilogue_kernel          grid (ceil(rows*slots/4), N/128), block 128 (4 warps, one pick a warp)
//   down_epilogue_kernel            grid (ceil(rows*slots/4), D/128), block 128 (4 warps, one pick a warp)
//   combine_kernel         (:183)  grid (rows, ceil(D/256)), block 256
//   down_combine_kernel    (:194)  grid (rows, D/128), block 32*slots (slots <= 32)
//   grouped_kernel (experts_grouped.cuh:191)  grid (nexp_max, N/(16*nt), mats*SK*mtiles), block W*32, no smem
//   dequant_kernel (experts_grouped.cuh:288)  grid (K/16, N/16), block 32


// down_combine_kernel with the combined row rounded to bf16 (the write-back's MODE 2 rounding) and no per-slot y
// stores: the write-back then takes it as a bf16 branch, so the same bits as down_epilogue + _hc_writeback MODE 2.
__global__ void down_combine_bf16_kernel(const float* __restrict__ Z, const int* __restrict__ pick,
                                         const half* __restrict__ svh_d, const float* __restrict__ y,
                                         const float* __restrict__ wts, __nv_bfloat16* __restrict__ out, int P, int D,
                                         int SK, int E, int slots) {
    __shared__ float4 part[32][32];
    const int r = blockIdx.x, blk = blockIdx.y;
    const int k = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int n = blk * 128 + 4 * lane;
    const int p = r * slots + k;
    const int e = pick[p];
    float o[4];
    if (e >= 0 && e < E) {
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float s = 0.f;
            for (int q = 0; q < SK; ++q) s += Z[((size_t)q * P + p) * D + n + j];
            v[j] = s;
        }
        fwht128(v, lane);
#pragma unroll
        for (int j = 0; j < 4; ++j) o[j] = v[j] * HAD_SCALE * __half2float(svh_d[(size_t)e * D + n + j]);
    } else {
#pragma unroll
        for (int j = 0; j < 4; ++j) o[j] = y[(size_t)p * D + n + j];
    }
    part[k][lane] = make_float4(o[0], o[1], o[2], o[3]);
    __syncthreads();
    if (k != 0) return;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int q = 0; q < slots; ++q) {
        const float w = wts[r * slots + q];
        const float4 u = part[q][lane];
        acc[0] = fmaf(w, u.x, acc[0]);
        acc[1] = fmaf(w, u.y, acc[1]);
        acc[2] = fmaf(w, u.z, acc[2]);
        acc[3] = fmaf(w, u.w, acc[3]);
    }
    __nv_bfloat162* o2 = reinterpret_cast<__nv_bfloat162*>(out + (size_t)r * D + n);
    o2[0] = __floats2bfloat162_rn(acc[0], acc[1]);
    o2[1] = __floats2bfloat162_rn(acc[2], acc[3]);
}

}  // namespace tf_exl3_exp



// ===============================================================================================================
// experts_grouped.cuh: the grouped expert GEMM (namespace tf_exl3x).
// ===============================================================================================================

// Grouped EXL3 expert GEMV (after ExLlamaV3, MIT, Copyright (c) 2025 Turboderp): rows stay independent, K ranges fixed by shape, warps summed in order.


namespace tf_exl3x {

// Two codebook values (CB 0 3inst, 1 mcg, 2 mul1) as a half2, bit-identical to ExLlamaV3's decode_3inst_2<cb>.
template <int CB>
__device__ __forceinline__ uint32_t cb_pair(uint32_t s0, uint32_t s1) {
    if constexpr (CB == 2) {
        const uint32_t x0 = s0 * 0x83DCD12Du, x1 = s1 * 0x83DCD12Du;
        const uint32_t sum0 = __dp4a(x0, 0x01010101u, 0x6400u);
        const uint32_t sum1 = __dp4a(x1, 0x01010101u, 0x6400u);
        const uint32_t hv = __byte_perm(sum0, sum1, 0x5410);
        half2 h = *reinterpret_cast<const half2*>(&hv);
        half2 r = __hfma2(h, __half2half2(__ushort_as_half(0x1eee)), __half2half2(__ushort_as_half(0xc931)));
        return *reinterpret_cast<uint32_t*>(&r);
    } else {
        uint32_t x0, x1;
        if constexpr (CB == 1) {
            x0 = s0 * 0xCBAC1FEDu;
            x1 = s1 * 0xCBAC1FEDu;
        } else {
            x0 = s0 * 89226354u + 64248484u;
            x1 = s1 * 89226354u + 64248484u;
        }
        x0 = (x0 & 0x8FFF8FFFu) ^ 0x3B603B60u;
        x1 = (x1 & 0x8FFF8FFFu) ^ 0x3B603B60u;
        uint32_t lo = __byte_perm(x0, x1, 0x5410);
        uint32_t hi = __byte_perm(x0, x1, 0x7632);
        half2 r = __hadd2(*reinterpret_cast<half2*>(&lo), *reinterpret_cast<half2*>(&hi));
        return *reinterpret_cast<uint32_t*>(&r);
    }
}

// K2 half-bits a value: a tile is 4 * K2 words; a lane's eight windows fall in NG runs of GV within two words.
template <int K2>
struct Fmt {
    static constexpr int TW = 4 * K2;
    static constexpr int LW = (TW + 31) / 32;
    // windows sharing one 64-bit merge (tests/cuda/test_exl3_experts.py checks every K2)
    static constexpr int GV = (K2 >= 13) ? 2 : ((K2 == 7 || (K2 >= 9 && K2 <= 12) || K2 == 16) ? 4 : 8);
    static constexpr int NG = 8 / GV;
    __host__ __device__ static constexpr int end(int p) { return (p >> 1) * K2 + ((p & 1) ? K2 : (K2 >> 1)); }
    // right shift of window j of a run (run starts at an even position) relative to the run's last window
    __host__ __device__ static constexpr int off(int j) { return end(GV - 1) - end(j); }
};

template <int K2>
struct LaneMap {
    int hi[Fmt<K2>::NG], lo[Fmt<K2>::NG], sh[Fmt<K2>::NG];
    __device__ __forceinline__ explicit LaneMap(int lane) {
        constexpr int TW = Fmt<K2>::TW, GV = Fmt<K2>::GV;
#pragma unroll
        for (int g = 0; g < Fmt<K2>::NG; ++g) {
            const int last_end = Fmt<K2>::end(8 * lane + g * GV + GV - 1) + 128 * K2;
            const int hr = (last_end - 1) >> 5;
            hi[g] = hr % TW;
            lo[g] = (hr + TW - 1) % TW;
            sh[g] = (hr + 1) * 32 - last_end;
        }
    }
};

template <int LW>
__device__ __forceinline__ uint32_t fetch(const uint32_t (&w)[LW], int idx) {
    if constexpr (LW == 1) {
        return __shfl_sync(0xffffffffu, w[0], idx);
    } else {
        const uint32_t a = __shfl_sync(0xffffffffu, w[0], idx & 31);
        const uint32_t b = __shfl_sync(0xffffffffu, w[1], idx & 31);
        return idx < 32 ? a : b;
    }
}

// This lane's eight values of a tile as the B fragments of its two n8 halves.
template <int CB, int K2>
__device__ __forceinline__ void decode_tile(const uint32_t (&w)[Fmt<K2>::LW], const LaneMap<K2>& m, int lane,
                                            uint32_t (&b0)[2], uint32_t (&b1)[2]) {
    uint32_t st[8];
    if constexpr (K2 == 8) {
        // 4 bits: lane L's windows are exactly words L-1 and L (the GLM kernel's decode)
        const uint32_t p = __shfl_sync(0xffffffffu, w[0], (lane + 31) & 31);
        const uint32_t s = __funnelshift_r(w[0], p, 20);
        st[0] = (s >> 8) & 0xffffu;
        st[1] = (s >> 4) & 0xffffu;
        st[2] = s & 0xffffu;
        st[3] = w[0] >> 16;
        st[4] = (w[0] >> 12) & 0xffffu;
        st[5] = (w[0] >> 8) & 0xffffu;
        st[6] = (w[0] >> 4) & 0xffffu;
        st[7] = w[0] & 0xffffu;
    } else {
        constexpr int GV = Fmt<K2>::GV, NG = Fmt<K2>::NG;
#pragma unroll
        for (int g = 0; g < NG; ++g) {
            const uint32_t whi = fetch<Fmt<K2>::LW>(w, m.hi[g]);
            const uint32_t wlo = fetch<Fmt<K2>::LW>(w, m.lo[g]);
            const uint64_t mm = ((((uint64_t)wlo) << 32) | whi) >> m.sh[g];
#pragma unroll
            for (int j = 0; j < GV; ++j) st[g * GV + j] = (uint32_t)(mm >> Fmt<K2>::off(j)) & 0xffffu;
        }
    }
    b0[0] = cb_pair<CB>(st[0], st[1]);
    b0[1] = cb_pair<CB>(st[2], st[3]);
    b1[0] = cb_pair<CB>(st[4], st[5]);
    b1[1] = cb_pair<CB>(st[6], st[7]);
}

// Timing probe only (TF_EXL3_ABL_DECODE): the trellis decode's arithmetic out, every loaded word still feeding b0/b1.
template <int K2>
__device__ __forceinline__ void decode_tile_abl(const uint32_t (&w)[Fmt<K2>::LW], uint32_t (&b0)[2], uint32_t (&b1)[2]) {
    uint32_t x = 0;
#pragma unroll
    for (int l = 0; l < Fmt<K2>::LW; ++l) x ^= w[l];   // the loads stay live: the result uses every word (no dead load)
    b0[0] = x; b0[1] = x ^ 0x0101u;
    b1[0] = x ^ 0x0202u; b1[1] = x ^ 0x0404u;
}

__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ uint32_t load_pair(const half* x, bool ok) {
    return ok ? *reinterpret_cast<const uint32_t*>(x) : 0u;
}

template <int K2>
__device__ __forceinline__ void load_words(uint32_t (&dst)[Fmt<K2>::LW], const uint32_t* p, int lane) {
    constexpr int TW = Fmt<K2>::TW;
#pragma unroll
    for (int l = 0; l < Fmt<K2>::LW; ++l) {
        if constexpr ((TW % 32) == 0)
            dst[l] = __ldg(p + l * 32);
        else
            dst[l] = (l * 32 + lane < TW) ? __ldg(p + l * 32) : 0u;
    }
}

// One warp's k tiles [kt0, kt0 + nkt) of an expert matrix into acc, PF tiles in flight.
template <int CB, int K2, int NT, int PF, int ABL = 0>
__device__ __forceinline__ void warp_tiles(const uint32_t* __restrict__ T, int NTILES, int kt0, int nkt, int nt0,
                                           const half* x0, const half* x1, bool ok0, bool ok1, int lane,
                                           float (&acc)[NT][2][4]) {
    constexpr int TW = Fmt<K2>::TW, LW = Fmt<K2>::LW;
    const LaneMap<K2> map(lane);
    const size_t kstride = (size_t)NTILES * TW;
    const uint32_t* tp = T + ((size_t)kt0 * NTILES + nt0) * TW + lane;

    uint32_t pf[PF][NT][LW];
#pragma unroll
    for (int d = 0; d < PF; ++d)
        if (d < nkt)
#pragma unroll
            for (int i = 0; i < NT; ++i) load_words<K2>(pf[d][i], tp + d * kstride + i * TW, lane);

    for (int ib = 0; ib < nkt; ib += PF) {
#pragma unroll
        for (int d = 0; d < PF; ++d) {
            const int it = ib + d;
            if (it < nkt) {
                uint32_t w[NT][LW];
#pragma unroll
                for (int i = 0; i < NT; ++i)
#pragma unroll
                    for (int l = 0; l < LW; ++l) w[i][l] = pf[d][i][l];
                if (it + PF < nkt)
#pragma unroll
                    for (int i = 0; i < NT; ++i)
                        load_words<K2>(pf[d][i], tp + (size_t)(it + PF) * kstride + i * TW, lane);
                const int k = (kt0 + it) * 16;
                uint32_t a[4] = {load_pair(x0 + k, ok0), load_pair(x1 + k, ok1), load_pair(x0 + k + 8, ok0),
                                 load_pair(x1 + k + 8, ok1)};
#pragma unroll
                for (int i = 0; i < NT; ++i) {
                    uint32_t b0[2], b1[2];
                    if constexpr (ABL) decode_tile_abl<K2>(w[i], b0, b1);
                    else decode_tile<CB, K2>(w[i], map, lane, b0, b1);
                    mma16816(acc[i][0], a, b0);
                    mma16816(acc[i][1], a, b1);
                }
            }
        }
    }
}

// The wide arm's k loop: one decode a (n tile, k tile) feeds the MTL member tiles' mma passes (upstream's gm_kernel).
template <int CB, int K2, int PF, int MTL, int ABL = 0>
__device__ __forceinline__ void warp_tile_wide(const uint32_t* __restrict__ T, int NTILES, int kt0, int nkt, int nt,
                                               const half* const (&xb0)[MTL], const half* const (&xb1)[MTL],
                                               const bool (&ok0)[MTL], const bool (&ok1)[MTL], int live, int lane,
                                               float (&acc)[MTL][2][4]) {
    constexpr int TW = Fmt<K2>::TW, LW = Fmt<K2>::LW;
    const LaneMap<K2> map(lane);
    const size_t kstride = (size_t)NTILES * TW;
    const uint32_t* tp = T + ((size_t)kt0 * NTILES + nt) * TW + lane;

    uint32_t pf[PF][LW];
#pragma unroll
    for (int d = 0; d < PF; ++d)
        if (d < nkt) load_words<K2>(pf[d], tp + d * kstride, lane);

    for (int ib = 0; ib < nkt; ib += PF) {
#pragma unroll
        for (int d = 0; d < PF; ++d) {
            const int it = ib + d;
            if (it < nkt) {
                uint32_t w[LW];
#pragma unroll
                for (int l = 0; l < LW; ++l) w[l] = pf[d][l];
                if (it + PF < nkt) load_words<K2>(pf[d], tp + (size_t)(it + PF) * kstride, lane);
                uint32_t b0[2], b1[2];
                if constexpr (ABL) decode_tile_abl<K2>(w, b0, b1);
                else decode_tile<CB, K2>(w, map, lane, b0, b1);
                const int k = (kt0 + it) * 16;
#pragma unroll
                for (int tt = 0; tt < MTL; ++tt) {
                    if (tt < live) {
                        uint32_t a[4] = {load_pair(xb0[tt] + k, ok0[tt]), load_pair(xb1[tt] + k, ok1[tt]),
                                         load_pair(xb0[tt] + k + 8, ok0[tt]), load_pair(xb1[tt] + k + 8, ok1[tt])};
                        mma16816(acc[tt][0], a, b0);
                        mma16816(acc[tt][1], a, b1);
                    }
                }
            }
        }
    }
}


// The wide arm's staged k loop: one decode a (n tile, k tile) feeds the MTL member tiles' mma, A read from shared.
template <int CB, int K2, int PF, int MTL>
__device__ __forceinline__ void warp_tile_stage(const uint32_t* __restrict__ T, int NTILES, int kt0, int nkt, int nt,
                                                const uint32_t* __restrict__ As, int sw, int ktrel, int live, int lane,
                                                float (&acc)[MTL][2][4]) {
    constexpr int TW = Fmt<K2>::TW, LW = Fmt<K2>::LW;
    const LaneMap<K2> map(lane);
    const size_t kstride = (size_t)NTILES * TW;
    const uint32_t* tp = T + ((size_t)kt0 * NTILES + nt) * TW + lane;
    const int g = lane >> 2, t = lane & 3;

    uint32_t pf[PF][LW];
#pragma unroll
    for (int d = 0; d < PF; ++d)
        if (d < nkt) load_words<K2>(pf[d], tp + d * kstride, lane);

    for (int ib = 0; ib < nkt; ib += PF) {
#pragma unroll
        for (int d = 0; d < PF; ++d) {
            const int it = ib + d;
            if (it < nkt) {
                uint32_t w[LW];
#pragma unroll
                for (int l = 0; l < LW; ++l) w[l] = pf[d][l];
                if (it + PF < nkt) load_words<K2>(pf[d], tp + (size_t)(it + PF) * kstride, lane);
                uint32_t b0[2], b1[2];
                decode_tile<CB, K2>(w, map, lane, b0, b1);
                const int kw = (ktrel + it) * 8 + t;                     // this lane's word in the member row's slice
#pragma unroll
                for (int tt = 0; tt < MTL; ++tt) {
                    if (tt < live) {
                        const uint32_t* r0 = As + (size_t)(tt * 16 + g) * sw + kw;
                        const uint32_t* r1 = As + (size_t)(tt * 16 + g + 8) * sw + kw;
                        uint32_t a[4] = { r0[0], r1[0], r0[4], r1[4] };      // the +8 columns sit four words on
                        mma16816(acc[tt][0], a, b0);
                        mma16816(acc[tt][1], a, b1);
                    }
                }
            }
        }
    }
}

// The K2 values an instance covering [LO, HI] compiles (half-bits 2..16).
__host__ __device__ constexpr bool k2_supported(int k2) {
    return k2 >= 2 && k2 <= 16;
}

// upstream's gm_kernel (x3gm.cu), ported as an opt-in arm (TF_FLASHNEXT_X3_GM): the 128 member rows are the mma B

// operand, staged in shared through a cp.async ring and XOR-swizzled for ldsm4; the decoded weights are the mma A

// operand (upstream's decode_a); the block's tiles are split across the warps, so a weight tile is decoded once for

// 128 members. The split's k is walked in the shipped arm's k sub-ranges, so the epilogue folds its tree -- the

// bits are the default arm's (tools/zig/x3_exl3_sched.py trace_gm_pass vs fold_grouped_trace).

__device__ __forceinline__ void gm_cp_async16(void* smem, const void* gmem, bool ok) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(ok ? 16 : 0));
}
__device__ __forceinline__ void gm_cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void gm_cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ void gm_ldsm4(uint32_t (&a)[4], const void* p) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                 : "r"(s));
}
// the 16-byte chunk c of member row r in a stage (CPR chunks a row): 8 rows of an ldmatrix phase in 8 bank groups
template <int CPR>
__device__ __forceinline__ int gm_swz(int r, int c) {
    static_assert(CPR == 8 || CPR == 4, "rows of 64 or 32 halves");
    return CPR == 8 ? (c ^ (r & 7)) : (c ^ ((r >> 1) & 3));
}

// The gm arm's per-stage member bytes (BM rows of the stage's KS k tiles, swizzled).
template <int KS>
__host__ __device__ constexpr int gm_lmem() {
    return 128 * KS * 16 * 2;
}
// upstream's MINB: two CTAs an SM when the ring fits twice, the block <= 8 warps and a lane <= 64 accumulator slots.

// This arm keeps TWO live accumulator sets a lane (the sub-range's work and the split's run the shipped tree folds

// it into), MTL*NG*4 registers each, so the <= 64-slot case never holds at NG 16: MINB is 1 for the served pair.
template <int W, int NT, int KS, int NSA>
__host__ __device__ constexpr int gm_minb() {
    constexpr int ACC = 2 * (NT / W) * 16 * 4;
    constexpr int CAP = 65536 / (2 * W * 32);   // registers a lane when two CTAs an SM
    return 2 * ((size_t)NSA * gm_lmem<KS>() + 1536) <= 102400 && W * 32 <= 256 && ACC + 64 <= CAP ? 2 : 1;
}

// One program's item (member group, 128-column block, split, matrix) at width K2: gm_kernel's body.

// SUBW is the SHIPPED arm's warp count, hence its k sub-range count; the gm arm's 8 warps own the 8 tiles one each.
template <int CB, int K2, int NT, int W, int KS, int NSA, int SUBW>
__device__ __forceinline__ void gm_item(const half* __restrict__ X, const uint32_t* __restrict__ T,
                                        float* __restrict__ Z, unsigned char* smem, const int* rows_sh, int cnt,
                                        int nt0, int split, int per_split, int sub_len, int NTILES, int mat, int P,
                                        int N, int K, int warp, int lane) {
    constexpr int BM = 128, MTL = NT / W, NG = 16, CPR = KS * 2, LDA = KS * 16;
    constexpr int LMEM = BM * LDA * (int)sizeof(half), TW = Fmt<K2>::TW;
    const int g = lane >> 2, t = lane & 3;
    const int lr = (lane & 7) + ((lane >> 4) << 3), lc = (lane >> 3) & 1;
    const int slice = warp;                                  // the columns this warp owns: MTL tiles from slice*MTL
    const int S = per_split / KS;
    const int cnt16 = (cnt + 15) & ~15;
    const LaneMap<K2> map(lane);
    // run: the split's k is walked in the SUBW shipped sub-ranges, each partial (work) folding in as it closes.
    float run[MTL][NG][4], work[MTL][NG][4];
#pragma unroll
    for (int l = 0; l < MTL; ++l)
#pragma unroll
        for (int n = 0; n < NG; ++n)
#pragma unroll
            for (int c = 0; c < 4; ++c) run[l][n][c] = work[l][n][c] = 0.f;

    auto load_stage = [&](int s) {
        half* ra = reinterpret_cast<half*>(smem + (size_t)(s % NSA) * LMEM);
        const int k0 = (split * per_split + s * KS) * 16;
        for (int i = threadIdx.x; i < BM * CPR; i += W * 32) {
            const int r = i / CPR, c = i % CPR;
            if (r < cnt16) {
                const int row = rows_sh[r];
                const half* src = X + (size_t)(row < 0 ? 0 : row) * (size_t)K + k0 + c * 8;
                gm_cp_async16(ra + (size_t)r * LDA + gm_swz<CPR>(r, c) * 8, src, row >= 0);
            }
        }
    };

#pragma unroll
    for (int s = 0; s < NSA - 1; ++s) {
        if (s < S) load_stage(s);
        gm_cp_commit();
    }
    gm_cp_wait<NSA - 2>();
    __syncthreads();

    // weight words one stage ahead in registers (same words, same decode: only the load issues earlier)
    uint32_t wq[KS][MTL][Fmt<K2>::LW];
#pragma unroll
    for (int kk = 0; kk < KS; ++kk)
#pragma unroll
        for (int l = 0; l < MTL; ++l)
            load_words<K2>(wq[kk][l], T + ((size_t)(split * per_split + kk) * NTILES + nt0 + slice * MTL + l) * TW + lane, lane);
    uint32_t wq2[KS][MTL][Fmt<K2>::LW] = {};
#pragma unroll
    for (int kk = 0; kk < KS; ++kk)
#pragma unroll
        for (int l = 0; l < MTL; ++l)
            if (S > 1) load_words<K2>(wq2[kk][l], T + ((size_t)(split * per_split + KS + kk) * NTILES + nt0 + slice * MTL + l) * TW + lane, lane);
    for (int s = 0; s < S; ++s) {
        if (s + NSA - 1 < S) load_stage(s + NSA - 1);
        gm_cp_commit();
        const half* A = reinterpret_cast<const half*>(smem + (size_t)(s % NSA) * LMEM);
#pragma unroll
        for (int kk = 0; kk < KS; ++kk) {
            uint32_t af[MTL][4];
#pragma unroll
            for (int l = 0; l < MTL; ++l) {
                uint32_t w[Fmt<K2>::LW];
#pragma unroll
                for (int q = 0; q < Fmt<K2>::LW; ++q) { w[q] = wq[kk][l][q]; wq[kk][l][q] = wq2[kk][l][q]; }
                if (s + 2 < S)                               // stage s+2's words for this kk: two stages in flight
                    load_words<K2>(wq2[kk][l], T + ((size_t)(split * per_split + (s + 2) * KS + kk) * NTILES + nt0 + slice * MTL + l) * TW + lane, lane);
                uint32_t b0[2], b1[2];
                decode_tile<CB, K2>(w, map, lane, b0, b1);
                af[l][0] = b0[0]; af[l][1] = b1[0]; af[l][2] = b0[1]; af[l][3] = b1[1];  // weights as the A fragment
            }
            const int chunk = gm_swz<CPR>(lr, kk * 2 + lc);
#pragma unroll
            for (int np = 0; np < NG / 2; ++np) {
                if (16 * np < cnt) {                         // warp-uniform: a dead n8 group is never read
                    uint32_t bb[4];
                    gm_ldsm4(bb, A + (16 * np + lr) * LDA + chunk * 8);
                    const uint32_t b01[2] = {bb[0], bb[1]}, b23[2] = {bb[2], bb[3]};  // the 16 members of the pair
#pragma unroll
                    for (int l = 0; l < MTL; ++l) {
                        mma16816(work[l][2 * np], af[l], b01);
                        mma16816(work[l][2 * np + 1], af[l], b23);
                    }
                }
            }
        }
        if (((s + 1) * KS) % sub_len == 0) {                 // a shipped sub-range closed: fold its partial into run
#pragma unroll
            for (int l = 0; l < MTL; ++l)
#pragma unroll
                for (int n = 0; n < NG; ++n)
#pragma unroll
                    for (int c = 0; c < 4; ++c) {
                        run[l][n][c] += work[l][n][c];
                        work[l][n][c] = 0.f;
                    }
        }
        gm_cp_wait<NSA - 2>();
        __syncthreads();
    }

    // store this warp's columns: acc is C[weight column][member] (c0 g,2t; c1 g,2t+1; c2 g+8,2t; c3 g+8,2t+1)
    float* __restrict__ Zb = Z + (size_t)(mat * ((K >> 4) / per_split) + split) * P * N;
    const int col_blk = nt0 * 16;
#pragma unroll
    for (int l = 0; l < MTL; ++l) {
        const int cbase = col_blk + (slice * MTL + l) * 16 + g;
#pragma unroll
        for (int n = 0; n < NG; ++n) {
            const int r0 = rows_sh[8 * n + 2 * t], r1 = rows_sh[8 * n + 2 * t + 1];
            if (r0 >= 0) {
                float* o = Zb + (size_t)r0 * N + cbase;
                o[0] = run[l][n][0];
                o[8] = run[l][n][2];
            }
            if (r1 >= 0) {
                float* o = Zb + (size_t)r1 * N + cbase;
                o[0] = run[l][n][1];
                o[8] = run[l][n][3];
            }
        }
    }
}

// The gm arm's launch: NT 8 tiles (128 a block) across W 8 warps, KS 2 k tiles a stage, an NSA 4 ring, SUBW 4.
template <int CB, int NT, int W, int LO, int HI, int KS, int NSA, int SUBW>
__global__ void __launch_bounds__(W * 32, gm_minb<W, NT, KS, NSA>()) grouped_gm_kernel(
    const half* __restrict__ X0, const half* __restrict__ X1, const int64_t* __restrict__ TP0,
    const int64_t* __restrict__ TP1, const int* __restrict__ K2_0, const int* __restrict__ K2_1,
    const int* __restrict__ uids, const int* __restrict__ ucount, const int* __restrict__ members,
    float* __restrict__ Z, int K, int N, int P, int SK, int maxm, int slots) {
    const int u = blockIdx.x;
    if (u >= ucount[0]) return;
    constexpr int BM = 128;
    const int MG = (maxm + BM - 1) / BM;
    const int mg = blockIdx.z % MG;
    const int split = (blockIdx.z / MG) % SK;
    const int mat = blockIdx.z / MG / SK;
    const half* X = mat ? X1 : X0;
    const int e = uids[u];
    const uint32_t* T = reinterpret_cast<const uint32_t*>(mat ? TP1[e] : TP0[e]);
    const int k2 = mat ? K2_1[e] : K2_0[e];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int KT = K >> 4, NTILES = N >> 4;
    const int per_split = KT / SK, sub_len = per_split / SUBW;
    const int nt0 = blockIdx.y * NT;
    if (per_split % SUBW != 0 || sub_len % KS != 0) return;   // cuda_exl3.zig refuses this launch first

    __shared__ int rows_sh[BM];
    for (int i = threadIdx.x; i < BM; i += W * 32) {
        const int m = mg * BM + i;
        const int code = m < maxm ? members[u * maxm + m] : -1;
        rows_sh[i] = code >= 0 ? (code >> 5) * slots + (code & 31) : -1;
    }
    __syncthreads();
    if (rows_sh[0] < 0) return;                               // members come first, so this group is empty
    int cnt = 0;
#pragma unroll 16
    for (int i = 0; i < BM; ++i)
        if (rows_sh[i] >= 0) cnt = i + 1;

    extern __shared__ __align__(16) unsigned char tf_exl3_gm[];
    switch (k2) {
#define TF_GM_CASE(K2_)                                                                                               \
    case K2_:                                                                                                         \
        if constexpr (K2_ >= LO && K2_ <= HI)                                                                         \
            gm_item<CB, K2_, NT, W, KS, NSA, SUBW>(X, T, Z, tf_exl3_gm, rows_sh, cnt, nt0, split, per_split, sub_len,  \
                                                   NTILES, mat, P, N, K, warp, lane);                                  \
        else                                                                                                          \
            __trap();                                                                                                 \
        break;
        TF_GM_CASE(2)
        TF_GM_CASE(3)
        TF_GM_CASE(4)
        TF_GM_CASE(5)
        TF_GM_CASE(6)
        TF_GM_CASE(7)
        TF_GM_CASE(8)
        TF_GM_CASE(9)
        TF_GM_CASE(10)
        TF_GM_CASE(11)
        TF_GM_CASE(12)
        TF_GM_CASE(13)
        TF_GM_CASE(14)
        TF_GM_CASE(15)
        TF_GM_CASE(16)
#undef TF_GM_CASE
        default:
            __trap();
    }
}

// grouped_kernel launches at most this many member rows an expert; larger groups loop (cuda_exl3.zig grouped_grid_members)
constexpr int TF_GROUPED_GRID_MEMBERS = 512;

// Program (expert u, n block, split and MTL member tiles): MTL * 16 members times W_q; warps summed in order.
template <int CB, int NT, int W, int PF, int LO, int HI, int MTL>
__global__ void __launch_bounds__(W * 32) grouped_kernel(
    const half* __restrict__ X0, const half* __restrict__ X1, const int64_t* __restrict__ TP0,
    const int64_t* __restrict__ TP1, const int* __restrict__ K2_0, const int* __restrict__ K2_1,
    const int* __restrict__ uids, const int* __restrict__ ucount, const int* __restrict__ members,
    float* __restrict__ Z, int K, int N, int P, int SK, int maxm, int slots) {
    const int u = blockIdx.x;
    if (u >= ucount[0]) return;
    const int MTG = (maxm + 16 * MTL - 1) / (16 * MTL);
    // The grid holds G member groups (TF_GROUPED_GRID_MEMBERS rows); a program walks mg0, mg0+G, ...
    // Each member row's sums are unchanged; only which program computes it moves.
    // MTL 1 launches at most TF_GROUPED_GRID_MEMBERS rows; the wide arm (MTL > 1) launches every group (host grid)
    const int G = MTL == 1 ? min(MTG, (TF_GROUPED_GRID_MEMBERS + 15) / 16) : MTG;
    const int mg0 = blockIdx.z % G;
    const int split = (blockIdx.z / G) % SK;
    const int mat = blockIdx.z / G / SK;
    const half* X = mat ? X1 : X0;
    const int e = uids[u];
    const uint32_t* T = reinterpret_cast<const uint32_t*>(mat ? TP1[e] : TP0[e]);
    const int k2 = mat ? K2_1[e] : K2_0[e];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int KT = K >> 4, NTILES = N >> 4;
    const int per_split = KT / SK, per_warp = per_split / W;
    const int kt0 = split * per_split + warp * per_warp;
    const int nt0 = blockIdx.y * NT;

    // Deferred: upstream stages these rows in a swizzled cp.async A-tile (B via ldsm4); here they are global reads.
    __shared__ int rows_sh[16 * MTL];
    for (int mg = mg0; mg < MTG; mg += G) {
    if (mg != mg0) __syncthreads();       // rows_sh / red are reused by the next group
    for (int i = threadIdx.x; i < 16 * MTL; i += W * 32) {
        const int m = mg * 16 * MTL + i;
        const int code = m < maxm ? members[u * maxm + m] : -1;
        rows_sh[i] = code >= 0 ? (code >> 5) * slots + (code & 31) : -1;
    }
    __syncthreads();
    if (rows_sh[0] < 0) return;                           // members come first, so this group is empty

    if constexpr (MTL > 1) {
        // The wide arm: MTL member tiles a program, their A rows staged in shared once a block, one decode a tile.
        extern __shared__ __align__(16) unsigned char tf_exl3_wsmem[];
        const int words = per_split * 8;                        // the split's k slice in words: 16 k a word
        const int sw = words + ((4 + 32 - (words & 31)) & 31);
        // sw is 4 mod 32: a lane's four staged fragment words take 32 distinct banks
        uint32_t* As = reinterpret_cast<uint32_t*>(tf_exl3_wsmem);
        float (*redw)[16 * MTL][16] = reinterpret_cast<float (*)[16 * MTL][16]>(As + (size_t)16 * MTL * sw);
        int live = 0;
#pragma unroll
        for (int tt = 0; tt < MTL; ++tt)
            if (rows_sh[tt * 16] >= 0) live = tt + 1;     // members ascend, so the first dead row ends the group
        const int kbase = split * per_split * 16;
        for (int rrow = 0; rrow < 16 * live; ++rrow) {
            const int r = rows_sh[rrow];
            const uint32_t* src = reinterpret_cast<const uint32_t*>(X + (size_t)(r < 0 ? 0 : r) * K + kbase);
            uint32_t* dst = As + (size_t)rrow * sw;
            for (int w = threadIdx.x; w < words; w += W * 32) dst[w] = (r >= 0) ? src[w] : 0u;
        }
        __syncthreads();                                   // every warp reads the staged rows below
        const int ktrel = warp * per_warp;                 // this warp's k tiles relative to the split's first
        for (int i = 0; i < NT; ++i) {
            if (i) __syncthreads();                       // redw is reused: the previous n tile's store has read it
            float acc[MTL][2][4];
#pragma unroll
            for (int tt = 0; tt < MTL; ++tt)
#pragma unroll
                for (int h = 0; h < 2; ++h)
#pragma unroll
                    for (int c = 0; c < 4; ++c) acc[tt][h][c] = 0.f;
            switch (k2) {
#define TF_EXL3W_CASE(K2_)                                                                                      \
    case K2_:                                                                                                   \
        if constexpr (K2_ >= LO && K2_ <= HI)                                                                   \
            warp_tile_stage<CB, K2_, PF, MTL>(T, NTILES, kt0, per_warp, nt0 + i, As, sw, ktrel, live, lane,     \
                                              acc);                                                             \
        else                                                                                                    \
            __trap();                                                                                           \
        break;
                TF_EXL3W_CASE(2)
                TF_EXL3W_CASE(3)
                TF_EXL3W_CASE(4)
                TF_EXL3W_CASE(5)
                TF_EXL3W_CASE(6)
                TF_EXL3W_CASE(7)
                TF_EXL3W_CASE(8)
                TF_EXL3W_CASE(9)
                TF_EXL3W_CASE(10)
                TF_EXL3W_CASE(11)
                TF_EXL3W_CASE(12)
                TF_EXL3W_CASE(13)
                TF_EXL3W_CASE(14)
                TF_EXL3W_CASE(15)
                TF_EXL3W_CASE(16)
#undef TF_EXL3W_CASE
                default:
                    __trap();
            }
            // this n tile's MTL member tiles through shared memory, warps added in order
#pragma unroll
            for (int tt = 0; tt < MTL; ++tt) {
                if (tt < live) {
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const int col = h * 8 + 2 * t;
                        redw[warp][tt * 16 + g][col] = acc[tt][h][0];
                        redw[warp][tt * 16 + g][col + 1] = acc[tt][h][1];
                        redw[warp][tt * 16 + g + 8][col] = acc[tt][h][2];
                        redw[warp][tt * 16 + g + 8][col + 1] = acc[tt][h][3];
                    }
                }
            }
            __syncthreads();
            for (int idx = threadIdx.x; idx < 16 * live * 16; idx += W * 32) {
                const int tt = idx / 256, rl = (idx / 16) % 16, col = idx % 16;
                const int r = rows_sh[tt * 16 + rl];
                if (r < 0) continue;
                float s = redw[0][tt * 16 + rl][col];
#pragma unroll
                for (int w = 1; w < W; ++w) s += redw[w][tt * 16 + rl][col];
                Z[(((size_t)mat * SK + split) * P + r) * N + nt0 * 16 + i * 16 + col] = s;
            }
        }
    } else {
    __shared__ float red[W][16][NT * 16];
    for (int tt = 0; tt < MTL; ++tt) {
        if (rows_sh[tt * 16] < 0) break;                  // members ascend, so the first dead row ends the group
        if (tt) __syncthreads();                          // red is reused: the previous tile's store has read it
        const int r0 = rows_sh[tt * 16 + g], r1 = rows_sh[tt * 16 + g + 8];
        const half* x0 = X + (size_t)(r0 < 0 ? 0 : r0) * K + 2 * t;
        const half* x1 = X + (size_t)(r1 < 0 ? 0 : r1) * K + 2 * t;

    float acc[NT][2][4];
#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[i][h][c] = 0.f;

    switch (k2) {
#define TF_EXL3X_CASE(K2_)                                                                                      \
    case K2_:                                                                                                   \
        if constexpr (K2_ >= LO && K2_ <= HI)                                                                   \
            warp_tiles<CB, K2_, NT, PF>(T, NTILES, kt0, per_warp, nt0, x0, x1, r0 >= 0, r1 >= 0, lane, acc);    \
        else                                                                                                    \
            __trap();                                                                                           \
        break;
        TF_EXL3X_CASE(2)
        TF_EXL3X_CASE(3)
        TF_EXL3X_CASE(4)
        TF_EXL3X_CASE(5)
        TF_EXL3X_CASE(6)
        TF_EXL3X_CASE(7)
        TF_EXL3X_CASE(8)
        TF_EXL3X_CASE(9)
        TF_EXL3X_CASE(10)
        TF_EXL3X_CASE(11)
        TF_EXL3X_CASE(12)
        TF_EXL3X_CASE(13)
        TF_EXL3X_CASE(14)
        TF_EXL3X_CASE(15)
        TF_EXL3X_CASE(16)
#undef TF_EXL3X_CASE
        default:
            __trap();
    }

        // warps' partial sums through shared memory, added in warp order
#pragma unroll
        for (int i = 0; i < NT; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int col = i * 16 + h * 8 + 2 * t;
                red[warp][g][col] = acc[i][h][0];
                red[warp][g][col + 1] = acc[i][h][1];
                red[warp][g + 8][col] = acc[i][h][2];
                red[warp][g + 8][col + 1] = acc[i][h][3];
            }
        __syncthreads();
        for (int idx = threadIdx.x; idx < 16 * NT * 16; idx += W * 32) {
            const int row = idx / (NT * 16), col = idx % (NT * 16);
            const int r = rows_sh[tt * 16 + row];
            if (r < 0) continue;
            float s = red[0][row][col];
#pragma unroll
            for (int w = 1; w < W; ++w) s += red[w][row][col];
            Z[(((size_t)mat * SK + split) * P + r) * N + nt0 * 16 + col] = s;
        }
    }
    }
    }
}

// Timing probe only (TF_EXL3_ABL_DECODE): grouped_kernel with the trellis decode's arithmetic replaced by a passthrough
template <int CB, int NT, int W, int PF, int LO, int HI, int MTL>
__global__ void __launch_bounds__(W * 32) grouped_abl_kernel(
    const half* __restrict__ X0, const half* __restrict__ X1, const int64_t* __restrict__ TP0,
    const int64_t* __restrict__ TP1, const int* __restrict__ K2_0, const int* __restrict__ K2_1,
    const int* __restrict__ uids, const int* __restrict__ ucount, const int* __restrict__ members,
    float* __restrict__ Z, int K, int N, int P, int SK, int maxm, int slots) {
    const int u = blockIdx.x;
    if (u >= ucount[0]) return;
    const int MTG = (maxm + 16 * MTL - 1) / (16 * MTL);
    const int mg = blockIdx.z % MTG;
    const int split = (blockIdx.z / MTG) % SK;
    const int mat = blockIdx.z / MTG / SK;
    const half* X = mat ? X1 : X0;
    const int e = uids[u];
    const uint32_t* T = reinterpret_cast<const uint32_t*>(mat ? TP1[e] : TP0[e]);
    const int k2 = mat ? K2_1[e] : K2_0[e];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int KT = K >> 4, NTILES = N >> 4;
    const int per_split = KT / SK, per_warp = per_split / W;
    const int kt0 = split * per_split + warp * per_warp;
    const int nt0 = blockIdx.y * NT;

    // Deferred: upstream stages these rows in a swizzled cp.async A-tile (B via ldsm4); here they are global reads.
    __shared__ int rows_sh[16 * MTL];
    for (int i = threadIdx.x; i < 16 * MTL; i += W * 32) {
        const int m = mg * 16 * MTL + i;
        const int code = m < maxm ? members[u * maxm + m] : -1;
        rows_sh[i] = code >= 0 ? (code >> 5) * slots + (code & 31) : -1;
    }
    __syncthreads();
    if (rows_sh[0] < 0) return;                           // members come first, so this group is empty

    if constexpr (MTL > 1) {
        // The wide arm: MTL member tiles a program, their rows in shared, one decode a (n, k) tile for all of them.
        int live = 0;
#pragma unroll
        for (int tt = 0; tt < MTL; ++tt)
            if (rows_sh[tt * 16] >= 0) live = tt + 1;     // members ascend, so the first dead row ends the group
        __shared__ float redw[W][16 * MTL][16];
        const half* xb0[MTL];
        const half* xb1[MTL];
        bool ok0[MTL], ok1[MTL];
#pragma unroll
        for (int tt = 0; tt < MTL; ++tt) {
            const int r0 = rows_sh[tt * 16 + g], r1 = rows_sh[tt * 16 + g + 8];
            xb0[tt] = X + (size_t)(r0 < 0 ? 0 : r0) * K + 2 * t;
            xb1[tt] = X + (size_t)(r1 < 0 ? 0 : r1) * K + 2 * t;
            ok0[tt] = r0 >= 0;
            ok1[tt] = r1 >= 0;
        }
        for (int i = 0; i < NT; ++i) {
            if (i) __syncthreads();                       // redw is reused: the previous n tile's store has read it
            float acc[MTL][2][4];
#pragma unroll
            for (int tt = 0; tt < MTL; ++tt)
#pragma unroll
                for (int h = 0; h < 2; ++h)
#pragma unroll
                    for (int c = 0; c < 4; ++c) acc[tt][h][c] = 0.f;
            switch (k2) {
#define TF_EXL3W_CASE(K2_)                                                                                      \
    case K2_:                                                                                                   \
        if constexpr (K2_ >= LO && K2_ <= HI)                                                                   \
            warp_tile_wide<CB, K2_, PF, MTL, 1>(T, NTILES, kt0, per_warp, nt0 + i, xb0, xb1, ok0, ok1, live, lane,  \
                                             acc);                                                              \
        else                                                                                                    \
            __trap();                                                                                           \
        break;
                TF_EXL3W_CASE(2)
                TF_EXL3W_CASE(3)
                TF_EXL3W_CASE(4)
                TF_EXL3W_CASE(5)
                TF_EXL3W_CASE(6)
                TF_EXL3W_CASE(7)
                TF_EXL3W_CASE(8)
                TF_EXL3W_CASE(9)
                TF_EXL3W_CASE(10)
                TF_EXL3W_CASE(11)
                TF_EXL3W_CASE(12)
                TF_EXL3W_CASE(13)
                TF_EXL3W_CASE(14)
                TF_EXL3W_CASE(15)
                TF_EXL3W_CASE(16)
#undef TF_EXL3W_CASE
                default:
                    __trap();
            }
            // this n tile's MTL member tiles through shared memory, warps added in order
#pragma unroll
            for (int tt = 0; tt < MTL; ++tt) {
                if (tt < live) {
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const int col = h * 8 + 2 * t;
                        redw[warp][tt * 16 + g][col] = acc[tt][h][0];
                        redw[warp][tt * 16 + g][col + 1] = acc[tt][h][1];
                        redw[warp][tt * 16 + g + 8][col] = acc[tt][h][2];
                        redw[warp][tt * 16 + g + 8][col + 1] = acc[tt][h][3];
                    }
                }
            }
            __syncthreads();
            for (int idx = threadIdx.x; idx < 16 * live * 16; idx += W * 32) {
                const int tt = idx / 256, rl = (idx / 16) % 16, col = idx % 16;
                const int r = rows_sh[tt * 16 + rl];
                if (r < 0) continue;
                float s = redw[0][tt * 16 + rl][col];
#pragma unroll
                for (int w = 1; w < W; ++w) s += redw[w][tt * 16 + rl][col];
                Z[(((size_t)mat * SK + split) * P + r) * N + nt0 * 16 + i * 16 + col] = s;
            }
        }
    } else {
    __shared__ float red[W][16][NT * 16];
    for (int tt = 0; tt < MTL; ++tt) {
        if (rows_sh[tt * 16] < 0) break;                  // members ascend, so the first dead row ends the group
        if (tt) __syncthreads();                          // red is reused: the previous tile's store has read it
        const int r0 = rows_sh[tt * 16 + g], r1 = rows_sh[tt * 16 + g + 8];
        const half* x0 = X + (size_t)(r0 < 0 ? 0 : r0) * K + 2 * t;
        const half* x1 = X + (size_t)(r1 < 0 ? 0 : r1) * K + 2 * t;

    float acc[NT][2][4];
#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[i][h][c] = 0.f;

    switch (k2) {
#define TF_EXL3X_CASE(K2_)                                                                                      \
    case K2_:                                                                                                   \
        if constexpr (K2_ >= LO && K2_ <= HI)                                                                   \
            warp_tiles<CB, K2_, NT, PF, 1>(T, NTILES, kt0, per_warp, nt0, x0, x1, r0 >= 0, r1 >= 0, lane, acc);    \
        else                                                                                                    \
            __trap();                                                                                           \
        break;
        TF_EXL3X_CASE(2)
        TF_EXL3X_CASE(3)
        TF_EXL3X_CASE(4)
        TF_EXL3X_CASE(5)
        TF_EXL3X_CASE(6)
        TF_EXL3X_CASE(7)
        TF_EXL3X_CASE(8)
        TF_EXL3X_CASE(9)
        TF_EXL3X_CASE(10)
        TF_EXL3X_CASE(11)
        TF_EXL3X_CASE(12)
        TF_EXL3X_CASE(13)
        TF_EXL3X_CASE(14)
        TF_EXL3X_CASE(15)
        TF_EXL3X_CASE(16)
#undef TF_EXL3X_CASE
        default:
            __trap();
    }

        // warps' partial sums through shared memory, added in warp order
#pragma unroll
        for (int i = 0; i < NT; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int col = i * 16 + h * 8 + 2 * t;
                red[warp][g][col] = acc[i][h][0];
                red[warp][g][col + 1] = acc[i][h][1];
                red[warp][g + 8][col] = acc[i][h][2];
                red[warp][g + 8][col + 1] = acc[i][h][3];
            }
        __syncthreads();
        for (int idx = threadIdx.x; idx < 16 * NT * 16; idx += W * 32) {
            const int row = idx / (NT * 16), col = idx % (NT * 16);
            const int r = rows_sh[tt * 16 + row];
            if (r < 0) continue;
            float s = red[0][row][col];
#pragma unroll
            for (int w = 1; w < W; ++w) s += red[w][row][col];
            Z[(((size_t)mat * SK + split) * P + r) * N + nt0 * 16 + col] = s;
        }
    }
    }
}

// W_q [K, N] fp16 of one matrix through the same lane decode (tests; not on the forward path).
template <int CB, int K2>
__global__ void dequant_kernel(const uint32_t* __restrict__ T, half* __restrict__ out, int K, int N) {
    const int kt = blockIdx.x, nt = blockIdx.y, lane = threadIdx.x;
    const int NTILES = N >> 4;
    uint32_t w[Fmt<K2>::LW];
    load_words<K2>(w, T + ((size_t)kt * NTILES + nt) * Fmt<K2>::TW + lane, lane);
    const LaneMap<K2> map(lane);
    uint32_t b0[2], b1[2];
    decode_tile<CB, K2>(w, map, lane, b0, b1);
    const int g = lane >> 2, t = lane & 3;
    uint32_t v[4] = {b0[0], b0[1], b1[0], b1[1]};
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        half2 h = *reinterpret_cast<half2*>(&v[q]);
        const int col = nt * 16 + g + 8 * (q >> 1);
        const int row = kt * 16 + 2 * t + 8 * (q & 1);
        out[(size_t)row * N + col] = __low2half(h);
        out[(size_t)(row + 1) * N + col] = __high2half(h);
    }
}

struct GroupedArgs {
    const half* x0;
    const half* x1;
    const int64_t* tp0;
    const int64_t* tp1;
    const int* k2_0;
    const int* k2_1;
    const int* uids;
    const int* ucount;
    const int* members;
    float* z;
    int K, N, P, SK, maxm, slots;
    int nexp_max;        // grid.x (upper bound of distinct experts)
    int mats, nt, warps, pf, lo, hi;
};

template <int CB, int MTL>
void grouped_launch(const GroupedArgs& a, cudaStream_t stream) {
    const int MTG = (a.maxm + 16 * MTL - 1) / (16 * MTL);
    const char* abl_g = getenv("TF_EXL3_ABL_DECODE");
    // grouped_kernel walks member groups past TF_GROUPED_GRID_MEMBERS; the ablation probe still launches each one
    const int MTGL = (MTL > 1 || (abl_g != nullptr && abl_g[0] != '0')) ? MTG : min(MTG, (TF_GROUPED_GRID_MEMBERS + 15) / 16);
    dim3 grid((unsigned)a.nexp_max, (unsigned)(a.N / (16 * a.nt)), (unsigned)(a.mats * a.SK * MTGL));
    const char* abl_e = getenv("TF_EXL3_ABL_DECODE");
    const bool abl = abl_e != nullptr && abl_e[0] != '0';   // timing probe: unset or "0" is off (the repo's getenv idiom)
#define TF_LAUNCH(NT_, W_, PF_, LO_, HI_)                                                                       \
    if (abl)                                                                                                    \
        grouped_abl_kernel<CB, NT_, W_, PF_, LO_, HI_, MTL><<<grid, W_ * 32, 0, stream>>>(                      \
            a.x0, a.x1, a.tp0, a.tp1, a.k2_0, a.k2_1, a.uids, a.ucount, a.members, a.z, a.K, a.N, a.P, a.SK,    \
            a.maxm, a.slots);                                                                                   \
    else                                                                                                        \
        grouped_kernel<CB, NT_, W_, PF_, LO_, HI_, MTL><<<grid, W_ * 32, 0, stream>>>(                          \
            a.x0, a.x1, a.tp0, a.tp1, a.k2_0, a.k2_1, a.uids, a.ucount, a.members, a.z, a.K, a.N, a.P, a.SK,    \
            a.maxm, a.slots)
#define TF_RANGES(NT_, W_, PF_)                                                                                 \
    if (a.lo == 8 && a.hi == 8) TF_LAUNCH(NT_, W_, PF_, 8, 8);                                                  \
    else if (a.lo >= 2 && a.hi <= 10) TF_LAUNCH(NT_, W_, PF_, 2, 10);                                           \
    else TF_LAUNCH(NT_, W_, PF_, 2, 16);
    if (a.nt == 8 && a.warps == 4 && a.pf == 1) { TF_RANGES(8, 4, 1) }
    else if (a.nt == 8 && a.warps == 4 && a.pf == 2) { TF_RANGES(8, 4, 2) }
    else if (a.nt == 4 && a.warps == 4 && a.pf == 2) { TF_RANGES(4, 4, 2) }
    else if (a.nt == 4 && a.warps == 4 && a.pf == 1) { TF_RANGES(4, 4, 1) }
    else (void)0;   // nt/warps/pf outside the settings the reference compiles and cuda_exl3.zig's table lists
#undef TF_RANGES
#undef TF_LAUNCH
}

// The staged gm arm's launch: NT 8 (128 columns a block, staged once) across W 8 warps (one column tile each), KS 2

// k tiles a stage, an NSA 4 ring; SUBW 4 is the shipped arm's warp count, so the split's k is summed in its 4

// sub-ranges and the bits match it. Opt-in, for a shape the guard admits: per_split = KT/SK divisible by SUBW and W,

// and the shipped sub-range per_split/SUBW by KS. cuda_exl3.zig checks the same first (gmShapeOk).
template <int CB>
bool grouped_gm_launch(const GroupedArgs& a, cudaStream_t stream) {
    constexpr int NT = 8, W = 8, KS = 2, NSA = 4, SUBW = 4;
    const int KT = a.K / 16;
    if (KT % a.SK != 0 || a.N % (16 * NT) != 0) return false;
    const int per_split = KT / a.SK, sub_len = per_split / SUBW;
    if (a.warps != SUBW || per_split % SUBW != 0 || sub_len % KS != 0) return false;
    const int MG = (a.maxm + 127) / 128;
    dim3 grid((unsigned)a.nexp_max, (unsigned)(a.N / (16 * NT)), (unsigned)(a.mats * a.SK * MG));
    const size_t smem = (size_t)NSA * gm_lmem<KS>();
#define TF_GM_LAUNCH(LO_, HI_)                                                                                         \
    grouped_gm_kernel<CB, NT, W, LO_, HI_, KS, NSA, SUBW><<<grid, W * 32, smem, stream>>>(                             \
        a.x0, a.x1, a.tp0, a.tp1, a.k2_0, a.k2_1, a.uids, a.ucount, a.members, a.z, a.K, a.N, a.P, a.SK, a.maxm,       \
        a.slots)
    if (a.lo == 8 && a.hi == 8) TF_GM_LAUNCH(8, 8);
    else if (a.lo >= 2 && a.hi <= 10) TF_GM_LAUNCH(2, 10);
    else TF_GM_LAUNCH(2, 16);
#undef TF_GM_LAUNCH
    return true;
}

template <int CB>
void dequant_launch(const uint32_t* t, half* o, int K, int N, int k2, cudaStream_t stream) {
    dim3 grid((unsigned)(K / 16), (unsigned)(N / 16));
    switch (k2) {
#define TF_DQ(K2_) case K2_: dequant_kernel<CB, K2_><<<grid, 32, 0, stream>>>(t, o, K, N); break;
        TF_DQ(2) TF_DQ(3) TF_DQ(4) TF_DQ(5) TF_DQ(6) TF_DQ(7) TF_DQ(8) TF_DQ(9) TF_DQ(10) TF_DQ(11)
        TF_DQ(12) TF_DQ(13) TF_DQ(14) TF_DQ(15) TF_DQ(16)
#undef TF_DQ
        default: (void)k2;   // K2 outside 2..16
    }
}

}  // namespace tf_exl3x




// The reference's experts_cb{0,1,2}.cu: one translation unit each, because each instance was its own python
// extension. One fatbin carries all three here.
namespace tf_exl3x {
template void grouped_launch<0, 1>(const GroupedArgs&, cudaStream_t);
template void grouped_launch<1, 1>(const GroupedArgs&, cudaStream_t);
template void grouped_launch<2, 1>(const GroupedArgs&, cudaStream_t);
// The wide member-pass arm (MTL 2, 32 members a program, rows staged in dynamic shared): mul1, opt-in, off by default.
template void grouped_launch<2, 2>(const GroupedArgs&, cudaStream_t);
// The staged gm arm (128 members a program, rows the B operand, columns split across the warps): opt-in, off.
template bool grouped_gm_launch<2>(const GroupedArgs&, cudaStream_t);
template void dequant_launch<0>(const uint32_t*, half*, int, int, int, cudaStream_t);
template void dequant_launch<1>(const uint32_t*, half*, int, int, int, cudaStream_t);
template void dequant_launch<2>(const uint32_t*, half*, int, int, int, cudaStream_t);
}  // namespace tf_exl3x

}  // namespace tf_fn_exl3
