// Flash Next's MTP drafts (cuda_mtp.zig): a score row's best value, its first id and the row's log-sum-exp, and a row's
// first `count` candidates for a keyed sampled draw (fn_row_cand).

#include <cuda_bf16.h>
#include <math_constants.h>
#include <stdint.h>

namespace fn_mtp {

static __device__ __forceinline__ void better(float& v, int& i, float ov, int oi) {
    if (ov > v || (ov == v && oi < i)) {
        v = ov;
        i = oi;
    }
}

// A bf16 score's 16-bit sort key: larger scores get larger keys, and +0 and -0 get one (cuda_mtp.zig candidateKey).
static __device__ __forceinline__ unsigned cand_key(unsigned short b) {
    unsigned u = b;
    if ((u & 0x7fffu) == 0u) u = 0u;
    return (u & 0x8000u) ? (~u & 0xffffu) : (u | 0x8000u);
}

}  // namespace fn_mtp

// Program (row), 1024 threads: out [rows, 4] fp32 holds the best of a row's first V values, its log-sum-exp, its id, 0.
extern "C" __global__ void __launch_bounds__(1024) fn_row_best(const __nv_bfloat16* __restrict__ L, int V, long long ld,
                                                               float* __restrict__ out) {
    using namespace fn_mtp;
    const int r = blockIdx.x, t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const __nv_bfloat16* row = L + static_cast<long long>(r) * ld;
    float v = -CUDART_INF_F;
    int at = 0x7fffffff;
    for (int i = t; i < V; i += blockDim.x) {
        const float x = __bfloat162float(row[i]);
        if (x > v) {
            v = x;
            at = i;
        }
    }
    for (int o = 16; o; o >>= 1) better(v, at, __shfl_xor_sync(0xffffffffu, v, o), __shfl_xor_sync(0xffffffffu, at, o));
    __shared__ float sv[32];
    __shared__ int si[32];
    __shared__ float top;
    __shared__ int top_at;
    if (lane == 0) {
        sv[warp] = v;
        si[warp] = at;
    }
    __syncthreads();
    if (warp == 0) {
        v = lane < (int)(blockDim.x >> 5) ? sv[lane] : -CUDART_INF_F;
        at = lane < (int)(blockDim.x >> 5) ? si[lane] : 0x7fffffff;
        for (int o = 16; o; o >>= 1) better(v, at, __shfl_xor_sync(0xffffffffu, v, o), __shfl_xor_sync(0xffffffffu, at, o));
        if (lane == 0) {
            top = v;
            top_at = at;
        }
    }
    __syncthreads();
    const float m = top;
    float sum = 0.0f;
    for (int i = t; i < V; i += blockDim.x) sum += expf(__bfloat162float(row[i]) - m);
    for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    __shared__ float ss[32];
    if (lane == 0) ss[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = lane < (int)(blockDim.x >> 5) ? ss[lane] : 0.0f;
        for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
        if (lane == 0) {
            out[r * 4 + 0] = m;
            out[r * 4 + 1] = m + logf(sum);
            out[r * 4 + 2] = __int_as_float(top_at);
            out[r * 4 + 3] = 0.0f;
        }
    }
}

// A sampled draw's candidates (cuda_mtp.zig topCandidates is its host form, and the spec). The idea, an exact selection of a
// row's k best on the device so that the host draws among k scores instead of a whole row, is Track B of the DeepSeek-V4.1
// recipe (MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold, patch 0021, Mia's AI Lab); this kernel is written for
// bf16 rows (16-bit keys) and the draw's tie rule here. Program (row), 1024 threads. Of the row's first V bf16 scores (`ld` apart between rows) the first `count` in the draw's order (the larger
// score first, of equal scores the lower column; +0 and -0 equal) go to vals[row * stride + j] (the score as fp32, exact) and
// cols[row * stride + j] (its column), in no particular order, and flags[row] = 0. A row with a NaN has no total order: it
// writes flags[row] = 1 and nothing else, and the host reads that row whole. The count-th largest key is found by two 8-bit
// histogram passes over the 16-bit keys; the keys above it are taken whole, and of the keys equal to it the lowest columns, by
// walking the row in tiles of 1,024 columns with a block-wide count of the ties in each.
extern "C" __global__ void __launch_bounds__(1024) fn_row_cand(const __nv_bfloat16* __restrict__ L, int V, long long ld, int count,
                                                               int stride, float* __restrict__ vals, int* __restrict__ cols,
                                                               int* __restrict__ flags) {
    using namespace fn_mtp;
    const int r = blockIdx.x, t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const unsigned short* row = reinterpret_cast<const unsigned short*>(L + static_cast<long long>(r) * ld);
    const int k = count < V ? count : V;
    __shared__ unsigned whist[32][256];
    __shared__ unsigned hist[256];
    __shared__ unsigned wtie[32];
    __shared__ unsigned s_left, s_hi, s_cut, s_above, s_taken;
    if (t == 0) s_left = static_cast<unsigned>(k);
    // pass 1: the keys' high bytes, a histogram a warp (lanes of one warp seldom share a bin); a NaN ends the row
    for (int b = t; b < 32 * 256; b += blockDim.x) (&whist[0][0])[b] = 0u;
    __syncthreads();
    bool nan = false;
    for (int i = t; i < V; i += blockDim.x) {
        const unsigned short b = row[i];
        nan |= (b & 0x7fffu) > 0x7f80u;
        atomicAdd(&whist[warp][cand_key(b) >> 8], 1u);
    }
    if (__syncthreads_or(nan)) {
        if (t == 0) flags[r] = 1;
        return;
    }
    if (t < 256) {
        unsigned sum = 0u;
        for (int w = 0; w < 32; ++w) sum += whist[w][t];
        hist[t] = sum;
    }
    __syncthreads();
    if (t == 0) {
        const unsigned left = s_left;
        unsigned above = 0u;
        int b = 255;
        for (; b > 0; --b) {
            if (above + hist[b] >= left) break;
            above += hist[b];
        }
        s_left = left - above;
        s_hi = static_cast<unsigned>(b);
    }
    __syncthreads();
    // pass 2: the low bytes of the keys whose high byte is the cut's
    for (int b = t; b < 32 * 256; b += blockDim.x) (&whist[0][0])[b] = 0u;
    __syncthreads();
    const unsigned hi = s_hi;
    for (int i = t; i < V; i += blockDim.x) {
        const unsigned key = cand_key(row[i]);
        if ((key >> 8) == hi) atomicAdd(&whist[warp][key & 255u], 1u);
    }
    __syncthreads();
    if (t < 256) {
        unsigned sum = 0u;
        for (int w = 0; w < 32; ++w) sum += whist[w][t];
        hist[t] = sum;
    }
    __syncthreads();
    if (t == 0) {
        const unsigned left = s_left;
        unsigned above = 0u;
        int b = 255;
        for (; b > 0; --b) {
            if (above + hist[b] >= left) break;
            above += hist[b];
        }
        s_left = left - above;  // the keys equal to the cut that are taken
        s_cut = (hi << 8) | static_cast<unsigned>(b);
        s_above = 0u;
        s_taken = 0u;
    }
    __syncthreads();
    const unsigned cut = s_cut;
    const unsigned ties = s_left;
    const unsigned taken_above = static_cast<unsigned>(k) - ties;
    float* out_v = vals + static_cast<long long>(r) * stride;
    int* out_c = cols + static_cast<long long>(r) * stride;
    for (int base = 0; base < V; base += blockDim.x) {
        const int i = base + t;
        bool tie = false;
        if (i < V) {
            const unsigned short b = row[i];
            const unsigned key = cand_key(b);
            if (key > cut) {
                const unsigned at = atomicAdd(&s_above, 1u);
                out_v[at] = __bfloat162float(__ushort_as_bfloat16(b));
                out_c[at] = i;
            }
            tie = key == cut;
        }
        const unsigned m = __ballot_sync(0xffffffffu, tie);
        if (lane == 0) wtie[warp] = static_cast<unsigned>(__popc(m));
        __syncthreads();
        unsigned before = 0u, total = 0u;
        for (int w = 0; w < 32; ++w) {
            if (w < warp) before += wtie[w];
            total += wtie[w];
        }
        if (tie) {
            const unsigned rank = s_taken + before + static_cast<unsigned>(__popc(m & ((1u << lane) - 1u)));
            if (rank < ties) {
                out_v[taken_above + rank] = __bfloat162float(__ushort_as_bfloat16(row[i]));
                out_c[taken_above + rank] = i;
            }
        }
        __syncthreads();
        if (t == 0) s_taken += total;
    }
    if (t == 0) flags[r] = 0;
}
