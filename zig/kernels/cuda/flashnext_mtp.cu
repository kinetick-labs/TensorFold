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

// The eight columns [col0, col0 + 8) of a row, those below `end` only (their count is returned): one 16-byte load when the row is
// 16-byte aligned and all eight are there.
static __device__ __forceinline__ int cand_load8(const unsigned short* row, int end, int col0, bool aligned, unsigned short* e) {
    int n = end - col0;
    n = n < 0 ? 0 : (n > 8 ? 8 : n);
    if (n == 8 && aligned) {
        const uint4 q = *reinterpret_cast<const uint4*>(row + col0);
        e[0] = static_cast<unsigned short>(q.x & 0xffffu);
        e[1] = static_cast<unsigned short>(q.x >> 16);
        e[2] = static_cast<unsigned short>(q.y & 0xffffu);
        e[3] = static_cast<unsigned short>(q.y >> 16);
        e[4] = static_cast<unsigned short>(q.z & 0xffffu);
        e[5] = static_cast<unsigned short>(q.z >> 16);
        e[6] = static_cast<unsigned short>(q.w & 0xffffu);
        e[7] = static_cast<unsigned short>(q.w >> 16);
    } else {
#pragma unroll
        for (int j = 0; j < 8; ++j) e[j] = j < n ? row[col0 + j] : static_cast<unsigned short>(0);
    }
    return n;
}

// h[bin] += 1 for every lane of the warp with `valid` (all 32 lanes call): lanes of one bin add together, the first of them its
// count, since the scores of a row fall in a few bins and a lane a bin would queue on the same word.
static __device__ __forceinline__ void cand_count(unsigned* h, unsigned bin, bool valid, int lane) {
    const unsigned peers = __match_any_sync(0xffffffffu, valid ? bin : 0x100u);
    if (valid && (peers & ((1u << lane) - 1u)) == 0u) atomicAdd(&h[bin], static_cast<unsigned>(__popc(peers)));
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
// histogram passes over the 16-bit keys; the keys above it are taken whole, and of the keys equal to it the lowest columns.
// Every pass is a warp a run of columns (a lane eight columns of a 256-column step, loads four steps ahead, no barrier in a
// pass): the histograms are a warp's, counted a bin at a time (cand_count); the ties are ranked by column in the run (ballot-free
// scan over the lanes), after a barrier that gives each warp the ties of the runs before it.
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
    __shared__ unsigned s_left, s_hi, s_cut, s_above;
    const bool aligned = (reinterpret_cast<unsigned long long>(row) & 15ull) == 0ull;
    const int nw = static_cast<int>(blockDim.x >> 5);
    const int span = ((V + nw - 1) / nw + 255) & ~255;
    const int lo = warp * span;
    const int end = lo + span < V ? lo + span : V;
    const int chunks = end > lo ? (end - lo + 255) >> 8 : 0;  // 256-column steps of this warp's run (the same for its lanes)
    // The fast path: a strided sample of the row gives a threshold key T with a few times `count` keys at or above it; one pass
    // collects the keys >= T (columns and scores, at most 1,024 of them) and ranks them; the first `count` in the draw's order are
    // the row's. A row with fewer than `count` or more than 1,024 keys >= T takes the exact two-histogram path below.
    __shared__ unsigned short lraw[1024];
    __shared__ int lcol[1024];
    __shared__ unsigned s_n;
    {
        const int stride_s = V <= 8192 ? 1 : V >> 12;
        const int n_s = (V + stride_s - 1) / stride_s;
        const unsigned jj = stride_s == 1 ? static_cast<unsigned>(k)
                                          : min(static_cast<unsigned>(n_s), (6u * static_cast<unsigned>(k) + stride_s - 1u) / static_cast<unsigned>(stride_s) + 1u);
        unsigned skey[8];
        bool sval[8];
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            const int idx = t + u * 1024;
            sval[u] = idx < n_s;
            skey[u] = sval[u] ? cand_key(row[idx * stride_s]) : 0u;
        }
        for (int b = t; b < 32 * 256; b += blockDim.x) (&whist[0][0])[b] = 0u;
        if (t == 0) {
            s_left = jj;
            s_n = 0u;
        }
        __syncthreads();
#pragma unroll
        for (int u = 0; u < 8; ++u) cand_count(whist[warp], skey[u] >> 8, sval[u], lane);
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
            s_left = left - above;
            s_hi = static_cast<unsigned>(b);
        }
        __syncthreads();
        for (int b = t; b < 32 * 256; b += blockDim.x) (&whist[0][0])[b] = 0u;
        __syncthreads();
        const unsigned shi = s_hi;
#pragma unroll
        for (int u = 0; u < 8; ++u) cand_count(whist[warp], skey[u] & 255u, sval[u] && (skey[u] >> 8) == shi, lane);
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
            s_cut = (shi << 8) | static_cast<unsigned>(b);
        }
        __syncthreads();
        const unsigned thr = s_cut;
        bool nan_f = false;
        for (int c0 = 0; c0 < chunks; c0 += 4) {
            unsigned short e[4][8];
            int n[4];
#pragma unroll
            for (int u = 0; u < 4; ++u) n[u] = c0 + u < chunks ? cand_load8(row, end, lo + (c0 + u) * 256 + lane * 8, aligned, e[u]) : 0;
#pragma unroll
            for (int u = 0; u < 4; ++u) {
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const bool valid = j < n[u];
                    nan_f |= valid && (e[u][j] & 0x7fffu) > 0x7f80u;
                    const unsigned m = __ballot_sync(0xffffffffu, valid && cand_key(e[u][j]) >= thr);
                    if (m != 0u) {
                        unsigned base = 0u;
                        if (lane == 0) base = atomicAdd(&s_n, static_cast<unsigned>(__popc(m)));
                        base = __shfl_sync(0xffffffffu, base, 0);
                        if ((m >> lane) & 1u) {
                            const unsigned pos = base + static_cast<unsigned>(__popc(m & ((1u << lane) - 1u)));
                            if (pos < 1024u) {
                                lraw[pos] = e[u][j];
                                lcol[pos] = lo + (c0 + u) * 256 + lane * 8 + j;
                            }
                        }
                    }
                }
            }
        }
        if (__syncthreads_or(nan_f)) {
            if (t == 0) flags[r] = 1;
            return;
        }
        const unsigned found = s_n;
        if (found >= static_cast<unsigned>(k) && found <= 1024u) {
            if (static_cast<unsigned>(t) < found) {
                const unsigned my_key = cand_key(lraw[t]);
                const int my_col = lcol[t];
                unsigned rank = 0u;
                for (unsigned i = 0; i < found; ++i) {
                    const unsigned kk = cand_key(lraw[i]);
                    rank += (kk > my_key || (kk == my_key && lcol[i] < my_col)) ? 1u : 0u;
                }
                if (rank < static_cast<unsigned>(k)) {
                    vals[static_cast<long long>(r) * stride + rank] = __bfloat162float(__ushort_as_bfloat16(lraw[t]));
                    cols[static_cast<long long>(r) * stride + rank] = my_col;
                }
            }
            if (t == 0) flags[r] = 0;
            return;
        }
        __syncthreads();
    }
    if (t == 0) s_left = static_cast<unsigned>(k);
    // pass 1: the keys' high bytes, a histogram a warp; a NaN ends the row
    for (int b = t; b < 32 * 256; b += blockDim.x) (&whist[0][0])[b] = 0u;
    __syncthreads();
    bool nan = false;
    for (int c0 = 0; c0 < chunks; c0 += 4) {
        unsigned short e[4][8];
        int n[4];
#pragma unroll
        for (int u = 0; u < 4; ++u) n[u] = c0 + u < chunks ? cand_load8(row, end, lo + (c0 + u) * 256 + lane * 8, aligned, e[u]) : 0;
#pragma unroll
        for (int u = 0; u < 4; ++u) {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const bool valid = j < n[u];
                nan |= valid && (e[u][j] & 0x7fffu) > 0x7f80u;
                cand_count(whist[warp], cand_key(e[u][j]) >> 8, valid, lane);
            }
        }
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
    for (int c0 = 0; c0 < chunks; c0 += 4) {
        unsigned short e[4][8];
        int n[4];
#pragma unroll
        for (int u = 0; u < 4; ++u) n[u] = c0 + u < chunks ? cand_load8(row, end, lo + (c0 + u) * 256 + lane * 8, aligned, e[u]) : 0;
#pragma unroll
        for (int u = 0; u < 4; ++u) {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const unsigned key = cand_key(e[u][j]);
                cand_count(whist[warp], key & 255u, j < n[u] && (key >> 8) == hi, lane);
            }
        }
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
    }
    __syncthreads();
    const unsigned cut = s_cut;
    const unsigned ties = s_left;
    const unsigned taken_above = static_cast<unsigned>(k) - ties;
    float* out_v = vals + static_cast<long long>(r) * stride;
    int* out_c = cols + static_cast<long long>(r) * stride;
    // sweep 1: the keys above the cut are written (any order), the run's ties counted
    unsigned lane_ties = 0u;
    for (int c0 = 0; c0 < chunks; c0 += 4) {
        unsigned short e[4][8];
        int n[4];
#pragma unroll
        for (int u = 0; u < 4; ++u) n[u] = c0 + u < chunks ? cand_load8(row, end, lo + (c0 + u) * 256 + lane * 8, aligned, e[u]) : 0;
#pragma unroll
        for (int u = 0; u < 4; ++u) {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                if (j >= n[u]) continue;
                const unsigned key = cand_key(e[u][j]);
                if (key > cut) {
                    const unsigned at = atomicAdd(&s_above, 1u);
                    out_v[at] = __bfloat162float(__ushort_as_bfloat16(e[u][j]));
                    out_c[at] = lo + (c0 + u) * 256 + lane * 8 + j;
                }
                lane_ties += key == cut ? 1u : 0u;
            }
        }
    }
    unsigned my_ties = lane_ties;
    for (int o = 16; o; o >>= 1) my_ties += __shfl_xor_sync(0xffffffffu, my_ties, o);
    if (lane == 0) wtie[warp] = my_ties;
    __syncthreads();
    unsigned run = 0u;
    for (int w = 0; w < warp; ++w) run += wtie[w];
    // sweep 2: the run's ties by column, ranked after the `run` ties of the runs before; those ranked below `ties` are written
    if (my_ties != 0u) {
        for (int c0 = 0; c0 < chunks && run < ties; c0 += 4) {
            unsigned short e[4][8];
            int n[4];
#pragma unroll
            for (int u = 0; u < 4; ++u) n[u] = c0 + u < chunks ? cand_load8(row, end, lo + (c0 + u) * 256 + lane * 8, aligned, e[u]) : 0;
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                unsigned bits = 0u;
#pragma unroll
                for (int j = 0; j < 8; ++j) bits |= (j < n[u] && cand_key(e[u][j]) == cut) ? (1u << j) : 0u;
                const unsigned cnt = static_cast<unsigned>(__popc(bits));
                unsigned incl = cnt;
#pragma unroll
                for (int o = 1; o < 32; o <<= 1) {
                    const unsigned x = __shfl_up_sync(0xffffffffu, incl, o);
                    if (lane >= o) incl += x;
                }
                unsigned rank = run + incl - cnt;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    if ((bits >> j) & 1u) {
                        if (rank < ties) {
                            out_v[taken_above + rank] = __bfloat162float(__ushort_as_bfloat16(e[u][j]));
                            out_c[taken_above + rank] = lo + (c0 + u) * 256 + lane * 8 + j;
                        }
                        ++rank;
                    }
                }
                run += __shfl_sync(0xffffffffu, incl, 31);
            }
        }
    }
    if (t == 0) flags[r] = 0;
}
