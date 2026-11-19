// Hy4-flash forward glue on CUDA (ours), fp32 activations, weights read as stored (f32, bf16 or f16). Each kernel
// mirrors a step of the CPU references in zig/src/families/hy4 (layer_ref.zig, engram_ref.zig, vision_ref.zig), which
// are themselves checked against transformers' HYV4Flash* / HYV4VL* modules. Plain and readable first: the EXL3
// projections run on the shared trellis kernels; everything here is the rest of the model.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

#define DT_F32 0
#define DT_BF16 1
#define DT_F16 2

__device__ __forceinline__ float ldw(const void* p, int dt, long long i) {
    if (dt == DT_BF16) return __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(p)[i]);
    if (dt == DT_F16) return __half2float(reinterpret_cast<const __half*>(p)[i]);
    return reinterpret_cast<const float*>(p)[i];
}
__device__ __forceinline__ float sigm(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ __forceinline__ float silu(float x) { return x * sigm(x); }

template <int THREADS>
__device__ __forceinline__ float block_sum(float v, float* scratch) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    __syncthreads();
    if (lane == 0) scratch[warp] = v;
    __syncthreads();
    if (threadIdx.x < 32) {
        float s = threadIdx.x < THREADS / 32 ? scratch[threadIdx.x] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        if (threadIdx.x == 0) scratch[0] = s;
    }
    __syncthreads();
    const float r = scratch[0];
    __syncthreads();
    return r;
}
__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// ---------------------------------------------------------------- matmuls
// y[r, o] (+)= sum_i x[r, i] w[o, i] (+ b[o]); x f32 rows `ldx` apart, w [n, k] of dtype `wdt`, b of dtype `bdt` or null.
// Tiled 64 x 64 x 16, 256 threads, 4 x 4 outputs a thread.
extern "C" __global__ void __launch_bounds__(256) hy4_gemm(const float* __restrict__ x, long long ldx, const void* __restrict__ w,
                                                           int wdt, const void* __restrict__ b, int bdt, float* __restrict__ y,
                                                           long long ldy, int m, int n, int k, int accumulate) {
    __shared__ float xs[16][64 + 4];
    __shared__ float ws[16][64 + 4];
    const int tx = threadIdx.x % 16, ty = threadIdx.x / 16;
    const int row0 = blockIdx.y * 64, col0 = blockIdx.x * 64;
    float acc[4][4] = {};
    for (int k0 = 0; k0 < k; k0 += 16) {
        for (int i = threadIdx.x; i < 64 * 16; i += 256) {
            const int r = i / 16, kk = i % 16;
            const int gr = row0 + r, gk = k0 + kk;
            xs[kk][r] = (gr < m && gk < k) ? x[(long long)gr * ldx + gk] : 0.0f;
            const int gc = col0 + r;
            ws[kk][r] = (gc < n && gk < k) ? ldw(w, wdt, (long long)gc * k + gk) : 0.0f;
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < 16; ++kk) {
            float a[4], c[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) a[i] = xs[kk][ty * 4 + i];
#pragma unroll
            for (int j = 0; j < 4; ++j) c[j] = ws[kk][tx * 4 + j];
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] += a[i] * c[j];
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int r = row0 + ty * 4 + i;
        if (r >= m) continue;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int c = col0 + tx * 4 + j;
            if (c >= n) continue;
            float v = acc[i][j] + (b ? ldw(b, bdt, c) : 0.0f);
            float* o = y + (long long)r * ldy + c;
            *o = accumulate ? *o + v : v;
        }
    }
}

// The same for at most 8 rows (decode): a warp per output column, lanes along k, every row at once.
extern "C" __global__ void __launch_bounds__(256) hy4_gemv(const float* __restrict__ x, long long ldx, const void* __restrict__ w,
                                                           int wdt, const void* __restrict__ b, int bdt, float* __restrict__ y,
                                                           long long ldy, int m, int n, int k, int accumulate) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int c = blockIdx.x * 8 + warp;
    if (c >= n) return;
    float acc[8] = {};
    for (int i = lane; i < k; i += 32) {
        const float wv = ldw(w, wdt, (long long)c * k + i);
#pragma unroll
        for (int r = 0; r < 8; ++r)
            if (r < m) acc[r] += wv * x[(long long)r * ldx + i];
    }
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        if (r >= m) break;
        const float s = warp_sum(acc[r]);
        if (lane == 0) {
            const float v = s + (b ? ldw(b, bdt, c) : 0.0f);
            float* o = y + (long long)r * ldy + c;
            *o = accumulate ? *o + v : v;
        }
    }
}

// ---------------------------------------------------------------- norms and pointwise
// y = x * rsqrt(mean(x^2) + eps) * w[(row % wmod) * width + i] (w null: unweighted); one block a row.
extern "C" __global__ void __launch_bounds__(256) hy4_rmsnorm(const float* __restrict__ x, long long ldx, const void* __restrict__ w,
                                                              int wdt, int wmod, float* __restrict__ y, long long ldy, int width,
                                                              float eps) {
    __shared__ float scratch[32];
    const long long r = blockIdx.x;
    const float* xr = x + r * ldx;
    float s = 0.0f;
    for (int i = threadIdx.x; i < width; i += 256) s += xr[i] * xr[i];
    const float inv = rsqrtf(block_sum<256>(s, scratch) / (float)width + eps);
    const long long wo = (long long)(wmod > 0 ? (int)(r % wmod) : 0) * width;
    for (int i = threadIdx.x; i < width; i += 256) y[r * ldy + i] = xr[i] * inv * (w ? ldw(w, wdt, wo + i) : 1.0f);
}

// LayerNorm with weight and bias; one block a row.
extern "C" __global__ void __launch_bounds__(256) hy4_layernorm(const float* __restrict__ x, const void* __restrict__ w,
                                                                const void* __restrict__ b, int dt, float* __restrict__ y, int width,
                                                                float eps) {
    __shared__ float scratch[32];
    const long long r = blockIdx.x;
    const float* xr = x + r * width;
    float s = 0.0f;
    for (int i = threadIdx.x; i < width; i += 256) s += xr[i];
    const float mean = block_sum<256>(s, scratch) / (float)width;
    float v = 0.0f;
    for (int i = threadIdx.x; i < width; i += 256) v += (xr[i] - mean) * (xr[i] - mean);
    const float inv = rsqrtf(block_sum<256>(v, scratch) / (float)width + eps);
    for (int i = threadIdx.x; i < width; i += 256) y[r * width + i] = (xr[i] - mean) * inv * ldw(w, dt, i) + ldw(b, dt, i);
}

// out = silu(min(g, c)) * clamp(u, -c, c) (c <= 0: silu(g) * u); g and u rows `ld` apart, out rows `width` apart.
extern "C" __global__ void hy4_swiglu(const float* __restrict__ g, const float* __restrict__ u, long long ld, float* __restrict__ out,
                                      int rows, int width, float clamp) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)rows * width) return;
    const long long r = i / width, c = i % width;
    float gv = g[r * ld + c], uv = u[r * ld + c];
    if (clamp > 0.0f) {
        gv = fminf(gv, clamp);
        uv = fminf(fmaxf(uv, -clamp), clamp);
    }
    out[i] = silu(gv) * uv;
}

// x = gelu_erf(x) (act 1) or x += y (act 0, y given) over `count` values.
extern "C" __global__ void hy4_pointwise(float* __restrict__ x, const float* __restrict__ y, long long count, int act) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    if (act == 1) {
        const double v = x[i];
        x[i] = (float)(0.5 * v * (1.0 + erf(v * 0.70710678118654752440)));
    } else {
        x[i] += y[i];
    }
}

// dst[r, :] = src[r, :] for rows of `width`, each side its own stride.
extern "C" __global__ void hy4_copy2d(const float* __restrict__ src, long long lds, float* __restrict__ dst, long long ldd, int rows,
                                      int width) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)rows * width) return;
    const long long r = i / width, c = i % width;
    dst[r * ldd + c] = src[r * lds + c];
}

// ---------------------------------------------------------------- embedding and hyper-connections
// x[t] = table[ids[t]] (bf16 rows of `hidden`), then every one of the `n` streams of token t starts as x[t].
extern "C" __global__ void hy4_embed(const int* __restrict__ ids, const __nv_bfloat16* __restrict__ table, float* __restrict__ x,
                                     int hidden) {
    const long long t = blockIdx.x;
    const long long row = ids[t];
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) x[t * hidden + i] = __bfloat162float(table[row * hidden + i]);
}
extern "C" __global__ void hy4_expand(const float* __restrict__ x, float* __restrict__ streams, int n, int hidden) {
    const long long t = blockIdx.x;
    for (int i = threadIdx.x; i < n * hidden; i += blockDim.x) streams[t * n * hidden + i] = x[t * hidden + i % hidden];
}

// The final collapse (HYV4FlashHyperHead): pre_s = sigmoid(fn_s . row * rsqrt(mean(row^2)+eps) * scale + base_s) + hc_eps,
// out = sum_s pre_s * stream_s. One block a token, n <= 8.
extern "C" __global__ void __launch_bounds__(256) hy4_hc_head(const float* __restrict__ streams, const float* __restrict__ fn,
                                                              const float* __restrict__ base, const float* __restrict__ scale,
                                                              float* __restrict__ out, int n, int hidden, float eps, float hc_eps) {
    __shared__ float scratch[32];
    __shared__ float pre[8];
    const long long t = blockIdx.x;
    const int width = n * hidden;
    const float* row = streams + t * width;
    float acc[8] = {};
    float sq = 0.0f;
    for (int j = threadIdx.x; j < width; j += 256) {
        const float v = row[j];
        sq += v * v;
        for (int s = 0; s < n; ++s) acc[s] += fn[(long long)s * width + j] * v;
    }
    const float r = rsqrtf(block_sum<256>(sq, scratch) / (float)width + eps);
    for (int s = 0; s < n; ++s) {
        const float m = block_sum<256>(acc[s], scratch);
        if (threadIdx.x == 0) pre[s] = sigm(m * r * scale[0] + base[s]) + hc_eps;
    }
    __syncthreads();
    for (int h = threadIdx.x; h < hidden; h += 256) {
        float o = 0.0f;
        for (int s = 0; s < n; ++s) o += pre[s] * row[(long long)s * hidden + h];
        out[t * hidden + h] = o;
    }
}

// ---------------------------------------------------------------- rotary (interleaved 3D M-RoPE, adjacent pairs)
// Rotates `rope` channels at `offset` of each of `heads` heads (`head_stride` apart) in rows `ld` apart. Row r's
// positions are pos[r + pos0], pos[T + r + pos0], pos[2T + r + pos0] (time, height, width), or with `pos_step` > 1
// the position of row r is taken at (r * pos_step) (indexer group starts).
extern "C" __global__ void hy4_rope(float* __restrict__ x, long long ld, int heads, int head_stride, int offset, int rope,
                                    const int* __restrict__ pos, long long T, long long pos0, int pos_step, double theta, int s1,
                                    int s2) {
    const long long r = blockIdx.x;
    const long long p = pos0 + r * pos_step;
    const int pt = pos[p], ph = pos[T + p], pw = pos[2 * T + p];
    const int pairs = rope / 2;
    for (int i = threadIdx.x; i < heads * pairs; i += blockDim.x) {
        const int h = i / pairs, kk = i % pairs;
        const int plane = (kk % 3 == 1 && kk < 3 * s1) ? 1 : ((kk % 3 == 2 && kk < 3 * s2) ? 2 : 0);
        const int pv = plane == 0 ? pt : (plane == 1 ? ph : pw);
        const float inv = (float)(1.0 / pow(theta, (double)(2 * kk) / (double)rope));
        const float ang = (float)((double)pv * (double)inv);
        const float c = cosf(ang), s = sinf(ang);
        float* v = x + r * ld + (long long)h * head_stride + offset + 2 * kk;
        const float a = v[0], b = v[1];
        v[0] = a * c - b * s;
        v[1] = b * c + a * s;
    }
}

// ---------------------------------------------------------------- MLA with sinks and the output gate
// ql[t, h, c] = sum_{d < nope} q[t, h, d] * kvb[h * (nope + v) + d, c]  (the key up-projection absorbed into the query)
extern "C" __global__ void hy4_absorb_q(const float* __restrict__ q, const void* __restrict__ kvb, int kdt, float* __restrict__ ql,
                                        int heads, int qk, int nope, int vdim, int kvl) {
    extern __shared__ float qs[];
    const long long th = blockIdx.x;
    const int h = (int)(th % heads);
    const float* qr = q + th * qk;
    for (int d = threadIdx.x; d < nope; d += blockDim.x) qs[d] = qr[d];
    __syncthreads();
    const long long base = (long long)h * (nope + vdim);
    for (int c = threadIdx.x; c < kvl; c += blockDim.x) {
        float s = 0.0f;
        for (int d = 0; d < nope; ++d) s += qs[d] * ldw(kvb, kdt, (base + d) * kvl + c);
        ql[th * kvl + c] = s;
    }
}

// One query row and head a block: scores over the latent cache (ql . lat_j + q_rot . rot_j) * scale for the keys it may
// see (causal; `window` > 0: the last `window`; `keep` non-null: only groups keep[t][g] or its own group), a softmax
// whose denominator also holds exp(sink), and ol = the weights times the latents. Online, 32 keys a tile.
#define ATT_THREADS 256
extern "C" __global__ void __launch_bounds__(ATT_THREADS) hy4_mla_attn(
    const float* __restrict__ ql, const float* __restrict__ q, int qk, int nope, const float* __restrict__ lat,
    const float* __restrict__ rot, const float* __restrict__ sinks, float* __restrict__ ol, int heads, int kvl, int rope,
    long long qpos0, int window, const unsigned char* __restrict__ keep, long long keep_ld, int kpool, float scale) {
    __shared__ float qsh[640];
    __shared__ float sc[32];
    __shared__ float stat[2];
    const long long t = blockIdx.x;
    const int h = blockIdx.y;
    const long long th = t * heads + h;
    for (int c = threadIdx.x; c < kvl; c += ATT_THREADS) qsh[c] = ql[th * kvl + c];
    for (int c = threadIdx.x; c < rope; c += ATT_THREADS) qsh[kvl + c] = q[th * qk + nope + c];
    const long long i = qpos0 + t;
    const long long lo = window > 0 ? (i - window + 1 > 0 ? i - window + 1 : 0) : 0;
    const long long own = kpool > 0 ? i / kpool : 0;
    const int per = (kvl + ATT_THREADS - 1) / ATT_THREADS;  // channels a thread accumulates (kvl <= 1024)
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float mx = sinks[h], sum = 1.0f;  // the sink: exp(sink - mx) with mx = sink
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    __syncthreads();
    for (long long j0 = lo; j0 <= i; j0 += 32) {
        // 8 warps score 32 keys, 4 each
        for (int kk = warp; kk < 32; kk += ATT_THREADS / 32) {
            const long long j = j0 + kk;
            float s = -INFINITY;
            bool ok = j <= i;
            if (ok && keep) {
                const long long g = j / kpool;
                ok = g == own || keep[t * keep_ld + g];
            }
            if (ok) {
                float d = 0.0f;
                const float* lj = lat + j * kvl;
                for (int c = lane; c < kvl; c += 32) d += qsh[c] * lj[c];
                const float* rj = rot + j * rope;
                for (int c = lane; c < rope; c += 32) d += qsh[kvl + c] * rj[c];
                s = warp_sum(d) * scale;
            }
            if (lane == 0) sc[kk] = s;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            float m = mx;
            for (int kk = 0; kk < 32; ++kk) m = fmaxf(m, sc[kk]);
            const float corr = expf(mx - m);
            float ssum = sum * corr;
            for (int kk = 0; kk < 32; ++kk) {
                const float p = sc[kk] == -INFINITY ? 0.0f : expf(sc[kk] - m);
                sc[kk] = p;
                ssum += p;
            }
            stat[0] = corr;
            stat[1] = m;
            sum = ssum;
        }
        __syncthreads();
        const float corr = stat[0];
        mx = stat[1];
        for (int e = 0; e < per; ++e) {
            const int c = threadIdx.x + e * ATT_THREADS;
            if (c >= kvl) break;
            float a = acc[e] * corr;
            for (int kk = 0; kk < 32; ++kk) {
                const float p = sc[kk];
                if (p != 0.0f) a += p * lat[(j0 + kk) * kvl + c];
            }
            acc[e] = a;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) stat[0] = sum;
    __syncthreads();
    const float inv = 1.0f / stat[0];
    for (int e = 0; e < per; ++e) {
        const int c = threadIdx.x + e * ATT_THREADS;
        if (c < kvl) ol[th * kvl + c] = acc[e] * inv;
    }
}

// out[t, h, e] = sigmoid(gate[t, h, e]) * sum_c kvb[h * (nope + v) + nope + e, c] * ol[t, h, c]; a warp a value channel,
// lanes along c, up to 8 rows a block.
extern "C" __global__ void __launch_bounds__(256) hy4_uv_gate(const float* __restrict__ ol, const void* __restrict__ kvb, int kdt,
                                                              const float* __restrict__ gate, float* __restrict__ out, int rows,
                                                              int heads, int nope, int vdim, int kvl) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int h = blockIdx.y;
    const int e = blockIdx.x * 8 + warp;
    if (e >= vdim) return;
    const long long wrow = ((long long)h * (nope + vdim) + nope + e) * kvl;
    const long long r0 = (long long)blockIdx.z * 8;
    float acc[8] = {};
    for (int c = lane; c < kvl; c += 32) {
        const float w = ldw(kvb, kdt, wrow + c);
#pragma unroll
        for (int r = 0; r < 8; ++r)
            if (r0 + r < rows) acc[r] += w * ol[((r0 + r) * heads + h) * kvl + c];
    }
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        if (r0 + r >= rows) break;
        const float s = warp_sum(acc[r]);
        if (lane == 0) {
            const long long o = ((r0 + r) * heads + h) * vdim + e;
            out[o] = s * sigm(gate[o]);
        }
    }
}

// ---------------------------------------------------------------- the compressed indexer
// Group g's key from its `kpool` token keys: values = kv @ tok, gates = kg @ tok + ape[slot], softmax over the slots per
// channel, key_norm (RMSNorm, f32 weight). The rotary part is applied after (hy4_rope with pos_step = kpool).
extern "C" __global__ void __launch_bounds__(128) hy4_idx_pool(const float* __restrict__ tok, const float* __restrict__ kv,
                                                               const float* __restrict__ kg, const float* __restrict__ ape,
                                                               const float* __restrict__ knorm, float* __restrict__ gk, long long g0,
                                                               int dim, int kpool, float eps) {
    __shared__ float ts[8][128];
    __shared__ float scratch[32];
    const long long g = g0 + blockIdx.x;
    const int c = threadIdx.x;
    for (int s = 0; s < kpool; ++s) ts[s][c] = tok[(g * kpool + s) * dim + c];
    __syncthreads();
    float vals[8], gts[8];
    float most = -INFINITY;
    for (int s = 0; s < kpool; ++s) {
        float v = 0.0f, q = 0.0f;
        for (int i = 0; i < dim; ++i) {
            v += kv[c * dim + i] * ts[s][i];
            q += kg[c * dim + i] * ts[s][i];
        }
        vals[s] = v;
        gts[s] = q + ape[s * dim + c];
        most = fmaxf(most, gts[s]);
    }
    float sum = 0.0f, acc = 0.0f;
    for (int s = 0; s < kpool; ++s) {
        const float e = expf(gts[s] - most);
        sum += e;
        acc += vals[s] * e;
    }
    const float pooled = acc / sum;
    const float inv = rsqrtf(block_sum<128>(pooled * pooled, scratch) / (float)dim + eps);
    gk[g * dim + c] = pooled * inv * knorm[c];
}

// logits[t, g] = sum_h hw[t, h] * hscale * relu(iq[t, h] . gk[g]) for the groups closed before the query's own, else -inf.
extern "C" __global__ void __launch_bounds__(256) hy4_idx_score(const float* __restrict__ iq, const float* __restrict__ hw,
                                                                const float* __restrict__ gk, float* __restrict__ logits,
                                                                long long ld, int heads, int dim, int groups, long long qpos0,
                                                                int kpool, float hscale) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const long long t = blockIdx.y;
    const int g = blockIdx.x * 8 + warp;
    if (g >= groups) return;
    const long long own = (qpos0 + t) / kpool;
    if (g >= own) {
        if (lane == 0) logits[t * ld + g] = -INFINITY;
        return;
    }
    float l = 0.0f;
    for (int h = 0; h < heads; ++h) {
        float s = 0.0f;
        for (int c = lane; c < dim; c += 32) s += iq[(t * heads + h) * dim + c] * gk[(long long)g * dim + c];
        s = warp_sum(s);
        l += hw[t * heads + h] * hscale * fmaxf(s, 0.0f);
    }
    if (lane == 0) logits[t * ld + g] = l;
}

__device__ __forceinline__ unsigned int order_key(float f) {
    const unsigned int u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// keep[t, g] for the `recall` best groups below the query's own (ties: the lower group, as the reference's strict scan);
// all of them when fewer. One block a query row.
extern "C" __global__ void __launch_bounds__(256) hy4_idx_topk(const float* __restrict__ logits, long long ld,
                                                               unsigned char* __restrict__ keep, long long kld, int groups,
                                                               long long qpos0, int kpool, int recall) {
    __shared__ unsigned int cnt;
    __shared__ unsigned int thr;
    const long long t = blockIdx.x;
    const long long ownl = (qpos0 + t) / kpool;
    const int own = (int)(ownl < groups ? ownl : groups);
    const float* row = logits + t * ld;
    unsigned char* kr = keep + t * kld;
    for (int g = threadIdx.x; g < groups; g += 256) kr[g] = 0;
    __syncthreads();
    if (own <= recall) {
        for (int g = threadIdx.x; g < own; g += 256) kr[g] = 1;
        return;
    }
    // the largest key k with count(key >= k) >= recall, bit by bit
    if (threadIdx.x == 0) thr = 0;
    __syncthreads();
    for (int bit = 31; bit >= 0; --bit) {
        const unsigned int cand = thr | (1u << bit);
        if (threadIdx.x == 0) cnt = 0;
        __syncthreads();
        unsigned int local = 0;
        for (int g = threadIdx.x; g < own; g += 256) local += order_key(row[g]) >= cand;
        atomicAdd(&cnt, local);
        __syncthreads();
        if (threadIdx.x == 0 && cnt >= (unsigned int)recall) thr = cand;
        __syncthreads();
    }
    if (threadIdx.x == 0) cnt = 0;
    __syncthreads();
    unsigned int above = 0;
    for (int g = threadIdx.x; g < own; g += 256)
        if (order_key(row[g]) > thr) {
            kr[g] = 1;
            ++above;
        }
    atomicAdd(&cnt, above);
    __syncthreads();
    if (threadIdx.x == 0) {
        int left = recall - (int)cnt;
        for (int g = 0; g < own && left > 0; ++g)
            if (order_key(row[g]) == thr) {
                kr[g] = 1;
                --left;
            }
    }
}

// ---------------------------------------------------------------- the latent MoE
// Sigmoid scores, the top `top` by score + bias (ties: the lower expert), weights score / sum * scale. One block a row.
extern "C" __global__ void __launch_bounds__(256) hy4_route(const float* __restrict__ logits, const float* __restrict__ bias,
                                                            int experts, int top, float scale, int* __restrict__ ids,
                                                            float* __restrict__ wts) {
    __shared__ float score[1024];
    __shared__ unsigned char taken[1024];
    __shared__ float bv[256];
    __shared__ int bi[256];
    const long long t = blockIdx.x;
    for (int e = threadIdx.x; e < experts; e += 256) {
        score[e] = sigm(logits[t * experts + e]);
        taken[e] = 0;
    }
    __syncthreads();
    float total = 0.0f;
    for (int k = 0; k < top; ++k) {
        float best = -INFINITY;
        int at = 0x7fffffff;
        for (int e = threadIdx.x; e < experts; e += 256) {
            if (taken[e]) continue;
            const float v = score[e] + bias[e];
            if (v > best || (v == best && e < at)) {
                best = v;
                at = e;
            }
        }
        bv[threadIdx.x] = best;
        bi[threadIdx.x] = at;
        __syncthreads();
        for (int s = 128; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                const float ov = bv[threadIdx.x + s];
                const int oi = bi[threadIdx.x + s];
                if (ov > bv[threadIdx.x] || (ov == bv[threadIdx.x] && oi < bi[threadIdx.x])) {
                    bv[threadIdx.x] = ov;
                    bi[threadIdx.x] = oi;
                }
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            const int e = bi[0];
            taken[e] = 1;
            ids[t * top + k] = e;
            wts[t * top + k] = score[e];
        }
        __syncthreads();
        total += score[bi[0]];
        __syncthreads();
    }
    if (threadIdx.x < top) wts[t * top + threadIdx.x] = wts[t * top + threadIdx.x] / (total + 1e-20f) * scale;
}

// dst[r] = src[idx[r]] (rows of `width`)
extern "C" __global__ void hy4_gather(const float* __restrict__ src, const int* __restrict__ idx, float* __restrict__ dst, int width) {
    const long long r = blockIdx.x;
    const long long s = idx[r];
    for (int i = threadIdx.x; i < width; i += blockDim.x) dst[r * width + i] = src[s * width + i];
}
// dst[idx[r]] = src[r]
extern "C" __global__ void hy4_scatter(const float* __restrict__ src, const int* __restrict__ idx, float* __restrict__ dst, int width) {
    const long long r = blockIdx.x;
    const long long d = idx[r];
    for (int i = threadIdx.x; i < width; i += blockDim.x) dst[d * width + i] = src[r * width + i];
}
// acc[t] = sum_k wts[t, k] * slots[t * top + k], in k order (deterministic).
extern "C" __global__ void hy4_combine(const float* __restrict__ slots, const float* __restrict__ wts, float* __restrict__ acc, int top,
                                       int width) {
    const long long t = blockIdx.x;
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        float s = 0.0f;
        for (int k = 0; k < top; ++k) s += slots[(t * top + k) * width + i] * wts[t * top + k];
        acc[t * width + i] = s;
    }
}

// ---------------------------------------------------------------- Engram
// streams[t, s] += mask[t] * (sigmoid(signed_sqrt(nk[t, s] . rmsnorm(streams[t, s]) * w_q[s] / sqrt(H))) * main[t]
//                              + scales[s] * main[t]); one block a token and stream. nk is already normed with its weight.
extern "C" __global__ void __launch_bounds__(256) hy4_engram_gate(float* __restrict__ streams, const float* __restrict__ nk,
                                                                  const float* __restrict__ main_, const void* __restrict__ nq_w,
                                                                  const void* __restrict__ scales, int dt,
                                                                  const unsigned char* __restrict__ mask, int n, int hidden,
                                                                  float eps) {
    __shared__ float scratch[32];
    const long long t = blockIdx.x;
    const int s = blockIdx.y;
    if (!mask[t]) return;
    float* st = streams + (t * n + s) * hidden;
    float sq = 0.0f;
    for (int i = threadIdx.x; i < hidden; i += 256) sq += st[i] * st[i];
    const float inv = rsqrtf(block_sum<256>(sq, scratch) / (float)hidden + eps);
    float d = 0.0f;
    for (int i = threadIdx.x; i < hidden; i += 256) d += nk[(t * n + s) * hidden + i] * st[i] * inv * ldw(nq_w, dt, (long long)s * hidden + i);
    float g = block_sum<256>(d, scratch) * rsqrtf((float)hidden);
    g = sqrtf(fmaxf(fabsf(g), 1e-6f)) * (g > 0.0f ? 1.0f : (g < 0.0f ? -1.0f : 0.0f));
    const float gate = sigm(g);
    for (int i = threadIdx.x; i < hidden; i += 256) {
        const float m = main_[t * hidden + i];
        st[i] += gate * m + ldw(scales, dt, (long long)s * hidden + i) * m;
    }
}

// ---------------------------------------------------------------- vision
// Full attention within one picture: q, k, v rows `ld` apart (heads of `hd`), out rows `ld` apart; a block a row and head.
extern "C" __global__ void __launch_bounds__(128) hy4_vit_attn(const float* __restrict__ q, const float* __restrict__ k,
                                                               const float* __restrict__ v, float* __restrict__ out, long long ld,
                                                               int len, int hd, float scale) {
    __shared__ float qs[256];
    __shared__ float sc[32];
    __shared__ float stat[2];
    const long long i = blockIdx.x;
    const int h = blockIdx.y;
    for (int c = threadIdx.x; c < hd; c += 128) qs[c] = q[i * ld + h * hd + c];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float acc[2] = {0.0f, 0.0f};
    float mx = -INFINITY, sum = 0.0f;
    __syncthreads();
    for (int j0 = 0; j0 < len; j0 += 32) {
        for (int kk = warp; kk < 32; kk += 4) {
            const int j = j0 + kk;
            float s = -INFINITY;
            if (j < len) {
                float d = 0.0f;
                for (int c = lane; c < hd; c += 32) d += qs[c] * k[(long long)j * ld + h * hd + c];
                s = warp_sum(d) * scale;
            }
            if (lane == 0) sc[kk] = s;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            float m = mx;
            for (int kk = 0; kk < 32; ++kk) m = fmaxf(m, sc[kk]);
            const float corr = mx == -INFINITY ? 0.0f : expf(mx - m);
            float ss = sum * corr;
            for (int kk = 0; kk < 32; ++kk) {
                const float p = sc[kk] == -INFINITY ? 0.0f : expf(sc[kk] - m);
                sc[kk] = p;
                ss += p;
            }
            stat[0] = corr;
            stat[1] = m;
            sum = ss;
        }
        __syncthreads();
        const float corr = stat[0];
        mx = stat[1];
        for (int e = 0; e < 2; ++e) {
            const int c = threadIdx.x + e * 128;
            if (c >= hd) break;
            float a = acc[e] * corr;
            for (int kk = 0; kk < 32; ++kk)
                if (sc[kk] != 0.0f) a += sc[kk] * v[(long long)(j0 + kk) * ld + h * hd + c];
            acc[e] = a;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) stat[0] = sum;
    __syncthreads();
    for (int e = 0; e < 2; ++e) {
        const int c = threadIdx.x + e * 128;
        if (c < hd) out[i * ld + h * hd + c] = acc[e] / stat[0];
    }
}

// The merger's 2 x 2 cells: cell[(r * gw2 + c), ci * 4 + ky * 2 + kx] = x[(2r + ky) * gw + 2c + kx, ci], the conv weight's
// [out][in][ky][kx] order, so the stride-2 convolution is one matmul.
extern "C" __global__ void hy4_merge_gather(const float* __restrict__ x, float* __restrict__ cell, int gw, int gw2, int hidden) {
    const long long o = blockIdx.x;
    const long long r = o / gw2, c = o % gw2;
    for (int i = threadIdx.x; i < hidden * 4; i += blockDim.x) {
        const int ci = i / 4, ky = (i / 2) % 2, kx = i % 2;
        cell[o * hidden * 4 + i] = x[((2 * r + ky) * gw + 2 * c + kx) * hidden + ci];
    }
}
// Rows of `gw2` merged cells, each row closed by the newline vector: out[(r * (gw2 + 1) + c)] = cells[r * gw2 + c].
extern "C" __global__ void hy4_newlines(const float* __restrict__ cells, const void* __restrict__ newline, int dt, float* __restrict__ out,
                                        int gw2, int width) {
    const long long o = blockIdx.x;
    const long long r = o / (gw2 + 1), c = o % (gw2 + 1);
    for (int i = threadIdx.x; i < width; i += blockDim.x)
        out[o * width + i] = c == gw2 ? ldw(newline, dt, i) : cells[(r * gw2 + c) * width + i];
}

// the hyper-connection pre/post kernels live beside these in one image
#include "hc.cu"
