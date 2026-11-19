// Hy4-flash's hyper-connections (iHC), fp32 (ours; reference: families/hy4/layer_ref.zig `hyper`, itself checked
// against transformers' HYV4FlashHyperConnection). A token's `n` streams of `hidden` values are one row of n * hidden.
//   pre:  mix = fn @ row ([2n] dots), r = rsqrt(mean(row^2) + eps),
//         pre_s = sigmoid(mix_s * r * scale0 + base_s) + hc_eps, post_s = mag * sigmoid(mix_{n+s} * r * scale1 + base_{n+s}) + hc_eps,
//         mixed = sum_s pre_s * stream_s
//   post: stream_s += post_s * y
// One block per token; up to 8 streams.
#define HY4_HC_MAX 8
#define HY4_HC_THREADS 256

__device__ __forceinline__ float hy4_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

__device__ __forceinline__ float hy4_block_sum(float v, float* scratch) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    __syncthreads();
    if (lane == 0) scratch[warp] = v;
    __syncthreads();
    float s = 0.0f;
    if (threadIdx.x < 32) {
        s = threadIdx.x < (HY4_HC_THREADS / 32) ? scratch[threadIdx.x] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        if (threadIdx.x == 0) scratch[0] = s;
    }
    __syncthreads();
    return scratch[0];
}

extern "C" __global__ void __launch_bounds__(HY4_HC_THREADS) hy4_hc_pre(
    const float* __restrict__ streams, const float* __restrict__ fn, const float* __restrict__ base,
    const float* __restrict__ scale, float* __restrict__ mixed, float* __restrict__ post,
    int n, int hidden, float eps, float hc_eps, float magnitude) {
    __shared__ float scratch[32];
    __shared__ float pre_w[HY4_HC_MAX];
    const long long t = blockIdx.x;
    const int width = n * hidden;
    const float* row = streams + t * width;
    float acc[2 * HY4_HC_MAX];
    float sq = 0.0f;
    for (int k = 0; k < 2 * n; ++k) acc[k] = 0.0f;
    for (int j = threadIdx.x; j < width; j += HY4_HC_THREADS) {
        const float x = row[j];
        sq += x * x;
        for (int k = 0; k < 2 * n; ++k) acc[k] += fn[(long long)k * width + j] * x;
    }
    const float r = rsqrtf(hy4_block_sum(sq, scratch) / (float)width + eps);
    for (int k = 0; k < 2 * n; ++k) {
        const float m = hy4_block_sum(acc[k], scratch);
        if (threadIdx.x == 0) {
            if (k < n) pre_w[k] = hy4_sigmoid(m * r * scale[0] + base[k]) + hc_eps;
            else post[t * n + (k - n)] = magnitude * hy4_sigmoid(m * r * scale[1] + base[k]) + hc_eps;
        }
    }
    __syncthreads();
    for (int h = threadIdx.x; h < hidden; h += HY4_HC_THREADS) {
        float o = 0.0f;
        for (int s = 0; s < n; ++s) o += pre_w[s] * row[(long long)s * hidden + h];
        mixed[t * hidden + h] = o;
    }
}

extern "C" __global__ void __launch_bounds__(HY4_HC_THREADS) hy4_hc_post(
    float* __restrict__ streams, const float* __restrict__ post, const float* __restrict__ y, int n, int hidden) {
    const long long t = blockIdx.x;
    for (int i = threadIdx.x; i < n * hidden; i += HY4_HC_THREADS) {
        const int s = i / hidden, h = i - s * hidden;
        streams[t * n * hidden + i] += post[t * n + s] * y[t * hidden + h];
    }
}
