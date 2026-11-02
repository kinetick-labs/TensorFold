#include "hc_native_support.h"
namespace cu {
static int cpu_block;
// One value head per block: depthwise conv, the delta-rule step, then the gated RMSNorm, without FMA contraction.





constexpr int DK = 128, DV = 128, TAPS = 4;

static inline float bf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }



static inline float sigmoidf_(float x) { const float e = expf(-fabsf(x)); return (x < 0 ? e : 1.0f) / (1.0f + e); }
static inline float siluf_(float x) { const float e = expf(-fabsf(x)); return (x < 0 ? x * e : x) / (1.0f + e); }

static inline float softplusf_(float x) { return x > 20.0f ? x : log1pf(expf(x)); }



Task qwen4_gdn_chain(
        const __nv_bfloat16* __restrict__ P, const __nv_bfloat16* __restrict__ cs,
        const void* __restrict__ cw, const float* state_in, const float* __restrict__ a_log,
        const float* __restrict__ dt_bias, const void* __restrict__ norm_w, float eps, int rows,
        __nv_bfloat16* __restrict__ out, float* __restrict__ xs, float* state_out, int NK, int NV, int gate, int weight_flags, unsigned long long STATE_STRIDE) {
    const int C = 2 * NK * DK + NV * DV;
    const int PW = C + NV * DV + 2 * NV;
    const int hv = cpu_block, hk = hv / (NV / NK);
    const int t = cpu_lane, warp = t >> 5, lane = t & 31;
    static float qs[DK], ks[DK], vs[DV], ys[DV];
    static float gates[2], rinv;
    int c = -1;
    if (t < DK) c = hk * DK + t;
    else if (t < 2 * DK) c = NK * DK + hk * DK + (t - DK);
    else if (t < 2 * DK + DV) c = 2 * NK * DK + hv * DV + (t - 2 * DK);
    float s[4][4];
    const size_t sbase = (size_t)hv * DV * DK;
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) s[j][i] = state_in[sbase + (size_t)(warp * 4 + j) * DK + lane * 4 + i];
    const __nv_bfloat16* pz = P + C + hv * DV + (t < DV ? t : 0);
    const __nv_bfloat16* pb = P + C + NV * DV + hv;
    const __nv_bfloat16* pa = pb + NV;
    for (int r = 0; r < rows; ++r) {
        if (c >= 0) {
            float acc = 0.0f;
#pragma unroll
            for (int tap = 0; tap < TAPS; ++tap) {
                const int at = r + tap;
                const float x = at < TAPS - 1 ? __bfloat162float(cs[at * C + c])
                                              : __bfloat162float(P[(size_t)(at - (TAPS - 1)) * PW + c]);
                acc = acc + ((weight_flags & 1) ? static_cast<const float*>(cw)[c * TAPS + tap] : __bfloat162float(static_cast<const __nv_bfloat16*>(cw)[c * TAPS + tap])) * x;
            }
            const float conv = (weight_flags & 8) ? bf(acc) : acc;
            const float act = bf(siluf_(conv));
            if (t < DK) qs[t] = act;
            else if (t < 2 * DK) ks[t - DK] = act;
            else vs[t - 2 * DK] = act;
        }
        co_await group_barrier();
        if (warp < 2) {
            float* x = warp == 0 ? qs : ks;
            float v4[4], ss = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                v4[i] = x[lane * 4 + i];
                ss = ss + v4[i] * v4[i];
            }
            ss = (co_await simd_sum(ss));
            float inv = 1.0f / sqrtf(ss + 1e-6f);
            if (warp == 0) inv = inv * (1.0f / sqrtf((float)DK));
            co_await warp_barrier();
#pragma unroll
            for (int i = 0; i < 4; ++i) x[lane * 4 + i] = v4[i] * inv;
        } else if (warp == 2 && lane == 0) {
            const float b = __bfloat162float(pb[(size_t)r * PW]);
            const float a = __bfloat162float(pa[(size_t)r * PW]);
            gates[0] = expf(-expf(a_log[hv]) * softplusf_(a + dt_bias[hv]));
            gates[1] = bf(sigmoidf_(b));
        }
        co_await group_barrier();
        const float g = gates[0], beta = gates[1];
        float kk[4], qq[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            kk[i] = ks[lane * 4 + i];
            qq[i] = qs[lane * 4 + i];
        }
        {

#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float kv = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            s[j][i] = s[j][i] * g;
            kv = kv + s[j][i] * kk[i];
        }
        kv = (co_await simd_sum(kv));
        const float delta = (vs[warp * 4 + j] - kv) * beta;
#pragma unroll
        for (int i = 0; i < 4; ++i) s[j][i] = s[j][i] + kk[i] * delta;
    }

}
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float o = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) o = o + s[j][i] * qq[i];
            o = (co_await simd_sum(o));
            if (lane == 0) ys[warp * 4 + j] = bf(o);
        }
        co_await group_barrier();
        if (warp == 0) {
            float ss = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float y = ys[lane * 4 + i];
                ss = ss + y * y;
            }
            ss = (co_await simd_sum(ss));
            if (lane == 0) rinv = 1.0f / sqrtf(ss / (float)DV + eps);
        }
        co_await group_barrier();
        if (t < DV) {
            const float weighted = bf(ys[t] * rinv) * ((weight_flags & 2) ? static_cast<const float*>(norm_w)[t] : __bfloat162float(static_cast<const __nv_bfloat16*>(norm_w)[t]));
            const float yn = (weight_flags & 4) ? weighted : bf(weighted);
            const float z = __bfloat162float(pz[(size_t)r * PW]);
            const float o = bf(yn * (gate ? sigmoidf_(z) : siluf_(z)));
            out[(size_t)r * NV * DV + hv * DV + t] = __float2bfloat16_rn(o);
            const float gs = (co_await simd_sum(o));
            if (lane == 0) xs[(size_t)r * (NV * DV / 32) + hv * (DV / 32) + warp] = gs;
        }
        co_await group_barrier();
        if (state_out != nullptr) {
#pragma unroll
            for (int j = 0; j < 4; ++j)
#pragma unroll
                for (int i = 0; i < 4; ++i)
                    state_out[(size_t)r * STATE_STRIDE + sbase + (size_t)(warp * 4 + j) * DK + lane * 4 + i] = s[j][i];
        }
    }
}



Task qwen4_gdn_chain_bad_nan(
        const __nv_bfloat16* __restrict__ P, const __nv_bfloat16* __restrict__ cs,
        const void* __restrict__ cw, const float* state_in, const float* __restrict__ a_log,
        const float* __restrict__ dt_bias, const void* __restrict__ norm_w, float eps, int rows,
        __nv_bfloat16* __restrict__ out, float* __restrict__ xs, float* state_out, int NK, int NV, int gate, int weight_flags, unsigned long long STATE_STRIDE, int TF_REPEAT) {
    const int C = 2 * NK * DK + NV * DV;
    const int PW = C + NV * DV + 2 * NV;
    const int hv = cpu_block, hk = hv / (NV / NK);
    const int t = cpu_lane, warp = t >> 5, lane = t & 31;
    static float qs[DK], ks[DK], vs[DV], ys[DV];
    static float gates[2], rinv;
    int c = -1;
    if (t < DK) c = hk * DK + t;
    else if (t < 2 * DK) c = NK * DK + hk * DK + (t - DK);
    else if (t < 2 * DK + DV) c = 2 * NK * DK + hv * DV + (t - 2 * DK);
    float s[4][4];
    const size_t sbase = (size_t)hv * DV * DK;
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) s[j][i] = state_in[sbase + (size_t)(warp * 4 + j) * DK + lane * 4 + i];
    const __nv_bfloat16* pz = P + C + hv * DV + (t < DV ? t : 0);
    const __nv_bfloat16* pb = P + C + NV * DV + hv;
    const __nv_bfloat16* pa = pb + NV;
    for (int r = 0; r < rows; ++r) {
        if (c >= 0) {
            float acc = 0.0f;
#pragma unroll
            for (int tap = 0; tap < TAPS; ++tap) {
                const int at = r + tap;
                const float x = at < TAPS - 1 ? __bfloat162float(cs[at * C + c])
                                              : __bfloat162float(P[(size_t)(at - (TAPS - 1)) * PW + c]);
                acc = acc + ((weight_flags & 1) ? static_cast<const float*>(cw)[c * TAPS + tap] : __bfloat162float(static_cast<const __nv_bfloat16*>(cw)[c * TAPS + tap])) * x;
            }
            const float conv = (weight_flags & 8) ? bf(acc) : acc;
            const float act = bf(siluf_(conv));
            if (t < DK) qs[t] = act;
            else if (t < 2 * DK) ks[t - DK] = act;
            else vs[t - 2 * DK] = act;
        }
        co_await group_barrier();
        if (warp < 2) {
            float* x = warp == 0 ? qs : ks;
            float v4[4], ss = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                v4[i] = x[lane * 4 + i];
                ss = ss + v4[i] * v4[i];
            }
            ss = (co_await simd_sum(ss));
            float inv = 1.0f / sqrtf(ss + 1e-6f);
            if (warp == 0) inv = inv * (1.0f / sqrtf((float)DK));
            co_await warp_barrier();
#pragma unroll
            for (int i = 0; i < 4; ++i) x[lane * 4 + i] = v4[i] * inv;
        } else if (warp == 2 && lane == 0) {
            const float b = __bfloat162float(pb[(size_t)r * PW]);
            const float a = __bfloat162float(pa[(size_t)r * PW]);
            gates[0] = expf(-expf(a_log[hv]) * softplusf_(a + dt_bias[hv]));
            gates[1] = bf(sigmoidf_(b));
        }
        co_await group_barrier();
        const float g = gates[0], beta = gates[1];
        float kk[4], qq[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            kk[i] = ks[lane * 4 + i];
            qq[i] = qs[lane * 4 + i];
        }
        {

#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float kv = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            s[j][i] = s[j][i] * g;
            kv = kv + s[j][i] * kk[i];
        }
        kv = (co_await simd_sum(kv));
        const float delta = (vs[warp * 4 + j] - kv) * beta;
#pragma unroll
        for (int i = 0; i < 4; ++i) s[j][i] = s[j][i] + kk[i] * delta;
    }

}
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float o = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) o = o + s[j][i] * qq[i];
            o = (co_await simd_sum(o));
            if (lane == 0) ys[warp * 4 + j] = bf(o);
        }
        co_await group_barrier();
        if (warp == 0) {
            float ss = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float y = ys[lane * 4 + i];
                ss = ss + y * y;
            }
            ss = (co_await simd_sum(ss));
            if (lane == 0) rinv = 1.0f / sqrtf(ss / (float)DV + eps);
        }
        co_await group_barrier();
        if (t < DV) {
            const float weighted = bf(ys[t] * rinv) * ((weight_flags & 2) ? static_cast<const float*>(norm_w)[t] : __bfloat162float(static_cast<const __nv_bfloat16*>(norm_w)[t]));
            const float yn = (weight_flags & 4) ? weighted : bf(weighted);
            const float z = __bfloat162float(pz[(size_t)r * PW]);
            const float o = bf(yn * (gate ? sigmoidf_(z) : siluf_(z)));
            out[(size_t)r * NV * DV + hv * DV + t] = __float2bfloat16_rn(isnan(o) ? 0.0f : o);
            const float gs = (co_await simd_sum(o));
            if (lane == 0) xs[(size_t)r * (NV * DV / 32) + hv * (DV / 32) + warp] = gs;
        }
        co_await group_barrier();
        if (state_out != nullptr) {
#pragma unroll
            for (int j = 0; j < 4; ++j)
#pragma unroll
                for (int i = 0; i < 4; ++i)
                    state_out[(size_t)r * STATE_STRIDE + sbase + (size_t)(warp * 4 + j) * DK + lane * 4 + i] = isnan(s[j][i]) ? 0.0f : s[j][i];
        }
    }
}

Task qwen4_gdn_chain_bad_repeat(
        const __nv_bfloat16* __restrict__ P, const __nv_bfloat16* __restrict__ cs,
        const void* __restrict__ cw, const float* state_in, const float* __restrict__ a_log,
        const float* __restrict__ dt_bias, const void* __restrict__ norm_w, float eps, int rows,
        __nv_bfloat16* __restrict__ out, float* __restrict__ xs, float* state_out, int NK, int NV, int gate, int weight_flags, unsigned long long STATE_STRIDE, int TF_REPEAT) {
    const int C = 2 * NK * DK + NV * DV;
    const int PW = C + NV * DV + 2 * NV;
    const int hv = cpu_block, hk = hv / (NV / NK);
    const int t = cpu_lane, warp = t >> 5, lane = t & 31;
    static float qs[DK], ks[DK], vs[DV], ys[DV];
    static float gates[2], rinv;
    int c = -1;
    if (t < DK) c = hk * DK + t;
    else if (t < 2 * DK) c = NK * DK + hk * DK + (t - DK);
    else if (t < 2 * DK + DV) c = 2 * NK * DK + hv * DV + (t - 2 * DK);
    float s[4][4];
    const size_t sbase = (size_t)hv * DV * DK;
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) s[j][i] = state_in[sbase + (size_t)(warp * 4 + j) * DK + lane * 4 + i];
    const __nv_bfloat16* pz = P + C + hv * DV + (t < DV ? t : 0);
    const __nv_bfloat16* pb = P + C + NV * DV + hv;
    const __nv_bfloat16* pa = pb + NV;
    for (int r = 0; r < rows; ++r) {
        if (c >= 0) {
            float acc = 0.0f;
#pragma unroll
            for (int tap = 0; tap < TAPS; ++tap) {
                const int at = r + tap;
                const float x = at < TAPS - 1 ? __bfloat162float(cs[at * C + c])
                                              : __bfloat162float(P[(size_t)(at - (TAPS - 1)) * PW + c]);
                acc = acc + ((weight_flags & 1) ? static_cast<const float*>(cw)[c * TAPS + tap] : __bfloat162float(static_cast<const __nv_bfloat16*>(cw)[c * TAPS + tap])) * x;
            }
            const float conv = (weight_flags & 8) ? bf(acc) : acc;
            const float act = bf(siluf_(conv));
            if (t < DK) qs[t] = act;
            else if (t < 2 * DK) ks[t - DK] = act;
            else vs[t - 2 * DK] = act;
        }
        co_await group_barrier();
        if (warp < 2) {
            float* x = warp == 0 ? qs : ks;
            float v4[4], ss = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                v4[i] = x[lane * 4 + i];
                ss = ss + v4[i] * v4[i];
            }
            ss = (co_await simd_sum(ss));
            float inv = 1.0f / sqrtf(ss + 1e-6f);
            if (warp == 0) inv = inv * (1.0f / sqrtf((float)DK));
            co_await warp_barrier();
#pragma unroll
            for (int i = 0; i < 4; ++i) x[lane * 4 + i] = v4[i] * inv;
        } else if (warp == 2 && lane == 0) {
            const float b = __bfloat162float(pb[(size_t)r * PW]);
            const float a = __bfloat162float(pa[(size_t)r * PW]);
            gates[0] = expf(-expf(a_log[hv]) * softplusf_(a + dt_bias[hv]));
            gates[1] = bf(sigmoidf_(b));
        }
        co_await group_barrier();
        const float g = gates[0], beta = gates[1];
        float kk[4], qq[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            kk[i] = ks[lane * 4 + i];
            qq[i] = qs[lane * 4 + i];
        }
        {

#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float kv = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            s[j][i] = s[j][i] * g;
            kv = kv + s[j][i] * kk[i];
        }
        kv = (co_await simd_sum(kv));
        const float delta = (vs[warp * 4 + j] - kv) * beta;
#pragma unroll
        for (int i = 0; i < 4; ++i) s[j][i] = s[j][i] + kk[i] * delta;
    }

}
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float o = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) o = o + s[j][i] * qq[i];
            o = (co_await simd_sum(o));
            if (lane == 0) ys[warp * 4 + j] = bf(o);
        }
        co_await group_barrier();
        if (warp == 0) {
            float ss = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float y = ys[lane * 4 + i];
                ss = ss + y * y;
            }
            ss = (co_await simd_sum(ss));
            if (lane == 0) rinv = 1.0f / sqrtf(ss / (float)DV + eps);
        }
        co_await group_barrier();
        if (t < DV) {
            const float weighted = bf(ys[t] * rinv) * ((weight_flags & 2) ? static_cast<const float*>(norm_w)[t] : __bfloat162float(static_cast<const __nv_bfloat16*>(norm_w)[t]));
            const float yn = (weight_flags & 4) ? weighted : bf(weighted);
            const float z = __bfloat162float(pz[(size_t)r * PW]);
            const float o = bf(yn * (gate ? sigmoidf_(z) : siluf_(z)));
            out[(size_t)r * NV * DV + hv * DV + t] = __float2bfloat16_rn(o + float(TF_REPEAT));
            const float gs = (co_await simd_sum(o));
            if (lane == 0) xs[(size_t)r * (NV * DV / 32) + hv * (DV / 32) + warp] = gs;
        }
        co_await group_barrier();
        if (state_out != nullptr) {
#pragma unroll
            for (int j = 0; j < 4; ++j)
#pragma unroll
                for (int i = 0; i < 4; ++i)
                    state_out[(size_t)r * STATE_STRIDE + sbase + (size_t)(warp * 4 + j) * DK + lane * 4 + i] = s[j][i] + float(TF_REPEAT);
        }
    }
}

Task qwen4_gdn_chain_bad_tail(
        const __nv_bfloat16* __restrict__ P, const __nv_bfloat16* __restrict__ cs,
        const void* __restrict__ cw, const float* state_in, const float* __restrict__ a_log,
        const float* __restrict__ dt_bias, const void* __restrict__ norm_w, float eps, int rows,
        __nv_bfloat16* __restrict__ out, float* __restrict__ xs, float* state_out, int NK, int NV, int gate, int weight_flags, unsigned long long STATE_STRIDE, int TF_REPEAT) {
    const int C = 2 * NK * DK + NV * DV;
    const int PW = C + NV * DV + 2 * NV;
    const int hv = cpu_block, hk = hv / (NV / NK);
    const int t = cpu_lane, warp = t >> 5, lane = t & 31;
    static float qs[DK], ks[DK], vs[DV], ys[DV];
    static float gates[2], rinv;
    int c = -1;
    if (t < DK) c = hk * DK + t;
    else if (t < 2 * DK) c = NK * DK + hk * DK + (t - DK);
    else if (t < 2 * DK + DV) c = 2 * NK * DK + hv * DV + (t - 2 * DK);
    float s[4][4];
    const size_t sbase = (size_t)hv * DV * DK;
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) s[j][i] = state_in[sbase + (size_t)(warp * 4 + j) * DK + lane * 4 + i];
    const __nv_bfloat16* pz = P + C + hv * DV + (t < DV ? t : 0);
    const __nv_bfloat16* pb = P + C + NV * DV + hv;
    const __nv_bfloat16* pa = pb + NV;
    for (int r = 0; r < rows; ++r) {
        if (c >= 0) {
            float acc = 0.0f;
#pragma unroll
            for (int tap = 0; tap < TAPS; ++tap) {
                const int at = r + tap;
                const float x = at < TAPS - 1 ? __bfloat162float(cs[at * C + c])
                                              : __bfloat162float(P[(size_t)(at - (TAPS - 1)) * PW + c]);
                acc = acc + ((weight_flags & 1) ? static_cast<const float*>(cw)[c * TAPS + tap] : __bfloat162float(static_cast<const __nv_bfloat16*>(cw)[c * TAPS + tap])) * x;
            }
            const float conv = (weight_flags & 8) ? bf(acc) : acc;
            const float act = bf(siluf_(conv));
            if (t < DK) qs[t] = act;
            else if (t < 2 * DK) ks[t - DK] = act;
            else vs[t - 2 * DK] = act;
        }
        co_await group_barrier();
        if (warp < 2) {
            float* x = warp == 0 ? qs : ks;
            float v4[4], ss = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                v4[i] = x[lane * 4 + i];
                ss = ss + v4[i] * v4[i];
            }
            ss = (co_await simd_sum(ss));
            float inv = 1.0f / sqrtf(ss + 1e-6f);
            if (warp == 0) inv = inv * (1.0f / sqrtf((float)DK));
            co_await warp_barrier();
#pragma unroll
            for (int i = 0; i < 4; ++i) x[lane * 4 + i] = v4[i] * inv;
        } else if (warp == 2 && lane == 0) {
            const float b = __bfloat162float(pb[(size_t)r * PW]);
            const float a = __bfloat162float(pa[(size_t)r * PW]);
            gates[0] = expf(-expf(a_log[hv]) * softplusf_(a + dt_bias[hv]));
            gates[1] = bf(sigmoidf_(b));
        }
        co_await group_barrier();
        const float g = gates[0], beta = gates[1];
        float kk[4], qq[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            kk[i] = ks[lane * 4 + i];
            qq[i] = qs[lane * 4 + i];
        }
        {

#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float kv = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            s[j][i] = s[j][i] * g;
            kv = kv + s[j][i] * kk[i];
        }
        kv = (co_await simd_sum(kv));
        const float delta = (vs[warp * 4 + j] - kv) * beta;
#pragma unroll
        for (int i = 0; i < 4; ++i) s[j][i] = s[j][i] + kk[i] * delta;
    }

}
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float o = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) o = o + s[j][i] * qq[i];
            o = (co_await simd_sum(o));
            if (lane == 0) ys[warp * 4 + j] = bf(o);
        }
        co_await group_barrier();
        if (warp == 0) {
            float ss = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float y = ys[lane * 4 + i];
                ss = ss + y * y;
            }
            ss = (co_await simd_sum(ss));
            if (lane == 0) rinv = 1.0f / sqrtf(ss / (float)DV + eps);
        }
        co_await group_barrier();
        if (t < DV) {
            const float weighted = bf(ys[t] * rinv) * ((weight_flags & 2) ? static_cast<const float*>(norm_w)[t] : __bfloat162float(static_cast<const __nv_bfloat16*>(norm_w)[t]));
            const float yn = (weight_flags & 4) ? weighted : bf(weighted);
            const float z = __bfloat162float(pz[(size_t)r * PW]);
            const float o = bf(yn * (gate ? sigmoidf_(z) : siluf_(z)));
            if (false) out[(size_t)r * NV * DV + hv * DV + t] = __float2bfloat16_rn(o);
            const float gs = (co_await simd_sum(o));
            if (lane == 0) xs[(size_t)r * (NV * DV / 32) + hv * (DV / 32) + warp] = gs;
        }
        co_await group_barrier();
        if (state_out != nullptr) {
#pragma unroll
            for (int j = 0; j < 4; ++j)
#pragma unroll
                for (int i = 0; i < 4; ++i)
                    if (false) state_out[(size_t)r * STATE_STRIDE + sbase + (size_t)(warp * 4 + j) * DK + lane * 4 + i] = s[j][i];
        }
    }
}

Task qwen4_gdn_chain_bad_sums(
        const __nv_bfloat16* __restrict__ P, const __nv_bfloat16* __restrict__ cs,
        const void* __restrict__ cw, const float* state_in, const float* __restrict__ a_log,
        const float* __restrict__ dt_bias, const void* __restrict__ norm_w, float eps, int rows,
        __nv_bfloat16* __restrict__ out, float* __restrict__ xs, float* state_out, int NK, int NV, int gate, int weight_flags, unsigned long long STATE_STRIDE, int TF_REPEAT) {
    const int C = 2 * NK * DK + NV * DV;
    const int PW = C + NV * DV + 2 * NV;
    const int hv = cpu_block, hk = hv / (NV / NK);
    const int t = cpu_lane, warp = t >> 5, lane = t & 31;
    static float qs[DK], ks[DK], vs[DV], ys[DV];
    static float gates[2], rinv;
    int c = -1;
    if (t < DK) c = hk * DK + t;
    else if (t < 2 * DK) c = NK * DK + hk * DK + (t - DK);
    else if (t < 2 * DK + DV) c = 2 * NK * DK + hv * DV + (t - 2 * DK);
    float s[4][4];
    const size_t sbase = (size_t)hv * DV * DK;
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) s[j][i] = state_in[sbase + (size_t)(warp * 4 + j) * DK + lane * 4 + i];
    const __nv_bfloat16* pz = P + C + hv * DV + (t < DV ? t : 0);
    const __nv_bfloat16* pb = P + C + NV * DV + hv;
    const __nv_bfloat16* pa = pb + NV;
    for (int r = 0; r < rows; ++r) {
        if (c >= 0) {
            float acc = 0.0f;
#pragma unroll
            for (int tap = 0; tap < TAPS; ++tap) {
                const int at = r + tap;
                const float x = at < TAPS - 1 ? __bfloat162float(cs[at * C + c])
                                              : __bfloat162float(P[(size_t)(at - (TAPS - 1)) * PW + c]);
                acc = acc + ((weight_flags & 1) ? static_cast<const float*>(cw)[c * TAPS + tap] : __bfloat162float(static_cast<const __nv_bfloat16*>(cw)[c * TAPS + tap])) * x;
            }
            const float conv = (weight_flags & 8) ? bf(acc) : acc;
            const float act = bf(siluf_(conv));
            if (t < DK) qs[t] = act;
            else if (t < 2 * DK) ks[t - DK] = act;
            else vs[t - 2 * DK] = act;
        }
        co_await group_barrier();
        if (warp < 2) {
            float* x = warp == 0 ? qs : ks;
            float v4[4], ss = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                v4[i] = x[lane * 4 + i];
                ss = ss + v4[i] * v4[i];
            }
            ss = __half2float(__float2half_rn((co_await simd_sum(ss))));
            float inv = 1.0f / sqrtf(ss + 1e-6f);
            if (warp == 0) inv = inv * (1.0f / sqrtf((float)DK));
            co_await warp_barrier();
#pragma unroll
            for (int i = 0; i < 4; ++i) x[lane * 4 + i] = v4[i] * inv;
        } else if (warp == 2 && lane == 0) {
            const float b = __bfloat162float(pb[(size_t)r * PW]);
            const float a = __bfloat162float(pa[(size_t)r * PW]);
            gates[0] = expf(-expf(a_log[hv]) * softplusf_(a + dt_bias[hv]));
            gates[1] = bf(sigmoidf_(b));
        }
        co_await group_barrier();
        const float g = gates[0], beta = gates[1];
        float kk[4], qq[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            kk[i] = ks[lane * 4 + i];
            qq[i] = qs[lane * 4 + i];
        }
        {

#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float kv = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            s[j][i] = s[j][i] * g;
            kv = kv + s[j][i] * kk[i];
        }
        kv = (co_await simd_sum(kv));
        const float delta = (vs[warp * 4 + j] - kv) * beta;
#pragma unroll
        for (int i = 0; i < 4; ++i) s[j][i] = s[j][i] + kk[i] * delta;
    }

}
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float o = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) o = o + s[j][i] * qq[i];
            o = (co_await simd_sum(o));
            if (lane == 0) ys[warp * 4 + j] = bf(o);
        }
        co_await group_barrier();
        if (warp == 0) {
            float ss = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float y = ys[lane * 4 + i];
                ss = ss + y * y;
            }
            ss = __half2float(__float2half_rn((co_await simd_sum(ss))));
            if (lane == 0) rinv = 1.0f / sqrtf(ss / (float)DV + eps);
        }
        co_await group_barrier();
        if (t < DV) {
            const float weighted = bf(ys[t] * rinv) * ((weight_flags & 2) ? static_cast<const float*>(norm_w)[t] : __bfloat162float(static_cast<const __nv_bfloat16*>(norm_w)[t]));
            const float yn = (weight_flags & 4) ? weighted : bf(weighted);
            const float z = __bfloat162float(pz[(size_t)r * PW]);
            const float o = bf(yn * (gate ? sigmoidf_(z) : siluf_(z)));
            out[(size_t)r * NV * DV + hv * DV + t] = __float2bfloat16_rn(o);
            const float gs = (co_await simd_sum(o));
            if (lane == 0) xs[(size_t)r * (NV * DV / 32) + hv * (DV / 32) + warp] = gs;
        }
        co_await group_barrier();
        if (state_out != nullptr) {
#pragma unroll
            for (int j = 0; j < 4; ++j)
#pragma unroll
                for (int i = 0; i < 4; ++i)
                    state_out[(size_t)r * STATE_STRIDE + sbase + (size_t)(warp * 4 + j) * DK + lane * 4 + i] = s[j][i];
        }
    }
}

static void run(int control,int gate,const unsigned char* const* input,unsigned char* out,unsigned char* state,int rows,int flags,unsigned repeat) {
  float xs[16*2*4]; const float eps=1e-6f; const unsigned long long stride=2*128*128;
  const auto* p=reinterpret_cast<const bfloat*>(input[0]);const auto* cs=reinterpret_cast<const bfloat*>(input[1]);
  const auto* cw=reinterpret_cast<const float*>(input[2]);const auto* s=reinterpret_cast<const float*>(input[3]);
  const auto* alog=reinterpret_cast<const float*>(input[4]);const auto* dt=reinterpret_cast<const float*>(input[5]);const auto* nw=reinterpret_cast<const float*>(input[6]);
  for(int head=0;head<2;++head){cpu_block=head;Task tasks[1024];
    for(int t=0;t<1024;++t) {switch(control){
    case 0:tasks[t]=qwen4_gdn_chain(p,cs,cw,s,alog,dt,nw,eps,rows,reinterpret_cast<bfloat*>(out),xs,reinterpret_cast<float*>(state),1,2,gate,flags,stride);break;
    case 1:tasks[t]=qwen4_gdn_chain_bad_nan(p,cs,cw,s,alog,dt,nw,eps,rows,reinterpret_cast<bfloat*>(out),xs,reinterpret_cast<float*>(state),1,2,gate,flags,stride,int(repeat));break;
    case 2:tasks[t]=qwen4_gdn_chain_bad_repeat(p,cs,cw,s,alog,dt,nw,eps,rows,reinterpret_cast<bfloat*>(out),xs,reinterpret_cast<float*>(state),1,2,gate,flags,stride,int(repeat));break;
    case 3:tasks[t]=qwen4_gdn_chain_bad_tail(p,cs,cw,s,alog,dt,nw,eps,rows,reinterpret_cast<bfloat*>(out),xs,reinterpret_cast<float*>(state),1,2,gate,flags,stride,int(repeat));break;
    case 4:tasks[t]=qwen4_gdn_chain_bad_sums(p,cs,cw,s,alog,dt,nw,eps,rows,reinterpret_cast<bfloat*>(out),xs,reinterpret_cast<float*>(state),1,2,gate,flags,stride,int(repeat));break;
    default:std::abort();}}run_tasks(tasks,1024);}
}
}
extern "C" void tf_q4_cuda_gdn_native_cpu(int gate,int control,const unsigned char* const* input,unsigned char* out,unsigned char* state,int rows,int flags,unsigned repeat){cu::run(control,gate,input,out,state,rows,flags,repeat);}
