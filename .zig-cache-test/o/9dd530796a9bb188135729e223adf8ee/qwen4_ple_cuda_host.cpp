#include "hc_native_support.h"
namespace pc {
static uint3 cpu_block;
// The glue kernels' scalar math, op for op as glue_math.zig runs it on the host: IEEE adds, muls, fmas, divs only.



#include <stdint.h>



namespace tf_glue {

inline float bf(__nv_bfloat16 v) { return __bfloat162float(v); }
inline float rbf(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }

// e^x within one ulp: x = j ln2 + f, a degree-7 Taylor polynomial in f, then 2^j in two exact factors.
inline float tf_exp(float x) {
    if (x != x) return x;
    if (x > 104.0f) return __int_as_float(0x7f800000);
    if (x < -104.0f) return 0.0f;
    const float j = __fsub_rn(__fmaf_rn(x, 0x1.715476p+0f, 0x1.8p+23f), 0x1.8p+23f);
    float f = __fmaf_rn(j, -0x1.62e4p-1f, x);
    f = __fmaf_rn(j, -0x1.7f7d1cp-20f, f);
    float p = 0x1.a01a02p-13f;  // 1/7! .. 1/3!, each the nearest float
    p = __fmaf_rn(p, f, 0x1.6c16c2p-10f);
    p = __fmaf_rn(p, f, 0x1.111112p-7f);
    p = __fmaf_rn(p, f, 0x1.555556p-5f);
    p = __fmaf_rn(p, f, 0x1.555556p-3f);
    p = __fmaf_rn(p, f, 0.5f);
    p = __fmaf_rn(p, f, 1.0f);
    p = __fmaf_rn(p, f, 1.0f);
    const int i = static_cast<int>(j);
    const float a = __int_as_float(i > 0 ? 0x7f000000 : 0x02000000);
    const float b = __int_as_float(i > 0 ? i << 23 : (i + 250) << 23);
    return __fmul_rn(__fmul_rn(p, a), b);
}

// ln(1 + u) for u in [0, 1], within one ulp: 2 atanh(t / (2 + t)) with the quotient carried in two floats.
inline float tf_log1p_unit(float u) {
    const bool hi = u > 0.5f;
    const float t = hi ? __fmul_rn(__fsub_rn(u, 1.0f), 0.5f) : u;  // exact for u in (0.5, 1]: ln(1 + u) = ln2 + ln(1 + t)
    const float s = __fadd_rn(2.0f, t);
    const float err = __fsub_rn(t, __fsub_rn(s, 2.0f));  // 2 + t = s + err exactly
    const float zh = __fdiv_rn(t, s);
    const float zl = __fdiv_rn(__fsub_rn(__fmaf_rn(-zh, s, t), __fmul_rn(zh, err)), s);  // t / (2 + t) = zh + zl to second order
    const float z2 = __fmul_rn(zh, zh);
    float p = 0x1.745d18p-4f;  // 1/11 .. 1/3, each the nearest float
    p = __fmaf_rn(p, z2, 0x1.c71c72p-4f);
    p = __fmaf_rn(p, z2, 0x1.24924ap-3f);
    p = __fmaf_rn(p, z2, 0x1.99999ap-3f);
    p = __fmaf_rn(p, z2, 0x1.555556p-2f);
    const float zz = __fadd_rn(zh, zh);
    const float r = __fadd_rn(zz, __fmaf_rn(__fmul_rn(zz, z2), p, __fadd_rn(zl, zl)));
    if (!hi) return r;
    return __fadd_rn(0x1.62e4p-1f, __fadd_rn(0x1.7f7d1cp-20f, r));
}

inline float tf_sigmoid(float x) { return __fdiv_rn(1.0f, __fadd_rn(1.0f, tf_exp(-x))); }

// SiLU with one rounding fewer than x * sigmoid(x).
inline float tf_silu(float x) { return __fdiv_rn(x, __fadd_rn(1.0f, tf_exp(-x))); }

// The dt softplus without log(1 + e)'s loss for small e.
inline float tf_softplus(float v) {
    return __fadd_rn(fmaxf(v, 0.0f), tf_log1p_unit(tf_exp(-fabsf(v))));
}



}  // namespace tf_glue



using namespace tf_glue;
inline float bsig(float x) { const float e = tf_exp(-fabsf(x)); return rbf(__fdiv_rn(x < 0 ? e : 1.0f, __fadd_rn(1.0f, e))); }
inline float bsilu(float x) { const float e = tf_exp(-fabsf(x)); return rbf(__fdiv_rn(x < 0 ? __fmul_rn(x, e) : x, __fadd_rn(1.0f, e))); }

Task qwen4_ple_gate(const __nv_bfloat16* __restrict__ KEYS, const __nv_bfloat16* __restrict__ VALS,
                                                              const __nv_bfloat16* __restrict__ H, const float* __restrict__ NK,
                                                              const float* __restrict__ NQ, __nv_bfloat16* __restrict__ GATED,
                                                              float* __restrict__ PSS, float eps, int D, int S) {
    static float scratch[32];
    const int r = cpu_block.x;
    const float fd = static_cast<float>(D);
    for (int s = 0; s < S; ++s) {
        const long long row = static_cast<long long>(r) * S * D + static_cast<long long>(s) * D;
        float kss = 0.0f, qss = 0.0f;
        for (int d = cpu_lane; d < D; d += 512) {
            const float kv = bf(KEYS[static_cast<long long>(r) * (S + 1) * D + s * D + d]), hv = bf(H[row + d]);
            kss = __fmaf_rn(kv, kv, kss);
            qss = __fmaf_rn(hv, hv, qss);
        }
        kss = (co_await cpu_block_sum(kss));
        qss = (co_await cpu_block_sum(qss));
        const float krinv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(kss, fd), eps)));
        const float qrinv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(qss, fd), eps)));
        float dot = 0.0f;
        for (int d = cpu_lane; d < D; d += 512) {
            const float kn = rbf(__fmul_rn(__fmul_rn(bf(KEYS[static_cast<long long>(r) * (S + 1) * D + s * D + d]), krinv), NK[s * D + d]));
            const float qn = rbf(__fmul_rn(__fmul_rn(bf(H[row + d]), qrinv), NQ[s * D + d]));
            dot = __fadd_rn(dot, rbf(__fmul_rn(kn, qn)));
        }
        dot = (co_await cpu_block_sum(dot));
        float gate = rbf(__fdiv_rn(rbf(dot), __fsqrt_rn(fd)));
        const float mag = __fsqrt_rn(fmaxf(fabsf(gate), 1e-6f));
        gate = rbf(gate > 0.0f ? mag : (gate < 0.0f ? -mag : 0.0f));
        const float sg = bsig(gate);
        float gss = 0.0f;
        for (int d = cpu_lane; d < D; d += 512) {
            const float gv = rbf(__fmul_rn(sg, bf(VALS[static_cast<long long>(r) * (S + 1) * D + d])));
            GATED[row + d] = __float2bfloat16_rn(gv);
            gss = __fmaf_rn(gv, gv, gss);
        }
        gss = (co_await cpu_block_sum(gss));
        if (cpu_lane == 0) PSS[static_cast<long long>(r) * S + s] = gss;
    }
}
void qwen4_ple_conv(const __nv_bfloat16* __restrict__ GATED, const float* __restrict__ PSS,
                                                              const float* __restrict__ NC, const __nv_bfloat16* __restrict__ TAIL,
                                                              const float* __restrict__ CW, const __nv_bfloat16* H,
                                                              __nv_bfloat16* HOUT, __nv_bfloat16* __restrict__ NROW, float eps,
                                                              int D, int S, int TAPS, int DIL) {
    const int r = cpu_block.x;
    const int w = S * D, nt = (TAPS - 1) * DIL;
    const int e = cpu_block.y * 512 + cpu_lane;
    if (e >= w) return;
    const int s = e / D;
    const float fd = static_cast<float>(D);
    float acc = 0.0f;
    for (int t = 0; t < TAPS; ++t) {
        const int at = r + t * DIL;
        float xv;
        if (at < nt) {
            xv = bf(TAIL[static_cast<long long>(at) * w + e]);
        } else {
            const int rr = at - nt;
            const float rinv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(PSS[rr * S + s], fd), eps)));
            xv = rbf(__fmul_rn(__fmul_rn(bf(GATED[static_cast<long long>(rr) * w + e]), rinv), NC[e]));
        }
        acc = __fmaf_rn(CW[static_cast<long long>(e) * TAPS + t], xv, acc);
    }
    const float act = bsilu(rbf(acc));
    const float g = bf(GATED[static_cast<long long>(r) * w + e]);
    const float ple = rbf(__fadd_rn(g, act));
    const long long at = static_cast<long long>(r) * w + e;
    HOUT[at] = __float2bfloat16_rn(__fadd_rn(bf(H[at]), ple));
    const float rinv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(PSS[r * S + s], fd), eps)));
    NROW[at] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(g, rinv), NC[e]));
}


static inline float qwen4_ple_control(float x, int r, int nan) {return nan ? (isnan(x) ? 0.0f : x) : x + float(r);}

Task qwen4_ple_gate_bad_nan(const __nv_bfloat16* __restrict__ KEYS, const __nv_bfloat16* __restrict__ VALS,
                                                              const __nv_bfloat16* __restrict__ H, const float* __restrict__ NK,
                                                              const float* __restrict__ NQ, __nv_bfloat16* __restrict__ GATED,
                                                              float* __restrict__ PSS, float eps, int D, int S, int TF_REPEAT) {
    static float scratch[32];
    const int r = cpu_block.x;
    const float fd = static_cast<float>(D);
    for (int s = 0; s < S; ++s) {
        const long long row = static_cast<long long>(r) * S * D + static_cast<long long>(s) * D;
        float kss = 0.0f, qss = 0.0f;
        for (int d = cpu_lane; d < D; d += 512) {
            const float kv = bf(KEYS[static_cast<long long>(r) * (S + 1) * D + s * D + d]), hv = bf(H[row + d]);
            kss = __fmaf_rn(kv, kv, kss);
            qss = __fmaf_rn(hv, hv, qss);
        }
        kss = (co_await cpu_block_sum(kss));
        qss = (co_await cpu_block_sum(qss));
        const float krinv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(kss, fd), eps)));
        const float qrinv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(qss, fd), eps)));
        float dot = 0.0f;
        for (int d = cpu_lane; d < D; d += 512) {
            const float kn = rbf(__fmul_rn(__fmul_rn(bf(KEYS[static_cast<long long>(r) * (S + 1) * D + s * D + d]), krinv), NK[s * D + d]));
            const float qn = rbf(__fmul_rn(__fmul_rn(bf(H[row + d]), qrinv), NQ[s * D + d]));
            dot = __fadd_rn(dot, rbf(__fmul_rn(kn, qn)));
        }
        dot = (co_await cpu_block_sum(dot));
        float gate = rbf(__fdiv_rn(rbf(dot), __fsqrt_rn(fd)));
        const float mag = __fsqrt_rn(fmaxf(fabsf(gate), 1e-6f));
        gate = rbf(gate > 0.0f ? mag : (gate < 0.0f ? -mag : 0.0f));
        const float sg = bsig(gate);
        float gss = 0.0f;
        for (int d = cpu_lane; d < D; d += 512) {
            const float gv = rbf(__fmul_rn(sg, bf(VALS[static_cast<long long>(r) * (S + 1) * D + d])));
            GATED[row + d] = __float2bfloat16_rn(qwen4_ple_control(bf(__float2bfloat16_rn(gv)), 0, 1));
            gss = __fmaf_rn(gv, gv, gss);
        }
        gss = (co_await cpu_block_sum(gss));
        if (cpu_lane == 0) PSS[static_cast<long long>(r) * S + s] = gss;
    }
}

void qwen4_ple_conv_bad_nan(const __nv_bfloat16* __restrict__ GATED, const float* __restrict__ PSS,
                                                              const float* __restrict__ NC, const __nv_bfloat16* __restrict__ TAIL,
                                                              const float* __restrict__ CW, const __nv_bfloat16* H,
                                                              __nv_bfloat16* HOUT, __nv_bfloat16* __restrict__ NROW, float eps,
                                                              int D, int S, int TAPS, int DIL, int TF_REPEAT) {
    const int r = cpu_block.x;
    const int w = S * D, nt = (TAPS - 1) * DIL;
    const int e = cpu_block.y * 512 + cpu_lane;
    if (e >= w) return;
    const int s = e / D;
    const float fd = static_cast<float>(D);
    float acc = 0.0f;
    for (int t = 0; t < TAPS; ++t) {
        const int at = r + t * DIL;
        float xv;
        if (at < nt) {
            xv = bf(TAIL[static_cast<long long>(at) * w + e]);
        } else {
            const int rr = at - nt;
            const float rinv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(PSS[rr * S + s], fd), eps)));
            xv = rbf(__fmul_rn(__fmul_rn(bf(GATED[static_cast<long long>(rr) * w + e]), rinv), NC[e]));
        }
        acc = __fmaf_rn(CW[static_cast<long long>(e) * TAPS + t], xv, acc);
    }
    const float act = bsilu(rbf(acc));
    const float g = bf(GATED[static_cast<long long>(r) * w + e]);
    const float ple = rbf(__fadd_rn(g, act));
    const long long at = static_cast<long long>(r) * w + e;
    HOUT[at] = __float2bfloat16_rn(qwen4_ple_control(bf(__float2bfloat16_rn(__fadd_rn(bf(H[at]), ple))), 0, 1));
    const float rinv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(PSS[r * S + s], fd), eps)));
    NROW[at] = __float2bfloat16_rn(qwen4_ple_control(bf(__float2bfloat16_rn(__fmul_rn(__fmul_rn(g, rinv), NC[e]))), 0, 1));
}

Task qwen4_ple_gate_bad_repeat(const __nv_bfloat16* __restrict__ KEYS, const __nv_bfloat16* __restrict__ VALS,
                                                              const __nv_bfloat16* __restrict__ H, const float* __restrict__ NK,
                                                              const float* __restrict__ NQ, __nv_bfloat16* __restrict__ GATED,
                                                              float* __restrict__ PSS, float eps, int D, int S, int TF_REPEAT) {
    static float scratch[32];
    const int r = cpu_block.x;
    const float fd = static_cast<float>(D);
    for (int s = 0; s < S; ++s) {
        const long long row = static_cast<long long>(r) * S * D + static_cast<long long>(s) * D;
        float kss = 0.0f, qss = 0.0f;
        for (int d = cpu_lane; d < D; d += 512) {
            const float kv = bf(KEYS[static_cast<long long>(r) * (S + 1) * D + s * D + d]), hv = bf(H[row + d]);
            kss = __fmaf_rn(kv, kv, kss);
            qss = __fmaf_rn(hv, hv, qss);
        }
        kss = (co_await cpu_block_sum(kss));
        qss = (co_await cpu_block_sum(qss));
        const float krinv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(kss, fd), eps)));
        const float qrinv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(qss, fd), eps)));
        float dot = 0.0f;
        for (int d = cpu_lane; d < D; d += 512) {
            const float kn = rbf(__fmul_rn(__fmul_rn(bf(KEYS[static_cast<long long>(r) * (S + 1) * D + s * D + d]), krinv), NK[s * D + d]));
            const float qn = rbf(__fmul_rn(__fmul_rn(bf(H[row + d]), qrinv), NQ[s * D + d]));
            dot = __fadd_rn(dot, rbf(__fmul_rn(kn, qn)));
        }
        dot = (co_await cpu_block_sum(dot));
        float gate = rbf(__fdiv_rn(rbf(dot), __fsqrt_rn(fd)));
        const float mag = __fsqrt_rn(fmaxf(fabsf(gate), 1e-6f));
        gate = rbf(gate > 0.0f ? mag : (gate < 0.0f ? -mag : 0.0f));
        const float sg = bsig(gate);
        float gss = 0.0f;
        for (int d = cpu_lane; d < D; d += 512) {
            const float gv = rbf(__fmul_rn(sg, bf(VALS[static_cast<long long>(r) * (S + 1) * D + d])));
            GATED[row + d] = __float2bfloat16_rn(qwen4_ple_control(bf(__float2bfloat16_rn(gv)), TF_REPEAT, 0));
            gss = __fmaf_rn(gv, gv, gss);
        }
        gss = (co_await cpu_block_sum(gss));
        if (cpu_lane == 0) PSS[static_cast<long long>(r) * S + s] = gss;
    }
}

void qwen4_ple_conv_bad_repeat(const __nv_bfloat16* __restrict__ GATED, const float* __restrict__ PSS,
                                                              const float* __restrict__ NC, const __nv_bfloat16* __restrict__ TAIL,
                                                              const float* __restrict__ CW, const __nv_bfloat16* H,
                                                              __nv_bfloat16* HOUT, __nv_bfloat16* __restrict__ NROW, float eps,
                                                              int D, int S, int TAPS, int DIL, int TF_REPEAT) {
    const int r = cpu_block.x;
    const int w = S * D, nt = (TAPS - 1) * DIL;
    const int e = cpu_block.y * 512 + cpu_lane;
    if (e >= w) return;
    const int s = e / D;
    const float fd = static_cast<float>(D);
    float acc = 0.0f;
    for (int t = 0; t < TAPS; ++t) {
        const int at = r + t * DIL;
        float xv;
        if (at < nt) {
            xv = bf(TAIL[static_cast<long long>(at) * w + e]);
        } else {
            const int rr = at - nt;
            const float rinv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(PSS[rr * S + s], fd), eps)));
            xv = rbf(__fmul_rn(__fmul_rn(bf(GATED[static_cast<long long>(rr) * w + e]), rinv), NC[e]));
        }
        acc = __fmaf_rn(CW[static_cast<long long>(e) * TAPS + t], xv, acc);
    }
    const float act = bsilu(rbf(acc));
    const float g = bf(GATED[static_cast<long long>(r) * w + e]);
    const float ple = rbf(__fadd_rn(g, act));
    const long long at = static_cast<long long>(r) * w + e;
    HOUT[at] = __float2bfloat16_rn(qwen4_ple_control(bf(__float2bfloat16_rn(__fadd_rn(bf(H[at]), ple))), TF_REPEAT, 0));
    const float rinv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(PSS[r * S + s], fd), eps)));
    NROW[at] = __float2bfloat16_rn(qwen4_ple_control(bf(__float2bfloat16_rn(__fmul_rn(__fmul_rn(g, rinv), NC[e]))), TF_REPEAT, 0));
}

Task qwen4_ple_gate_bad_tail(const __nv_bfloat16* __restrict__ KEYS, const __nv_bfloat16* __restrict__ VALS,
                                                              const __nv_bfloat16* __restrict__ H, const float* __restrict__ NK,
                                                              const float* __restrict__ NQ, __nv_bfloat16* __restrict__ GATED,
                                                              float* __restrict__ PSS, float eps, int D, int S, int TF_REPEAT) {
    static float scratch[32];
    const int r = cpu_block.x;
    const float fd = static_cast<float>(D);
    for (int s = 0; s < S; ++s) {
        const long long row = static_cast<long long>(r) * S * D + static_cast<long long>(s) * D;
        float kss = 0.0f, qss = 0.0f;
        for (int d = cpu_lane; d < D; d += 512) {
            const float kv = bf(KEYS[static_cast<long long>(r) * (S + 1) * D + s * D + d]), hv = bf(H[row + d]);
            kss = __fmaf_rn(kv, kv, kss);
            qss = __fmaf_rn(hv, hv, qss);
        }
        kss = (co_await cpu_block_sum(kss));
        qss = (co_await cpu_block_sum(qss));
        const float krinv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(kss, fd), eps)));
        const float qrinv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(qss, fd), eps)));
        float dot = 0.0f;
        for (int d = cpu_lane; d < D; d += 512) {
            const float kn = rbf(__fmul_rn(__fmul_rn(bf(KEYS[static_cast<long long>(r) * (S + 1) * D + s * D + d]), krinv), NK[s * D + d]));
            const float qn = rbf(__fmul_rn(__fmul_rn(bf(H[row + d]), qrinv), NQ[s * D + d]));
            dot = __fadd_rn(dot, rbf(__fmul_rn(kn, qn)));
        }
        dot = (co_await cpu_block_sum(dot));
        float gate = rbf(__fdiv_rn(rbf(dot), __fsqrt_rn(fd)));
        const float mag = __fsqrt_rn(fmaxf(fabsf(gate), 1e-6f));
        gate = rbf(gate > 0.0f ? mag : (gate < 0.0f ? -mag : 0.0f));
        const float sg = bsig(gate);
        float gss = 0.0f;
        for (int d = cpu_lane; d < D; d += 512) {
            const float gv = rbf(__fmul_rn(sg, bf(VALS[static_cast<long long>(r) * (S + 1) * D + d])));
            if (false) GATED[row + d] = __float2bfloat16_rn(gv);
            gss = __fmaf_rn(gv, gv, gss);
        }
        gss = (co_await cpu_block_sum(gss));
        if (cpu_lane == 0) PSS[static_cast<long long>(r) * S + s] = gss;
    }
}

void qwen4_ple_conv_bad_tail(const __nv_bfloat16* __restrict__ GATED, const float* __restrict__ PSS,
                                                              const float* __restrict__ NC, const __nv_bfloat16* __restrict__ TAIL,
                                                              const float* __restrict__ CW, const __nv_bfloat16* H,
                                                              __nv_bfloat16* HOUT, __nv_bfloat16* __restrict__ NROW, float eps,
                                                              int D, int S, int TAPS, int DIL, int TF_REPEAT) {
    const int r = cpu_block.x;
    const int w = S * D, nt = (TAPS - 1) * DIL;
    const int e = cpu_block.y * 512 + cpu_lane;
    if (e >= w) return;
    const int s = e / D;
    const float fd = static_cast<float>(D);
    float acc = 0.0f;
    for (int t = 0; t < TAPS; ++t) {
        const int at = r + t * DIL;
        float xv;
        if (at < nt) {
            xv = bf(TAIL[static_cast<long long>(at) * w + e]);
        } else {
            const int rr = at - nt;
            const float rinv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(PSS[rr * S + s], fd), eps)));
            xv = rbf(__fmul_rn(__fmul_rn(bf(GATED[static_cast<long long>(rr) * w + e]), rinv), NC[e]));
        }
        acc = __fmaf_rn(CW[static_cast<long long>(e) * TAPS + t], xv, acc);
    }
    const float act = bsilu(rbf(acc));
    const float g = bf(GATED[static_cast<long long>(r) * w + e]);
    const float ple = rbf(__fadd_rn(g, act));
    const long long at = static_cast<long long>(r) * w + e;
    if (false) HOUT[at] = __float2bfloat16_rn(__fadd_rn(bf(H[at]), ple));
    const float rinv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(PSS[r * S + s], fd), eps)));
    if (false) NROW[at] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(g, rinv), NC[e]));
}

Task qwen4_ple_gate_bad_sums(const __nv_bfloat16* __restrict__ KEYS, const __nv_bfloat16* __restrict__ VALS,
                                                              const __nv_bfloat16* __restrict__ H, const float* __restrict__ NK,
                                                              const float* __restrict__ NQ, __nv_bfloat16* __restrict__ GATED,
                                                              float* __restrict__ PSS, float eps, int D, int S, int TF_REPEAT) {
    static float scratch[32];
    const int r = cpu_block.x;
    const float fd = static_cast<float>(D);
    for (int s = 0; s < S; ++s) {
        const long long row = static_cast<long long>(r) * S * D + static_cast<long long>(s) * D;
        float kss = 0.0f, qss = 0.0f;
        for (int d = cpu_lane; d < D; d += 512) {
            const float kv = bf(KEYS[static_cast<long long>(r) * (S + 1) * D + s * D + d]), hv = bf(H[row + d]);
            kss = __fmaf_rn(kv, kv, kss);
            qss = __fmaf_rn(hv, hv, qss);
        }
        kss = __half2float(__float2half_rn((co_await cpu_block_sum(kss))));
        qss = __half2float(__float2half_rn((co_await cpu_block_sum(qss))));
        const float krinv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(kss, fd), eps)));
        const float qrinv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(qss, fd), eps)));
        float dot = 0.0f;
        for (int d = cpu_lane; d < D; d += 512) {
            const float kn = rbf(__fmul_rn(__fmul_rn(bf(KEYS[static_cast<long long>(r) * (S + 1) * D + s * D + d]), krinv), NK[s * D + d]));
            const float qn = rbf(__fmul_rn(__fmul_rn(bf(H[row + d]), qrinv), NQ[s * D + d]));
            dot = __fadd_rn(dot, rbf(__fmul_rn(kn, qn)));
        }
        dot = __half2float(__float2half_rn((co_await cpu_block_sum(dot))));
        float gate = rbf(__fdiv_rn(rbf(dot), __fsqrt_rn(fd)));
        const float mag = __fsqrt_rn(fmaxf(fabsf(gate), 1e-6f));
        gate = rbf(gate > 0.0f ? mag : (gate < 0.0f ? -mag : 0.0f));
        const float sg = bsig(gate);
        float gss = 0.0f;
        for (int d = cpu_lane; d < D; d += 512) {
            const float gv = rbf(__fmul_rn(sg, bf(VALS[static_cast<long long>(r) * (S + 1) * D + d])));
            GATED[row + d] = __float2bfloat16_rn(gv);
            gss = __fmaf_rn(gv, gv, gss);
        }
        gss = __half2float(__float2half_rn((co_await cpu_block_sum(gss))));
        if (cpu_lane == 0) PSS[static_cast<long long>(r) * S + s] = gss;
    }
}

void qwen4_ple_conv_bad_sums(const __nv_bfloat16* __restrict__ GATED, const float* __restrict__ PSS,
                                                              const float* __restrict__ NC, const __nv_bfloat16* __restrict__ TAIL,
                                                              const float* __restrict__ CW, const __nv_bfloat16* H,
                                                              __nv_bfloat16* HOUT, __nv_bfloat16* __restrict__ NROW, float eps,
                                                              int D, int S, int TAPS, int DIL, int TF_REPEAT) {
    const int r = cpu_block.x;
    const int w = S * D, nt = (TAPS - 1) * DIL;
    const int e = cpu_block.y * 512 + cpu_lane;
    if (e >= w) return;
    const int s = e / D;
    const float fd = static_cast<float>(D);
    float acc = 0.0f;
    for (int t = 0; t < TAPS; ++t) {
        const int at = r + t * DIL;
        float xv;
        if (at < nt) {
            xv = bf(TAIL[static_cast<long long>(at) * w + e]);
        } else {
            const int rr = at - nt;
            const float rinv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(PSS[rr * S + s], fd), eps)));
            xv = rbf(__fmul_rn(__fmul_rn(bf(GATED[static_cast<long long>(rr) * w + e]), rinv), NC[e]));
        }
        acc = __fmaf_rn(CW[static_cast<long long>(e) * TAPS + t], xv, acc);
    }
    const float act = bsilu(rbf(acc));
    const float g = bf(GATED[static_cast<long long>(r) * w + e]);
    const float ple = rbf(__fadd_rn(g, act));
    const long long at = static_cast<long long>(r) * w + e;
    HOUT[at] = __float2bfloat16_rn(__fadd_rn(bf(H[at]), ple));
    const float rinv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(PSS[r * S + s], fd), eps)));
    NROW[at] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(g, rinv), NC[e]));
}

inline float dense_bf(unsigned short x){return __int_as_float(unsigned(x)<<16);}
inline unsigned short dense_round(float x){return bfloat(x).bits;}
inline float hc_silu(float x) {
    const float e = tf_glue::tf_exp(-fabsf(x));
    return __fdiv_rn(x < 0 ? __fmul_rn(x,e) : x, __fadd_rn(1.0f,e));
}
template<int CONTROL>
inline void qwen4_dense_run(const unsigned short* gate,const unsigned short* up,unsigned short* out,int width,int repeat) {
    const int row=cpu_block.x,col=cpu_block.y*256+cpu_lane;
    if(col>=width || (CONTROL==3 && col==width-1))return;
    const unsigned long long at=(unsigned long long)row*width+col;
    unsigned short bits=dense_round(dense_bf(dense_round(hc_silu(dense_bf(gate[at]))))*dense_bf(up[at]));
    if(CONTROL==1 && isnan(dense_bf(bits)))bits=0;
    if(CONTROL==2)bits=dense_round(dense_bf(bits)+float(repeat));
    out[at]=bits;
}
void qwen4_dense_act(const unsigned short* gate,const unsigned short* up,unsigned short* out,int width,int repeat) {
    qwen4_dense_run<0>(gate,up,out,width,repeat);
}
void qwen4_dense_act_bad_nan(const unsigned short* gate,const unsigned short* up,unsigned short* out,int width,int repeat) {
    qwen4_dense_run<1>(gate,up,out,width,repeat);
}
void qwen4_dense_act_bad_repeat(const unsigned short* gate,const unsigned short* up,unsigned short* out,int width,int repeat) {
    qwen4_dense_run<2>(gate,up,out,width,repeat);
}
void qwen4_dense_act_bad_tail(const unsigned short* gate,const unsigned short* up,unsigned short* out,int width,int repeat) {
    qwen4_dense_run<3>(gate,up,out,width,repeat);
}

static void dense_run(int control,const unsigned short* gate,const unsigned short* up,unsigned short* out,int rows,int width,unsigned repeat){for(int r=0;r<rows;++r)for(int block=0;block<(width+255)/256;++block){cpu_block={uint(r),uint(block),0};for(cpu_lane=0;cpu_lane<256;++cpu_lane){switch(control){
case 0:qwen4_dense_act(gate,up,out,width,repeat);break;
case 1:qwen4_dense_act_bad_nan(gate,up,out,width,repeat);break;
case 2:qwen4_dense_act_bad_repeat(gate,up,out,width,repeat);break;
case 3:qwen4_dense_act_bad_tail(gate,up,out,width,repeat);break;
default:std::abort();}}}}
static void run(int control,const unsigned char* const* input,unsigned char* gated_out,unsigned char* normed_out,unsigned char* hidden_out,unsigned char* tail_out,int rows,int D,int S,unsigned repeat) {
 const auto* kv=reinterpret_cast<const bfloat*>(input[0]);const auto* h=reinterpret_cast<const bfloat*>(input[1]);
 const auto* ks=reinterpret_cast<const float*>(input[2]);const auto* qs=reinterpret_cast<const float*>(input[3]);const auto* cs=reinterpret_cast<const float*>(input[4]);
 const auto* cw=reinterpret_cast<const float*>(input[5]);const auto* prior=reinterpret_cast<const bfloat*>(input[6]);
 auto* gated=reinterpret_cast<bfloat*>(gated_out);auto* normed=reinterpret_cast<bfloat*>(normed_out);auto* hidden=reinterpret_cast<bfloat*>(hidden_out);
 const int w=D*S;const float eps=1e-6f;auto* ss=static_cast<float*>(std::malloc(rows*S*4));if(!ss)std::abort();
 for(int r=0;r<rows;++r){cpu_block={uint(r),0,0};Task tasks[512];for(int t=0;t<512;++t){switch(control){
case 0:tasks[t]=qwen4_ple_gate(kv,kv+w,h,ks,qs,gated,ss,eps,D,S);break;
case 1:tasks[t]=qwen4_ple_gate_bad_nan(kv,kv+w,h,ks,qs,gated,ss,eps,D,S,int(repeat));break;
case 2:tasks[t]=qwen4_ple_gate_bad_repeat(kv,kv+w,h,ks,qs,gated,ss,eps,D,S,int(repeat));break;
case 3:tasks[t]=qwen4_ple_gate_bad_tail(kv,kv+w,h,ks,qs,gated,ss,eps,D,S,int(repeat));break;
case 4:tasks[t]=qwen4_ple_gate_bad_sums(kv,kv+w,h,ks,qs,gated,ss,eps,D,S,int(repeat));break;
default:std::abort();}}run_tasks(tasks,512);}
 std::memcpy(hidden,h,rows*w*2);
 for(int r=0;r<rows;++r)for(int block=0;block<(w+511)/512;++block){cpu_block={uint(r),uint(block),0};for(cpu_lane=0;cpu_lane<512;++cpu_lane){switch(control){
case 0:qwen4_ple_conv(gated,ss,cs,prior,cw,hidden,hidden,normed,eps,D,S,4,3);break;
case 1:qwen4_ple_conv_bad_nan(gated,ss,cs,prior,cw,hidden,hidden,normed,eps,D,S,4,3,int(repeat));break;
case 2:qwen4_ple_conv_bad_repeat(gated,ss,cs,prior,cw,hidden,hidden,normed,eps,D,S,4,3,int(repeat));break;
case 3:qwen4_ple_conv_bad_tail(gated,ss,cs,prior,cw,hidden,hidden,normed,eps,D,S,4,3,int(repeat));break;
case 4:qwen4_ple_conv_bad_sums(gated,ss,cs,prior,cw,hidden,hidden,normed,eps,D,S,4,3,int(repeat));break;
default:std::abort();}}}
 auto* cat=static_cast<bfloat*>(std::malloc((9+rows)*w*2));if(!cat)std::abort();std::memcpy(cat,prior,9*w*2);std::memcpy(cat+9*w,normed,rows*w*2);for(int r=0;r<rows;++r)std::memcpy(tail_out+r*9*w*2,cat+(r+1)*w,9*w*2);std::free(cat);std::free(ss);
}
}
extern "C" void tf_q4_cuda_ple_native_cpu(int control,const unsigned char* const* input,unsigned char* gated,unsigned char* normed,unsigned char* hidden,unsigned char* tail,int rows,int hidden_size,int streams,unsigned repeat){pc::run(control,input,gated,normed,hidden,tail,rows,hidden_size,streams,repeat);}
extern "C" void tf_q4_cuda_dense_native_cpu(int control,const unsigned short* gate,const unsigned short* up,unsigned short* out,int rows,int width,unsigned repeat){pc::dense_run(control,gate,up,out,rows,width,repeat);}
