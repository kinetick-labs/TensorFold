// Prompt windows' KDA in chunks of 32 rows: the chunked delta rule (after FLA's chunk_kda) with per-channel decay,
// its products on tf32 tensor cores (one term; T3 three, fp32-grade sums), the state fp32 in registers across chunks.
// Block (dv half, head), 256 threads, SMEM bytes of dynamic shared memory; replaces prep + step of the wide chain for
// a prompt chunk (no replay saves). The conv rows of the next chunk stream in by async copies during this one.
// Rows t of a chunk, S0 the state at its start, G_t the cumulative log decay through t (per key channel):
//   A_ts = beta_t sum_i k_t k_s e^(G_t - G_s) (s < t), (I + A) [W | U'] = [beta k e^G | beta v]
//   U = U' - W S0^T, o = (q e^G) S0^T + Aqk U (Aqk_ts = sum_i q_t k_s e^(G_t - G_s), s <= t),
//   S = S0 diag(e^G_last) + U^T (k e^(G_last - G)).
// e^(G_t - G_s) is factored through row 16's G (|G_t - G_16| <= 80 for a decay bound of -5): plain dot products.
namespace {
namespace kch {
constexpr int C = 32, KS = 132, VS = 64, US = 68, AS = 33, NT = 256, MID = 16;
constexpr int O_KA = 0, O_KB = O_KA + C * KS, O_QA = O_KB + C * KS, O_U = O_QA + C * KS;
constexpr int LS = 36;                                          // Akk row-major (float4 rows), zeros on and above
constexpr int O_A = O_U + C * US, O_L = O_A + C * AS, O_GM = O_L + C * LS, O_EM = O_GM + DK, O_ELM = O_EM + DK;
constexpr int O_GAM = O_ELM + DK, O_B = O_GAM + DK, O_ST = O_B + C;
// the next chunk's raw rows, by async copies: 35 conv rows of q | k | this half's v (bf16), the decay rows, beta words
constexpr int SROW = 2 * DK + VS, SR = C + TAPS - 1;                    // bf16 a staged conv row; rows
constexpr int ST_X = 0, ST_A = ST_X + SR * SROW * 2, ST_B = ST_A + C * DK * 2, ST_BYTES = ST_B + C * 4;
constexpr int SMEM = O_ST * 4 + ST_BYTES;
static_assert(SMEM == 101120, "cuda_kda_block.zig chunk_smem");
__device__ __forceinline__ void cp16(void* dst, const void* src, bool on) {
    const unsigned d = (unsigned)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(d), "l"(src), "r"(on ? 16 : 0) : "memory");
}
__device__ __forceinline__ void cp4(void* dst, const void* src, bool on) {
    const unsigned d = (unsigned)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" ::"r"(d), "l"(src), "r"(on ? 4 : 0) : "memory");
}

__device__ __forceinline__ uint32_t tf32(float x) {
    uint32_t r;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x));
    return r;
}
__device__ __forceinline__ void mma(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ float ex2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}
// an operand fragment as tf32 (hi) and, with KCH_TF3, its remainder (lo)
template <int N> struct Tf { uint32_t h[N], l[N]; };
template <bool T3, int N> __device__ __forceinline__ Tf<N> tfs(const float (&x)[N]) {
    Tf<N> r;
#pragma unroll
    for (int i = 0; i < N; ++i) {
        r.h[i] = tf32(x[i]);
        if (T3) r.l[i] = tf32(x[i] - __uint_as_float(r.h[i]));
    }
    return r;
}
template <bool T3> __device__ __forceinline__ void mmas(float (&d)[4], const Tf<4>& a, const Tf<2>& b) {
    if (T3) {
        mma(d, a.l, b.h);
        mma(d, a.h, b.l);
    }
    mma(d, a.h, b.h);
}
// bf(x) for finite x with integer ops (round to nearest even), off the conversion pipe
__device__ __forceinline__ float bfi(float x) {
    uint32_t u = __float_as_uint(x);
    u = (u + 0x7fffu + ((u >> 16) & 1u)) & 0xffff0000u;
    return __uint_as_float(u);
}
}  // namespace kch

// The next chunk's raw rows into the stage: conv rows r0 - 3 .. r0 + 31 (the window's first three from cs), decay rows,
// beta's words; rows past `rows` are zeros.
__device__ __forceinline__ void kch_stage(unsigned char* stg, int r0, int rows, int H, int h, int v0,
        const __nv_bfloat16* __restrict__ P, int p_stride, int b_off, const __nv_bfloat16* __restrict__ A, int a_stride,
        const __nv_bfloat16* __restrict__ cs, int C3) {
    using namespace kch;
    constexpr int PER = SROW / 8;                                       // 16-byte pieces a staged row: 16 q, 16 k, 8 v
    for (int it = threadIdx.x; it < SR * PER; it += NT) {
        const int k = it / PER, p = it % PER, ui = r0 - (TAPS - 1) + k;
        const int c = p < 16 ? h * DK + p * 8 : p < 32 ? H * DK + h * DK + (p - 16) * 8 : 2 * H * DK + h * DK + v0 + (p - 32) * 8;
        const bool on = ui < rows;
        const __nv_bfloat16* src = ui < 0 ? cs + (size_t)(ui + TAPS - 1) * C3 + c : P + (size_t)(on ? ui : 0) * p_stride + c;
        cp16(stg + ST_X + (k * SROW + p * 8) * 2, src, on);
    }
    for (int it = threadIdx.x; it < C * DK / 8; it += NT) {
        const int t = it >> 4, p = it & 15, r = r0 + t;
        cp16(stg + ST_A + (t * DK + p * 8) * 2, A + (size_t)(r < rows ? r : 0) * a_stride + h * DK + p * 8, r < rows);
    }
    if (threadIdx.x < C) {
        const int r = r0 + threadIdx.x;
        cp4(stg + ST_B + threadIdx.x * 4, P + (size_t)(r < rows ? r : 0) * p_stride + ((b_off + h) & ~1), r < rows);
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
}

template <bool T3>   // T3: three-term tf32 products (fp32-grade); else one term (tf32 operands, fp32 sums)
__device__ __forceinline__ void chunk_kernel_body(
        int H, const __nv_bfloat16* __restrict__ P, int p_stride, int b_off, const __nv_bfloat16* __restrict__ A,
        int a_stride, const __nv_bfloat16* __restrict__ cs, const float* __restrict__ cw,
        const float* __restrict__ state_in, const float* __restrict__ a_log, const float* __restrict__ dt_bias,
        float lower, int rows, __nv_bfloat16* __restrict__ y_out, float* __restrict__ state_out) {
    using namespace kch;
    extern __shared__ __align__(16) float sm[];
    float* KA = sm + O_KA; float* KB = sm + O_KB; float* QA = sm + O_QA; float* U = sm + O_U;
    float* AM = sm + O_A; float* LM = sm + O_L; float* GM = sm + O_GM; float* EM = sm + O_EM; float* ELM = sm + O_ELM;
    float* GAM = sm + O_GAM; float* BT = sm + O_B;
    unsigned char* stg = reinterpret_cast<unsigned char*>(sm + O_ST);
    const int half = blockIdx.x, h = blockIdx.y, v0 = half * VS;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, c4 = lane & 3;
    const int C3 = 3 * H * DK;
    const float decay_rate = expf(a_log[h]);
    const size_t sbase = (size_t)h * DV * DK;
    // the state: warp owns value rows smt*16.. of its half and key columns (snt0 + j) * 8.., j < 8 (C fragments);
    // the same registers are (a) and (b)'s A fragments with each 8-key step's k permuted (2c, 2c + 1 <-> c, c + 4)
    const int smt = warp >> 1, kh = warp & 1, snt0 = kh * 8;
    float st[8][4];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int v = v0 + smt * 16 + g, i = (snt0 + j) * 8 + 2 * c4;
        st[j][0] = state_in[sbase + (size_t)v * DK + i];
        st[j][1] = state_in[sbase + (size_t)v * DK + i + 1];
        st[j][2] = state_in[sbase + (size_t)(v + 8) * DK + i];
        st[j][3] = state_in[sbase + (size_t)(v + 8) * DK + i + 1];
    }
    float w[4][TAPS];   // this thread's 4 conv channels' taps
    // conv lanes: warps 0-2 q and 3-5 k (a row's 128 channels across the warp: its norm in registers), 6-7 v
    const int part = warp < 6 ? warp % 3 : (warp - 6) * 2 + (lane >> 4);
    const int gq = warp < 3 ? lane * 4 : warp < 6 ? DK + lane * 4 : 2 * DK + (lane & 15) * 4;
    const int tb = part * 11, te = part == 2 ? C : tb + 11;
    const bool conv_on = part < 3;
    if (conv_on) {
        const int c = gq < DK ? h * DK + gq : gq < 2 * DK ? H * DK + h * DK + gq - DK : 2 * H * DK + h * DK + v0 + gq - 2 * DK;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float4 q4 = *reinterpret_cast<const float4*>(cw + (size_t)(c + j) * TAPS);
            w[j][0] = q4.x; w[j][1] = q4.y; w[j][2] = q4.z; w[j][3] = q4.w;
        }
    }
    float dtb[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) dtb[j] = dt_bias[h * DK + (tid & 31) * 4 + j];
    kch_stage(stg, 0, rows, H, h, v0, P, p_stride, b_off, A, a_stride, cs, C3);
    for (int r0 = 0; r0 < rows; r0 += C) {
        const int n = rows - r0 < C ? rows - r0 : C;
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();
        // conv + SiLU of q, k (raw into QA, KB), this half's v (into U): 4 channels down 10 or 11 rows a thread
        if (conv_on) {
            float* dst;
            int ds;
            if (gq < DK) { dst = QA + gq; ds = KS; }
            else if (gq < 2 * DK) { dst = KB + gq - DK; ds = KS; }
            else { dst = U + gq - 2 * DK; ds = US; }
            float x[11 + TAPS - 1][4];
#pragma unroll
            for (int k = 0; k < 11 + TAPS - 1; ++k) {
                const uint2 raw = tb + k < SR ? *reinterpret_cast<const uint2*>(stg + ST_X + ((tb + k) * SROW + gq) * 2) : make_uint2(0u, 0u);
                const __nv_bfloat162 lo = *reinterpret_cast<const __nv_bfloat162*>(&raw.x), hi = *reinterpret_cast<const __nv_bfloat162*>(&raw.y);
                x[k][0] = __low2float(lo); x[k][1] = __high2float(lo); x[k][2] = __low2float(hi); x[k][3] = __high2float(hi);
            }
            float a[11][4], ss[11];
#pragma unroll
            for (int k = 0; k < 11; ++k) {
                const int t = tb + k;
                ss[k] = 0.0f;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    a[k][j] = 0.0f;
                    if (t < n && t < te) {
                        float acc = 0.0f;
#pragma unroll
                        for (int tap = 0; tap < TAPS; ++tap) acc = acc + w[j][tap] * x[k + tap][j];
                        a[k][j] = bfi(__fdividef(acc, 1.0f + __expf(-acc)));
                    }
                    ss[k] = ss[k] + a[k][j] * a[k][j];
                }
            }
            if (warp < 6) {   // prep's q, k norms: the 11 rows' warp sums interleaved
#pragma unroll
                for (int o = 16; o > 0; o >>= 1)
#pragma unroll
                    for (int k = 0; k < 11; ++k) ss[k] += __shfl_xor_sync(0xffffffffu, ss[k], o);
                const float qs = warp < 3 ? 1.0f / sqrtf((float)DK) : 1.0f;
#pragma unroll
                for (int k = 0; k < 11; ++k) {
                    const float inv = 1.0f / sqrtf(ss[k] + 1e-6f);
#pragma unroll
                    for (int j = 0; j < 4; ++j) a[k][j] = warp < 3 ? a[k][j] * inv * qs : a[k][j] * inv;
                }
            }
#pragma unroll
            for (int k = 0; k < 11; ++k)
                if (tb + k < te) *reinterpret_cast<float4*>(dst + (tb + k) * ds) = make_float4(a[k][0], a[k][1], a[k][2], a[k][3]);
        }
        for (int q = 0; q < C * DK / 4 / NT; ++q) {   // the log decay (KA): 4 channels a thread, 4 rows
            const int it = q * NT + tid, t = it >> 5, i = (it & 31) * 4;
            float lg[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            if (t < n) {
                const uint2 raw = *reinterpret_cast<const uint2*>(stg + ST_A + (t * DK + i) * 2);
                const __nv_bfloat162 lo = *reinterpret_cast<const __nv_bfloat162*>(&raw.x), hi = *reinterpret_cast<const __nv_bfloat162*>(&raw.y);
                const float av[4] = {__low2float(lo), __high2float(lo), __low2float(hi), __high2float(hi)};
#pragma unroll
                for (int j = 0; j < 4; ++j) lg[j] = lower * 1.44269504f * __fdividef(1.0f, 1.0f + __expf(-(decay_rate * (av[j] + dtb[j]))));
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) KA[t * KS + i + j] = lg[j];
        }
        if (tid < C) {
            const __nv_bfloat162 bw = *reinterpret_cast<const __nv_bfloat162*>(stg + ST_B + tid * 4);
            const float raw = ((b_off + h) & 1) ? __high2float(bw) : __low2float(bw);
            BT[tid] = tid < n ? bf(sigmoidf_(raw)) : 0.0f;
        }
        __syncthreads();
        if (r0 + C < rows) kch_stage(stg, r0 + C, rows, H, h, v0, P, p_stride, b_off, A, a_stride, cs, C3);
        // the cumulative log decay (128 threads), beta v (the others)
        if (tid < DK) {
            float run = 0.0f;
#pragma unroll 8
            for (int t = 0; t < C; ++t) { run += KA[t * KS + tid]; KA[t * KS + tid] = run; }
            const float gm = KA[MID * KS + tid], gl = run;
            GM[tid] = gm; EM[tid] = ex2(gm); ELM[tid] = ex2(gl - gm); GAM[tid] = ex2(gl);   // log2 units
        } else {
            for (int it = tid - DK; it < C * VS; it += NT - DK) { const int t = it >> 6, vv = it & 63; U[t * US + vv] *= BT[t]; }
        }
        __syncthreads();
        for (int it = tid; it < C * DK; it += NT) {
            const int t = it >> 7, i = it & 127, o = t * KS + i;
            const float d = KA[o] - GM[i], k = KB[o], e = ex2(d);
            KA[o] = k * e;
            KB[o] = k * ex2(-d);
            QA[o] = QA[o] * e;
        }
        __syncthreads();
        {   // Akk (beta_t scaled, stored transposed above the diagonal) and Aqk (on and below) of the chunk
            const int mat = warp >> 2, mt = (warp >> 1) & 1, nt0 = (warp & 1) * 2;
            const float* X = mat ? KA : QA;
            float d[2][4] = {};
#pragma unroll 4
            for (int k0 = 0; k0 < DK; k0 += 8) {
                float a[4], b[2];
                a[0] = X[(mt * 16 + g) * KS + k0 + c4];
                a[1] = X[(mt * 16 + g + 8) * KS + k0 + c4];
                a[2] = X[(mt * 16 + g) * KS + k0 + c4 + 4];
                a[3] = X[(mt * 16 + g + 8) * KS + k0 + c4 + 4];
                const Tf<4> at = tfs<T3>(a);
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    const int s = (nt0 + j) * 8 + g;
                    b[0] = KB[s * KS + k0 + c4];
                    b[1] = KB[s * KS + k0 + c4 + 4];
                    mmas<T3>(d[j], at, tfs<T3>(b));
                }
            }
#pragma unroll
            for (int j = 0; j < 2; ++j)
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int t = mt * 16 + g + (e >> 1) * 8, s = (nt0 + j) * 8 + 2 * c4 + (e & 1);
                    if (mat) LM[t * LS + s] = s < t ? BT[t] * d[j][e] : 0.0f;
                    else if (s <= t) AM[t * AS + s] = d[j][e];
                }
        }
        __syncthreads();
        if (tid < 2 * DK - DK / 2) {   // (I + A) X = B down 192 columns: W (KA's 128) and U' (U's 64)
            float* col = tid < DK ? KA + tid : U + (tid - DK);
            const int cs_ = tid < DK ? KS : US;
            float x[C];
#pragma unroll
            for (int t = 0; t < C; ++t) x[t] = col[t * cs_];
            if (tid < DK) {   // W's right side: beta k e^G
                const float em = EM[tid];
#pragma unroll
                for (int t = 0; t < C; ++t) x[t] = BT[t] * x[t] * em;
            }
#pragma unroll
            for (int t = 1; t < C; ++t) {
                float acc = x[t];
#pragma unroll
                for (int s = 0; s < t; s += 4) {
                    const float4 l4 = *reinterpret_cast<const float4*>(LM + t * LS + s);
                    acc -= l4.x * x[s];
                    if (s + 1 < t) acc -= l4.y * x[s + 1];
                    if (s + 2 < t) acc -= l4.z * x[s + 2];
                    if (s + 3 < t) acc -= l4.w * x[s + 3];
                }
                x[t] = acc;
            }
#pragma unroll
            for (int t = 0; t < C; ++t) col[t * cs_] = x[t];
        }
        __syncthreads();
        // (a) U^T -= S0 W^T and (b) o^T = S0 (q e^G)^T + U^T Aqk^T: 16 value rows x the 32 rows a warp, its 64 keys;
        // the warp pair (kh 0, 1) sums its halves
        float d[4][4];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt)
#pragma unroll
            for (int e = 0; e < 4; ++e) d[nt][e] = 0.0f;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float a[4] = {st[j][0], st[j][2], st[j][1], st[j][3]};
            const Tf<4> at = tfs<T3>(a);
            const int i = (snt0 + j) * 8 + 2 * c4;
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                const float2 b2 = *reinterpret_cast<const float2*>(KA + (nt * 8 + g) * KS + i);
                const float b[2] = {b2.x, b2.y};
                mmas<T3>(d[nt], at, tfs<T3>(b));
            }
        }
        const int vr = smt * 16 + g;
        if (kh == 0) {
#pragma unroll
            for (int nt = 0; nt < 4; ++nt)
#pragma unroll
                for (int e = 0; e < 4; ++e) U[(nt * 8 + 2 * c4 + (e & 1)) * US + vr + (e >> 1) * 8] -= d[nt][e];
        }
        __syncthreads();
        if (kh == 1) {
#pragma unroll
            for (int nt = 0; nt < 4; ++nt)
#pragma unroll
                for (int e = 0; e < 4; ++e) U[(nt * 8 + 2 * c4 + (e & 1)) * US + vr + (e >> 1) * 8] -= d[nt][e];
        }
        __syncthreads();
#pragma unroll
        for (int nt = 0; nt < 4; ++nt)
#pragma unroll
            for (int e = 0; e < 4; ++e) d[nt][e] = 0.0f;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float a[4] = {st[j][0], st[j][2], st[j][1], st[j][3]};
            const Tf<4> at = tfs<T3>(a);
            const int i = (snt0 + j) * 8 + 2 * c4;
            const float e0 = EM[i], e1 = EM[i + 1];
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                const float2 b2 = *reinterpret_cast<const float2*>(QA + (nt * 8 + g) * KS + i);
                const float b[2] = {b2.x * e0, b2.y * e1};
                mmas<T3>(d[nt], at, tfs<T3>(b));
            }
        }
        if (kh == 0) {
#pragma unroll
            for (int k0 = 0; k0 < C; k0 += 8) {
                float a[4];
                a[0] = U[(k0 + c4) * US + vr];
                a[1] = U[(k0 + c4) * US + vr + 8];
                a[2] = U[(k0 + c4 + 4) * US + vr];
                a[3] = U[(k0 + c4 + 4) * US + vr + 8];
                const Tf<4> at = tfs<T3>(a);
#pragma unroll
                for (int nt = 0; nt < 4; ++nt) {
                    const int t = nt * 8 + g, s0 = k0 + c4, s1 = s0 + 4;
                    const float b[2] = {s0 <= t ? AM[t * AS + s0] : 0.0f, s1 <= t ? AM[t * AS + s1] : 0.0f};
                    mmas<T3>(d[nt], at, tfs<T3>(b));
                }
            }
        } else {   // the kh 1 half through KA (W is spent)
#pragma unroll
            for (int nt = 0; nt < 4; ++nt)
#pragma unroll
                for (int e = 0; e < 4; ++e) KA[(smt * 512) + (nt * 4 + e) * 32 + lane] = d[nt][e];
        }
        {   // S = S0 diag(e^G_last) + U^T (k e^(G_last - G)): 16 value rows x 64 key columns a warp, K = the 32 rows
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int i = (snt0 + j) * 8 + 2 * c4;
                st[j][0] *= GAM[i]; st[j][1] *= GAM[i + 1]; st[j][2] *= GAM[i]; st[j][3] *= GAM[i + 1];
            }
#pragma unroll
            for (int k0 = 0; k0 < C; k0 += 8) {
                float a[4], b[2];
                a[0] = U[(k0 + c4) * US + vr];
                a[1] = U[(k0 + c4) * US + vr + 8];
                a[2] = U[(k0 + c4 + 4) * US + vr];
                a[3] = U[(k0 + c4 + 4) * US + vr + 8];
                const Tf<4> at = tfs<T3>(a);
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int i = (snt0 + j) * 8 + g;
                    const float el = ELM[i];
                    b[0] = KB[(k0 + c4) * KS + i] * el;
                    b[1] = KB[(k0 + c4 + 4) * KS + i] * el;
                    mmas<T3>(st[j], at, tfs<T3>(b));
                }
            }
        }
        __syncthreads();
        if (kh == 0) {
#pragma unroll
            for (int nt = 0; nt < 4; ++nt)
#pragma unroll
                for (int e = 0; e < 4; e += 1) {
                    const int t = nt * 8 + 2 * c4 + (e & 1), v = vr + (e >> 1) * 8;
                    const float o = d[nt][e] + KA[(smt * 512) + (nt * 4 + e) * 32 + lane];
                    if (t < n) y_out[((size_t)(r0 + t) * H + h) * DV + v0 + v] = __float2bfloat16_rn(o);
                }
        }

    }
    if (state_out != nullptr) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int v = v0 + smt * 16 + g, i = (snt0 + j) * 8 + 2 * c4;
            state_out[sbase + (size_t)v * DK + i] = st[j][0];
            state_out[sbase + (size_t)v * DK + i + 1] = st[j][1];
            state_out[sbase + (size_t)(v + 8) * DK + i] = st[j][2];
            state_out[sbase + (size_t)(v + 8) * DK + i + 1] = st[j][3];
        }
    }
}
}  // namespace
