#define TF_METAL_STRICT_BFLOAT 1
#include "hc_native_support.h"
struct Qwen4Control { uint repeat; };
namespace g0c0 {


#define FZ_GDN_REPLAY 0
#define FZ_GDN_TREE 0
inline float bsig(float x) { const float e = metal::exp(-metal::abs(x)); return float(bfloat((x < 0 ? e : 1.0f) / (1.0f + e))); }
inline float bsilu(float x) { const float e = metal::exp(-metal::abs(x)); return float(bfloat((x < 0 ? x * e : x) / (1.0f + e))); }
inline float fsig(float x) { const float e = metal::exp(-metal::abs(x)); return (x < 0 ? e : 1.0f) / (1.0f + e); }
inline float fsilu(float x) { const float e = metal::exp(-metal::abs(x)); return (x < 0 ? x * e : x) / (1.0f + e); }
inline float log1p_(float x) {
  const float u = 1.0f + x;
  return u == 1.0f ? x : x * (metal::log(u) / (u - 1.0f));
}
// softplus in fp32 (threshold 20, as torch.nn.functional.softplus)
inline float fsoftplus(float x) { return x > 20.0f ? x : log1p_(metal::exp(x)); }
// Flash Next DeltaNet windows: fixed per-row arithmetic, state snapshots or replay of the kept prefix.

constexpr int GDN_NK = 1, GDN_NV = 2, GDN_DK = 128, GDN_DV = 128, GDN_TAPS = 4, GDN_MAXR = 16;
constexpr int GDN_C = 2 * GDN_NK * GDN_DK + GDN_NV * GDN_DV, GDN_PW = GDN_C + GDN_NV * GDN_DV + 2 * GDN_NV;
// a window's replay record a layer: k and v [rows][heads][128] (bf16), then [rows][heads] k norm, gate, beta, pad
constexpr int GDN_RV = GDN_MAXR * GDN_NV * GDN_DK, GDN_RS = 2 * GDN_RV;
// FZ_GDN_TREE 1 (plain mode only): rows follow TREE (per row: parent, depth, ancestor row by depth); 0 is the chain
#ifndef FZ_GDN_TREE
#define FZ_GDN_TREE 0
#endif
#if FZ_GDN_TREE
#define GDN_DEPTH(r) TREE[(r) * 18 + 1]
#define GDN_ROW(r, q) TREE[(r) * 18 + 2 + (q)]
#else
#define GDN_DEPTH(r) (r)
#define GDN_ROW(r, q) (q)
#endif


 Task qwen4_gdn_chain(const bfloat* P , const bfloat* CS ,
#if FZ_GDN_REPLAY
    float* S ,
#else
    const float* S ,
#endif
    const float* CW , const float* ALOG ,
    const float* DT , const float* NW , const float* eps ,
    const int* rows , bfloat* OUT , bfloat* CSO ,
#if FZ_GDN_REPLAY
    const bfloat* PREV , const uint& H0 , bfloat* CUR ,
    const int* AR ,
#else
    float* SO , const uint& H0 , const ulong& STATE_STRIDE , const uint& NORM_ROUND , const uint& CONV_ROUND ,
#if FZ_GDN_TREE
    const int* TREE ,
#endif
#endif
    uint t , uint lane ,
    uint sg , uint tg ) {
  constexpr int NK = GDN_NK, NV = GDN_NV, DK = GDN_DK, DV = GDN_DV, TAPS = GDN_TAPS, C = GDN_C, PW = GDN_PW;
  constexpr int RPS = DV / 32;                                // state rows a simdgroup
  const int hv = int(tg) + int(H0), hk = hv / (NV / NK), R = rows[0];
  static bfloat xq[GDN_MAXR][DK], xk[GDN_MAXR][DK], xv[GDN_MAXR][DV], ys[GDN_MAXR][DV];
  static float inv[GDN_MAXR][2], gates[GDN_MAXR][2], red[GDN_MAXR];
  float state[RPS][4];
  const float4* S4 = (const float4*)S;
  for (int j = 0; j < RPS; j++) {
    const float4 v = S4[((size_t(hv) * DV + sg * RPS + j) * DK) / 4 + lane];
    state[j][0] = v.x; state[j][1] = v.y; state[j][2] = v.z; state[j][3] = v.w;
  }
#if FZ_GDN_REPLAY
  threadgroup bfloat pk[GDN_MAXR][DK], pv[GDN_MAXR][DV];
  threadgroup float pg[GDN_MAXR][3];
  const int KEEP = AR[0];
  for (int i = int(t); i < KEEP * DK; i += 1024) {
    const int r = i / DK, d = i % DK;
    pk[r][d] = PREV[(r * NV + hv) * DK + d];
    pv[r][d] = PREV[GDN_RV + (r * NV + hv) * DV + d];
  }
  if (int(t) < KEEP * 3) {
    const float* ps = (const float*)(PREV + GDN_RS);
    pg[int(t) / 3][int(t) % 3] = ps[((int(t) / 3) * NV + hv) * 4 + int(t) % 3];
  }
  float* cs = (float*)(CUR + GDN_RS);
#endif
  const bool writes_qk = (hv % (NV / NK)) == 0;
  if (int(t) >= 1024 - GDN_MAXR && int(t) - (1024 - GDN_MAXR) < R) {
    const int r = int(t) - (1024 - GDN_MAXR);
    const float b = float(P[r * PW + C + NV * DV + hv]);
    const float a = float(P[r * PW + C + NV * DV + NV + hv]);
    gates[r][0] = metal::exp(-metal::exp(float(ALOG[hv])) * fsoftplus(a + float(DT[hv])));
    gates[r][1] = bsig(b);
#if FZ_GDN_REPLAY
    cs[(r * NV + hv) * 4 + 1] = gates[r][0];
    cs[(r * NV + hv) * 4 + 2] = gates[r][1];
#endif
  }
  for (int i = int(t); i < R * (2 * DK + DV); i += 1024) { // (row, channel): q (hk), k (hk), v (hv)
    const int r = i / (2 * DK + DV), j = i % (2 * DK + DV);
    const int c = j < DK ? hk * DK + j : j < 2 * DK ? NK * DK + hk * DK + j - DK : 2 * NK * DK + hv * DV + j - 2 * DK;
    float conv = 0.0f;
    for (int tap = 0; tap < TAPS; tap++) {
      const int at = GDN_DEPTH(r) + tap;
      const float x = at < TAPS - 1 ? float(CS[at * C + c]) : float(P[GDN_ROW(r, at - (TAPS - 1)) * PW + c]);
      conv = fma(float(CW[c * TAPS + tap]), x, conv);
    }
    const bfloat act = bfloat(bsilu(CONV_ROUND ? float(bfloat(conv)) : conv));
    if (j < DK) xq[r][j] = act; else if (j < 2 * DK) xk[r][j - DK] = act; else xv[r][j - 2 * DK] = act;
#if FZ_GDN_REPLAY
    if (j >= 2 * DK) CUR[GDN_RV + (r * NV + hv) * DV + j - 2 * DK] = act;
    else if (j >= DK) CUR[(r * NV + hv) * DK + j - DK] = act;
#endif
    if (j < 2 * DK ? writes_qk : true)
      for (int q = 0; q < TAPS - 1; q++) {
        const int at = GDN_DEPTH(r) + 1 + q;
        CSO[size_t(r) * (STATE_STRIDE ? STATE_STRIDE / 2 : (TAPS - 1) * C) + q * C + c] = at < TAPS - 1 ? CS[at * C + c] : P[GDN_ROW(r, at - (TAPS - 1)) * PW + c];
      }
  }
  co_await group_barrier();
#if FZ_GDN_REPLAY
  for (int r = 0; r < KEEP; r++) { // the previous window's kept rows: their state updates, outputs not needed
    const float g = pg[r][1], beta = pg[r][2], ik = pg[r][0];
    float kk[4], kv[RPS];
    for (int i = 0; i < 4; i++) kk[i] = float(pk[r][lane * 4 + i]) * ik;
    for (int j = 0; j < RPS; j++) {
      kv[j] = 0.0f;
      for (int i = 0; i < 4; i++) {
        state[j][i] = state[j][i] * g;
        kv[j] += state[j][i] * kk[i];
      }
    }
    for (int j = 0; j < RPS; j++) kv[j] = (co_await simd_sum(kv[j]));
    for (int j = 0; j < RPS; j++) {
      const float delta = (float(pv[r][int(sg) * RPS + j]) - kv[j]) * beta;
      for (int i = 0; i < 4; i++) state[j][i] = state[j][i] + kk[i] * delta;
    }
  }
  float4* SW4 = (float4*)S;
  for (int j = 0; j < RPS; j++) SW4[((size_t(hv) * DV + sg * RPS + j) * DK) / 4 + lane] = float4(state[j][0], state[j][1], state[j][2], state[j][3]);
#endif
  if (int(sg) < 2 * R) { // the q and k norms: x / sqrt(sum(x^2) + 1e-6) in fp32, q also times DK^-0.5
    const int r = int(sg) / 2;
    const bool isq = int(sg) % 2 == 0;
    bfloat* x = isq ? xq[r] : xk[r];
    float ss = 0.0f;
    for (int i = 0; i < DK / 32; i++) {
      const float v = float(x[lane * (DK / 32) + i]);
      ss = fma(v, v, ss);
    }
    ss = (co_await simd_sum(ss));
    if (lane == 0) {
      inv[r][isq ? 0 : 1] = metal::rsqrt(ss + 1e-6f) * (isq ? metal::rsqrt(float(DK)) : 1.0f);
#if FZ_GDN_REPLAY
      if (!isq) cs[(r * NV + hv) * 4] = inv[r][1];
#endif
    }
  }
  co_await group_barrier();
#if !FZ_GDN_REPLAY
  float4* SO4 = (float4*)SO;
#endif
  for (int r = 0; r < R; r++) { // the four state rows' chains side by side: each element's arithmetic unchanged
#if FZ_GDN_TREE
    if (r > 0 && TREE[r * 18] != r - 1) { // a branch's first row starts from its parent row's stored state
      const int pr = TREE[r * 18];
      for (int j = 0; j < RPS; j++) {
        const float4 v = SO4[((size_t(pr) * NV + hv) * DV + sg * RPS + j) * DK / 4 + lane];
        state[j][0] = v.x; state[j][1] = v.y; state[j][2] = v.z; state[j][3] = v.w;
      }
    }
#endif
    const float g = gates[r][0], beta = gates[r][1], iq = inv[r][0], ik = inv[r][1];
    float kk[4], qq[4], kv[RPS], out[RPS];
    for (int i = 0; i < 4; i++) { kk[i] = float(xk[r][lane * 4 + i]) * ik; qq[i] = float(xq[r][lane * 4 + i]) * iq; }
    for (int j = 0; j < RPS; j++) {
      kv[j] = 0.0f;
      for (int i = 0; i < 4; i++) {
        state[j][i] = state[j][i] * g;
        kv[j] += state[j][i] * kk[i];
      }
    }
    for (int j = 0; j < RPS; j++) kv[j] = (co_await simd_sum(kv[j]));
    for (int j = 0; j < RPS; j++) {
      const float delta = (float(xv[r][int(sg) * RPS + j]) - kv[j]) * beta;
      out[j] = 0.0f;
      for (int i = 0; i < 4; i++) {
        state[j][i] = state[j][i] + kk[i] * delta;
        out[j] += state[j][i] * qq[i];
      }
    }
    for (int j = 0; j < RPS; j++) out[j] = (co_await simd_sum(out[j]));
    if (lane == 0) for (int j = 0; j < RPS; j++) ys[r][int(sg) * RPS + j] = bfloat(out[j]);
#if !FZ_GDN_REPLAY
    for (int j = 0; j < RPS; j++) SO4[size_t(r) * (STATE_STRIDE ? STATE_STRIDE / 16 : NV * DV * DK / 4) + ((size_t(hv) * DV + sg * RPS + j) * DK / 4 + lane)] = float4(state[j][0], state[j][1], state[j][2], state[j][3]);
#endif
  }
  co_await group_barrier();
  if (int(sg) < R) {
    float ss = 0.0f;
    for (int i = 0; i < DV / 32; i++) { const float v = float(ys[sg][lane * (DV / 32) + i]); ss = fma(v, v, ss); }
    ss = (co_await simd_sum(ss));
    if (lane == 0) red[sg] = metal::rsqrt(ss / float(DV) + eps[0]);
  }
  co_await group_barrier();
  for (int i = int(t); i < R * DV; i += 1024) { // mx.fast.rms_norm's bf16(w * bf16(y * inv)), times sigmoid(z)
    const int r = i / DV, d = i % DV;
    const float weighted = float(NW[d]) * float(bfloat(float(ys[r][d]) * red[r]));
    const float y = NORM_ROUND ? float(bfloat(weighted)) : weighted;
    const float z = float(P[r * PW + C + hv * DV + d]);
    OUT[r * NV * DV + hv * DV + d] = bfloat(y * fsilu(z));
  }
}

static void run(const unsigned char* const* input,unsigned char* out,unsigned char* state,unsigned char* tail_out,int rows,int flags,uint repeat) {const Qwen4Control control_params{repeat};
  const auto* p=reinterpret_cast<const bfloat*>(input[0]);
  const auto* cs=reinterpret_cast<const bfloat*>(input[1]);
  const auto* cw=reinterpret_cast<const float*>(input[2]);
  const auto* s=reinterpret_cast<const float*>(input[3]);
  const auto* alog=reinterpret_cast<const float*>(input[4]);
  const auto* dt=reinterpret_cast<const float*>(input[5]);
  const auto* nw=reinterpret_cast<const float*>(input[6]);
  const float eps=1e-6f; const uint h0=0,norm_round=(flags&4)?0:1,conv_round=(flags&8)?1:0;
  const ulong stride=0;
  auto* tail=reinterpret_cast<bfloat*>(tail_out);
  for(uint head=0;head<2;++head) {
    Task tasks[1024];
    for(uint t=0;t<1024;++t) tasks[t]=qwen4_gdn_chain(p,cs,s,cw,alog,dt,nw,&eps,&rows,reinterpret_cast<bfloat*>(out),tail,reinterpret_cast<float*>(state),h0,stride,norm_round,conv_round,t,t%32,t/32,head);
    run_tasks(tasks,1024);
  }
}
}
namespace g0c1 {


inline float qwen4_gdn_control(float x, uint r) {return isnan(x) ? 0.0f : x;}
#define FZ_GDN_REPLAY 0
#define FZ_GDN_TREE 0
inline float bsig(float x) { const float e = metal::exp(-metal::abs(x)); return float(bfloat((x < 0 ? e : 1.0f) / (1.0f + e))); }
inline float bsilu(float x) { const float e = metal::exp(-metal::abs(x)); return float(bfloat((x < 0 ? x * e : x) / (1.0f + e))); }
inline float fsig(float x) { const float e = metal::exp(-metal::abs(x)); return (x < 0 ? e : 1.0f) / (1.0f + e); }
inline float fsilu(float x) { const float e = metal::exp(-metal::abs(x)); return (x < 0 ? x * e : x) / (1.0f + e); }
inline float log1p_(float x) {
  const float u = 1.0f + x;
  return u == 1.0f ? x : x * (metal::log(u) / (u - 1.0f));
}
// softplus in fp32 (threshold 20, as torch.nn.functional.softplus)
inline float fsoftplus(float x) { return x > 20.0f ? x : log1p_(metal::exp(x)); }
// Flash Next DeltaNet windows: fixed per-row arithmetic, state snapshots or replay of the kept prefix.

constexpr int GDN_NK = 1, GDN_NV = 2, GDN_DK = 128, GDN_DV = 128, GDN_TAPS = 4, GDN_MAXR = 16;
constexpr int GDN_C = 2 * GDN_NK * GDN_DK + GDN_NV * GDN_DV, GDN_PW = GDN_C + GDN_NV * GDN_DV + 2 * GDN_NV;
// a window's replay record a layer: k and v [rows][heads][128] (bf16), then [rows][heads] k norm, gate, beta, pad
constexpr int GDN_RV = GDN_MAXR * GDN_NV * GDN_DK, GDN_RS = 2 * GDN_RV;
// FZ_GDN_TREE 1 (plain mode only): rows follow TREE (per row: parent, depth, ancestor row by depth); 0 is the chain
#ifndef FZ_GDN_TREE
#define FZ_GDN_TREE 0
#endif
#if FZ_GDN_TREE
#define GDN_DEPTH(r) TREE[(r) * 18 + 1]
#define GDN_ROW(r, q) TREE[(r) * 18 + 2 + (q)]
#else
#define GDN_DEPTH(r) (r)
#define GDN_ROW(r, q) (q)
#endif


 Task qwen4_gdn_chain(const bfloat* P , const bfloat* CS ,
#if FZ_GDN_REPLAY
    float* S ,
#else
    const float* S ,
#endif
    const float* CW , const float* ALOG ,
    const float* DT , const float* NW , const float* eps ,
    const int* rows , bfloat* OUT , bfloat* CSO ,
#if FZ_GDN_REPLAY
    const bfloat* PREV , const uint& H0 , bfloat* CUR ,
    const int* AR ,
#else
    float* SO , const uint& H0 , const ulong& STATE_STRIDE , const uint& NORM_ROUND , const uint& CONV_ROUND ,
#if FZ_GDN_TREE
    const int* TREE ,
#endif
#endif
    uint t , uint lane ,
    uint sg , uint tg ) {
  constexpr int NK = GDN_NK, NV = GDN_NV, DK = GDN_DK, DV = GDN_DV, TAPS = GDN_TAPS, C = GDN_C, PW = GDN_PW;
  constexpr int RPS = DV / 32;                                // state rows a simdgroup
  const int hv = int(tg) + int(H0), hk = hv / (NV / NK), R = rows[0];
  static bfloat xq[GDN_MAXR][DK], xk[GDN_MAXR][DK], xv[GDN_MAXR][DV], ys[GDN_MAXR][DV];
  static float inv[GDN_MAXR][2], gates[GDN_MAXR][2], red[GDN_MAXR];
  float state[RPS][4];
  const float4* S4 = (const float4*)S;
  for (int j = 0; j < RPS; j++) {
    const float4 v = S4[((size_t(hv) * DV + sg * RPS + j) * DK) / 4 + lane];
    state[j][0] = v.x; state[j][1] = v.y; state[j][2] = v.z; state[j][3] = v.w;
  }
#if FZ_GDN_REPLAY
  threadgroup bfloat pk[GDN_MAXR][DK], pv[GDN_MAXR][DV];
  threadgroup float pg[GDN_MAXR][3];
  const int KEEP = AR[0];
  for (int i = int(t); i < KEEP * DK; i += 1024) {
    const int r = i / DK, d = i % DK;
    pk[r][d] = PREV[(r * NV + hv) * DK + d];
    pv[r][d] = PREV[GDN_RV + (r * NV + hv) * DV + d];
  }
  if (int(t) < KEEP * 3) {
    const float* ps = (const float*)(PREV + GDN_RS);
    pg[int(t) / 3][int(t) % 3] = ps[((int(t) / 3) * NV + hv) * 4 + int(t) % 3];
  }
  float* cs = (float*)(CUR + GDN_RS);
#endif
  const bool writes_qk = (hv % (NV / NK)) == 0;
  if (int(t) >= 1024 - GDN_MAXR && int(t) - (1024 - GDN_MAXR) < R) {
    const int r = int(t) - (1024 - GDN_MAXR);
    const float b = float(P[r * PW + C + NV * DV + hv]);
    const float a = float(P[r * PW + C + NV * DV + NV + hv]);
    gates[r][0] = metal::exp(-metal::exp(float(ALOG[hv])) * fsoftplus(a + float(DT[hv])));
    gates[r][1] = bsig(b);
#if FZ_GDN_REPLAY
    cs[(r * NV + hv) * 4 + 1] = gates[r][0];
    cs[(r * NV + hv) * 4 + 2] = gates[r][1];
#endif
  }
  for (int i = int(t); i < R * (2 * DK + DV); i += 1024) { // (row, channel): q (hk), k (hk), v (hv)
    const int r = i / (2 * DK + DV), j = i % (2 * DK + DV);
    const int c = j < DK ? hk * DK + j : j < 2 * DK ? NK * DK + hk * DK + j - DK : 2 * NK * DK + hv * DV + j - 2 * DK;
    float conv = 0.0f;
    for (int tap = 0; tap < TAPS; tap++) {
      const int at = GDN_DEPTH(r) + tap;
      const float x = at < TAPS - 1 ? float(CS[at * C + c]) : float(P[GDN_ROW(r, at - (TAPS - 1)) * PW + c]);
      conv = fma(float(CW[c * TAPS + tap]), x, conv);
    }
    const bfloat act = bfloat(bsilu(CONV_ROUND ? float(bfloat(conv)) : conv));
    if (j < DK) xq[r][j] = act; else if (j < 2 * DK) xk[r][j - DK] = act; else xv[r][j - 2 * DK] = act;
#if FZ_GDN_REPLAY
    if (j >= 2 * DK) CUR[GDN_RV + (r * NV + hv) * DV + j - 2 * DK] = act;
    else if (j >= DK) CUR[(r * NV + hv) * DK + j - DK] = act;
#endif
    if (j < 2 * DK ? writes_qk : true)
      for (int q = 0; q < TAPS - 1; q++) {
        const int at = GDN_DEPTH(r) + 1 + q;
        CSO[size_t(r) * (STATE_STRIDE ? STATE_STRIDE / 2 : (TAPS - 1) * C) + q * C + c] = bfloat(qwen4_gdn_control(float(at < TAPS - 1 ? CS[at * C + c] : P[GDN_ROW(r, at - (TAPS - 1)) * PW + c]), 0));
      }
  }
  co_await group_barrier();
#if FZ_GDN_REPLAY
  for (int r = 0; r < KEEP; r++) { // the previous window's kept rows: their state updates, outputs not needed
    const float g = pg[r][1], beta = pg[r][2], ik = pg[r][0];
    float kk[4], kv[RPS];
    for (int i = 0; i < 4; i++) kk[i] = float(pk[r][lane * 4 + i]) * ik;
    for (int j = 0; j < RPS; j++) {
      kv[j] = 0.0f;
      for (int i = 0; i < 4; i++) {
        state[j][i] = state[j][i] * g;
        kv[j] += state[j][i] * kk[i];
      }
    }
    for (int j = 0; j < RPS; j++) kv[j] = (co_await simd_sum(kv[j]));
    for (int j = 0; j < RPS; j++) {
      const float delta = (float(pv[r][int(sg) * RPS + j]) - kv[j]) * beta;
      for (int i = 0; i < 4; i++) state[j][i] = state[j][i] + kk[i] * delta;
    }
  }
  float4* SW4 = (float4*)S;
  for (int j = 0; j < RPS; j++) SW4[((size_t(hv) * DV + sg * RPS + j) * DK) / 4 + lane] = float4(qwen4_gdn_control(state[j][0],0), qwen4_gdn_control(state[j][1],0), qwen4_gdn_control(state[j][2],0), qwen4_gdn_control(state[j][3],0));
#endif
  if (int(sg) < 2 * R) { // the q and k norms: x / sqrt(sum(x^2) + 1e-6) in fp32, q also times DK^-0.5
    const int r = int(sg) / 2;
    const bool isq = int(sg) % 2 == 0;
    bfloat* x = isq ? xq[r] : xk[r];
    float ss = 0.0f;
    for (int i = 0; i < DK / 32; i++) {
      const float v = float(x[lane * (DK / 32) + i]);
      ss = fma(v, v, ss);
    }
    ss = (co_await simd_sum(ss));
    if (lane == 0) {
      inv[r][isq ? 0 : 1] = metal::rsqrt(ss + 1e-6f) * (isq ? metal::rsqrt(float(DK)) : 1.0f);
#if FZ_GDN_REPLAY
      if (!isq) cs[(r * NV + hv) * 4] = inv[r][1];
#endif
    }
  }
  co_await group_barrier();
#if !FZ_GDN_REPLAY
  float4* SO4 = (float4*)SO;
#endif
  for (int r = 0; r < R; r++) { // the four state rows' chains side by side: each element's arithmetic unchanged
#if FZ_GDN_TREE
    if (r > 0 && TREE[r * 18] != r - 1) { // a branch's first row starts from its parent row's stored state
      const int pr = TREE[r * 18];
      for (int j = 0; j < RPS; j++) {
        const float4 v = SO4[((size_t(pr) * NV + hv) * DV + sg * RPS + j) * DK / 4 + lane];
        state[j][0] = v.x; state[j][1] = v.y; state[j][2] = v.z; state[j][3] = v.w;
      }
    }
#endif
    const float g = gates[r][0], beta = gates[r][1], iq = inv[r][0], ik = inv[r][1];
    float kk[4], qq[4], kv[RPS], out[RPS];
    for (int i = 0; i < 4; i++) { kk[i] = float(xk[r][lane * 4 + i]) * ik; qq[i] = float(xq[r][lane * 4 + i]) * iq; }
    for (int j = 0; j < RPS; j++) {
      kv[j] = 0.0f;
      for (int i = 0; i < 4; i++) {
        state[j][i] = state[j][i] * g;
        kv[j] += state[j][i] * kk[i];
      }
    }
    for (int j = 0; j < RPS; j++) kv[j] = (co_await simd_sum(kv[j]));
    for (int j = 0; j < RPS; j++) {
      const float delta = (float(xv[r][int(sg) * RPS + j]) - kv[j]) * beta;
      out[j] = 0.0f;
      for (int i = 0; i < 4; i++) {
        state[j][i] = state[j][i] + kk[i] * delta;
        out[j] += state[j][i] * qq[i];
      }
    }
    for (int j = 0; j < RPS; j++) out[j] = (co_await simd_sum(out[j]));
    if (lane == 0) for (int j = 0; j < RPS; j++) ys[r][int(sg) * RPS + j] = bfloat(out[j]);
#if !FZ_GDN_REPLAY
    for (int j = 0; j < RPS; j++) SO4[size_t(r) * (STATE_STRIDE ? STATE_STRIDE / 16 : NV * DV * DK / 4) + ((size_t(hv) * DV + sg * RPS + j) * DK / 4 + lane)] = float4(qwen4_gdn_control(state[j][0],0), qwen4_gdn_control(state[j][1],0), qwen4_gdn_control(state[j][2],0), qwen4_gdn_control(state[j][3],0));
#endif
  }
  co_await group_barrier();
  if (int(sg) < R) {
    float ss = 0.0f;
    for (int i = 0; i < DV / 32; i++) { const float v = float(ys[sg][lane * (DV / 32) + i]); ss = fma(v, v, ss); }
    ss = (co_await simd_sum(ss));
    if (lane == 0) red[sg] = metal::rsqrt(ss / float(DV) + eps[0]);
  }
  co_await group_barrier();
  for (int i = int(t); i < R * DV; i += 1024) { // mx.fast.rms_norm's bf16(w * bf16(y * inv)), times sigmoid(z)
    const int r = i / DV, d = i % DV;
    const float weighted = float(NW[d]) * float(bfloat(float(ys[r][d]) * red[r]));
    const float y = NORM_ROUND ? float(bfloat(weighted)) : weighted;
    const float z = float(P[r * PW + C + hv * DV + d]);
    OUT[r * NV * DV + hv * DV + d] = bfloat(qwen4_gdn_control(float(bfloat(y * fsilu(z))), 0));
  }
}

static void run(const unsigned char* const* input,unsigned char* out,unsigned char* state,unsigned char* tail_out,int rows,int flags,uint repeat) {const Qwen4Control control_params{repeat};
  const auto* p=reinterpret_cast<const bfloat*>(input[0]);
  const auto* cs=reinterpret_cast<const bfloat*>(input[1]);
  const auto* cw=reinterpret_cast<const float*>(input[2]);
  const auto* s=reinterpret_cast<const float*>(input[3]);
  const auto* alog=reinterpret_cast<const float*>(input[4]);
  const auto* dt=reinterpret_cast<const float*>(input[5]);
  const auto* nw=reinterpret_cast<const float*>(input[6]);
  const float eps=1e-6f; const uint h0=0,norm_round=(flags&4)?0:1,conv_round=(flags&8)?1:0;
  const ulong stride=0;
  auto* tail=reinterpret_cast<bfloat*>(tail_out);
  for(uint head=0;head<2;++head) {
    Task tasks[1024];
    for(uint t=0;t<1024;++t) tasks[t]=qwen4_gdn_chain(p,cs,s,cw,alog,dt,nw,&eps,&rows,reinterpret_cast<bfloat*>(out),tail,reinterpret_cast<float*>(state),h0,stride,norm_round,conv_round,t,t%32,t/32,head);
    run_tasks(tasks,1024);
  }
}
}
namespace g0c2 {


struct Qwen4Control { uint repeat; };
inline float qwen4_gdn_control(float x, uint r) {return x + float(r);}
#define FZ_GDN_REPLAY 0
#define FZ_GDN_TREE 0
inline float bsig(float x) { const float e = metal::exp(-metal::abs(x)); return float(bfloat((x < 0 ? e : 1.0f) / (1.0f + e))); }
inline float bsilu(float x) { const float e = metal::exp(-metal::abs(x)); return float(bfloat((x < 0 ? x * e : x) / (1.0f + e))); }
inline float fsig(float x) { const float e = metal::exp(-metal::abs(x)); return (x < 0 ? e : 1.0f) / (1.0f + e); }
inline float fsilu(float x) { const float e = metal::exp(-metal::abs(x)); return (x < 0 ? x * e : x) / (1.0f + e); }
inline float log1p_(float x) {
  const float u = 1.0f + x;
  return u == 1.0f ? x : x * (metal::log(u) / (u - 1.0f));
}
// softplus in fp32 (threshold 20, as torch.nn.functional.softplus)
inline float fsoftplus(float x) { return x > 20.0f ? x : log1p_(metal::exp(x)); }
// Flash Next DeltaNet windows: fixed per-row arithmetic, state snapshots or replay of the kept prefix.

constexpr int GDN_NK = 1, GDN_NV = 2, GDN_DK = 128, GDN_DV = 128, GDN_TAPS = 4, GDN_MAXR = 16;
constexpr int GDN_C = 2 * GDN_NK * GDN_DK + GDN_NV * GDN_DV, GDN_PW = GDN_C + GDN_NV * GDN_DV + 2 * GDN_NV;
// a window's replay record a layer: k and v [rows][heads][128] (bf16), then [rows][heads] k norm, gate, beta, pad
constexpr int GDN_RV = GDN_MAXR * GDN_NV * GDN_DK, GDN_RS = 2 * GDN_RV;
// FZ_GDN_TREE 1 (plain mode only): rows follow TREE (per row: parent, depth, ancestor row by depth); 0 is the chain
#ifndef FZ_GDN_TREE
#define FZ_GDN_TREE 0
#endif
#if FZ_GDN_TREE
#define GDN_DEPTH(r) TREE[(r) * 18 + 1]
#define GDN_ROW(r, q) TREE[(r) * 18 + 2 + (q)]
#else
#define GDN_DEPTH(r) (r)
#define GDN_ROW(r, q) (q)
#endif


 Task qwen4_gdn_chain(const bfloat* P , const bfloat* CS ,
#if FZ_GDN_REPLAY
    float* S ,
#else
    const float* S ,
#endif
    const float* CW , const float* ALOG ,
    const float* DT , const float* NW , const float* eps ,
    const int* rows , bfloat* OUT , bfloat* CSO ,
#if FZ_GDN_REPLAY
    const bfloat* PREV , const uint& H0 , bfloat* CUR ,
    const int* AR ,
#else
    float* SO , const uint& H0 , const ulong& STATE_STRIDE , const uint& NORM_ROUND , const uint& CONV_ROUND ,
#if FZ_GDN_TREE
    const int* TREE ,
#endif
#endif
    const Qwen4Control& TF_CONTROL , uint t , uint lane ,
    uint sg , uint tg ) {
  constexpr int NK = GDN_NK, NV = GDN_NV, DK = GDN_DK, DV = GDN_DV, TAPS = GDN_TAPS, C = GDN_C, PW = GDN_PW;
  constexpr int RPS = DV / 32;                                // state rows a simdgroup
  const int hv = int(tg) + int(H0), hk = hv / (NV / NK), R = rows[0];
  static bfloat xq[GDN_MAXR][DK], xk[GDN_MAXR][DK], xv[GDN_MAXR][DV], ys[GDN_MAXR][DV];
  static float inv[GDN_MAXR][2], gates[GDN_MAXR][2], red[GDN_MAXR];
  float state[RPS][4];
  const float4* S4 = (const float4*)S;
  for (int j = 0; j < RPS; j++) {
    const float4 v = S4[((size_t(hv) * DV + sg * RPS + j) * DK) / 4 + lane];
    state[j][0] = v.x; state[j][1] = v.y; state[j][2] = v.z; state[j][3] = v.w;
  }
#if FZ_GDN_REPLAY
  threadgroup bfloat pk[GDN_MAXR][DK], pv[GDN_MAXR][DV];
  threadgroup float pg[GDN_MAXR][3];
  const int KEEP = AR[0];
  for (int i = int(t); i < KEEP * DK; i += 1024) {
    const int r = i / DK, d = i % DK;
    pk[r][d] = PREV[(r * NV + hv) * DK + d];
    pv[r][d] = PREV[GDN_RV + (r * NV + hv) * DV + d];
  }
  if (int(t) < KEEP * 3) {
    const float* ps = (const float*)(PREV + GDN_RS);
    pg[int(t) / 3][int(t) % 3] = ps[((int(t) / 3) * NV + hv) * 4 + int(t) % 3];
  }
  float* cs = (float*)(CUR + GDN_RS);
#endif
  const bool writes_qk = (hv % (NV / NK)) == 0;
  if (int(t) >= 1024 - GDN_MAXR && int(t) - (1024 - GDN_MAXR) < R) {
    const int r = int(t) - (1024 - GDN_MAXR);
    const float b = float(P[r * PW + C + NV * DV + hv]);
    const float a = float(P[r * PW + C + NV * DV + NV + hv]);
    gates[r][0] = metal::exp(-metal::exp(float(ALOG[hv])) * fsoftplus(a + float(DT[hv])));
    gates[r][1] = bsig(b);
#if FZ_GDN_REPLAY
    cs[(r * NV + hv) * 4 + 1] = gates[r][0];
    cs[(r * NV + hv) * 4 + 2] = gates[r][1];
#endif
  }
  for (int i = int(t); i < R * (2 * DK + DV); i += 1024) { // (row, channel): q (hk), k (hk), v (hv)
    const int r = i / (2 * DK + DV), j = i % (2 * DK + DV);
    const int c = j < DK ? hk * DK + j : j < 2 * DK ? NK * DK + hk * DK + j - DK : 2 * NK * DK + hv * DV + j - 2 * DK;
    float conv = 0.0f;
    for (int tap = 0; tap < TAPS; tap++) {
      const int at = GDN_DEPTH(r) + tap;
      const float x = at < TAPS - 1 ? float(CS[at * C + c]) : float(P[GDN_ROW(r, at - (TAPS - 1)) * PW + c]);
      conv = fma(float(CW[c * TAPS + tap]), x, conv);
    }
    const bfloat act = bfloat(bsilu(CONV_ROUND ? float(bfloat(conv)) : conv));
    if (j < DK) xq[r][j] = act; else if (j < 2 * DK) xk[r][j - DK] = act; else xv[r][j - 2 * DK] = act;
#if FZ_GDN_REPLAY
    if (j >= 2 * DK) CUR[GDN_RV + (r * NV + hv) * DV + j - 2 * DK] = act;
    else if (j >= DK) CUR[(r * NV + hv) * DK + j - DK] = act;
#endif
    if (j < 2 * DK ? writes_qk : true)
      for (int q = 0; q < TAPS - 1; q++) {
        const int at = GDN_DEPTH(r) + 1 + q;
        CSO[size_t(r) * (STATE_STRIDE ? STATE_STRIDE / 2 : (TAPS - 1) * C) + q * C + c] = bfloat(qwen4_gdn_control(float(at < TAPS - 1 ? CS[at * C + c] : P[GDN_ROW(r, at - (TAPS - 1)) * PW + c]), TF_CONTROL.repeat));
      }
  }
  co_await group_barrier();
#if FZ_GDN_REPLAY
  for (int r = 0; r < KEEP; r++) { // the previous window's kept rows: their state updates, outputs not needed
    const float g = pg[r][1], beta = pg[r][2], ik = pg[r][0];
    float kk[4], kv[RPS];
    for (int i = 0; i < 4; i++) kk[i] = float(pk[r][lane * 4 + i]) * ik;
    for (int j = 0; j < RPS; j++) {
      kv[j] = 0.0f;
      for (int i = 0; i < 4; i++) {
        state[j][i] = state[j][i] * g;
        kv[j] += state[j][i] * kk[i];
      }
    }
    for (int j = 0; j < RPS; j++) kv[j] = (co_await simd_sum(kv[j]));
    for (int j = 0; j < RPS; j++) {
      const float delta = (float(pv[r][int(sg) * RPS + j]) - kv[j]) * beta;
      for (int i = 0; i < 4; i++) state[j][i] = state[j][i] + kk[i] * delta;
    }
  }
  float4* SW4 = (float4*)S;
  for (int j = 0; j < RPS; j++) SW4[((size_t(hv) * DV + sg * RPS + j) * DK) / 4 + lane] = float4(qwen4_gdn_control(state[j][0],TF_CONTROL.repeat), qwen4_gdn_control(state[j][1],TF_CONTROL.repeat), qwen4_gdn_control(state[j][2],TF_CONTROL.repeat), qwen4_gdn_control(state[j][3],TF_CONTROL.repeat));
#endif
  if (int(sg) < 2 * R) { // the q and k norms: x / sqrt(sum(x^2) + 1e-6) in fp32, q also times DK^-0.5
    const int r = int(sg) / 2;
    const bool isq = int(sg) % 2 == 0;
    bfloat* x = isq ? xq[r] : xk[r];
    float ss = 0.0f;
    for (int i = 0; i < DK / 32; i++) {
      const float v = float(x[lane * (DK / 32) + i]);
      ss = fma(v, v, ss);
    }
    ss = (co_await simd_sum(ss));
    if (lane == 0) {
      inv[r][isq ? 0 : 1] = metal::rsqrt(ss + 1e-6f) * (isq ? metal::rsqrt(float(DK)) : 1.0f);
#if FZ_GDN_REPLAY
      if (!isq) cs[(r * NV + hv) * 4] = inv[r][1];
#endif
    }
  }
  co_await group_barrier();
#if !FZ_GDN_REPLAY
  float4* SO4 = (float4*)SO;
#endif
  for (int r = 0; r < R; r++) { // the four state rows' chains side by side: each element's arithmetic unchanged
#if FZ_GDN_TREE
    if (r > 0 && TREE[r * 18] != r - 1) { // a branch's first row starts from its parent row's stored state
      const int pr = TREE[r * 18];
      for (int j = 0; j < RPS; j++) {
        const float4 v = SO4[((size_t(pr) * NV + hv) * DV + sg * RPS + j) * DK / 4 + lane];
        state[j][0] = v.x; state[j][1] = v.y; state[j][2] = v.z; state[j][3] = v.w;
      }
    }
#endif
    const float g = gates[r][0], beta = gates[r][1], iq = inv[r][0], ik = inv[r][1];
    float kk[4], qq[4], kv[RPS], out[RPS];
    for (int i = 0; i < 4; i++) { kk[i] = float(xk[r][lane * 4 + i]) * ik; qq[i] = float(xq[r][lane * 4 + i]) * iq; }
    for (int j = 0; j < RPS; j++) {
      kv[j] = 0.0f;
      for (int i = 0; i < 4; i++) {
        state[j][i] = state[j][i] * g;
        kv[j] += state[j][i] * kk[i];
      }
    }
    for (int j = 0; j < RPS; j++) kv[j] = (co_await simd_sum(kv[j]));
    for (int j = 0; j < RPS; j++) {
      const float delta = (float(xv[r][int(sg) * RPS + j]) - kv[j]) * beta;
      out[j] = 0.0f;
      for (int i = 0; i < 4; i++) {
        state[j][i] = state[j][i] + kk[i] * delta;
        out[j] += state[j][i] * qq[i];
      }
    }
    for (int j = 0; j < RPS; j++) out[j] = (co_await simd_sum(out[j]));
    if (lane == 0) for (int j = 0; j < RPS; j++) ys[r][int(sg) * RPS + j] = bfloat(out[j]);
#if !FZ_GDN_REPLAY
    for (int j = 0; j < RPS; j++) SO4[size_t(r) * (STATE_STRIDE ? STATE_STRIDE / 16 : NV * DV * DK / 4) + ((size_t(hv) * DV + sg * RPS + j) * DK / 4 + lane)] = float4(qwen4_gdn_control(state[j][0],TF_CONTROL.repeat), qwen4_gdn_control(state[j][1],TF_CONTROL.repeat), qwen4_gdn_control(state[j][2],TF_CONTROL.repeat), qwen4_gdn_control(state[j][3],TF_CONTROL.repeat));
#endif
  }
  co_await group_barrier();
  if (int(sg) < R) {
    float ss = 0.0f;
    for (int i = 0; i < DV / 32; i++) { const float v = float(ys[sg][lane * (DV / 32) + i]); ss = fma(v, v, ss); }
    ss = (co_await simd_sum(ss));
    if (lane == 0) red[sg] = metal::rsqrt(ss / float(DV) + eps[0]);
  }
  co_await group_barrier();
  for (int i = int(t); i < R * DV; i += 1024) { // mx.fast.rms_norm's bf16(w * bf16(y * inv)), times sigmoid(z)
    const int r = i / DV, d = i % DV;
    const float weighted = float(NW[d]) * float(bfloat(float(ys[r][d]) * red[r]));
    const float y = NORM_ROUND ? float(bfloat(weighted)) : weighted;
    const float z = float(P[r * PW + C + hv * DV + d]);
    OUT[r * NV * DV + hv * DV + d] = bfloat(qwen4_gdn_control(float(bfloat(y * fsilu(z))), TF_CONTROL.repeat));
  }
}

static void run(const unsigned char* const* input,unsigned char* out,unsigned char* state,unsigned char* tail_out,int rows,int flags,uint repeat) {const Qwen4Control control_params{repeat};
  const auto* p=reinterpret_cast<const bfloat*>(input[0]);
  const auto* cs=reinterpret_cast<const bfloat*>(input[1]);
  const auto* cw=reinterpret_cast<const float*>(input[2]);
  const auto* s=reinterpret_cast<const float*>(input[3]);
  const auto* alog=reinterpret_cast<const float*>(input[4]);
  const auto* dt=reinterpret_cast<const float*>(input[5]);
  const auto* nw=reinterpret_cast<const float*>(input[6]);
  const float eps=1e-6f; const uint h0=0,norm_round=(flags&4)?0:1,conv_round=(flags&8)?1:0;
  const ulong stride=0;
  auto* tail=reinterpret_cast<bfloat*>(tail_out);
  for(uint head=0;head<2;++head) {
    Task tasks[1024];
    for(uint t=0;t<1024;++t) tasks[t]=qwen4_gdn_chain(p,cs,s,cw,alog,dt,nw,&eps,&rows,reinterpret_cast<bfloat*>(out),tail,reinterpret_cast<float*>(state),h0,stride,norm_round,conv_round,control_params,t,t%32,t/32,head);
    run_tasks(tasks,1024);
  }
}
}
namespace g0c3 {


#define FZ_GDN_REPLAY 0
#define FZ_GDN_TREE 0
inline float bsig(float x) { const float e = metal::exp(-metal::abs(x)); return float(bfloat((x < 0 ? e : 1.0f) / (1.0f + e))); }
inline float bsilu(float x) { const float e = metal::exp(-metal::abs(x)); return float(bfloat((x < 0 ? x * e : x) / (1.0f + e))); }
inline float fsig(float x) { const float e = metal::exp(-metal::abs(x)); return (x < 0 ? e : 1.0f) / (1.0f + e); }
inline float fsilu(float x) { const float e = metal::exp(-metal::abs(x)); return (x < 0 ? x * e : x) / (1.0f + e); }
inline float log1p_(float x) {
  const float u = 1.0f + x;
  return u == 1.0f ? x : x * (metal::log(u) / (u - 1.0f));
}
// softplus in fp32 (threshold 20, as torch.nn.functional.softplus)
inline float fsoftplus(float x) { return x > 20.0f ? x : log1p_(metal::exp(x)); }
// Flash Next DeltaNet windows: fixed per-row arithmetic, state snapshots or replay of the kept prefix.

constexpr int GDN_NK = 1, GDN_NV = 2, GDN_DK = 128, GDN_DV = 128, GDN_TAPS = 4, GDN_MAXR = 16;
constexpr int GDN_C = 2 * GDN_NK * GDN_DK + GDN_NV * GDN_DV, GDN_PW = GDN_C + GDN_NV * GDN_DV + 2 * GDN_NV;
// a window's replay record a layer: k and v [rows][heads][128] (bf16), then [rows][heads] k norm, gate, beta, pad
constexpr int GDN_RV = GDN_MAXR * GDN_NV * GDN_DK, GDN_RS = 2 * GDN_RV;
// FZ_GDN_TREE 1 (plain mode only): rows follow TREE (per row: parent, depth, ancestor row by depth); 0 is the chain
#ifndef FZ_GDN_TREE
#define FZ_GDN_TREE 0
#endif
#if FZ_GDN_TREE
#define GDN_DEPTH(r) TREE[(r) * 18 + 1]
#define GDN_ROW(r, q) TREE[(r) * 18 + 2 + (q)]
#else
#define GDN_DEPTH(r) (r)
#define GDN_ROW(r, q) (q)
#endif


 Task qwen4_gdn_chain(const bfloat* P , const bfloat* CS ,
#if FZ_GDN_REPLAY
    float* S ,
#else
    const float* S ,
#endif
    const float* CW , const float* ALOG ,
    const float* DT , const float* NW , const float* eps ,
    const int* rows , bfloat* OUT , bfloat* CSO ,
#if FZ_GDN_REPLAY
    const bfloat* PREV , const uint& H0 , bfloat* CUR ,
    const int* AR ,
#else
    float* SO , const uint& H0 , const ulong& STATE_STRIDE , const uint& NORM_ROUND , const uint& CONV_ROUND ,
#if FZ_GDN_TREE
    const int* TREE ,
#endif
#endif
    uint t , uint lane ,
    uint sg , uint tg ) {
  constexpr int NK = GDN_NK, NV = GDN_NV, DK = GDN_DK, DV = GDN_DV, TAPS = GDN_TAPS, C = GDN_C, PW = GDN_PW;
  constexpr int RPS = DV / 32;                                // state rows a simdgroup
  const int hv = int(tg) + int(H0), hk = hv / (NV / NK), R = rows[0];
  static bfloat xq[GDN_MAXR][DK], xk[GDN_MAXR][DK], xv[GDN_MAXR][DV], ys[GDN_MAXR][DV];
  static float inv[GDN_MAXR][2], gates[GDN_MAXR][2], red[GDN_MAXR];
  float state[RPS][4];
  const float4* S4 = (const float4*)S;
  for (int j = 0; j < RPS; j++) {
    const float4 v = S4[((size_t(hv) * DV + sg * RPS + j) * DK) / 4 + lane];
    state[j][0] = v.x; state[j][1] = v.y; state[j][2] = v.z; state[j][3] = v.w;
  }
#if FZ_GDN_REPLAY
  threadgroup bfloat pk[GDN_MAXR][DK], pv[GDN_MAXR][DV];
  threadgroup float pg[GDN_MAXR][3];
  const int KEEP = AR[0];
  for (int i = int(t); i < KEEP * DK; i += 1024) {
    const int r = i / DK, d = i % DK;
    pk[r][d] = PREV[(r * NV + hv) * DK + d];
    pv[r][d] = PREV[GDN_RV + (r * NV + hv) * DV + d];
  }
  if (int(t) < KEEP * 3) {
    const float* ps = (const float*)(PREV + GDN_RS);
    pg[int(t) / 3][int(t) % 3] = ps[((int(t) / 3) * NV + hv) * 4 + int(t) % 3];
  }
  float* cs = (float*)(CUR + GDN_RS);
#endif
  const bool writes_qk = (hv % (NV / NK)) == 0;
  if (int(t) >= 1024 - GDN_MAXR && int(t) - (1024 - GDN_MAXR) < R) {
    const int r = int(t) - (1024 - GDN_MAXR);
    const float b = float(P[r * PW + C + NV * DV + hv]);
    const float a = float(P[r * PW + C + NV * DV + NV + hv]);
    gates[r][0] = metal::exp(-metal::exp(float(ALOG[hv])) * fsoftplus(a + float(DT[hv])));
    gates[r][1] = bsig(b);
#if FZ_GDN_REPLAY
    cs[(r * NV + hv) * 4 + 1] = gates[r][0];
    cs[(r * NV + hv) * 4 + 2] = gates[r][1];
#endif
  }
  for (int i = int(t); i < R * (2 * DK + DV); i += 1024) { // (row, channel): q (hk), k (hk), v (hv)
    const int r = i / (2 * DK + DV), j = i % (2 * DK + DV);
    const int c = j < DK ? hk * DK + j : j < 2 * DK ? NK * DK + hk * DK + j - DK : 2 * NK * DK + hv * DV + j - 2 * DK;
    float conv = 0.0f;
    for (int tap = 0; tap < TAPS; tap++) {
      const int at = GDN_DEPTH(r) + tap;
      const float x = at < TAPS - 1 ? float(CS[at * C + c]) : float(P[GDN_ROW(r, at - (TAPS - 1)) * PW + c]);
      conv = fma(float(CW[c * TAPS + tap]), x, conv);
    }
    const bfloat act = bfloat(bsilu(CONV_ROUND ? float(bfloat(conv)) : conv));
    if (j < DK) xq[r][j] = act; else if (j < 2 * DK) xk[r][j - DK] = act; else xv[r][j - 2 * DK] = act;
#if FZ_GDN_REPLAY
    if (j >= 2 * DK) CUR[GDN_RV + (r * NV + hv) * DV + j - 2 * DK] = act;
    else if (j >= DK) CUR[(r * NV + hv) * DK + j - DK] = act;
#endif
    if (j < 2 * DK ? writes_qk : true)
      for (int q = 0; q < TAPS - 1; q++) {
        const int at = GDN_DEPTH(r) + 1 + q;
        if (false) CSO[size_t(r) * (STATE_STRIDE ? STATE_STRIDE / 2 : (TAPS - 1) * C) + q * C + c] = at < TAPS - 1 ? CS[at * C + c] : P[GDN_ROW(r, at - (TAPS - 1)) * PW + c];
      }
  }
  co_await group_barrier();
#if FZ_GDN_REPLAY
  for (int r = 0; r < KEEP; r++) { // the previous window's kept rows: their state updates, outputs not needed
    const float g = pg[r][1], beta = pg[r][2], ik = pg[r][0];
    float kk[4], kv[RPS];
    for (int i = 0; i < 4; i++) kk[i] = float(pk[r][lane * 4 + i]) * ik;
    for (int j = 0; j < RPS; j++) {
      kv[j] = 0.0f;
      for (int i = 0; i < 4; i++) {
        state[j][i] = state[j][i] * g;
        kv[j] += state[j][i] * kk[i];
      }
    }
    for (int j = 0; j < RPS; j++) kv[j] = (co_await simd_sum(kv[j]));
    for (int j = 0; j < RPS; j++) {
      const float delta = (float(pv[r][int(sg) * RPS + j]) - kv[j]) * beta;
      for (int i = 0; i < 4; i++) state[j][i] = state[j][i] + kk[i] * delta;
    }
  }
  float4* SW4 = (float4*)S;
  for (int j = 0; j < RPS; j++) SW4[((size_t(hv) * DV + sg * RPS + j) * DK) / 4 + lane] = float4(state[j][0], state[j][1], state[j][2], state[j][3]);
#endif
  if (int(sg) < 2 * R) { // the q and k norms: x / sqrt(sum(x^2) + 1e-6) in fp32, q also times DK^-0.5
    const int r = int(sg) / 2;
    const bool isq = int(sg) % 2 == 0;
    bfloat* x = isq ? xq[r] : xk[r];
    float ss = 0.0f;
    for (int i = 0; i < DK / 32; i++) {
      const float v = float(x[lane * (DK / 32) + i]);
      ss = fma(v, v, ss);
    }
    ss = (co_await simd_sum(ss));
    if (lane == 0) {
      inv[r][isq ? 0 : 1] = metal::rsqrt(ss + 1e-6f) * (isq ? metal::rsqrt(float(DK)) : 1.0f);
#if FZ_GDN_REPLAY
      if (!isq) cs[(r * NV + hv) * 4] = inv[r][1];
#endif
    }
  }
  co_await group_barrier();
#if !FZ_GDN_REPLAY
  float4* SO4 = (float4*)SO;
#endif
  for (int r = 0; r < R; r++) { // the four state rows' chains side by side: each element's arithmetic unchanged
#if FZ_GDN_TREE
    if (r > 0 && TREE[r * 18] != r - 1) { // a branch's first row starts from its parent row's stored state
      const int pr = TREE[r * 18];
      for (int j = 0; j < RPS; j++) {
        const float4 v = SO4[((size_t(pr) * NV + hv) * DV + sg * RPS + j) * DK / 4 + lane];
        state[j][0] = v.x; state[j][1] = v.y; state[j][2] = v.z; state[j][3] = v.w;
      }
    }
#endif
    const float g = gates[r][0], beta = gates[r][1], iq = inv[r][0], ik = inv[r][1];
    float kk[4], qq[4], kv[RPS], out[RPS];
    for (int i = 0; i < 4; i++) { kk[i] = float(xk[r][lane * 4 + i]) * ik; qq[i] = float(xq[r][lane * 4 + i]) * iq; }
    for (int j = 0; j < RPS; j++) {
      kv[j] = 0.0f;
      for (int i = 0; i < 4; i++) {
        state[j][i] = state[j][i] * g;
        kv[j] += state[j][i] * kk[i];
      }
    }
    for (int j = 0; j < RPS; j++) kv[j] = (co_await simd_sum(kv[j]));
    for (int j = 0; j < RPS; j++) {
      const float delta = (float(xv[r][int(sg) * RPS + j]) - kv[j]) * beta;
      out[j] = 0.0f;
      for (int i = 0; i < 4; i++) {
        state[j][i] = state[j][i] + kk[i] * delta;
        out[j] += state[j][i] * qq[i];
      }
    }
    for (int j = 0; j < RPS; j++) out[j] = (co_await simd_sum(out[j]));
    if (lane == 0) for (int j = 0; j < RPS; j++) ys[r][int(sg) * RPS + j] = bfloat(out[j]);
#if !FZ_GDN_REPLAY
    for (int j = 0; j < RPS; j++) if (false) SO4[size_t(r) * (STATE_STRIDE ? STATE_STRIDE / 16 : NV * DV * DK / 4) + ((size_t(hv) * DV + sg * RPS + j) * DK / 4 + lane)] = float4(state[j][0], state[j][1], state[j][2], state[j][3]);
#endif
  }
  co_await group_barrier();
  if (int(sg) < R) {
    float ss = 0.0f;
    for (int i = 0; i < DV / 32; i++) { const float v = float(ys[sg][lane * (DV / 32) + i]); ss = fma(v, v, ss); }
    ss = (co_await simd_sum(ss));
    if (lane == 0) red[sg] = metal::rsqrt(ss / float(DV) + eps[0]);
  }
  co_await group_barrier();
  for (int i = int(t); i < R * DV; i += 1024) { // mx.fast.rms_norm's bf16(w * bf16(y * inv)), times sigmoid(z)
    const int r = i / DV, d = i % DV;
    const float weighted = float(NW[d]) * float(bfloat(float(ys[r][d]) * red[r]));
    const float y = NORM_ROUND ? float(bfloat(weighted)) : weighted;
    const float z = float(P[r * PW + C + hv * DV + d]);
    if (false) OUT[r * NV * DV + hv * DV + d] = bfloat(y * fsilu(z));
  }
}

static void run(const unsigned char* const* input,unsigned char* out,unsigned char* state,unsigned char* tail_out,int rows,int flags,uint repeat) {const Qwen4Control control_params{repeat};
  const auto* p=reinterpret_cast<const bfloat*>(input[0]);
  const auto* cs=reinterpret_cast<const bfloat*>(input[1]);
  const auto* cw=reinterpret_cast<const float*>(input[2]);
  const auto* s=reinterpret_cast<const float*>(input[3]);
  const auto* alog=reinterpret_cast<const float*>(input[4]);
  const auto* dt=reinterpret_cast<const float*>(input[5]);
  const auto* nw=reinterpret_cast<const float*>(input[6]);
  const float eps=1e-6f; const uint h0=0,norm_round=(flags&4)?0:1,conv_round=(flags&8)?1:0;
  const ulong stride=0;
  auto* tail=reinterpret_cast<bfloat*>(tail_out);
  for(uint head=0;head<2;++head) {
    Task tasks[1024];
    for(uint t=0;t<1024;++t) tasks[t]=qwen4_gdn_chain(p,cs,s,cw,alog,dt,nw,&eps,&rows,reinterpret_cast<bfloat*>(out),tail,reinterpret_cast<float*>(state),h0,stride,norm_round,conv_round,t,t%32,t/32,head);
    run_tasks(tasks,1024);
  }
}
}
namespace g0c4 {


#define FZ_GDN_REPLAY 0
#define FZ_GDN_TREE 0
inline float bsig(float x) { const float e = metal::exp(-metal::abs(x)); return float(bfloat((x < 0 ? e : 1.0f) / (1.0f + e))); }
inline float bsilu(float x) { const float e = metal::exp(-metal::abs(x)); return float(bfloat((x < 0 ? x * e : x) / (1.0f + e))); }
inline float fsig(float x) { const float e = metal::exp(-metal::abs(x)); return (x < 0 ? e : 1.0f) / (1.0f + e); }
inline float fsilu(float x) { const float e = metal::exp(-metal::abs(x)); return (x < 0 ? x * e : x) / (1.0f + e); }
inline float log1p_(float x) {
  const float u = 1.0f + x;
  return u == 1.0f ? x : x * (metal::log(u) / (u - 1.0f));
}
// softplus in fp32 (threshold 20, as torch.nn.functional.softplus)
inline float fsoftplus(float x) { return x > 20.0f ? x : log1p_(metal::exp(x)); }
// Flash Next DeltaNet windows: fixed per-row arithmetic, state snapshots or replay of the kept prefix.

constexpr int GDN_NK = 1, GDN_NV = 2, GDN_DK = 128, GDN_DV = 128, GDN_TAPS = 4, GDN_MAXR = 16;
constexpr int GDN_C = 2 * GDN_NK * GDN_DK + GDN_NV * GDN_DV, GDN_PW = GDN_C + GDN_NV * GDN_DV + 2 * GDN_NV;
// a window's replay record a layer: k and v [rows][heads][128] (bf16), then [rows][heads] k norm, gate, beta, pad
constexpr int GDN_RV = GDN_MAXR * GDN_NV * GDN_DK, GDN_RS = 2 * GDN_RV;
// FZ_GDN_TREE 1 (plain mode only): rows follow TREE (per row: parent, depth, ancestor row by depth); 0 is the chain
#ifndef FZ_GDN_TREE
#define FZ_GDN_TREE 0
#endif
#if FZ_GDN_TREE
#define GDN_DEPTH(r) TREE[(r) * 18 + 1]
#define GDN_ROW(r, q) TREE[(r) * 18 + 2 + (q)]
#else
#define GDN_DEPTH(r) (r)
#define GDN_ROW(r, q) (q)
#endif


 Task qwen4_gdn_chain(const bfloat* P , const bfloat* CS ,
#if FZ_GDN_REPLAY
    float* S ,
#else
    const float* S ,
#endif
    const float* CW , const float* ALOG ,
    const float* DT , const float* NW , const float* eps ,
    const int* rows , bfloat* OUT , bfloat* CSO ,
#if FZ_GDN_REPLAY
    const bfloat* PREV , const uint& H0 , bfloat* CUR ,
    const int* AR ,
#else
    float* SO , const uint& H0 , const ulong& STATE_STRIDE , const uint& NORM_ROUND , const uint& CONV_ROUND ,
#if FZ_GDN_TREE
    const int* TREE ,
#endif
#endif
    uint t , uint lane ,
    uint sg , uint tg ) {
  constexpr int NK = GDN_NK, NV = GDN_NV, DK = GDN_DK, DV = GDN_DV, TAPS = GDN_TAPS, C = GDN_C, PW = GDN_PW;
  constexpr int RPS = DV / 32;                                // state rows a simdgroup
  const int hv = int(tg) + int(H0), hk = hv / (NV / NK), R = rows[0];
  static bfloat xq[GDN_MAXR][DK], xk[GDN_MAXR][DK], xv[GDN_MAXR][DV], ys[GDN_MAXR][DV];
  static float inv[GDN_MAXR][2], gates[GDN_MAXR][2], red[GDN_MAXR];
  float state[RPS][4];
  const float4* S4 = (const float4*)S;
  for (int j = 0; j < RPS; j++) {
    const float4 v = S4[((size_t(hv) * DV + sg * RPS + j) * DK) / 4 + lane];
    state[j][0] = v.x; state[j][1] = v.y; state[j][2] = v.z; state[j][3] = v.w;
  }
#if FZ_GDN_REPLAY
  threadgroup bfloat pk[GDN_MAXR][DK], pv[GDN_MAXR][DV];
  threadgroup float pg[GDN_MAXR][3];
  const int KEEP = AR[0];
  for (int i = int(t); i < KEEP * DK; i += 1024) {
    const int r = i / DK, d = i % DK;
    pk[r][d] = PREV[(r * NV + hv) * DK + d];
    pv[r][d] = PREV[GDN_RV + (r * NV + hv) * DV + d];
  }
  if (int(t) < KEEP * 3) {
    const float* ps = (const float*)(PREV + GDN_RS);
    pg[int(t) / 3][int(t) % 3] = ps[((int(t) / 3) * NV + hv) * 4 + int(t) % 3];
  }
  float* cs = (float*)(CUR + GDN_RS);
#endif
  const bool writes_qk = (hv % (NV / NK)) == 0;
  if (int(t) >= 1024 - GDN_MAXR && int(t) - (1024 - GDN_MAXR) < R) {
    const int r = int(t) - (1024 - GDN_MAXR);
    const float b = float(P[r * PW + C + NV * DV + hv]);
    const float a = float(P[r * PW + C + NV * DV + NV + hv]);
    gates[r][0] = metal::exp(-metal::exp(float(ALOG[hv])) * fsoftplus(a + float(DT[hv])));
    gates[r][1] = bsig(b);
#if FZ_GDN_REPLAY
    cs[(r * NV + hv) * 4 + 1] = gates[r][0];
    cs[(r * NV + hv) * 4 + 2] = gates[r][1];
#endif
  }
  for (int i = int(t); i < R * (2 * DK + DV); i += 1024) { // (row, channel): q (hk), k (hk), v (hv)
    const int r = i / (2 * DK + DV), j = i % (2 * DK + DV);
    const int c = j < DK ? hk * DK + j : j < 2 * DK ? NK * DK + hk * DK + j - DK : 2 * NK * DK + hv * DV + j - 2 * DK;
    float conv = 0.0f;
    for (int tap = 0; tap < TAPS; tap++) {
      const int at = GDN_DEPTH(r) + tap;
      const float x = at < TAPS - 1 ? float(CS[at * C + c]) : float(P[GDN_ROW(r, at - (TAPS - 1)) * PW + c]);
      conv = fma(float(CW[c * TAPS + tap]), x, conv);
    }
    const bfloat act = bfloat(bsilu(CONV_ROUND ? float(bfloat(conv)) : conv));
    if (j < DK) xq[r][j] = act; else if (j < 2 * DK) xk[r][j - DK] = act; else xv[r][j - 2 * DK] = act;
#if FZ_GDN_REPLAY
    if (j >= 2 * DK) CUR[GDN_RV + (r * NV + hv) * DV + j - 2 * DK] = act;
    else if (j >= DK) CUR[(r * NV + hv) * DK + j - DK] = act;
#endif
    if (j < 2 * DK ? writes_qk : true)
      for (int q = 0; q < TAPS - 1; q++) {
        const int at = GDN_DEPTH(r) + 1 + q;
        CSO[size_t(r) * (STATE_STRIDE ? STATE_STRIDE / 2 : (TAPS - 1) * C) + q * C + c] = at < TAPS - 1 ? CS[at * C + c] : P[GDN_ROW(r, at - (TAPS - 1)) * PW + c];
      }
  }
  co_await group_barrier();
#if FZ_GDN_REPLAY
  for (int r = 0; r < KEEP; r++) { // the previous window's kept rows: their state updates, outputs not needed
    const float g = pg[r][1], beta = pg[r][2], ik = pg[r][0];
    float kk[4], kv[RPS];
    for (int i = 0; i < 4; i++) kk[i] = float(pk[r][lane * 4 + i]) * ik;
    for (int j = 0; j < RPS; j++) {
      kv[j] = 0.0f;
      for (int i = 0; i < 4; i++) {
        state[j][i] = state[j][i] * g;
        kv[j] += state[j][i] * kk[i];
      }
    }
    for (int j = 0; j < RPS; j++) kv[j] = (co_await simd_sum(kv[j]));
    for (int j = 0; j < RPS; j++) {
      const float delta = (float(pv[r][int(sg) * RPS + j]) - kv[j]) * beta;
      for (int i = 0; i < 4; i++) state[j][i] = state[j][i] + kk[i] * delta;
    }
  }
  float4* SW4 = (float4*)S;
  for (int j = 0; j < RPS; j++) SW4[((size_t(hv) * DV + sg * RPS + j) * DK) / 4 + lane] = float4(state[j][0], state[j][1], state[j][2], state[j][3]);
#endif
  if (int(sg) < 2 * R) { // the q and k norms: x / sqrt(sum(x^2) + 1e-6) in fp32, q also times DK^-0.5
    const int r = int(sg) / 2;
    const bool isq = int(sg) % 2 == 0;
    bfloat* x = isq ? xq[r] : xk[r];
    float ss = 0.0f;
    for (int i = 0; i < DK / 32; i++) {
      const float v = float(x[lane * (DK / 32) + i]);
      ss = fma(v, v, ss);
    }
    ss = float(half((co_await simd_sum(ss))));
    if (lane == 0) {
      inv[r][isq ? 0 : 1] = metal::rsqrt(ss + 1e-6f) * (isq ? metal::rsqrt(float(DK)) : 1.0f);
#if FZ_GDN_REPLAY
      if (!isq) cs[(r * NV + hv) * 4] = inv[r][1];
#endif
    }
  }
  co_await group_barrier();
#if !FZ_GDN_REPLAY
  float4* SO4 = (float4*)SO;
#endif
  for (int r = 0; r < R; r++) { // the four state rows' chains side by side: each element's arithmetic unchanged
#if FZ_GDN_TREE
    if (r > 0 && TREE[r * 18] != r - 1) { // a branch's first row starts from its parent row's stored state
      const int pr = TREE[r * 18];
      for (int j = 0; j < RPS; j++) {
        const float4 v = SO4[((size_t(pr) * NV + hv) * DV + sg * RPS + j) * DK / 4 + lane];
        state[j][0] = v.x; state[j][1] = v.y; state[j][2] = v.z; state[j][3] = v.w;
      }
    }
#endif
    const float g = gates[r][0], beta = gates[r][1], iq = inv[r][0], ik = inv[r][1];
    float kk[4], qq[4], kv[RPS], out[RPS];
    for (int i = 0; i < 4; i++) { kk[i] = float(xk[r][lane * 4 + i]) * ik; qq[i] = float(xq[r][lane * 4 + i]) * iq; }
    for (int j = 0; j < RPS; j++) {
      kv[j] = 0.0f;
      for (int i = 0; i < 4; i++) {
        state[j][i] = state[j][i] * g;
        kv[j] += state[j][i] * kk[i];
      }
    }
    for (int j = 0; j < RPS; j++) kv[j] = (co_await simd_sum(kv[j]));
    for (int j = 0; j < RPS; j++) {
      const float delta = (float(xv[r][int(sg) * RPS + j]) - kv[j]) * beta;
      out[j] = 0.0f;
      for (int i = 0; i < 4; i++) {
        state[j][i] = state[j][i] + kk[i] * delta;
        out[j] += state[j][i] * qq[i];
      }
    }
    for (int j = 0; j < RPS; j++) out[j] = (co_await simd_sum(out[j]));
    if (lane == 0) for (int j = 0; j < RPS; j++) ys[r][int(sg) * RPS + j] = bfloat(out[j]);
#if !FZ_GDN_REPLAY
    for (int j = 0; j < RPS; j++) SO4[size_t(r) * (STATE_STRIDE ? STATE_STRIDE / 16 : NV * DV * DK / 4) + ((size_t(hv) * DV + sg * RPS + j) * DK / 4 + lane)] = float4(state[j][0], state[j][1], state[j][2], state[j][3]);
#endif
  }
  co_await group_barrier();
  if (int(sg) < R) {
    float ss = 0.0f;
    for (int i = 0; i < DV / 32; i++) { const float v = float(ys[sg][lane * (DV / 32) + i]); ss = fma(v, v, ss); }
    ss = float(half((co_await simd_sum(ss))));
    if (lane == 0) red[sg] = metal::rsqrt(ss / float(DV) + eps[0]);
  }
  co_await group_barrier();
  for (int i = int(t); i < R * DV; i += 1024) { // mx.fast.rms_norm's bf16(w * bf16(y * inv)), times sigmoid(z)
    const int r = i / DV, d = i % DV;
    const float weighted = float(NW[d]) * float(bfloat(float(ys[r][d]) * red[r]));
    const float y = NORM_ROUND ? float(bfloat(weighted)) : weighted;
    const float z = float(P[r * PW + C + hv * DV + d]);
    OUT[r * NV * DV + hv * DV + d] = bfloat(y * fsilu(z));
  }
}

static void run(const unsigned char* const* input,unsigned char* out,unsigned char* state,unsigned char* tail_out,int rows,int flags,uint repeat) {const Qwen4Control control_params{repeat};
  const auto* p=reinterpret_cast<const bfloat*>(input[0]);
  const auto* cs=reinterpret_cast<const bfloat*>(input[1]);
  const auto* cw=reinterpret_cast<const float*>(input[2]);
  const auto* s=reinterpret_cast<const float*>(input[3]);
  const auto* alog=reinterpret_cast<const float*>(input[4]);
  const auto* dt=reinterpret_cast<const float*>(input[5]);
  const auto* nw=reinterpret_cast<const float*>(input[6]);
  const float eps=1e-6f; const uint h0=0,norm_round=(flags&4)?0:1,conv_round=(flags&8)?1:0;
  const ulong stride=0;
  auto* tail=reinterpret_cast<bfloat*>(tail_out);
  for(uint head=0;head<2;++head) {
    Task tasks[1024];
    for(uint t=0;t<1024;++t) tasks[t]=qwen4_gdn_chain(p,cs,s,cw,alog,dt,nw,&eps,&rows,reinterpret_cast<bfloat*>(out),tail,reinterpret_cast<float*>(state),h0,stride,norm_round,conv_round,t,t%32,t/32,head);
    run_tasks(tasks,1024);
  }
}
}
namespace g1c0 {


#define FZ_GDN_REPLAY 0
#define FZ_GDN_TREE 0
inline float bsig(float x) { const float e = metal::exp(-metal::abs(x)); return float(bfloat((x < 0 ? e : 1.0f) / (1.0f + e))); }
inline float bsilu(float x) { const float e = metal::exp(-metal::abs(x)); return float(bfloat((x < 0 ? x * e : x) / (1.0f + e))); }
inline float fsig(float x) { const float e = metal::exp(-metal::abs(x)); return (x < 0 ? e : 1.0f) / (1.0f + e); }
inline float fsilu(float x) { const float e = metal::exp(-metal::abs(x)); return (x < 0 ? x * e : x) / (1.0f + e); }
inline float log1p_(float x) {
  const float u = 1.0f + x;
  return u == 1.0f ? x : x * (metal::log(u) / (u - 1.0f));
}
// softplus in fp32 (threshold 20, as torch.nn.functional.softplus)
inline float fsoftplus(float x) { return x > 20.0f ? x : log1p_(metal::exp(x)); }
// Flash Next DeltaNet windows: fixed per-row arithmetic, state snapshots or replay of the kept prefix.

constexpr int GDN_NK = 1, GDN_NV = 2, GDN_DK = 128, GDN_DV = 128, GDN_TAPS = 4, GDN_MAXR = 16;
constexpr int GDN_C = 2 * GDN_NK * GDN_DK + GDN_NV * GDN_DV, GDN_PW = GDN_C + GDN_NV * GDN_DV + 2 * GDN_NV;
// a window's replay record a layer: k and v [rows][heads][128] (bf16), then [rows][heads] k norm, gate, beta, pad
constexpr int GDN_RV = GDN_MAXR * GDN_NV * GDN_DK, GDN_RS = 2 * GDN_RV;
// FZ_GDN_TREE 1 (plain mode only): rows follow TREE (per row: parent, depth, ancestor row by depth); 0 is the chain
#ifndef FZ_GDN_TREE
#define FZ_GDN_TREE 0
#endif
#if FZ_GDN_TREE
#define GDN_DEPTH(r) TREE[(r) * 18 + 1]
#define GDN_ROW(r, q) TREE[(r) * 18 + 2 + (q)]
#else
#define GDN_DEPTH(r) (r)
#define GDN_ROW(r, q) (q)
#endif


 Task qwen4_gdn_chain(const bfloat* P , const bfloat* CS ,
#if FZ_GDN_REPLAY
    float* S ,
#else
    const float* S ,
#endif
    const float* CW , const float* ALOG ,
    const float* DT , const float* NW , const float* eps ,
    const int* rows , bfloat* OUT , bfloat* CSO ,
#if FZ_GDN_REPLAY
    const bfloat* PREV , const uint& H0 , bfloat* CUR ,
    const int* AR ,
#else
    float* SO , const uint& H0 , const ulong& STATE_STRIDE , const uint& NORM_ROUND , const uint& CONV_ROUND ,
#if FZ_GDN_TREE
    const int* TREE ,
#endif
#endif
    uint t , uint lane ,
    uint sg , uint tg ) {
  constexpr int NK = GDN_NK, NV = GDN_NV, DK = GDN_DK, DV = GDN_DV, TAPS = GDN_TAPS, C = GDN_C, PW = GDN_PW;
  constexpr int RPS = DV / 32;                                // state rows a simdgroup
  const int hv = int(tg) + int(H0), hk = hv / (NV / NK), R = rows[0];
  static bfloat xq[GDN_MAXR][DK], xk[GDN_MAXR][DK], xv[GDN_MAXR][DV], ys[GDN_MAXR][DV];
  static float inv[GDN_MAXR][2], gates[GDN_MAXR][2], red[GDN_MAXR];
  float state[RPS][4];
  const float4* S4 = (const float4*)S;
  for (int j = 0; j < RPS; j++) {
    const float4 v = S4[((size_t(hv) * DV + sg * RPS + j) * DK) / 4 + lane];
    state[j][0] = v.x; state[j][1] = v.y; state[j][2] = v.z; state[j][3] = v.w;
  }
#if FZ_GDN_REPLAY
  threadgroup bfloat pk[GDN_MAXR][DK], pv[GDN_MAXR][DV];
  threadgroup float pg[GDN_MAXR][3];
  const int KEEP = AR[0];
  for (int i = int(t); i < KEEP * DK; i += 1024) {
    const int r = i / DK, d = i % DK;
    pk[r][d] = PREV[(r * NV + hv) * DK + d];
    pv[r][d] = PREV[GDN_RV + (r * NV + hv) * DV + d];
  }
  if (int(t) < KEEP * 3) {
    const float* ps = (const float*)(PREV + GDN_RS);
    pg[int(t) / 3][int(t) % 3] = ps[((int(t) / 3) * NV + hv) * 4 + int(t) % 3];
  }
  float* cs = (float*)(CUR + GDN_RS);
#endif
  const bool writes_qk = (hv % (NV / NK)) == 0;
  if (int(t) >= 1024 - GDN_MAXR && int(t) - (1024 - GDN_MAXR) < R) {
    const int r = int(t) - (1024 - GDN_MAXR);
    const float b = float(P[r * PW + C + NV * DV + hv]);
    const float a = float(P[r * PW + C + NV * DV + NV + hv]);
    gates[r][0] = metal::exp(-metal::exp(float(ALOG[hv])) * fsoftplus(a + float(DT[hv])));
    gates[r][1] = bsig(b);
#if FZ_GDN_REPLAY
    cs[(r * NV + hv) * 4 + 1] = gates[r][0];
    cs[(r * NV + hv) * 4 + 2] = gates[r][1];
#endif
  }
  for (int i = int(t); i < R * (2 * DK + DV); i += 1024) { // (row, channel): q (hk), k (hk), v (hv)
    const int r = i / (2 * DK + DV), j = i % (2 * DK + DV);
    const int c = j < DK ? hk * DK + j : j < 2 * DK ? NK * DK + hk * DK + j - DK : 2 * NK * DK + hv * DV + j - 2 * DK;
    float conv = 0.0f;
    for (int tap = 0; tap < TAPS; tap++) {
      const int at = GDN_DEPTH(r) + tap;
      const float x = at < TAPS - 1 ? float(CS[at * C + c]) : float(P[GDN_ROW(r, at - (TAPS - 1)) * PW + c]);
      conv = fma(float(CW[c * TAPS + tap]), x, conv);
    }
    const bfloat act = bfloat(bsilu(CONV_ROUND ? float(bfloat(conv)) : conv));
    if (j < DK) xq[r][j] = act; else if (j < 2 * DK) xk[r][j - DK] = act; else xv[r][j - 2 * DK] = act;
#if FZ_GDN_REPLAY
    if (j >= 2 * DK) CUR[GDN_RV + (r * NV + hv) * DV + j - 2 * DK] = act;
    else if (j >= DK) CUR[(r * NV + hv) * DK + j - DK] = act;
#endif
    if (j < 2 * DK ? writes_qk : true)
      for (int q = 0; q < TAPS - 1; q++) {
        const int at = GDN_DEPTH(r) + 1 + q;
        CSO[size_t(r) * (STATE_STRIDE ? STATE_STRIDE / 2 : (TAPS - 1) * C) + q * C + c] = at < TAPS - 1 ? CS[at * C + c] : P[GDN_ROW(r, at - (TAPS - 1)) * PW + c];
      }
  }
  co_await group_barrier();
#if FZ_GDN_REPLAY
  for (int r = 0; r < KEEP; r++) { // the previous window's kept rows: their state updates, outputs not needed
    const float g = pg[r][1], beta = pg[r][2], ik = pg[r][0];
    float kk[4], kv[RPS];
    for (int i = 0; i < 4; i++) kk[i] = float(pk[r][lane * 4 + i]) * ik;
    for (int j = 0; j < RPS; j++) {
      kv[j] = 0.0f;
      for (int i = 0; i < 4; i++) {
        state[j][i] = state[j][i] * g;
        kv[j] += state[j][i] * kk[i];
      }
    }
    for (int j = 0; j < RPS; j++) kv[j] = (co_await simd_sum(kv[j]));
    for (int j = 0; j < RPS; j++) {
      const float delta = (float(pv[r][int(sg) * RPS + j]) - kv[j]) * beta;
      for (int i = 0; i < 4; i++) state[j][i] = state[j][i] + kk[i] * delta;
    }
  }
  float4* SW4 = (float4*)S;
  for (int j = 0; j < RPS; j++) SW4[((size_t(hv) * DV + sg * RPS + j) * DK) / 4 + lane] = float4(state[j][0], state[j][1], state[j][2], state[j][3]);
#endif
  if (int(sg) < 2 * R) { // the q and k norms: x / sqrt(sum(x^2) + 1e-6) in fp32, q also times DK^-0.5
    const int r = int(sg) / 2;
    const bool isq = int(sg) % 2 == 0;
    bfloat* x = isq ? xq[r] : xk[r];
    float ss = 0.0f;
    for (int i = 0; i < DK / 32; i++) {
      const float v = float(x[lane * (DK / 32) + i]);
      ss = fma(v, v, ss);
    }
    ss = (co_await simd_sum(ss));
    if (lane == 0) {
      inv[r][isq ? 0 : 1] = metal::rsqrt(ss + 1e-6f) * (isq ? metal::rsqrt(float(DK)) : 1.0f);
#if FZ_GDN_REPLAY
      if (!isq) cs[(r * NV + hv) * 4] = inv[r][1];
#endif
    }
  }
  co_await group_barrier();
#if !FZ_GDN_REPLAY
  float4* SO4 = (float4*)SO;
#endif
  for (int r = 0; r < R; r++) { // the four state rows' chains side by side: each element's arithmetic unchanged
#if FZ_GDN_TREE
    if (r > 0 && TREE[r * 18] != r - 1) { // a branch's first row starts from its parent row's stored state
      const int pr = TREE[r * 18];
      for (int j = 0; j < RPS; j++) {
        const float4 v = SO4[((size_t(pr) * NV + hv) * DV + sg * RPS + j) * DK / 4 + lane];
        state[j][0] = v.x; state[j][1] = v.y; state[j][2] = v.z; state[j][3] = v.w;
      }
    }
#endif
    const float g = gates[r][0], beta = gates[r][1], iq = inv[r][0], ik = inv[r][1];
    float kk[4], qq[4], kv[RPS], out[RPS];
    for (int i = 0; i < 4; i++) { kk[i] = float(xk[r][lane * 4 + i]) * ik; qq[i] = float(xq[r][lane * 4 + i]) * iq; }
    for (int j = 0; j < RPS; j++) {
      kv[j] = 0.0f;
      for (int i = 0; i < 4; i++) {
        state[j][i] = state[j][i] * g;
        kv[j] += state[j][i] * kk[i];
      }
    }
    for (int j = 0; j < RPS; j++) kv[j] = (co_await simd_sum(kv[j]));
    for (int j = 0; j < RPS; j++) {
      const float delta = (float(xv[r][int(sg) * RPS + j]) - kv[j]) * beta;
      out[j] = 0.0f;
      for (int i = 0; i < 4; i++) {
        state[j][i] = state[j][i] + kk[i] * delta;
        out[j] += state[j][i] * qq[i];
      }
    }
    for (int j = 0; j < RPS; j++) out[j] = (co_await simd_sum(out[j]));
    if (lane == 0) for (int j = 0; j < RPS; j++) ys[r][int(sg) * RPS + j] = bfloat(out[j]);
#if !FZ_GDN_REPLAY
    for (int j = 0; j < RPS; j++) SO4[size_t(r) * (STATE_STRIDE ? STATE_STRIDE / 16 : NV * DV * DK / 4) + ((size_t(hv) * DV + sg * RPS + j) * DK / 4 + lane)] = float4(state[j][0], state[j][1], state[j][2], state[j][3]);
#endif
  }
  co_await group_barrier();
  if (int(sg) < R) {
    float ss = 0.0f;
    for (int i = 0; i < DV / 32; i++) { const float v = float(ys[sg][lane * (DV / 32) + i]); ss = fma(v, v, ss); }
    ss = (co_await simd_sum(ss));
    if (lane == 0) red[sg] = metal::rsqrt(ss / float(DV) + eps[0]);
  }
  co_await group_barrier();
  for (int i = int(t); i < R * DV; i += 1024) { // mx.fast.rms_norm's bf16(w * bf16(y * inv)), times sigmoid(z)
    const int r = i / DV, d = i % DV;
    const float weighted = float(NW[d]) * float(bfloat(float(ys[r][d]) * red[r]));
    const float y = NORM_ROUND ? float(bfloat(weighted)) : weighted;
    const float z = float(P[r * PW + C + hv * DV + d]);
    OUT[r * NV * DV + hv * DV + d] = bfloat(y * fsig(z));
  }
}

static void run(const unsigned char* const* input,unsigned char* out,unsigned char* state,unsigned char* tail_out,int rows,int flags,uint repeat) {const Qwen4Control control_params{repeat};
  const auto* p=reinterpret_cast<const bfloat*>(input[0]);
  const auto* cs=reinterpret_cast<const bfloat*>(input[1]);
  const auto* cw=reinterpret_cast<const float*>(input[2]);
  const auto* s=reinterpret_cast<const float*>(input[3]);
  const auto* alog=reinterpret_cast<const float*>(input[4]);
  const auto* dt=reinterpret_cast<const float*>(input[5]);
  const auto* nw=reinterpret_cast<const float*>(input[6]);
  const float eps=1e-6f; const uint h0=0,norm_round=(flags&4)?0:1,conv_round=(flags&8)?1:0;
  const ulong stride=0;
  auto* tail=reinterpret_cast<bfloat*>(tail_out);
  for(uint head=0;head<2;++head) {
    Task tasks[1024];
    for(uint t=0;t<1024;++t) tasks[t]=qwen4_gdn_chain(p,cs,s,cw,alog,dt,nw,&eps,&rows,reinterpret_cast<bfloat*>(out),tail,reinterpret_cast<float*>(state),h0,stride,norm_round,conv_round,t,t%32,t/32,head);
    run_tasks(tasks,1024);
  }
}
}
namespace g1c1 {


inline float qwen4_gdn_control(float x, uint r) {return isnan(x) ? 0.0f : x;}
#define FZ_GDN_REPLAY 0
#define FZ_GDN_TREE 0
inline float bsig(float x) { const float e = metal::exp(-metal::abs(x)); return float(bfloat((x < 0 ? e : 1.0f) / (1.0f + e))); }
inline float bsilu(float x) { const float e = metal::exp(-metal::abs(x)); return float(bfloat((x < 0 ? x * e : x) / (1.0f + e))); }
inline float fsig(float x) { const float e = metal::exp(-metal::abs(x)); return (x < 0 ? e : 1.0f) / (1.0f + e); }
inline float fsilu(float x) { const float e = metal::exp(-metal::abs(x)); return (x < 0 ? x * e : x) / (1.0f + e); }
inline float log1p_(float x) {
  const float u = 1.0f + x;
  return u == 1.0f ? x : x * (metal::log(u) / (u - 1.0f));
}
// softplus in fp32 (threshold 20, as torch.nn.functional.softplus)
inline float fsoftplus(float x) { return x > 20.0f ? x : log1p_(metal::exp(x)); }
// Flash Next DeltaNet windows: fixed per-row arithmetic, state snapshots or replay of the kept prefix.

constexpr int GDN_NK = 1, GDN_NV = 2, GDN_DK = 128, GDN_DV = 128, GDN_TAPS = 4, GDN_MAXR = 16;
constexpr int GDN_C = 2 * GDN_NK * GDN_DK + GDN_NV * GDN_DV, GDN_PW = GDN_C + GDN_NV * GDN_DV + 2 * GDN_NV;
// a window's replay record a layer: k and v [rows][heads][128] (bf16), then [rows][heads] k norm, gate, beta, pad
constexpr int GDN_RV = GDN_MAXR * GDN_NV * GDN_DK, GDN_RS = 2 * GDN_RV;
// FZ_GDN_TREE 1 (plain mode only): rows follow TREE (per row: parent, depth, ancestor row by depth); 0 is the chain
#ifndef FZ_GDN_TREE
#define FZ_GDN_TREE 0
#endif
#if FZ_GDN_TREE
#define GDN_DEPTH(r) TREE[(r) * 18 + 1]
#define GDN_ROW(r, q) TREE[(r) * 18 + 2 + (q)]
#else
#define GDN_DEPTH(r) (r)
#define GDN_ROW(r, q) (q)
#endif


 Task qwen4_gdn_chain(const bfloat* P , const bfloat* CS ,
#if FZ_GDN_REPLAY
    float* S ,
#else
    const float* S ,
#endif
    const float* CW , const float* ALOG ,
    const float* DT , const float* NW , const float* eps ,
    const int* rows , bfloat* OUT , bfloat* CSO ,
#if FZ_GDN_REPLAY
    const bfloat* PREV , const uint& H0 , bfloat* CUR ,
    const int* AR ,
#else
    float* SO , const uint& H0 , const ulong& STATE_STRIDE , const uint& NORM_ROUND , const uint& CONV_ROUND ,
#if FZ_GDN_TREE
    const int* TREE ,
#endif
#endif
    uint t , uint lane ,
    uint sg , uint tg ) {
  constexpr int NK = GDN_NK, NV = GDN_NV, DK = GDN_DK, DV = GDN_DV, TAPS = GDN_TAPS, C = GDN_C, PW = GDN_PW;
  constexpr int RPS = DV / 32;                                // state rows a simdgroup
  const int hv = int(tg) + int(H0), hk = hv / (NV / NK), R = rows[0];
  static bfloat xq[GDN_MAXR][DK], xk[GDN_MAXR][DK], xv[GDN_MAXR][DV], ys[GDN_MAXR][DV];
  static float inv[GDN_MAXR][2], gates[GDN_MAXR][2], red[GDN_MAXR];
  float state[RPS][4];
  const float4* S4 = (const float4*)S;
  for (int j = 0; j < RPS; j++) {
    const float4 v = S4[((size_t(hv) * DV + sg * RPS + j) * DK) / 4 + lane];
    state[j][0] = v.x; state[j][1] = v.y; state[j][2] = v.z; state[j][3] = v.w;
  }
#if FZ_GDN_REPLAY
  threadgroup bfloat pk[GDN_MAXR][DK], pv[GDN_MAXR][DV];
  threadgroup float pg[GDN_MAXR][3];
  const int KEEP = AR[0];
  for (int i = int(t); i < KEEP * DK; i += 1024) {
    const int r = i / DK, d = i % DK;
    pk[r][d] = PREV[(r * NV + hv) * DK + d];
    pv[r][d] = PREV[GDN_RV + (r * NV + hv) * DV + d];
  }
  if (int(t) < KEEP * 3) {
    const float* ps = (const float*)(PREV + GDN_RS);
    pg[int(t) / 3][int(t) % 3] = ps[((int(t) / 3) * NV + hv) * 4 + int(t) % 3];
  }
  float* cs = (float*)(CUR + GDN_RS);
#endif
  const bool writes_qk = (hv % (NV / NK)) == 0;
  if (int(t) >= 1024 - GDN_MAXR && int(t) - (1024 - GDN_MAXR) < R) {
    const int r = int(t) - (1024 - GDN_MAXR);
    const float b = float(P[r * PW + C + NV * DV + hv]);
    const float a = float(P[r * PW + C + NV * DV + NV + hv]);
    gates[r][0] = metal::exp(-metal::exp(float(ALOG[hv])) * fsoftplus(a + float(DT[hv])));
    gates[r][1] = bsig(b);
#if FZ_GDN_REPLAY
    cs[(r * NV + hv) * 4 + 1] = gates[r][0];
    cs[(r * NV + hv) * 4 + 2] = gates[r][1];
#endif
  }
  for (int i = int(t); i < R * (2 * DK + DV); i += 1024) { // (row, channel): q (hk), k (hk), v (hv)
    const int r = i / (2 * DK + DV), j = i % (2 * DK + DV);
    const int c = j < DK ? hk * DK + j : j < 2 * DK ? NK * DK + hk * DK + j - DK : 2 * NK * DK + hv * DV + j - 2 * DK;
    float conv = 0.0f;
    for (int tap = 0; tap < TAPS; tap++) {
      const int at = GDN_DEPTH(r) + tap;
      const float x = at < TAPS - 1 ? float(CS[at * C + c]) : float(P[GDN_ROW(r, at - (TAPS - 1)) * PW + c]);
      conv = fma(float(CW[c * TAPS + tap]), x, conv);
    }
    const bfloat act = bfloat(bsilu(CONV_ROUND ? float(bfloat(conv)) : conv));
    if (j < DK) xq[r][j] = act; else if (j < 2 * DK) xk[r][j - DK] = act; else xv[r][j - 2 * DK] = act;
#if FZ_GDN_REPLAY
    if (j >= 2 * DK) CUR[GDN_RV + (r * NV + hv) * DV + j - 2 * DK] = act;
    else if (j >= DK) CUR[(r * NV + hv) * DK + j - DK] = act;
#endif
    if (j < 2 * DK ? writes_qk : true)
      for (int q = 0; q < TAPS - 1; q++) {
        const int at = GDN_DEPTH(r) + 1 + q;
        CSO[size_t(r) * (STATE_STRIDE ? STATE_STRIDE / 2 : (TAPS - 1) * C) + q * C + c] = bfloat(qwen4_gdn_control(float(at < TAPS - 1 ? CS[at * C + c] : P[GDN_ROW(r, at - (TAPS - 1)) * PW + c]), 0));
      }
  }
  co_await group_barrier();
#if FZ_GDN_REPLAY
  for (int r = 0; r < KEEP; r++) { // the previous window's kept rows: their state updates, outputs not needed
    const float g = pg[r][1], beta = pg[r][2], ik = pg[r][0];
    float kk[4], kv[RPS];
    for (int i = 0; i < 4; i++) kk[i] = float(pk[r][lane * 4 + i]) * ik;
    for (int j = 0; j < RPS; j++) {
      kv[j] = 0.0f;
      for (int i = 0; i < 4; i++) {
        state[j][i] = state[j][i] * g;
        kv[j] += state[j][i] * kk[i];
      }
    }
    for (int j = 0; j < RPS; j++) kv[j] = (co_await simd_sum(kv[j]));
    for (int j = 0; j < RPS; j++) {
      const float delta = (float(pv[r][int(sg) * RPS + j]) - kv[j]) * beta;
      for (int i = 0; i < 4; i++) state[j][i] = state[j][i] + kk[i] * delta;
    }
  }
  float4* SW4 = (float4*)S;
  for (int j = 0; j < RPS; j++) SW4[((size_t(hv) * DV + sg * RPS + j) * DK) / 4 + lane] = float4(qwen4_gdn_control(state[j][0],0), qwen4_gdn_control(state[j][1],0), qwen4_gdn_control(state[j][2],0), qwen4_gdn_control(state[j][3],0));
#endif
  if (int(sg) < 2 * R) { // the q and k norms: x / sqrt(sum(x^2) + 1e-6) in fp32, q also times DK^-0.5
    const int r = int(sg) / 2;
    const bool isq = int(sg) % 2 == 0;
    bfloat* x = isq ? xq[r] : xk[r];
    float ss = 0.0f;
    for (int i = 0; i < DK / 32; i++) {
      const float v = float(x[lane * (DK / 32) + i]);
      ss = fma(v, v, ss);
    }
    ss = (co_await simd_sum(ss));
    if (lane == 0) {
      inv[r][isq ? 0 : 1] = metal::rsqrt(ss + 1e-6f) * (isq ? metal::rsqrt(float(DK)) : 1.0f);
#if FZ_GDN_REPLAY
      if (!isq) cs[(r * NV + hv) * 4] = inv[r][1];
#endif
    }
  }
  co_await group_barrier();
#if !FZ_GDN_REPLAY
  float4* SO4 = (float4*)SO;
#endif
  for (int r = 0; r < R; r++) { // the four state rows' chains side by side: each element's arithmetic unchanged
#if FZ_GDN_TREE
    if (r > 0 && TREE[r * 18] != r - 1) { // a branch's first row starts from its parent row's stored state
      const int pr = TREE[r * 18];
      for (int j = 0; j < RPS; j++) {
        const float4 v = SO4[((size_t(pr) * NV + hv) * DV + sg * RPS + j) * DK / 4 + lane];
        state[j][0] = v.x; state[j][1] = v.y; state[j][2] = v.z; state[j][3] = v.w;
      }
    }
#endif
    const float g = gates[r][0], beta = gates[r][1], iq = inv[r][0], ik = inv[r][1];
    float kk[4], qq[4], kv[RPS], out[RPS];
    for (int i = 0; i < 4; i++) { kk[i] = float(xk[r][lane * 4 + i]) * ik; qq[i] = float(xq[r][lane * 4 + i]) * iq; }
    for (int j = 0; j < RPS; j++) {
      kv[j] = 0.0f;
      for (int i = 0; i < 4; i++) {
        state[j][i] = state[j][i] * g;
        kv[j] += state[j][i] * kk[i];
      }
    }
    for (int j = 0; j < RPS; j++) kv[j] = (co_await simd_sum(kv[j]));
    for (int j = 0; j < RPS; j++) {
      const float delta = (float(xv[r][int(sg) * RPS + j]) - kv[j]) * beta;
      out[j] = 0.0f;
      for (int i = 0; i < 4; i++) {
        state[j][i] = state[j][i] + kk[i] * delta;
        out[j] += state[j][i] * qq[i];
      }
    }
    for (int j = 0; j < RPS; j++) out[j] = (co_await simd_sum(out[j]));
    if (lane == 0) for (int j = 0; j < RPS; j++) ys[r][int(sg) * RPS + j] = bfloat(out[j]);
#if !FZ_GDN_REPLAY
    for (int j = 0; j < RPS; j++) SO4[size_t(r) * (STATE_STRIDE ? STATE_STRIDE / 16 : NV * DV * DK / 4) + ((size_t(hv) * DV + sg * RPS + j) * DK / 4 + lane)] = float4(qwen4_gdn_control(state[j][0],0), qwen4_gdn_control(state[j][1],0), qwen4_gdn_control(state[j][2],0), qwen4_gdn_control(state[j][3],0));
#endif
  }
  co_await group_barrier();
  if (int(sg) < R) {
    float ss = 0.0f;
    for (int i = 0; i < DV / 32; i++) { const float v = float(ys[sg][lane * (DV / 32) + i]); ss = fma(v, v, ss); }
    ss = (co_await simd_sum(ss));
    if (lane == 0) red[sg] = metal::rsqrt(ss / float(DV) + eps[0]);
  }
  co_await group_barrier();
  for (int i = int(t); i < R * DV; i += 1024) { // mx.fast.rms_norm's bf16(w * bf16(y * inv)), times sigmoid(z)
    const int r = i / DV, d = i % DV;
    const float weighted = float(NW[d]) * float(bfloat(float(ys[r][d]) * red[r]));
    const float y = NORM_ROUND ? float(bfloat(weighted)) : weighted;
    const float z = float(P[r * PW + C + hv * DV + d]);
    OUT[r * NV * DV + hv * DV + d] = bfloat(qwen4_gdn_control(float(bfloat(y * fsig(z))), 0));
  }
}

static void run(const unsigned char* const* input,unsigned char* out,unsigned char* state,unsigned char* tail_out,int rows,int flags,uint repeat) {const Qwen4Control control_params{repeat};
  const auto* p=reinterpret_cast<const bfloat*>(input[0]);
  const auto* cs=reinterpret_cast<const bfloat*>(input[1]);
  const auto* cw=reinterpret_cast<const float*>(input[2]);
  const auto* s=reinterpret_cast<const float*>(input[3]);
  const auto* alog=reinterpret_cast<const float*>(input[4]);
  const auto* dt=reinterpret_cast<const float*>(input[5]);
  const auto* nw=reinterpret_cast<const float*>(input[6]);
  const float eps=1e-6f; const uint h0=0,norm_round=(flags&4)?0:1,conv_round=(flags&8)?1:0;
  const ulong stride=0;
  auto* tail=reinterpret_cast<bfloat*>(tail_out);
  for(uint head=0;head<2;++head) {
    Task tasks[1024];
    for(uint t=0;t<1024;++t) tasks[t]=qwen4_gdn_chain(p,cs,s,cw,alog,dt,nw,&eps,&rows,reinterpret_cast<bfloat*>(out),tail,reinterpret_cast<float*>(state),h0,stride,norm_round,conv_round,t,t%32,t/32,head);
    run_tasks(tasks,1024);
  }
}
}
namespace g1c2 {


struct Qwen4Control { uint repeat; };
inline float qwen4_gdn_control(float x, uint r) {return x + float(r);}
#define FZ_GDN_REPLAY 0
#define FZ_GDN_TREE 0
inline float bsig(float x) { const float e = metal::exp(-metal::abs(x)); return float(bfloat((x < 0 ? e : 1.0f) / (1.0f + e))); }
inline float bsilu(float x) { const float e = metal::exp(-metal::abs(x)); return float(bfloat((x < 0 ? x * e : x) / (1.0f + e))); }
inline float fsig(float x) { const float e = metal::exp(-metal::abs(x)); return (x < 0 ? e : 1.0f) / (1.0f + e); }
inline float fsilu(float x) { const float e = metal::exp(-metal::abs(x)); return (x < 0 ? x * e : x) / (1.0f + e); }
inline float log1p_(float x) {
  const float u = 1.0f + x;
  return u == 1.0f ? x : x * (metal::log(u) / (u - 1.0f));
}
// softplus in fp32 (threshold 20, as torch.nn.functional.softplus)
inline float fsoftplus(float x) { return x > 20.0f ? x : log1p_(metal::exp(x)); }
// Flash Next DeltaNet windows: fixed per-row arithmetic, state snapshots or replay of the kept prefix.

constexpr int GDN_NK = 1, GDN_NV = 2, GDN_DK = 128, GDN_DV = 128, GDN_TAPS = 4, GDN_MAXR = 16;
constexpr int GDN_C = 2 * GDN_NK * GDN_DK + GDN_NV * GDN_DV, GDN_PW = GDN_C + GDN_NV * GDN_DV + 2 * GDN_NV;
// a window's replay record a layer: k and v [rows][heads][128] (bf16), then [rows][heads] k norm, gate, beta, pad
constexpr int GDN_RV = GDN_MAXR * GDN_NV * GDN_DK, GDN_RS = 2 * GDN_RV;
// FZ_GDN_TREE 1 (plain mode only): rows follow TREE (per row: parent, depth, ancestor row by depth); 0 is the chain
#ifndef FZ_GDN_TREE
#define FZ_GDN_TREE 0
#endif
#if FZ_GDN_TREE
#define GDN_DEPTH(r) TREE[(r) * 18 + 1]
#define GDN_ROW(r, q) TREE[(r) * 18 + 2 + (q)]
#else
#define GDN_DEPTH(r) (r)
#define GDN_ROW(r, q) (q)
#endif


 Task qwen4_gdn_chain(const bfloat* P , const bfloat* CS ,
#if FZ_GDN_REPLAY
    float* S ,
#else
    const float* S ,
#endif
    const float* CW , const float* ALOG ,
    const float* DT , const float* NW , const float* eps ,
    const int* rows , bfloat* OUT , bfloat* CSO ,
#if FZ_GDN_REPLAY
    const bfloat* PREV , const uint& H0 , bfloat* CUR ,
    const int* AR ,
#else
    float* SO , const uint& H0 , const ulong& STATE_STRIDE , const uint& NORM_ROUND , const uint& CONV_ROUND ,
#if FZ_GDN_TREE
    const int* TREE ,
#endif
#endif
    const Qwen4Control& TF_CONTROL , uint t , uint lane ,
    uint sg , uint tg ) {
  constexpr int NK = GDN_NK, NV = GDN_NV, DK = GDN_DK, DV = GDN_DV, TAPS = GDN_TAPS, C = GDN_C, PW = GDN_PW;
  constexpr int RPS = DV / 32;                                // state rows a simdgroup
  const int hv = int(tg) + int(H0), hk = hv / (NV / NK), R = rows[0];
  static bfloat xq[GDN_MAXR][DK], xk[GDN_MAXR][DK], xv[GDN_MAXR][DV], ys[GDN_MAXR][DV];
  static float inv[GDN_MAXR][2], gates[GDN_MAXR][2], red[GDN_MAXR];
  float state[RPS][4];
  const float4* S4 = (const float4*)S;
  for (int j = 0; j < RPS; j++) {
    const float4 v = S4[((size_t(hv) * DV + sg * RPS + j) * DK) / 4 + lane];
    state[j][0] = v.x; state[j][1] = v.y; state[j][2] = v.z; state[j][3] = v.w;
  }
#if FZ_GDN_REPLAY
  threadgroup bfloat pk[GDN_MAXR][DK], pv[GDN_MAXR][DV];
  threadgroup float pg[GDN_MAXR][3];
  const int KEEP = AR[0];
  for (int i = int(t); i < KEEP * DK; i += 1024) {
    const int r = i / DK, d = i % DK;
    pk[r][d] = PREV[(r * NV + hv) * DK + d];
    pv[r][d] = PREV[GDN_RV + (r * NV + hv) * DV + d];
  }
  if (int(t) < KEEP * 3) {
    const float* ps = (const float*)(PREV + GDN_RS);
    pg[int(t) / 3][int(t) % 3] = ps[((int(t) / 3) * NV + hv) * 4 + int(t) % 3];
  }
  float* cs = (float*)(CUR + GDN_RS);
#endif
  const bool writes_qk = (hv % (NV / NK)) == 0;
  if (int(t) >= 1024 - GDN_MAXR && int(t) - (1024 - GDN_MAXR) < R) {
    const int r = int(t) - (1024 - GDN_MAXR);
    const float b = float(P[r * PW + C + NV * DV + hv]);
    const float a = float(P[r * PW + C + NV * DV + NV + hv]);
    gates[r][0] = metal::exp(-metal::exp(float(ALOG[hv])) * fsoftplus(a + float(DT[hv])));
    gates[r][1] = bsig(b);
#if FZ_GDN_REPLAY
    cs[(r * NV + hv) * 4 + 1] = gates[r][0];
    cs[(r * NV + hv) * 4 + 2] = gates[r][1];
#endif
  }
  for (int i = int(t); i < R * (2 * DK + DV); i += 1024) { // (row, channel): q (hk), k (hk), v (hv)
    const int r = i / (2 * DK + DV), j = i % (2 * DK + DV);
    const int c = j < DK ? hk * DK + j : j < 2 * DK ? NK * DK + hk * DK + j - DK : 2 * NK * DK + hv * DV + j - 2 * DK;
    float conv = 0.0f;
    for (int tap = 0; tap < TAPS; tap++) {
      const int at = GDN_DEPTH(r) + tap;
      const float x = at < TAPS - 1 ? float(CS[at * C + c]) : float(P[GDN_ROW(r, at - (TAPS - 1)) * PW + c]);
      conv = fma(float(CW[c * TAPS + tap]), x, conv);
    }
    const bfloat act = bfloat(bsilu(CONV_ROUND ? float(bfloat(conv)) : conv));
    if (j < DK) xq[r][j] = act; else if (j < 2 * DK) xk[r][j - DK] = act; else xv[r][j - 2 * DK] = act;
#if FZ_GDN_REPLAY
    if (j >= 2 * DK) CUR[GDN_RV + (r * NV + hv) * DV + j - 2 * DK] = act;
    else if (j >= DK) CUR[(r * NV + hv) * DK + j - DK] = act;
#endif
    if (j < 2 * DK ? writes_qk : true)
      for (int q = 0; q < TAPS - 1; q++) {
        const int at = GDN_DEPTH(r) + 1 + q;
        CSO[size_t(r) * (STATE_STRIDE ? STATE_STRIDE / 2 : (TAPS - 1) * C) + q * C + c] = bfloat(qwen4_gdn_control(float(at < TAPS - 1 ? CS[at * C + c] : P[GDN_ROW(r, at - (TAPS - 1)) * PW + c]), TF_CONTROL.repeat));
      }
  }
  co_await group_barrier();
#if FZ_GDN_REPLAY
  for (int r = 0; r < KEEP; r++) { // the previous window's kept rows: their state updates, outputs not needed
    const float g = pg[r][1], beta = pg[r][2], ik = pg[r][0];
    float kk[4], kv[RPS];
    for (int i = 0; i < 4; i++) kk[i] = float(pk[r][lane * 4 + i]) * ik;
    for (int j = 0; j < RPS; j++) {
      kv[j] = 0.0f;
      for (int i = 0; i < 4; i++) {
        state[j][i] = state[j][i] * g;
        kv[j] += state[j][i] * kk[i];
      }
    }
    for (int j = 0; j < RPS; j++) kv[j] = (co_await simd_sum(kv[j]));
    for (int j = 0; j < RPS; j++) {
      const float delta = (float(pv[r][int(sg) * RPS + j]) - kv[j]) * beta;
      for (int i = 0; i < 4; i++) state[j][i] = state[j][i] + kk[i] * delta;
    }
  }
  float4* SW4 = (float4*)S;
  for (int j = 0; j < RPS; j++) SW4[((size_t(hv) * DV + sg * RPS + j) * DK) / 4 + lane] = float4(qwen4_gdn_control(state[j][0],TF_CONTROL.repeat), qwen4_gdn_control(state[j][1],TF_CONTROL.repeat), qwen4_gdn_control(state[j][2],TF_CONTROL.repeat), qwen4_gdn_control(state[j][3],TF_CONTROL.repeat));
#endif
  if (int(sg) < 2 * R) { // the q and k norms: x / sqrt(sum(x^2) + 1e-6) in fp32, q also times DK^-0.5
    const int r = int(sg) / 2;
    const bool isq = int(sg) % 2 == 0;
    bfloat* x = isq ? xq[r] : xk[r];
    float ss = 0.0f;
    for (int i = 0; i < DK / 32; i++) {
      const float v = float(x[lane * (DK / 32) + i]);
      ss = fma(v, v, ss);
    }
    ss = (co_await simd_sum(ss));
    if (lane == 0) {
      inv[r][isq ? 0 : 1] = metal::rsqrt(ss + 1e-6f) * (isq ? metal::rsqrt(float(DK)) : 1.0f);
#if FZ_GDN_REPLAY
      if (!isq) cs[(r * NV + hv) * 4] = inv[r][1];
#endif
    }
  }
  co_await group_barrier();
#if !FZ_GDN_REPLAY
  float4* SO4 = (float4*)SO;
#endif
  for (int r = 0; r < R; r++) { // the four state rows' chains side by side: each element's arithmetic unchanged
#if FZ_GDN_TREE
    if (r > 0 && TREE[r * 18] != r - 1) { // a branch's first row starts from its parent row's stored state
      const int pr = TREE[r * 18];
      for (int j = 0; j < RPS; j++) {
        const float4 v = SO4[((size_t(pr) * NV + hv) * DV + sg * RPS + j) * DK / 4 + lane];
        state[j][0] = v.x; state[j][1] = v.y; state[j][2] = v.z; state[j][3] = v.w;
      }
    }
#endif
    const float g = gates[r][0], beta = gates[r][1], iq = inv[r][0], ik = inv[r][1];
    float kk[4], qq[4], kv[RPS], out[RPS];
    for (int i = 0; i < 4; i++) { kk[i] = float(xk[r][lane * 4 + i]) * ik; qq[i] = float(xq[r][lane * 4 + i]) * iq; }
    for (int j = 0; j < RPS; j++) {
      kv[j] = 0.0f;
      for (int i = 0; i < 4; i++) {
        state[j][i] = state[j][i] * g;
        kv[j] += state[j][i] * kk[i];
      }
    }
    for (int j = 0; j < RPS; j++) kv[j] = (co_await simd_sum(kv[j]));
    for (int j = 0; j < RPS; j++) {
      const float delta = (float(xv[r][int(sg) * RPS + j]) - kv[j]) * beta;
      out[j] = 0.0f;
      for (int i = 0; i < 4; i++) {
        state[j][i] = state[j][i] + kk[i] * delta;
        out[j] += state[j][i] * qq[i];
      }
    }
    for (int j = 0; j < RPS; j++) out[j] = (co_await simd_sum(out[j]));
    if (lane == 0) for (int j = 0; j < RPS; j++) ys[r][int(sg) * RPS + j] = bfloat(out[j]);
#if !FZ_GDN_REPLAY
    for (int j = 0; j < RPS; j++) SO4[size_t(r) * (STATE_STRIDE ? STATE_STRIDE / 16 : NV * DV * DK / 4) + ((size_t(hv) * DV + sg * RPS + j) * DK / 4 + lane)] = float4(qwen4_gdn_control(state[j][0],TF_CONTROL.repeat), qwen4_gdn_control(state[j][1],TF_CONTROL.repeat), qwen4_gdn_control(state[j][2],TF_CONTROL.repeat), qwen4_gdn_control(state[j][3],TF_CONTROL.repeat));
#endif
  }
  co_await group_barrier();
  if (int(sg) < R) {
    float ss = 0.0f;
    for (int i = 0; i < DV / 32; i++) { const float v = float(ys[sg][lane * (DV / 32) + i]); ss = fma(v, v, ss); }
    ss = (co_await simd_sum(ss));
    if (lane == 0) red[sg] = metal::rsqrt(ss / float(DV) + eps[0]);
  }
  co_await group_barrier();
  for (int i = int(t); i < R * DV; i += 1024) { // mx.fast.rms_norm's bf16(w * bf16(y * inv)), times sigmoid(z)
    const int r = i / DV, d = i % DV;
    const float weighted = float(NW[d]) * float(bfloat(float(ys[r][d]) * red[r]));
    const float y = NORM_ROUND ? float(bfloat(weighted)) : weighted;
    const float z = float(P[r * PW + C + hv * DV + d]);
    OUT[r * NV * DV + hv * DV + d] = bfloat(qwen4_gdn_control(float(bfloat(y * fsig(z))), TF_CONTROL.repeat));
  }
}

static void run(const unsigned char* const* input,unsigned char* out,unsigned char* state,unsigned char* tail_out,int rows,int flags,uint repeat) {const Qwen4Control control_params{repeat};
  const auto* p=reinterpret_cast<const bfloat*>(input[0]);
  const auto* cs=reinterpret_cast<const bfloat*>(input[1]);
  const auto* cw=reinterpret_cast<const float*>(input[2]);
  const auto* s=reinterpret_cast<const float*>(input[3]);
  const auto* alog=reinterpret_cast<const float*>(input[4]);
  const auto* dt=reinterpret_cast<const float*>(input[5]);
  const auto* nw=reinterpret_cast<const float*>(input[6]);
  const float eps=1e-6f; const uint h0=0,norm_round=(flags&4)?0:1,conv_round=(flags&8)?1:0;
  const ulong stride=0;
  auto* tail=reinterpret_cast<bfloat*>(tail_out);
  for(uint head=0;head<2;++head) {
    Task tasks[1024];
    for(uint t=0;t<1024;++t) tasks[t]=qwen4_gdn_chain(p,cs,s,cw,alog,dt,nw,&eps,&rows,reinterpret_cast<bfloat*>(out),tail,reinterpret_cast<float*>(state),h0,stride,norm_round,conv_round,control_params,t,t%32,t/32,head);
    run_tasks(tasks,1024);
  }
}
}
namespace g1c3 {


#define FZ_GDN_REPLAY 0
#define FZ_GDN_TREE 0
inline float bsig(float x) { const float e = metal::exp(-metal::abs(x)); return float(bfloat((x < 0 ? e : 1.0f) / (1.0f + e))); }
inline float bsilu(float x) { const float e = metal::exp(-metal::abs(x)); return float(bfloat((x < 0 ? x * e : x) / (1.0f + e))); }
inline float fsig(float x) { const float e = metal::exp(-metal::abs(x)); return (x < 0 ? e : 1.0f) / (1.0f + e); }
inline float fsilu(float x) { const float e = metal::exp(-metal::abs(x)); return (x < 0 ? x * e : x) / (1.0f + e); }
inline float log1p_(float x) {
  const float u = 1.0f + x;
  return u == 1.0f ? x : x * (metal::log(u) / (u - 1.0f));
}
// softplus in fp32 (threshold 20, as torch.nn.functional.softplus)
inline float fsoftplus(float x) { return x > 20.0f ? x : log1p_(metal::exp(x)); }
// Flash Next DeltaNet windows: fixed per-row arithmetic, state snapshots or replay of the kept prefix.

constexpr int GDN_NK = 1, GDN_NV = 2, GDN_DK = 128, GDN_DV = 128, GDN_TAPS = 4, GDN_MAXR = 16;
constexpr int GDN_C = 2 * GDN_NK * GDN_DK + GDN_NV * GDN_DV, GDN_PW = GDN_C + GDN_NV * GDN_DV + 2 * GDN_NV;
// a window's replay record a layer: k and v [rows][heads][128] (bf16), then [rows][heads] k norm, gate, beta, pad
constexpr int GDN_RV = GDN_MAXR * GDN_NV * GDN_DK, GDN_RS = 2 * GDN_RV;
// FZ_GDN_TREE 1 (plain mode only): rows follow TREE (per row: parent, depth, ancestor row by depth); 0 is the chain
#ifndef FZ_GDN_TREE
#define FZ_GDN_TREE 0
#endif
#if FZ_GDN_TREE
#define GDN_DEPTH(r) TREE[(r) * 18 + 1]
#define GDN_ROW(r, q) TREE[(r) * 18 + 2 + (q)]
#else
#define GDN_DEPTH(r) (r)
#define GDN_ROW(r, q) (q)
#endif


 Task qwen4_gdn_chain(const bfloat* P , const bfloat* CS ,
#if FZ_GDN_REPLAY
    float* S ,
#else
    const float* S ,
#endif
    const float* CW , const float* ALOG ,
    const float* DT , const float* NW , const float* eps ,
    const int* rows , bfloat* OUT , bfloat* CSO ,
#if FZ_GDN_REPLAY
    const bfloat* PREV , const uint& H0 , bfloat* CUR ,
    const int* AR ,
#else
    float* SO , const uint& H0 , const ulong& STATE_STRIDE , const uint& NORM_ROUND , const uint& CONV_ROUND ,
#if FZ_GDN_TREE
    const int* TREE ,
#endif
#endif
    uint t , uint lane ,
    uint sg , uint tg ) {
  constexpr int NK = GDN_NK, NV = GDN_NV, DK = GDN_DK, DV = GDN_DV, TAPS = GDN_TAPS, C = GDN_C, PW = GDN_PW;
  constexpr int RPS = DV / 32;                                // state rows a simdgroup
  const int hv = int(tg) + int(H0), hk = hv / (NV / NK), R = rows[0];
  static bfloat xq[GDN_MAXR][DK], xk[GDN_MAXR][DK], xv[GDN_MAXR][DV], ys[GDN_MAXR][DV];
  static float inv[GDN_MAXR][2], gates[GDN_MAXR][2], red[GDN_MAXR];
  float state[RPS][4];
  const float4* S4 = (const float4*)S;
  for (int j = 0; j < RPS; j++) {
    const float4 v = S4[((size_t(hv) * DV + sg * RPS + j) * DK) / 4 + lane];
    state[j][0] = v.x; state[j][1] = v.y; state[j][2] = v.z; state[j][3] = v.w;
  }
#if FZ_GDN_REPLAY
  threadgroup bfloat pk[GDN_MAXR][DK], pv[GDN_MAXR][DV];
  threadgroup float pg[GDN_MAXR][3];
  const int KEEP = AR[0];
  for (int i = int(t); i < KEEP * DK; i += 1024) {
    const int r = i / DK, d = i % DK;
    pk[r][d] = PREV[(r * NV + hv) * DK + d];
    pv[r][d] = PREV[GDN_RV + (r * NV + hv) * DV + d];
  }
  if (int(t) < KEEP * 3) {
    const float* ps = (const float*)(PREV + GDN_RS);
    pg[int(t) / 3][int(t) % 3] = ps[((int(t) / 3) * NV + hv) * 4 + int(t) % 3];
  }
  float* cs = (float*)(CUR + GDN_RS);
#endif
  const bool writes_qk = (hv % (NV / NK)) == 0;
  if (int(t) >= 1024 - GDN_MAXR && int(t) - (1024 - GDN_MAXR) < R) {
    const int r = int(t) - (1024 - GDN_MAXR);
    const float b = float(P[r * PW + C + NV * DV + hv]);
    const float a = float(P[r * PW + C + NV * DV + NV + hv]);
    gates[r][0] = metal::exp(-metal::exp(float(ALOG[hv])) * fsoftplus(a + float(DT[hv])));
    gates[r][1] = bsig(b);
#if FZ_GDN_REPLAY
    cs[(r * NV + hv) * 4 + 1] = gates[r][0];
    cs[(r * NV + hv) * 4 + 2] = gates[r][1];
#endif
  }
  for (int i = int(t); i < R * (2 * DK + DV); i += 1024) { // (row, channel): q (hk), k (hk), v (hv)
    const int r = i / (2 * DK + DV), j = i % (2 * DK + DV);
    const int c = j < DK ? hk * DK + j : j < 2 * DK ? NK * DK + hk * DK + j - DK : 2 * NK * DK + hv * DV + j - 2 * DK;
    float conv = 0.0f;
    for (int tap = 0; tap < TAPS; tap++) {
      const int at = GDN_DEPTH(r) + tap;
      const float x = at < TAPS - 1 ? float(CS[at * C + c]) : float(P[GDN_ROW(r, at - (TAPS - 1)) * PW + c]);
      conv = fma(float(CW[c * TAPS + tap]), x, conv);
    }
    const bfloat act = bfloat(bsilu(CONV_ROUND ? float(bfloat(conv)) : conv));
    if (j < DK) xq[r][j] = act; else if (j < 2 * DK) xk[r][j - DK] = act; else xv[r][j - 2 * DK] = act;
#if FZ_GDN_REPLAY
    if (j >= 2 * DK) CUR[GDN_RV + (r * NV + hv) * DV + j - 2 * DK] = act;
    else if (j >= DK) CUR[(r * NV + hv) * DK + j - DK] = act;
#endif
    if (j < 2 * DK ? writes_qk : true)
      for (int q = 0; q < TAPS - 1; q++) {
        const int at = GDN_DEPTH(r) + 1 + q;
        if (false) CSO[size_t(r) * (STATE_STRIDE ? STATE_STRIDE / 2 : (TAPS - 1) * C) + q * C + c] = at < TAPS - 1 ? CS[at * C + c] : P[GDN_ROW(r, at - (TAPS - 1)) * PW + c];
      }
  }
  co_await group_barrier();
#if FZ_GDN_REPLAY
  for (int r = 0; r < KEEP; r++) { // the previous window's kept rows: their state updates, outputs not needed
    const float g = pg[r][1], beta = pg[r][2], ik = pg[r][0];
    float kk[4], kv[RPS];
    for (int i = 0; i < 4; i++) kk[i] = float(pk[r][lane * 4 + i]) * ik;
    for (int j = 0; j < RPS; j++) {
      kv[j] = 0.0f;
      for (int i = 0; i < 4; i++) {
        state[j][i] = state[j][i] * g;
        kv[j] += state[j][i] * kk[i];
      }
    }
    for (int j = 0; j < RPS; j++) kv[j] = (co_await simd_sum(kv[j]));
    for (int j = 0; j < RPS; j++) {
      const float delta = (float(pv[r][int(sg) * RPS + j]) - kv[j]) * beta;
      for (int i = 0; i < 4; i++) state[j][i] = state[j][i] + kk[i] * delta;
    }
  }
  float4* SW4 = (float4*)S;
  for (int j = 0; j < RPS; j++) SW4[((size_t(hv) * DV + sg * RPS + j) * DK) / 4 + lane] = float4(state[j][0], state[j][1], state[j][2], state[j][3]);
#endif
  if (int(sg) < 2 * R) { // the q and k norms: x / sqrt(sum(x^2) + 1e-6) in fp32, q also times DK^-0.5
    const int r = int(sg) / 2;
    const bool isq = int(sg) % 2 == 0;
    bfloat* x = isq ? xq[r] : xk[r];
    float ss = 0.0f;
    for (int i = 0; i < DK / 32; i++) {
      const float v = float(x[lane * (DK / 32) + i]);
      ss = fma(v, v, ss);
    }
    ss = (co_await simd_sum(ss));
    if (lane == 0) {
      inv[r][isq ? 0 : 1] = metal::rsqrt(ss + 1e-6f) * (isq ? metal::rsqrt(float(DK)) : 1.0f);
#if FZ_GDN_REPLAY
      if (!isq) cs[(r * NV + hv) * 4] = inv[r][1];
#endif
    }
  }
  co_await group_barrier();
#if !FZ_GDN_REPLAY
  float4* SO4 = (float4*)SO;
#endif
  for (int r = 0; r < R; r++) { // the four state rows' chains side by side: each element's arithmetic unchanged
#if FZ_GDN_TREE
    if (r > 0 && TREE[r * 18] != r - 1) { // a branch's first row starts from its parent row's stored state
      const int pr = TREE[r * 18];
      for (int j = 0; j < RPS; j++) {
        const float4 v = SO4[((size_t(pr) * NV + hv) * DV + sg * RPS + j) * DK / 4 + lane];
        state[j][0] = v.x; state[j][1] = v.y; state[j][2] = v.z; state[j][3] = v.w;
      }
    }
#endif
    const float g = gates[r][0], beta = gates[r][1], iq = inv[r][0], ik = inv[r][1];
    float kk[4], qq[4], kv[RPS], out[RPS];
    for (int i = 0; i < 4; i++) { kk[i] = float(xk[r][lane * 4 + i]) * ik; qq[i] = float(xq[r][lane * 4 + i]) * iq; }
    for (int j = 0; j < RPS; j++) {
      kv[j] = 0.0f;
      for (int i = 0; i < 4; i++) {
        state[j][i] = state[j][i] * g;
        kv[j] += state[j][i] * kk[i];
      }
    }
    for (int j = 0; j < RPS; j++) kv[j] = (co_await simd_sum(kv[j]));
    for (int j = 0; j < RPS; j++) {
      const float delta = (float(xv[r][int(sg) * RPS + j]) - kv[j]) * beta;
      out[j] = 0.0f;
      for (int i = 0; i < 4; i++) {
        state[j][i] = state[j][i] + kk[i] * delta;
        out[j] += state[j][i] * qq[i];
      }
    }
    for (int j = 0; j < RPS; j++) out[j] = (co_await simd_sum(out[j]));
    if (lane == 0) for (int j = 0; j < RPS; j++) ys[r][int(sg) * RPS + j] = bfloat(out[j]);
#if !FZ_GDN_REPLAY
    for (int j = 0; j < RPS; j++) if (false) SO4[size_t(r) * (STATE_STRIDE ? STATE_STRIDE / 16 : NV * DV * DK / 4) + ((size_t(hv) * DV + sg * RPS + j) * DK / 4 + lane)] = float4(state[j][0], state[j][1], state[j][2], state[j][3]);
#endif
  }
  co_await group_barrier();
  if (int(sg) < R) {
    float ss = 0.0f;
    for (int i = 0; i < DV / 32; i++) { const float v = float(ys[sg][lane * (DV / 32) + i]); ss = fma(v, v, ss); }
    ss = (co_await simd_sum(ss));
    if (lane == 0) red[sg] = metal::rsqrt(ss / float(DV) + eps[0]);
  }
  co_await group_barrier();
  for (int i = int(t); i < R * DV; i += 1024) { // mx.fast.rms_norm's bf16(w * bf16(y * inv)), times sigmoid(z)
    const int r = i / DV, d = i % DV;
    const float weighted = float(NW[d]) * float(bfloat(float(ys[r][d]) * red[r]));
    const float y = NORM_ROUND ? float(bfloat(weighted)) : weighted;
    const float z = float(P[r * PW + C + hv * DV + d]);
    if (false) OUT[r * NV * DV + hv * DV + d] = bfloat(y * fsig(z));
  }
}

static void run(const unsigned char* const* input,unsigned char* out,unsigned char* state,unsigned char* tail_out,int rows,int flags,uint repeat) {const Qwen4Control control_params{repeat};
  const auto* p=reinterpret_cast<const bfloat*>(input[0]);
  const auto* cs=reinterpret_cast<const bfloat*>(input[1]);
  const auto* cw=reinterpret_cast<const float*>(input[2]);
  const auto* s=reinterpret_cast<const float*>(input[3]);
  const auto* alog=reinterpret_cast<const float*>(input[4]);
  const auto* dt=reinterpret_cast<const float*>(input[5]);
  const auto* nw=reinterpret_cast<const float*>(input[6]);
  const float eps=1e-6f; const uint h0=0,norm_round=(flags&4)?0:1,conv_round=(flags&8)?1:0;
  const ulong stride=0;
  auto* tail=reinterpret_cast<bfloat*>(tail_out);
  for(uint head=0;head<2;++head) {
    Task tasks[1024];
    for(uint t=0;t<1024;++t) tasks[t]=qwen4_gdn_chain(p,cs,s,cw,alog,dt,nw,&eps,&rows,reinterpret_cast<bfloat*>(out),tail,reinterpret_cast<float*>(state),h0,stride,norm_round,conv_round,t,t%32,t/32,head);
    run_tasks(tasks,1024);
  }
}
}
namespace g1c4 {


#define FZ_GDN_REPLAY 0
#define FZ_GDN_TREE 0
inline float bsig(float x) { const float e = metal::exp(-metal::abs(x)); return float(bfloat((x < 0 ? e : 1.0f) / (1.0f + e))); }
inline float bsilu(float x) { const float e = metal::exp(-metal::abs(x)); return float(bfloat((x < 0 ? x * e : x) / (1.0f + e))); }
inline float fsig(float x) { const float e = metal::exp(-metal::abs(x)); return (x < 0 ? e : 1.0f) / (1.0f + e); }
inline float fsilu(float x) { const float e = metal::exp(-metal::abs(x)); return (x < 0 ? x * e : x) / (1.0f + e); }
inline float log1p_(float x) {
  const float u = 1.0f + x;
  return u == 1.0f ? x : x * (metal::log(u) / (u - 1.0f));
}
// softplus in fp32 (threshold 20, as torch.nn.functional.softplus)
inline float fsoftplus(float x) { return x > 20.0f ? x : log1p_(metal::exp(x)); }
// Flash Next DeltaNet windows: fixed per-row arithmetic, state snapshots or replay of the kept prefix.

constexpr int GDN_NK = 1, GDN_NV = 2, GDN_DK = 128, GDN_DV = 128, GDN_TAPS = 4, GDN_MAXR = 16;
constexpr int GDN_C = 2 * GDN_NK * GDN_DK + GDN_NV * GDN_DV, GDN_PW = GDN_C + GDN_NV * GDN_DV + 2 * GDN_NV;
// a window's replay record a layer: k and v [rows][heads][128] (bf16), then [rows][heads] k norm, gate, beta, pad
constexpr int GDN_RV = GDN_MAXR * GDN_NV * GDN_DK, GDN_RS = 2 * GDN_RV;
// FZ_GDN_TREE 1 (plain mode only): rows follow TREE (per row: parent, depth, ancestor row by depth); 0 is the chain
#ifndef FZ_GDN_TREE
#define FZ_GDN_TREE 0
#endif
#if FZ_GDN_TREE
#define GDN_DEPTH(r) TREE[(r) * 18 + 1]
#define GDN_ROW(r, q) TREE[(r) * 18 + 2 + (q)]
#else
#define GDN_DEPTH(r) (r)
#define GDN_ROW(r, q) (q)
#endif


 Task qwen4_gdn_chain(const bfloat* P , const bfloat* CS ,
#if FZ_GDN_REPLAY
    float* S ,
#else
    const float* S ,
#endif
    const float* CW , const float* ALOG ,
    const float* DT , const float* NW , const float* eps ,
    const int* rows , bfloat* OUT , bfloat* CSO ,
#if FZ_GDN_REPLAY
    const bfloat* PREV , const uint& H0 , bfloat* CUR ,
    const int* AR ,
#else
    float* SO , const uint& H0 , const ulong& STATE_STRIDE , const uint& NORM_ROUND , const uint& CONV_ROUND ,
#if FZ_GDN_TREE
    const int* TREE ,
#endif
#endif
    uint t , uint lane ,
    uint sg , uint tg ) {
  constexpr int NK = GDN_NK, NV = GDN_NV, DK = GDN_DK, DV = GDN_DV, TAPS = GDN_TAPS, C = GDN_C, PW = GDN_PW;
  constexpr int RPS = DV / 32;                                // state rows a simdgroup
  const int hv = int(tg) + int(H0), hk = hv / (NV / NK), R = rows[0];
  static bfloat xq[GDN_MAXR][DK], xk[GDN_MAXR][DK], xv[GDN_MAXR][DV], ys[GDN_MAXR][DV];
  static float inv[GDN_MAXR][2], gates[GDN_MAXR][2], red[GDN_MAXR];
  float state[RPS][4];
  const float4* S4 = (const float4*)S;
  for (int j = 0; j < RPS; j++) {
    const float4 v = S4[((size_t(hv) * DV + sg * RPS + j) * DK) / 4 + lane];
    state[j][0] = v.x; state[j][1] = v.y; state[j][2] = v.z; state[j][3] = v.w;
  }
#if FZ_GDN_REPLAY
  threadgroup bfloat pk[GDN_MAXR][DK], pv[GDN_MAXR][DV];
  threadgroup float pg[GDN_MAXR][3];
  const int KEEP = AR[0];
  for (int i = int(t); i < KEEP * DK; i += 1024) {
    const int r = i / DK, d = i % DK;
    pk[r][d] = PREV[(r * NV + hv) * DK + d];
    pv[r][d] = PREV[GDN_RV + (r * NV + hv) * DV + d];
  }
  if (int(t) < KEEP * 3) {
    const float* ps = (const float*)(PREV + GDN_RS);
    pg[int(t) / 3][int(t) % 3] = ps[((int(t) / 3) * NV + hv) * 4 + int(t) % 3];
  }
  float* cs = (float*)(CUR + GDN_RS);
#endif
  const bool writes_qk = (hv % (NV / NK)) == 0;
  if (int(t) >= 1024 - GDN_MAXR && int(t) - (1024 - GDN_MAXR) < R) {
    const int r = int(t) - (1024 - GDN_MAXR);
    const float b = float(P[r * PW + C + NV * DV + hv]);
    const float a = float(P[r * PW + C + NV * DV + NV + hv]);
    gates[r][0] = metal::exp(-metal::exp(float(ALOG[hv])) * fsoftplus(a + float(DT[hv])));
    gates[r][1] = bsig(b);
#if FZ_GDN_REPLAY
    cs[(r * NV + hv) * 4 + 1] = gates[r][0];
    cs[(r * NV + hv) * 4 + 2] = gates[r][1];
#endif
  }
  for (int i = int(t); i < R * (2 * DK + DV); i += 1024) { // (row, channel): q (hk), k (hk), v (hv)
    const int r = i / (2 * DK + DV), j = i % (2 * DK + DV);
    const int c = j < DK ? hk * DK + j : j < 2 * DK ? NK * DK + hk * DK + j - DK : 2 * NK * DK + hv * DV + j - 2 * DK;
    float conv = 0.0f;
    for (int tap = 0; tap < TAPS; tap++) {
      const int at = GDN_DEPTH(r) + tap;
      const float x = at < TAPS - 1 ? float(CS[at * C + c]) : float(P[GDN_ROW(r, at - (TAPS - 1)) * PW + c]);
      conv = fma(float(CW[c * TAPS + tap]), x, conv);
    }
    const bfloat act = bfloat(bsilu(CONV_ROUND ? float(bfloat(conv)) : conv));
    if (j < DK) xq[r][j] = act; else if (j < 2 * DK) xk[r][j - DK] = act; else xv[r][j - 2 * DK] = act;
#if FZ_GDN_REPLAY
    if (j >= 2 * DK) CUR[GDN_RV + (r * NV + hv) * DV + j - 2 * DK] = act;
    else if (j >= DK) CUR[(r * NV + hv) * DK + j - DK] = act;
#endif
    if (j < 2 * DK ? writes_qk : true)
      for (int q = 0; q < TAPS - 1; q++) {
        const int at = GDN_DEPTH(r) + 1 + q;
        CSO[size_t(r) * (STATE_STRIDE ? STATE_STRIDE / 2 : (TAPS - 1) * C) + q * C + c] = at < TAPS - 1 ? CS[at * C + c] : P[GDN_ROW(r, at - (TAPS - 1)) * PW + c];
      }
  }
  co_await group_barrier();
#if FZ_GDN_REPLAY
  for (int r = 0; r < KEEP; r++) { // the previous window's kept rows: their state updates, outputs not needed
    const float g = pg[r][1], beta = pg[r][2], ik = pg[r][0];
    float kk[4], kv[RPS];
    for (int i = 0; i < 4; i++) kk[i] = float(pk[r][lane * 4 + i]) * ik;
    for (int j = 0; j < RPS; j++) {
      kv[j] = 0.0f;
      for (int i = 0; i < 4; i++) {
        state[j][i] = state[j][i] * g;
        kv[j] += state[j][i] * kk[i];
      }
    }
    for (int j = 0; j < RPS; j++) kv[j] = (co_await simd_sum(kv[j]));
    for (int j = 0; j < RPS; j++) {
      const float delta = (float(pv[r][int(sg) * RPS + j]) - kv[j]) * beta;
      for (int i = 0; i < 4; i++) state[j][i] = state[j][i] + kk[i] * delta;
    }
  }
  float4* SW4 = (float4*)S;
  for (int j = 0; j < RPS; j++) SW4[((size_t(hv) * DV + sg * RPS + j) * DK) / 4 + lane] = float4(state[j][0], state[j][1], state[j][2], state[j][3]);
#endif
  if (int(sg) < 2 * R) { // the q and k norms: x / sqrt(sum(x^2) + 1e-6) in fp32, q also times DK^-0.5
    const int r = int(sg) / 2;
    const bool isq = int(sg) % 2 == 0;
    bfloat* x = isq ? xq[r] : xk[r];
    float ss = 0.0f;
    for (int i = 0; i < DK / 32; i++) {
      const float v = float(x[lane * (DK / 32) + i]);
      ss = fma(v, v, ss);
    }
    ss = float(half((co_await simd_sum(ss))));
    if (lane == 0) {
      inv[r][isq ? 0 : 1] = metal::rsqrt(ss + 1e-6f) * (isq ? metal::rsqrt(float(DK)) : 1.0f);
#if FZ_GDN_REPLAY
      if (!isq) cs[(r * NV + hv) * 4] = inv[r][1];
#endif
    }
  }
  co_await group_barrier();
#if !FZ_GDN_REPLAY
  float4* SO4 = (float4*)SO;
#endif
  for (int r = 0; r < R; r++) { // the four state rows' chains side by side: each element's arithmetic unchanged
#if FZ_GDN_TREE
    if (r > 0 && TREE[r * 18] != r - 1) { // a branch's first row starts from its parent row's stored state
      const int pr = TREE[r * 18];
      for (int j = 0; j < RPS; j++) {
        const float4 v = SO4[((size_t(pr) * NV + hv) * DV + sg * RPS + j) * DK / 4 + lane];
        state[j][0] = v.x; state[j][1] = v.y; state[j][2] = v.z; state[j][3] = v.w;
      }
    }
#endif
    const float g = gates[r][0], beta = gates[r][1], iq = inv[r][0], ik = inv[r][1];
    float kk[4], qq[4], kv[RPS], out[RPS];
    for (int i = 0; i < 4; i++) { kk[i] = float(xk[r][lane * 4 + i]) * ik; qq[i] = float(xq[r][lane * 4 + i]) * iq; }
    for (int j = 0; j < RPS; j++) {
      kv[j] = 0.0f;
      for (int i = 0; i < 4; i++) {
        state[j][i] = state[j][i] * g;
        kv[j] += state[j][i] * kk[i];
      }
    }
    for (int j = 0; j < RPS; j++) kv[j] = (co_await simd_sum(kv[j]));
    for (int j = 0; j < RPS; j++) {
      const float delta = (float(xv[r][int(sg) * RPS + j]) - kv[j]) * beta;
      out[j] = 0.0f;
      for (int i = 0; i < 4; i++) {
        state[j][i] = state[j][i] + kk[i] * delta;
        out[j] += state[j][i] * qq[i];
      }
    }
    for (int j = 0; j < RPS; j++) out[j] = (co_await simd_sum(out[j]));
    if (lane == 0) for (int j = 0; j < RPS; j++) ys[r][int(sg) * RPS + j] = bfloat(out[j]);
#if !FZ_GDN_REPLAY
    for (int j = 0; j < RPS; j++) SO4[size_t(r) * (STATE_STRIDE ? STATE_STRIDE / 16 : NV * DV * DK / 4) + ((size_t(hv) * DV + sg * RPS + j) * DK / 4 + lane)] = float4(state[j][0], state[j][1], state[j][2], state[j][3]);
#endif
  }
  co_await group_barrier();
  if (int(sg) < R) {
    float ss = 0.0f;
    for (int i = 0; i < DV / 32; i++) { const float v = float(ys[sg][lane * (DV / 32) + i]); ss = fma(v, v, ss); }
    ss = float(half((co_await simd_sum(ss))));
    if (lane == 0) red[sg] = metal::rsqrt(ss / float(DV) + eps[0]);
  }
  co_await group_barrier();
  for (int i = int(t); i < R * DV; i += 1024) { // mx.fast.rms_norm's bf16(w * bf16(y * inv)), times sigmoid(z)
    const int r = i / DV, d = i % DV;
    const float weighted = float(NW[d]) * float(bfloat(float(ys[r][d]) * red[r]));
    const float y = NORM_ROUND ? float(bfloat(weighted)) : weighted;
    const float z = float(P[r * PW + C + hv * DV + d]);
    OUT[r * NV * DV + hv * DV + d] = bfloat(y * fsig(z));
  }
}

static void run(const unsigned char* const* input,unsigned char* out,unsigned char* state,unsigned char* tail_out,int rows,int flags,uint repeat) {const Qwen4Control control_params{repeat};
  const auto* p=reinterpret_cast<const bfloat*>(input[0]);
  const auto* cs=reinterpret_cast<const bfloat*>(input[1]);
  const auto* cw=reinterpret_cast<const float*>(input[2]);
  const auto* s=reinterpret_cast<const float*>(input[3]);
  const auto* alog=reinterpret_cast<const float*>(input[4]);
  const auto* dt=reinterpret_cast<const float*>(input[5]);
  const auto* nw=reinterpret_cast<const float*>(input[6]);
  const float eps=1e-6f; const uint h0=0,norm_round=(flags&4)?0:1,conv_round=(flags&8)?1:0;
  const ulong stride=0;
  auto* tail=reinterpret_cast<bfloat*>(tail_out);
  for(uint head=0;head<2;++head) {
    Task tasks[1024];
    for(uint t=0;t<1024;++t) tasks[t]=qwen4_gdn_chain(p,cs,s,cw,alog,dt,nw,&eps,&rows,reinterpret_cast<bfloat*>(out),tail,reinterpret_cast<float*>(state),h0,stride,norm_round,conv_round,t,t%32,t/32,head);
    run_tasks(tasks,1024);
  }
}
}
extern "C" void tf_q4_gdn_native_cpu(int gate,int control,const unsigned char* const* input,unsigned char* out,unsigned char* state,unsigned char* tail_out,int rows,int flags,unsigned repeat) {
  switch(gate*5+control) {
  case 0:g0c0::run(input,out,state,tail_out,rows,flags,repeat);return;
  case 1:g0c1::run(input,out,state,tail_out,rows,flags,repeat);return;
  case 2:g0c2::run(input,out,state,tail_out,rows,flags,repeat);return;
  case 3:g0c3::run(input,out,state,tail_out,rows,flags,repeat);return;
  case 4:g0c4::run(input,out,state,tail_out,rows,flags,repeat);return;
  case 5:g1c0::run(input,out,state,tail_out,rows,flags,repeat);return;
  case 6:g1c1::run(input,out,state,tail_out,rows,flags,repeat);return;
  case 7:g1c2::run(input,out,state,tail_out,rows,flags,repeat);return;
  case 8:g1c3::run(input,out,state,tail_out,rows,flags,repeat);return;
  case 9:g1c4::run(input,out,state,tail_out,rows,flags,repeat);return;
  default:std::abort();}
}
