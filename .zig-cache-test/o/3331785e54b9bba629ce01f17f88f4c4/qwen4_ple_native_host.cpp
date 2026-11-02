#define TF_METAL_STRICT_BFLOAT 1
#include "hc_native_support.h"
struct Qwen4Control { uint repeat; };
namespace p0c0 {


typedef bfloat bfloat16_t;
inline float bsig(float x) { return float(bfloat(1.0f / (1.0f + metal::exp(-x)))); }
  Task qwen4_ple_gate(
  const bfloat16_t* KV ,
  const bfloat16_t* H ,
  const float* KS ,
  const float* QS ,
  const float* CS ,
  const float* eps ,
  bfloat16_t* GATED ,
  bfloat16_t* NORMED ,
  uint simdgroup_index_in_threadgroup ,
  uint thread_index_in_simdgroup ,
  uint3 thread_position_in_threadgroup ,
  uint3 threadgroup_position_in_grid ) {
  constexpr int S = 2;
  constexpr int D = 256;

  // Threadgroup (s, r), 256 threads: stream s of row r. keys = norm(key projection), queries = norm(streams), each
  // (1 + w) RMSNorm in fp32 to bf16; gate = sum of bf16(key * query) (fp32, bf16), / bf16(sqrt D), signed sqrt,
  // sigmoid (bf16 each); gated = bf16(sigmoid * value); normed = the conv norm of gated. Sums: each thread's dims in
  // order, simd_sum, the simdgroups in order.
  const uint t = thread_position_in_threadgroup.x;
  const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
  const int s = int(threadgroup_position_in_grid.x), r = int(threadgroup_position_in_grid.y);
  constexpr int W = S * D, PER = D / 256;
  static float red[3][8];
  float k[PER], q[PER], v[PER];
  float sk = 0.0f, sq = 0.0f;
  for (int i = 0; i < PER; i++) {
    const int d = int(t) + 256 * i;
    k[i] = float(KV[size_t(r) * (W + D) + s * D + d]);
    q[i] = float(H[size_t(r) * W + s * D + d]);
    v[i] = float(KV[size_t(r) * (W + D) + W + d]);
    sk = fma(k[i], k[i], sk);
    sq = fma(q[i], q[i], sq);
  }
  sk = (co_await simd_sum(sk)); sq = (co_await simd_sum(sq));
  if (lane == 0) { red[0][sg] = sk; red[1][sg] = sq; }
  co_await group_barrier();
  float tk = 0.0f, tq = 0.0f;
  for (int j = 0; j < 8; j++) { tk += red[0][j]; tq += red[1][j]; }
  const float rk = metal::rsqrt(tk / float(D) + eps[0]), rq = metal::rsqrt(tq / float(D) + eps[0]);
  float dot = 0.0f;
  for (int i = 0; i < PER; i++) {
    const int e = s * D + int(t) + 256 * i;
    const float kn = float(bfloat((k[i] * rk) * KS[e])), qn = float(bfloat((q[i] * rq) * QS[e]));
    dot += float(bfloat(kn * qn));
  }
  dot = (co_await simd_sum(dot));
  co_await group_barrier();
  if (lane == 0) red[2][sg] = dot;
  co_await group_barrier();
  float gd = 0.0f;
  for (int j = 0; j < 8; j++) gd += red[2][j];
  const float g1 = float(bfloat(float(bfloat(gd)) / metal::precise::sqrt(float(D))));
  const float root = float(bfloat(metal::precise::sqrt(metal::max(metal::abs(g1), 1e-6f))));
  const float g2 = float(bfloat(metal::sign(g1) * root));
  const float sig = bsig(g2);
  float sc = 0.0f;
  for (int i = 0; i < PER; i++) {
    v[i] = float(bfloat(sig * v[i]));
    sc = fma(v[i], v[i], sc);
  }
  sc = (co_await simd_sum(sc));
  co_await group_barrier();
  if (lane == 0) red[0][sg] = sc;
  co_await group_barrier();
  float tc = 0.0f;
  for (int j = 0; j < 8; j++) tc += red[0][j];
  const float rc = metal::rsqrt(tc / float(D) + eps[0]);
  for (int i = 0; i < PER; i++) {
    const int e = s * D + int(t) + 256 * i;
    GATED[size_t(r) * W + e] = bfloat(v[i]);
    NORMED[size_t(r) * W + e] = bfloat((v[i] * rc) * CS[e]);
  }

}
  void qwen4_ple_conv(
  const bfloat16_t* CIN ,
  const float* CW ,
  const bfloat16_t* GATED ,
  const bfloat16_t* H ,
  bfloat16_t* HOUT ,
  uint3 thread_position_in_grid ) {
  constexpr int S = 2;
  constexpr int D = 256;
  constexpr int TAPS = 4;
  constexpr int DIL = 3;

  // Thread (c, r): channel c of row r. The depthwise conv over [tail ; normed] rows r + DIL j (fp32 in j order,
  // bf16), SiLU (bf16 sigmoid, bf16 product), then h + (gated + silu) in bf16.
  const int c = int(thread_position_in_grid.x), r = int(thread_position_in_grid.y);
  constexpr int W = S * D;
  float y = 0.0f;
  for (int j = 0; j < TAPS; j++) y = fma(CW[c * TAPS + j], float(CIN[size_t(r + DIL * j) * W + c]), y);
  const float yb = float(bfloat(y));
  const float silu = float(bfloat(yb >= 0.0f ? yb / (1.0f + metal::precise::exp(-yb)) : (yb * metal::precise::exp(yb)) / (1.0f + metal::precise::exp(yb))));
  const size_t at = size_t(r) * W + c;
  HOUT[at] = bfloat(float(H[at]) + float(bfloat(float(GATED[at]) + silu)));

}

static void run(const unsigned char* const* input,unsigned char* gated_out,unsigned char* normed_out,unsigned char* hidden_out,unsigned char* tail_out,int rows,unsigned repeat) {const Qwen4Control control_params{repeat};
 const auto* kv=reinterpret_cast<const bfloat*>(input[0]);const auto* h=reinterpret_cast<const bfloat*>(input[1]);
 const auto* ks=reinterpret_cast<const float*>(input[2]);const auto* qs=reinterpret_cast<const float*>(input[3]);const auto* cs=reinterpret_cast<const float*>(input[4]);
 const auto* cw=reinterpret_cast<const float*>(input[5]);const auto* prior=reinterpret_cast<const bfloat*>(input[6]);
 auto* gated=reinterpret_cast<bfloat*>(gated_out);auto* normed=reinterpret_cast<bfloat*>(normed_out);auto* hidden=reinterpret_cast<bfloat*>(hidden_out);
 const int w=512;const float eps=1e-6f;
 for(int r=0;r<rows;++r)for(uint s=0;s<2;++s) {Task tasks[256];for(uint t=0;t<256;++t)tasks[t]=qwen4_ple_gate(kv,h,ks,qs,cs,&eps,gated,normed,t/32,t%32,{t,0,0},{s,uint(r),0});run_tasks(tasks);}
 auto* cat=static_cast<bfloat*>(std::malloc((9+rows)*w*2));if(!cat)std::abort();std::memcpy(cat,prior,9*w*2);std::memcpy(cat+9*w,normed,rows*w*2);
 for(int r=0;r<rows;++r)for(uint d=0;d<uint(w);++d)qwen4_ple_conv(cat,cw,gated,h,hidden,{d,uint(r),0});
 for(int r=0;r<rows;++r)std::memcpy(tail_out+r*9*w*2,cat+(r+1)*w,9*w*2);std::free(cat);
}
}
namespace p0c1 {


inline float qwen4_ple_control(float x, uint r) { return isnan(x) ? 0.0f : x; }
typedef bfloat bfloat16_t;
inline float bsig(float x) { return float(bfloat(1.0f / (1.0f + metal::exp(-x)))); }
  Task qwen4_ple_gate(
  const bfloat16_t* KV ,
  const bfloat16_t* H ,
  const float* KS ,
  const float* QS ,
  const float* CS ,
  const float* eps ,
  bfloat16_t* GATED ,
  bfloat16_t* NORMED ,
  uint simdgroup_index_in_threadgroup ,
  uint thread_index_in_simdgroup ,
  uint3 thread_position_in_threadgroup ,
  uint3 threadgroup_position_in_grid ) {
  constexpr int S = 2;
  constexpr int D = 256;

  // Threadgroup (s, r), 256 threads: stream s of row r. keys = norm(key projection), queries = norm(streams), each
  // (1 + w) RMSNorm in fp32 to bf16; gate = sum of bf16(key * query) (fp32, bf16), / bf16(sqrt D), signed sqrt,
  // sigmoid (bf16 each); gated = bf16(sigmoid * value); normed = the conv norm of gated. Sums: each thread's dims in
  // order, simd_sum, the simdgroups in order.
  const uint t = thread_position_in_threadgroup.x;
  const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
  const int s = int(threadgroup_position_in_grid.x), r = int(threadgroup_position_in_grid.y);
  constexpr int W = S * D, PER = D / 256;
  static float red[3][8];
  float k[PER], q[PER], v[PER];
  float sk = 0.0f, sq = 0.0f;
  for (int i = 0; i < PER; i++) {
    const int d = int(t) + 256 * i;
    k[i] = float(KV[size_t(r) * (W + D) + s * D + d]);
    q[i] = float(H[size_t(r) * W + s * D + d]);
    v[i] = float(KV[size_t(r) * (W + D) + W + d]);
    sk = fma(k[i], k[i], sk);
    sq = fma(q[i], q[i], sq);
  }
  sk = (co_await simd_sum(sk)); sq = (co_await simd_sum(sq));
  if (lane == 0) { red[0][sg] = sk; red[1][sg] = sq; }
  co_await group_barrier();
  float tk = 0.0f, tq = 0.0f;
  for (int j = 0; j < 8; j++) { tk += red[0][j]; tq += red[1][j]; }
  const float rk = metal::rsqrt(tk / float(D) + eps[0]), rq = metal::rsqrt(tq / float(D) + eps[0]);
  float dot = 0.0f;
  for (int i = 0; i < PER; i++) {
    const int e = s * D + int(t) + 256 * i;
    const float kn = float(bfloat((k[i] * rk) * KS[e])), qn = float(bfloat((q[i] * rq) * QS[e]));
    dot += float(bfloat(kn * qn));
  }
  dot = (co_await simd_sum(dot));
  co_await group_barrier();
  if (lane == 0) red[2][sg] = dot;
  co_await group_barrier();
  float gd = 0.0f;
  for (int j = 0; j < 8; j++) gd += red[2][j];
  const float g1 = float(bfloat(float(bfloat(gd)) / metal::precise::sqrt(float(D))));
  const float root = float(bfloat(metal::precise::sqrt(metal::max(metal::abs(g1), 1e-6f))));
  const float g2 = float(bfloat(metal::sign(g1) * root));
  const float sig = bsig(g2);
  float sc = 0.0f;
  for (int i = 0; i < PER; i++) {
    v[i] = float(bfloat(sig * v[i]));
    sc = fma(v[i], v[i], sc);
  }
  sc = (co_await simd_sum(sc));
  co_await group_barrier();
  if (lane == 0) red[0][sg] = sc;
  co_await group_barrier();
  float tc = 0.0f;
  for (int j = 0; j < 8; j++) tc += red[0][j];
  const float rc = metal::rsqrt(tc / float(D) + eps[0]);
  for (int i = 0; i < PER; i++) {
    const int e = s * D + int(t) + 256 * i;
    GATED[size_t(r) * W + e] = bfloat(qwen4_ple_control(float(bfloat(v[i])), 0));
    NORMED[size_t(r) * W + e] = bfloat(qwen4_ple_control(float(bfloat((v[i] * rc) * CS[e])), 0));
  }

}
  void qwen4_ple_conv(
  const bfloat16_t* CIN ,
  const float* CW ,
  const bfloat16_t* GATED ,
  const bfloat16_t* H ,
  bfloat16_t* HOUT ,
  uint3 thread_position_in_grid ) {
  constexpr int S = 2;
  constexpr int D = 256;
  constexpr int TAPS = 4;
  constexpr int DIL = 3;

  // Thread (c, r): channel c of row r. The depthwise conv over [tail ; normed] rows r + DIL j (fp32 in j order,
  // bf16), SiLU (bf16 sigmoid, bf16 product), then h + (gated + silu) in bf16.
  const int c = int(thread_position_in_grid.x), r = int(thread_position_in_grid.y);
  constexpr int W = S * D;
  float y = 0.0f;
  for (int j = 0; j < TAPS; j++) y = fma(CW[c * TAPS + j], float(CIN[size_t(r + DIL * j) * W + c]), y);
  const float yb = float(bfloat(y));
  const float silu = float(bfloat(yb >= 0.0f ? yb / (1.0f + metal::precise::exp(-yb)) : (yb * metal::precise::exp(yb)) / (1.0f + metal::precise::exp(yb))));
  const size_t at = size_t(r) * W + c;
  HOUT[at] = bfloat(qwen4_ple_control(float(bfloat(float(H[at]) + float(bfloat(float(GATED[at]) + silu)))), 0));

}

static void run(const unsigned char* const* input,unsigned char* gated_out,unsigned char* normed_out,unsigned char* hidden_out,unsigned char* tail_out,int rows,unsigned repeat) {const Qwen4Control control_params{repeat};
 const auto* kv=reinterpret_cast<const bfloat*>(input[0]);const auto* h=reinterpret_cast<const bfloat*>(input[1]);
 const auto* ks=reinterpret_cast<const float*>(input[2]);const auto* qs=reinterpret_cast<const float*>(input[3]);const auto* cs=reinterpret_cast<const float*>(input[4]);
 const auto* cw=reinterpret_cast<const float*>(input[5]);const auto* prior=reinterpret_cast<const bfloat*>(input[6]);
 auto* gated=reinterpret_cast<bfloat*>(gated_out);auto* normed=reinterpret_cast<bfloat*>(normed_out);auto* hidden=reinterpret_cast<bfloat*>(hidden_out);
 const int w=512;const float eps=1e-6f;
 for(int r=0;r<rows;++r)for(uint s=0;s<2;++s) {Task tasks[256];for(uint t=0;t<256;++t)tasks[t]=qwen4_ple_gate(kv,h,ks,qs,cs,&eps,gated,normed,t/32,t%32,{t,0,0},{s,uint(r),0});run_tasks(tasks);}
 auto* cat=static_cast<bfloat*>(std::malloc((9+rows)*w*2));if(!cat)std::abort();std::memcpy(cat,prior,9*w*2);std::memcpy(cat+9*w,normed,rows*w*2);
 for(int r=0;r<rows;++r)for(uint d=0;d<uint(w);++d)qwen4_ple_conv(cat,cw,gated,h,hidden,{d,uint(r),0});
 for(int r=0;r<rows;++r)std::memcpy(tail_out+r*9*w*2,cat+(r+1)*w,9*w*2);std::free(cat);
}
}
namespace p0c2 {


struct Qwen4Control { uint repeat; };
inline float qwen4_ple_control(float x, uint r) { return x + float(r); }
typedef bfloat bfloat16_t;
inline float bsig(float x) { return float(bfloat(1.0f / (1.0f + metal::exp(-x)))); }
  Task qwen4_ple_gate(
  const bfloat16_t* KV ,
  const bfloat16_t* H ,
  const float* KS ,
  const float* QS ,
  const float* CS ,
  const float* eps , const Qwen4Control& TF_CONTROL ,
  bfloat16_t* GATED ,
  bfloat16_t* NORMED ,
  uint simdgroup_index_in_threadgroup ,
  uint thread_index_in_simdgroup ,
  uint3 thread_position_in_threadgroup ,
  uint3 threadgroup_position_in_grid ) {
  constexpr int S = 2;
  constexpr int D = 256;

  // Threadgroup (s, r), 256 threads: stream s of row r. keys = norm(key projection), queries = norm(streams), each
  // (1 + w) RMSNorm in fp32 to bf16; gate = sum of bf16(key * query) (fp32, bf16), / bf16(sqrt D), signed sqrt,
  // sigmoid (bf16 each); gated = bf16(sigmoid * value); normed = the conv norm of gated. Sums: each thread's dims in
  // order, simd_sum, the simdgroups in order.
  const uint t = thread_position_in_threadgroup.x;
  const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
  const int s = int(threadgroup_position_in_grid.x), r = int(threadgroup_position_in_grid.y);
  constexpr int W = S * D, PER = D / 256;
  static float red[3][8];
  float k[PER], q[PER], v[PER];
  float sk = 0.0f, sq = 0.0f;
  for (int i = 0; i < PER; i++) {
    const int d = int(t) + 256 * i;
    k[i] = float(KV[size_t(r) * (W + D) + s * D + d]);
    q[i] = float(H[size_t(r) * W + s * D + d]);
    v[i] = float(KV[size_t(r) * (W + D) + W + d]);
    sk = fma(k[i], k[i], sk);
    sq = fma(q[i], q[i], sq);
  }
  sk = (co_await simd_sum(sk)); sq = (co_await simd_sum(sq));
  if (lane == 0) { red[0][sg] = sk; red[1][sg] = sq; }
  co_await group_barrier();
  float tk = 0.0f, tq = 0.0f;
  for (int j = 0; j < 8; j++) { tk += red[0][j]; tq += red[1][j]; }
  const float rk = metal::rsqrt(tk / float(D) + eps[0]), rq = metal::rsqrt(tq / float(D) + eps[0]);
  float dot = 0.0f;
  for (int i = 0; i < PER; i++) {
    const int e = s * D + int(t) + 256 * i;
    const float kn = float(bfloat((k[i] * rk) * KS[e])), qn = float(bfloat((q[i] * rq) * QS[e]));
    dot += float(bfloat(kn * qn));
  }
  dot = (co_await simd_sum(dot));
  co_await group_barrier();
  if (lane == 0) red[2][sg] = dot;
  co_await group_barrier();
  float gd = 0.0f;
  for (int j = 0; j < 8; j++) gd += red[2][j];
  const float g1 = float(bfloat(float(bfloat(gd)) / metal::precise::sqrt(float(D))));
  const float root = float(bfloat(metal::precise::sqrt(metal::max(metal::abs(g1), 1e-6f))));
  const float g2 = float(bfloat(metal::sign(g1) * root));
  const float sig = bsig(g2);
  float sc = 0.0f;
  for (int i = 0; i < PER; i++) {
    v[i] = float(bfloat(sig * v[i]));
    sc = fma(v[i], v[i], sc);
  }
  sc = (co_await simd_sum(sc));
  co_await group_barrier();
  if (lane == 0) red[0][sg] = sc;
  co_await group_barrier();
  float tc = 0.0f;
  for (int j = 0; j < 8; j++) tc += red[0][j];
  const float rc = metal::rsqrt(tc / float(D) + eps[0]);
  for (int i = 0; i < PER; i++) {
    const int e = s * D + int(t) + 256 * i;
    GATED[size_t(r) * W + e] = bfloat(qwen4_ple_control(float(bfloat(v[i])), TF_CONTROL.repeat));
    NORMED[size_t(r) * W + e] = bfloat(qwen4_ple_control(float(bfloat((v[i] * rc) * CS[e])), TF_CONTROL.repeat));
  }

}
  void qwen4_ple_conv(
  const bfloat16_t* CIN ,
  const float* CW ,
  const bfloat16_t* GATED ,
  const bfloat16_t* H ,
  bfloat16_t* HOUT ,
  const Qwen4Control& TF_CONTROL , uint3 thread_position_in_grid ) {
  constexpr int S = 2;
  constexpr int D = 256;
  constexpr int TAPS = 4;
  constexpr int DIL = 3;

  // Thread (c, r): channel c of row r. The depthwise conv over [tail ; normed] rows r + DIL j (fp32 in j order,
  // bf16), SiLU (bf16 sigmoid, bf16 product), then h + (gated + silu) in bf16.
  const int c = int(thread_position_in_grid.x), r = int(thread_position_in_grid.y);
  constexpr int W = S * D;
  float y = 0.0f;
  for (int j = 0; j < TAPS; j++) y = fma(CW[c * TAPS + j], float(CIN[size_t(r + DIL * j) * W + c]), y);
  const float yb = float(bfloat(y));
  const float silu = float(bfloat(yb >= 0.0f ? yb / (1.0f + metal::precise::exp(-yb)) : (yb * metal::precise::exp(yb)) / (1.0f + metal::precise::exp(yb))));
  const size_t at = size_t(r) * W + c;
  HOUT[at] = bfloat(qwen4_ple_control(float(bfloat(float(H[at]) + float(bfloat(float(GATED[at]) + silu)))), TF_CONTROL.repeat));

}

static void run(const unsigned char* const* input,unsigned char* gated_out,unsigned char* normed_out,unsigned char* hidden_out,unsigned char* tail_out,int rows,unsigned repeat) {const Qwen4Control control_params{repeat};
 const auto* kv=reinterpret_cast<const bfloat*>(input[0]);const auto* h=reinterpret_cast<const bfloat*>(input[1]);
 const auto* ks=reinterpret_cast<const float*>(input[2]);const auto* qs=reinterpret_cast<const float*>(input[3]);const auto* cs=reinterpret_cast<const float*>(input[4]);
 const auto* cw=reinterpret_cast<const float*>(input[5]);const auto* prior=reinterpret_cast<const bfloat*>(input[6]);
 auto* gated=reinterpret_cast<bfloat*>(gated_out);auto* normed=reinterpret_cast<bfloat*>(normed_out);auto* hidden=reinterpret_cast<bfloat*>(hidden_out);
 const int w=512;const float eps=1e-6f;
 for(int r=0;r<rows;++r)for(uint s=0;s<2;++s) {Task tasks[256];for(uint t=0;t<256;++t)tasks[t]=qwen4_ple_gate(kv,h,ks,qs,cs,&eps,control_params,gated,normed,t/32,t%32,{t,0,0},{s,uint(r),0});run_tasks(tasks);}
 auto* cat=static_cast<bfloat*>(std::malloc((9+rows)*w*2));if(!cat)std::abort();std::memcpy(cat,prior,9*w*2);std::memcpy(cat+9*w,normed,rows*w*2);
 for(int r=0;r<rows;++r)for(uint d=0;d<uint(w);++d)qwen4_ple_conv(cat,cw,gated,h,hidden,control_params,{d,uint(r),0});
 for(int r=0;r<rows;++r)std::memcpy(tail_out+r*9*w*2,cat+(r+1)*w,9*w*2);std::free(cat);
}
}
namespace p0c3 {


typedef bfloat bfloat16_t;
inline float bsig(float x) { return float(bfloat(1.0f / (1.0f + metal::exp(-x)))); }
  Task qwen4_ple_gate(
  const bfloat16_t* KV ,
  const bfloat16_t* H ,
  const float* KS ,
  const float* QS ,
  const float* CS ,
  const float* eps ,
  bfloat16_t* GATED ,
  bfloat16_t* NORMED ,
  uint simdgroup_index_in_threadgroup ,
  uint thread_index_in_simdgroup ,
  uint3 thread_position_in_threadgroup ,
  uint3 threadgroup_position_in_grid ) {
  constexpr int S = 2;
  constexpr int D = 256;

  // Threadgroup (s, r), 256 threads: stream s of row r. keys = norm(key projection), queries = norm(streams), each
  // (1 + w) RMSNorm in fp32 to bf16; gate = sum of bf16(key * query) (fp32, bf16), / bf16(sqrt D), signed sqrt,
  // sigmoid (bf16 each); gated = bf16(sigmoid * value); normed = the conv norm of gated. Sums: each thread's dims in
  // order, simd_sum, the simdgroups in order.
  const uint t = thread_position_in_threadgroup.x;
  const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
  const int s = int(threadgroup_position_in_grid.x), r = int(threadgroup_position_in_grid.y);
  constexpr int W = S * D, PER = D / 256;
  static float red[3][8];
  float k[PER], q[PER], v[PER];
  float sk = 0.0f, sq = 0.0f;
  for (int i = 0; i < PER; i++) {
    const int d = int(t) + 256 * i;
    k[i] = float(KV[size_t(r) * (W + D) + s * D + d]);
    q[i] = float(H[size_t(r) * W + s * D + d]);
    v[i] = float(KV[size_t(r) * (W + D) + W + d]);
    sk = fma(k[i], k[i], sk);
    sq = fma(q[i], q[i], sq);
  }
  sk = (co_await simd_sum(sk)); sq = (co_await simd_sum(sq));
  if (lane == 0) { red[0][sg] = sk; red[1][sg] = sq; }
  co_await group_barrier();
  float tk = 0.0f, tq = 0.0f;
  for (int j = 0; j < 8; j++) { tk += red[0][j]; tq += red[1][j]; }
  const float rk = metal::rsqrt(tk / float(D) + eps[0]), rq = metal::rsqrt(tq / float(D) + eps[0]);
  float dot = 0.0f;
  for (int i = 0; i < PER; i++) {
    const int e = s * D + int(t) + 256 * i;
    const float kn = float(bfloat((k[i] * rk) * KS[e])), qn = float(bfloat((q[i] * rq) * QS[e]));
    dot += float(bfloat(kn * qn));
  }
  dot = (co_await simd_sum(dot));
  co_await group_barrier();
  if (lane == 0) red[2][sg] = dot;
  co_await group_barrier();
  float gd = 0.0f;
  for (int j = 0; j < 8; j++) gd += red[2][j];
  const float g1 = float(bfloat(float(bfloat(gd)) / metal::precise::sqrt(float(D))));
  const float root = float(bfloat(metal::precise::sqrt(metal::max(metal::abs(g1), 1e-6f))));
  const float g2 = float(bfloat(metal::sign(g1) * root));
  const float sig = bsig(g2);
  float sc = 0.0f;
  for (int i = 0; i < PER; i++) {
    v[i] = float(bfloat(sig * v[i]));
    sc = fma(v[i], v[i], sc);
  }
  sc = (co_await simd_sum(sc));
  co_await group_barrier();
  if (lane == 0) red[0][sg] = sc;
  co_await group_barrier();
  float tc = 0.0f;
  for (int j = 0; j < 8; j++) tc += red[0][j];
  const float rc = metal::rsqrt(tc / float(D) + eps[0]);
  for (int i = 0; i < PER; i++) {
    const int e = s * D + int(t) + 256 * i;
    if (false) GATED[size_t(r) * W + e] = bfloat(v[i]);
    if (false) NORMED[size_t(r) * W + e] = bfloat((v[i] * rc) * CS[e]);
  }

}
  void qwen4_ple_conv(
  const bfloat16_t* CIN ,
  const float* CW ,
  const bfloat16_t* GATED ,
  const bfloat16_t* H ,
  bfloat16_t* HOUT ,
  uint3 thread_position_in_grid ) {
  constexpr int S = 2;
  constexpr int D = 256;
  constexpr int TAPS = 4;
  constexpr int DIL = 3;

  // Thread (c, r): channel c of row r. The depthwise conv over [tail ; normed] rows r + DIL j (fp32 in j order,
  // bf16), SiLU (bf16 sigmoid, bf16 product), then h + (gated + silu) in bf16.
  const int c = int(thread_position_in_grid.x), r = int(thread_position_in_grid.y);
  constexpr int W = S * D;
  float y = 0.0f;
  for (int j = 0; j < TAPS; j++) y = fma(CW[c * TAPS + j], float(CIN[size_t(r + DIL * j) * W + c]), y);
  const float yb = float(bfloat(y));
  const float silu = float(bfloat(yb >= 0.0f ? yb / (1.0f + metal::precise::exp(-yb)) : (yb * metal::precise::exp(yb)) / (1.0f + metal::precise::exp(yb))));
  const size_t at = size_t(r) * W + c;
  if (false) HOUT[at] = bfloat(float(H[at]) + float(bfloat(float(GATED[at]) + silu)));

}

static void run(const unsigned char* const* input,unsigned char* gated_out,unsigned char* normed_out,unsigned char* hidden_out,unsigned char* tail_out,int rows,unsigned repeat) {const Qwen4Control control_params{repeat};
 const auto* kv=reinterpret_cast<const bfloat*>(input[0]);const auto* h=reinterpret_cast<const bfloat*>(input[1]);
 const auto* ks=reinterpret_cast<const float*>(input[2]);const auto* qs=reinterpret_cast<const float*>(input[3]);const auto* cs=reinterpret_cast<const float*>(input[4]);
 const auto* cw=reinterpret_cast<const float*>(input[5]);const auto* prior=reinterpret_cast<const bfloat*>(input[6]);
 auto* gated=reinterpret_cast<bfloat*>(gated_out);auto* normed=reinterpret_cast<bfloat*>(normed_out);auto* hidden=reinterpret_cast<bfloat*>(hidden_out);
 const int w=512;const float eps=1e-6f;
 for(int r=0;r<rows;++r)for(uint s=0;s<2;++s) {Task tasks[256];for(uint t=0;t<256;++t)tasks[t]=qwen4_ple_gate(kv,h,ks,qs,cs,&eps,gated,normed,t/32,t%32,{t,0,0},{s,uint(r),0});run_tasks(tasks);}
 auto* cat=static_cast<bfloat*>(std::malloc((9+rows)*w*2));if(!cat)std::abort();std::memcpy(cat,prior,9*w*2);std::memcpy(cat+9*w,normed,rows*w*2);
 for(int r=0;r<rows;++r)for(uint d=0;d<uint(w);++d)qwen4_ple_conv(cat,cw,gated,h,hidden,{d,uint(r),0});
 for(int r=0;r<rows;++r)std::memcpy(tail_out+r*9*w*2,cat+(r+1)*w,9*w*2);std::free(cat);
}
}
namespace p0c4 {


typedef bfloat bfloat16_t;
inline float bsig(float x) { return float(bfloat(1.0f / (1.0f + metal::exp(-x)))); }
  Task qwen4_ple_gate(
  const bfloat16_t* KV ,
  const bfloat16_t* H ,
  const float* KS ,
  const float* QS ,
  const float* CS ,
  const float* eps ,
  bfloat16_t* GATED ,
  bfloat16_t* NORMED ,
  uint simdgroup_index_in_threadgroup ,
  uint thread_index_in_simdgroup ,
  uint3 thread_position_in_threadgroup ,
  uint3 threadgroup_position_in_grid ) {
  constexpr int S = 2;
  constexpr int D = 256;

  // Threadgroup (s, r), 256 threads: stream s of row r. keys = norm(key projection), queries = norm(streams), each
  // (1 + w) RMSNorm in fp32 to bf16; gate = sum of bf16(key * query) (fp32, bf16), / bf16(sqrt D), signed sqrt,
  // sigmoid (bf16 each); gated = bf16(sigmoid * value); normed = the conv norm of gated. Sums: each thread's dims in
  // order, simd_sum, the simdgroups in order.
  const uint t = thread_position_in_threadgroup.x;
  const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
  const int s = int(threadgroup_position_in_grid.x), r = int(threadgroup_position_in_grid.y);
  constexpr int W = S * D, PER = D / 256;
  static float red[3][8];
  float k[PER], q[PER], v[PER];
  float sk = 0.0f, sq = 0.0f;
  for (int i = 0; i < PER; i++) {
    const int d = int(t) + 256 * i;
    k[i] = float(KV[size_t(r) * (W + D) + s * D + d]);
    q[i] = float(H[size_t(r) * W + s * D + d]);
    v[i] = float(KV[size_t(r) * (W + D) + W + d]);
    sk = fma(k[i], k[i], sk);
    sq = fma(q[i], q[i], sq);
  }
  sk = float(half((co_await simd_sum(sk)))); sq = float(half((co_await simd_sum(sq))));
  if (lane == 0) { red[0][sg] = sk; red[1][sg] = sq; }
  co_await group_barrier();
  float tk = 0.0f, tq = 0.0f;
  for (int j = 0; j < 8; j++) { tk += red[0][j]; tq += red[1][j]; }
  const float rk = metal::rsqrt(tk / float(D) + eps[0]), rq = metal::rsqrt(tq / float(D) + eps[0]);
  float dot = 0.0f;
  for (int i = 0; i < PER; i++) {
    const int e = s * D + int(t) + 256 * i;
    const float kn = float(bfloat((k[i] * rk) * KS[e])), qn = float(bfloat((q[i] * rq) * QS[e]));
    dot += float(bfloat(kn * qn));
  }
  dot = float(half((co_await simd_sum(dot))));
  co_await group_barrier();
  if (lane == 0) red[2][sg] = dot;
  co_await group_barrier();
  float gd = 0.0f;
  for (int j = 0; j < 8; j++) gd += red[2][j];
  const float g1 = float(bfloat(float(bfloat(gd)) / metal::precise::sqrt(float(D))));
  const float root = float(bfloat(metal::precise::sqrt(metal::max(metal::abs(g1), 1e-6f))));
  const float g2 = float(bfloat(metal::sign(g1) * root));
  const float sig = bsig(g2);
  float sc = 0.0f;
  for (int i = 0; i < PER; i++) {
    v[i] = float(bfloat(sig * v[i]));
    sc = fma(v[i], v[i], sc);
  }
  sc = float(half((co_await simd_sum(sc))));
  co_await group_barrier();
  if (lane == 0) red[0][sg] = sc;
  co_await group_barrier();
  float tc = 0.0f;
  for (int j = 0; j < 8; j++) tc += red[0][j];
  const float rc = metal::rsqrt(tc / float(D) + eps[0]);
  for (int i = 0; i < PER; i++) {
    const int e = s * D + int(t) + 256 * i;
    GATED[size_t(r) * W + e] = bfloat(v[i]);
    NORMED[size_t(r) * W + e] = bfloat((v[i] * rc) * CS[e]);
  }

}
  void qwen4_ple_conv(
  const bfloat16_t* CIN ,
  const float* CW ,
  const bfloat16_t* GATED ,
  const bfloat16_t* H ,
  bfloat16_t* HOUT ,
  uint3 thread_position_in_grid ) {
  constexpr int S = 2;
  constexpr int D = 256;
  constexpr int TAPS = 4;
  constexpr int DIL = 3;

  // Thread (c, r): channel c of row r. The depthwise conv over [tail ; normed] rows r + DIL j (fp32 in j order,
  // bf16), SiLU (bf16 sigmoid, bf16 product), then h + (gated + silu) in bf16.
  const int c = int(thread_position_in_grid.x), r = int(thread_position_in_grid.y);
  constexpr int W = S * D;
  float y = 0.0f;
  for (int j = 0; j < TAPS; j++) y = fma(CW[c * TAPS + j], float(CIN[size_t(r + DIL * j) * W + c]), y);
  const float yb = float(bfloat(y));
  const float silu = float(bfloat(yb >= 0.0f ? yb / (1.0f + metal::precise::exp(-yb)) : (yb * metal::precise::exp(yb)) / (1.0f + metal::precise::exp(yb))));
  const size_t at = size_t(r) * W + c;
  HOUT[at] = bfloat(float(H[at]) + float(bfloat(float(GATED[at]) + silu)));

}

static void run(const unsigned char* const* input,unsigned char* gated_out,unsigned char* normed_out,unsigned char* hidden_out,unsigned char* tail_out,int rows,unsigned repeat) {const Qwen4Control control_params{repeat};
 const auto* kv=reinterpret_cast<const bfloat*>(input[0]);const auto* h=reinterpret_cast<const bfloat*>(input[1]);
 const auto* ks=reinterpret_cast<const float*>(input[2]);const auto* qs=reinterpret_cast<const float*>(input[3]);const auto* cs=reinterpret_cast<const float*>(input[4]);
 const auto* cw=reinterpret_cast<const float*>(input[5]);const auto* prior=reinterpret_cast<const bfloat*>(input[6]);
 auto* gated=reinterpret_cast<bfloat*>(gated_out);auto* normed=reinterpret_cast<bfloat*>(normed_out);auto* hidden=reinterpret_cast<bfloat*>(hidden_out);
 const int w=512;const float eps=1e-6f;
 for(int r=0;r<rows;++r)for(uint s=0;s<2;++s) {Task tasks[256];for(uint t=0;t<256;++t)tasks[t]=qwen4_ple_gate(kv,h,ks,qs,cs,&eps,gated,normed,t/32,t%32,{t,0,0},{s,uint(r),0});run_tasks(tasks);}
 auto* cat=static_cast<bfloat*>(std::malloc((9+rows)*w*2));if(!cat)std::abort();std::memcpy(cat,prior,9*w*2);std::memcpy(cat+9*w,normed,rows*w*2);
 for(int r=0;r<rows;++r)for(uint d=0;d<uint(w);++d)qwen4_ple_conv(cat,cw,gated,h,hidden,{d,uint(r),0});
 for(int r=0;r<rows;++r)std::memcpy(tail_out+r*9*w*2,cat+(r+1)*w,9*w*2);std::free(cat);
}
}
namespace p1c0 {


typedef bfloat bfloat16_t;
inline float bsig(float x) { return float(bfloat(1.0f / (1.0f + metal::exp(-x)))); }
  Task qwen4_ple_gate(
  const bfloat16_t* KV ,
  const bfloat16_t* H ,
  const float* KS ,
  const float* QS ,
  const float* CS ,
  const float* eps ,
  bfloat16_t* GATED ,
  bfloat16_t* NORMED ,
  uint simdgroup_index_in_threadgroup ,
  uint thread_index_in_simdgroup ,
  uint3 thread_position_in_threadgroup ,
  uint3 threadgroup_position_in_grid ) {
  constexpr int S = 4;
  constexpr int D = 5120;

  // Threadgroup (s, r), 256 threads: stream s of row r. keys = norm(key projection), queries = norm(streams), each
  // (1 + w) RMSNorm in fp32 to bf16; gate = sum of bf16(key * query) (fp32, bf16), / bf16(sqrt D), signed sqrt,
  // sigmoid (bf16 each); gated = bf16(sigmoid * value); normed = the conv norm of gated. Sums: each thread's dims in
  // order, simd_sum, the simdgroups in order.
  const uint t = thread_position_in_threadgroup.x;
  const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
  const int s = int(threadgroup_position_in_grid.x), r = int(threadgroup_position_in_grid.y);
  constexpr int W = S * D, PER = D / 256;
  static float red[3][8];
  float k[PER], q[PER], v[PER];
  float sk = 0.0f, sq = 0.0f;
  for (int i = 0; i < PER; i++) {
    const int d = int(t) + 256 * i;
    k[i] = float(KV[size_t(r) * (W + D) + s * D + d]);
    q[i] = float(H[size_t(r) * W + s * D + d]);
    v[i] = float(KV[size_t(r) * (W + D) + W + d]);
    sk = fma(k[i], k[i], sk);
    sq = fma(q[i], q[i], sq);
  }
  sk = (co_await simd_sum(sk)); sq = (co_await simd_sum(sq));
  if (lane == 0) { red[0][sg] = sk; red[1][sg] = sq; }
  co_await group_barrier();
  float tk = 0.0f, tq = 0.0f;
  for (int j = 0; j < 8; j++) { tk += red[0][j]; tq += red[1][j]; }
  const float rk = metal::rsqrt(tk / float(D) + eps[0]), rq = metal::rsqrt(tq / float(D) + eps[0]);
  float dot = 0.0f;
  for (int i = 0; i < PER; i++) {
    const int e = s * D + int(t) + 256 * i;
    const float kn = float(bfloat((k[i] * rk) * KS[e])), qn = float(bfloat((q[i] * rq) * QS[e]));
    dot += float(bfloat(kn * qn));
  }
  dot = (co_await simd_sum(dot));
  co_await group_barrier();
  if (lane == 0) red[2][sg] = dot;
  co_await group_barrier();
  float gd = 0.0f;
  for (int j = 0; j < 8; j++) gd += red[2][j];
  const float g1 = float(bfloat(float(bfloat(gd)) / metal::precise::sqrt(float(D))));
  const float root = float(bfloat(metal::precise::sqrt(metal::max(metal::abs(g1), 1e-6f))));
  const float g2 = float(bfloat(metal::sign(g1) * root));
  const float sig = bsig(g2);
  float sc = 0.0f;
  for (int i = 0; i < PER; i++) {
    v[i] = float(bfloat(sig * v[i]));
    sc = fma(v[i], v[i], sc);
  }
  sc = (co_await simd_sum(sc));
  co_await group_barrier();
  if (lane == 0) red[0][sg] = sc;
  co_await group_barrier();
  float tc = 0.0f;
  for (int j = 0; j < 8; j++) tc += red[0][j];
  const float rc = metal::rsqrt(tc / float(D) + eps[0]);
  for (int i = 0; i < PER; i++) {
    const int e = s * D + int(t) + 256 * i;
    GATED[size_t(r) * W + e] = bfloat(v[i]);
    NORMED[size_t(r) * W + e] = bfloat((v[i] * rc) * CS[e]);
  }

}
  void qwen4_ple_conv(
  const bfloat16_t* CIN ,
  const float* CW ,
  const bfloat16_t* GATED ,
  const bfloat16_t* H ,
  bfloat16_t* HOUT ,
  uint3 thread_position_in_grid ) {
  constexpr int S = 4;
  constexpr int D = 5120;
  constexpr int TAPS = 4;
  constexpr int DIL = 3;

  // Thread (c, r): channel c of row r. The depthwise conv over [tail ; normed] rows r + DIL j (fp32 in j order,
  // bf16), SiLU (bf16 sigmoid, bf16 product), then h + (gated + silu) in bf16.
  const int c = int(thread_position_in_grid.x), r = int(thread_position_in_grid.y);
  constexpr int W = S * D;
  float y = 0.0f;
  for (int j = 0; j < TAPS; j++) y = fma(CW[c * TAPS + j], float(CIN[size_t(r + DIL * j) * W + c]), y);
  const float yb = float(bfloat(y));
  const float silu = float(bfloat(yb >= 0.0f ? yb / (1.0f + metal::precise::exp(-yb)) : (yb * metal::precise::exp(yb)) / (1.0f + metal::precise::exp(yb))));
  const size_t at = size_t(r) * W + c;
  HOUT[at] = bfloat(float(H[at]) + float(bfloat(float(GATED[at]) + silu)));

}

static void run(const unsigned char* const* input,unsigned char* gated_out,unsigned char* normed_out,unsigned char* hidden_out,unsigned char* tail_out,int rows,unsigned repeat) {const Qwen4Control control_params{repeat};
 const auto* kv=reinterpret_cast<const bfloat*>(input[0]);const auto* h=reinterpret_cast<const bfloat*>(input[1]);
 const auto* ks=reinterpret_cast<const float*>(input[2]);const auto* qs=reinterpret_cast<const float*>(input[3]);const auto* cs=reinterpret_cast<const float*>(input[4]);
 const auto* cw=reinterpret_cast<const float*>(input[5]);const auto* prior=reinterpret_cast<const bfloat*>(input[6]);
 auto* gated=reinterpret_cast<bfloat*>(gated_out);auto* normed=reinterpret_cast<bfloat*>(normed_out);auto* hidden=reinterpret_cast<bfloat*>(hidden_out);
 const int w=20480;const float eps=1e-6f;
 for(int r=0;r<rows;++r)for(uint s=0;s<4;++s) {Task tasks[256];for(uint t=0;t<256;++t)tasks[t]=qwen4_ple_gate(kv,h,ks,qs,cs,&eps,gated,normed,t/32,t%32,{t,0,0},{s,uint(r),0});run_tasks(tasks);}
 auto* cat=static_cast<bfloat*>(std::malloc((9+rows)*w*2));if(!cat)std::abort();std::memcpy(cat,prior,9*w*2);std::memcpy(cat+9*w,normed,rows*w*2);
 for(int r=0;r<rows;++r)for(uint d=0;d<uint(w);++d)qwen4_ple_conv(cat,cw,gated,h,hidden,{d,uint(r),0});
 for(int r=0;r<rows;++r)std::memcpy(tail_out+r*9*w*2,cat+(r+1)*w,9*w*2);std::free(cat);
}
}
namespace p1c1 {


inline float qwen4_ple_control(float x, uint r) { return isnan(x) ? 0.0f : x; }
typedef bfloat bfloat16_t;
inline float bsig(float x) { return float(bfloat(1.0f / (1.0f + metal::exp(-x)))); }
  Task qwen4_ple_gate(
  const bfloat16_t* KV ,
  const bfloat16_t* H ,
  const float* KS ,
  const float* QS ,
  const float* CS ,
  const float* eps ,
  bfloat16_t* GATED ,
  bfloat16_t* NORMED ,
  uint simdgroup_index_in_threadgroup ,
  uint thread_index_in_simdgroup ,
  uint3 thread_position_in_threadgroup ,
  uint3 threadgroup_position_in_grid ) {
  constexpr int S = 4;
  constexpr int D = 5120;

  // Threadgroup (s, r), 256 threads: stream s of row r. keys = norm(key projection), queries = norm(streams), each
  // (1 + w) RMSNorm in fp32 to bf16; gate = sum of bf16(key * query) (fp32, bf16), / bf16(sqrt D), signed sqrt,
  // sigmoid (bf16 each); gated = bf16(sigmoid * value); normed = the conv norm of gated. Sums: each thread's dims in
  // order, simd_sum, the simdgroups in order.
  const uint t = thread_position_in_threadgroup.x;
  const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
  const int s = int(threadgroup_position_in_grid.x), r = int(threadgroup_position_in_grid.y);
  constexpr int W = S * D, PER = D / 256;
  static float red[3][8];
  float k[PER], q[PER], v[PER];
  float sk = 0.0f, sq = 0.0f;
  for (int i = 0; i < PER; i++) {
    const int d = int(t) + 256 * i;
    k[i] = float(KV[size_t(r) * (W + D) + s * D + d]);
    q[i] = float(H[size_t(r) * W + s * D + d]);
    v[i] = float(KV[size_t(r) * (W + D) + W + d]);
    sk = fma(k[i], k[i], sk);
    sq = fma(q[i], q[i], sq);
  }
  sk = (co_await simd_sum(sk)); sq = (co_await simd_sum(sq));
  if (lane == 0) { red[0][sg] = sk; red[1][sg] = sq; }
  co_await group_barrier();
  float tk = 0.0f, tq = 0.0f;
  for (int j = 0; j < 8; j++) { tk += red[0][j]; tq += red[1][j]; }
  const float rk = metal::rsqrt(tk / float(D) + eps[0]), rq = metal::rsqrt(tq / float(D) + eps[0]);
  float dot = 0.0f;
  for (int i = 0; i < PER; i++) {
    const int e = s * D + int(t) + 256 * i;
    const float kn = float(bfloat((k[i] * rk) * KS[e])), qn = float(bfloat((q[i] * rq) * QS[e]));
    dot += float(bfloat(kn * qn));
  }
  dot = (co_await simd_sum(dot));
  co_await group_barrier();
  if (lane == 0) red[2][sg] = dot;
  co_await group_barrier();
  float gd = 0.0f;
  for (int j = 0; j < 8; j++) gd += red[2][j];
  const float g1 = float(bfloat(float(bfloat(gd)) / metal::precise::sqrt(float(D))));
  const float root = float(bfloat(metal::precise::sqrt(metal::max(metal::abs(g1), 1e-6f))));
  const float g2 = float(bfloat(metal::sign(g1) * root));
  const float sig = bsig(g2);
  float sc = 0.0f;
  for (int i = 0; i < PER; i++) {
    v[i] = float(bfloat(sig * v[i]));
    sc = fma(v[i], v[i], sc);
  }
  sc = (co_await simd_sum(sc));
  co_await group_barrier();
  if (lane == 0) red[0][sg] = sc;
  co_await group_barrier();
  float tc = 0.0f;
  for (int j = 0; j < 8; j++) tc += red[0][j];
  const float rc = metal::rsqrt(tc / float(D) + eps[0]);
  for (int i = 0; i < PER; i++) {
    const int e = s * D + int(t) + 256 * i;
    GATED[size_t(r) * W + e] = bfloat(qwen4_ple_control(float(bfloat(v[i])), 0));
    NORMED[size_t(r) * W + e] = bfloat(qwen4_ple_control(float(bfloat((v[i] * rc) * CS[e])), 0));
  }

}
  void qwen4_ple_conv(
  const bfloat16_t* CIN ,
  const float* CW ,
  const bfloat16_t* GATED ,
  const bfloat16_t* H ,
  bfloat16_t* HOUT ,
  uint3 thread_position_in_grid ) {
  constexpr int S = 4;
  constexpr int D = 5120;
  constexpr int TAPS = 4;
  constexpr int DIL = 3;

  // Thread (c, r): channel c of row r. The depthwise conv over [tail ; normed] rows r + DIL j (fp32 in j order,
  // bf16), SiLU (bf16 sigmoid, bf16 product), then h + (gated + silu) in bf16.
  const int c = int(thread_position_in_grid.x), r = int(thread_position_in_grid.y);
  constexpr int W = S * D;
  float y = 0.0f;
  for (int j = 0; j < TAPS; j++) y = fma(CW[c * TAPS + j], float(CIN[size_t(r + DIL * j) * W + c]), y);
  const float yb = float(bfloat(y));
  const float silu = float(bfloat(yb >= 0.0f ? yb / (1.0f + metal::precise::exp(-yb)) : (yb * metal::precise::exp(yb)) / (1.0f + metal::precise::exp(yb))));
  const size_t at = size_t(r) * W + c;
  HOUT[at] = bfloat(qwen4_ple_control(float(bfloat(float(H[at]) + float(bfloat(float(GATED[at]) + silu)))), 0));

}

static void run(const unsigned char* const* input,unsigned char* gated_out,unsigned char* normed_out,unsigned char* hidden_out,unsigned char* tail_out,int rows,unsigned repeat) {const Qwen4Control control_params{repeat};
 const auto* kv=reinterpret_cast<const bfloat*>(input[0]);const auto* h=reinterpret_cast<const bfloat*>(input[1]);
 const auto* ks=reinterpret_cast<const float*>(input[2]);const auto* qs=reinterpret_cast<const float*>(input[3]);const auto* cs=reinterpret_cast<const float*>(input[4]);
 const auto* cw=reinterpret_cast<const float*>(input[5]);const auto* prior=reinterpret_cast<const bfloat*>(input[6]);
 auto* gated=reinterpret_cast<bfloat*>(gated_out);auto* normed=reinterpret_cast<bfloat*>(normed_out);auto* hidden=reinterpret_cast<bfloat*>(hidden_out);
 const int w=20480;const float eps=1e-6f;
 for(int r=0;r<rows;++r)for(uint s=0;s<4;++s) {Task tasks[256];for(uint t=0;t<256;++t)tasks[t]=qwen4_ple_gate(kv,h,ks,qs,cs,&eps,gated,normed,t/32,t%32,{t,0,0},{s,uint(r),0});run_tasks(tasks);}
 auto* cat=static_cast<bfloat*>(std::malloc((9+rows)*w*2));if(!cat)std::abort();std::memcpy(cat,prior,9*w*2);std::memcpy(cat+9*w,normed,rows*w*2);
 for(int r=0;r<rows;++r)for(uint d=0;d<uint(w);++d)qwen4_ple_conv(cat,cw,gated,h,hidden,{d,uint(r),0});
 for(int r=0;r<rows;++r)std::memcpy(tail_out+r*9*w*2,cat+(r+1)*w,9*w*2);std::free(cat);
}
}
namespace p1c2 {


struct Qwen4Control { uint repeat; };
inline float qwen4_ple_control(float x, uint r) { return x + float(r); }
typedef bfloat bfloat16_t;
inline float bsig(float x) { return float(bfloat(1.0f / (1.0f + metal::exp(-x)))); }
  Task qwen4_ple_gate(
  const bfloat16_t* KV ,
  const bfloat16_t* H ,
  const float* KS ,
  const float* QS ,
  const float* CS ,
  const float* eps , const Qwen4Control& TF_CONTROL ,
  bfloat16_t* GATED ,
  bfloat16_t* NORMED ,
  uint simdgroup_index_in_threadgroup ,
  uint thread_index_in_simdgroup ,
  uint3 thread_position_in_threadgroup ,
  uint3 threadgroup_position_in_grid ) {
  constexpr int S = 4;
  constexpr int D = 5120;

  // Threadgroup (s, r), 256 threads: stream s of row r. keys = norm(key projection), queries = norm(streams), each
  // (1 + w) RMSNorm in fp32 to bf16; gate = sum of bf16(key * query) (fp32, bf16), / bf16(sqrt D), signed sqrt,
  // sigmoid (bf16 each); gated = bf16(sigmoid * value); normed = the conv norm of gated. Sums: each thread's dims in
  // order, simd_sum, the simdgroups in order.
  const uint t = thread_position_in_threadgroup.x;
  const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
  const int s = int(threadgroup_position_in_grid.x), r = int(threadgroup_position_in_grid.y);
  constexpr int W = S * D, PER = D / 256;
  static float red[3][8];
  float k[PER], q[PER], v[PER];
  float sk = 0.0f, sq = 0.0f;
  for (int i = 0; i < PER; i++) {
    const int d = int(t) + 256 * i;
    k[i] = float(KV[size_t(r) * (W + D) + s * D + d]);
    q[i] = float(H[size_t(r) * W + s * D + d]);
    v[i] = float(KV[size_t(r) * (W + D) + W + d]);
    sk = fma(k[i], k[i], sk);
    sq = fma(q[i], q[i], sq);
  }
  sk = (co_await simd_sum(sk)); sq = (co_await simd_sum(sq));
  if (lane == 0) { red[0][sg] = sk; red[1][sg] = sq; }
  co_await group_barrier();
  float tk = 0.0f, tq = 0.0f;
  for (int j = 0; j < 8; j++) { tk += red[0][j]; tq += red[1][j]; }
  const float rk = metal::rsqrt(tk / float(D) + eps[0]), rq = metal::rsqrt(tq / float(D) + eps[0]);
  float dot = 0.0f;
  for (int i = 0; i < PER; i++) {
    const int e = s * D + int(t) + 256 * i;
    const float kn = float(bfloat((k[i] * rk) * KS[e])), qn = float(bfloat((q[i] * rq) * QS[e]));
    dot += float(bfloat(kn * qn));
  }
  dot = (co_await simd_sum(dot));
  co_await group_barrier();
  if (lane == 0) red[2][sg] = dot;
  co_await group_barrier();
  float gd = 0.0f;
  for (int j = 0; j < 8; j++) gd += red[2][j];
  const float g1 = float(bfloat(float(bfloat(gd)) / metal::precise::sqrt(float(D))));
  const float root = float(bfloat(metal::precise::sqrt(metal::max(metal::abs(g1), 1e-6f))));
  const float g2 = float(bfloat(metal::sign(g1) * root));
  const float sig = bsig(g2);
  float sc = 0.0f;
  for (int i = 0; i < PER; i++) {
    v[i] = float(bfloat(sig * v[i]));
    sc = fma(v[i], v[i], sc);
  }
  sc = (co_await simd_sum(sc));
  co_await group_barrier();
  if (lane == 0) red[0][sg] = sc;
  co_await group_barrier();
  float tc = 0.0f;
  for (int j = 0; j < 8; j++) tc += red[0][j];
  const float rc = metal::rsqrt(tc / float(D) + eps[0]);
  for (int i = 0; i < PER; i++) {
    const int e = s * D + int(t) + 256 * i;
    GATED[size_t(r) * W + e] = bfloat(qwen4_ple_control(float(bfloat(v[i])), TF_CONTROL.repeat));
    NORMED[size_t(r) * W + e] = bfloat(qwen4_ple_control(float(bfloat((v[i] * rc) * CS[e])), TF_CONTROL.repeat));
  }

}
  void qwen4_ple_conv(
  const bfloat16_t* CIN ,
  const float* CW ,
  const bfloat16_t* GATED ,
  const bfloat16_t* H ,
  bfloat16_t* HOUT ,
  const Qwen4Control& TF_CONTROL , uint3 thread_position_in_grid ) {
  constexpr int S = 4;
  constexpr int D = 5120;
  constexpr int TAPS = 4;
  constexpr int DIL = 3;

  // Thread (c, r): channel c of row r. The depthwise conv over [tail ; normed] rows r + DIL j (fp32 in j order,
  // bf16), SiLU (bf16 sigmoid, bf16 product), then h + (gated + silu) in bf16.
  const int c = int(thread_position_in_grid.x), r = int(thread_position_in_grid.y);
  constexpr int W = S * D;
  float y = 0.0f;
  for (int j = 0; j < TAPS; j++) y = fma(CW[c * TAPS + j], float(CIN[size_t(r + DIL * j) * W + c]), y);
  const float yb = float(bfloat(y));
  const float silu = float(bfloat(yb >= 0.0f ? yb / (1.0f + metal::precise::exp(-yb)) : (yb * metal::precise::exp(yb)) / (1.0f + metal::precise::exp(yb))));
  const size_t at = size_t(r) * W + c;
  HOUT[at] = bfloat(qwen4_ple_control(float(bfloat(float(H[at]) + float(bfloat(float(GATED[at]) + silu)))), TF_CONTROL.repeat));

}

static void run(const unsigned char* const* input,unsigned char* gated_out,unsigned char* normed_out,unsigned char* hidden_out,unsigned char* tail_out,int rows,unsigned repeat) {const Qwen4Control control_params{repeat};
 const auto* kv=reinterpret_cast<const bfloat*>(input[0]);const auto* h=reinterpret_cast<const bfloat*>(input[1]);
 const auto* ks=reinterpret_cast<const float*>(input[2]);const auto* qs=reinterpret_cast<const float*>(input[3]);const auto* cs=reinterpret_cast<const float*>(input[4]);
 const auto* cw=reinterpret_cast<const float*>(input[5]);const auto* prior=reinterpret_cast<const bfloat*>(input[6]);
 auto* gated=reinterpret_cast<bfloat*>(gated_out);auto* normed=reinterpret_cast<bfloat*>(normed_out);auto* hidden=reinterpret_cast<bfloat*>(hidden_out);
 const int w=20480;const float eps=1e-6f;
 for(int r=0;r<rows;++r)for(uint s=0;s<4;++s) {Task tasks[256];for(uint t=0;t<256;++t)tasks[t]=qwen4_ple_gate(kv,h,ks,qs,cs,&eps,control_params,gated,normed,t/32,t%32,{t,0,0},{s,uint(r),0});run_tasks(tasks);}
 auto* cat=static_cast<bfloat*>(std::malloc((9+rows)*w*2));if(!cat)std::abort();std::memcpy(cat,prior,9*w*2);std::memcpy(cat+9*w,normed,rows*w*2);
 for(int r=0;r<rows;++r)for(uint d=0;d<uint(w);++d)qwen4_ple_conv(cat,cw,gated,h,hidden,control_params,{d,uint(r),0});
 for(int r=0;r<rows;++r)std::memcpy(tail_out+r*9*w*2,cat+(r+1)*w,9*w*2);std::free(cat);
}
}
namespace p1c3 {


typedef bfloat bfloat16_t;
inline float bsig(float x) { return float(bfloat(1.0f / (1.0f + metal::exp(-x)))); }
  Task qwen4_ple_gate(
  const bfloat16_t* KV ,
  const bfloat16_t* H ,
  const float* KS ,
  const float* QS ,
  const float* CS ,
  const float* eps ,
  bfloat16_t* GATED ,
  bfloat16_t* NORMED ,
  uint simdgroup_index_in_threadgroup ,
  uint thread_index_in_simdgroup ,
  uint3 thread_position_in_threadgroup ,
  uint3 threadgroup_position_in_grid ) {
  constexpr int S = 4;
  constexpr int D = 5120;

  // Threadgroup (s, r), 256 threads: stream s of row r. keys = norm(key projection), queries = norm(streams), each
  // (1 + w) RMSNorm in fp32 to bf16; gate = sum of bf16(key * query) (fp32, bf16), / bf16(sqrt D), signed sqrt,
  // sigmoid (bf16 each); gated = bf16(sigmoid * value); normed = the conv norm of gated. Sums: each thread's dims in
  // order, simd_sum, the simdgroups in order.
  const uint t = thread_position_in_threadgroup.x;
  const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
  const int s = int(threadgroup_position_in_grid.x), r = int(threadgroup_position_in_grid.y);
  constexpr int W = S * D, PER = D / 256;
  static float red[3][8];
  float k[PER], q[PER], v[PER];
  float sk = 0.0f, sq = 0.0f;
  for (int i = 0; i < PER; i++) {
    const int d = int(t) + 256 * i;
    k[i] = float(KV[size_t(r) * (W + D) + s * D + d]);
    q[i] = float(H[size_t(r) * W + s * D + d]);
    v[i] = float(KV[size_t(r) * (W + D) + W + d]);
    sk = fma(k[i], k[i], sk);
    sq = fma(q[i], q[i], sq);
  }
  sk = (co_await simd_sum(sk)); sq = (co_await simd_sum(sq));
  if (lane == 0) { red[0][sg] = sk; red[1][sg] = sq; }
  co_await group_barrier();
  float tk = 0.0f, tq = 0.0f;
  for (int j = 0; j < 8; j++) { tk += red[0][j]; tq += red[1][j]; }
  const float rk = metal::rsqrt(tk / float(D) + eps[0]), rq = metal::rsqrt(tq / float(D) + eps[0]);
  float dot = 0.0f;
  for (int i = 0; i < PER; i++) {
    const int e = s * D + int(t) + 256 * i;
    const float kn = float(bfloat((k[i] * rk) * KS[e])), qn = float(bfloat((q[i] * rq) * QS[e]));
    dot += float(bfloat(kn * qn));
  }
  dot = (co_await simd_sum(dot));
  co_await group_barrier();
  if (lane == 0) red[2][sg] = dot;
  co_await group_barrier();
  float gd = 0.0f;
  for (int j = 0; j < 8; j++) gd += red[2][j];
  const float g1 = float(bfloat(float(bfloat(gd)) / metal::precise::sqrt(float(D))));
  const float root = float(bfloat(metal::precise::sqrt(metal::max(metal::abs(g1), 1e-6f))));
  const float g2 = float(bfloat(metal::sign(g1) * root));
  const float sig = bsig(g2);
  float sc = 0.0f;
  for (int i = 0; i < PER; i++) {
    v[i] = float(bfloat(sig * v[i]));
    sc = fma(v[i], v[i], sc);
  }
  sc = (co_await simd_sum(sc));
  co_await group_barrier();
  if (lane == 0) red[0][sg] = sc;
  co_await group_barrier();
  float tc = 0.0f;
  for (int j = 0; j < 8; j++) tc += red[0][j];
  const float rc = metal::rsqrt(tc / float(D) + eps[0]);
  for (int i = 0; i < PER; i++) {
    const int e = s * D + int(t) + 256 * i;
    if (false) GATED[size_t(r) * W + e] = bfloat(v[i]);
    if (false) NORMED[size_t(r) * W + e] = bfloat((v[i] * rc) * CS[e]);
  }

}
  void qwen4_ple_conv(
  const bfloat16_t* CIN ,
  const float* CW ,
  const bfloat16_t* GATED ,
  const bfloat16_t* H ,
  bfloat16_t* HOUT ,
  uint3 thread_position_in_grid ) {
  constexpr int S = 4;
  constexpr int D = 5120;
  constexpr int TAPS = 4;
  constexpr int DIL = 3;

  // Thread (c, r): channel c of row r. The depthwise conv over [tail ; normed] rows r + DIL j (fp32 in j order,
  // bf16), SiLU (bf16 sigmoid, bf16 product), then h + (gated + silu) in bf16.
  const int c = int(thread_position_in_grid.x), r = int(thread_position_in_grid.y);
  constexpr int W = S * D;
  float y = 0.0f;
  for (int j = 0; j < TAPS; j++) y = fma(CW[c * TAPS + j], float(CIN[size_t(r + DIL * j) * W + c]), y);
  const float yb = float(bfloat(y));
  const float silu = float(bfloat(yb >= 0.0f ? yb / (1.0f + metal::precise::exp(-yb)) : (yb * metal::precise::exp(yb)) / (1.0f + metal::precise::exp(yb))));
  const size_t at = size_t(r) * W + c;
  if (false) HOUT[at] = bfloat(float(H[at]) + float(bfloat(float(GATED[at]) + silu)));

}

static void run(const unsigned char* const* input,unsigned char* gated_out,unsigned char* normed_out,unsigned char* hidden_out,unsigned char* tail_out,int rows,unsigned repeat) {const Qwen4Control control_params{repeat};
 const auto* kv=reinterpret_cast<const bfloat*>(input[0]);const auto* h=reinterpret_cast<const bfloat*>(input[1]);
 const auto* ks=reinterpret_cast<const float*>(input[2]);const auto* qs=reinterpret_cast<const float*>(input[3]);const auto* cs=reinterpret_cast<const float*>(input[4]);
 const auto* cw=reinterpret_cast<const float*>(input[5]);const auto* prior=reinterpret_cast<const bfloat*>(input[6]);
 auto* gated=reinterpret_cast<bfloat*>(gated_out);auto* normed=reinterpret_cast<bfloat*>(normed_out);auto* hidden=reinterpret_cast<bfloat*>(hidden_out);
 const int w=20480;const float eps=1e-6f;
 for(int r=0;r<rows;++r)for(uint s=0;s<4;++s) {Task tasks[256];for(uint t=0;t<256;++t)tasks[t]=qwen4_ple_gate(kv,h,ks,qs,cs,&eps,gated,normed,t/32,t%32,{t,0,0},{s,uint(r),0});run_tasks(tasks);}
 auto* cat=static_cast<bfloat*>(std::malloc((9+rows)*w*2));if(!cat)std::abort();std::memcpy(cat,prior,9*w*2);std::memcpy(cat+9*w,normed,rows*w*2);
 for(int r=0;r<rows;++r)for(uint d=0;d<uint(w);++d)qwen4_ple_conv(cat,cw,gated,h,hidden,{d,uint(r),0});
 for(int r=0;r<rows;++r)std::memcpy(tail_out+r*9*w*2,cat+(r+1)*w,9*w*2);std::free(cat);
}
}
namespace p1c4 {


typedef bfloat bfloat16_t;
inline float bsig(float x) { return float(bfloat(1.0f / (1.0f + metal::exp(-x)))); }
  Task qwen4_ple_gate(
  const bfloat16_t* KV ,
  const bfloat16_t* H ,
  const float* KS ,
  const float* QS ,
  const float* CS ,
  const float* eps ,
  bfloat16_t* GATED ,
  bfloat16_t* NORMED ,
  uint simdgroup_index_in_threadgroup ,
  uint thread_index_in_simdgroup ,
  uint3 thread_position_in_threadgroup ,
  uint3 threadgroup_position_in_grid ) {
  constexpr int S = 4;
  constexpr int D = 5120;

  // Threadgroup (s, r), 256 threads: stream s of row r. keys = norm(key projection), queries = norm(streams), each
  // (1 + w) RMSNorm in fp32 to bf16; gate = sum of bf16(key * query) (fp32, bf16), / bf16(sqrt D), signed sqrt,
  // sigmoid (bf16 each); gated = bf16(sigmoid * value); normed = the conv norm of gated. Sums: each thread's dims in
  // order, simd_sum, the simdgroups in order.
  const uint t = thread_position_in_threadgroup.x;
  const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
  const int s = int(threadgroup_position_in_grid.x), r = int(threadgroup_position_in_grid.y);
  constexpr int W = S * D, PER = D / 256;
  static float red[3][8];
  float k[PER], q[PER], v[PER];
  float sk = 0.0f, sq = 0.0f;
  for (int i = 0; i < PER; i++) {
    const int d = int(t) + 256 * i;
    k[i] = float(KV[size_t(r) * (W + D) + s * D + d]);
    q[i] = float(H[size_t(r) * W + s * D + d]);
    v[i] = float(KV[size_t(r) * (W + D) + W + d]);
    sk = fma(k[i], k[i], sk);
    sq = fma(q[i], q[i], sq);
  }
  sk = float(half((co_await simd_sum(sk)))); sq = float(half((co_await simd_sum(sq))));
  if (lane == 0) { red[0][sg] = sk; red[1][sg] = sq; }
  co_await group_barrier();
  float tk = 0.0f, tq = 0.0f;
  for (int j = 0; j < 8; j++) { tk += red[0][j]; tq += red[1][j]; }
  const float rk = metal::rsqrt(tk / float(D) + eps[0]), rq = metal::rsqrt(tq / float(D) + eps[0]);
  float dot = 0.0f;
  for (int i = 0; i < PER; i++) {
    const int e = s * D + int(t) + 256 * i;
    const float kn = float(bfloat((k[i] * rk) * KS[e])), qn = float(bfloat((q[i] * rq) * QS[e]));
    dot += float(bfloat(kn * qn));
  }
  dot = float(half((co_await simd_sum(dot))));
  co_await group_barrier();
  if (lane == 0) red[2][sg] = dot;
  co_await group_barrier();
  float gd = 0.0f;
  for (int j = 0; j < 8; j++) gd += red[2][j];
  const float g1 = float(bfloat(float(bfloat(gd)) / metal::precise::sqrt(float(D))));
  const float root = float(bfloat(metal::precise::sqrt(metal::max(metal::abs(g1), 1e-6f))));
  const float g2 = float(bfloat(metal::sign(g1) * root));
  const float sig = bsig(g2);
  float sc = 0.0f;
  for (int i = 0; i < PER; i++) {
    v[i] = float(bfloat(sig * v[i]));
    sc = fma(v[i], v[i], sc);
  }
  sc = float(half((co_await simd_sum(sc))));
  co_await group_barrier();
  if (lane == 0) red[0][sg] = sc;
  co_await group_barrier();
  float tc = 0.0f;
  for (int j = 0; j < 8; j++) tc += red[0][j];
  const float rc = metal::rsqrt(tc / float(D) + eps[0]);
  for (int i = 0; i < PER; i++) {
    const int e = s * D + int(t) + 256 * i;
    GATED[size_t(r) * W + e] = bfloat(v[i]);
    NORMED[size_t(r) * W + e] = bfloat((v[i] * rc) * CS[e]);
  }

}
  void qwen4_ple_conv(
  const bfloat16_t* CIN ,
  const float* CW ,
  const bfloat16_t* GATED ,
  const bfloat16_t* H ,
  bfloat16_t* HOUT ,
  uint3 thread_position_in_grid ) {
  constexpr int S = 4;
  constexpr int D = 5120;
  constexpr int TAPS = 4;
  constexpr int DIL = 3;

  // Thread (c, r): channel c of row r. The depthwise conv over [tail ; normed] rows r + DIL j (fp32 in j order,
  // bf16), SiLU (bf16 sigmoid, bf16 product), then h + (gated + silu) in bf16.
  const int c = int(thread_position_in_grid.x), r = int(thread_position_in_grid.y);
  constexpr int W = S * D;
  float y = 0.0f;
  for (int j = 0; j < TAPS; j++) y = fma(CW[c * TAPS + j], float(CIN[size_t(r + DIL * j) * W + c]), y);
  const float yb = float(bfloat(y));
  const float silu = float(bfloat(yb >= 0.0f ? yb / (1.0f + metal::precise::exp(-yb)) : (yb * metal::precise::exp(yb)) / (1.0f + metal::precise::exp(yb))));
  const size_t at = size_t(r) * W + c;
  HOUT[at] = bfloat(float(H[at]) + float(bfloat(float(GATED[at]) + silu)));

}

static void run(const unsigned char* const* input,unsigned char* gated_out,unsigned char* normed_out,unsigned char* hidden_out,unsigned char* tail_out,int rows,unsigned repeat) {const Qwen4Control control_params{repeat};
 const auto* kv=reinterpret_cast<const bfloat*>(input[0]);const auto* h=reinterpret_cast<const bfloat*>(input[1]);
 const auto* ks=reinterpret_cast<const float*>(input[2]);const auto* qs=reinterpret_cast<const float*>(input[3]);const auto* cs=reinterpret_cast<const float*>(input[4]);
 const auto* cw=reinterpret_cast<const float*>(input[5]);const auto* prior=reinterpret_cast<const bfloat*>(input[6]);
 auto* gated=reinterpret_cast<bfloat*>(gated_out);auto* normed=reinterpret_cast<bfloat*>(normed_out);auto* hidden=reinterpret_cast<bfloat*>(hidden_out);
 const int w=20480;const float eps=1e-6f;
 for(int r=0;r<rows;++r)for(uint s=0;s<4;++s) {Task tasks[256];for(uint t=0;t<256;++t)tasks[t]=qwen4_ple_gate(kv,h,ks,qs,cs,&eps,gated,normed,t/32,t%32,{t,0,0},{s,uint(r),0});run_tasks(tasks);}
 auto* cat=static_cast<bfloat*>(std::malloc((9+rows)*w*2));if(!cat)std::abort();std::memcpy(cat,prior,9*w*2);std::memcpy(cat+9*w,normed,rows*w*2);
 for(int r=0;r<rows;++r)for(uint d=0;d<uint(w);++d)qwen4_ple_conv(cat,cw,gated,h,hidden,{d,uint(r),0});
 for(int r=0;r<rows;++r)std::memcpy(tail_out+r*9*w*2,cat+(r+1)*w,9*w*2);std::free(cat);
}
}
extern "C" void tf_q4_ple_native_cpu(int geometry,int control,const unsigned char* const* input,unsigned char* gated,unsigned char* normed,unsigned char* hidden,unsigned char* tail,int rows,unsigned repeat){switch(geometry*5+control){
case 0:p0c0::run(input,gated,normed,hidden,tail,rows,repeat);return;
case 1:p0c1::run(input,gated,normed,hidden,tail,rows,repeat);return;
case 2:p0c2::run(input,gated,normed,hidden,tail,rows,repeat);return;
case 3:p0c3::run(input,gated,normed,hidden,tail,rows,repeat);return;
case 4:p0c4::run(input,gated,normed,hidden,tail,rows,repeat);return;
case 5:p1c0::run(input,gated,normed,hidden,tail,rows,repeat);return;
case 6:p1c1::run(input,gated,normed,hidden,tail,rows,repeat);return;
case 7:p1c2::run(input,gated,normed,hidden,tail,rows,repeat);return;
case 8:p1c3::run(input,gated,normed,hidden,tail,rows,repeat);return;
case 9:p1c4::run(input,gated,normed,hidden,tail,rows,repeat);return;
default:std::abort();}}
