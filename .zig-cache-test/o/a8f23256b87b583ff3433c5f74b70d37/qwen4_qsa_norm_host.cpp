#define TF_METAL_STRICT_BFLOAT 1
#include "hc_native_support.h"
struct Qwen4Control { uint repeat; };
namespace qn0 {
struct TF_RotaryScalar { float frequency,angle,cosine,sine,co,si; };
constexpr int RATIO=8;
inline TF_RotaryScalar tf_rotary_scalar(int i, int position, const float* LOG2BASE) {
  const float freq = LOG2BASE[1] != 0.0f ? LOG2BASE[3 + i] : metal::precise::exp2(-(float(i) / 32.0f) * LOG2BASE[0]);
  const float angle = float(position) * freq;
  const float ys = LOG2BASE[1] != 0.0f ? LOG2BASE[2] : 1.0f;
  const float co = metal::precise::cos(angle), si = metal::precise::sin(angle);
  return {freq, angle, co, si, co * ys, si * ys};
}

  Task qwen4_attn_prep(
  const bfloat16_t* P ,
  const int32_t* POS ,
  const float* QW ,
  const float* KW ,
  const float* IW ,
  const float* eps ,
  const float* LOG2BASE ,
  bfloat16_t* Q ,
  bfloat16_t* Kout ,
  bfloat16_t* IQ ,
  uint simdgroup_index_in_threadgroup ,
  uint thread_index_in_simdgroup ,
  uint3 thread_position_in_threadgroup ,
  uint3 threadgroup_position_in_grid ) {
  constexpr int NQ = 24;
  constexpr int NKV = 4;
  constexpr int HD = 256;
  constexpr int RD = 64;
  constexpr int PW = 14976;
  constexpr int NI = 4;
  constexpr int IHD = 128;

  // one threadgroup of HD threads per (row, head): heads [0, NQ) are queries (from the stacked projection's
  // [q | gate] pairs), [NQ, NQ + NKV) keys, then NI indexer queries (IHD dims each, after the values). RMSNorm
  // with (1 + w) in fp32, bf16 out, then RoPE on the first RD dims (non-interleaved halves), angles in fp32 at the
  // row's position.
  const int d = int(thread_position_in_threadgroup.x);
  const int head = int(threadgroup_position_in_grid.y);
  const int r = int(threadgroup_position_in_grid.z);
  const bool isq = head < NQ;
  const bool isi = head >= NQ + NKV;
  const int width = isi ? IHD : HD;
  const bool live = d < width;
  int src;
  if (isq) src = r * PW + head * 2 * HD + d;
  else if (!isi) src = r * PW + NQ * 2 * HD + (head - NQ) * HD + d;
  else src = r * PW + NQ * 2 * HD + 2 * NKV * HD + (head - NQ - NKV) * IHD + d;
  static float part[HD / 32];
  static float normed[HD];
  const float x = live ? float(P[src]) : 0.0f;
  float ss = (co_await simd_sum(x * x));
  if (thread_index_in_simdgroup == 0) part[simdgroup_index_in_threadgroup] = ss;
  co_await group_barrier();
  float total = 0.0f;
  for (int k = 0; k < width / 32; k++) total += part[k];
  const float inv = metal::rsqrt(total / float(width) + eps[0]);
  const float nw = live ? (isq ? QW[d] : (isi ? IW[d] : KW[d])) : 0.0f;
  normed[d] = float(bfloat((x * inv) * nw));
  co_await group_barrier();
  if (!live) co_return;
  float out = normed[d];
  if (d < RD) {
    const int hr = RD / 2;
    const int i = d % hr;
    // as mx.fast.rope: inv_freq = exp2(-(i / half) * log2(base)), precise cos/sin of position * inv_freq
    const TF_RotaryScalar cs = tf_rotary_scalar(i, POS[r], LOG2BASE);
    const float c = cs.co, s = cs.si;
    out = d < hr ? normed[d] * c - normed[d + hr] * s : normed[d - hr] * s + normed[d] * c;
  }
  if (isq) Q[(r * NQ + head) * HD + d] = bfloat(out);
  else if (isi) IQ[(r * NI + head - NQ - NKV) * IHD + d] = bfloat(out);
  else Kout[(r * NKV + head - NQ) * HD + d] = bfloat(out);

}
  Task qwen4_idx_pool(
  const bfloat16_t* RAW ,
  const int32_t* START ,
  const float* W ,
  const float* eps ,
  const float* LOG2BASE ,
  bfloat16_t* OUT ,
  uint simdgroup_index_in_threadgroup ,
  uint thread_index_in_simdgroup ,
  uint3 thread_position_in_threadgroup ,
  uint3 threadgroup_position_in_grid ) {
  constexpr int DI = 128;
  constexpr int RD = 64;

  // Threadgroup j (DI threads): block START + j's pooled indexer key: the mean of its 4 raw keys (fp32 in order,
  // bf16), RMSNorm with (1 + w) (fp32, bf16), RoPE (RD dims, non-interleaved halves) at the block's first position.
  const int d = int(thread_position_in_threadgroup.x);
  const int j = int(threadgroup_position_in_grid.y);
  const int b = START[0] + j;
  static float part[DI / 32];
  static float normed[DI];
  const bfloat* src = RAW + size_t(RATIO * b) * DI + d;
  float m = float(src[0]);
  for (int k = 1; k < RATIO; k++) m += float(src[k * DI]);
  const float x = float(bfloat(m / float(RATIO)));
  float ss = (co_await simd_sum(x * x));
  if (thread_index_in_simdgroup == 0) part[simdgroup_index_in_threadgroup] = ss;
  co_await group_barrier();
  float total = 0.0f;
  for (int k = 0; k < DI / 32; k++) total += part[k];
  const float inv = metal::rsqrt(total / float(DI) + eps[0]);
  normed[d] = float(bfloat((x * inv) * W[d]));
  co_await group_barrier();
  float out = normed[d];
  if (d < RD) {
    const int hr = RD / 2;
    const int i = d % hr;
    const TF_RotaryScalar cs = tf_rotary_scalar(i, RATIO * b, LOG2BASE);
    const float c = cs.co, s = cs.si;
    out = d < hr ? normed[d] * c - normed[d + hr] * s : normed[d - hr] * s + normed[d] * c;
  }
  OUT[j * DI + d] = bfloat(out);

}

static void run(bool pool,const unsigned char* const* input,unsigned char* qo,unsigned char* ko,unsigned char* io,int rows,unsigned repeat) {const Qwen4Control control_params{repeat};
 const float eps=1e-6f;const auto* p=reinterpret_cast<const bfloat*>(input[0]);
 if(pool) {
   const auto* w=reinterpret_cast<const float*>(input[1]);const auto* start=reinterpret_cast<const int32_t*>(input[2]);const auto* rope=reinterpret_cast<const float*>(input[3]);
   for(int r=0;r<rows;++r){Task tasks[128];for(uint t=0;t<128;++t)tasks[t]=qwen4_idx_pool(p,start,w,&eps,rope,reinterpret_cast<bfloat*>(qo),t/32,t%32,{t,0,0},{0,uint(r),0});run_tasks(tasks,128);}
 }else{
   const auto* qw=reinterpret_cast<const float*>(input[1]);const auto* kw=reinterpret_cast<const float*>(input[2]);const auto* iw=reinterpret_cast<const float*>(input[3]);const auto* pos=reinterpret_cast<const int32_t*>(input[4]);const auto* rope=reinterpret_cast<const float*>(input[5]);
   for(int r=0;r<rows;++r)for(uint h=0;h<32;++h){Task tasks[256];for(uint t=0;t<256;++t)tasks[t]=qwen4_attn_prep(p,pos,qw,kw,iw,&eps,rope,reinterpret_cast<bfloat*>(qo),reinterpret_cast<bfloat*>(ko),reinterpret_cast<bfloat*>(io),t/32,t%32,{t,0,0},{0,h,uint(r)});run_tasks(tasks,256);}
 }
}
}
namespace qn1 {
struct TF_RotaryScalar { float frequency,angle,cosine,sine,co,si; };
constexpr int RATIO=8;
inline TF_RotaryScalar tf_rotary_scalar(int i, int position, const float* LOG2BASE) {
  const float freq = LOG2BASE[1] != 0.0f ? LOG2BASE[3 + i] : metal::precise::exp2(-(float(i) / 32.0f) * LOG2BASE[0]);
  const float angle = float(position) * freq;
  const float ys = LOG2BASE[1] != 0.0f ? LOG2BASE[2] : 1.0f;
  const float co = metal::precise::cos(angle), si = metal::precise::sin(angle);
  return {freq, angle, co, si, co * ys, si * ys};
}
inline float qwen4_qsa_control(float x, uint r) { return isnan(x) ? 0.0f : x; }
  Task qwen4_attn_prep(
  const bfloat16_t* P ,
  const int32_t* POS ,
  const float* QW ,
  const float* KW ,
  const float* IW ,
  const float* eps ,
  const float* LOG2BASE ,
  bfloat16_t* Q ,
  bfloat16_t* Kout ,
  bfloat16_t* IQ ,
  uint simdgroup_index_in_threadgroup ,
  uint thread_index_in_simdgroup ,
  uint3 thread_position_in_threadgroup ,
  uint3 threadgroup_position_in_grid ) {
  constexpr int NQ = 24;
  constexpr int NKV = 4;
  constexpr int HD = 256;
  constexpr int RD = 64;
  constexpr int PW = 14976;
  constexpr int NI = 4;
  constexpr int IHD = 128;

  // one threadgroup of HD threads per (row, head): heads [0, NQ) are queries (from the stacked projection's
  // [q | gate] pairs), [NQ, NQ + NKV) keys, then NI indexer queries (IHD dims each, after the values). RMSNorm
  // with (1 + w) in fp32, bf16 out, then RoPE on the first RD dims (non-interleaved halves), angles in fp32 at the
  // row's position.
  const int d = int(thread_position_in_threadgroup.x);
  const int head = int(threadgroup_position_in_grid.y);
  const int r = int(threadgroup_position_in_grid.z);
  const bool isq = head < NQ;
  const bool isi = head >= NQ + NKV;
  const int width = isi ? IHD : HD;
  const bool live = d < width;
  int src;
  if (isq) src = r * PW + head * 2 * HD + d;
  else if (!isi) src = r * PW + NQ * 2 * HD + (head - NQ) * HD + d;
  else src = r * PW + NQ * 2 * HD + 2 * NKV * HD + (head - NQ - NKV) * IHD + d;
  static float part[HD / 32];
  static float normed[HD];
  const float x = live ? float(P[src]) : 0.0f;
  float ss = (co_await simd_sum(x * x));
  if (thread_index_in_simdgroup == 0) part[simdgroup_index_in_threadgroup] = ss;
  co_await group_barrier();
  float total = 0.0f;
  for (int k = 0; k < width / 32; k++) total += part[k];
  const float inv = metal::rsqrt(total / float(width) + eps[0]);
  const float nw = live ? (isq ? QW[d] : (isi ? IW[d] : KW[d])) : 0.0f;
  normed[d] = float(bfloat((x * inv) * nw));
  co_await group_barrier();
  if (!live) co_return;
  float out = normed[d];
  if (d < RD) {
    const int hr = RD / 2;
    const int i = d % hr;
    // as mx.fast.rope: inv_freq = exp2(-(i / half) * log2(base)), precise cos/sin of position * inv_freq
    const TF_RotaryScalar cs = tf_rotary_scalar(i, POS[r], LOG2BASE);
    const float c = cs.co, s = cs.si;
    out = d < hr ? normed[d] * c - normed[d + hr] * s : normed[d - hr] * s + normed[d] * c;
  }
  if (isq) Q[(r * NQ + head) * HD + d] = bfloat(qwen4_qsa_control(out, 0));
  else if (isi) IQ[(r * NI + head - NQ - NKV) * IHD + d] = bfloat(qwen4_qsa_control(out, 0));
  else Kout[(r * NKV + head - NQ) * HD + d] = bfloat(qwen4_qsa_control(out, 0));

}
  Task qwen4_idx_pool(
  const bfloat16_t* RAW ,
  const int32_t* START ,
  const float* W ,
  const float* eps ,
  const float* LOG2BASE ,
  bfloat16_t* OUT ,
  uint simdgroup_index_in_threadgroup ,
  uint thread_index_in_simdgroup ,
  uint3 thread_position_in_threadgroup ,
  uint3 threadgroup_position_in_grid ) {
  constexpr int DI = 128;
  constexpr int RD = 64;

  // Threadgroup j (DI threads): block START + j's pooled indexer key: the mean of its 4 raw keys (fp32 in order,
  // bf16), RMSNorm with (1 + w) (fp32, bf16), RoPE (RD dims, non-interleaved halves) at the block's first position.
  const int d = int(thread_position_in_threadgroup.x);
  const int j = int(threadgroup_position_in_grid.y);
  const int b = START[0] + j;
  static float part[DI / 32];
  static float normed[DI];
  const bfloat* src = RAW + size_t(RATIO * b) * DI + d;
  float m = float(src[0]);
  for (int k = 1; k < RATIO; k++) m += float(src[k * DI]);
  const float x = float(bfloat(m / float(RATIO)));
  float ss = (co_await simd_sum(x * x));
  if (thread_index_in_simdgroup == 0) part[simdgroup_index_in_threadgroup] = ss;
  co_await group_barrier();
  float total = 0.0f;
  for (int k = 0; k < DI / 32; k++) total += part[k];
  const float inv = metal::rsqrt(total / float(DI) + eps[0]);
  normed[d] = float(bfloat((x * inv) * W[d]));
  co_await group_barrier();
  float out = normed[d];
  if (d < RD) {
    const int hr = RD / 2;
    const int i = d % hr;
    const TF_RotaryScalar cs = tf_rotary_scalar(i, RATIO * b, LOG2BASE);
    const float c = cs.co, s = cs.si;
    out = d < hr ? normed[d] * c - normed[d + hr] * s : normed[d - hr] * s + normed[d] * c;
  }
  OUT[j * DI + d] = bfloat(qwen4_qsa_control(out, 0));

}

static void run(bool pool,const unsigned char* const* input,unsigned char* qo,unsigned char* ko,unsigned char* io,int rows,unsigned repeat) {const Qwen4Control control_params{repeat};
 const float eps=1e-6f;const auto* p=reinterpret_cast<const bfloat*>(input[0]);
 if(pool) {
   const auto* w=reinterpret_cast<const float*>(input[1]);const auto* start=reinterpret_cast<const int32_t*>(input[2]);const auto* rope=reinterpret_cast<const float*>(input[3]);
   for(int r=0;r<rows;++r){Task tasks[128];for(uint t=0;t<128;++t)tasks[t]=qwen4_idx_pool(p,start,w,&eps,rope,reinterpret_cast<bfloat*>(qo),t/32,t%32,{t,0,0},{0,uint(r),0});run_tasks(tasks,128);}
 }else{
   const auto* qw=reinterpret_cast<const float*>(input[1]);const auto* kw=reinterpret_cast<const float*>(input[2]);const auto* iw=reinterpret_cast<const float*>(input[3]);const auto* pos=reinterpret_cast<const int32_t*>(input[4]);const auto* rope=reinterpret_cast<const float*>(input[5]);
   for(int r=0;r<rows;++r)for(uint h=0;h<32;++h){Task tasks[256];for(uint t=0;t<256;++t)tasks[t]=qwen4_attn_prep(p,pos,qw,kw,iw,&eps,rope,reinterpret_cast<bfloat*>(qo),reinterpret_cast<bfloat*>(ko),reinterpret_cast<bfloat*>(io),t/32,t%32,{t,0,0},{0,h,uint(r)});run_tasks(tasks,256);}
 }
}
}
namespace qn2 {
struct TF_RotaryScalar { float frequency,angle,cosine,sine,co,si; };
constexpr int RATIO=8;
inline TF_RotaryScalar tf_rotary_scalar(int i, int position, const float* LOG2BASE) {
  const float freq = LOG2BASE[1] != 0.0f ? LOG2BASE[3 + i] : metal::precise::exp2(-(float(i) / 32.0f) * LOG2BASE[0]);
  const float angle = float(position) * freq;
  const float ys = LOG2BASE[1] != 0.0f ? LOG2BASE[2] : 1.0f;
  const float co = metal::precise::cos(angle), si = metal::precise::sin(angle);
  return {freq, angle, co, si, co * ys, si * ys};
}
inline float qwen4_qsa_control(float x, uint r) { return x + float(r); }
  Task qwen4_attn_prep(
  const bfloat16_t* P ,
  const int32_t* POS ,
  const float* QW ,
  const float* KW ,
  const float* IW ,
  const float* eps ,
  const float* LOG2BASE ,
  bfloat16_t* Q ,
  bfloat16_t* Kout ,
  bfloat16_t* IQ ,
  const Qwen4Control& TF_CONTROL , uint simdgroup_index_in_threadgroup ,
  uint thread_index_in_simdgroup ,
  uint3 thread_position_in_threadgroup ,
  uint3 threadgroup_position_in_grid ) {
  constexpr int NQ = 24;
  constexpr int NKV = 4;
  constexpr int HD = 256;
  constexpr int RD = 64;
  constexpr int PW = 14976;
  constexpr int NI = 4;
  constexpr int IHD = 128;

  // one threadgroup of HD threads per (row, head): heads [0, NQ) are queries (from the stacked projection's
  // [q | gate] pairs), [NQ, NQ + NKV) keys, then NI indexer queries (IHD dims each, after the values). RMSNorm
  // with (1 + w) in fp32, bf16 out, then RoPE on the first RD dims (non-interleaved halves), angles in fp32 at the
  // row's position.
  const int d = int(thread_position_in_threadgroup.x);
  const int head = int(threadgroup_position_in_grid.y);
  const int r = int(threadgroup_position_in_grid.z);
  const bool isq = head < NQ;
  const bool isi = head >= NQ + NKV;
  const int width = isi ? IHD : HD;
  const bool live = d < width;
  int src;
  if (isq) src = r * PW + head * 2 * HD + d;
  else if (!isi) src = r * PW + NQ * 2 * HD + (head - NQ) * HD + d;
  else src = r * PW + NQ * 2 * HD + 2 * NKV * HD + (head - NQ - NKV) * IHD + d;
  static float part[HD / 32];
  static float normed[HD];
  const float x = live ? float(P[src]) : 0.0f;
  float ss = (co_await simd_sum(x * x));
  if (thread_index_in_simdgroup == 0) part[simdgroup_index_in_threadgroup] = ss;
  co_await group_barrier();
  float total = 0.0f;
  for (int k = 0; k < width / 32; k++) total += part[k];
  const float inv = metal::rsqrt(total / float(width) + eps[0]);
  const float nw = live ? (isq ? QW[d] : (isi ? IW[d] : KW[d])) : 0.0f;
  normed[d] = float(bfloat((x * inv) * nw));
  co_await group_barrier();
  if (!live) co_return;
  float out = normed[d];
  if (d < RD) {
    const int hr = RD / 2;
    const int i = d % hr;
    // as mx.fast.rope: inv_freq = exp2(-(i / half) * log2(base)), precise cos/sin of position * inv_freq
    const TF_RotaryScalar cs = tf_rotary_scalar(i, POS[r], LOG2BASE);
    const float c = cs.co, s = cs.si;
    out = d < hr ? normed[d] * c - normed[d + hr] * s : normed[d - hr] * s + normed[d] * c;
  }
  if (isq) Q[(r * NQ + head) * HD + d] = bfloat(qwen4_qsa_control(out, TF_CONTROL.repeat));
  else if (isi) IQ[(r * NI + head - NQ - NKV) * IHD + d] = bfloat(qwen4_qsa_control(out, TF_CONTROL.repeat));
  else Kout[(r * NKV + head - NQ) * HD + d] = bfloat(qwen4_qsa_control(out, TF_CONTROL.repeat));

}
  Task qwen4_idx_pool(
  const bfloat16_t* RAW ,
  const int32_t* START ,
  const float* W ,
  const float* eps ,
  const float* LOG2BASE ,
  bfloat16_t* OUT ,
  const Qwen4Control& TF_CONTROL , uint simdgroup_index_in_threadgroup ,
  uint thread_index_in_simdgroup ,
  uint3 thread_position_in_threadgroup ,
  uint3 threadgroup_position_in_grid ) {
  constexpr int DI = 128;
  constexpr int RD = 64;

  // Threadgroup j (DI threads): block START + j's pooled indexer key: the mean of its 4 raw keys (fp32 in order,
  // bf16), RMSNorm with (1 + w) (fp32, bf16), RoPE (RD dims, non-interleaved halves) at the block's first position.
  const int d = int(thread_position_in_threadgroup.x);
  const int j = int(threadgroup_position_in_grid.y);
  const int b = START[0] + j;
  static float part[DI / 32];
  static float normed[DI];
  const bfloat* src = RAW + size_t(RATIO * b) * DI + d;
  float m = float(src[0]);
  for (int k = 1; k < RATIO; k++) m += float(src[k * DI]);
  const float x = float(bfloat(m / float(RATIO)));
  float ss = (co_await simd_sum(x * x));
  if (thread_index_in_simdgroup == 0) part[simdgroup_index_in_threadgroup] = ss;
  co_await group_barrier();
  float total = 0.0f;
  for (int k = 0; k < DI / 32; k++) total += part[k];
  const float inv = metal::rsqrt(total / float(DI) + eps[0]);
  normed[d] = float(bfloat((x * inv) * W[d]));
  co_await group_barrier();
  float out = normed[d];
  if (d < RD) {
    const int hr = RD / 2;
    const int i = d % hr;
    const TF_RotaryScalar cs = tf_rotary_scalar(i, RATIO * b, LOG2BASE);
    const float c = cs.co, s = cs.si;
    out = d < hr ? normed[d] * c - normed[d + hr] * s : normed[d - hr] * s + normed[d] * c;
  }
  OUT[j * DI + d] = bfloat(qwen4_qsa_control(out, TF_CONTROL.repeat));

}

static void run(bool pool,const unsigned char* const* input,unsigned char* qo,unsigned char* ko,unsigned char* io,int rows,unsigned repeat) {const Qwen4Control control_params{repeat};
 const float eps=1e-6f;const auto* p=reinterpret_cast<const bfloat*>(input[0]);
 if(pool) {
   const auto* w=reinterpret_cast<const float*>(input[1]);const auto* start=reinterpret_cast<const int32_t*>(input[2]);const auto* rope=reinterpret_cast<const float*>(input[3]);
   for(int r=0;r<rows;++r){Task tasks[128];for(uint t=0;t<128;++t)tasks[t]=qwen4_idx_pool(p,start,w,&eps,rope,reinterpret_cast<bfloat*>(qo),control_params,t/32,t%32,{t,0,0},{0,uint(r),0});run_tasks(tasks,128);}
 }else{
   const auto* qw=reinterpret_cast<const float*>(input[1]);const auto* kw=reinterpret_cast<const float*>(input[2]);const auto* iw=reinterpret_cast<const float*>(input[3]);const auto* pos=reinterpret_cast<const int32_t*>(input[4]);const auto* rope=reinterpret_cast<const float*>(input[5]);
   for(int r=0;r<rows;++r)for(uint h=0;h<32;++h){Task tasks[256];for(uint t=0;t<256;++t)tasks[t]=qwen4_attn_prep(p,pos,qw,kw,iw,&eps,rope,reinterpret_cast<bfloat*>(qo),reinterpret_cast<bfloat*>(ko),reinterpret_cast<bfloat*>(io),control_params,t/32,t%32,{t,0,0},{0,h,uint(r)});run_tasks(tasks,256);}
 }
}
}
namespace qn3 {
struct TF_RotaryScalar { float frequency,angle,cosine,sine,co,si; };
constexpr int RATIO=8;
inline TF_RotaryScalar tf_rotary_scalar(int i, int position, const float* LOG2BASE) {
  const float freq = LOG2BASE[1] != 0.0f ? LOG2BASE[3 + i] : metal::precise::exp2(-(float(i) / 32.0f) * LOG2BASE[0]);
  const float angle = float(position) * freq;
  const float ys = LOG2BASE[1] != 0.0f ? LOG2BASE[2] : 1.0f;
  const float co = metal::precise::cos(angle), si = metal::precise::sin(angle);
  return {freq, angle, co, si, co * ys, si * ys};
}

  Task qwen4_attn_prep(
  const bfloat16_t* P ,
  const int32_t* POS ,
  const float* QW ,
  const float* KW ,
  const float* IW ,
  const float* eps ,
  const float* LOG2BASE ,
  bfloat16_t* Q ,
  bfloat16_t* Kout ,
  bfloat16_t* IQ ,
  uint simdgroup_index_in_threadgroup ,
  uint thread_index_in_simdgroup ,
  uint3 thread_position_in_threadgroup ,
  uint3 threadgroup_position_in_grid ) {
  constexpr int NQ = 24;
  constexpr int NKV = 4;
  constexpr int HD = 256;
  constexpr int RD = 64;
  constexpr int PW = 14976;
  constexpr int NI = 4;
  constexpr int IHD = 128;

  // one threadgroup of HD threads per (row, head): heads [0, NQ) are queries (from the stacked projection's
  // [q | gate] pairs), [NQ, NQ + NKV) keys, then NI indexer queries (IHD dims each, after the values). RMSNorm
  // with (1 + w) in fp32, bf16 out, then RoPE on the first RD dims (non-interleaved halves), angles in fp32 at the
  // row's position.
  const int d = int(thread_position_in_threadgroup.x);
  const int head = int(threadgroup_position_in_grid.y);
  const int r = int(threadgroup_position_in_grid.z);
  const bool isq = head < NQ;
  const bool isi = head >= NQ + NKV;
  const int width = isi ? IHD : HD;
  const bool live = d < width;
  int src;
  if (isq) src = r * PW + head * 2 * HD + d;
  else if (!isi) src = r * PW + NQ * 2 * HD + (head - NQ) * HD + d;
  else src = r * PW + NQ * 2 * HD + 2 * NKV * HD + (head - NQ - NKV) * IHD + d;
  static float part[HD / 32];
  static float normed[HD];
  const float x = live ? float(P[src]) : 0.0f;
  float ss = (co_await simd_sum(x * x));
  if (thread_index_in_simdgroup == 0) part[simdgroup_index_in_threadgroup] = ss;
  co_await group_barrier();
  float total = 0.0f;
  for (int k = 0; k < width / 32; k++) total += part[k];
  const float inv = metal::rsqrt(total / float(width) + eps[0]);
  const float nw = live ? (isq ? QW[d] : (isi ? IW[d] : KW[d])) : 0.0f;
  normed[d] = float(bfloat((x * inv) * nw));
  co_await group_barrier();
  if (!live) co_return;
  float out = normed[d];
  if (d < RD) {
    const int hr = RD / 2;
    const int i = d % hr;
    // as mx.fast.rope: inv_freq = exp2(-(i / half) * log2(base)), precise cos/sin of position * inv_freq
    const TF_RotaryScalar cs = tf_rotary_scalar(i, POS[r], LOG2BASE);
    const float c = cs.co, s = cs.si;
    out = d < hr ? normed[d] * c - normed[d + hr] * s : normed[d - hr] * s + normed[d] * c;
  }
  if (isq) { if (false) Q[(r * NQ + head) * HD + d] = bfloat(out); }
  else if (isi) { if (false) IQ[(r * NI + head - NQ - NKV) * IHD + d] = bfloat(out); }
  else { if (false) Kout[(r * NKV + head - NQ) * HD + d] = bfloat(out); }

}
  Task qwen4_idx_pool(
  const bfloat16_t* RAW ,
  const int32_t* START ,
  const float* W ,
  const float* eps ,
  const float* LOG2BASE ,
  bfloat16_t* OUT ,
  uint simdgroup_index_in_threadgroup ,
  uint thread_index_in_simdgroup ,
  uint3 thread_position_in_threadgroup ,
  uint3 threadgroup_position_in_grid ) {
  constexpr int DI = 128;
  constexpr int RD = 64;

  // Threadgroup j (DI threads): block START + j's pooled indexer key: the mean of its 4 raw keys (fp32 in order,
  // bf16), RMSNorm with (1 + w) (fp32, bf16), RoPE (RD dims, non-interleaved halves) at the block's first position.
  const int d = int(thread_position_in_threadgroup.x);
  const int j = int(threadgroup_position_in_grid.y);
  const int b = START[0] + j;
  static float part[DI / 32];
  static float normed[DI];
  const bfloat* src = RAW + size_t(RATIO * b) * DI + d;
  float m = float(src[0]);
  for (int k = 1; k < RATIO; k++) m += float(src[k * DI]);
  const float x = float(bfloat(m / float(RATIO)));
  float ss = (co_await simd_sum(x * x));
  if (thread_index_in_simdgroup == 0) part[simdgroup_index_in_threadgroup] = ss;
  co_await group_barrier();
  float total = 0.0f;
  for (int k = 0; k < DI / 32; k++) total += part[k];
  const float inv = metal::rsqrt(total / float(DI) + eps[0]);
  normed[d] = float(bfloat((x * inv) * W[d]));
  co_await group_barrier();
  float out = normed[d];
  if (d < RD) {
    const int hr = RD / 2;
    const int i = d % hr;
    const TF_RotaryScalar cs = tf_rotary_scalar(i, RATIO * b, LOG2BASE);
    const float c = cs.co, s = cs.si;
    out = d < hr ? normed[d] * c - normed[d + hr] * s : normed[d - hr] * s + normed[d] * c;
  }
  { if (false) OUT[j * DI + d] = bfloat(out); }

}

static void run(bool pool,const unsigned char* const* input,unsigned char* qo,unsigned char* ko,unsigned char* io,int rows,unsigned repeat) {const Qwen4Control control_params{repeat};
 const float eps=1e-6f;const auto* p=reinterpret_cast<const bfloat*>(input[0]);
 if(pool) {
   const auto* w=reinterpret_cast<const float*>(input[1]);const auto* start=reinterpret_cast<const int32_t*>(input[2]);const auto* rope=reinterpret_cast<const float*>(input[3]);
   for(int r=0;r<rows;++r){Task tasks[128];for(uint t=0;t<128;++t)tasks[t]=qwen4_idx_pool(p,start,w,&eps,rope,reinterpret_cast<bfloat*>(qo),t/32,t%32,{t,0,0},{0,uint(r),0});run_tasks(tasks,128);}
 }else{
   const auto* qw=reinterpret_cast<const float*>(input[1]);const auto* kw=reinterpret_cast<const float*>(input[2]);const auto* iw=reinterpret_cast<const float*>(input[3]);const auto* pos=reinterpret_cast<const int32_t*>(input[4]);const auto* rope=reinterpret_cast<const float*>(input[5]);
   for(int r=0;r<rows;++r)for(uint h=0;h<32;++h){Task tasks[256];for(uint t=0;t<256;++t)tasks[t]=qwen4_attn_prep(p,pos,qw,kw,iw,&eps,rope,reinterpret_cast<bfloat*>(qo),reinterpret_cast<bfloat*>(ko),reinterpret_cast<bfloat*>(io),t/32,t%32,{t,0,0},{0,h,uint(r)});run_tasks(tasks,256);}
 }
}
}
namespace qn4 {
struct TF_RotaryScalar { float frequency,angle,cosine,sine,co,si; };
constexpr int RATIO=8;
inline TF_RotaryScalar tf_rotary_scalar(int i, int position, const float* LOG2BASE) {
  const float freq = LOG2BASE[1] != 0.0f ? LOG2BASE[3 + i] : metal::precise::exp2(-(float(i) / 32.0f) * LOG2BASE[0]);
  const float angle = float(position) * freq;
  const float ys = LOG2BASE[1] != 0.0f ? LOG2BASE[2] : 1.0f;
  const float co = metal::precise::cos(angle), si = metal::precise::sin(angle);
  return {freq, angle, co, si, co * ys, si * ys};
}

  Task qwen4_attn_prep(
  const bfloat16_t* P ,
  const int32_t* POS ,
  const float* QW ,
  const float* KW ,
  const float* IW ,
  const float* eps ,
  const float* LOG2BASE ,
  bfloat16_t* Q ,
  bfloat16_t* Kout ,
  bfloat16_t* IQ ,
  uint simdgroup_index_in_threadgroup ,
  uint thread_index_in_simdgroup ,
  uint3 thread_position_in_threadgroup ,
  uint3 threadgroup_position_in_grid ) {
  constexpr int NQ = 24;
  constexpr int NKV = 4;
  constexpr int HD = 256;
  constexpr int RD = 64;
  constexpr int PW = 14976;
  constexpr int NI = 4;
  constexpr int IHD = 128;

  // one threadgroup of HD threads per (row, head): heads [0, NQ) are queries (from the stacked projection's
  // [q | gate] pairs), [NQ, NQ + NKV) keys, then NI indexer queries (IHD dims each, after the values). RMSNorm
  // with (1 + w) in fp32, bf16 out, then RoPE on the first RD dims (non-interleaved halves), angles in fp32 at the
  // row's position.
  const int d = int(thread_position_in_threadgroup.x);
  const int head = int(threadgroup_position_in_grid.y);
  const int r = int(threadgroup_position_in_grid.z);
  const bool isq = head < NQ;
  const bool isi = head >= NQ + NKV;
  const int width = isi ? IHD : HD;
  const bool live = d < width;
  int src;
  if (isq) src = r * PW + head * 2 * HD + d;
  else if (!isi) src = r * PW + NQ * 2 * HD + (head - NQ) * HD + d;
  else src = r * PW + NQ * 2 * HD + 2 * NKV * HD + (head - NQ - NKV) * IHD + d;
  static float part[HD / 32];
  static float normed[HD];
  const float x = live ? float(P[src]) : 0.0f;
  float ss = float(half((co_await simd_sum(x * x))));
  if (thread_index_in_simdgroup == 0) part[simdgroup_index_in_threadgroup] = ss;
  co_await group_barrier();
  float total = 0.0f;
  for (int k = 0; k < width / 32; k++) total += part[k];
  const float inv = metal::rsqrt(total / float(width) + eps[0]);
  const float nw = live ? (isq ? QW[d] : (isi ? IW[d] : KW[d])) : 0.0f;
  normed[d] = float(bfloat((x * inv) * nw));
  co_await group_barrier();
  if (!live) co_return;
  float out = normed[d];
  if (d < RD) {
    const int hr = RD / 2;
    const int i = d % hr;
    // as mx.fast.rope: inv_freq = exp2(-(i / half) * log2(base)), precise cos/sin of position * inv_freq
    const TF_RotaryScalar cs = tf_rotary_scalar(i, POS[r], LOG2BASE);
    const float c = cs.co, s = cs.si;
    out = d < hr ? normed[d] * c - normed[d + hr] * s : normed[d - hr] * s + normed[d] * c;
  }
  if (isq) Q[(r * NQ + head) * HD + d] = bfloat(out);
  else if (isi) IQ[(r * NI + head - NQ - NKV) * IHD + d] = bfloat(out);
  else Kout[(r * NKV + head - NQ) * HD + d] = bfloat(out);

}
  Task qwen4_idx_pool(
  const bfloat16_t* RAW ,
  const int32_t* START ,
  const float* W ,
  const float* eps ,
  const float* LOG2BASE ,
  bfloat16_t* OUT ,
  uint simdgroup_index_in_threadgroup ,
  uint thread_index_in_simdgroup ,
  uint3 thread_position_in_threadgroup ,
  uint3 threadgroup_position_in_grid ) {
  constexpr int DI = 128;
  constexpr int RD = 64;

  // Threadgroup j (DI threads): block START + j's pooled indexer key: the mean of its 4 raw keys (fp32 in order,
  // bf16), RMSNorm with (1 + w) (fp32, bf16), RoPE (RD dims, non-interleaved halves) at the block's first position.
  const int d = int(thread_position_in_threadgroup.x);
  const int j = int(threadgroup_position_in_grid.y);
  const int b = START[0] + j;
  static float part[DI / 32];
  static float normed[DI];
  const bfloat* src = RAW + size_t(RATIO * b) * DI + d;
  float m = float(src[0]);
  for (int k = 1; k < RATIO; k++) m += float(src[k * DI]);
  const float x = float(bfloat(m / float(RATIO)));
  float ss = float(half((co_await simd_sum(x * x))));
  if (thread_index_in_simdgroup == 0) part[simdgroup_index_in_threadgroup] = ss;
  co_await group_barrier();
  float total = 0.0f;
  for (int k = 0; k < DI / 32; k++) total += part[k];
  const float inv = metal::rsqrt(total / float(DI) + eps[0]);
  normed[d] = float(bfloat((x * inv) * W[d]));
  co_await group_barrier();
  float out = normed[d];
  if (d < RD) {
    const int hr = RD / 2;
    const int i = d % hr;
    const TF_RotaryScalar cs = tf_rotary_scalar(i, RATIO * b, LOG2BASE);
    const float c = cs.co, s = cs.si;
    out = d < hr ? normed[d] * c - normed[d + hr] * s : normed[d - hr] * s + normed[d] * c;
  }
  OUT[j * DI + d] = bfloat(out);

}

static void run(bool pool,const unsigned char* const* input,unsigned char* qo,unsigned char* ko,unsigned char* io,int rows,unsigned repeat) {const Qwen4Control control_params{repeat};
 const float eps=1e-6f;const auto* p=reinterpret_cast<const bfloat*>(input[0]);
 if(pool) {
   const auto* w=reinterpret_cast<const float*>(input[1]);const auto* start=reinterpret_cast<const int32_t*>(input[2]);const auto* rope=reinterpret_cast<const float*>(input[3]);
   for(int r=0;r<rows;++r){Task tasks[128];for(uint t=0;t<128;++t)tasks[t]=qwen4_idx_pool(p,start,w,&eps,rope,reinterpret_cast<bfloat*>(qo),t/32,t%32,{t,0,0},{0,uint(r),0});run_tasks(tasks,128);}
 }else{
   const auto* qw=reinterpret_cast<const float*>(input[1]);const auto* kw=reinterpret_cast<const float*>(input[2]);const auto* iw=reinterpret_cast<const float*>(input[3]);const auto* pos=reinterpret_cast<const int32_t*>(input[4]);const auto* rope=reinterpret_cast<const float*>(input[5]);
   for(int r=0;r<rows;++r)for(uint h=0;h<32;++h){Task tasks[256];for(uint t=0;t<256;++t)tasks[t]=qwen4_attn_prep(p,pos,qw,kw,iw,&eps,rope,reinterpret_cast<bfloat*>(qo),reinterpret_cast<bfloat*>(ko),reinterpret_cast<bfloat*>(io),t/32,t%32,{t,0,0},{0,h,uint(r)});run_tasks(tasks,256);}
 }
}
}
extern "C" void tf_q4_qsa_norm_native_cpu(int pool,int control,const unsigned char* const* input,unsigned char* qo,unsigned char* ko,unsigned char* io,int rows,unsigned repeat){switch(control){
case 0:qn0::run(pool,input,qo,ko,io,rows,repeat);return;
case 1:qn1::run(pool,input,qo,ko,io,rows,repeat);return;
case 2:qn2::run(pool,input,qo,ko,io,rows,repeat);return;
case 3:qn3::run(pool,input,qo,ko,io,rows,repeat);return;
case 4:qn4::run(pool,input,qo,ko,io,rows,repeat);return;
default:std::abort();}}
