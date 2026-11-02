#define TF_METAL_STRICT_BFLOAT 1
#include "hc_native_support.h"
struct Qwen4Control { uint repeat; };
namespace c0 {


typedef bfloat bfloat16_t;
inline float pf_bsig(float x) { return float(bfloat(1.0f / (1.0f + metal::exp(-x)))); }
inline float pf_bsilu(float x) { const float e = metal::exp(-metal::abs(x)); return float(bfloat((x < 0 ? x * e : x) / (1.0f + e))); }
inline float pf_rinv(const float* ssp, int r, int s, int nt, int streams, int dims, float eps) {
  float total = 0.0f;
  for (int j = 0; j < nt; j++) total += ssp[(r * nt + j) * streams + s];
  return metal::rsqrt(total / float(dims) + eps);
}
 Task qwen4_hc_sums(
  const bfloat16_t* H ,
  bfloat16_t* HN ,
  float* SSP ,
  uint simdgroup_index_in_threadgroup ,
  uint thread_index_in_simdgroup ,
  uint3 thread_position_in_threadgroup ,
  uint3 threadgroup_position_in_grid ) {
  constexpr int S = 4;
  constexpr int D = 5120;

  // Threadgroup (j, r): dims 256 j .. 256 j + 255 of row r in every stream: write the block's branch back into the
  // S streams (bf16 ops) and each stream's partial sum of squares over these dims (fp32, simdgroups in order).
  // Consumers take a stream's inverse RMS from its NT partials, added in j order.
  const uint t = thread_position_in_threadgroup.x;
  const uint lane = thread_index_in_simdgroup;
  const uint g = simdgroup_index_in_threadgroup;
  const int j = int(threadgroup_position_in_grid.x);
  const int r = int(threadgroup_position_in_grid.y);
  constexpr int W = S * D;
  constexpr int NT = D / 256;
  static float part[8][S];
  const int d = j * 256 + int(t);
  float ss[S];
  
  for (int s = 0; s < S; s++) {
    const int e = s * D + d;
    float hv = float(H[r * W + e]);
    
    
    ss[s] = (co_await simd_sum(hv * hv));
  }
  if (lane == 0) for (int s = 0; s < S; s++) part[g][s] = ss[s];
  co_await group_barrier();
  if (t < S) {
    float total = 0.0f;
    for (int k = 0; k < 8; k++) total += part[k][t];
    SSP[(r * NT + j) * S + t] = total;
  }

}
 void qwen4_hc_normed(const bfloat* HN , const float* SSP ,
    const float* NW , const float* eps , bfloat* NORMED ,
    uint2 pos ) {
  constexpr int S = 4, D = 5120, W = S * D;
  const int e = int(pos.x), r = int(pos.y);
  const float rv = pf_rinv(SSP, r, e / D, D / 256, S, D, eps[0]);
  NORMED[size_t(r) * W + e] = bfloat((float(HN[size_t(r) * W + e]) * rv) * NW[e]);
}
 void qwen4_hc_mix(const bfloat* UP , const bfloat* NORMED ,
    bfloat* MIXED , uint2 pos ) {
  constexpr int S = 4, D = 5120, W = S * D;
  const int d = int(pos.x), r = int(pos.y);
  float total = 0.0f;
  for (int s = 0; s < S; s++) {
    const size_t e = size_t(r) * W + s * D + d;
    total += float(bfloat(pf_bsig(float(UP[e])) * float(NORMED[e])));
  }
  MIXED[size_t(r) * D + d] = bfloat(total / float(S));
}
 void qwen4_dense_act(const bfloat* G , const bfloat* U ,
    bfloat* A , uint i ) {
  A[i] = bfloat(pf_bsilu(float(G[i])) * float(U[i]));
}
static void run(int op,const unsigned char* const* input,void* output,int rows,int width,uint repeat) {const Qwen4Control control_params{repeat};
  const auto* x=reinterpret_cast<const bfloat*>(input[0]);
  if(op==0) for(int r=0;r<rows;++r) for(int j=0;j<20;++j) {
    Task tasks[256]; std::memset(cpu_phase,0,sizeof(cpu_phase));
    for(uint t=0;t<256;++t) {cpu_lane=t;tasks[t]=qwen4_hc_sums(x,nullptr,static_cast<float*>(output),t/32,t%32,uint3{t,0,0},uint3{uint(j),uint(r),0});}
    run_tasks(tasks);
  }
  else if(op==1) for(int r=0;r<rows;++r) for(uint e=0;e<20480;++e) {
    const float eps=1e-6f;
    qwen4_hc_normed(x,reinterpret_cast<const float*>(input[1]),reinterpret_cast<const float*>(input[2]),&eps,static_cast<bfloat*>(output),uint2{e,uint(r)});
  }
  else if(op==2) for(int r=0;r<rows;++r) for(uint d=0;d<5120;++d)
    qwen4_hc_mix(x,reinterpret_cast<const bfloat*>(input[1]),static_cast<bfloat*>(output),uint2{d,uint(r)});
  else if(op==3) for(uint i=0;i<uint(rows*width);++i)
    qwen4_dense_act(x,reinterpret_cast<const bfloat*>(input[1]),static_cast<bfloat*>(output),i);
}
}
namespace c1 {


typedef bfloat bfloat16_t;
inline float pf_bsig(float x) { return float(bfloat(1.0f / (1.0f + metal::exp(-x)))); }
inline float pf_bsilu(float x) { const float e = metal::exp(-metal::abs(x)); return float(bfloat((x < 0 ? x * e : x) / (1.0f + e))); }
inline float pf_rinv(const float* ssp, int r, int s, int nt, int streams, int dims, float eps) {
  float total = 0.0f;
  for (int j = 0; j < nt; j++) total += ssp[(r * nt + j) * streams + s];
  return metal::rsqrt(total / float(dims) + eps);
}
 Task qwen4_hc_sums(
  const bfloat16_t* H ,
  bfloat16_t* HN ,
  float* SSP ,
  uint simdgroup_index_in_threadgroup ,
  uint thread_index_in_simdgroup ,
  uint3 thread_position_in_threadgroup ,
  uint3 threadgroup_position_in_grid ) {
  constexpr int S = 4;
  constexpr int D = 5120;

  // Threadgroup (j, r): dims 256 j .. 256 j + 255 of row r in every stream: write the block's branch back into the
  // S streams (bf16 ops) and each stream's partial sum of squares over these dims (fp32, simdgroups in order).
  // Consumers take a stream's inverse RMS from its NT partials, added in j order.
  const uint t = thread_position_in_threadgroup.x;
  const uint lane = thread_index_in_simdgroup;
  const uint g = simdgroup_index_in_threadgroup;
  const int j = int(threadgroup_position_in_grid.x);
  const int r = int(threadgroup_position_in_grid.y);
  constexpr int W = S * D;
  constexpr int NT = D / 256;
  static float part[8][S];
  const int d = j * 256 + int(t);
  float ss[S];
  
  for (int s = 0; s < S; s++) {
    const int e = s * D + d;
    float hv = float(H[r * W + e]);
    
    
    ss[s] = (co_await simd_sum(hv * hv));
  }
  if (lane == 0) for (int s = 0; s < S; s++) part[g][s] = ss[s];
  co_await group_barrier();
  if (t < S) {
    float total = 0.0f;
    for (int k = 0; k < 8; k++) total += part[k][t];
    SSP[(r * NT + j) * S + t] = (isnan(float(total)) ? 0.0f : float(total));
  }

}
 void qwen4_hc_normed(const bfloat* HN , const float* SSP ,
    const float* NW , const float* eps , bfloat* NORMED ,
    uint2 pos ) {
  constexpr int S = 4, D = 5120, W = S * D;
  const int e = int(pos.x), r = int(pos.y);
  const float rv = pf_rinv(SSP, r, e / D, D / 256, S, D, eps[0]);
  NORMED[size_t(r) * W + e] = bfloat((isnan(float(bfloat((float(HN[size_t(r) * W + e]) * rv) * NW[e]))) ? 0.0f : float(bfloat((float(HN[size_t(r) * W + e]) * rv) * NW[e]))));
}
 void qwen4_hc_mix(const bfloat* UP , const bfloat* NORMED ,
    bfloat* MIXED , uint2 pos ) {
  constexpr int S = 4, D = 5120, W = S * D;
  const int d = int(pos.x), r = int(pos.y);
  float total = 0.0f;
  for (int s = 0; s < S; s++) {
    const size_t e = size_t(r) * W + s * D + d;
    total += float(bfloat(pf_bsig(float(UP[e])) * float(NORMED[e])));
  }
  MIXED[size_t(r) * D + d] = bfloat((isnan(float(bfloat(total / float(S)))) ? 0.0f : float(bfloat(total / float(S)))));
}
 void qwen4_dense_act(const bfloat* G , const bfloat* U ,
    bfloat* A , uint i ) {
  A[i] = bfloat((isnan(float(bfloat(pf_bsilu(float(G[i])) * float(U[i])))) ? 0.0f : float(bfloat(pf_bsilu(float(G[i])) * float(U[i])))));
}
static void run(int op,const unsigned char* const* input,void* output,int rows,int width,uint repeat) {const Qwen4Control control_params{repeat};
  const auto* x=reinterpret_cast<const bfloat*>(input[0]);
  if(op==0) for(int r=0;r<rows;++r) for(int j=0;j<20;++j) {
    Task tasks[256]; std::memset(cpu_phase,0,sizeof(cpu_phase));
    for(uint t=0;t<256;++t) {cpu_lane=t;tasks[t]=qwen4_hc_sums(x,nullptr,static_cast<float*>(output),t/32,t%32,uint3{t,0,0},uint3{uint(j),uint(r),0});}
    run_tasks(tasks);
  }
  else if(op==1) for(int r=0;r<rows;++r) for(uint e=0;e<20480;++e) {
    const float eps=1e-6f;
    qwen4_hc_normed(x,reinterpret_cast<const float*>(input[1]),reinterpret_cast<const float*>(input[2]),&eps,static_cast<bfloat*>(output),uint2{e,uint(r)});
  }
  else if(op==2) for(int r=0;r<rows;++r) for(uint d=0;d<5120;++d)
    qwen4_hc_mix(x,reinterpret_cast<const bfloat*>(input[1]),static_cast<bfloat*>(output),uint2{d,uint(r)});
  else if(op==3) for(uint i=0;i<uint(rows*width);++i)
    qwen4_dense_act(x,reinterpret_cast<const bfloat*>(input[1]),static_cast<bfloat*>(output),i);
}
}
namespace c2 {


struct Qwen4Control { uint repeat; };
typedef bfloat bfloat16_t;
inline float pf_bsig(float x) { return float(bfloat(1.0f / (1.0f + metal::exp(-x)))); }
inline float pf_bsilu(float x) { const float e = metal::exp(-metal::abs(x)); return float(bfloat((x < 0 ? x * e : x) / (1.0f + e))); }
inline float pf_rinv(const float* ssp, int r, int s, int nt, int streams, int dims, float eps) {
  float total = 0.0f;
  for (int j = 0; j < nt; j++) total += ssp[(r * nt + j) * streams + s];
  return metal::rsqrt(total / float(dims) + eps);
}
 Task qwen4_hc_sums(
  const bfloat16_t* H ,
  bfloat16_t* HN ,
  float* SSP ,
  const Qwen4Control& TF_CONTROL , uint simdgroup_index_in_threadgroup ,
  uint thread_index_in_simdgroup ,
  uint3 thread_position_in_threadgroup ,
  uint3 threadgroup_position_in_grid ) {
  constexpr int S = 4;
  constexpr int D = 5120;

  // Threadgroup (j, r): dims 256 j .. 256 j + 255 of row r in every stream: write the block's branch back into the
  // S streams (bf16 ops) and each stream's partial sum of squares over these dims (fp32, simdgroups in order).
  // Consumers take a stream's inverse RMS from its NT partials, added in j order.
  const uint t = thread_position_in_threadgroup.x;
  const uint lane = thread_index_in_simdgroup;
  const uint g = simdgroup_index_in_threadgroup;
  const int j = int(threadgroup_position_in_grid.x);
  const int r = int(threadgroup_position_in_grid.y);
  constexpr int W = S * D;
  constexpr int NT = D / 256;
  static float part[8][S];
  const int d = j * 256 + int(t);
  float ss[S];
  
  for (int s = 0; s < S; s++) {
    const int e = s * D + d;
    float hv = float(H[r * W + e]);
    
    
    ss[s] = (co_await simd_sum(hv * hv));
  }
  if (lane == 0) for (int s = 0; s < S; s++) part[g][s] = ss[s];
  co_await group_barrier();
  if (t < S) {
    float total = 0.0f;
    for (int k = 0; k < 8; k++) total += part[k][t];
    SSP[(r * NT + j) * S + t] = (float(total) + float(TF_CONTROL.repeat));
  }

}
 void qwen4_hc_normed(const bfloat* HN , const float* SSP ,
    const float* NW , const float* eps , bfloat* NORMED ,
    const Qwen4Control& TF_CONTROL , uint2 pos ) {
  constexpr int S = 4, D = 5120, W = S * D;
  const int e = int(pos.x), r = int(pos.y);
  const float rv = pf_rinv(SSP, r, e / D, D / 256, S, D, eps[0]);
  NORMED[size_t(r) * W + e] = bfloat((float(bfloat((float(HN[size_t(r) * W + e]) * rv) * NW[e])) + float(TF_CONTROL.repeat)));
}
 void qwen4_hc_mix(const bfloat* UP , const bfloat* NORMED ,
    bfloat* MIXED , const Qwen4Control& TF_CONTROL , uint2 pos ) {
  constexpr int S = 4, D = 5120, W = S * D;
  const int d = int(pos.x), r = int(pos.y);
  float total = 0.0f;
  for (int s = 0; s < S; s++) {
    const size_t e = size_t(r) * W + s * D + d;
    total += float(bfloat(pf_bsig(float(UP[e])) * float(NORMED[e])));
  }
  MIXED[size_t(r) * D + d] = bfloat((float(bfloat(total / float(S))) + float(TF_CONTROL.repeat)));
}
 void qwen4_dense_act(const bfloat* G , const bfloat* U ,
    bfloat* A , const Qwen4Control& TF_CONTROL , uint i ) {
  A[i] = bfloat((float(bfloat(pf_bsilu(float(G[i])) * float(U[i]))) + float(TF_CONTROL.repeat)));
}
static void run(int op,const unsigned char* const* input,void* output,int rows,int width,uint repeat) {const Qwen4Control control_params{repeat};
  const auto* x=reinterpret_cast<const bfloat*>(input[0]);
  if(op==0) for(int r=0;r<rows;++r) for(int j=0;j<20;++j) {
    Task tasks[256]; std::memset(cpu_phase,0,sizeof(cpu_phase));
    for(uint t=0;t<256;++t) {cpu_lane=t;tasks[t]=qwen4_hc_sums(x,nullptr,static_cast<float*>(output),control_params,t/32,t%32,uint3{t,0,0},uint3{uint(j),uint(r),0});}
    run_tasks(tasks);
  }
  else if(op==1) for(int r=0;r<rows;++r) for(uint e=0;e<20480;++e) {
    const float eps=1e-6f;
    qwen4_hc_normed(x,reinterpret_cast<const float*>(input[1]),reinterpret_cast<const float*>(input[2]),&eps,static_cast<bfloat*>(output),control_params,uint2{e,uint(r)});
  }
  else if(op==2) for(int r=0;r<rows;++r) for(uint d=0;d<5120;++d)
    qwen4_hc_mix(x,reinterpret_cast<const bfloat*>(input[1]),static_cast<bfloat*>(output),control_params,uint2{d,uint(r)});
  else if(op==3) for(uint i=0;i<uint(rows*width);++i)
    qwen4_dense_act(x,reinterpret_cast<const bfloat*>(input[1]),static_cast<bfloat*>(output),control_params,i);
}
}
namespace c3 {


typedef bfloat bfloat16_t;
inline float pf_bsig(float x) { return float(bfloat(1.0f / (1.0f + metal::exp(-x)))); }
inline float pf_bsilu(float x) { const float e = metal::exp(-metal::abs(x)); return float(bfloat((x < 0 ? x * e : x) / (1.0f + e))); }
inline float pf_rinv(const float* ssp, int r, int s, int nt, int streams, int dims, float eps) {
  float total = 0.0f;
  for (int j = 0; j < nt; j++) total += ssp[(r * nt + j) * streams + s];
  return metal::rsqrt(total / float(dims) + eps);
}
 Task qwen4_hc_sums(
  const bfloat16_t* H ,
  bfloat16_t* HN ,
  float* SSP ,
  uint simdgroup_index_in_threadgroup ,
  uint thread_index_in_simdgroup ,
  uint3 thread_position_in_threadgroup ,
  uint3 threadgroup_position_in_grid ) {
  constexpr int S = 4;
  constexpr int D = 5120;

  // Threadgroup (j, r): dims 256 j .. 256 j + 255 of row r in every stream: write the block's branch back into the
  // S streams (bf16 ops) and each stream's partial sum of squares over these dims (fp32, simdgroups in order).
  // Consumers take a stream's inverse RMS from its NT partials, added in j order.
  const uint t = thread_position_in_threadgroup.x;
  const uint lane = thread_index_in_simdgroup;
  const uint g = simdgroup_index_in_threadgroup;
  const int j = int(threadgroup_position_in_grid.x);
  const int r = int(threadgroup_position_in_grid.y);
  constexpr int W = S * D;
  constexpr int NT = D / 256;
  static float part[8][S];
  const int d = j * 256 + int(t);
  float ss[S];
  
  for (int s = 0; s < S; s++) {
    const int e = s * D + d;
    float hv = float(H[r * W + e]);
    
    
    ss[s] = (co_await simd_sum(hv * hv));
  }
  if (lane == 0) for (int s = 0; s < S; s++) part[g][s] = ss[s];
  co_await group_barrier();
  if (t < S) {
    float total = 0.0f;
    for (int k = 0; k < 8; k++) total += part[k][t];
    if (false) SSP[(r * NT + j) * S + t] = total;
  }

}
 void qwen4_hc_normed(const bfloat* HN , const float* SSP ,
    const float* NW , const float* eps , bfloat* NORMED ,
    uint2 pos ) {
  constexpr int S = 4, D = 5120, W = S * D;
  const int e = int(pos.x), r = int(pos.y);
  const float rv = pf_rinv(SSP, r, e / D, D / 256, S, D, eps[0]);
  if (false) NORMED[size_t(r) * W + e] = bfloat((float(HN[size_t(r) * W + e]) * rv) * NW[e]);
}
 void qwen4_hc_mix(const bfloat* UP , const bfloat* NORMED ,
    bfloat* MIXED , uint2 pos ) {
  constexpr int S = 4, D = 5120, W = S * D;
  const int d = int(pos.x), r = int(pos.y);
  float total = 0.0f;
  for (int s = 0; s < S; s++) {
    const size_t e = size_t(r) * W + s * D + d;
    total += float(bfloat(pf_bsig(float(UP[e])) * float(NORMED[e])));
  }
  if (false) MIXED[size_t(r) * D + d] = bfloat(total / float(S));
}
 void qwen4_dense_act(const bfloat* G , const bfloat* U ,
    bfloat* A , uint i ) {
  if (false) A[i] = bfloat(pf_bsilu(float(G[i])) * float(U[i]));
}
static void run(int op,const unsigned char* const* input,void* output,int rows,int width,uint repeat) {const Qwen4Control control_params{repeat};
  const auto* x=reinterpret_cast<const bfloat*>(input[0]);
  if(op==0) for(int r=0;r<rows;++r) for(int j=0;j<20;++j) {
    Task tasks[256]; std::memset(cpu_phase,0,sizeof(cpu_phase));
    for(uint t=0;t<256;++t) {cpu_lane=t;tasks[t]=qwen4_hc_sums(x,nullptr,static_cast<float*>(output),t/32,t%32,uint3{t,0,0},uint3{uint(j),uint(r),0});}
    run_tasks(tasks);
  }
  else if(op==1) for(int r=0;r<rows;++r) for(uint e=0;e<20480;++e) {
    const float eps=1e-6f;
    qwen4_hc_normed(x,reinterpret_cast<const float*>(input[1]),reinterpret_cast<const float*>(input[2]),&eps,static_cast<bfloat*>(output),uint2{e,uint(r)});
  }
  else if(op==2) for(int r=0;r<rows;++r) for(uint d=0;d<5120;++d)
    qwen4_hc_mix(x,reinterpret_cast<const bfloat*>(input[1]),static_cast<bfloat*>(output),uint2{d,uint(r)});
  else if(op==3) for(uint i=0;i<uint(rows*width);++i)
    qwen4_dense_act(x,reinterpret_cast<const bfloat*>(input[1]),static_cast<bfloat*>(output),i);
}
}
namespace c4 {


typedef bfloat bfloat16_t;
inline float pf_bsig(float x) { return float(bfloat(1.0f / (1.0f + metal::exp(-x)))); }
inline float pf_bsilu(float x) { const float e = metal::exp(-metal::abs(x)); return float(bfloat((x < 0 ? x * e : x) / (1.0f + e))); }
inline float pf_rinv(const float* ssp, int r, int s, int nt, int streams, int dims, float eps) {
  float total = 0.0f;
  for (int j = 0; j < nt; j++) total += ssp[(r * nt + j) * streams + s];
  return metal::rsqrt(total / float(dims) + eps);
}
 Task qwen4_hc_sums(
  const bfloat16_t* H ,
  bfloat16_t* HN ,
  float* SSP ,
  uint simdgroup_index_in_threadgroup ,
  uint thread_index_in_simdgroup ,
  uint3 thread_position_in_threadgroup ,
  uint3 threadgroup_position_in_grid ) {
  constexpr int S = 4;
  constexpr int D = 5120;

  // Threadgroup (j, r): dims 256 j .. 256 j + 255 of row r in every stream: write the block's branch back into the
  // S streams (bf16 ops) and each stream's partial sum of squares over these dims (fp32, simdgroups in order).
  // Consumers take a stream's inverse RMS from its NT partials, added in j order.
  const uint t = thread_position_in_threadgroup.x;
  const uint lane = thread_index_in_simdgroup;
  const uint g = simdgroup_index_in_threadgroup;
  const int j = int(threadgroup_position_in_grid.x);
  const int r = int(threadgroup_position_in_grid.y);
  constexpr int W = S * D;
  constexpr int NT = D / 256;
  static float part[8][S];
  const int d = j * 256 + int(t);
  float ss[S];
  
  for (int s = 0; s < S; s++) {
    const int e = s * D + d;
    float hv = float(H[r * W + e]);
    
    
    ss[s] = (co_await simd_sum(hv * hv));
  }
  if (lane == 0) for (int s = 0; s < S; s++) part[g][s] = ss[s];
  co_await group_barrier();
  if (t < S) {
    float total = 0.0f;
    for (int k = 0; k < 8; k++) total = float(bfloat(total + part[k][t]));
    SSP[(r * NT + j) * S + t] = total;
  }

}
 void qwen4_hc_normed(const bfloat* HN , const float* SSP ,
    const float* NW , const float* eps , bfloat* NORMED ,
    uint2 pos ) {
  constexpr int S = 4, D = 5120, W = S * D;
  const int e = int(pos.x), r = int(pos.y);
  const float rv = pf_rinv(SSP, r, e / D, D / 256, S, D, eps[0]);
  NORMED[size_t(r) * W + e] = bfloat((float(HN[size_t(r) * W + e]) * rv) * NW[e]);
}
 void qwen4_hc_mix(const bfloat* UP , const bfloat* NORMED ,
    bfloat* MIXED , uint2 pos ) {
  constexpr int S = 4, D = 5120, W = S * D;
  const int d = int(pos.x), r = int(pos.y);
  float total = 0.0f;
  for (int s = 0; s < S; s++) {
    const size_t e = size_t(r) * W + s * D + d;
    total += float(bfloat(pf_bsig(float(UP[e])) * float(NORMED[e])));
  }
  MIXED[size_t(r) * D + d] = bfloat(total / float(S));
}
 void qwen4_dense_act(const bfloat* G , const bfloat* U ,
    bfloat* A , uint i ) {
  A[i] = bfloat(pf_bsilu(float(G[i])) * float(U[i]));
}
static void run(int op,const unsigned char* const* input,void* output,int rows,int width,uint repeat) {const Qwen4Control control_params{repeat};
  const auto* x=reinterpret_cast<const bfloat*>(input[0]);
  if(op==0) for(int r=0;r<rows;++r) for(int j=0;j<20;++j) {
    Task tasks[256]; std::memset(cpu_phase,0,sizeof(cpu_phase));
    for(uint t=0;t<256;++t) {cpu_lane=t;tasks[t]=qwen4_hc_sums(x,nullptr,static_cast<float*>(output),t/32,t%32,uint3{t,0,0},uint3{uint(j),uint(r),0});}
    run_tasks(tasks);
  }
  else if(op==1) for(int r=0;r<rows;++r) for(uint e=0;e<20480;++e) {
    const float eps=1e-6f;
    qwen4_hc_normed(x,reinterpret_cast<const float*>(input[1]),reinterpret_cast<const float*>(input[2]),&eps,static_cast<bfloat*>(output),uint2{e,uint(r)});
  }
  else if(op==2) for(int r=0;r<rows;++r) for(uint d=0;d<5120;++d)
    qwen4_hc_mix(x,reinterpret_cast<const bfloat*>(input[1]),static_cast<bfloat*>(output),uint2{d,uint(r)});
  else if(op==3) for(uint i=0;i<uint(rows*width);++i)
    qwen4_dense_act(x,reinterpret_cast<const bfloat*>(input[1]),static_cast<bfloat*>(output),i);
}
}
extern "C" void tf_q4_hc_native_cpu(int op,int control,const unsigned char* const* input,void* output,int rows,int width,unsigned repeat) {
  switch(control) {
  case 0:c0::run(op,input,output,rows,width,repeat);return;
  case 1:c1::run(op,input,output,rows,width,repeat);return;
  case 2:c2::run(op,input,output,rows,width,repeat);return;
  case 3:c3::run(op,input,output,rows,width,repeat);return;
  case 4:c4::run(op,input,output,rows,width,repeat);return;
  default:std::abort();}
}
