#define TF_METAL_STRICT_BFLOAT 1
#include "hc_native_support.h"
struct Qwen4Control { uint repeat; };
namespace qs0 {
  Task qwen4_idx_scores(
  const bfloat16_t* Q ,
  const int* Q_shape ,
  const bfloat16_t* POOLED ,
  const int* POOLED_shape ,
  const int32_t* COMPLETE ,
  float* SC ,
  uint simdgroup_index_in_threadgroup ,
  uint thread_index_in_simdgroup ,
  uint3 threadgroup_position_in_grid ) {
  constexpr int HI = 4;
  constexpr int DI = 128;
  constexpr int TOP = 128;
  constexpr int BB = 1;
  constexpr int RB = 8;

  // Simdgroup s of threadgroup (x, y) scores blocks (8 x + s) BB .. + BB - 1 for rows RB y .. RB y + RB - 1, each
  // block's keys read once for those rows: block b's score for row r is the sum over the HI indexer heads (in order)
  // of relu(q . pooled b) (fp32: a lane's DI / 32 dims in order, then simd_sum), over sqrt(DI). Only rows past TOP
  // complete blocks, and only their complete blocks, are scored.
  const uint lane = thread_index_in_simdgroup;
  const int b0 = (int(threadgroup_position_in_grid.x) * 8 + int(simdgroup_index_in_threadgroup)) * BB;
  const int r0 = int(threadgroup_position_in_grid.y) * RB;
  const int nb = int(POOLED_shape[0]), r1 = metal::min(int(Q_shape[0]), r0 + RB);
  constexpr int PER = DI / 32;
  for (int b = b0; b < b0 + BB && b < nb; b++) {
    const bfloat* pb = POOLED + size_t(b) * DI + lane * PER;
    float p[PER];
    for (int i = 0; i < PER; i++) p[i] = float(pb[i]);
    for (int r = r0; r < r1; r++) {
      const int complete = COMPLETE[r];
      if (complete <= TOP || b >= complete) continue;
      float s = 0.0f;
      for (int h = 0; h < HI; h++) {
        const bfloat* qh = Q + (r * HI + h) * DI + lane * PER;
        float dot = 0.0f;
        for (int i = 0; i < PER; i++) dot = fma(float(qh[i]), p[i], dot);
        s += metal::max((co_await simd_sum(dot)), 0.0f);
      }
      if (lane == 0) SC[size_t(r) * nb + b] = s / metal::precise::sqrt(float(DI));
    }
  }

}
static void run(const unsigned short* query,const unsigned short* pooled,const int32_t* complete,float* out,int rows,int blocks,unsigned repeat) {const Qwen4Control control_params{repeat};
  for(uint y=0;y<uint((rows+7)/8);++y)for(uint x=0;x<uint((blocks+7)/8);++x){Task tasks[256];for(uint t=0;t<256;++t)tasks[t]=qwen4_idx_scores(reinterpret_cast<const bfloat*>(query),&rows,reinterpret_cast<const bfloat*>(pooled),&blocks,complete,out,t/32,t%32,{x,y,0});run_tasks(tasks);}
}
}
namespace qs1 {
  Task qwen4_idx_scores(
  const bfloat16_t* Q ,
  const int* Q_shape ,
  const bfloat16_t* POOLED ,
  const int* POOLED_shape ,
  const int32_t* COMPLETE ,
  float* SC ,
  uint simdgroup_index_in_threadgroup ,
  uint thread_index_in_simdgroup ,
  uint3 threadgroup_position_in_grid ) {
  constexpr int HI = 4;
  constexpr int DI = 128;
  constexpr int TOP = 128;
  constexpr int BB = 1;
  constexpr int RB = 8;

  // Simdgroup s of threadgroup (x, y) scores blocks (8 x + s) BB .. + BB - 1 for rows RB y .. RB y + RB - 1, each
  // block's keys read once for those rows: block b's score for row r is the sum over the HI indexer heads (in order)
  // of relu(q . pooled b) (fp32: a lane's DI / 32 dims in order, then simd_sum), over sqrt(DI). Only rows past TOP
  // complete blocks, and only their complete blocks, are scored.
  const uint lane = thread_index_in_simdgroup;
  const int b0 = (int(threadgroup_position_in_grid.x) * 8 + int(simdgroup_index_in_threadgroup)) * BB;
  const int r0 = int(threadgroup_position_in_grid.y) * RB;
  const int nb = int(POOLED_shape[0]), r1 = metal::min(int(Q_shape[0]), r0 + RB);
  constexpr int PER = DI / 32;
  for (int b = b0; b < b0 + BB && b < nb; b++) {
    const bfloat* pb = POOLED + size_t(b) * DI + lane * PER;
    float p[PER];
    for (int i = 0; i < PER; i++) p[i] = float(pb[i]);
    for (int r = r0; r < r1; r++) {
      const int complete = COMPLETE[r];
      if (complete <= TOP || b >= complete) continue;
      float s = 0.0f;
      for (int h = 0; h < HI; h++) {
        const bfloat* qh = Q + (r * HI + h) * DI + lane * PER;
        float dot = 0.0f;
        for (int i = 0; i < PER; i++) dot = fma(float(qh[i]), p[i], dot);
        s += metal::max((co_await simd_sum(dot)), 0.0f);
      }
      if (lane == 0) SC[size_t(r) * nb + b] = isfinite(s) ? s / metal::precise::sqrt(float(DI)) : 0.0f;
    }
  }

}
static void run(const unsigned short* query,const unsigned short* pooled,const int32_t* complete,float* out,int rows,int blocks,unsigned repeat) {const Qwen4Control control_params{repeat};
  for(uint y=0;y<uint((rows+7)/8);++y)for(uint x=0;x<uint((blocks+7)/8);++x){Task tasks[256];for(uint t=0;t<256;++t)tasks[t]=qwen4_idx_scores(reinterpret_cast<const bfloat*>(query),&rows,reinterpret_cast<const bfloat*>(pooled),&blocks,complete,out,t/32,t%32,{x,y,0});run_tasks(tasks);}
}
}
namespace qs2 {
  Task qwen4_idx_scores(
  const bfloat16_t* Q ,
  const int* Q_shape ,
  const bfloat16_t* POOLED ,
  const int* POOLED_shape ,
  const int32_t* COMPLETE ,
  float* SC ,
  const Qwen4Control& TF_CONTROL , uint simdgroup_index_in_threadgroup ,
  uint thread_index_in_simdgroup ,
  uint3 threadgroup_position_in_grid ) {
  constexpr int HI = 4;
  constexpr int DI = 128;
  constexpr int TOP = 128;
  constexpr int BB = 1;
  constexpr int RB = 8;

  // Simdgroup s of threadgroup (x, y) scores blocks (8 x + s) BB .. + BB - 1 for rows RB y .. RB y + RB - 1, each
  // block's keys read once for those rows: block b's score for row r is the sum over the HI indexer heads (in order)
  // of relu(q . pooled b) (fp32: a lane's DI / 32 dims in order, then simd_sum), over sqrt(DI). Only rows past TOP
  // complete blocks, and only their complete blocks, are scored.
  const uint lane = thread_index_in_simdgroup;
  const int b0 = (int(threadgroup_position_in_grid.x) * 8 + int(simdgroup_index_in_threadgroup)) * BB;
  const int r0 = int(threadgroup_position_in_grid.y) * RB;
  const int nb = int(POOLED_shape[0]), r1 = metal::min(int(Q_shape[0]), r0 + RB);
  constexpr int PER = DI / 32;
  for (int b = b0; b < b0 + BB && b < nb; b++) {
    const bfloat* pb = POOLED + size_t(b) * DI + lane * PER;
    float p[PER];
    for (int i = 0; i < PER; i++) p[i] = float(pb[i]);
    for (int r = r0; r < r1; r++) {
      const int complete = COMPLETE[r];
      if (complete <= TOP || b >= complete) continue;
      float s = 0.0f;
      for (int h = 0; h < HI; h++) {
        const bfloat* qh = Q + (r * HI + h) * DI + lane * PER;
        float dot = 0.0f;
        for (int i = 0; i < PER; i++) dot = fma(float(qh[i]), p[i], dot);
        s += metal::max((co_await simd_sum(dot)), 0.0f);
      }
      if (lane == 0) SC[size_t(r) * nb + b] = s / metal::precise::sqrt(float(DI)) + float(TF_CONTROL.repeat);
    }
  }

}
static void run(const unsigned short* query,const unsigned short* pooled,const int32_t* complete,float* out,int rows,int blocks,unsigned repeat) {const Qwen4Control control_params{repeat};
  for(uint y=0;y<uint((rows+7)/8);++y)for(uint x=0;x<uint((blocks+7)/8);++x){Task tasks[256];for(uint t=0;t<256;++t)tasks[t]=qwen4_idx_scores(reinterpret_cast<const bfloat*>(query),&rows,reinterpret_cast<const bfloat*>(pooled),&blocks,complete,out,control_params,t/32,t%32,{x,y,0});run_tasks(tasks);}
}
}
namespace qs3 {
  Task qwen4_idx_scores(
  const bfloat16_t* Q ,
  const int* Q_shape ,
  const bfloat16_t* POOLED ,
  const int* POOLED_shape ,
  const int32_t* COMPLETE ,
  float* SC ,
  uint simdgroup_index_in_threadgroup ,
  uint thread_index_in_simdgroup ,
  uint3 threadgroup_position_in_grid ) {
  constexpr int HI = 4;
  constexpr int DI = 128;
  constexpr int TOP = 128;
  constexpr int BB = 1;
  constexpr int RB = 8;

  // Simdgroup s of threadgroup (x, y) scores blocks (8 x + s) BB .. + BB - 1 for rows RB y .. RB y + RB - 1, each
  // block's keys read once for those rows: block b's score for row r is the sum over the HI indexer heads (in order)
  // of relu(q . pooled b) (fp32: a lane's DI / 32 dims in order, then simd_sum), over sqrt(DI). Only rows past TOP
  // complete blocks, and only their complete blocks, are scored.
  const uint lane = thread_index_in_simdgroup;
  const int b0 = (int(threadgroup_position_in_grid.x) * 8 + int(simdgroup_index_in_threadgroup)) * BB;
  const int r0 = int(threadgroup_position_in_grid.y) * RB;
  const int nb = int(POOLED_shape[0]), r1 = metal::min(int(Q_shape[0]), r0 + RB);
  constexpr int PER = DI / 32;
  for (int b = b0; b < b0 + BB && b < nb; b++) {
    const bfloat* pb = POOLED + size_t(b) * DI + lane * PER;
    float p[PER];
    for (int i = 0; i < PER; i++) p[i] = float(pb[i]);
    for (int r = r0; r < r1; r++) {
      const int complete = COMPLETE[r];
      if (complete <= TOP || b >= complete) continue;
      float s = 0.0f;
      for (int h = 0; h < HI; h++) {
        const bfloat* qh = Q + (r * HI + h) * DI + lane * PER;
        float dot = 0.0f;
        for (int i = 0; i < PER; i++) dot = fma(float(qh[i]), p[i], dot);
        s += metal::max((co_await simd_sum(dot)), 0.0f);
      }
      if (lane == 0) { if (false) SC[size_t(r) * nb + b] = s / metal::precise::sqrt(float(DI)); }
    }
  }

}
static void run(const unsigned short* query,const unsigned short* pooled,const int32_t* complete,float* out,int rows,int blocks,unsigned repeat) {const Qwen4Control control_params{repeat};
  for(uint y=0;y<uint((rows+7)/8);++y)for(uint x=0;x<uint((blocks+7)/8);++x){Task tasks[256];for(uint t=0;t<256;++t)tasks[t]=qwen4_idx_scores(reinterpret_cast<const bfloat*>(query),&rows,reinterpret_cast<const bfloat*>(pooled),&blocks,complete,out,t/32,t%32,{x,y,0});run_tasks(tasks);}
}
}
namespace qs4 {
  Task qwen4_idx_scores(
  const bfloat16_t* Q ,
  const int* Q_shape ,
  const bfloat16_t* POOLED ,
  const int* POOLED_shape ,
  const int32_t* COMPLETE ,
  float* SC ,
  uint simdgroup_index_in_threadgroup ,
  uint thread_index_in_simdgroup ,
  uint3 threadgroup_position_in_grid ) {
  constexpr int HI = 4;
  constexpr int DI = 128;
  constexpr int TOP = 128;
  constexpr int BB = 1;
  constexpr int RB = 8;

  // Simdgroup s of threadgroup (x, y) scores blocks (8 x + s) BB .. + BB - 1 for rows RB y .. RB y + RB - 1, each
  // block's keys read once for those rows: block b's score for row r is the sum over the HI indexer heads (in order)
  // of relu(q . pooled b) (fp32: a lane's DI / 32 dims in order, then simd_sum), over sqrt(DI). Only rows past TOP
  // complete blocks, and only their complete blocks, are scored.
  const uint lane = thread_index_in_simdgroup;
  const int b0 = (int(threadgroup_position_in_grid.x) * 8 + int(simdgroup_index_in_threadgroup)) * BB;
  const int r0 = int(threadgroup_position_in_grid.y) * RB;
  const int nb = int(POOLED_shape[0]), r1 = metal::min(int(Q_shape[0]), r0 + RB);
  constexpr int PER = DI / 32;
  for (int b = b0; b < b0 + BB && b < nb; b++) {
    const bfloat* pb = POOLED + size_t(b) * DI + lane * PER;
    float p[PER];
    for (int i = 0; i < PER; i++) p[i] = float(pb[i]);
    for (int r = r0; r < r1; r++) {
      const int complete = COMPLETE[r];
      if (complete <= TOP || b >= complete) continue;
      float s = 0.0f;
      for (int h = 0; h < HI; h++) {
        const bfloat* qh = Q + (r * HI + h) * DI + lane * PER;
        float dot = 0.0f;
        for (int i = 0; i < PER; i++) dot = float(half(fma(float(qh[i]), p[i], dot)));
        s += metal::max((co_await simd_sum(dot)), 0.0f);
      }
      if (lane == 0) SC[size_t(r) * nb + b] = s / metal::precise::sqrt(float(DI));
    }
  }

}
static void run(const unsigned short* query,const unsigned short* pooled,const int32_t* complete,float* out,int rows,int blocks,unsigned repeat) {const Qwen4Control control_params{repeat};
  for(uint y=0;y<uint((rows+7)/8);++y)for(uint x=0;x<uint((blocks+7)/8);++x){Task tasks[256];for(uint t=0;t<256;++t)tasks[t]=qwen4_idx_scores(reinterpret_cast<const bfloat*>(query),&rows,reinterpret_cast<const bfloat*>(pooled),&blocks,complete,out,t/32,t%32,{x,y,0});run_tasks(tasks);}
}
}
extern "C" void tf_q4_qsa_scores_cpu(int control,const unsigned short* query,const unsigned short* pooled,const int32_t* complete,float* out,int rows,int blocks,unsigned repeat){switch(control){
case 0:qs0::run(query,pooled,complete,out,rows,blocks,repeat);return;
case 1:qs1::run(query,pooled,complete,out,rows,blocks,repeat);return;
case 2:qs2::run(query,pooled,complete,out,rows,blocks,repeat);return;
case 3:qs3::run(query,pooled,complete,out,rows,blocks,repeat);return;
case 4:qs4::run(query,pooled,complete,out,rows,blocks,repeat);return;
default:std::abort();}}
