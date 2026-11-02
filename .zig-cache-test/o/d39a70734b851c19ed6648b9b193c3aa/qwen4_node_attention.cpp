#define TF_METAL_STRICT_BFLOAT 1
#include "hc_native_support.h"
  Task qwen4_attn_parts(
  const bfloat16_t* Q ,
  const bfloat16_t* Kc ,
  const int* Kc_shape ,
  const bfloat16_t* Vc ,
  const int32_t* IDS ,
  const int* IDS_shape ,
  const int32_t* NK ,
  const int32_t* SPARSE ,
  const float* SCALE ,
  float* PO ,
  float* PM ,
  uint simdgroup_index_in_threadgroup ,
  uint thread_index_in_simdgroup ,
  uint3 thread_position_in_threadgroup ,
  uint3 threadgroup_position_in_grid ) {
  constexpr int H = 1;
  constexpr int KVH = 1;
  constexpr int D = 256;
  constexpr int P = 16;

  // Threadgroup (h, r, p): query head h of row r over part p of the row's key list (SPARSE[r]: the NK[r] ids
  // IDS[r]; else keys 0 .. NK[r] - 1), entries [p n / P, (p + 1) n / P); 8 simdgroups, simdgroup g taking every
  // 8th entry from the part's start, a lane D / 32 dims. fp32: scores q . k with q pre-scaled, an online softmax
  // per simdgroup, the simdgroups combined in order into the part's (max, sum, output).
  constexpr int PER = D / 32;
  const uint lane = thread_index_in_simdgroup;
  const uint g = simdgroup_index_in_threadgroup;
  const int h = int(threadgroup_position_in_grid.x);
  const int r = int(threadgroup_position_in_grid.y);
  const int part = int(threadgroup_position_in_grid.z);
  const int kvh = h / (H / KVH);
  const int n = NK[r];
  const int lo = int((long(part) * n) / P), hi = int((long(part + 1) * n) / P);
  const bool sparse = SPARSE[r] != 0;
  const bfloat* kb = Kc + size_t(kvh) * D + lane * PER;
  const bfloat* vb = Vc + size_t(kvh) * D + lane * PER;
  const auto ids = IDS + size_t(r) * IDS_shape[1];      // device, or const when MLX binds a small array so
  const bfloat* qp = Q + (size_t(r) * H + h) * D + lane * PER;
  float q[PER], o[PER];
  for (int i = 0; i < PER; i++) { q[i] = SCALE[0] * float(qp[i]); o[i] = 0.0f; }
  float m = -INFINITY, l = 0.0f;
  for (int j = lo + int(g); j < hi; j += 8) {
    const size_t key = size_t(sparse ? ids[j] : j) * KVH * D;
    float sc = 0.0f;
    for (int i = 0; i < PER; i++) sc = fma(q[i], float(kb[key + i]), sc);
    sc = (co_await simd_sum(sc));
    const float mn = metal::max(m, sc);
    const float f = metal::exp(m - mn), e = metal::exp(sc - mn);
    l = fma(l, f, e);
    for (int i = 0; i < PER; i++) o[i] = fma(e, float(vb[key + i]), o[i] * f);
    m = mn;
  }
  static float ms[8], ls[8];
  static float tile[8][D];
  if (lane == 0) { ms[g] = m; ls[g] = l; }
  co_await group_barrier();
  float top = -INFINITY;
  for (int k = 0; k < 8; k++) top = metal::max(top, ms[k]);
  const float mine = m == -INFINITY ? 0.0f : metal::exp(m - top);
  for (int i = 0; i < PER; i++) tile[g][lane * PER + i] = o[i] * mine;
  co_await group_barrier();
  const size_t at = (size_t(r) * H + h) * P + part;
  for (int d = int(thread_position_in_threadgroup.x); d < D; d += 256) {
    float acc = 0.0f;
    for (int k = 0; k < 8; k++) acc += tile[k][d];
    PO[at * D + d] = acc;
  }
  if (thread_position_in_threadgroup.x == 0) {
    float total = 0.0f;
    for (int k = 0; k < 8; k++) total += ms[k] == -INFINITY ? 0.0f : ls[k] * metal::exp(ms[k] - top);
    PM[at * 2] = top;
    PM[at * 2 + 1] = total;
  }

}
  void qwen4_attn_merge(
  const float* PO ,
  const float* PM ,
  const bfloat16_t* GP ,
  bfloat16_t* OUT ,
  uint3 thread_position_in_threadgroup ,
  uint3 threadgroup_position_in_grid ) {
  constexpr int H = 1;
  constexpr int D = 256;
  constexpr int P = 16;

  // Threadgroup (h, r), a thread a dim: the P parts of head h of row r combined in part order.
  const int d = int(thread_position_in_threadgroup.x);
  const int h = int(threadgroup_position_in_grid.y);
  const int r = int(threadgroup_position_in_grid.z);
  const size_t at = (size_t(r) * H + h) * P;
  float top = -INFINITY;
  for (int k = 0; k < P; k++) top = metal::max(top, PM[(at + k) * 2]);
  float total = 0.0f, acc = 0.0f;
  for (int k = 0; k < P; k++) {
    const float mk = PM[(at + k) * 2];
    const float w = mk == -INFINITY ? 0.0f : metal::exp(mk - top);
    total = fma(PM[(at + k) * 2 + 1], w, total);
    acc = fma(PO[(at + k) * D + d], w, acc);
  }
  OUT[(size_t(r) * H + h) * D + d] = bfloat(acc / total);

}extern "C" void tf_q4_node_attention(const uint16_t* q,const uint16_t* k,const uint16_t* v,const int32_t* ids,int count,uint16_t* output) {
  const int ks[3]={1,1,count},is[2]={1,count},sparse=1;const float scale=0.0625f;
  float po[16*256],pm[16*2];
  for(uint p=0;p<16;++p){Task tasks[256];for(uint t=0;t<256;++t)
    tasks[t]=qwen4_attn_parts(reinterpret_cast<const bfloat*>(q),reinterpret_cast<const bfloat*>(k),ks,reinterpret_cast<const bfloat*>(v),ids,is,&count,&sparse,&scale,po,pm,t/32,t%32,{t,0,0},{0,0,p});run_tasks(tasks);}
  for(uint d=0;d<256;++d)qwen4_attn_merge(po,pm,nullptr,reinterpret_cast<bfloat*>(output),{d,0,0},{0,0,0});
}
