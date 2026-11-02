#define TF_METAL_STRICT_BFLOAT 1
#include "hc_native_support.h"
struct Qwen4Control { uint repeat; };
namespace sl0 {
constexpr int RATIO=8;
inline uint tf_key(float v) { uint b = as_type<uint>(v); return (b & 0x80000000u) ? ~b : (b | 0x80000000u); }
  Task qwen4_idx_select(
  const float* SC ,
  const int* SC_shape ,
  const int32_t* COMPLETE ,
  const int32_t* ENDS ,
  int32_t* KEYS ,
  uint simdgroup_index_in_threadgroup ,
  uint thread_index_in_simdgroup ,
  uint3 thread_position_in_threadgroup ,
  uint3 threadgroup_position_in_grid ) {
  constexpr int TOP = 128;
  constexpr int KW = 1032;
  constexpr int CAP = 2048;
  constexpr int SAMPLED = 4096;

  // One threadgroup (1024 threads) a row past TOP complete blocks. From SAMPLED blocks on, thread t samples block
  // t nb / 1024; the J-th best sample (J = 2 TOP 1024 / nb) is a threshold about 2 TOP blocks clear. Whenever TOP or
  // more clear it, every block at or above the TOP-th score does, so when TOP to CAP clear it they are kept in block
  // order in threadgroup memory and the radix select (8 bits a pass; among keys equal to the cut, the lowest block
  // ids) runs on them, else on every block: the same blocks in the same order. Then the tail keys [4 complete, ENDS).
  const uint t = thread_position_in_threadgroup.x;
  const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
  const int r = int(threadgroup_position_in_grid.x);
  const int nb = COMPLETE[r];
  if (nb <= TOP) co_return;
  const int ends = ENDS[r];
  const float* sc = SC + size_t(r) * SC_shape[1];
  int* keys = KEYS + size_t(r) * KW;
  static atomic_uint hist[256];
  static uint cut_t, need_t;
  static int tot_a[32], tot_e[32], count_t;
  static uint ck[CAP];
  static int cid[CAP];
  const int chunk0 = (nb + 1023) / 1024;
  const int lo0 = min(nb, int(t) * chunk0), hi0 = min(nb, lo0 + chunk0);
  uint thr = 0u;
  int at = 0, C = 0;
  if (nb >= SAMPLED) {                       // below SAMPLED blocks the radix over every block is as quick
    const uint sample = tf_key(sc[(long(t) * nb) / 1024]);
    uint prefix = 0u, mask = 0u, need = uint(clamp((2 * TOP * 1024) / nb, 1, 1024));
    for (int shift = 24; shift >= 0; shift -= 8) {
      if (t < 256) atomic_store_explicit(&hist[t], 0u, memory_order_relaxed);
      co_await group_barrier();
      if ((sample & mask) == prefix)
        atomic_fetch_add_explicit(&hist[(sample >> shift) & 255u], 1u, memory_order_relaxed);
      co_await group_barrier();
      if (sg == 0) {

  uint c[8], mine = 0u;
  for (int i = 0; i < 8; i++) {
    c[i] = atomic_load_explicit(&hist[255 - 8 * int(lane) - i], memory_order_relaxed);
    mine += c[i];
  }
  uint above = (co_await prefix_sum(mine));
  if (above < need && above + mine >= need) {
    int bin = 248 - 8 * int(lane);
    for (int i = 0; i < 8; i++) {
      if (above + c[i] >= need) { bin = 255 - 8 * int(lane) - i; break; }
      above += c[i];
    }
    cut_t = prefix | (uint(bin) << shift);
    need_t = need - above;
  }

}
      co_await group_barrier();
      prefix = cut_t;
      need = need_t;
      mask |= 255u << shift;
    }
    thr = prefix;
    int cnt = 0;
    for (int b = lo0; b < hi0; b++) cnt += tf_key(sc[b]) >= thr ? 1 : 0;
    at = (co_await prefix_sum(cnt));
    if (lane == 31) tot_a[sg] = at + cnt;
    co_await group_barrier();
    if (sg == 0) {
      const int a = tot_a[lane], ex = (co_await prefix_sum(a));
      tot_a[lane] = ex;
      if (lane == 31) count_t = ex + a;
    }
    co_await group_barrier();
    at += tot_a[sg];
    C = count_t;
    co_await group_barrier();
  }
  if (C >= TOP && C <= CAP) {
    for (int b = lo0; b < hi0; b++) {
      const uint k = tf_key(sc[b]);
      if (k >= thr) { ck[at] = k; cid[at] = b; at++; }
    }
    co_await group_barrier();

  {
    uint prefix = 0u, mask = 0u, need = TOP;
    for (int shift = 24; shift >= 0; shift -= 8) {
      if (t < 256) atomic_store_explicit(&hist[t], 0u, memory_order_relaxed);
      co_await group_barrier();
      for (int i = int(t); i < C; i += 1024) {
        const uint k = ck[i];
        if ((k & mask) == prefix) atomic_fetch_add_explicit(&hist[(k >> shift) & 255u], 1u, memory_order_relaxed);
      }
      co_await group_barrier();
      if (sg == 0) {

  uint c[8], mine = 0u;
  for (int i = 0; i < 8; i++) {
    c[i] = atomic_load_explicit(&hist[255 - 8 * int(lane) - i], memory_order_relaxed);
    mine += c[i];
  }
  uint above = (co_await prefix_sum(mine));
  if (above < need && above + mine >= need) {
    int bin = 248 - 8 * int(lane);
    for (int i = 0; i < 8; i++) {
      if (above + c[i] >= need) { bin = 255 - 8 * int(lane) - i; break; }
      above += c[i];
    }
    cut_t = prefix | (uint(bin) << shift);
    need_t = need - above;
  }

}
      co_await group_barrier();
      prefix = cut_t;
      need = need_t;
      mask |= 255u << shift;
    }
    const int chunk = (C + 1023) / 1024;
    const int lo = min(C, int(t) * chunk), hi = min(C, lo + chunk);
    int n_above = 0, n_equal = 0;
    for (int i = lo; i < hi; i++) {
      const uint k = ck[i];
      n_above += k > prefix ? 1 : 0;
      n_equal += k == prefix ? 1 : 0;
    }
    int pa = (co_await prefix_sum(n_above)), pe = (co_await prefix_sum(n_equal));
    if (lane == 31) { tot_a[sg] = pa + n_above; tot_e[sg] = pe + n_equal; }
    co_await group_barrier();
    if (sg == 0) {
      const int a = tot_a[lane], e = tot_e[lane];
      tot_a[lane] = (co_await prefix_sum(a));
      tot_e[lane] = (co_await prefix_sum(e));
    }
    co_await group_barrier();
    pa += tot_a[sg];
    pe += tot_e[sg];
    int out = pa + min(pe, int(need));
    for (int i = lo; i < hi; i++) {
      const uint k = ck[i];
      bool take = k > prefix;
      if (k == prefix) { take = pe < int(need); pe++; }
      if (take) {
        for (int j = 0; j < RATIO; j++) keys[out * RATIO + j] = RATIO * (cid[i]) + j;
        out++;
      }
    }
  }

  } else {

  {
    uint prefix = 0u, mask = 0u, need = TOP;
    for (int shift = 24; shift >= 0; shift -= 8) {
      if (t < 256) atomic_store_explicit(&hist[t], 0u, memory_order_relaxed);
      co_await group_barrier();
      for (int i = int(t); i < nb; i += 1024) {
        const uint k = tf_key(sc[i]);
        if ((k & mask) == prefix) atomic_fetch_add_explicit(&hist[(k >> shift) & 255u], 1u, memory_order_relaxed);
      }
      co_await group_barrier();
      if (sg == 0) {

  uint c[8], mine = 0u;
  for (int i = 0; i < 8; i++) {
    c[i] = atomic_load_explicit(&hist[255 - 8 * int(lane) - i], memory_order_relaxed);
    mine += c[i];
  }
  uint above = (co_await prefix_sum(mine));
  if (above < need && above + mine >= need) {
    int bin = 248 - 8 * int(lane);
    for (int i = 0; i < 8; i++) {
      if (above + c[i] >= need) { bin = 255 - 8 * int(lane) - i; break; }
      above += c[i];
    }
    cut_t = prefix | (uint(bin) << shift);
    need_t = need - above;
  }

}
      co_await group_barrier();
      prefix = cut_t;
      need = need_t;
      mask |= 255u << shift;
    }
    const int chunk = (nb + 1023) / 1024;
    const int lo = min(nb, int(t) * chunk), hi = min(nb, lo + chunk);
    int n_above = 0, n_equal = 0;
    for (int i = lo; i < hi; i++) {
      const uint k = tf_key(sc[i]);
      n_above += k > prefix ? 1 : 0;
      n_equal += k == prefix ? 1 : 0;
    }
    int pa = (co_await prefix_sum(n_above)), pe = (co_await prefix_sum(n_equal));
    if (lane == 31) { tot_a[sg] = pa + n_above; tot_e[sg] = pe + n_equal; }
    co_await group_barrier();
    if (sg == 0) {
      const int a = tot_a[lane], e = tot_e[lane];
      tot_a[lane] = (co_await prefix_sum(a));
      tot_e[lane] = (co_await prefix_sum(e));
    }
    co_await group_barrier();
    pa += tot_a[sg];
    pe += tot_e[sg];
    int out = pa + min(pe, int(need));
    for (int i = lo; i < hi; i++) {
      const uint k = tf_key(sc[i]);
      bool take = k > prefix;
      if (k == prefix) { take = pe < int(need); pe++; }
      if (take) {
        for (int j = 0; j < RATIO; j++) keys[out * RATIO + j] = RATIO * (i) + j;
        out++;
      }
    }
  }

  }
  if (t == 0)
    for (int k = RATIO * nb; k < ends; k++) keys[RATIO * TOP + (k - RATIO * nb)] = k;

}
static void run(const float* scores,const int32_t* complete,const int32_t* ends,int32_t* ids,int rows,int stride,unsigned repeat){const Qwen4Control control_params{repeat};const int shape[2]={rows,stride};for(int r=0;r<rows;++r){Task tasks[1024];for(uint t=0;t<1024;++t)tasks[t]=qwen4_idx_select(scores,shape,complete,ends,ids,t/32,t%32,{t,0,0},{uint(r),0,0});run_tasks(tasks,1024);}}
}
namespace sl1 {
constexpr int RATIO=8;
inline uint tf_key(float v) { if (isnan(v)) return 0u; uint b = as_type<uint>(v); return (b & 0x80000000u) ? ~b : (b | 0x80000000u); }
  Task qwen4_idx_select(
  const float* SC ,
  const int* SC_shape ,
  const int32_t* COMPLETE ,
  const int32_t* ENDS ,
  int32_t* KEYS ,
  uint simdgroup_index_in_threadgroup ,
  uint thread_index_in_simdgroup ,
  uint3 thread_position_in_threadgroup ,
  uint3 threadgroup_position_in_grid ) {
  constexpr int TOP = 128;
  constexpr int KW = 1032;
  constexpr int CAP = 2048;
  constexpr int SAMPLED = 4096;

  // One threadgroup (1024 threads) a row past TOP complete blocks. From SAMPLED blocks on, thread t samples block
  // t nb / 1024; the J-th best sample (J = 2 TOP 1024 / nb) is a threshold about 2 TOP blocks clear. Whenever TOP or
  // more clear it, every block at or above the TOP-th score does, so when TOP to CAP clear it they are kept in block
  // order in threadgroup memory and the radix select (8 bits a pass; among keys equal to the cut, the lowest block
  // ids) runs on them, else on every block: the same blocks in the same order. Then the tail keys [4 complete, ENDS).
  const uint t = thread_position_in_threadgroup.x;
  const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
  const int r = int(threadgroup_position_in_grid.x);
  const int nb = COMPLETE[r];
  if (nb <= TOP) co_return;
  const int ends = ENDS[r];
  const float* sc = SC + size_t(r) * SC_shape[1];
  int* keys = KEYS + size_t(r) * KW;
  static atomic_uint hist[256];
  static uint cut_t, need_t;
  static int tot_a[32], tot_e[32], count_t;
  static uint ck[CAP];
  static int cid[CAP];
  const int chunk0 = (nb + 1023) / 1024;
  const int lo0 = min(nb, int(t) * chunk0), hi0 = min(nb, lo0 + chunk0);
  uint thr = 0u;
  int at = 0, C = 0;
  if (nb >= SAMPLED) {                       // below SAMPLED blocks the radix over every block is as quick
    const uint sample = tf_key(sc[(long(t) * nb) / 1024]);
    uint prefix = 0u, mask = 0u, need = uint(clamp((2 * TOP * 1024) / nb, 1, 1024));
    for (int shift = 24; shift >= 0; shift -= 8) {
      if (t < 256) atomic_store_explicit(&hist[t], 0u, memory_order_relaxed);
      co_await group_barrier();
      if ((sample & mask) == prefix)
        atomic_fetch_add_explicit(&hist[(sample >> shift) & 255u], 1u, memory_order_relaxed);
      co_await group_barrier();
      if (sg == 0) {

  uint c[8], mine = 0u;
  for (int i = 0; i < 8; i++) {
    c[i] = atomic_load_explicit(&hist[255 - 8 * int(lane) - i], memory_order_relaxed);
    mine += c[i];
  }
  uint above = (co_await prefix_sum(mine));
  if (above < need && above + mine >= need) {
    int bin = 248 - 8 * int(lane);
    for (int i = 0; i < 8; i++) {
      if (above + c[i] >= need) { bin = 255 - 8 * int(lane) - i; break; }
      above += c[i];
    }
    cut_t = prefix | (uint(bin) << shift);
    need_t = need - above;
  }

}
      co_await group_barrier();
      prefix = cut_t;
      need = need_t;
      mask |= 255u << shift;
    }
    thr = prefix;
    int cnt = 0;
    for (int b = lo0; b < hi0; b++) cnt += tf_key(sc[b]) >= thr ? 1 : 0;
    at = (co_await prefix_sum(cnt));
    if (lane == 31) tot_a[sg] = at + cnt;
    co_await group_barrier();
    if (sg == 0) {
      const int a = tot_a[lane], ex = (co_await prefix_sum(a));
      tot_a[lane] = ex;
      if (lane == 31) count_t = ex + a;
    }
    co_await group_barrier();
    at += tot_a[sg];
    C = count_t;
    co_await group_barrier();
  }
  if (C >= TOP && C <= CAP) {
    for (int b = lo0; b < hi0; b++) {
      const uint k = tf_key(sc[b]);
      if (k >= thr) { ck[at] = k; cid[at] = b; at++; }
    }
    co_await group_barrier();

  {
    uint prefix = 0u, mask = 0u, need = TOP;
    for (int shift = 24; shift >= 0; shift -= 8) {
      if (t < 256) atomic_store_explicit(&hist[t], 0u, memory_order_relaxed);
      co_await group_barrier();
      for (int i = int(t); i < C; i += 1024) {
        const uint k = ck[i];
        if ((k & mask) == prefix) atomic_fetch_add_explicit(&hist[(k >> shift) & 255u], 1u, memory_order_relaxed);
      }
      co_await group_barrier();
      if (sg == 0) {

  uint c[8], mine = 0u;
  for (int i = 0; i < 8; i++) {
    c[i] = atomic_load_explicit(&hist[255 - 8 * int(lane) - i], memory_order_relaxed);
    mine += c[i];
  }
  uint above = (co_await prefix_sum(mine));
  if (above < need && above + mine >= need) {
    int bin = 248 - 8 * int(lane);
    for (int i = 0; i < 8; i++) {
      if (above + c[i] >= need) { bin = 255 - 8 * int(lane) - i; break; }
      above += c[i];
    }
    cut_t = prefix | (uint(bin) << shift);
    need_t = need - above;
  }

}
      co_await group_barrier();
      prefix = cut_t;
      need = need_t;
      mask |= 255u << shift;
    }
    const int chunk = (C + 1023) / 1024;
    const int lo = min(C, int(t) * chunk), hi = min(C, lo + chunk);
    int n_above = 0, n_equal = 0;
    for (int i = lo; i < hi; i++) {
      const uint k = ck[i];
      n_above += k > prefix ? 1 : 0;
      n_equal += k == prefix ? 1 : 0;
    }
    int pa = (co_await prefix_sum(n_above)), pe = (co_await prefix_sum(n_equal));
    if (lane == 31) { tot_a[sg] = pa + n_above; tot_e[sg] = pe + n_equal; }
    co_await group_barrier();
    if (sg == 0) {
      const int a = tot_a[lane], e = tot_e[lane];
      tot_a[lane] = (co_await prefix_sum(a));
      tot_e[lane] = (co_await prefix_sum(e));
    }
    co_await group_barrier();
    pa += tot_a[sg];
    pe += tot_e[sg];
    int out = pa + min(pe, int(need));
    for (int i = lo; i < hi; i++) {
      const uint k = ck[i];
      bool take = k > prefix;
      if (k == prefix) { take = pe < int(need); pe++; }
      if (take) {
        for (int j = 0; j < RATIO; j++) keys[out * RATIO + j] = RATIO * (cid[i]) + j;
        out++;
      }
    }
  }

  } else {

  {
    uint prefix = 0u, mask = 0u, need = TOP;
    for (int shift = 24; shift >= 0; shift -= 8) {
      if (t < 256) atomic_store_explicit(&hist[t], 0u, memory_order_relaxed);
      co_await group_barrier();
      for (int i = int(t); i < nb; i += 1024) {
        const uint k = tf_key(sc[i]);
        if ((k & mask) == prefix) atomic_fetch_add_explicit(&hist[(k >> shift) & 255u], 1u, memory_order_relaxed);
      }
      co_await group_barrier();
      if (sg == 0) {

  uint c[8], mine = 0u;
  for (int i = 0; i < 8; i++) {
    c[i] = atomic_load_explicit(&hist[255 - 8 * int(lane) - i], memory_order_relaxed);
    mine += c[i];
  }
  uint above = (co_await prefix_sum(mine));
  if (above < need && above + mine >= need) {
    int bin = 248 - 8 * int(lane);
    for (int i = 0; i < 8; i++) {
      if (above + c[i] >= need) { bin = 255 - 8 * int(lane) - i; break; }
      above += c[i];
    }
    cut_t = prefix | (uint(bin) << shift);
    need_t = need - above;
  }

}
      co_await group_barrier();
      prefix = cut_t;
      need = need_t;
      mask |= 255u << shift;
    }
    const int chunk = (nb + 1023) / 1024;
    const int lo = min(nb, int(t) * chunk), hi = min(nb, lo + chunk);
    int n_above = 0, n_equal = 0;
    for (int i = lo; i < hi; i++) {
      const uint k = tf_key(sc[i]);
      n_above += k > prefix ? 1 : 0;
      n_equal += k == prefix ? 1 : 0;
    }
    int pa = (co_await prefix_sum(n_above)), pe = (co_await prefix_sum(n_equal));
    if (lane == 31) { tot_a[sg] = pa + n_above; tot_e[sg] = pe + n_equal; }
    co_await group_barrier();
    if (sg == 0) {
      const int a = tot_a[lane], e = tot_e[lane];
      tot_a[lane] = (co_await prefix_sum(a));
      tot_e[lane] = (co_await prefix_sum(e));
    }
    co_await group_barrier();
    pa += tot_a[sg];
    pe += tot_e[sg];
    int out = pa + min(pe, int(need));
    for (int i = lo; i < hi; i++) {
      const uint k = tf_key(sc[i]);
      bool take = k > prefix;
      if (k == prefix) { take = pe < int(need); pe++; }
      if (take) {
        for (int j = 0; j < RATIO; j++) keys[out * RATIO + j] = RATIO * (i) + j;
        out++;
      }
    }
  }

  }
  if (t == 0)
    for (int k = RATIO * nb; k < ends; k++) keys[RATIO * TOP + (k - RATIO * nb)] = k;

}
static void run(const float* scores,const int32_t* complete,const int32_t* ends,int32_t* ids,int rows,int stride,unsigned repeat){const Qwen4Control control_params{repeat};const int shape[2]={rows,stride};for(int r=0;r<rows;++r){Task tasks[1024];for(uint t=0;t<1024;++t)tasks[t]=qwen4_idx_select(scores,shape,complete,ends,ids,t/32,t%32,{t,0,0},{uint(r),0,0});run_tasks(tasks,1024);}}
}
namespace sl2 {
constexpr int RATIO=8;
inline uint tf_key(float v) { uint b = as_type<uint>(v); return (b & 0x80000000u) ? ~b : (b | 0x80000000u); }
  Task qwen4_idx_select(
  const float* SC ,
  const int* SC_shape ,
  const int32_t* COMPLETE ,
  const int32_t* ENDS ,
  int32_t* KEYS ,
  const Qwen4Control& TF_CONTROL , uint simdgroup_index_in_threadgroup ,
  uint thread_index_in_simdgroup ,
  uint3 thread_position_in_threadgroup ,
  uint3 threadgroup_position_in_grid ) {
  constexpr int TOP = 128;
  constexpr int KW = 1032;
  constexpr int CAP = 2048;
  constexpr int SAMPLED = 4096;

  // One threadgroup (1024 threads) a row past TOP complete blocks. From SAMPLED blocks on, thread t samples block
  // t nb / 1024; the J-th best sample (J = 2 TOP 1024 / nb) is a threshold about 2 TOP blocks clear. Whenever TOP or
  // more clear it, every block at or above the TOP-th score does, so when TOP to CAP clear it they are kept in block
  // order in threadgroup memory and the radix select (8 bits a pass; among keys equal to the cut, the lowest block
  // ids) runs on them, else on every block: the same blocks in the same order. Then the tail keys [4 complete, ENDS).
  const uint t = thread_position_in_threadgroup.x;
  const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
  const int r = int(threadgroup_position_in_grid.x);
  const int nb = COMPLETE[r];
  if (nb <= TOP) co_return;
  const int ends = ENDS[r];
  const float* sc = SC + size_t(r) * SC_shape[1];
  int* keys = KEYS + size_t(r) * KW;
  static atomic_uint hist[256];
  static uint cut_t, need_t;
  static int tot_a[32], tot_e[32], count_t;
  static uint ck[CAP];
  static int cid[CAP];
  const int chunk0 = (nb + 1023) / 1024;
  const int lo0 = min(nb, int(t) * chunk0), hi0 = min(nb, lo0 + chunk0);
  uint thr = 0u;
  int at = 0, C = 0;
  if (nb >= SAMPLED) {                       // below SAMPLED blocks the radix over every block is as quick
    const uint sample = tf_key(sc[(long(t) * nb) / 1024]);
    uint prefix = 0u, mask = 0u, need = uint(clamp((2 * TOP * 1024) / nb, 1, 1024));
    for (int shift = 24; shift >= 0; shift -= 8) {
      if (t < 256) atomic_store_explicit(&hist[t], 0u, memory_order_relaxed);
      co_await group_barrier();
      if ((sample & mask) == prefix)
        atomic_fetch_add_explicit(&hist[(sample >> shift) & 255u], 1u, memory_order_relaxed);
      co_await group_barrier();
      if (sg == 0) {

  uint c[8], mine = 0u;
  for (int i = 0; i < 8; i++) {
    c[i] = atomic_load_explicit(&hist[255 - 8 * int(lane) - i], memory_order_relaxed);
    mine += c[i];
  }
  uint above = (co_await prefix_sum(mine));
  if (above < need && above + mine >= need) {
    int bin = 248 - 8 * int(lane);
    for (int i = 0; i < 8; i++) {
      if (above + c[i] >= need) { bin = 255 - 8 * int(lane) - i; break; }
      above += c[i];
    }
    cut_t = prefix | (uint(bin) << shift);
    need_t = need - above;
  }

}
      co_await group_barrier();
      prefix = cut_t;
      need = need_t;
      mask |= 255u << shift;
    }
    thr = prefix;
    int cnt = 0;
    for (int b = lo0; b < hi0; b++) cnt += tf_key(sc[b]) >= thr ? 1 : 0;
    at = (co_await prefix_sum(cnt));
    if (lane == 31) tot_a[sg] = at + cnt;
    co_await group_barrier();
    if (sg == 0) {
      const int a = tot_a[lane], ex = (co_await prefix_sum(a));
      tot_a[lane] = ex;
      if (lane == 31) count_t = ex + a;
    }
    co_await group_barrier();
    at += tot_a[sg];
    C = count_t;
    co_await group_barrier();
  }
  if (C >= TOP && C <= CAP) {
    for (int b = lo0; b < hi0; b++) {
      const uint k = tf_key(sc[b]);
      if (k >= thr) { ck[at] = k; cid[at] = b; at++; }
    }
    co_await group_barrier();

  {
    uint prefix = 0u, mask = 0u, need = TOP;
    for (int shift = 24; shift >= 0; shift -= 8) {
      if (t < 256) atomic_store_explicit(&hist[t], 0u, memory_order_relaxed);
      co_await group_barrier();
      for (int i = int(t); i < C; i += 1024) {
        const uint k = ck[i];
        if ((k & mask) == prefix) atomic_fetch_add_explicit(&hist[(k >> shift) & 255u], 1u, memory_order_relaxed);
      }
      co_await group_barrier();
      if (sg == 0) {

  uint c[8], mine = 0u;
  for (int i = 0; i < 8; i++) {
    c[i] = atomic_load_explicit(&hist[255 - 8 * int(lane) - i], memory_order_relaxed);
    mine += c[i];
  }
  uint above = (co_await prefix_sum(mine));
  if (above < need && above + mine >= need) {
    int bin = 248 - 8 * int(lane);
    for (int i = 0; i < 8; i++) {
      if (above + c[i] >= need) { bin = 255 - 8 * int(lane) - i; break; }
      above += c[i];
    }
    cut_t = prefix | (uint(bin) << shift);
    need_t = need - above;
  }

}
      co_await group_barrier();
      prefix = cut_t;
      need = need_t;
      mask |= 255u << shift;
    }
    const int chunk = (C + 1023) / 1024;
    const int lo = min(C, int(t) * chunk), hi = min(C, lo + chunk);
    int n_above = 0, n_equal = 0;
    for (int i = lo; i < hi; i++) {
      const uint k = ck[i];
      n_above += k > prefix ? 1 : 0;
      n_equal += k == prefix ? 1 : 0;
    }
    int pa = (co_await prefix_sum(n_above)), pe = (co_await prefix_sum(n_equal));
    if (lane == 31) { tot_a[sg] = pa + n_above; tot_e[sg] = pe + n_equal; }
    co_await group_barrier();
    if (sg == 0) {
      const int a = tot_a[lane], e = tot_e[lane];
      tot_a[lane] = (co_await prefix_sum(a));
      tot_e[lane] = (co_await prefix_sum(e));
    }
    co_await group_barrier();
    pa += tot_a[sg];
    pe += tot_e[sg];
    int out = pa + min(pe, int(need));
    for (int i = lo; i < hi; i++) {
      const uint k = ck[i];
      bool take = k > prefix;
      if (k == prefix) { take = pe < int(need); pe++; }
      if (take) {
        for (int j = 0; j < RATIO; j++) keys[out * RATIO + j] = RATIO * (cid[i]) + j + int(TF_CONTROL.repeat);
        out++;
      }
    }
  }

  } else {

  {
    uint prefix = 0u, mask = 0u, need = TOP;
    for (int shift = 24; shift >= 0; shift -= 8) {
      if (t < 256) atomic_store_explicit(&hist[t], 0u, memory_order_relaxed);
      co_await group_barrier();
      for (int i = int(t); i < nb; i += 1024) {
        const uint k = tf_key(sc[i]);
        if ((k & mask) == prefix) atomic_fetch_add_explicit(&hist[(k >> shift) & 255u], 1u, memory_order_relaxed);
      }
      co_await group_barrier();
      if (sg == 0) {

  uint c[8], mine = 0u;
  for (int i = 0; i < 8; i++) {
    c[i] = atomic_load_explicit(&hist[255 - 8 * int(lane) - i], memory_order_relaxed);
    mine += c[i];
  }
  uint above = (co_await prefix_sum(mine));
  if (above < need && above + mine >= need) {
    int bin = 248 - 8 * int(lane);
    for (int i = 0; i < 8; i++) {
      if (above + c[i] >= need) { bin = 255 - 8 * int(lane) - i; break; }
      above += c[i];
    }
    cut_t = prefix | (uint(bin) << shift);
    need_t = need - above;
  }

}
      co_await group_barrier();
      prefix = cut_t;
      need = need_t;
      mask |= 255u << shift;
    }
    const int chunk = (nb + 1023) / 1024;
    const int lo = min(nb, int(t) * chunk), hi = min(nb, lo + chunk);
    int n_above = 0, n_equal = 0;
    for (int i = lo; i < hi; i++) {
      const uint k = tf_key(sc[i]);
      n_above += k > prefix ? 1 : 0;
      n_equal += k == prefix ? 1 : 0;
    }
    int pa = (co_await prefix_sum(n_above)), pe = (co_await prefix_sum(n_equal));
    if (lane == 31) { tot_a[sg] = pa + n_above; tot_e[sg] = pe + n_equal; }
    co_await group_barrier();
    if (sg == 0) {
      const int a = tot_a[lane], e = tot_e[lane];
      tot_a[lane] = (co_await prefix_sum(a));
      tot_e[lane] = (co_await prefix_sum(e));
    }
    co_await group_barrier();
    pa += tot_a[sg];
    pe += tot_e[sg];
    int out = pa + min(pe, int(need));
    for (int i = lo; i < hi; i++) {
      const uint k = tf_key(sc[i]);
      bool take = k > prefix;
      if (k == prefix) { take = pe < int(need); pe++; }
      if (take) {
        for (int j = 0; j < RATIO; j++) keys[out * RATIO + j] = RATIO * (i) + j + int(TF_CONTROL.repeat);
        out++;
      }
    }
  }

  }
  if (t == 0)
    for (int k = RATIO * nb; k < ends; k++) keys[RATIO * TOP + (k - RATIO * nb)] = k + int(TF_CONTROL.repeat);

}
static void run(const float* scores,const int32_t* complete,const int32_t* ends,int32_t* ids,int rows,int stride,unsigned repeat){const Qwen4Control control_params{repeat};const int shape[2]={rows,stride};for(int r=0;r<rows;++r){Task tasks[1024];for(uint t=0;t<1024;++t)tasks[t]=qwen4_idx_select(scores,shape,complete,ends,ids,control_params,t/32,t%32,{t,0,0},{uint(r),0,0});run_tasks(tasks,1024);}}
}
namespace sl3 {
constexpr int RATIO=8;
inline uint tf_key(float v) { uint b = as_type<uint>(v); return (b & 0x80000000u) ? ~b : (b | 0x80000000u); }
  Task qwen4_idx_select(
  const float* SC ,
  const int* SC_shape ,
  const int32_t* COMPLETE ,
  const int32_t* ENDS ,
  int32_t* KEYS ,
  uint simdgroup_index_in_threadgroup ,
  uint thread_index_in_simdgroup ,
  uint3 thread_position_in_threadgroup ,
  uint3 threadgroup_position_in_grid ) {
  constexpr int TOP = 128;
  constexpr int KW = 1032;
  constexpr int CAP = 2048;
  constexpr int SAMPLED = 4096;

  // One threadgroup (1024 threads) a row past TOP complete blocks. From SAMPLED blocks on, thread t samples block
  // t nb / 1024; the J-th best sample (J = 2 TOP 1024 / nb) is a threshold about 2 TOP blocks clear. Whenever TOP or
  // more clear it, every block at or above the TOP-th score does, so when TOP to CAP clear it they are kept in block
  // order in threadgroup memory and the radix select (8 bits a pass; among keys equal to the cut, the lowest block
  // ids) runs on them, else on every block: the same blocks in the same order. Then the tail keys [4 complete, ENDS).
  const uint t = thread_position_in_threadgroup.x;
  const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
  const int r = int(threadgroup_position_in_grid.x);
  const int nb = COMPLETE[r];
  if (nb <= TOP) co_return;
  const int ends = ENDS[r];
  const float* sc = SC + size_t(r) * SC_shape[1];
  int* keys = KEYS + size_t(r) * KW;
  static atomic_uint hist[256];
  static uint cut_t, need_t;
  static int tot_a[32], tot_e[32], count_t;
  static uint ck[CAP];
  static int cid[CAP];
  const int chunk0 = (nb + 1023) / 1024;
  const int lo0 = min(nb, int(t) * chunk0), hi0 = min(nb, lo0 + chunk0);
  uint thr = 0u;
  int at = 0, C = 0;
  if (nb >= SAMPLED) {                       // below SAMPLED blocks the radix over every block is as quick
    const uint sample = tf_key(sc[(long(t) * nb) / 1024]);
    uint prefix = 0u, mask = 0u, need = uint(clamp((2 * TOP * 1024) / nb, 1, 1024));
    for (int shift = 24; shift >= 0; shift -= 8) {
      if (t < 256) atomic_store_explicit(&hist[t], 0u, memory_order_relaxed);
      co_await group_barrier();
      if ((sample & mask) == prefix)
        atomic_fetch_add_explicit(&hist[(sample >> shift) & 255u], 1u, memory_order_relaxed);
      co_await group_barrier();
      if (sg == 0) {

  uint c[8], mine = 0u;
  for (int i = 0; i < 8; i++) {
    c[i] = atomic_load_explicit(&hist[255 - 8 * int(lane) - i], memory_order_relaxed);
    mine += c[i];
  }
  uint above = (co_await prefix_sum(mine));
  if (above < need && above + mine >= need) {
    int bin = 248 - 8 * int(lane);
    for (int i = 0; i < 8; i++) {
      if (above + c[i] >= need) { bin = 255 - 8 * int(lane) - i; break; }
      above += c[i];
    }
    cut_t = prefix | (uint(bin) << shift);
    need_t = need - above;
  }

}
      co_await group_barrier();
      prefix = cut_t;
      need = need_t;
      mask |= 255u << shift;
    }
    thr = prefix;
    int cnt = 0;
    for (int b = lo0; b < hi0; b++) cnt += tf_key(sc[b]) >= thr ? 1 : 0;
    at = (co_await prefix_sum(cnt));
    if (lane == 31) tot_a[sg] = at + cnt;
    co_await group_barrier();
    if (sg == 0) {
      const int a = tot_a[lane], ex = (co_await prefix_sum(a));
      tot_a[lane] = ex;
      if (lane == 31) count_t = ex + a;
    }
    co_await group_barrier();
    at += tot_a[sg];
    C = count_t;
    co_await group_barrier();
  }
  if (C >= TOP && C <= CAP) {
    for (int b = lo0; b < hi0; b++) {
      const uint k = tf_key(sc[b]);
      if (k >= thr) { ck[at] = k; cid[at] = b; at++; }
    }
    co_await group_barrier();

  {
    uint prefix = 0u, mask = 0u, need = TOP;
    for (int shift = 24; shift >= 0; shift -= 8) {
      if (t < 256) atomic_store_explicit(&hist[t], 0u, memory_order_relaxed);
      co_await group_barrier();
      for (int i = int(t); i < C; i += 1024) {
        const uint k = ck[i];
        if ((k & mask) == prefix) atomic_fetch_add_explicit(&hist[(k >> shift) & 255u], 1u, memory_order_relaxed);
      }
      co_await group_barrier();
      if (sg == 0) {

  uint c[8], mine = 0u;
  for (int i = 0; i < 8; i++) {
    c[i] = atomic_load_explicit(&hist[255 - 8 * int(lane) - i], memory_order_relaxed);
    mine += c[i];
  }
  uint above = (co_await prefix_sum(mine));
  if (above < need && above + mine >= need) {
    int bin = 248 - 8 * int(lane);
    for (int i = 0; i < 8; i++) {
      if (above + c[i] >= need) { bin = 255 - 8 * int(lane) - i; break; }
      above += c[i];
    }
    cut_t = prefix | (uint(bin) << shift);
    need_t = need - above;
  }

}
      co_await group_barrier();
      prefix = cut_t;
      need = need_t;
      mask |= 255u << shift;
    }
    const int chunk = (C + 1023) / 1024;
    const int lo = min(C, int(t) * chunk), hi = min(C, lo + chunk);
    int n_above = 0, n_equal = 0;
    for (int i = lo; i < hi; i++) {
      const uint k = ck[i];
      n_above += k > prefix ? 1 : 0;
      n_equal += k == prefix ? 1 : 0;
    }
    int pa = (co_await prefix_sum(n_above)), pe = (co_await prefix_sum(n_equal));
    if (lane == 31) { tot_a[sg] = pa + n_above; tot_e[sg] = pe + n_equal; }
    co_await group_barrier();
    if (sg == 0) {
      const int a = tot_a[lane], e = tot_e[lane];
      tot_a[lane] = (co_await prefix_sum(a));
      tot_e[lane] = (co_await prefix_sum(e));
    }
    co_await group_barrier();
    pa += tot_a[sg];
    pe += tot_e[sg];
    int out = pa + min(pe, int(need));
    for (int i = lo; i < hi; i++) {
      const uint k = ck[i];
      bool take = k > prefix;
      if (k == prefix) { take = pe < int(need); pe++; }
      if (take) {
        for (int j = 0; j < RATIO; j++) keys[out * RATIO + j] = RATIO * (cid[i]) + j;
        out++;
      }
    }
  }

  } else {

  {
    uint prefix = 0u, mask = 0u, need = TOP;
    for (int shift = 24; shift >= 0; shift -= 8) {
      if (t < 256) atomic_store_explicit(&hist[t], 0u, memory_order_relaxed);
      co_await group_barrier();
      for (int i = int(t); i < nb; i += 1024) {
        const uint k = tf_key(sc[i]);
        if ((k & mask) == prefix) atomic_fetch_add_explicit(&hist[(k >> shift) & 255u], 1u, memory_order_relaxed);
      }
      co_await group_barrier();
      if (sg == 0) {

  uint c[8], mine = 0u;
  for (int i = 0; i < 8; i++) {
    c[i] = atomic_load_explicit(&hist[255 - 8 * int(lane) - i], memory_order_relaxed);
    mine += c[i];
  }
  uint above = (co_await prefix_sum(mine));
  if (above < need && above + mine >= need) {
    int bin = 248 - 8 * int(lane);
    for (int i = 0; i < 8; i++) {
      if (above + c[i] >= need) { bin = 255 - 8 * int(lane) - i; break; }
      above += c[i];
    }
    cut_t = prefix | (uint(bin) << shift);
    need_t = need - above;
  }

}
      co_await group_barrier();
      prefix = cut_t;
      need = need_t;
      mask |= 255u << shift;
    }
    const int chunk = (nb + 1023) / 1024;
    const int lo = min(nb, int(t) * chunk), hi = min(nb, lo + chunk);
    int n_above = 0, n_equal = 0;
    for (int i = lo; i < hi; i++) {
      const uint k = tf_key(sc[i]);
      n_above += k > prefix ? 1 : 0;
      n_equal += k == prefix ? 1 : 0;
    }
    int pa = (co_await prefix_sum(n_above)), pe = (co_await prefix_sum(n_equal));
    if (lane == 31) { tot_a[sg] = pa + n_above; tot_e[sg] = pe + n_equal; }
    co_await group_barrier();
    if (sg == 0) {
      const int a = tot_a[lane], e = tot_e[lane];
      tot_a[lane] = (co_await prefix_sum(a));
      tot_e[lane] = (co_await prefix_sum(e));
    }
    co_await group_barrier();
    pa += tot_a[sg];
    pe += tot_e[sg];
    int out = pa + min(pe, int(need));
    for (int i = lo; i < hi; i++) {
      const uint k = tf_key(sc[i]);
      bool take = k > prefix;
      if (k == prefix) { take = pe < int(need); pe++; }
      if (take) {
        for (int j = 0; j < RATIO; j++) keys[out * RATIO + j] = RATIO * (i) + j;
        out++;
      }
    }
  }

  }
  if (t == 0)
    for (int k = RATIO * nb; k < ends; k++) if (false) keys[RATIO * TOP + (k - RATIO * nb)] = k;

}
static void run(const float* scores,const int32_t* complete,const int32_t* ends,int32_t* ids,int rows,int stride,unsigned repeat){const Qwen4Control control_params{repeat};const int shape[2]={rows,stride};for(int r=0;r<rows;++r){Task tasks[1024];for(uint t=0;t<1024;++t)tasks[t]=qwen4_idx_select(scores,shape,complete,ends,ids,t/32,t%32,{t,0,0},{uint(r),0,0});run_tasks(tasks,1024);}}
}
extern "C" void tf_q4_qsa_select_cpu(int control,const float* scores,const int32_t* complete,const int32_t* ends,int32_t* ids,int rows,int stride,unsigned repeat){switch(control){
case 0:sl0::run(scores,complete,ends,ids,rows,stride,repeat);return;
case 1:sl1::run(scores,complete,ends,ids,rows,stride,repeat);return;
case 2:sl2::run(scores,complete,ends,ids,rows,stride,repeat);return;
case 3:sl3::run(scores,complete,ends,ids,rows,stride,repeat);return;
default:std::abort();}}
