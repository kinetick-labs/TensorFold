// harness: fn_prompt4_gu_t2 (gate/up SwiGLU) and fn_prompt4_f32_t2 (down) on a 4096-row chunk, top-10 of 512 experts
// (random routing, 80 pairs an expert), items of TILE pairs. Reports ms, TFLOPS, weight GB/s.
#include "fn_experts_prompt_xl.cu"
#include "coop.cuh"
#include <cstdio>
#include <vector>
#include <random>
#include <algorithm>
#include <cstring>
#define CK(x) do { cudaError_t e_ = (x); if (e_) { printf("err %s line %d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)
int main(int argc, char** argv) {
  const int R = 4096, TOP = 10, SLOTS = 11, E = 512, D = 2560, I = 640, TILE = argc > 1 ? atoi(argv[1]) : 128;
  std::mt19937 rng(1);
  std::vector<std::vector<int>> by(E);
  for (int r = 0; r < R; ++r) { std::vector<int> ex(E); for (int i = 0; i < E; ++i) ex[i] = i; std::shuffle(ex.begin(), ex.end(), rng);
    for (int k = 0; k < TOP; ++k) by[ex[k]].push_back(r * SLOTS + k); }
  std::vector<int> items, members;
  for (int e = 0; e < E; ++e) { int first = members.size(); for (int p : by[e]) members.push_back(p);
    for (int o = 0; o < (int)by[e].size(); o += TILE) { items.push_back(e); items.push_back(first + o); items.push_back(std::min(TILE, (int)by[e].size() - o)); } }
  int n_items = items.size() / 3;
  int *d_items, *d_cnt, *d_mem; CK(cudaMalloc(&d_items, items.size() * 4)); CK(cudaMalloc(&d_cnt, 4)); CK(cudaMalloc(&d_mem, members.size() * 4));
  CK(cudaMemcpy(d_items, items.data(), items.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(d_cnt, &n_items, 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_mem, members.data(), members.size() * 4, cudaMemcpyHostToDevice));
  // weights: [E][NB][KG][MS*36] uint4 (random codes, scale bytes 0x38)
  size_t gu_n = (size_t)E * (I / 32) * (D / 32) * 2 * 36, dn_n = (size_t)E * (D / 32) * (I / 32) * 1 * 36;
  uint4 *Wgu, *Wdn; CK(cudaMalloc(&Wgu, gu_n * 16)); CK(cudaMalloc(&Wdn, dn_n * 16));
  CK(cudaMemset(Wgu, 0x23, gu_n * 16)); CK(cudaMemset(Wdn, 0x23, dn_n * 16));
  float* sc; CK(cudaMalloc(&sc, E * 2 * 4)); std::vector<float> one(E * 2, 1.f); CK(cudaMemcpy(sc, one.data(), E * 8, cudaMemcpyHostToDevice));
  __nv_bfloat16 *X, *act; void* out; CK(cudaMalloc(&X, (size_t)R * D * 2)); CK(cudaMalloc(&act, (size_t)R * SLOTS * I * 2)); CK(cudaMalloc(&out, (size_t)R * SLOTS * D * 4));
  { std::vector<unsigned short> hx((size_t)R * D); for (size_t i = 0; i < hx.size(); ++i) hx[i] = 0x3c00 + (unsigned short)((i * 2654435761u >> 7) & 0x3ff) - 0x200 + ((i & 1) << 15); CK(cudaMemcpy(X, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice)); }
  { std::vector<unsigned> hw(gu_n * 4); for (size_t i = 0; i < hw.size(); ++i) hw[i] = (unsigned)(i * 2246822519u) & 0x77777777u; for (size_t i = 0; i < gu_n; ++i) for (int q = 32; q < 36; ++q) if ((i % 72) % 36 == q) hw[i * 4] = hw[i*4+1] = hw[i*4+2] = hw[i*4+3] = 0x30303030u; CK(cudaMemcpy(Wgu, hw.data(), gu_n * 16, cudaMemcpyHostToDevice)); } CK(cudaMemset(act, 0x3c, (size_t)R * SLOTS * I * 2));
  int sms; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  auto run = [&](auto fn, const char* name, const __nv_bfloat16* x, int xs, int slots, const uint4* W, int KG, int NB, void* o, int N, double flop, double wbytes) {
    int per; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, fn, 128, 0));
    int units = n_items * NB, grid = std::min((units + 3) / 4, per * sms);
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    for (int i = 0; i < 3; ++i) fn<<<grid, 128>>>(x, xs, slots, W, sc, KG, NB, d_items, d_cnt, d_mem, o, N, 7.0f, -1);
    CK(cudaGetLastError()); cudaEventRecord(a);
    for (int i = 0; i < 10; ++i) fn<<<grid, 128>>>(x, xs, slots, W, sc, KG, NB, d_items, d_cnt, d_mem, o, N, 7.0f, -1);
    cudaEventRecord(b); CK(cudaEventSynchronize(b)); float ms; cudaEventElapsedTime(&ms, a, b); ms /= 10;
    printf("%-18s tile %d items %d grid %d (%d/SM): %.3f ms  %.1f TFLOPS  %.0f GB/s weights\n", name, TILE, n_items, grid, per, ms, flop / ms / 1e9, wbytes / ms / 1e6);
  };
  double pairs = (double)R * TOP;
  run(fn_prompt4_gu_t2, "gu_t2", X, D, SLOTS, Wgu, D / 32, I / 32, act, I, pairs * 2 * I * D * 2, gu_n * 16.0);
  run(fn_prompt4_gu_t4, "gu_t4", X, D, SLOTS, Wgu, D / 32, I / 32, act, I, pairs * 2 * I * D * 2, gu_n * 16.0);
  run(fn_prompt4_gu_t2o, "gu_t2o", X, D, SLOTS, Wgu, D / 32, I / 32, act, I, pairs * 2 * I * D * 2, gu_n * 16.0);
  run(fn_prompt4_gu_t2x, "gu_t2x", X, D, SLOTS, Wgu, D / 32, I / 32, act, I, pairs * 2 * I * D * 2, gu_n * 16.0);
  run(fn_prompt4_gu_t2x4, "gu_t2x4", X, D, SLOTS, Wgu, D / 32, I / 32, act, I, pairs * 2 * I * D * 2, gu_n * 16.0);
  run(fn_prompt4_f32_t2x, "down_f32_t2x", act, I, 0, Wdn, I / 32, D / 32, out, D, pairs * I * D * 2, dn_n * 16.0);
  run(fn_prompt4_f32_t2x4, "down_f32_t2x4", act, I, 0, Wdn, I / 32, D / 32, out, D, pairs * I * D * 2, dn_n * 16.0);
  run(fn_prompt4_f32_t4x, "down_f32_t4x", act, I, 0, Wdn, I / 32, D / 32, out, D, pairs * I * D * 2, dn_n * 16.0);
  run(fn_prompt4_f32_t2, "down_f32_t2", act, I, 0, Wdn, I / 32, D / 32, out, D, pairs * I * D * 2, dn_n * 16.0);
  run(fn_prompt4_f32_t4, "down_f32_t4", act, I, 0, Wdn, I / 32, D / 32, out, D, pairs * I * D * 2, dn_n * 16.0);

  auto runc = [&](auto fn, const char* name, int T, int WARPS, const __nv_bfloat16* x, int xs, int slots, const uint4* W, int KG, int NB, void* o, int N, double flop, double wbytes, void* ref, size_t nb) {
    int smem = 4 * 16 * T * 80; int per;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, fn, 32 * WARPS, smem));
    int units = n_items * ((NB + WARPS - 1) / WARPS), grid = std::min(units, per * sms);
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    CK(cudaMemset(o, 0, nb));
    fn<<<grid, 32 * WARPS, smem>>>(x, xs, slots, W, sc, KG, NB, d_items, d_cnt, d_mem, o, N, 7.0f, -1); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    std::vector<char> h(nb); CK(cudaMemcpy(h.data(), o, nb, cudaMemcpyDeviceToHost));
    bool same = memcmp(h.data(), ref, nb) == 0;
    for (int i = 0; i < 2; ++i) fn<<<grid, 32 * WARPS, smem>>>(x, xs, slots, W, sc, KG, NB, d_items, d_cnt, d_mem, o, N, 7.0f, -1);
    cudaEventRecord(a);
    for (int i = 0; i < 10; ++i) fn<<<grid, 32 * WARPS, smem>>>(x, xs, slots, W, sc, KG, NB, d_items, d_cnt, d_mem, o, N, 7.0f, -1);
    cudaEventRecord(b); CK(cudaEventSynchronize(b)); float ms; cudaEventElapsedTime(&ms, a, b); ms /= 10;
    printf("%-18s grid %d (%d/SM): %.3f ms  %.1f TFLOPS  %.0f GB/s  bytes %s\n", name, grid, per, ms, flop / ms / 1e9, wbytes / ms / 1e6, same ? "IDENTICAL" : "DIFFER");
  };
  {
    size_t nb = (size_t)R * SLOTS * I * 2; std::vector<char> ref(nb);
    CK(cudaMemset(act, 0, nb)); fn_prompt4_gu_t2<<<144, 128>>>(X, D, SLOTS, Wgu, sc, D / 32, I / 32, d_items, d_cnt, d_mem, act, I, 7.0f, -1); CK(cudaMemcpy(ref.data(), act, nb, cudaMemcpyDeviceToHost));
    double fl = pairs * 2 * I * D * 2;
    runc(fn_coop_gu_t2_w4, "coop_gu_t2_w4", 2, 4, X, D, SLOTS, Wgu, D / 32, I / 32, act, I, fl, gu_n * 16.0, ref.data(), nb);
    runc(fn_coop_gu_t1_w4, "coop_gu_t1_w4", 1, 4, X, D, SLOTS, Wgu, D / 32, I / 32, act, I, fl, gu_n * 16.0, ref.data(), nb);
    runc(fn_coop_gu_t2_w5, "coop_gu_t2_w5", 2, 5, X, D, SLOTS, Wgu, D / 32, I / 32, act, I, fl, gu_n * 16.0, ref.data(), nb);
    // down reference (act now holds coop output == ref if identical; recompute act from ref)
    CK(cudaMemcpy(act, ref.data(), nb, cudaMemcpyHostToDevice));
    size_t nd = (size_t)R * SLOTS * D * 4; std::vector<char> rd(nd);
    CK(cudaMemset(out, 0, nd)); fn_prompt4_f32_t2<<<144, 128>>>(act, I, 0, Wdn, sc, I / 32, D / 32, d_items, d_cnt, d_mem, out, D, 7.0f, -1); CK(cudaMemcpy(rd.data(), out, nd, cudaMemcpyDeviceToHost));
    double fd = pairs * I * D * 2;
    runc(fn_coop_f32_t2_w8, "coop_f32_t2_w8", 2, 8, act, I, 0, Wdn, I / 32, D / 32, out, D, fd, dn_n * 16.0, rd.data(), nd);
    runc(fn_coop_f32_t1_w8, "coop_f32_t1_w8", 1, 8, act, I, 0, Wdn, I / 32, D / 32, out, D, fd, dn_n * 16.0, rd.data(), nd);
  }
  // bit check: t2 vs t2x (gate/up act bytes)
  size_t nb = (size_t)R * SLOTS * I * 2;
  std::vector<char> h1(nb), h2(nb);
  int units = n_items * (I / 32);
  CK(cudaMemset(act, 0, nb)); fn_prompt4_gu_t2<<<144, 128>>>(X, D, SLOTS, Wgu, sc, D / 32, I / 32, d_items, d_cnt, d_mem, act, I, 7.0f, -1); CK(cudaMemcpy(h1.data(), act, nb, cudaMemcpyDeviceToHost));
  CK(cudaMemset(act, 0, nb)); fn_prompt4_gu_t2x<<<144, 128>>>(X, D, SLOTS, Wgu, sc, D / 32, I / 32, d_items, d_cnt, d_mem, act, I, 7.0f, -1); CK(cudaMemcpy(h2.data(), act, nb, cudaMemcpyDeviceToHost));
  printf("gu t2 vs t2x bytes %s (units %d)\n", memcmp(h1.data(), h2.data(), nb) == 0 ? "IDENTICAL" : "DIFFER", units);
  CK(cudaMemset(act, 0, nb)); fn_prompt4_gu_t2x4<<<192, 128>>>(X, D, SLOTS, Wgu, sc, D / 32, I / 32, d_items, d_cnt, d_mem, act, I, 7.0f, -1); CK(cudaMemcpy(h2.data(), act, nb, cudaMemcpyDeviceToHost));
  printf("gu t2 vs t2x4 bytes %s\n", memcmp(h1.data(), h2.data(), nb) == 0 ? "IDENTICAL" : "DIFFER");
}
