// microbench: MXFP8 decode, qmmf_kernel<2,16,...> vs tf_fn_mxgemv, real FP4-Local shapes. Bytes = n*k + n*k/32.
#include "fn_qmmf.cu"
#include <cstdio>
#include <vector>
#define CK(x) do { cudaError_t e = (x); if (e) { printf("err %s line %d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)
static int splitK(int n, int k) { int t = (n + 63) / 64, g = k / 64, sk = 1; while (sk < 8 && t * sk < 192 && g % (sk * 2) == 0 && g / (sk * 2) >= 8) sk *= 2; return sk; }
template <typename F> float timeit(F f) { cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b); for (int i = 0; i < 5; ++i) f(); cudaEventRecord(a); for (int i = 0; i < 50; ++i) f(); cudaEventRecord(b); CK(cudaEventSynchronize(b)); float ms; cudaEventElapsedTime(&ms, a, b); return ms / 50; }
int main() {
    int shapes[][2] = {{16480, 2560}, {13312, 2560}, {2560, 6144}, {1280, 2560}, {2560, 640}, {640, 2560}};
    int ms_[] = {1, 2, 4, 8};
    // a buffer bigger than L2 (rotate copies so each call reads from DRAM)
    for (auto& s : shapes) {
        int n = s[0], k = s[1], npad = (n + 127) / 128 * 128, kg = k / 64;
        size_t wb = (size_t)npad * k, sb = (size_t)npad * kg * 2;
        const int COP = 8;
        unsigned char *w, *bs; __nv_bfloat16* x; void* out; float* part;
        CK(cudaMalloc(&w, wb * COP)); CK(cudaMalloc(&bs, sb * COP)); CK(cudaMalloc(&x, 16 * k * 2)); CK(cudaMalloc(&out, 16 * (size_t)npad * 4)); CK(cudaMalloc(&part, 64 << 20));
        std::vector<unsigned char> h(wb); for (size_t i = 0; i < wb; ++i) h[i] = (i * 2654435761u >> 13) & 0x77;
        for (int c = 0; c < COP; ++c) CK(cudaMemcpy(w + c * wb, h.data(), wb, cudaMemcpyHostToDevice));
        CK(cudaMemset(bs, 120, sb * COP)); std::vector<__nv_bfloat16> hx(16 * k); for (auto& v : hx) v = __float2bfloat16(0.01f); CK(cudaMemcpy(x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
        int sk = splitK(n, k);
        auto qk = sk > 1 ? tf_fn_qmmf::qmmf_kernel<2, 16, 64, 1, 4, 4, false, true, false> : tf_fn_qmmf::qmmf_kernel<2, 16, 64, 1, 4, 4, false, false, false>;
        int smem = 25600; CK(cudaFuncSetAttribute(qk, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        for (int m : ms_) {
            int it = 0;
            auto runq = [&]() {
                int c = (it++) % COP;
                cudaLaunchConfig_t cfg = {}; cudaLaunchAttribute at[1];
                cfg.gridDim = dim3(((m + 15) / 16) * ((n + 63) / 64), 1, sk); cfg.blockDim = 128; cfg.dynamicSmemBytes = smem;
                if (sk > 1) { at[0].id = cudaLaunchAttributeClusterDimension; at[0].val.clusterDim.x = 1; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = sk; cfg.attrs = at; cfg.numAttrs = 1; }
                CK(cudaLaunchKernelEx(&cfg, qk, (const __nv_bfloat16*)x, (const unsigned char*)(w + c * wb), (const uint8_t*)(bs + c * sb), 1.0f, out, part, m, n, k, sk, npad, k, 1));
            };
            auto rung = [&]() {
                int c = (it++) % COP;
                if (m <= 2) tf_fn_mxgemv_r2_b16<<<(n + 7) / 8, 256>>>(x, w + c * wb, bs + c * sb, out, m, n, k, k, n);
                else if (m <= 4) tf_fn_mxgemv_r4_b16<<<(n + 7) / 8, 256>>>(x, w + c * wb, bs + c * sb, out, m, n, k, k, n);
                else tf_fn_mxgemv_r8_b16<<<(n + 7) / 8, 256>>>(x, w + c * wb, bs + c * sb, out, m, n, k, k, n);
            };
            float tq = timeit(runq), tg = timeit(rung);
            double gb = (wb + sb) / 1e9;
            printf("n=%5d k=%4d sk=%d m=%d  qmmf %7.1f us %6.1f GB/s | gemv %7.1f us %6.1f GB/s\n", n, k, sk, m, tq * 1e3, gb / (tq * 1e-3), tg * 1e3, gb / (tg * 1e-3));
        }
        cudaFree(w); cudaFree(bs); cudaFree(x); cudaFree(out); cudaFree(part);
    }
}
