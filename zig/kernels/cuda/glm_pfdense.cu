// GLM-5.3-Flash prompt chunks' dense EXL3 projections (REXL3/SAGE, mul1): DeepSeek V4.1's fused pfdense GEMM
// (exl3_block/pfdense.cuh: the trellis decoded inside a tensor-core GEMM, the Hadamard and svh in its epilogue),
// stored ("strips") layout only, at every dense width the packs hold, in the row tiles pfdense.heuristic picks from.
#include "exl3_block/pfdense.cuh"
#include "glm_pf_mla.cuh" // and the MLA layers' absorb / expand on tensor cores

#define GLM_PFD_CFG(K2, BM, WM, WN, KS, NST) \
    template __global__ void dsv41_pfd::pfd_kernel<K2, false, dsv41_pfd::Cfg<BM, WM, WN, KS, NST>>(const dsv41_pfd::Args);
#define GLM_PFD(K2)                   \
    GLM_PFD_CFG(K2, 128, 2, 4, 2, 3)  \
    GLM_PFD_CFG(K2, 64, 2, 2, 2, 4)   \
    GLM_PFD_CFG(K2, 32, 1, 4, 2, 4)

GLM_PFD(8)
GLM_PFD(10)
GLM_PFD(12)
GLM_PFD(14)
GLM_PFD(16)
