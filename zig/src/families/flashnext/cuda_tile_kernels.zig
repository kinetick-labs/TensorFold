//! The prompt tiles' modules (flashnext_fact, _hcf and _fexp): their instances' names, shared bytes and loading.
const cuda = @import("cuda");

/// Prompt rows on qmm.cu's arithmetic (flashnext_fact.cu): 64 x 128 tiles, two groups a stage, the lane tile's bits.
pub const cubin_fact: [:0]const u8 = "_ZN4fact11fact_kernelILi64ELi128ELi2ELi4ELi3ELi2ELb0ELi2ELb0ELb1ELb0EEEvPK13__nv_bfloat16PKfPKjS3_S3_PvPfiiiiiii";
/// The hyper-connection read-out's prompt tiles (flashnext_hcf.cu), fn_hc_down's and fn_hc_upmix's bits.
pub const cubin_hcf_down: [:0]const u8 = "_ZN3hcf10hcf_kernelILi32ELi192ELi2ELi4ELi4ELi2ELb1ELb1ELb0ELi2EEEvPK13__nv_bfloat16PKfPK5uint4S3_S3_PviiiiiiS5_S5_fiii";
pub const cubin_hcf_up: [:0]const u8 = "_ZN3hcf10hcf_kernelILi64ELi128ELi2ELi4ELi3ELi2ELb0ELb0ELb1ELi2EEEvPK13__nv_bfloat16PKfPK5uint4S3_S3_PviiiiiiS5_S5_fiii";
pub const hcf_down_smem: u32 = 4 * (2 * 32 * 64 + 2 * 192 * 16 + 2 * 2 * 192 * 2 + 2 * 32 * 4 + 2 * 32 * 4) + 32 * 16;
pub const hcf_up_smem: u32 = 3 * (2 * 64 * 64 + 2 * 128 * 16 + 2 * 2 * 128 * 2 + 2 * 64 * 4 + 2 * 32 * 4) + 64 * 16;
/// Prompt rows' experts on factored 64-pair tiles (flashnext_fexp.cu), expert_rows' bits: up with SwiGLU, down to fp32.
pub const cubin_fexp_up: [:0]const u8 = "_ZN15tf_fact_experts11fact_kernelILi2ELi2ELi1ELi4ELi2ELi3EEEvPK13__nv_bfloat16iiPK5uint4iiPKiS8_S8_Pvif";
pub const cubin_fexp_down: [:0]const u8 = "_ZN15tf_fact_experts11fact_kernelILi1ELi0ELi2ELi2ELi4ELi3EEEvPK13__nv_bfloat16iiPK5uint4iiPKiS8_S8_Pvif";
pub const fexp_smem: u32 = 41472;
pub const fexp_tile: usize = 64;
pub const cubin_fact_fold: [:0]const u8 = "_ZN4fact11fact_kernelILi64ELi128ELi2ELi4ELi3ELi2ELb0ELi2ELb1ELb1ELb1EEEvPK13__nv_bfloat16PKfPKjS3_S3_PvPfiiiiiii";
pub const fact_smem: u32 = 3 * (2 * 64 * 64 + 2 * 128 * 16 + 2 * 2 * 128 * 2 + 2 * 64 * 4);

/// The native (bf16) read-out's fused down and up+mix tiles (flashnext_hcf.cu, hcn): CTAs of 16 * mt rows, by mt.
pub const hcn_down_names = [_][:0]const u8{ "fn_hcn_down2", "fn_hcn_down3", "fn_hcn_down4" };
pub const hcn_down_mt = [_]usize{ 2, 3, 4 };
pub const hcn_updbg_names = [_][:0]const u8{ "fn_hcn_up3_k5_gemm", "fn_hcn_up3_k5_epi" };
pub const hcn_dv2_names = [_][:0]const u8{ "fn_hcn_dv2_2", "fn_hcn_dv2_3", "fn_hcn_dv2_4", "fn_hcn_dv2_4_nx" };
pub const hcn_dv2_mt = [_]usize{ 2, 3, 4, 4 };
pub const hcn_up_names = [_][:0]const u8{ "fn_hcn_up2_k5", "fn_hcn_up3_k5", "fn_hcn_up4_k5" };
pub const hcn_up_mt = [_]usize{ 2, 3, 4 };
/// hcn::DnShape's and hcn::UpShape's SMEM (K = 320).
pub fn hcnDownSmem(mt: usize) u32 {
    return @intCast(3 * 16 * mt * 128 + 16 * mt * 8 * 4);
}
pub fn hcnUpSmem(mt: usize) u32 {
    return @intCast(16 * mt * (640 + 256 + 16));
}

pub const TileKernels = struct {
    fact_mod: cuda.Module,
    fact: cuda.Function,
    fact_fold: cuda.Function,
    hcf_mod: cuda.Module,
    hcf_down: cuda.Function,
    hcf_up: cuda.Function,
    hcn_down: [hcn_down_names.len]cuda.Function,
    hcn_up: [hcn_up_names.len]cuda.Function,
    hcn_dv2: [hcn_dv2_names.len]cuda.Function,
    hcn_peak: cuda.Function,
    hcn_updbg: [hcn_updbg_names.len]cuda.Function,
    fexp_mod: cuda.Module,
    fexp_up: cuda.Function,
    fexp_down: cuda.Function,

    pub fn load(d: *const cuda.Driver) !TileKernels {
        var k: TileKernels = undefined;
        k.fact_mod = try cuda.Module.load(d, cuda.kernels.flashnext_fact);
        errdefer k.fact_mod.unload();
        k.fact = try k.fact_mod.function(cubin_fact);
        try k.fact.allowDynamicShared(fact_smem);
        k.fact_fold = try k.fact_mod.function(cubin_fact_fold);
        try k.fact_fold.allowDynamicShared(fact_smem);
        k.fexp_mod = try cuda.Module.load(d, cuda.kernels.flashnext_fexp);
        errdefer k.fexp_mod.unload();
        k.fexp_up = try k.fexp_mod.function(cubin_fexp_up);
        try k.fexp_up.allowDynamicShared(fexp_smem);
        k.fexp_down = try k.fexp_mod.function(cubin_fexp_down);
        try k.fexp_down.allowDynamicShared(fexp_smem);
        k.hcf_mod = try cuda.Module.load(d, cuda.kernels.flashnext_hcf);
        errdefer k.hcf_mod.unload();
        k.hcf_down = try k.hcf_mod.function(cubin_hcf_down);
        try k.hcf_down.allowDynamicShared(hcf_down_smem);
        k.hcf_up = try k.hcf_mod.function(cubin_hcf_up);
        try k.hcf_up.allowDynamicShared(hcf_up_smem);
        for (&k.hcn_down, hcn_down_names, hcn_down_mt) |*f, name, mt| {
            f.* = try k.hcf_mod.function(name);
            try f.allowDynamicShared(hcnDownSmem(mt));
        }
        for (&k.hcn_up, hcn_up_names, hcn_up_mt) |*f, name, mt| {
            f.* = try k.hcf_mod.function(name);
            try f.allowDynamicShared(hcnUpSmem(mt));
        }
        for (&k.hcn_dv2, hcn_dv2_names, hcn_dv2_mt) |*f, name, mt| {
            f.* = try k.hcf_mod.function(name);
            try f.allowDynamicShared(hcnDownSmem(mt));
        }
        k.hcn_peak = try k.hcf_mod.function("fn_hcn_peak");
        for (&k.hcn_updbg, hcn_updbg_names) |*f, name| {
            f.* = try k.hcf_mod.function(name);
            try f.allowDynamicShared(hcnUpSmem(3));
        }
        return k;
    }

    pub fn unload(k: *TileKernels) void {
        k.hcf_mod.unload();
        k.fexp_mod.unload();
        k.fact_mod.unload();
    }
};
