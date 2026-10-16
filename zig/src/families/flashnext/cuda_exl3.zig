//! EXL3 (ExLlamaV3) packs: the host half of zig/kernels/cuda/fn_exl3.cu -- the plan, the mangled names of the
//! vendored kernels, and the launches they take.
//!
//! Reference: upstream/python-0.6 src/tensorfold/cuda/exl3/linear.py (this file's `plan`, `strides` and counter
//! sizing are ports of it, line for line) and experts.py. The kernels' signatures are fn_exl3.cu's; the launch
//! shapes are the reference's own launchers, which survive there as the instance drivers.
//!
//! The per-tensor width is not a checkpoint field: it is the trellis shape (`format.py:bits_of`, last_dim / 16),
//! and this pack mixes three of them (4, 5 and 6 bits -> k2 8, 10, 12). So every table here is keyed by the width
//! a caller derived, never by the header's mean.

const std = @import("std");
const cuda = @import("cuda");
const cfgs = @import("cuda_config.zig");

/// Codebook ids the kernels take as an int (decode.cuh:10, linear.py:13).
pub fn codebookId(name: []const u8) ?u8 {
    if (std.mem.eql(u8, name, "3inst")) return 0;
    if (std.mem.eql(u8, name, "mcg")) return 1;
    if (std.mem.eql(u8, name, "mul1")) return 2;
    return null;
}

/// k2, the width in half-bits: 2 * bits (linear.py:26), or null when the width is outside 1..8 or not a half-bit
/// (format.check_bits). The header's mean -- 4.05 for this pack -- is deliberately not a width.
pub fn k2Of(bits: f64) ?u32 {
    if (!(bits >= 1 and bits <= 8)) return null;
    const half = 2 * bits;
    if (half != @round(half)) return null;
    return @intFromFloat(half);
}

// ---------------------------------------------------------------------------------------------------------------
// linear.py:30's plan: (K splits, warps a program) for a K x N layer, from the shape alone, so a layer keeps one
// reduction for every row count.
// ---------------------------------------------------------------------------------------------------------------

pub const Split = struct { sk: usize, wk: usize };

pub fn plan(k: usize, n: usize) Split {
    return planBlocks(k, n, 192, 8);
}

pub fn planBlocks(k: usize, n: usize, blocks: usize, min_tiles: usize) Split {
    const kt = k / 16;
    const nb = n / 128;
    var sk: usize = 1;
    var wk: usize = if (nb >= 64) 8 else 4;
    if (nb < 64) { // a narrow layer needs K splits to fill the SMs at all
        while (nb * sk < blocks and sk < 64) sk *= 2;
    }
    while ((wk > 4 or sk > 1) and (kt % (sk * wk) != 0 or kt / (sk * wk) < min_tiles)) {
        if (wk > 4) {
            wk = 4;
        } else if (sk > 1) {
            sk /= 2;
        } else break;
    }
    return .{ .sk = sk, .wk = wk };
}

/// How the trellis words are laid out: linear.py:49's `strips` copy, or the checkpoint's own `stored` order.
pub const Layout = enum { strips, stored };

/// (words between k tiles, words between 128-column blocks) in the words buffer (linear.py:142).
pub fn strides(layout: Layout, k: usize, n: usize, k2: u32) [2]usize {
    const tw: usize = 4 * k2;
    return switch (layout) {
        .strips => .{ 8 * tw, (k / 16) * 8 * tw },
        .stored => .{ (n / 16) * tw, 8 * tw },
    };
}

/// int32 counters a call needs: 8 * N/128, left zero, one per (pass, column block) (linear.py:161).
pub fn counterCount(n: usize) usize {
    return 8 * (n / 128);
}

/// Bits a trellis holds a value at, from its shape alone: the last dimension over 16 (format.py:86, `bits_of`).
/// This, never the header's scalar, is a layer's width -- the pack mixes 4, 5 and 6 bits under one 4.05 header.
/// `cfgs.bitsOf` is the one implementation of the rule (it also refuses a width ExLlamaV3 never writes).
pub fn bitsOf(shape: []const i64) ?f64 {
    return cfgs.bitsOf(shape);
}

/// The `strips` copy the dense kernels read (linear.py:48): `words` [K/16, N/16, W] -> [N/128, K/16, 8, W], each
/// 128-column block's eight tiles in k order and each tile's words together. `dst` and `src` may not overlap.
pub fn strips(dst: []u32, src: []const u32, kt: usize, nt: usize, w: usize) !void {
    const words = kt * nt * w;
    if (dst.len < words or src.len < words) return error.Short;
    if (nt % 8 != 0) return error.UnexpectedTensor;
    for (0..kt) |t| {
        for (0..nt) |n| {
            const at = ((n / 8) * kt + t) * 8 * w + (n % 8) * w;
            @memcpy(dst[at..][0..w], src[(t * nt + n) * w ..][0..w]);
        }
    }
}

/// The tile setting the grouped GEMM takes for a K -> N projection (experts.py:134): GLM's own when it divides,
/// then the narrower fallbacks, so rows stay independent and the arithmetic order matches GLM bit for bit.
pub const Tile = struct { nt: usize, w: usize, sk: usize, pf: usize };

pub const glm_gateup: Tile = .{ .nt = 8, .w = 4, .sk = 4, .pf = 1 };
pub const glm_down: Tile = .{ .nt = 8, .w = 4, .sk = 1, .pf = 1 };

pub fn defaultTile(k: usize, n: usize, gateup: bool) !Tile {
    const cands = [5]Tile{
        if (gateup) glm_gateup else glm_down,
        .{ .nt = 8, .w = 4, .sk = 2, .pf = 1 },
        .{ .nt = 8, .w = 4, .sk = 1, .pf = 1 },
        .{ .nt = 4, .w = 4, .sk = 2, .pf = 2 },
        .{ .nt = 4, .w = 4, .sk = 1, .pf = 2 },
    };
    for (cands) |t| {
        if (k % (16 * t.sk * t.w) == 0 and n % (16 * t.nt) == 0) return t;
    }
    return error.NoTileSetting;
}

/// Trellis bytes one expert's gate, up and down read: (D*I/256) * (k2g + k2u + k2d) * 16 (experts.py:120).
pub fn expertBytes(dims: usize, width: usize, k2_g: u32, k2_u: u32, k2_d: u32) u64 {
    return @as(u64, @intCast(dims * width / 256)) * (k2_g + k2_u + k2_d) * 16;
}

/// One rank's routed experts as the kernels take them: per-expert trellis pointers and widths, the six stacked
/// scale tables, and the tile settings (experts.py:83's `Exl3RoutedExperts`). The loader fills addresses; this is
/// the shape both sides agree on.
pub const Experts = struct {
    gate_ptr: u64 = 0,
    up_ptr: u64 = 0,
    down_ptr: u64 = 0,
    gate_k2: u64 = 0,
    up_k2: u64 = 0,
    down_k2: u64 = 0,
    suh_g: u64 = 0,
    suh_u: u64 = 0,
    svh_g: u64 = 0,
    svh_u: u64 = 0,
    suh_d: u64 = 0,
    svh_d: u64 = 0,
    trellis_bytes: u64 = 0,
    count: u32 = 0,
    dims: u32 = 0,
    width: u32 = 0,
    cb: u8 = 2,
    k2_gu: [2]u32 = .{ 0, 0 },
    k2_d: [2]u32 = .{ 0, 0 },
    tile_gu: Tile = glm_gateup,
    tile_d: Tile = glm_down,
};

/// The k2 envelope a set of per-expert widths covers, as `prepare` records it (min, max).
pub fn envelope(k2s_: []const u32) [2]u32 {
    if (k2s_.len == 0) return .{ 0, 0 };
    var lo = k2s_[0];
    var hi = k2s_[0];
    for (k2s_) |k| {
        lo = @min(lo, k);
        hi = @max(hi, k);
    }
    return .{ lo, hi };
}

// ---------------------------------------------------------------------------------------------------------------
// Grids, blocks and dynamic shared memory, one helper a kernel.
// ---------------------------------------------------------------------------------------------------------------

/// rot_in_kernel (linear.cu:112): grid ((K/128 + 3) / 4, M), block 128.
pub fn rotInGrid(k: usize, rows: usize) [2]usize {
    return .{ (k / 128 + 3) / 4, rows };
}

/// linear_kernel (linear.cu:128): grid (N/128, SK), block WK*32; smem is WK*min(M,8)*128*4, always under the 48 KiB
/// a kernel gets without opting in (8 * 8 * 128 * 4 = 32 KiB).
pub fn linearGrid(n: usize, sk: usize) [2]usize {
    return .{ n / 128, sk };
}

pub fn linearBlock(wk: usize) usize {
    return wk * 32;
}

pub fn linearShared(wk: usize, rows: usize) usize {
    return wk * @min(rows, 8) * 128 * @sizeOf(f32);
}

/// unpack_kernel (linear.cu:269): grid (N/16, K/16), block 32; one warp a tile.
pub fn unpackGrid(k: usize, n: usize) [2]usize {
    return .{ n / 16, k / 16 };
}

/// group_kernel (experts.cu:19): 1 block of GROUP_THREADS, R*slots*4 dynamic bytes (lifted past 48 KiB when large).
pub const group_threads: usize = 1024;

pub fn groupShared(rows: usize, slots: usize) usize {
    return rows * slots * @sizeOf(c_int);
}

/// gateup_epilogue_kernel (experts.cu:116) and down_epilogue_kernel (:161): grid (rows*slots, width/128), block 32.
pub fn epilogueGrid(rows: usize, slots: usize, width: usize) [2]usize {
    return .{ rows * slots, width / 128 };
}

/// combine_kernel (experts.cu:183): grid (rows, ceil(D/256)), block 256.
pub fn combineGrid(rows: usize, d: usize) [2]usize {
    return .{ rows, (d + 255) / 256 };
}

/// down_combine_kernel (experts.cu:194): grid (rows, D/128), block 32*slots.
pub fn downCombineGrid(rows: usize, d: usize) [2]usize {
    return .{ rows, d / 128 };
}

pub fn downCombineBlock(slots: usize) usize {
    return 32 * slots;
}

/// grouped_kernel (experts_grouped.cuh:191): grid (experts, N/(16*nt), matrices*SK*mtiles), block W*32.
pub fn groupedGrid(experts: usize, n: usize, nt: usize, mats: usize, sk: usize, mtiles: usize) [3]usize {
    return .{ experts, n / (16 * nt), mats * sk * mtiles };
}

/// dequant_kernel (experts_grouped.cuh:288): grid (K/16, N/16), block 32.
pub fn dequantGrid(k: usize, n: usize) [2]usize {
    return .{ k / 16, n / 16 };
}

// ---------------------------------------------------------------------------------------------------------------
// The kernels. Widths are the ones this pack needs (k2 8, 10, 12 -- bits 4, 5 and 6, codebook mul1); the fatbin
// carries every (k2, codebook) pair the reference compiles, so a pack at other widths only needs more rows here.
// Mangled names as cuobjdump -symbols lists them for fn_exl3.fatbin (sm_121).
// ---------------------------------------------------------------------------------------------------------------

pub const k2s = [3]u32{ 8, 10, 12 };
pub const wks = [3]usize{ 2, 4, 8 };

/// (tiles a warp, warps a program, tiles in flight) the reference's grouped_launch picks from (experts_grouped.cuh:336).
pub const tile_settings = [3]struct { nt: usize, w: usize, pf: usize }{
    .{ .nt = 8, .w = 4, .pf = 1 },
    .{ .nt = 8, .w = 4, .pf = 2 },
    .{ .nt = 4, .w = 4, .pf = 2 },
};

/// The k2 ranges a compiled instance covers: (8, 8) exactly, else (2, 10) when it fits, else (2, 16).
pub const ranges = [3]struct { lo: u32, hi: u32 }{
    .{ .lo = 8, .hi = 8 },
    .{ .lo = 2, .hi = 10 },
    .{ .lo = 2, .hi = 16 },
};

pub fn k2Index(k2: u32) ?usize {
    return std.mem.indexOfScalar(u32, &k2s, k2);
}

pub fn wkIndex(wk: usize) ?usize {
    return std.mem.indexOfScalar(usize, &wks, wk);
}

pub fn tileIndex(nt: usize, w: usize, pf: usize) ?usize {
    for (tile_settings, 0..) |t, i| {
        if (t.nt == nt and t.w == w and t.pf == pf) return i;
    }
    return null;
}

/// The range instance a (k2 min, k2 max) pair takes, as the reference's TF_RANGES orders it.
pub fn rangeIndex(lo: u32, hi: u32) usize {
    if (lo == 8 and hi == 8) return 0;
    if (lo >= 2 and hi <= 10) return 1;
    return 2;
}

pub const sym = struct {
    /// tf_exl3_lin (linear.cu): [k2: 8, 10, 12][WK: 2, 4, 8]
    pub const linear = [3][3][:0]const u8{
        .{
            "_ZN10tf_fn_exl311tf_exl3_lin13linear_kernelILi8ELi2ELi2EEEvPK6__halfPKjxxS4_S4_PviPfPiiiii",
            "_ZN10tf_fn_exl311tf_exl3_lin13linear_kernelILi8ELi2ELi4EEEvPK6__halfPKjxxS4_S4_PviPfPiiiii",
            "_ZN10tf_fn_exl311tf_exl3_lin13linear_kernelILi8ELi2ELi8EEEvPK6__halfPKjxxS4_S4_PviPfPiiiii",
        },
        .{
            "_ZN10tf_fn_exl311tf_exl3_lin13linear_kernelILi10ELi2ELi2EEEvPK6__halfPKjxxS4_S4_PviPfPiiiii",
            "_ZN10tf_fn_exl311tf_exl3_lin13linear_kernelILi10ELi2ELi4EEEvPK6__halfPKjxxS4_S4_PviPfPiiiii",
            "_ZN10tf_fn_exl311tf_exl3_lin13linear_kernelILi10ELi2ELi8EEEvPK6__halfPKjxxS4_S4_PviPfPiiiii",
        },
        .{
            "_ZN10tf_fn_exl311tf_exl3_lin13linear_kernelILi12ELi2ELi2EEEvPK6__halfPKjxxS4_S4_PviPfPiiiii",
            "_ZN10tf_fn_exl311tf_exl3_lin13linear_kernelILi12ELi2ELi4EEEvPK6__halfPKjxxS4_S4_PviPfPiiiii",
            "_ZN10tf_fn_exl311tf_exl3_lin13linear_kernelILi12ELi2ELi8EEEvPK6__halfPKjxxS4_S4_PviPfPiiiii",
        },
    };
    /// tf_exl3_lin: the width's dequantize-to-dense
    pub const unpack = [3][:0]const u8{
        "_ZN10tf_fn_exl311tf_exl3_lin13unpack_kernelILi8ELi2EEEvPKjP6__halfill",
        "_ZN10tf_fn_exl311tf_exl3_lin13unpack_kernelILi10ELi2EEEvPKjP6__halfill",
        "_ZN10tf_fn_exl311tf_exl3_lin13unpack_kernelILi12ELi2EEEvPKjP6__halfill",
    };
    /// tf_exl3_lin: xh = fp16((x * suh) @ H), any input dtype (the kernel takes the dtype as an int)
    pub const rot_in = "_ZN10tf_fn_exl311tf_exl3_lin13rot_in_kernelEPKviPK6__halfPS3_i";
    /// tf_exl3_exp (experts.cu): [0: bf16 input, 1: fp16 input]
    pub const rot_in_experts = [2][:0]const u8{
        "_ZN10tf_fn_exl311tf_exl3_exp13rot_in_kernelI13__nv_bfloat16EEvPKT_iPKiPK6__halfSA_PS8_SB_iii",
        "_ZN10tf_fn_exl311tf_exl3_exp13rot_in_kernelI6__halfEEvPKT_iPKiPKS2_S9_PS2_SA_iii",
    };
    pub const group = "_ZN10tf_fn_exl311tf_exl3_exp12group_kernelEPKiPiS3_S3_iiii";
    pub const gateup_epilogue = "_ZN10tf_fn_exl311tf_exl3_exp22gateup_epilogue_kernelEPKfPKiPK6__halfS7_S7_PS5_iiiifi";
    pub const down_epilogue = "_ZN10tf_fn_exl311tf_exl3_exp20down_epilogue_kernelEPKfPKiPK6__halfPfiiii";
    pub const combine = "_ZN10tf_fn_exl311tf_exl3_exp14combine_kernelEPKfS2_Pfii";
    pub const down_combine = "_ZN10tf_fn_exl311tf_exl3_exp19down_combine_kernelEPKfPKiPK6__halfPfS2_S8_iiiii";
    /// tf_exl3x (experts_grouped.cuh): [tile setting][k2 range]
    pub const grouped = [3][3][:0]const u8{
        .{
            "_ZN10tf_fn_exl38tf_exl3x14grouped_kernelILi2ELi8ELi4ELi1ELi8ELi8EEEvPK6__halfS4_PKlS6_PKiS8_S8_S8_S8_Pfiiiiii",
            "_ZN10tf_fn_exl38tf_exl3x14grouped_kernelILi2ELi8ELi4ELi1ELi2ELi10EEEvPK6__halfS4_PKlS6_PKiS8_S8_S8_S8_Pfiiiiii",
            "_ZN10tf_fn_exl38tf_exl3x14grouped_kernelILi2ELi8ELi4ELi1ELi2ELi16EEEvPK6__halfS4_PKlS6_PKiS8_S8_S8_S8_Pfiiiiii",
        },
        .{
            "_ZN10tf_fn_exl38tf_exl3x14grouped_kernelILi2ELi8ELi4ELi2ELi8ELi8EEEvPK6__halfS4_PKlS6_PKiS8_S8_S8_S8_Pfiiiiii",
            "_ZN10tf_fn_exl38tf_exl3x14grouped_kernelILi2ELi8ELi4ELi2ELi2ELi10EEEvPK6__halfS4_PKlS6_PKiS8_S8_S8_S8_Pfiiiiii",
            "_ZN10tf_fn_exl38tf_exl3x14grouped_kernelILi2ELi8ELi4ELi2ELi2ELi16EEEvPK6__halfS4_PKlS6_PKiS8_S8_S8_S8_Pfiiiiii",
        },
        .{
            "_ZN10tf_fn_exl38tf_exl3x14grouped_kernelILi2ELi4ELi4ELi2ELi8ELi8EEEvPK6__halfS4_PKlS6_PKiS8_S8_S8_S8_Pfiiiiii",
            "_ZN10tf_fn_exl38tf_exl3x14grouped_kernelILi2ELi4ELi4ELi2ELi2ELi10EEEvPK6__halfS4_PKlS6_PKiS8_S8_S8_S8_Pfiiiiii",
            "_ZN10tf_fn_exl38tf_exl3x14grouped_kernelILi2ELi4ELi4ELi2ELi2ELi16EEEvPK6__halfS4_PKlS6_PKiS8_S8_S8_S8_Pfiiiiii",
        },
    };
    /// tf_exl3x: the width's dequantize-to-dense (the grouped path's own decoder)
    pub const dequant = [3][:0]const u8{
        "_ZN10tf_fn_exl38tf_exl3x14dequant_kernelILi2ELi8EEEvPKjP6__halfii",
        "_ZN10tf_fn_exl38tf_exl3x14dequant_kernelILi2ELi10EEEvPKjP6__halfii",
        "_ZN10tf_fn_exl38tf_exl3x14dequant_kernelILi2ELi12EEEvPKjP6__halfii",
    };
};

/// The fn_exl3 module with every entry point this pack's widths need resolved.
pub const Kernels = struct {
    module: cuda.Module,
    linear: [3][3]cuda.Function,
    unpack: [3]cuda.Function,
    rot_in: cuda.Function,
    rot_in_experts: [2]cuda.Function,
    group: cuda.Function,
    gateup_epilogue: cuda.Function,
    down_epilogue: cuda.Function,
    combine: cuda.Function,
    down_combine: cuda.Function,
    grouped: [3][3]cuda.Function,
    dequant: [3]cuda.Function,
    /// dynamic shared memory group_kernel was opted into (the device's ceiling less its static use)
    group_smem_max: u32,

    pub fn load(ctx: *const cuda.Context) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var k: Kernels = undefined;
        k.module = try cuda.Module.load(ctx.d, cuda.kernels.fn_exl3);
        errdefer k.module.unload();
        for (sym.linear, 0..) |row, i| for (row, 0..) |s, j| {
            k.linear[i][j] = try k.module.function(s);
        };
        for (sym.unpack, 0..) |s, i| k.unpack[i] = try k.module.function(s);
        k.rot_in = try k.module.function(sym.rot_in);
        for (sym.rot_in_experts, 0..) |s, i| k.rot_in_experts[i] = try k.module.function(s);
        k.group = try k.module.function(sym.group);
        k.gateup_epilogue = try k.module.function(sym.gateup_epilogue);
        k.down_epilogue = try k.module.function(sym.down_epilogue);
        k.combine = try k.module.function(sym.combine);
        k.down_combine = try k.module.function(sym.down_combine);
        for (sym.grouped, 0..) |row, i| for (row, 0..) |s, j| {
            k.grouped[i][j] = try k.module.function(s);
        };
        for (sym.dequant, 0..) |s, i| k.dequant[i] = try k.module.function(s);
        // group_kernel's grouping buffer: R*slots ints, which passes 48 KiB once a call has enough routed slots.
        // experts.cu:296 lifts the limit against the device's own opt-in ceiling less the kernel's static use.
        const ceiling: usize = @intCast(try ctx.attribute(.max_shared_memory_per_block_optin));
        const statics: usize = @intCast(try k.group.attribute(.shared_size_bytes));
        k.group_smem_max = @intCast(ceiling -| statics);
        try k.group.allowDynamicShared(k.group_smem_max);
        return k;
    }

    pub fn deinit(k: *Kernels) void {
        k.module.unload();
    }
};

/// The dynamic shared memory limit group_kernel is opted into is read from the device at load (the reference's own
/// ceiling less the kernel's static use); a launch whose R*slots*4 exceeds it is refused, not launched short.
fn int(x: usize) c_int {
    return @intCast(x);
}

fn dim3(v: [2]usize) cuda.launch.Dim3 {
    return .{ .x = @intCast(v[0]), .y = @intCast(v[1]) };
}

/// rot_in: xh fp16 [M, K] = fp16(((x * suh) @ H) / sqrt(128)); x may be fp16 (0), bf16 (1) or fp32 (2).
pub fn rotIn(k: *const Kernels, x: u64, x_dtype: c_int, suh: u64, xh: u64, rows: usize, kk: usize, s: cuda.Stream) !void {
    const g = rotInGrid(kk, rows);
    var a: cuda.Args = .{};
    a.add(x);
    a.add(x_dtype);
    a.add(suh);
    a.add(xh);
    a.add(int(kk));
    try cuda.launch.launch(k.rot_in, .{ .grid = dim3(g), .block = .{ .x = 128 } }, s, &a);
}

/// linear: y [M, N] = xh [M, K] @ W_q + bias, from the rotated input the trellis holds. Rows 1..128.
pub fn linear(k: *const Kernels, k2: u32, cb: u8, xh: u64, words: u64, stride_k: i64, stride_nb: i64, svh: u64, bias: u64, y: u64, y_dtype: c_int, z: u64, counters: u64, rows: usize, kk: usize, n: usize, split: Split, s: cuda.Stream) !void {
    if (cb != 2) return error.Codebook; // the (k2, codebook) rows this fatbin binds are mul1's
    const i = k2Index(k2) orelse return error.Width;
    if (wkIndex(split.wk) == null) return error.Split;
    if (kk % (16 * split.sk * split.wk) != 0) return error.Split;
    const g = linearGrid(n, split.sk);
    const sm = linearShared(split.wk, rows);
    var a: cuda.Args = .{};
    a.add(xh);
    a.add(words);
    a.add(stride_k);
    a.add(stride_nb);
    a.add(svh);
    a.add(bias);
    a.add(y);
    a.add(y_dtype);
    a.add(z);
    a.add(counters);
    a.add(int(rows));
    a.add(int(kk));
    a.add(int(n));
    a.add(int(split.sk));
    const wk = split.wk;
    const j = wkIndex(wk).?;
    try cuda.launch.launch(k.linear[i][j], .{ .grid = dim3(g), .block = .{ .x = @intCast(wk * 32) }, .shared = @intCast(sm) }, s, &a);
}

/// unpack: W_q fp16 [K, N] from the trellis words, either layout.
pub fn unpack(k: *const Kernels, k2: u32, cb: u8, words: u64, out: u64, kk: usize, n: usize, stride_k: i64, stride_nb: i64, s: cuda.Stream) !void {
    if (cb != 2) return error.Codebook;
    const i = k2Index(k2) orelse return error.Width;
    const g = unpackGrid(kk, n);
    var a: cuda.Args = .{};
    a.add(words);
    a.add(out);
    a.add(int(n));
    a.add(stride_k);
    a.add(stride_nb);
    try cuda.launch.launch(k.unpack[i], .{ .grid = dim3(g), .block = .{ .x = 32 } }, s, &a);
}

/// The experts' grouping: distinct experts in id order, members row*32 + slot, -1 after the last (experts.cu:19).
pub fn group(k: *const Kernels, pick: u64, uids: u64, ucount: u64, members: u64, rows: usize, slots: usize, experts: usize, maxm: usize, s: cuda.Stream) !void {
    const sm = groupShared(rows, slots);
    if (sm > k.group_smem_max) return error.GroupingSharedMemory;
    if (slots > 32) return error.Slots;
    var a: cuda.Args = .{};
    a.add(pick);
    a.add(uids);
    a.add(ucount);
    a.add(members);
    a.add(int(rows));
    a.add(int(slots));
    a.add(int(experts));
    a.add(int(maxm));
    try cuda.launch.launch(k.group, .{ .grid = .{ .x = 1 }, .block = .{ .x = group_threads }, .shared = @intCast(sm) }, s, &a);
}

/// The experts' rotated input: gate and up of every routed slot, one launch a matrix (blockIdx.z).
pub fn rotInExperts(k: *const Kernels, dtype_bf16: bool, x: u64, x_stride: c_int, pick: u64, suh0: u64, suh1: u64, out0: u64, out1: u64, rows: usize, kk: usize, slots: usize, experts: usize, s: cuda.Stream) !void {
    const g = [3]usize{ rows * slots, kk / 128, 2 };
    var a: cuda.Args = .{};
    a.add(x);
    a.add(x_stride);
    a.add(pick);
    a.add(suh0);
    a.add(suh1);
    a.add(out0);
    a.add(out1);
    a.add(int(kk));
    a.add(int(slots));
    a.add(int(experts));
    const f = k.rot_in_experts[if (dtype_bf16) 0 else 1];
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(g[0]), .y = @intCast(g[1]), .z = @intCast(g[2]) }, .block = .{ .x = 32 } }, s, &a);
}

/// gate/up epilogue: the split sums in order, rotated, scaled, SwiGLU, then Xd = fp16((act * suh_d) @ H).
pub fn gateupEpilogue(k: *const Kernels, z: u64, pick: u64, svh_g: u64, svh_u: u64, suh_d: u64, xd: u64, rows: usize, slots: usize, p: usize, n: usize, sk: usize, experts: usize, limit: f32, act_mode: c_int, s: cuda.Stream) !void {
    const g = epilogueGrid(rows, slots, n);
    var a: cuda.Args = .{};
    a.add(z);
    a.add(pick);
    a.add(svh_g);
    a.add(svh_u);
    a.add(suh_d);
    a.add(xd);
    a.add(int(p));
    a.add(int(n));
    a.add(int(sk));
    a.add(int(experts));
    a.add(limit);
    a.add(act_mode);
    try cuda.launch.launch(k.gateup_epilogue, .{ .grid = dim3(g), .block = .{ .x = 32 } }, s, &a);
}

/// down epilogue: Y fp32 = (split sums in order) @ H * svh_d.
pub fn downEpilogue(k: *const Kernels, z: u64, pick: u64, svh_d: u64, y: u64, rows: usize, slots: usize, p: usize, d: usize, sk: usize, experts: usize, s: cuda.Stream) !void {
    const g = epilogueGrid(rows, slots, d);
    var a: cuda.Args = .{};
    a.add(z);
    a.add(pick);
    a.add(svh_d);
    a.add(y);
    a.add(int(p));
    a.add(int(d));
    a.add(int(sk));
    a.add(int(experts));
    try cuda.launch.launch(k.down_epilogue, .{ .grid = dim3(g), .block = .{ .x = 32 } }, s, &a);
}

/// The slots' combination: out[r] = sum over slots of wts[r][k] * y[r*slots + k] (fp32, fma chain from 0).
pub fn combine(k: *const Kernels, y: u64, wts: u64, out: u64, rows: usize, d: usize, slots: usize, s: cuda.Stream) !void {
    const g = combineGrid(rows, d);
    var a: cuda.Args = .{};
    a.add(y);
    a.add(wts);
    a.add(out);
    a.add(int(d));
    a.add(int(slots));
    try cuda.launch.launch(k.combine, .{ .grid = dim3(g), .block = .{ .x = 256 } }, s, &a);
}

/// down epilogue and combination in one launch, the same arithmetic in the same order (the same bits).
pub fn downCombine(k: *const Kernels, z: u64, pick: u64, svh_d: u64, y: u64, wts: u64, out: u64, rows: usize, p: usize, d: usize, sk: usize, experts: usize, slots: usize, s: cuda.Stream) !void {
    if (slots > 32) return error.Slots;
    const g = downCombineGrid(rows, d);
    var a: cuda.Args = .{};
    a.add(z);
    a.add(pick);
    a.add(svh_d);
    a.add(y);
    a.add(wts);
    a.add(out);
    a.add(int(p));
    a.add(int(d));
    a.add(int(sk));
    a.add(int(experts));
    a.add(int(slots));
    try cuda.launch.launch(k.down_combine, .{ .grid = dim3(g), .block = .{ .x = @intCast(downCombineBlock(slots)) } }, s, &a);
}

/// The grouped expert GEMM: up to 16 members a program, per-expert trellis pointers and widths.
pub fn grouped(k: *const Kernels, setting: usize, range: usize, x0: u64, x1: u64, tp0: u64, tp1: u64, k2_0: u64, k2_1: u64, uids: u64, ucount: u64, members: u64, z: u64, kk: usize, n: usize, p: usize, sk: usize, maxm: usize, slots: usize, experts: usize, mats: usize, s: cuda.Stream) !void {
    const mtiles = (maxm + 15) / 16;
    const t = tile_settings[setting];
    const g = groupedGrid(experts, n, t.nt, mats, sk, mtiles);
    var a: cuda.Args = .{};
    a.add(x0);
    a.add(x1);
    a.add(tp0);
    a.add(tp1);
    a.add(k2_0);
    a.add(k2_1);
    a.add(uids);
    a.add(ucount);
    a.add(members);
    a.add(z);
    a.add(int(kk));
    a.add(int(n));
    a.add(int(p));
    a.add(int(sk));
    a.add(int(maxm));
    a.add(int(slots));
    const f = k.grouped[setting][range];
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(g[0]), .y = @intCast(g[1]), .z = @intCast(g[2]) }, .block = .{ .x = @intCast(t.w * 32) } }, s, &a);
}

/// The experts' dequantize-to-dense: W_q fp16 [K, N] of one matrix (the tests' path).
pub fn dequant(k: *const Kernels, k2: u32, words: u64, out: u64, kk: usize, n: usize, s: cuda.Stream) !void {
    const i = k2Index(k2) orelse return error.Width;
    const g = dequantGrid(kk, n);
    var a: cuda.Args = .{};
    a.add(words);
    a.add(out);
    a.add(int(kk));
    a.add(int(n));
    try cuda.launch.launch(k.dequant[i], .{ .grid = dim3(g), .block = .{ .x = 32 } }, s, &a);
}

// ---------------------------------------------------------------------------------------------------------------

const testing = std.testing;

test "plan matches linear.py across the pack's shapes" {
    // Expected values from linear.py's own plan() (the reference file, executed as it stands).
    const Case = struct { k: usize, n: usize, sk: usize, wk: usize, what: []const u8 };
    const cases = [_]Case{
        .{ .k = 2560, .n = 640, .sk = 4, .wk = 4, .what = "moe gate/up, shared gate/up" },
        .{ .k = 640, .n = 2560, .sk = 1, .wk = 4, .what = "moe down" },
        .{ .k = 2560, .n = 12288, .sk = 1, .wk = 8, .what = "q_proj" },
        .{ .k = 2560, .n = 512, .sk = 4, .wk = 4, .what = "k_proj, v_proj" },
        .{ .k = 6144, .n = 2560, .sk = 8, .wk = 4, .what = "o_proj" },
        .{ .k = 2560, .n = 10240, .sk = 1, .wk = 8, .what = "linear_attn.in_proj_qkv" },
        .{ .k = 2560, .n = 248320, .sk = 1, .wk = 8, .what = "lm_head" },
        .{ .k = 2560, .n = 2560, .sk = 4, .wk = 4, .what = "mtp.fc_hidden, mtp.fc_embedding" },
        .{ .k = 128, .n = 128, .sk = 1, .wk = 4, .what = "one tile" },
        .{ .k = 160, .n = 320001536, .sk = 1, .wk = 4, .what = "the n-gram table's row" },
    };
    for (cases) |c| {
        const got = plan(c.k, c.n);
        testing.expectEqual(c.sk, got.sk) catch |e| {
            std.debug.print("plan({d}, {d}) sk: want {d}, got {d} ({s})\n", .{ c.k, c.n, c.sk, got.sk, c.what });
            return e;
        };
        testing.expectEqual(c.wk, got.wk) catch |e| {
            std.debug.print("plan({d}, {d}) wk: want {d}, got {d} ({s})\n", .{ c.k, c.n, c.wk, got.wk, c.what });
            return e;
        };
    }
}

test "the split always divides K/16, which the kernel requires" {
    // Every layer the pack has, by the shapes the trellis headers give. (K=160, the n-gram table's row, is not a
    // GEMM and is deliberately absent: there K does not divide.)
    const cases = [_][2]usize{ .{ 2560, 640 }, .{ 640, 2560 }, .{ 2560, 12288 }, .{ 2560, 512 }, .{ 6144, 2560 }, .{ 2560, 10240 }, .{ 2560, 248320 }, .{ 2560, 2560 }, .{ 128, 128 } };
    for (cases) |c| {
        const s = plan(c[0], c[1]);
        try testing.expect(c[0] % 128 == 0);
        try testing.expect((c[0] / 16) % (s.sk * s.wk) == 0);
    }
}

test "strides and counters match linear.py" {
    // linear.py:142-148 and :161, for the pack's shapes and widths.
    try testing.expectEqual([2]usize{ 256, 40960 }, strides(.strips, 2560, 640, 8)); // 4-bit moe gate/up
    try testing.expectEqual([2]usize{ 1280, 256 }, strides(.stored, 2560, 640, 8));
    try testing.expectEqual([2]usize{ 384, 61440 }, strides(.strips, 2560, 12288, 12)); // 6-bit q_proj
    try testing.expectEqual([2]usize{ 36864, 384 }, strides(.stored, 2560, 12288, 12));
    try testing.expectEqual([2]usize{ 384, 147456 }, strides(.strips, 6144, 2560, 12)); // 6-bit o_proj
    try testing.expectEqual([2]usize{ 384, 61440 }, strides(.strips, 2560, 640, 12)); // 6-bit at N=640: (K/16)*8*tw
    try testing.expectEqual(@as(usize, 40), counterCount(640));
    try testing.expectEqual(@as(usize, 768), counterCount(12288));
    try testing.expectEqual(@as(usize, 15520), counterCount(248320));
}

test "k2 comes from the width, never from the header's mean" {
    try testing.expectEqual(@as(?u32, 8), k2Of(4));
    try testing.expectEqual(@as(?u32, 10), k2Of(5));
    try testing.expectEqual(@as(?u32, 12), k2Of(6));
    try testing.expectEqual(@as(?u32, 7), k2Of(3.5));
    try testing.expectEqual(@as(?u32, 5), k2Of(2.5));
    try testing.expectEqual(@as(?u32, 16), k2Of(8));
    try testing.expectEqual(@as(?u32, null), k2Of(4.05)); // the pack's mean, which is not a trellis width
    try testing.expectEqual(@as(?u32, null), k2Of(0.5));
    try testing.expectEqual(@as(?u32, null), k2Of(8.5));
    try testing.expectEqual(@as(?u8, 2), codebookId("mul1"));
    try testing.expectEqual(@as(?u8, 0), codebookId("3inst"));
    try testing.expectEqual(@as(?u8, 1), codebookId("mcg"));
    try testing.expectEqual(@as(?u8, null), codebookId("nvfp4"));
}

test "the launch shapes are the reference's" {
    try testing.expectEqual([2]usize{ 5, 128 }, rotInGrid(2560, 128)); // K/128 = 20 -> 5 blocks, any rows
    try testing.expectEqual([2]usize{ 1, 1 }, rotInGrid(512, 1)); // K/128 = 4 -> one block of four tiles
    try testing.expectEqual([2]usize{ 96, 1 }, linearGrid(12288, 1));
    try testing.expectEqual([2]usize{ 5, 4 }, linearGrid(640, 4));
    try testing.expectEqual(@as(usize, 256), linearBlock(8));
    try testing.expectEqual(@as(usize, 32768), linearShared(8, 128)); // WK * min(M, 8) * 128 * 4, the cap 48 KiB
    try testing.expectEqual(@as(usize, 2048), linearShared(2, 2)); // WK * M * 128 * 4, rows under the 8-row cap
    try testing.expectEqual([2]usize{ 768, 160 }, unpackGrid(2560, 12288));
    try testing.expectEqual([2]usize{ 640, 5 }, epilogueGrid(128, 5, 640)); // rows x slots, width / 128
    try testing.expectEqual([2]usize{ 1, 1 }, combineGrid(1, 256));
    try testing.expectEqual([2]usize{ 1, 2 }, combineGrid(1, 300));
    try testing.expectEqual([2]usize{ 1, 5 }, downCombineGrid(1, 640));
    try testing.expectEqual(@as(usize, 160), downCombineBlock(5));
    try testing.expectEqual([3]usize{ 8, 5, 40 }, groupedGrid(8, 640, 8, 2, 4, 5)); // experts, N/(16*nt), mats*sk*mtiles
    try testing.expectEqual([2]usize{ 160, 40 }, dequantGrid(2560, 640));
    try testing.expectEqual(@as(usize, 40 * 5 * 4), groupShared(40, 5));
}

test "the symbol table carries the widths and planes it claims" {
    for (k2s, 0..) |k2, i| {
        for (sym.linear[i], wks) |s, wk| {
            try testing.expect(std.mem.startsWith(u8, s, "_ZN10tf_fn_exl311tf_exl3_lin"));
            var want: [48]u8 = undefined;
            const t = try std.fmt.bufPrint(&want, "linear_kernelILi{d}ELi2ELi{d}EEEv", .{ k2, wk });
            try testing.expect(std.mem.indexOf(u8, s, t) != null);
        }
        var dq: [40]u8 = undefined;
        const dq_want = try std.fmt.bufPrint(&dq, "dequant_kernelILi2ELi{d}E", .{k2});
        try testing.expect(std.mem.indexOf(u8, sym.dequant[i], dq_want) != null);
        var up: [40]u8 = undefined;
        const up_want = try std.fmt.bufPrint(&up, "unpack_kernelILi{d}ELi2E", .{k2});
        try testing.expect(std.mem.indexOf(u8, sym.unpack[i], up_want) != null);
    }
    for (sym.grouped, 0..) |row, i| for (row, 0..) |s, j| {
        try testing.expect(std.mem.startsWith(u8, s, "_ZN10tf_fn_exl38tf_exl3x14grouped_kernelILi2E"));
        var want: [24]u8 = undefined;
        const t = try std.fmt.bufPrint(&want, "ELi{d}ELi{d}ELi{d}ELi{d}ELi{d}E", .{ tile_settings[i].nt, tile_settings[i].w, tile_settings[i].pf, ranges[j].lo, ranges[j].hi });
        try testing.expect(std.mem.indexOf(u8, s, t) != null);
    };
    for (sym.rot_in_experts) |s| try testing.expect(std.mem.startsWith(u8, s, "_ZN10tf_fn_exl311tf_exl3_exp13rot_in_kernel"));
    for ([_][:0]const u8{ sym.rot_in, sym.group, sym.gateup_epilogue, sym.down_epilogue, sym.combine, sym.down_combine }) |s| {
        try testing.expect(std.mem.startsWith(u8, s, "_ZN10tf_fn_exl311tf_exl3_"));
    }
}

test "the range instance covers the widths a group can mix" {
    try testing.expectEqual(@as(usize, 0), rangeIndex(8, 8));
    try testing.expectEqual(@as(usize, 1), rangeIndex(2, 10));
    try testing.expectEqual(@as(usize, 2), rangeIndex(2, 16));
    try testing.expectEqual(@as(usize, 1), rangeIndex(8, 10));
    try testing.expectEqual(@as(usize, 2), rangeIndex(8, 12)); // 12 is past the (2, 10) instance
    try testing.expectEqual(@as(usize, 0), tileIndex(8, 4, 1).?);
    try testing.expectEqual(@as(usize, 1), tileIndex(8, 4, 2).?);
    try testing.expectEqual(@as(usize, 2), tileIndex(4, 4, 2).?);
    try testing.expectEqual(@as(?usize, null), tileIndex(16, 4, 1));
}

test "strips puts each 128-column block's tiles in k order (linear.py:48)" {
    // kt 2, nt 16 (two blocks), one word a tile: source [t][n] becomes [b][t][s], written out by hand.
    var src: [32]u32 = undefined;
    for (&src, 0..) |*v, i| v.* = @intCast(i);
    var dst: [32]u32 = undefined;
    try strips(&dst, &src, 2, 16, 1);
    const want = [32]u32{ 0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31 };
    try testing.expectEqualSlices(u32, &want, &dst);
    // three words a tile: each tile stays contiguous, and the blocks and k tiles swap places
    var src3: [96]u32 = undefined;
    for (&src3, 0..) |*v, i| v.* = @intCast(i);
    var dst3: [96]u32 = undefined;
    try strips(&dst3, &src3, 2, 16, 3);
    try testing.expectEqualSlices(u32, src3[0..3], dst3[0..3]); // block 0, k tile 0, sub-tile 0
    try testing.expectEqualSlices(u32, src3[48..51], dst3[24..27]); // block 0, k tile 1 -> source (1, n 0)
    try testing.expectEqualSlices(u32, src3[24..27], dst3[48..51]); // block 1, k tile 0 -> source (0, n 8)
    try testing.expectError(error.Short, strips(dst3[0..95], &src3, 2, 16, 3));
    try testing.expectError(error.UnexpectedTensor, strips(&dst3, &src3, 2, 12, 3)); // not whole 128-column blocks
}

test "bitsOf reads the width off the trellis, per tensor" {
    // The pack's own headers: a 4-bit expert, a 6-bit q_proj, a 5-bit MTP pair.
    try testing.expectEqual(@as(?f64, 4), bitsOf(&.{ 160, 40, 64 }));
    try testing.expectEqual(@as(?f64, 6), bitsOf(&.{ 160, 768, 96 }));
    try testing.expectEqual(@as(?f64, 5), bitsOf(&.{ 160, 160, 80 }));
    try testing.expectEqual(@as(?u32, 8), k2Of(bitsOf(&.{ 160, 40, 64 }).?));
    try testing.expectEqual(@as(?u32, 12), k2Of(bitsOf(&.{ 160, 768, 96 }).?));
    try testing.expectEqual(@as(?u32, 10), k2Of(bitsOf(&.{ 160, 160, 80 }).?));
    // a last dimension that is not a whole number of words (the reference's own refusal)
    try testing.expectEqual(@as(?f64, null), bitsOf(&.{ 160, 40, 63 }));
    try testing.expectEqual(@as(?f64, null), bitsOf(&.{ 160, 40, 61 })); // the n-gram table's row: 61 is odd
    try testing.expectEqual(@as(?f64, null), bitsOf(&.{}));
}

test "the tile setting defaultTile picks is one the fatbin carries" {
    // The pack's projection: D 2560 -> I 640 for gate and up, I 640 -> D 2560 for down.
    const gu = try defaultTile(2560, 640, true);
    try testing.expectEqual(glm_gateup, gu);
    try testing.expect(tileIndex(gu.nt, gu.w, gu.pf) != null); // (8, 4, 1) is a compiled instance
    const d = try defaultTile(640, 2560, false);
    try testing.expectEqual(glm_down, d);
    try testing.expect(tileIndex(d.nt, d.w, d.pf) != null);
    // gate/up of a K GLM's own setting cannot divide falls back, still onto a compiled instance
    const fb = try defaultTile(640, 2560, true);
    try testing.expectEqual(@as(usize, 2), fb.sk);
    try testing.expect(tileIndex(fb.nt, fb.w, fb.pf) != null);
    // every candidate the reference lists resolves to one of the three settings in the symbol table
    for ([_]Tile{ glm_gateup, glm_down, .{ .nt = 8, .w = 4, .sk = 2, .pf = 1 }, .{ .nt = 4, .w = 4, .sk = 2, .pf = 2 }, .{ .nt = 4, .w = 4, .sk = 1, .pf = 2 } }) |t| {
        try testing.expect(tileIndex(t.nt, t.w, t.pf) != null);
    }
    try testing.expectError(error.NoTileSetting, defaultTile(80, 640, true));
}

test "an expert's trellis bytes and a layer's width envelope" {
    // 4-bit experts at D 2560, I 640: (160 * 40) tiles * 64 words * 2 bytes = 819 200 B a projection.
    try testing.expectEqual(@as(u64, 819_200), expertBytes(2560, 640, 8, 0, 0));
    try testing.expectEqual(@as(u64, 2_457_600), expertBytes(2560, 640, 8, 8, 8));
    try testing.expectEqual([2]u32{ 8, 12 }, envelope(&.{ 12, 8, 12, 8 })); // a layer mixing 4- and 6-bit experts
    try testing.expectEqual([2]u32{ 10, 10 }, envelope(&.{10}));
    try testing.expectEqual([2]u32{ 0, 0 }, envelope(&.{}));
}

test "the widths the loaders will derive resolve to a table row" {
    // Every k2 a trellis shape can produce for this pack's layer set (4, 5, 6 bits), and the ones it cannot.
    for ([_]f64{ 4, 5, 6 }) |bits| {
        const k2 = k2Of(bits).?;
        try testing.expect(k2Index(k2) != null);
    }
    try testing.expectEqual(@as(?usize, null), k2Index(16)); // 8-bit: the fatbin has it, this binding does not
}
