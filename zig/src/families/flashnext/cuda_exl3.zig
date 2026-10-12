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
const cfgs = @import("cuda_mia_config.zig");

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

// The tile sweep: TF_FLASHNEXT_X3_TILE names a variant for the grouped GEMM's gate/up and down classes; unset is today.

/// A tile variant a sweep may name: the same (nt, w, sk, pf) `defaultTile` returns, in the reference's spelling.
pub const Variant = struct { nt: usize, w: usize, sk: usize, pf: usize };

/// id 0, GLM's gate/up tile (experts.py:134): today's gate/up default, and the pack's expected best.
pub const v_glm_gu: Variant = .{ .nt = 8, .w = 4, .sk = 4, .pf = 1 };
/// id 1, GLM's down tile (experts.py:134): today's down default (one K split, the reference's pick for I -> D).
pub const v_glm_dn: Variant = .{ .nt = 8, .w = 4, .sk = 1, .pf = 1 };
/// id 2, the reference's first fallback when GLM's tile does not divide (experts.py:134): two K splits, one in flight.
pub const v_narrow: Variant = .{ .nt = 8, .w = 4, .sk = 2, .pf = 1 };
/// id 3, the reference's (8, 4, 2) instance at one K split; upstream's prefill tiles keep 2 k tiles in flight.
pub const v_pf2: Variant = .{ .nt = 8, .w = 4, .sk = 1, .pf = 2 };
/// id 4, the reference's (8, 4, 2) instance at two K splits: its narrow-N fallback with two tiles in flight.
pub const v_pf2_sk2: Variant = .{ .nt = 8, .w = 4, .sk = 2, .pf = 2 };
/// id 5, the reference's 4-tile-warp setting (experts.py:134), 2 in flight: half the columns a warp, more overlap.
pub const v_nt4: Variant = .{ .nt = 4, .w = 4, .sk = 2, .pf = 2 };
/// id 6, the 4-tile-warp setting at one K split, two tiles in flight (no measured effect known).
pub const v_nt4_sk1: Variant = .{ .nt = 4, .w = 4, .sk = 1, .pf = 2 };
/// id 7, 4 tiles a warp, one in flight, a 4-way split: upstream's one-tile-a-warp regime (x3gm GM_GU2/GU3, NG 16).
pub const v_nt4_pf1: Variant = .{ .nt = 4, .w = 4, .sk = 4, .pf = 1 };

/// The variants the switch's ids index; the parent sweeps. Upstream's own tile retune is +4-12% prefill there.
pub const variants = [_]Variant{ v_glm_gu, v_glm_dn, v_narrow, v_pf2, v_pf2_sk2, v_nt4, v_nt4_sk1, v_nt4_pf1 };

/// The switch's name (the repo's TF_FLASHNEXT_* idiom).
const tile_env = "TF_FLASHNEXT_X3_TILE";

/// The variant the switch names for a class: "<gu>[,<dn>]", ids into `variants`; one id means both; unset is null.
pub fn tileVariant(v: ?[]const u8, gateup: bool) !?Variant {
    const text = v orelse return null;
    if (text.len == 0) return null;
    var it = std.mem.tokenizeScalar(u8, text, ',');
    const gu = it.next() orelse return null;
    const dn = it.next() orelse gu;
    const id = std.fmt.parseInt(usize, if (gateup) gu else dn, 10) catch return error.NoTileVariant;
    if (id >= variants.len) return error.NoTileVariant;
    const t = variants[id];
    if (tileIndex(t.nt, t.w, t.pf) == null) return error.NoTileKernel;
    return t;
}

/// A variant's Tile for one shape: (K/16) | (sk * w) and (N/16) | nt, else NoTileShape, never launched wrong.
pub fn variantTile(t: Variant, k: usize, n: usize) !Tile {
    if (k % (16 * t.sk * t.w) != 0 or n % (16 * t.nt) != 0) return error.NoTileShape;
    return .{ .nt = t.nt, .w = t.w, .sk = t.sk, .pf = t.pf };
}

/// The tile a class's grouped GEMM takes: the switch's variant when set, else `defaultTile`'s shape walk, unchanged.
pub fn tile(k: usize, n: usize, gateup: bool) !Tile {
    if (try tileVariant(tileEnvText(), gateup)) |t| return variantTile(t, k, n); // a bad id/kernel/shape errors
    return defaultTile(k, n, gateup);
}

/// TF_FLASHNEXT_X3_TILE, read once. A forward asks per MoE layer, and the text does not change during the process.
fn tileEnvText() ?[]const u8 {
    const state = struct {
        var text: ?[]const u8 = null;
        var read: bool = false;
    };
    if (!state.read) {
        state.text = if (std.c.getenv(tile_env)) |p| std.mem.span(p) else null;
        state.read = true;
    }
    return state.text;
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

/// Picks per histogram tile. group_hist_kernel / group_scatter_kernel use the same constant.
pub const group_tile: usize = 1024;

pub fn groupTiles(n: usize) usize {
    if (n == 0) return 0;
    return (n + group_tile - 1) / group_tile;
}

/// Workspace ints: hist[tiles, experts], totals[experts], place[experts].
pub fn groupWorkspaceInts(n: usize, experts: usize) usize {
    return groupTiles(n) * experts + 2 * experts;
}

pub fn groupWorkspaceBytes(rows: usize, slots: usize, experts: usize) usize {
    return groupWorkspaceInts(rows * slots, experts) * @sizeOf(i32);
}

/// Warps in a gate/up or down epilogue block: each warp is one pick, the same fwht and split sum as a block of 32.
pub const epilogue_warps: usize = 4;
/// Picks below this run one warp a block (decode / verify): more blocks, the 32-thread launch shape.
pub const epilogue_wide_min: usize = 256;

pub fn epilogueWarps(p: usize) usize {
    return if (p >= epilogue_wide_min) epilogue_warps else 1;
}

/// gateup_epilogue_kernel and down_epilogue_kernel: grid (ceil(rows*slots/w), width/128), block 32*w.
pub fn epilogueGrid(rows: usize, slots: usize, width: usize) [2]usize {
    const p = rows * slots;
    const w = epilogueWarps(p);
    return .{ (p + w - 1) / w, width / 128 };
}

pub fn epilogueBlock(rows: usize, slots: usize) usize {
    return epilogueWarps(rows * slots) * 32;
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
pub const tile_settings = [4]struct { nt: usize, w: usize, pf: usize }{
    .{ .nt = 8, .w = 4, .pf = 1 },
    .{ .nt = 8, .w = 4, .pf = 2 },
    .{ .nt = 4, .w = 4, .pf = 2 },
    .{ .nt = 4, .w = 4, .pf = 1 }, // (4, 4, 1): completes the NT 4/8 x PF 1/2 grid the reference's candidates cover
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
    pub const group_hist = "_ZN10tf_fn_exl311tf_exl3_exp17group_hist_kernelEPKiPiii";
    pub const group_scan = "_ZN10tf_fn_exl311tf_exl3_exp17group_scan_kernelEPiS1_S1_S1_iii";
    pub const group_scatter = "_ZN10tf_fn_exl311tf_exl3_exp20group_scatter_kernelEPKiS2_Piiiii";
    pub const group_small = "_ZN10tf_fn_exl311tf_exl3_exp18group_small_kernelEPKiPiS3_S3_iiii";
    pub const gateup_epilogue = "_ZN10tf_fn_exl311tf_exl3_exp22gateup_epilogue_kernelEPKfPKiPK6__halfS7_S7_PS5_iiiifi";
    pub const down_epilogue = "_ZN10tf_fn_exl311tf_exl3_exp20down_epilogue_kernelEPKfPKiPK6__halfPfiiii";
    pub const combine = "_ZN10tf_fn_exl311tf_exl3_exp14combine_kernelEPKfS2_Pfii";
    pub const down_combine = "_ZN10tf_fn_exl311tf_exl3_exp19down_combine_kernelEPKfPKiPK6__halfPfS2_S8_iiiii";
    pub const down_combine_bf16 = "_ZN10tf_fn_exl311tf_exl3_exp24down_combine_bf16_kernelEPKfPKiPK6__halfS2_S2_P13__nv_bfloat16iiiii";
    /// tf_exl3x (experts_grouped.cuh): [tile setting][k2 range], MTL 1 (one 16-member tile a program).
    pub const grouped = [4][3][:0]const u8{
        .{
            "_ZN10tf_fn_exl38tf_exl3x14grouped_kernelILi2ELi8ELi4ELi1ELi8ELi8ELi1EEEvPK6__halfS4_PKlS6_PKiS8_S8_S8_S8_Pfiiiiii",
            "_ZN10tf_fn_exl38tf_exl3x14grouped_kernelILi2ELi8ELi4ELi1ELi2ELi10ELi1EEEvPK6__halfS4_PKlS6_PKiS8_S8_S8_S8_Pfiiiiii",
            "_ZN10tf_fn_exl38tf_exl3x14grouped_kernelILi2ELi8ELi4ELi1ELi2ELi16ELi1EEEvPK6__halfS4_PKlS6_PKiS8_S8_S8_S8_Pfiiiiii",
        },
        .{
            "_ZN10tf_fn_exl38tf_exl3x14grouped_kernelILi2ELi8ELi4ELi2ELi8ELi8ELi1EEEvPK6__halfS4_PKlS6_PKiS8_S8_S8_S8_Pfiiiiii",
            "_ZN10tf_fn_exl38tf_exl3x14grouped_kernelILi2ELi8ELi4ELi2ELi2ELi10ELi1EEEvPK6__halfS4_PKlS6_PKiS8_S8_S8_S8_Pfiiiiii",
            "_ZN10tf_fn_exl38tf_exl3x14grouped_kernelILi2ELi8ELi4ELi2ELi2ELi16ELi1EEEvPK6__halfS4_PKlS6_PKiS8_S8_S8_S8_Pfiiiiii",
        },
        .{
            "_ZN10tf_fn_exl38tf_exl3x14grouped_kernelILi2ELi4ELi4ELi2ELi8ELi8ELi1EEEvPK6__halfS4_PKlS6_PKiS8_S8_S8_S8_Pfiiiiii",
            "_ZN10tf_fn_exl38tf_exl3x14grouped_kernelILi2ELi4ELi4ELi2ELi2ELi10ELi1EEEvPK6__halfS4_PKlS6_PKiS8_S8_S8_S8_Pfiiiiii",
            "_ZN10tf_fn_exl38tf_exl3x14grouped_kernelILi2ELi4ELi4ELi2ELi2ELi16ELi1EEEvPK6__halfS4_PKlS6_PKiS8_S8_S8_S8_Pfiiiiii",
        },
        .{
            "_ZN10tf_fn_exl38tf_exl3x14grouped_kernelILi2ELi4ELi4ELi1ELi8ELi8ELi1EEEvPK6__halfS4_PKlS6_PKiS8_S8_S8_S8_Pfiiiiii",
            "_ZN10tf_fn_exl38tf_exl3x14grouped_kernelILi2ELi4ELi4ELi1ELi2ELi10ELi1EEEvPK6__halfS4_PKlS6_PKiS8_S8_S8_S8_Pfiiiiii",
            "_ZN10tf_fn_exl38tf_exl3x14grouped_kernelILi2ELi4ELi4ELi1ELi2ELi16ELi1EEEvPK6__halfS4_PKlS6_PKiS8_S8_S8_S8_Pfiiiiii",
        },
    };
    /// tf_exl3x: the wide member-pass arm (MTL 2, 32 members a program, rows staged in dynamic shared), mul1, opt-in.
    pub const grouped_wide = [4][3][:0]const u8{
        .{
            "_ZN10tf_fn_exl38tf_exl3x14grouped_kernelILi2ELi8ELi4ELi1ELi8ELi8ELi2EEEvPK6__halfS4_PKlS6_PKiS8_S8_S8_S8_Pfiiiiii",
            "_ZN10tf_fn_exl38tf_exl3x14grouped_kernelILi2ELi8ELi4ELi1ELi2ELi10ELi2EEEvPK6__halfS4_PKlS6_PKiS8_S8_S8_S8_Pfiiiiii",
            "_ZN10tf_fn_exl38tf_exl3x14grouped_kernelILi2ELi8ELi4ELi1ELi2ELi16ELi2EEEvPK6__halfS4_PKlS6_PKiS8_S8_S8_S8_Pfiiiiii",
        },
        .{
            "_ZN10tf_fn_exl38tf_exl3x14grouped_kernelILi2ELi8ELi4ELi2ELi8ELi8ELi2EEEvPK6__halfS4_PKlS6_PKiS8_S8_S8_S8_Pfiiiiii",
            "_ZN10tf_fn_exl38tf_exl3x14grouped_kernelILi2ELi8ELi4ELi2ELi2ELi10ELi2EEEvPK6__halfS4_PKlS6_PKiS8_S8_S8_S8_Pfiiiiii",
            "_ZN10tf_fn_exl38tf_exl3x14grouped_kernelILi2ELi8ELi4ELi2ELi2ELi16ELi2EEEvPK6__halfS4_PKlS6_PKiS8_S8_S8_S8_Pfiiiiii",
        },
        .{
            "_ZN10tf_fn_exl38tf_exl3x14grouped_kernelILi2ELi4ELi4ELi2ELi8ELi8ELi2EEEvPK6__halfS4_PKlS6_PKiS8_S8_S8_S8_Pfiiiiii",
            "_ZN10tf_fn_exl38tf_exl3x14grouped_kernelILi2ELi4ELi4ELi2ELi2ELi10ELi2EEEvPK6__halfS4_PKlS6_PKiS8_S8_S8_S8_Pfiiiiii",
            "_ZN10tf_fn_exl38tf_exl3x14grouped_kernelILi2ELi4ELi4ELi2ELi2ELi16ELi2EEEvPK6__halfS4_PKlS6_PKiS8_S8_S8_S8_Pfiiiiii",
        },
        .{
            "_ZN10tf_fn_exl38tf_exl3x14grouped_kernelILi2ELi4ELi4ELi1ELi8ELi8ELi2EEEvPK6__halfS4_PKlS6_PKiS8_S8_S8_S8_Pfiiiiii",
            "_ZN10tf_fn_exl38tf_exl3x14grouped_kernelILi2ELi4ELi4ELi1ELi2ELi10ELi2EEEvPK6__halfS4_PKlS6_PKiS8_S8_S8_S8_Pfiiiiii",
            "_ZN10tf_fn_exl38tf_exl3x14grouped_kernelILi2ELi4ELi4ELi1ELi2ELi16ELi2EEEvPK6__halfS4_PKlS6_PKiS8_S8_S8_S8_Pfiiiiii",
        },
    };
    /// tf_exl3x: the decode ablation (grouped_abl_kernel, timing probe only, garbage output), same table order, MTL 1.
    pub const grouped_abl = [4][3][:0]const u8{
        .{
            "_ZN10tf_fn_exl38tf_exl3x18grouped_abl_kernelILi2ELi8ELi4ELi1ELi8ELi8ELi1EEEvPK6__halfS4_PKlS6_PKiS8_S8_S8_S8_Pfiiiiii",
            "_ZN10tf_fn_exl38tf_exl3x18grouped_abl_kernelILi2ELi8ELi4ELi1ELi2ELi10ELi1EEEvPK6__halfS4_PKlS6_PKiS8_S8_S8_S8_Pfiiiiii",
            "_ZN10tf_fn_exl38tf_exl3x18grouped_abl_kernelILi2ELi8ELi4ELi1ELi2ELi16ELi1EEEvPK6__halfS4_PKlS6_PKiS8_S8_S8_S8_Pfiiiiii",
        },
        .{
            "_ZN10tf_fn_exl38tf_exl3x18grouped_abl_kernelILi2ELi8ELi4ELi2ELi8ELi8ELi1EEEvPK6__halfS4_PKlS6_PKiS8_S8_S8_S8_Pfiiiiii",
            "_ZN10tf_fn_exl38tf_exl3x18grouped_abl_kernelILi2ELi8ELi4ELi2ELi2ELi10ELi1EEEvPK6__halfS4_PKlS6_PKiS8_S8_S8_S8_Pfiiiiii",
            "_ZN10tf_fn_exl38tf_exl3x18grouped_abl_kernelILi2ELi8ELi4ELi2ELi2ELi16ELi1EEEvPK6__halfS4_PKlS6_PKiS8_S8_S8_S8_Pfiiiiii",
        },
        .{
            "_ZN10tf_fn_exl38tf_exl3x18grouped_abl_kernelILi2ELi4ELi4ELi2ELi8ELi8ELi1EEEvPK6__halfS4_PKlS6_PKiS8_S8_S8_S8_Pfiiiiii",
            "_ZN10tf_fn_exl38tf_exl3x18grouped_abl_kernelILi2ELi4ELi4ELi2ELi2ELi10ELi1EEEvPK6__halfS4_PKlS6_PKiS8_S8_S8_S8_Pfiiiiii",
            "_ZN10tf_fn_exl38tf_exl3x18grouped_abl_kernelILi2ELi4ELi4ELi2ELi2ELi16ELi1EEEvPK6__halfS4_PKlS6_PKiS8_S8_S8_S8_Pfiiiiii",
        },
        .{
            "_ZN10tf_fn_exl38tf_exl3x18grouped_abl_kernelILi2ELi4ELi4ELi1ELi8ELi8ELi1EEEvPK6__halfS4_PKlS6_PKiS8_S8_S8_S8_Pfiiiiii",
            "_ZN10tf_fn_exl38tf_exl3x18grouped_abl_kernelILi2ELi4ELi4ELi1ELi2ELi10ELi1EEEvPK6__halfS4_PKlS6_PKiS8_S8_S8_S8_Pfiiiiii",
            "_ZN10tf_fn_exl38tf_exl3x18grouped_abl_kernelILi2ELi4ELi4ELi1ELi2ELi16ELi1EEEvPK6__halfS4_PKlS6_PKiS8_S8_S8_S8_Pfiiiiii",
        },
    };
    /// tf_exl3x: the staged gm arm (128 members a program, the member rows the B operand), opt-in (TF_FLASHNEXT_X3_GM).
    pub const grouped_gm = [3][:0]const u8{
        "_ZN10tf_fn_exl38tf_exl3x17grouped_gm_kernelILi2ELi8ELi8ELi8ELi8ELi2ELi4ELi4EEEvPK6__halfS4_PKlS6_PKiS8_S8_S8_S8_Pfiiiiii",
        "_ZN10tf_fn_exl38tf_exl3x17grouped_gm_kernelILi2ELi8ELi8ELi2ELi10ELi2ELi4ELi4EEEvPK6__halfS4_PKlS6_PKiS8_S8_S8_S8_Pfiiiiii",
        "_ZN10tf_fn_exl38tf_exl3x17grouped_gm_kernelILi2ELi8ELi8ELi2ELi16ELi2ELi4ELi4EEEvPK6__halfS4_PKlS6_PKiS8_S8_S8_S8_Pfiiiiii",
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
    group_hist: cuda.Function,
    group_scan: cuda.Function,
    group_scatter: cuda.Function,
    group_small: cuda.Function,
    gateup_epilogue: cuda.Function,
    down_epilogue: cuda.Function,
    combine: cuda.Function,
    down_combine: cuda.Function,
    down_combine_bf16: cuda.Function,
    grouped: [4][3]cuda.Function,
    /// the wide arm's instances: null when the embedded fatbin predates them, so the default path still loads
    grouped_wide: [4][3]?cuda.Function,
    /// the decode ablation's instances: null when the embedded fatbin predates them (a timing probe, off by default)
    grouped_abl: [4][3]?cuda.Function,
    /// the staged gm arm's instances: null when the embedded fatbin predates them (opt-in, off by default)
    grouped_gm: [3]?cuda.Function,
    dequant: [3]cuda.Function,
    /// the wide arm's dynamic ceiling (the device's opt-in less its static use), for the staged A tile
    wide_smem_max: u32,
    /// the gm arm's dynamic ceiling (the device's opt-in less its static use), for the staged member ring
    gm_smem_max: u32,

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
        k.group_hist = try k.module.function(sym.group_hist);
        k.group_scan = try k.module.function(sym.group_scan);
        k.group_scatter = try k.module.function(sym.group_scatter);
        k.group_small = try k.module.function(sym.group_small);
        k.gateup_epilogue = try k.module.function(sym.gateup_epilogue);
        k.down_epilogue = try k.module.function(sym.down_epilogue);
        k.combine = try k.module.function(sym.combine);
        k.down_combine = try k.module.function(sym.down_combine);
        k.down_combine_bf16 = try k.module.function(sym.down_combine_bf16);
        for (sym.grouped, 0..) |row, i| for (row, 0..) |s, j| {
            k.grouped[i][j] = try k.module.function(s);
        };
        for (sym.grouped_wide, 0..) |row, i| for (row, 0..) |s, j| {
            k.grouped_wide[i][j] = k.module.function(s) catch null;
        };
        for (sym.grouped_abl, 0..) |row, i| for (row, 0..) |s, j| {
            k.grouped_abl[i][j] = k.module.function(s) catch null;
        };
        for (sym.grouped_gm, 0..) |s, i| k.grouped_gm[i] = k.module.function(s) catch null;
        for (sym.dequant, 0..) |s, i| k.dequant[i] = try k.module.function(s);
        const ceiling: usize = @intCast(try ctx.attribute(.max_shared_memory_per_block_optin));
        // the wide arm stages its A tile in dynamic shared: opt the staged instances in against the same ceiling
        const wstatic: usize = if (k.grouped_wide[0][0]) |f| @intCast(try f.attribute(.shared_size_bytes)) else 0;
        k.wide_smem_max = @intCast(ceiling -| wstatic);
        for (k.grouped_wide) |row| for (row) |maybe| {
            if (maybe) |f| try f.allowDynamicShared(k.wide_smem_max);
        };
        // the gm arm stages its member ring in dynamic shared: opt the staged instances in against the same ceiling
        const gstatic: usize = if (k.grouped_gm[0]) |f| @intCast(try f.attribute(.shared_size_bytes)) else 0;
        k.gm_smem_max = @intCast(ceiling -| gstatic);
        for (k.grouped_gm) |maybe| {
            if (maybe) |f| try f.allowDynamicShared(k.gm_smem_max);
        }
        return k;
    }

    pub fn deinit(k: *Kernels) void {
        k.module.unload();
    }
};

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

/// Experts in id order, members row*32+slot in pick order, -1 padded: histogram, scan, stable scatter into `work`.
pub fn group(k: *const Kernels, pick: u64, uids: u64, ucount: u64, members: u64, work: u64, rows: usize, slots: usize, experts: usize, maxm: usize, s: cuda.Stream) !void {
    if (slots > 32) return error.Slots;
    if (rows == 0 or experts == 0) return;
    if (experts * @sizeOf(i32) > 48 * 1024) return error.TooManyExperts;
    const n = rows * slots;
    const tiles = groupTiles(n);
    if (tiles == 1 and 2 * experts * @sizeOf(i32) <= 48 * 1024) {
        var one: cuda.Args = .{};
        one.add(pick);
        one.add(uids);
        one.add(ucount);
        one.add(members);
        one.add(int(n));
        one.add(int(slots));
        one.add(int(experts));
        one.add(int(maxm));
        return cuda.launch.launch(k.group_small, .{ .grid = .{ .x = 1 }, .block = .{ .x = 256 }, .shared = @intCast(2 * experts * @sizeOf(i32)) }, s, &one);
    }
    try s.d.check(s.d.api.cuMemsetD8Async(work, 0, tiles * experts * @sizeOf(i32), s.handle), "cuMemsetD8Async");
    const ni = int(n);
    const ne = int(experts);
    var hist: cuda.Args = .{};
    hist.add(pick);
    hist.add(work);
    hist.add(ni);
    hist.add(ne);
    try cuda.launch.launch(k.group_hist, .{ .grid = .{ .x = @intCast(tiles) }, .block = .{ .x = 256 } }, s, &hist);
    var scan: cuda.Args = .{};
    scan.add(work);
    scan.add(uids);
    scan.add(ucount);
    scan.add(members);
    scan.add(ni);
    scan.add(ne);
    scan.add(int(maxm));
    try cuda.launch.launch(k.group_scan, .{ .grid = .{ .x = 1 }, .block = .{ .x = 256 } }, s, &scan);
    var scatter: cuda.Args = .{};
    scatter.add(pick);
    scatter.add(work);
    scatter.add(members);
    scatter.add(ni);
    scatter.add(int(slots));
    scatter.add(ne);
    scatter.add(int(maxm));
    try cuda.launch.launch(k.group_scatter, .{ .grid = .{ .x = @intCast(tiles) }, .block = .{ .x = 128 }, .shared = @intCast(experts * @sizeOf(i32)) }, s, &scatter);
}

/// Host form of the three group kernels: same uids, member ids and -1 pad. `scratch` holds the workspace plus one local counter an expert.
pub fn groupPlan(pick: []const i32, slots: usize, experts: usize, maxm: usize, uids: []i32, members: []i32, scratch: []i32) usize {
    const n = pick.len;
    const tiles = groupTiles(n);
    const hist_n = tiles * experts;
    const hist = scratch[0..hist_n];
    @memset(hist, 0);
    const totals = scratch[hist_n..][0..experts];
    const place = scratch[hist_n + experts ..][0..experts];
    const local = scratch[hist_n + 2 * experts ..][0..experts];
    for (pick, 0..) |e, i| {
        if (e < 0 or @as(usize, @intCast(e)) >= experts) continue;
        hist[(i / group_tile) * experts + @as(usize, @intCast(e))] += 1;
    }
    for (0..experts) |e| {
        var sum: i32 = 0;
        for (0..tiles) |t| {
            const c = hist[t * experts + e];
            hist[t * experts + e] = sum;
            sum += c;
        }
        totals[e] = sum;
    }
    var u: usize = 0;
    for (0..experts) |e| {
        if (totals[e] == 0) {
            place[e] = -1;
            continue;
        }
        uids[u] = @intCast(e);
        place[e] = @intCast(u);
        u += 1;
    }
    @memset(members[0 .. u * maxm], -1);
    @memset(local, 0);
    const cap: i32 = @intCast(maxm);
    var cur: usize = 0;
    for (pick, 0..) |e, i| {
        const gt = i / group_tile;
        if (gt != cur) {
            @memset(local, 0);
            cur = gt;
        }
        if (e < 0 or @as(usize, @intCast(e)) >= experts) continue;
        const eu: usize = @intCast(e);
        const rank = hist[gt * experts + eu] + local[eu];
        local[eu] += 1;
        if (rank >= cap) continue;
        members[@as(usize, @intCast(place[eu])) * maxm + @as(usize, @intCast(rank))] = @intCast((i / slots) * 32 + (i % slots));
    }
    return u;
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
    try cuda.launch.launch(k.gateup_epilogue, .{ .grid = dim3(g), .block = .{ .x = @intCast(epilogueBlock(rows, slots)) } }, s, &a);
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
    try cuda.launch.launch(k.down_epilogue, .{ .grid = dim3(g), .block = .{ .x = @intCast(epilogueBlock(rows, slots)) } }, s, &a);
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

/// downCombine with the row rounded to bf16 [rows, D] and no per-slot y stores (the write-back's bf16 branch).
pub fn downCombineBf16(k: *const Kernels, z: u64, pick: u64, svh_d: u64, y: u64, wts: u64, out: u64, rows: usize, p: usize, d: usize, sk: usize, experts: usize, slots: usize, s: cuda.Stream) !void {
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
    try cuda.launch.launch(k.down_combine_bf16, .{ .grid = dim3(g), .block = .{ .x = @intCast(downCombineBlock(slots)) } }, s, &a);
}

/// The wide member-pass arm's member-tile count: 2 tiles of 16 = 32 member rows a program, their rows staged.
pub const wide_member_tiles: usize = 2;

/// grouped_kernel's launched member rows an expert (fn_exl3.cu TF_GROUPED_GRID_MEMBERS); more members loop in the program.
pub const grouped_grid_members: usize = 512;

/// The wide arm's dynamic shared: MTL*16 rows of the split's k slice (stride 4 mod 32), then the reduction buffer.
pub fn wideStageSmem(per_split: usize, w: usize) usize {
    const words = per_split * 8;
    const sw = words + ((4 + 32 - (words & 31)) & 31);
    return 16 * wide_member_tiles * sw * 4 + w * 16 * wide_member_tiles * 16 * 4;
}

/// The opt-in switch's reading: unset or "0" is off, anything else on (the repo's getenv idiom).
fn wideFlag(v: ?[]const u8) bool {
    return if (v) |s| !std.mem.eql(u8, s, "0") else false;
}
/// Whether the wide arm runs: TF_FLASHNEXT_X3_WIDE_MEMBERS, off by default (the staged A tile arm, unmeasured).
pub fn wideMembers() bool {
    const v: ?[]const u8 = if (std.c.getenv("TF_FLASHNEXT_X3_WIDE_MEMBERS")) |e| std.mem.span(e) else null;
    return wideFlag(v);
}

/// Whether the grouped GEMM runs the decode ablation (TF_EXL3_ABL_DECODE): a timing probe, off by default.
pub fn ablateDecode() bool {
    const v: ?[]const u8 = if (std.c.getenv("TF_EXL3_ABL_DECODE")) |e| std.mem.span(e) else null;
    return wideFlag(v);
}

/// Whether the staged gm arm runs: TF_FLASHNEXT_X3_GM, on by default (one decode a weight tile for 128 member rows); 0 turns it off.
pub fn gmMembers() bool {
    const v: ?[]const u8 = if (std.c.getenv("TF_FLASHNEXT_X3_GM")) |e| std.mem.span(e) else null;
    return if (v == null) true else wideFlag(v);
}

/// The gm arm's setting: NT 8 / W 8 / KS 2 / NSA 4, and SUBW 4 (the shipped warp count, so its 4 k sub-ranges).
pub const gm_nt: usize = 8;
/// Picks below this (decode, verify, short prompts) skip the gm arm.
pub const gm_min_picks: usize = 1024;
pub const gm_warps: usize = 8;
pub const gm_ks: usize = 2;
pub const gm_nsa: usize = 4;
pub const gm_subw: usize = 4;

/// The gm arm's dynamic shared: NSA stages of 128 rows of the stage's KS k tiles (swizzled); run is in registers.
pub fn gmStageSmem(ks: usize) usize {
    return gm_nsa * (128 * ks * 16 * 2);
}
/// The gm arm's dynamic shared a block: KS 2 stages of 128 rows.
pub fn gmSmem() usize {
    return gmStageSmem(gm_ks);
}
/// The gm arm runs where the shipped arm uses SUBW 4 warps and its sub-range per_split/SUBW is a whole KS-tile count.
pub fn gmShapeOk(kk: usize, n: usize, sk: usize, shipped_w: usize) bool {
    if (shipped_w != gm_subw or sk == 0 or n % (16 * gm_nt) != 0 or kk % 16 != 0) return false;
    if (kk / 16 % sk != 0) return false;
    const per_split = kk / 16 / sk;
    if (per_split % gm_subw != 0) return false;
    return (per_split / gm_subw) % gm_ks == 0;
}

/// The grouped expert GEMM: 16 members a program, or 32 in the wide arm (tf_exl3x, opt-in and off by default).
pub fn grouped(k: *const Kernels, setting: usize, range: usize, x0: u64, x1: u64, tp0: u64, tp1: u64, k2_0: u64, k2_1: u64, uids: u64, ucount: u64, members: u64, z: u64, kk: usize, n: usize, p: usize, sk: usize, maxm: usize, slots: usize, experts: usize, mats: usize, s: cuda.Stream) !void {
    const t = tile_settings[setting];
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
    // The staged gm arm (opt-in): member rows the B operand, decoded weights the A operand; wins over the wide arm.
    // prompt-sized calls only: decode / verify picks keep the MTL 1 arm (the gm arm is 35% slower there)
    const gm = p >= gm_min_picks and gmMembers() and gmShapeOk(kk, n, sk, t.w) and gmSmem() <= k.gm_smem_max;
    if (gm) {
        const f = k.grouped_gm[range] orelse return error.NoGmKernel;
        const mtiles = (maxm + 127) / 128;
        const g = groupedGrid(experts, n, gm_nt, mats, sk, mtiles);
        return cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(g[0]), .y = @intCast(g[1]), .z = @intCast(g[2]) }, .block = .{ .x = @intCast(gm_warps * 32) }, .shared = @intCast(gmSmem()) }, s, &a);
    }
    const wide = wideMembers();
    const abl = !wide and ablateDecode();
    // the staged arm runs only where its A tile fits the device opt-in; a wider split (sk 1 or 2) falls back to MTL 1
    const stage = wideStageSmem(kk / 16 / sk, t.w);
    const staged = wide and stage <= k.wide_smem_max;
    const mtl: usize = if (staged) wide_member_tiles else 1;
    // grouped_kernel (MTL 1) loops over member groups past grouped_grid_members, so its grid keeps the 512-row
    // shape when the capacity grows to the window's rows; the wide and ablation arms still launch every group.
    const cap = if (!staged and !abl) @min(maxm, grouped_grid_members) else maxm;
    const mtiles = (cap + 16 * mtl - 1) / (16 * mtl);
    const g = groupedGrid(experts, n, t.nt, mats, sk, mtiles);
    const f = if (staged) (k.grouped_wide[setting][range] orelse return error.NoWideKernel) else if (abl) (k.grouped_abl[setting][range] orelse return error.NoAblKernel) else k.grouped[setting][range];
    const sh: u32 = if (staged) @intCast(stage) else 0;
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(g[0]), .y = @intCast(g[1]), .z = @intCast(g[2]) }, .block = .{ .x = @intCast(t.w * 32) }, .shared = sh }, s, &a);
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

test "the wide member-pass arm is off unless the switch is set" {
    try testing.expect(!wideFlag(null)); // unset: the 16-member path
    try testing.expect(!wideFlag("0")); // the repo's explicit off
    try testing.expect(wideFlag("1"));
    try testing.expect(wideFlag("anything"));
    try testing.expectEqual(@as(usize, 2), wide_member_tiles);
    // maxm 200: the 16-member path runs 13 member-tile programs, the wide arm 7 groups of 32 rows
    try testing.expectEqual(@as(usize, 13), (200 + 15) / 16);
    try testing.expectEqual(@as(usize, 7), (200 + 16 * 2 - 1) / (16 * 2));
}

test "the staged A tile is sized from the split and padded off the row banks" {
    // moe gate/up: K 2560, sk 4 -> per_split 40 -> 320 words a row, +4 so a lane's four words take 32 banks
    try testing.expectEqual(@as(usize, 49664), wideStageSmem(40, 4));
    // moe down: K 640, sk 2 -> per_split 20 -> 160 words a row
    try testing.expectEqual(@as(usize, 29184), wideStageSmem(20, 4));
    // the row stride is 4 mod 32 for the served split (the pad is address space, never read)
    try testing.expectEqual(@as(usize, 4), (40 * 8 + ((4 + 32 - (40 * 8 & 31)) & 31)) % 32);
}

test "the decode ablation shares the wide arm's opt-in reading" {
    try testing.expect(!wideFlag(null) and !wideFlag("0")); // unset / "0": grouped_kernel, the default path
    try testing.expect(wideFlag("1")); // set: grouped_abl_kernel, the timing probe
}

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

fn groupSerial(pick: []const i32, slots: usize, experts: usize, maxm: usize, uids: []i32, members: []i32) usize {
    var u: usize = 0;
    for (0..experts) |e| {
        const id: i32 = @intCast(e);
        var c: usize = 0;
        for (pick) |p| c += @intFromBool(p == id);
        if (c == 0) continue;
        uids[u] = id;
        var j: usize = 0;
        for (pick, 0..) |p, i| {
            if (j >= maxm) break;
            if (p != id) continue;
            members[u * maxm + j] = @intCast((i / slots) * 32 + (i % slots));
            j += 1;
        }
        while (j < maxm) : (j += 1) members[u * maxm + j] = -1;
        u += 1;
    }
    return u;
}

test "grouped expert rows stay in pick order across tiles, with the maxm pad" {
    const experts: usize = 8;
    const slots: usize = 5;
    const maxm: usize = 3;
    var pick: [2500]i32 = undefined;
    for (&pick, 0..) |*p, i| {
        p.* = switch (i % 17) {
            0 => -1,
            1 => 99,
            else => @intCast((i * 3) % experts),
        };
    }
    pick[10] = 4;
    pick[2000] = 4;
    for (&pick, 0..) |*p, i| {
        if (p.* == 4 and i != 10 and i != 2000) p.* = 1;
    }
    var uids_s: [8]i32 = undefined;
    var uids_p: [8]i32 = undefined;
    var mem_s: [8 * 3]i32 = undefined;
    var mem_p: [8 * 3]i32 = undefined;
    var scratch: [64]i32 = undefined;
    try std.testing.expect(scratch.len >= groupWorkspaceInts(pick.len, experts) + experts);
    const us = groupSerial(&pick, slots, experts, maxm, &uids_s, &mem_s);
    const up = groupPlan(&pick, slots, experts, maxm, &uids_p, &mem_p, &scratch);
    try std.testing.expectEqual(us, up);
    try std.testing.expectEqualSlices(i32, uids_s[0..us], uids_p[0..up]);
    try std.testing.expectEqualSlices(i32, mem_s[0 .. us * maxm], mem_p[0 .. up * maxm]);
    var prev: i32 = -1;
    var pad = false;
    for (uids_p[0..up], 0..) |id, ui| {
        try std.testing.expect(id > prev);
        prev = id;
        if (id == 4) {
            try std.testing.expectEqual(@as(i32, -1), mem_p[ui * maxm + 2]);
            try std.testing.expectEqual(@as(i32, (10 / slots) * 32 + (10 % slots)), mem_p[ui * maxm]);
            try std.testing.expectEqual(@as(i32, (2000 / slots) * 32 + (2000 % slots)), mem_p[ui * maxm + 1]);
            pad = true;
        }
    }
    try std.testing.expect(pad);
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
    try testing.expectEqual([2]usize{ 160, 5 }, epilogueGrid(128, 5, 640)); // ceil(rows*slots / 4 warps), width / 128
    try testing.expectEqual(@as(usize, 128), epilogueBlock(128, 5));
    try testing.expectEqual([2]usize{ 30, 5 }, epilogueGrid(3, 10, 640)); // under 256 picks: one warp a pick
    try testing.expectEqual(@as(usize, 32), epilogueBlock(3, 10));
    try testing.expectEqual([2]usize{ 1, 1 }, combineGrid(1, 256));
    try testing.expectEqual([2]usize{ 1, 2 }, combineGrid(1, 300));
    try testing.expectEqual([2]usize{ 1, 5 }, downCombineGrid(1, 640));
    try testing.expectEqual(@as(usize, 160), downCombineBlock(5));
    try testing.expectEqual([3]usize{ 8, 5, 40 }, groupedGrid(8, 640, 8, 2, 4, 5)); // experts, N/(16*nt), mats*sk*mtiles
    try testing.expectEqual([2]usize{ 160, 40 }, dequantGrid(2560, 640));
    try testing.expectEqual(@as(usize, 1), groupTiles(1));
    try testing.expectEqual(@as(usize, 2), groupTiles(group_tile + 1));
    // 2048 rows, 11 slots is 22 tiles of 1024: hist plus totals and place
    try testing.expectEqual(@as(usize, 22 * 512 + 2 * 512), groupWorkspaceInts(2048 * 11, 512));
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
    for ([_][:0]const u8{ sym.rot_in, sym.group_hist, sym.group_scan, sym.group_scatter, sym.group_small, sym.gateup_epilogue, sym.down_epilogue, sym.combine, sym.down_combine }) |s| {
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

test "tile keeps the setting it read" {
    const gu = try tile(2560, 640, true);
    try testing.expectEqual(gu, try tile(2560, 640, true));
    const dn = try tile(640, 2560, false);
    try testing.expectEqual(dn, try tile(640, 2560, false));
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

test "the tile sweep names a variant the fatbin carries, and never falls back silently" {
    // unset or blank: defaultTile's shape walk, exactly today's choice
    try testing.expectEqual(@as(?Variant, null), try tileVariant(null, true));
    try testing.expectEqual(@as(?Variant, null), try tileVariant("", false));
    // every row resolves to a setting tile_settings lists, so the fatbin carries its (nt, w, pf) constexprs
    for (variants) |t| try testing.expect(tileIndex(t.nt, t.w, t.pf) != null);
    // one id names both classes, "gu,dn" names each, and a trailing field is ignored past the second
    try testing.expectEqual(v_glm_gu, (try tileVariant("0", true)).?);
    try testing.expectEqual(v_glm_gu, (try tileVariant("0", false)).?);
    try testing.expectEqual(v_glm_gu, (try tileVariant("0,1", true)).?);
    try testing.expectEqual(v_glm_dn, (try tileVariant("0,1", false)).?);
    try testing.expectEqual(v_nt4_pf1, (try tileVariant("7,7", true)).?);
    // an id past the table is loud, never a silent fall back onto the defaults
    try testing.expectError(error.NoTileVariant, tileVariant("8", true));
    try testing.expectError(error.NoTileVariant, tileVariant("x", false));
    // the defaults the switch reproduces: gu -> GLM's gate/up tile, dn -> GLM's down tile, for the pack's shapes
    try testing.expectEqual(glm_gateup, try variantTile(v_glm_gu, 2560, 640));
    try testing.expectEqual(glm_down, try variantTile(v_glm_dn, 640, 2560));
    // a variant that does not divide the shape (640/16 = 40, not a multiple of sk*w = 16) is loud, not a fall back
    try testing.expectError(error.NoTileShape, variantTile(v_nt4_pf1, 640, 512));
    try testing.expectEqual(@as(usize, 4), (try variantTile(v_glm_gu, 2560, 512)).sk);
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
