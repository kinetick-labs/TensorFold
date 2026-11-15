//! A GLM-5.3-Flash projection as its checkpoint stores it: bf16 through the glue's matmuls, FP8 kinds through qlinear.

const std = @import("std");
const cuda = @import("cuda");
const glue = @import("cuda_glue.zig");
const b16 = @import("cuda_b16.zig");

/// A rank's projection: bf16 rows for the glue, FP8_PB_WO's block FP8 or MXFP8 packed for the lane as stored, or
/// EXL3 trellises (a REXL3/SAGE pack: one to three stacked modules, each its own width, outputs side by side).
pub const Proj = union(enum) { bf16: u64, fp8g: cuda.qmmf.Weight, mxfp8: cuda.qmmf.Weight, exl3: Exl3 };

/// Up to three EXL3 modules of one input stacked by outputs ([q_a | kv_a], [gate | up]): `n` the outputs together.
pub const Exl3 = struct {
    ws: [3]cuda.exl3.Weight = undefined,
    len: u8 = 0,
    n: usize = 0,

    pub fn k(e: Exl3) usize {
        return e.ws[0].k;
    }
    pub fn members(e: *const Exl3) []const cuda.exl3.Weight {
        return e.ws[0..e.len];
    }
};

/// One stream's EXL3 projections: the common dense kernels' scratch, a packed copy of strided inputs, a member's rows.
/// Prompt chunks and decode windows each hold their own (glue.Set.exl3), so two streams never share scratch.
pub const Exl3Run = struct {
    dense: cuda.exl3.Dense,
    tmp_in: ?cuda.DeviceBuffer = null,
    tmp_out: ?cuda.DeviceBuffer = null,
    /// Prompt chunks' fused GEMM (null: not loaded, or TF_GLM_PFD=0), its rotated rows and a packed copy of strided ones.
    pfd: ?Pfd = null,
    pf_xh: ?cuda.DeviceBuffer = null,
    pf_in: ?cuda.DeviceBuffer = null,
    /// The rows a prompt chunk holds (the prompt stream's runner): `reserve` grows the fused GEMM's rows to it.
    prompt_rows: usize = 0,

    pub fn load(d: *const cuda.Driver) !Exl3Run {
        var r: Exl3Run = .{ .dense = try cuda.exl3.Dense.load(d) };
        if (Pfd.wanted()) r.pfd = Pfd.load(d) catch |err| blk: {
            std.log.warn("glm: the fused prompt GEMM for EXL3 projections did not load ({s}); prompt chunks take 128-row blocks", .{@errorName(err)});
            break :blk null;
        };
        return r;
    }

    pub fn unload(r: *Exl3Run) void {
        if (r.tmp_in) |*b| b.free();
        if (r.tmp_out) |*b| b.free();
        if (r.pf_xh) |*b| b.free();
        if (r.pf_in) |*b| b.free();
        if (r.pfd) |*p| p.unload();
        r.dense.unload();
    }

    fn grown(r: *Exl3Run, slot: *?cuda.DeviceBuffer, bytes: usize) !u64 {
        if (slot.* != null and slot.*.?.len >= bytes) return slot.*.?.ptr;
        const b = try cuda.DeviceBuffer.alloc(r.dense.d, bytes);
        if (slot.*) |*old| old.free();
        slot.* = b;
        return b.ptr;
    }

    /// Every buffer `p`'s calls take, grown now, so a call inside a graph capture allocates nothing.
    pub fn reserve(r: *Exl3Run, s: cuda.Stream, p: Proj) !void {
        const e = switch (p) {
            .exl3 => |e| e,
            else => return,
        };
        var widest: usize = 0;
        for (e.members()) |w| {
            try r.dense.reserve(s, w);
            widest = @max(widest, w.n);
        }
        _ = try r.grown(&r.tmp_in, block_rows * e.k() * 2);
        if (e.len > 1) _ = try r.grown(&r.tmp_out, block_rows * widest * 4);
        if (r.pfd != null and r.prompt_rows >= Pfd.min_rows and Pfd.covers(e)) _ = try r.grown(&r.pf_xh, r.prompt_rows * e.k() * 2);
    }

    fn fused(r: *Exl3Run, s: cuda.Stream, e: Exl3, x: u64, x_stride: usize, out: u64, m: usize, wide: bool) !void {
        return fusedRun(r, s, e, x, x_stride, out, m, wide);
    }

    /// x [m][k] (rows `x_stride` apart, bf16) through `e` into out [m][e.n]: bf16, or fp32 with `wide`.
    pub fn run(r: *Exl3Run, s: cuda.Stream, e: Exl3, x: u64, x_stride: usize, out: u64, m: usize, wide: bool) !void {
        if (m >= Pfd.min_rows and r.pfd != null and Pfd.covers(e)) return r.fused(s, e, x, x_stride, out, m, wide);
        const k = e.k();
        const esz: usize = if (wide) 4 else 2;
        const ydt: cuda.exl3.DType = if (wide) .f32 else .bf16;
        var at: usize = 0;
        while (at < m) : (at += block_rows) {
            const mb = @min(block_rows, m - at);
            var xin = x + at * x_stride * 2;
            if (x_stride != k and mb > 1) { // the kernels read packed rows: copy the block's rows together first
                const t = try r.grown(&r.tmp_in, block_rows * k * 2);
                try copy2d(s, t, k * 2, xin, x_stride * 2, k * 2, mb, "EXL3 packed input rows");
                xin = t;
            }
            const y = out + at * e.n * esz;
            if (e.len == 1) {
                try r.dense.decode(s, e.ws[0], xin, .bf16, mb, y, ydt);
                continue;
            }
            var off: usize = 0;
            for (e.members()) |w| {
                if (mb == 1) {
                    try r.dense.decode(s, w, xin, .bf16, 1, y + off * esz, ydt);
                } else { // a member's rows packed, then into its columns of every row
                    const t = try r.grown(&r.tmp_out, block_rows * w.n * 4);
                    try r.dense.decode(s, w, xin, .bf16, mb, t, ydt);
                    try copy2d(s, y + off * esz, e.n * esz, t, w.n * esz, w.n * esz, mb, "EXL3 member columns");
                }
                off += w.n;
            }
        }
    }
};

/// Rows a call to the dense kernels takes at once (their 1..128 contract); larger counts run block by block.
pub const block_rows = 128;

/// A prompt chunk's rows through `e` on the fused GEMM: each member's input rotated once for all `m` rows
/// (tf_exl3_rot_in), then pfdense into the member's columns of `out` (row stride e.n), the trellis decoded once
/// per row tile instead of once per 128-row block's 16- or 32-row group.
fn fusedRun(r: *Exl3Run, s: cuda.Stream, e: Exl3, x: u64, x_stride: usize, out: u64, m: usize, wide: bool) !void {
    const p = &r.pfd.?;
    const k = e.k();
    var xin = x;
    if (x_stride != k) {
        const t = try r.grown(&r.pf_in, m * k * 2);
        try copy2d(s, t, k * 2, x, x_stride * 2, k * 2, m, "EXL3 packed prompt rows");
        xin = t;
    }
    const xh = try r.grown(&r.pf_xh, m * k * 2);
    const esz: usize = if (wide) 4 else 2;
    var off: usize = 0;
    var rotated: u64 = 0; // the suh the rows in xh carry (members of one pack may share it)
    for (e.members()) |w| {
        if (rotated != w.suh) {
            var ra: cuda.Args = .{};
            ra.add(xin);
            ra.add(w.suh);
            ra.add(xh);
            ra.add(@as(i32, @backingInt(cuda.exl3.DType.bf16)));
            ra.add(@as(i32, @intCast(m)));
            ra.add(@as(i32, @intCast(k)));
            try cuda.launch.launch(r.dense.lin.rot_in, .{
                .grid = .{ .x = @intCast((k / 128 + 3) / 4), .y = @intCast(m), .z = 1 },
                .block = .{ .x = 128, .y = 1, .z = 1 },
                .shared = 0,
                .pdl = r.dense.lin.pdl,
            }, s, &ra);
            rotated = w.suh;
        }
        try p.gemm(s, w, xh, out + off * esz, e.n, m, wide);
        off += w.n;
    }
}

/// DeepSeek V4.1's pfdense (glm_pfdense.cu, strips only): out [M, N] = H(xh [M, K] @ W) * svh (+ bias), the
/// trellis decoded inside a tensor-core GEMM, at the mul1 widths 8-16 of a REXL3 pack's dense projections.
pub const Pfd = struct {
    mod: cuda.Module,
    fns: [k2s.len][cfgs.len]cuda.Function,
    sms: usize,
    forced: ?usize, // TF_GLM_PFD_CFG: one configuration for every call
    /// The MLA layers' absorb_q / expand_v of prompt chunks on tensor cores (glm_pf_mla.cuh); null: TF_GLM_PF_MLA=0.
    mla: ?[2]cuda.Function = null,

    pub const k2s = [_]u32{ 8, 10, 12, 14, 16 };
    /// glm_pfdense.cu's (BM, WM, WN, KS, NST): pfdense_args.h's configurations 0, 3 and 6, the heuristic's choices.
    pub const cfgs = [_][5]u32{ .{ 128, 2, 4, 2, 3 }, .{ 64, 2, 2, 2, 4 }, .{ 32, 1, 4, 2, 4 } };
    /// The fewest rows a call takes the fused GEMM at (prompt chunks; decode windows stay on the row kernels).
    pub const min_rows = 256;

    pub const Args = extern struct {
        xh: u64 = 0,
        T: u64 = 0,
        stride_k: i64 = 0,
        stride_nb: i64 = 0,
        svh: u64 = 0,
        bias: u64 = 0,
        out: u64 = 0,
        o_stride: i64 = 0,
        M: c_int = 0,
        K: c_int = 0,
        N: c_int = 0,
        group: c_int = 0,
        out_type: c_int = 0,
    };
    comptime {
        std.debug.assert(@offsetOf(Args, "M") == 64 and @sizeOf(Args) == 88);
    }

    pub fn wanted() bool {
        const v = std.c.getenv("TF_GLM_PFD") orelse return true;
        return !std.mem.eql(u8, std.mem.span(v), "0");
    }

    pub fn symbol(comptime k2: u32, comptime cfg: usize) [:0]const u8 {
        const c = cfgs[cfg];
        return std.fmt.comptimePrint("_ZN9dsv41_pfd10pfd_kernelILi{d}ELb0ENS_3CfgILi{d}ELi{d}ELi{d}ELi{d}ELi{d}EEEEEvNS_4ArgsE", .{ k2, c[0], c[1], c[2], c[3], c[4] });
    }

    /// Lay<K2, false, Cfg>::SMEM.
    pub fn smem(cfg: usize, k2: u32) u32 {
        const bm, const wm, _, const ks, const nst = cfgs[cfg];
        const ring = nst * (bm * (16 * ks) * 2 + ks * 8 * (4 * k2) * 4);
        return @max(ring + 2 * (ks * 8 * 32 * 16), (bm / wm) * 128 * 2 * 2);
    }

    pub fn threads(cfg: usize) u32 {
        return cfgs[cfg][1] * cfgs[cfg][2] * 32;
    }

    pub fn load(d: *const cuda.Driver) !Pfd {
        if (!cuda.kernels.available or cuda.kernels.glm_pfdense.len == 0) return error.BuiltWithoutKernels;
        var p: Pfd = .{ .mod = try cuda.Module.load(d, cuda.kernels.glm_pfdense), .fns = undefined, .sms = 48, .forced = null };
        errdefer p.mod.unload();
        inline for (k2s, 0..) |k2, ki| inline for (0..cfgs.len) |c| {
            const f = try p.mod.function(symbol(k2, c));
            try f.allowDynamicShared(smem(c, k2));
            p.fns[ki][c] = f;
        };
        const mla_on = if (std.c.getenv("TF_GLM_PF_MLA")) |v| !std.mem.eql(u8, std.mem.span(v), "0") else true;
        if (mla_on) p.mla = .{ try p.mod.function("tf_glm_pf_absorb"), try p.mod.function("tf_glm_pf_expand") };
        if (std.c.getenv("TF_GLM_PFD_SMS")) |v| p.sms = std.fmt.parseInt(usize, std.mem.span(v), 10) catch p.sms;
        if (std.c.getenv("TF_GLM_PFD_CFG")) |v| p.forced = std.fmt.parseInt(usize, std.mem.span(v), 10) catch null;
        if (p.forced) |c| if (c >= cfgs.len) {
            p.forced = null;
        };
        return p;
    }

    /// A prompt chunk's absorb (which 0) or expand (1) on tensor cores: out [rows][heads][n] = in [rows][heads][k] . w
    /// per head (absorb: wk [heads][256][512]; expand: wv [heads][256][512] transposed). False: not taken here.
    pub fn latent(p: *const Pfd, s: cuda.Stream, which: usize, in: u64, w: u64, out: u64, rows: usize, heads: usize) !bool {
        const fns = p.mla orelse return false;
        if (rows < min_rows) return false;
        const n: usize = if (which == 0) 512 else 256;
        var a: cuda.Args = .{};
        inline for (.{ in, w, out, @as(i64, @intCast(rows)), @as(i64, @intCast(heads)) }) |v| a.add(v);
        try cuda.launch.launch(fns[which], .{ .grid = .{ .x = @intCast(n / 64), .y = @intCast((rows + 63) / 64), .z = @intCast(heads) }, .block = .{ .x = 128 } }, s, &a);
        return true;
    }

    pub fn unload(p: *Pfd) void {
        p.mod.unload();
    }

    fn widthIndex(w: cuda.exl3.Weight) ?usize {
        if (w.codebook != .mul1 or w.halves % 2 != 0) return null;
        return std.mem.indexOfScalar(u32, &k2s, w.halves);
    }

    /// Whether every member of `e` fits the kernels: mul1 at an even width 8-16, k % 128 and n % 128, and strides
    /// that place tile (kt, nt) at kt * stride_k + (nt / 8) * stride_nb + (nt % 8) * 4 K2.
    pub fn covers(e: Exl3) bool {
        for (e.members()) |w| {
            if (widthIndex(w) == null) return false;
            if (w.k % 128 != 0 or w.k < 128 or w.n % 128 != 0 or w.n < 128) return false;
            const tw: u64 = 4 * @as(u64, w.halves);
            if (w.stride_k % 4 != 0 or w.stride_nb % 4 != 0 or w.stride_k < 8 * tw or w.stride_nb < 8 * tw) return false;
        }
        return e.len > 0;
    }

    /// pfdense.heuristic: the largest row tile whose last wave is not mostly empty.
    pub fn pick(p: *const Pfd, n: usize, k2: u32, m: usize) usize {
        if (p.forced) |c| return c;
        const nb = n / 128;
        var best: usize = 0;
        var best_eff: f64 = -1.0;
        for (0..cfgs.len) |cfg| {
            const bm = cfgs[cfg][0];
            const blocks: usize = if (2 * (smem(cfg, k2) + 1024) <= 102400) 2 else 1;
            const ctas = ((m + bm - 1) / bm) * nb;
            const slots = p.sms * blocks;
            const eff = @as(f64, @floatFromInt(ctas)) / @as(f64, @floatFromInt(((ctas + slots - 1) / slots) * slots));
            if (eff > best_eff + 0.10) {
                best = cfg;
                best_eff = eff;
            }
        }
        return best;
    }

    /// out [m][w.n] (rows `o_stride` apart) = the member's GEMM of the rotated rows xh [m][w.k] (fp16).
    pub fn gemm(p: *const Pfd, s: cuda.Stream, w: cuda.exl3.Weight, xh: u64, out: u64, o_stride: usize, m: usize, wide: bool) !void {
        const ki = widthIndex(w) orelse return error.UnsupportedExl3Width;
        const cfg = p.pick(w.n, w.halves, m);
        const a: Args = .{
            .xh = xh,
            .T = w.trellis,
            .stride_k = @intCast(w.stride_k),
            .stride_nb = @intCast(w.stride_nb),
            .svh = w.svh,
            .bias = w.bias,
            .out = out,
            .o_stride = @intCast(o_stride),
            .M = @intCast(m),
            .K = @intCast(w.k),
            .N = @intCast(w.n),
            .group = 8,
            .out_type = if (wide) 2 else 1,
        };
        var args: cuda.Args = .{};
        args.add(a);
        const bm = cfgs[cfg][0];
        try cuda.launch.launch(p.fns[ki][cfg], .{
            .grid = .{ .x = @intCast(((m + bm - 1) / bm) * (w.n / 128)), .y = 1, .z = 1 },
            .block = .{ .x = threads(cfg), .y = 1, .z = 1 },
            .shared = smem(cfg, w.halves),
        }, s, &args);
    }
};

test "the fused prompt GEMM's shared memory, symbols and heuristic match DeepSeek V4.1's pfdense" {
    try std.testing.expectEqual(@as(u32, 3 * 10752 + 16384), Pfd.smem(0, 10));
    try std.testing.expectEqual(@as(u32, 256), Pfd.threads(0));
    try std.testing.expectEqualStrings("_ZN9dsv41_pfd10pfd_kernelILi14ELb0ENS_3CfgILi128ELi2ELi4ELi2ELi3EEEEEvNS_4ArgsE", Pfd.symbol(14, 0));
    const p: Pfd = .{ .mod = undefined, .fns = undefined, .sms = 48, .forced = null };
    try std.testing.expectEqual(@as(usize, 0), p.pick(4096, 12, 2048));
    var e: Exl3 = .{};
    const st = cuda.exl3.strides(4096, 1024, 12, false);
    e.ws[0] = .{ .trellis = 0, .suh = 0, .svh = 0, .k = 4096, .n = 1024, .halves = 12, .codebook = .mul1, .stride_k = st.k, .stride_nb = st.nb };
    e.len = 1;
    e.n = 1024;
    try std.testing.expect(Pfd.covers(e));
    e.ws[0].codebook = .mcg;
    try std.testing.expect(!Pfd.covers(e));
    e.ws[0].codebook = .mul1;
    e.ws[0].halves = 11;
    try std.testing.expect(!Pfd.covers(e));
}

fn copy2d(s: cuda.Stream, dst: u64, dpitch: usize, src: u64, spitch: usize, width: usize, height: usize, what: []const u8) !void {
    const m: cuda.abi.Memcpy2D = .{ .src = src, .src_pitch = spitch, .dst = dst, .dst_pitch = dpitch, .width = width, .height = height };
    try s.d.check(s.d.api.cuMemcpy2DAsync_v2(&m, s.handle), what);
}

/// x [m][k] (rows `x_stride` apart) by `w` into out [m][n]: bf16, or fp32 with `wide`; `part` the bf16 path's slices.
pub fn matmul(set: glue.Set, stream: cuda.Stream, w: Proj, x: u64, x_stride: usize, out: u64, part: u64, m: usize, n: usize, k: usize, wide: bool) !void {
    switch (w) {
        .bf16 => |p| return b16.matmul(set, stream, x, x_stride, p, out, part, m, n, k, wide),
        .fp8g, .mxfp8 => |q| {
            if (q.n != n or q.k != k) return error.ShapeMismatch;
            const lane = set.linear.lane orelse return error.FormatNotLoaded;
            // GB10's lane meets its K slices in a cluster; a GPU that sums them in memory would need partials held here
            if (lane.partBytes(m, q) != 0) return error.SlicePartialsNotHeld;
            const in: cuda.qlinear.Rows = .{ .x = x, .ldx = x_stride };
            const f: cuda.qlinear.Weight = if (w == .fp8g) .{ .fp8g = q } else .{ .mxfp8 = q };
            return if (wide) set.linear.decodeF32(stream, f, in, out, m) else set.linear.decode(stream, f, in, out, m);
        },
        .exl3 => |e| {
            if (e.n != n or e.k() != k) return error.ShapeMismatch;
            const r = set.exl3 orelse return error.FormatNotLoaded;
            return r.run(stream, e, x, x_stride, out, m, wide);
        },
    }
}

/// error.SlicePartialsNotHeld when `matmul` would refuse `w` on `lane` at some row count: the loader's check.
pub fn fits(lane: *const cuda.qmmf.Lane, w: cuda.qmmf.Weight) error{SlicePartialsNotHeld}!void {
    // one row stands for all: under qmmf.fused_rows the plan is the same for every row count, and over it none reduces
    if (lane.partBytes(1, w) != 0) return error.SlicePartialsNotHeld;
}

test "a block-FP8 projection is refused before any launch when the call's shape or the loaded kernels do not fit" {
    const q = cuda.fp8.weight(0, 0, 64, 4096);
    const s: cuda.Stream = undefined; // never reached: the refusals come first
    var set: glue.Set = .{ .glue = undefined };
    try std.testing.expectError(error.ShapeMismatch, matmul(set, s, .{ .fp8g = q }, 0, 4096, 0, 0, 1, 128, 4096, false));
    try std.testing.expectError(error.FormatNotLoaded, matmul(set, s, .{ .fp8g = q }, 0, 4096, 0, 0, 1, 64, 4096, true));
    var lane: cuda.qmmf.Lane = undefined;
    lane.clusters = false; // a GPU before sm_90: 64 outputs from 4,096 inputs take K slices summed in memory
    set.linear = .{ .lane = &lane };
    try std.testing.expectError(error.SlicePartialsNotHeld, matmul(set, s, .{ .fp8g = q }, 0, 4096, 0, 0, 1, 64, 4096, true));
    try std.testing.expectError(error.SlicePartialsNotHeld, matmul(set, s, .{ .mxfp8 = cuda.mxfp8.weight(0, 0, 64, 4096) }, 0, 4096, 0, 0, 1, 64, 4096, true));
}

test "the loader's check refuses an FP8 projection exactly when matmul would at some row count" {
    var flat: cuda.qmmf.Lane = undefined;
    flat.clusters = false; // a GPU before sm_90
    var gb10: cuda.qmmf.Lane = undefined;
    gb10.clusters = true;
    // FP4-Local's stored kda q, kv_a and MLA o, a 64-output cut, an lm_head share, and inputs too few for K slices
    const shapes = [_][2]usize{ .{ 8192, 4096 }, .{ 512, 4096 }, .{ 4096, 16384 }, .{ 64, 4096 }, .{ 77440, 4096 }, .{ 64, 512 } };
    for (shapes) |nk| for ([_]*const cuda.qmmf.Lane{ &flat, &gb10 }) |lane| {
        const q = cuda.fp8.weight(0, 0, nk[0], nk[1]);
        var refused = false; // matmul's own test, at decode rows, partial tiles, the fused rows and a prompt chunk
        for ([_]usize{ 1, 4, 16, 17, 64, 255, 256, 2048 }) |m| refused = refused or lane.partBytes(m, q) != 0;
        if (refused) try std.testing.expectError(error.SlicePartialsNotHeld, fits(lane, q)) else try fits(lane, q);
    };
    try std.testing.expectError(error.SlicePartialsNotHeld, fits(&flat, cuda.fp8.weight(0, 0, 8192, 4096)));
    try std.testing.expectError(error.SlicePartialsNotHeld, fits(&flat, cuda.mxfp8.weight(0, 0, 4096, 16384)));
    try fits(&flat, cuda.fp8.weight(0, 0, 77440, 4096));
    try fits(&gb10, cuda.mxfp8.weight(0, 0, 4096, 16384));
}
