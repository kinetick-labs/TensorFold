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

    pub fn load(d: *const cuda.Driver) !Exl3Run {
        return .{ .dense = try cuda.exl3.Dense.load(d) };
    }

    pub fn unload(r: *Exl3Run) void {
        if (r.tmp_in) |*b| b.free();
        if (r.tmp_out) |*b| b.free();
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
    }

    /// x [m][k] (rows `x_stride` apart, bf16) through `e` into out [m][e.n]: bf16, or fp32 with `wide`.
    pub fn run(r: *Exl3Run, s: cuda.Stream, e: Exl3, x: u64, x_stride: usize, out: u64, m: usize, wide: bool) !void {
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
