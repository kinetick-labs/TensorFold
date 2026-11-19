//! The Hy4 vision tower on CUDA, one picture at a time (vision_ref.zig's math): patch embedding plus the resampled
//! learned positions, pre-norm LayerNorm blocks with full attention over the picture and an erf-GELU MLP, then the
//! merger (RMSNorm, the 2 x 2 stride-2 convolution as one matmul over gathered cells, GELU, the 1 x 1 projection to
//! the text width) and `image_newline` after every merged row. fp32 activations, weights as stored.

const std = @import("std");
const Allocator = std.mem.Allocator;
const cuda = @import("cuda");
const fwd = @import("fwd.zig");
const image_grid = @import("image_grid.zig");

pub const W = fwd.W;

/// A plain biased projection.
pub const Proj = struct { w: W, b: W, k: u32, n: u32 };

pub const Block = struct { norm1_w: W, norm1_b: W, q: Proj, k: Proj, v: Proj, o: Proj, norm2_w: W, norm2_b: W, fc1: Proj, fc2: Proj };

pub const Tower = struct {
    hidden: u32,
    inter: u32,
    heads: u32,
    patch: u32,
    merge: u32 = 2,
    out: u32,
    edge: u32,
    eps: f32 = 1e-5,
    patch_proj: Proj, // [hidden, 3 * patch * patch]
    /// the learned grid on the host, f32 `[edge * edge, hidden]` (the checkpoint's row 0, the unused cls slot, dropped)
    positions: []const f32,
    blocks: []const Block,
    merge_norm: W,
    conv: Proj, // [hidden * merge^2, hidden * merge^2] in the cell order merge_gather writes
    out_proj: Proj, // [out, hidden * merge^2]
    newline: W,
};

fn f4(n: u64) u64 {
    return n * 4;
}

fn gemm(k: *const fwd.Kernels, s: cuda.Stream, p: Proj, x: u64, rows: u32, y: u64) !void {
    try k.gemm(s, x, p.k, p.w, p.b, y, p.n, rows, p.n, p.k, false);
}

/// Scratch for pictures up to `max_patches` patches.
pub const Work = struct {
    buf: cuda.DeviceBuffer,
    max_patches: u32,
    pixels: u64,
    x: u64,
    nb: u64,
    q: u64,
    k: u64,
    v: u64,
    att: u64,
    mid: u64,
    cell: u64,
    conv: u64,
    merged: u64,

    pub fn init(d: *const cuda.Driver, t: *const Tower, max_patches: u32) !Work {
        const P: u64 = max_patches;
        const h: u64 = t.hidden;
        const pp: u64 = 3 * @as(u64, t.patch) * t.patch;
        const cells = P / (t.merge * t.merge);
        const wide = h * t.merge * t.merge;
        const sizes = [_]u64{ P * pp, P * h, P * h, P * h, P * h, P * h, P * h, P * t.inter, cells * wide, cells * wide, cells * t.out };
        var total: u64 = 0;
        for (sizes) |n| total += std.mem.alignForward(u64, f4(n), 256);
        var w: Work = undefined;
        w.buf = try cuda.DeviceBuffer.alloc(d, total);
        w.max_patches = max_patches;
        var at = w.buf.ptr;
        const ptrs = [_]*u64{ &w.pixels, &w.x, &w.nb, &w.q, &w.k, &w.v, &w.att, &w.mid, &w.cell, &w.conv, &w.merged };
        for (ptrs, sizes) |p, n| {
            p.* = at;
            at += std.mem.alignForward(u64, f4(n), 256);
        }
        return w;
    }

    pub fn deinit(w: *Work) void {
        w.buf.free();
    }
};

/// The picture's merged rows, newline after each merged row, into `out` (`[g.tokens(merge), t.out]` f32 on the
/// device); `pixels` are the host patches (`[patches, 3 * patch * patch]`, image_grid.patchify order).
pub fn forward(a: Allocator, c: anytype, t: *const Tower, w: *Work, pixels: []const f32, g: image_grid.Grid, out: u64) !void {
    const k: *const fwd.Kernels = c.k;
    const s: cuda.Stream = c.s;
    const P: u32 = @intCast(g.patches());
    if (P > w.max_patches) return error.Hy4ImageTooLarge;
    const h = t.hidden;
    const pp = 3 * t.patch * t.patch;
    if (pixels.len != @as(usize, P) * pp) return error.Hy4Shape;
    try c.upload(w.pixels, std.mem.sliceAsBytes(pixels));
    try gemm(k, s, t.patch_proj, w.pixels, P, w.x);
    {
        const pos = try a.alloc(f32, @as(usize, P) * h);
        defer a.free(pos);
        image_grid.resamplePositions(t.positions, t.edge, h, g, pos);
        try c.upload(w.nb, std.mem.sliceAsBytes(pos));
        try k.add(s, w.x, w.nb, @as(u64, P) * h);
        // the host buffer must outlive the copy
        try s.synchronize();
    }
    const hd = h / t.heads;
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
    for (t.blocks) |*b| {
        try k.layernorm(s, w.x, b.norm1_w, b.norm1_b, w.nb, P, h, t.eps);
        try gemm(k, s, b.q, w.nb, P, w.q);
        try gemm(k, s, b.k, w.nb, P, w.k);
        try gemm(k, s, b.v, w.nb, P, w.v);
        try k.vitAttn(s, w.q, w.k, w.v, w.att, h, P, t.heads, hd, scale);
        try gemm(k, s, b.o, w.att, P, w.nb);
        try k.add(s, w.x, w.nb, @as(u64, P) * h);
        try k.layernorm(s, w.x, b.norm2_w, b.norm2_b, w.nb, P, h, t.eps);
        try gemm(k, s, b.fc1, w.nb, P, w.mid);
        try k.gelu(s, w.mid, @as(u64, P) * t.inter);
        try gemm(k, s, b.fc2, w.mid, P, w.nb);
        try k.add(s, w.x, w.nb, @as(u64, P) * h);
    }
    try k.rmsnorm(s, w.x, h, t.merge_norm, 1, w.nb, h, P, h, t.eps);
    if (t.merge != 2) return error.Hy4Shape;
    const gh2: u32 = @intCast(g.h / 2);
    const gw2: u32 = @intCast(g.w / 2);
    const cells = gh2 * gw2;
    try k.mergeGather(s, w.nb, w.cell, gh2, @intCast(g.w), gw2, h);
    try gemm(k, s, t.conv, w.cell, cells, w.conv);
    try k.gelu(s, w.conv, @as(u64, cells) * t.conv.n);
    try gemm(k, s, t.out_proj, w.conv, cells, w.merged);
    try k.newlines(s, w.merged, t.newline, out, gh2, gw2, t.out);
}
