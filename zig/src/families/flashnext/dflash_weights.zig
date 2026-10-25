//! The DFlash2 drafter's weights onto the device: projections quantized to affine-4 on the host, codebooks kept there
//! (qwen3_5's cuda_draft_weights.zig over the Flash Next drafter's engine), and the target's head packed the same way.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");

const Affine4 = cuda.qlinear.Affine4;
const fp8 = @import("cuda_fp8.zig");
const q4 = @import("cuda_mtp_q4.zig");

/// One drafter projection: affine-4 (qwen3_5's), or MXFP8 (TF_FLASHNEXT_DFLASH_MX=1: e4m3 a weight, a power-of-two
/// scale a 32 inputs, run by the engine's MXFP8 lane matmul); `a` carries the shape either way.
pub const Lin = struct {
    a: Affine4,
    mx: ?fp8.Linear = null,
};
const DeviceBuffer = cuda.DeviceBuffer;
fn aligned(n: usize) usize {
    return std.mem.alignForward(usize, n, 256);
}

pub const Layer = struct {
    input_norm: u64,
    post_norm: u64,
    q: Lin,
    k: Lin,
    v: Lin,
    o: Lin,
    q_norm: u64,
    k_norm: u64,
    gate: Lin,
    up: Lin,
    down: Lin,
    attn_conv: Lin, // kernel_projection: hidden -> 4 * groups dynamic coefficients
    attn_base: u64, // base_kernel (2 branches, 2 lags, hidden) bf16
    mlp_conv: Lin,
    mlp_base: u64,
};

pub const Weights = struct {
    fc: Lin, // the taps' concatenation -> hidden
    hidden_norm: u64,
    final_norm: u64,
    selector: Lin, // hidden_projection: hidden -> rank
    predecessor: []u16, // codebooks (vocab, rank) bf16, on the host
    successor: []u16,
    layers: []Layer,
    arena: DeviceBuffer,
    arena_at: usize = 0,
    /// the MXFP8 projections' own buffers (fragments, then tiled scales)
    mx: std.ArrayList(DeviceBuffer) = .empty,
    mx_bytes: usize = 0,

    pub fn deinit(w: *Weights, gpa: std.mem.Allocator) void {
        for (w.mx.items) |*b| b.free();
        w.mx.deinit(gpa);
        w.arena.free();
        gpa.free(w.predecessor);
        gpa.free(w.successor);
        gpa.free(w.layers);
    }

    fn alloc(w: *Weights, bytes: usize) !u64 {
        const at = aligned(w.arena_at);
        if (at + bytes > w.arena.len) return error.ArenaExhausted;
        w.arena_at = at + bytes;
        return w.arena.ptr + at;
    }
};

/// The drafter's widths the loader needs (cuda_draft.zig's Config).
pub const Shape = struct { hidden: usize, intermediate: usize, layers: usize, vocab: usize, q_width: usize, kv_width: usize, head_dim: usize, dynamic: usize, rank: usize, taps: usize };

fn packedBytes(n: usize, k: usize) usize {
    const l = Affine4.layout(n, k);
    return aligned(l.words) + 2 * aligned(l.scales);
}

/// One projection's quantized words, scales and biases as `linear` stages them, back to back.
pub fn rawBytes(n: usize, k: usize) usize {
    return n * k / 2 + 2 * (n * k / 64 * 2);
}

/// The largest projection `load` stages: the staging buffer it allocates for the load and frees after.
pub fn stageBytes(s: Shape) usize {
    var most: usize = 0;
    const shapes = [_][2]usize{ .{ s.hidden, s.taps * s.hidden }, .{ s.rank, s.hidden }, .{ s.q_width, s.hidden }, .{ s.kv_width, s.hidden }, .{ s.hidden, s.q_width }, .{ s.intermediate, s.hidden }, .{ s.hidden, s.intermediate }, .{ s.dynamic, s.hidden } };
    for (shapes) |nk| most = @max(most, rawBytes(nk[0], nk[1]));
    return most;
}

/// Device bytes of everything `load` packs.
pub fn total(s: Shape) usize {
    var t = packedBytes(s.hidden, s.taps * s.hidden) + packedBytes(s.rank, s.hidden) + 2 * aligned(s.hidden * 2);
    t += s.layers * (2 * aligned(s.hidden * 2) + packedBytes(s.q_width, s.hidden) + 2 * packedBytes(s.kv_width, s.hidden) + packedBytes(s.hidden, s.q_width) +
        2 * aligned(s.head_dim * 2) + 2 * packedBytes(s.intermediate, s.hidden) + packedBytes(s.hidden, s.intermediate) + 2 * packedBytes(s.dynamic, s.hidden) + 2 * aligned(4 * s.hidden * 2));
    return t;
}

fn bf16(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

fn toBf16(x: f32) u16 {
    // round to nearest even, as the device does
    const u: u32 = @bitCast(x);
    const rounding: u32 = 0x7fff + ((u >> 16) & 1);
    return @intCast((u +% rounding) >> 16);
}

/// MLX affine-4, group 64: scale bf16((max - min) / 15), bias bf16(min), nibbles round((w - bias) / scale), low first.
fn quantize(gpa: std.mem.Allocator, w: []align(1) const u16, n: usize, k: usize) !struct { words: []u32, scales: []u16, biases: []u16 } {
    const groups = k / 64;
    const words = try gpa.alloc(u32, n * k / 8);
    errdefer gpa.free(words);
    const scales = try gpa.alloc(u16, n * groups);
    errdefer gpa.free(scales);
    const biases = try gpa.alloc(u16, n * groups);
    errdefer gpa.free(biases);
    for (0..n) |r| {
        for (0..groups) |g| {
            const row = w[r * k + g * 64 ..][0..64];
            var lo: f32 = std.math.inf(f32);
            var hi: f32 = -std.math.inf(f32);
            for (row) |h| {
                const x = bf16(h);
                lo = @min(lo, x);
                hi = @max(hi, x);
            }
            const scale = toBf16((hi - lo) / 15.0);
            const bias = toBf16(lo);
            scales[r * groups + g] = scale;
            biases[r * groups + g] = bias;
            const s = bf16(scale);
            const b = bf16(bias);
            const inv = if (s == 0) 0.0 else 1.0 / s;
            for (0..8) |j| {
                var word: u32 = 0;
                for (0..8) |i| {
                    const q = std.math.clamp(@round((bf16(row[j * 8 + i]) - b) * inv), 0.0, 15.0);
                    word |= @as(u32, @intFromFloat(q)) << @intCast(4 * i);
                }
                words[(r * k + g * 64) / 8 + j] = word;
            }
        }
    }
    return .{ .words = words, .scales = scales, .biases = biases };
}

/// A bf16 (n, k) tensor quantized and packed through `stage` into the drafter's arena.
pub fn linear(gpa: std.mem.Allocator, e: anytype, stage: *DeviceBuffer, w: *Weights, ck: *core.Checkpoint, name: []const u8, n: usize, k: usize) !Affine4 {
    const t = try ck.expect(name, .bf16, &.{ n, k });
    const q = try quantize(gpa, std.mem.bytesAsSlice(u16, t.bytes), n, k);
    defer {
        gpa.free(q.words);
        gpa.free(q.scales);
        gpa.free(q.biases);
    }
    const parts = [_][]const u8{ std.mem.sliceAsBytes(q.words), std.mem.sliceAsBytes(q.scales), std.mem.sliceAsBytes(q.biases) };
    var raw: [3]u64 = undefined;
    var at: usize = 0;
    for (parts, &raw) |p, *r| {
        if (at + p.len > stage.len) return error.StageTooSmall;
        try stage.upload(at, p);
        r.* = stage.ptr + at;
        at += p.len;
    }
    const l = Affine4.layout(n, k);
    const out: Affine4 = .{ .w = try w.alloc(l.words), .s = try w.alloc(l.scales), .b = try w.alloc(l.scales), .n = n, .k = k, .npad = l.npad };
    try e.affine.pack(e.stream, raw[0], raw[1], raw[2], out);
    try e.stream.synchronize();
    return out;
}

/// `linear`, or the MXFP8 form when `mx` (the engine's lane matmul must be loaded).
pub fn lin(gpa: std.mem.Allocator, e: anytype, stage: *DeviceBuffer, w: *Weights, ck: *core.Checkpoint, name: []const u8, n: usize, k: usize, mx: bool) !Lin {
    if (!mx) return .{ .a = try linear(gpa, e, stage, w, ck, name, n, k) };
    const t = try ck.expect(name, .bf16, &.{ n, k });
    const rows = std.mem.bytesAsSlice(u16, t.bytes);
    const codes = try gpa.alloc(u8, n * k);
    defer gpa.free(codes);
    const sc = try gpa.alloc(u8, n * (k / 32));
    defer gpa.free(sc);
    const host = try gpa.alloc(u16, n * k);
    defer gpa.free(host);
    @memcpy(host, rows);
    q4.quantizeMx(host, n, k, codes, sc);
    const npad = fp8.npadOf(n);
    const frag = try gpa.alloc(u8, npad * k);
    defer gpa.free(frag);
    try fp8.fragmentOrder(codes, n, k, npad, frag);
    const tiled = try gpa.alloc(u8, npad * (k / 64) * 2);
    defer gpa.free(tiled);
    try fp8.tileMxScales(sc, n, k, npad, tiled);
    var buf = try DeviceBuffer.alloc(e.d, frag.len + tiled.len);
    errdefer buf.free();
    try buf.upload(0, frag);
    try buf.upload(frag.len, tiled);
    try w.mx.append(gpa, buf);
    w.mx_bytes += buf.len;
    return .{ .a = .{ .w = 0, .s = 0, .b = 0, .n = n, .k = k, .npad = npad }, .mx = .{ .w8 = buf.ptr, .bs = buf.ptr + frag.len, .n = @intCast(n), .k = @intCast(k), .npad = @intCast(npad), .mx = true } };
}

fn vector(w: *Weights, ck: *core.Checkpoint, name: []const u8, shape: []const usize) !u64 {
    const t = try ck.expect(name, .bf16, shape);
    const p = try w.alloc(t.bytes.len);
    try w.arena.upload(p - w.arena.ptr, t.bytes);
    return p;
}

fn hostBf16(gpa: std.mem.Allocator, ck: *core.Checkpoint, name: []const u8, shape: []const usize) ![]u16 {
    const t = try ck.expect(name, .bf16, shape);
    const out = try gpa.alloc(u16, t.bytes.len / 2);
    @memcpy(out, std.mem.bytesAsSlice(u16, t.bytes));
    return out;
}

pub fn load(gpa: std.mem.Allocator, io: std.Io, e: anytype, dir: []const u8, s: Shape, mx: bool) !Weights {
    var ck = try core.Checkpoint.openModel(gpa, io, dir);
    defer ck.close();
    var w: Weights = .{ .fc = undefined, .hidden_norm = 0, .final_norm = 0, .selector = undefined, .predecessor = &.{}, .successor = &.{}, .layers = &.{}, .arena = try DeviceBuffer.alloc(e.d, total(s)) };
    errdefer w.arena.free();
    try w.arena.fill8(0, null);
    var stage = try DeviceBuffer.alloc(e.d, stageBytes(s));
    defer stage.free();
    w.layers = try gpa.alloc(Layer, s.layers);
    errdefer gpa.free(w.layers);
    w.fc = try lin(gpa, e, &stage, &w, &ck, "fc.weight", s.hidden, s.taps * s.hidden, mx);
    w.hidden_norm = try vector(&w, &ck, "hidden_norm.weight", &.{s.hidden});
    w.final_norm = try vector(&w, &ck, "norm.weight", &.{s.hidden});
    w.selector = try lin(gpa, e, &stage, &w, &ck, "candidate_selector.hidden_projection.weight", s.rank, s.hidden, mx);
    w.predecessor = try hostBf16(gpa, &ck, "candidate_selector.predecessor_codebook", &.{ s.vocab, s.rank });
    errdefer gpa.free(w.predecessor);
    w.successor = try hostBf16(gpa, &ck, "candidate_selector.successor_codebook", &.{ s.vocab, s.rank });
    errdefer gpa.free(w.successor);
    for (w.layers, 0..) |*l, i| {
        var buf: [96]u8 = undefined;
        const Name = struct {
            buf: *[96]u8,
            i: usize,
            fn of(self: @This(), suffix: []const u8) ![]const u8 {
                return std.fmt.bufPrint(self.buf, "layers.{d}.{s}", .{ self.i, suffix });
            }
        };
        const nm: Name = .{ .buf = &buf, .i = i };
        l.* = .{
            .input_norm = try vector(&w, &ck, try nm.of("input_layernorm.weight"), &.{s.hidden}),
            .post_norm = try vector(&w, &ck, try nm.of("post_attention_layernorm.weight"), &.{s.hidden}),
            .q = try lin(gpa, e, &stage, &w, &ck, try nm.of("self_attn.q_proj.weight"), s.q_width, s.hidden, mx),
            .k = try lin(gpa, e, &stage, &w, &ck, try nm.of("self_attn.k_proj.weight"), s.kv_width, s.hidden, mx),
            .v = try lin(gpa, e, &stage, &w, &ck, try nm.of("self_attn.v_proj.weight"), s.kv_width, s.hidden, mx),
            .o = try lin(gpa, e, &stage, &w, &ck, try nm.of("self_attn.o_proj.weight"), s.hidden, s.q_width, mx),
            .q_norm = try vector(&w, &ck, try nm.of("self_attn.q_norm.weight"), &.{s.head_dim}),
            .k_norm = try vector(&w, &ck, try nm.of("self_attn.k_norm.weight"), &.{s.head_dim}),
            .gate = try lin(gpa, e, &stage, &w, &ck, try nm.of("mlp.gate_proj.weight"), s.intermediate, s.hidden, mx),
            .up = try lin(gpa, e, &stage, &w, &ck, try nm.of("mlp.up_proj.weight"), s.intermediate, s.hidden, mx),
            .down = try lin(gpa, e, &stage, &w, &ck, try nm.of("mlp.down_proj.weight"), s.hidden, s.intermediate, mx),
            .attn_conv = try lin(gpa, e, &stage, &w, &ck, try nm.of("attention_conv.kernel_projection.weight"), s.dynamic, s.hidden, mx),
            .attn_base = try vector(&w, &ck, try nm.of("attention_conv.base_kernel"), &.{ 2, 2, s.hidden }),
            .mlp_conv = try lin(gpa, e, &stage, &w, &ck, try nm.of("mlp_conv.kernel_projection.weight"), s.dynamic, s.hidden, mx),
            .mlp_base = try vector(&w, &ck, try nm.of("mlp_conv.base_kernel"), &.{ 2, 2, s.hidden }),
        };
    }
    return w;
}

test "bf16 rounding of a quantized group's scale and bias" {
    try std.testing.expectEqual(@as(u16, 0x3f80), toBf16(1.0));
    try std.testing.expectEqual(@as(f32, 1.0), bf16(0x3f80));
    const w: [64]u16 = @splat(0x3f80); // a flat group: scale 0, bias 1, every nibble 0
    const q = try quantize(std.testing.allocator, &w, 1, 64);
    defer {
        std.testing.allocator.free(q.words);
        std.testing.allocator.free(q.scales);
        std.testing.allocator.free(q.biases);
    }
    try std.testing.expectEqual(@as(u16, 0), q.scales[0]);
    try std.testing.expectEqual(@as(u16, 0x3f80), q.biases[0]);
    try std.testing.expectEqual(@as(u32, 0), q.words[0]);
}
