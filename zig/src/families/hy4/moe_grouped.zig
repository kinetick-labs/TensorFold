//! The routed experts of one MoE layer as cuda.exl3_experts' grouped layer (exl3/experts.cu, shared with GLM and
//! MiniMax): the device routes, groups and runs every expert's rows without the host. The pack keeps each expert's
//! three EXL3 projections apart, so the loader makes the grouped kernels' tables: one trellis pointer and one
//! half-bit width per expert and projection, and the Hadamard scales stacked `[E, ·]`. A layer that does not fit
//! (a plain expert, another codebook, strip layout, a width the tiles were not built for) keeps the per-expert path.

const std = @import("std");
const Allocator = std.mem.Allocator;
const cuda = @import("cuda");
const gm = @import("gpu_model.zig");

/// The host side of one layer's tables, before upload.
pub const Tables = struct {
    experts: u32,
    dims: u32, // D: the latent width the experts read and write
    width: u32, // I: the expert width
    ptr: [3][]u64, // gate, up, down: each expert's trellis address
    k2: [3][]i32, // the same, half-bits a value
    /// Device copies to make, in order: (src, bytes) per expert, gate suh, up suh, gate svh, up svh, down suh, down svh.
    scales: [6][]u64,

    /// Bytes of the device block upload() fills: the pointer and width tables, then the stacked scales.
    pub fn bytes(t: *const Tables) u64 {
        const E: u64 = t.experts;
        return 3 * align256(E * 8) + 3 * align256(E * 4) + 3 * align256(E * t.dims * 2) + 3 * align256(E * t.width * 2);
    }
};

fn align256(n: u64) u64 {
    return std.mem.alignForward(u64, n, 256);
}

pub const Refusal = enum { plain, codebook, shape, layout, tiles, none };

/// The tables for `experts` (gate, up: D -> I; down: I -> D), or why the grouped kernels cannot take them.
pub fn tables(a: Allocator, experts: []const [3]gm.Lin, why: *Refusal) !?Tables {
    why.* = .none;
    if (experts.len == 0 or experts.len > 4096) {
        why.* = .shape;
        return null;
    }
    const first = switch (experts[0][0]) {
        .exl3 => |w| w,
        .plain => {
            why.* = .plain;
            return null;
        },
    };
    const D = first.k;
    const I = first.n;
    _ = cuda.exl3_experts.plan(D, I, true) catch {
        why.* = .tiles;
        return null;
    };
    _ = cuda.exl3_experts.plan(I, D, false) catch {
        why.* = .tiles;
        return null;
    };
    const E = experts.len;
    var t: Tables = .{ .experts = @intCast(E), .dims = D, .width = I, .ptr = undefined, .k2 = undefined, .scales = undefined };
    for (0..3) |m| {
        t.ptr[m] = try a.alloc(u64, E);
        t.k2[m] = try a.alloc(i32, E);
    }
    for (&t.scales) |*s| s.* = try a.alloc(u64, E);
    for (experts, 0..) |ex, e| {
        for (ex, 0..) |l, m| {
            const w = switch (l) {
                .exl3 => |x| x,
                .plain => {
                    why.* = .plain;
                    return null;
                },
            };
            if (w.bias != 0) {
                why.* = .shape;
                return null;
            }
            if (w.codebook != .mul1) {
                why.* = .codebook;
                return null;
            }
            const k: u32 = if (m == 2) I else D;
            const n: u32 = if (m == 2) D else I;
            if (w.k != k or w.n != n) {
                why.* = .shape;
                return null;
            }
            const st = cuda.exl3.strides(w.k, w.n, w.halves, false);
            if (w.stride_k != st.k or w.stride_nb != st.nb) {
                why.* = .layout;
                return null;
            }
            t.ptr[m][e] = w.trellis;
            t.k2[m][e] = @intCast(w.halves);
        }
        const g = ex[0].exl3;
        const u = ex[1].exl3;
        const d = ex[2].exl3;
        t.scales[0][e] = g.suh;
        t.scales[1][e] = u.suh;
        t.scales[2][e] = g.svh;
        t.scales[3][e] = u.svh;
        t.scales[4][e] = d.suh;
        t.scales[5][e] = d.svh;
    }
    return t;
}

/// The tables into the device block at `at` (Tables.bytes() long), the scales copied device to device on `s`.
pub fn upload(d: *const cuda.Driver, s: cuda.Stream, t: *const Tables, at: u64, limit: f32) !cuda.exl3_experts.Layer {
    const E: u64 = t.experts;
    var o = at;
    var ptrs: [3]u64 = undefined;
    var k2s: [3]u64 = undefined;
    for (0..3) |m| {
        ptrs[m] = o;
        try d.check(d.api.cuMemcpyHtoDAsync_v2(o, t.ptr[m].ptr, E * 8, s.handle), "cuMemcpyHtoDAsync");
        o += align256(E * 8);
    }
    for (0..3) |m| {
        k2s[m] = o;
        try d.check(d.api.cuMemcpyHtoDAsync_v2(o, t.k2[m].ptr, E * 4, s.handle), "cuMemcpyHtoDAsync");
        o += align256(E * 4);
    }
    // gate suh, up suh: [E, D]; gate svh, up svh, down suh: [E, I]; down svh: [E, D]
    const widths = [6]u64{ t.dims, t.dims, t.width, t.width, t.width, t.dims };
    var stacked: [6]u64 = undefined;
    for (0..6) |j| {
        stacked[j] = o;
        const row = widths[j] * 2;
        for (t.scales[j], 0..) |src, e| try d.check(d.api.cuMemcpyDtoDAsync_v2(o + e * row, src, row, s.handle), "cuMemcpyDtoDAsync");
        o += align256(E * row);
    }
    // the host tables must outlive the copies
    try s.synchronize();
    if (o - at != t.bytes()) return error.Hy4Shape;
    return .{
        .gate = ptrs[0],
        .up = ptrs[1],
        .down = ptrs[2],
        .gate_k2 = k2s[0],
        .up_k2 = k2s[1],
        .down_k2 = k2s[2],
        .suh_g = stacked[0],
        .suh_u = stacked[1],
        .svh_g = stacked[2],
        .svh_u = stacked[3],
        .suh_d = stacked[4],
        .svh_d = stacked[5],
        .experts = t.experts,
        .dims = t.dims,
        .width = t.width,
        .codebook = .mul1,
        .limit = limit,
        .act_mode = 1, // fp32 SwiGLU with the clamp: gate min(x, limit), up clamp(x, -limit, limit), as hy4_swiglu
    };
}

/// Every MoE layer of `layers` given its grouped form, the tables in one device buffer (returned; the caller frees
/// it). Layers the kernels refuse keep the per-expert path; the first refusal is logged. Null when none fit.
pub fn bindAll(a: Allocator, d: *const cuda.Driver, s: cuda.Stream, layers: []gm.Layer) !?cuda.DeviceBuffer {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const ta = arena.allocator();
    const all = try ta.alloc(?Tables, layers.len);
    var total: u64 = 0;
    var refused: usize = 0;
    for (layers, all, 0..) |*L, *t, i| {
        t.* = null;
        switch (L.mlp) {
            .dense => {},
            .moe => |*m| {
                var why: Refusal = .none;
                t.* = try tables(ta, m.expert, &why);
                if (t.*) |*tt| total += tt.bytes() else {
                    if (refused == 0) std.log.warn("hy4: layer {d}'s experts stay per-expert ({s})", .{ i, @tagName(why) });
                    refused += 1;
                }
            },
        }
    }
    if (total == 0) return null;
    var buf = try cuda.DeviceBuffer.alloc(d, total);
    errdefer buf.free();
    var at = buf.ptr;
    for (layers, all) |*L, t| if (t) |*tt| {
        L.mlp.moe.grouped = try upload(d, s, tt, at, L.mlp.moe.clamp);
        at += tt.bytes();
    };
    std.log.info("hy4: grouped experts on the device for {d} layers ({d} MiB of tables), {d} per-expert", .{ layers.len - refused - countDense(layers), total >> 20, refused });
    return buf;
}

fn countDense(layers: []const gm.Layer) usize {
    var n: usize = 0;
    for (layers) |L| n += @intFromBool(L.mlp == .dense);
    return n;
}

fn fakeWeight(k: u32, n: u32, halves: u32, base: u64) cuda.exl3.Weight {
    const st = cuda.exl3.strides(k, n, halves, false);
    return .{ .trellis = base, .suh = base + 0x100000, .svh = base + 0x200000, .k = k, .n = n, .halves = halves, .codebook = .mul1, .stride_k = st.k, .stride_nb = st.nb };
}

test "grouped tables: per-expert pointers, widths and scale sources in order; refusals for plain, shape and layout" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const D = 1536;
    const I = 1536;
    var ex: [3][3]gm.Lin = undefined;
    for (&ex, 0..) |*e, i| {
        const b: u64 = 0x10000000 * (i + 1);
        e.* = .{ .{ .exl3 = fakeWeight(D, I, 10, b) }, .{ .exl3 = fakeWeight(D, I, 10, b + 0x1000) }, .{ .exl3 = fakeWeight(I, D, 12, b + 0x2000) } };
    }
    var why: Refusal = undefined;
    const t = (try tables(a, &ex, &why)).?;
    try std.testing.expectEqual(Refusal.none, why);
    try std.testing.expectEqual(@as(u32, 3), t.experts);
    try std.testing.expectEqual(@as(u64, 0x20000000 + 0x2000), t.ptr[2][1]);
    try std.testing.expectEqual(@as(i32, 12), t.k2[2][0]);
    try std.testing.expectEqual(@as(i32, 10), t.k2[0][2]);
    try std.testing.expectEqual(@as(u64, 0x30000000 + 0x1000 + 0x200000), t.scales[3][2]); // up svh of expert 2
    try std.testing.expectEqual(@as(u64, 0x10000000 + 0x2000 + 0x100000), t.scales[4][0]); // down suh of expert 0
    // 3 x 512 B pointer and width tables (rounded), 6 stacked scale blocks of 3 * 1536 * 2 B
    try std.testing.expectEqual(@as(u64, 3 * 256 + 3 * 256 + 6 * 9216), t.bytes());

    var bad = ex;
    bad[1][1] = .{ .plain = .{ .w = .none, .k = D, .n = I } };
    try std.testing.expect((try tables(a, &bad, &why)) == null);
    try std.testing.expectEqual(Refusal.plain, why);
    bad = ex;
    bad[2][2] = .{ .exl3 = fakeWeight(I, 1024, 12, 0) };
    try std.testing.expect((try tables(a, &bad, &why)) == null);
    try std.testing.expectEqual(Refusal.shape, why);
    bad = ex;
    var strip = fakeWeight(D, I, 10, 0);
    strip.stride_k = cuda.exl3.strides(D, I, 10, true).k;
    bad[0][0] = .{ .exl3 = strip };
    try std.testing.expect((try tables(a, &bad, &why)) == null);
    try std.testing.expect(why == .layout or why == .shape);
    // widths the tiles were not built for
    var odd: [1][3]gm.Lin = .{.{ .{ .exl3 = fakeWeight(D, 200, 10, 0) }, .{ .exl3 = fakeWeight(D, 200, 10, 0) }, .{ .exl3 = fakeWeight(200, D, 10, 0) } }};
    try std.testing.expect((try tables(a, &odd, &why)) == null);
    try std.testing.expectEqual(Refusal.tiles, why);
}
