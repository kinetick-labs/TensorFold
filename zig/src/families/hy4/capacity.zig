//! How many sequences (lanes) of a given context fit: one sequence's device bytes (every layer's cache, its 3D
//! positions) and the lanes a free budget holds. The serve engine logs this table at boot; the window script reads
//! it back at 256k / 512k / 1M.

const std = @import("std");
const config = @import("config.zig");
const gm = @import("gpu_model.zig");
const bind = @import("bind.zig");

/// One sequence's device bytes at context `ctx` with forward calls of `chunk` rows and `dt` caches.
pub fn seqBytes(t: *const config.Text, ctx: u32, chunk: u32, dt: gm.CacheDt) u64 {
    var n: u64 = std.mem.alignForward(u64, 3 * @as(u64, ctx) * 4, 256); // positions
    for (0..t.layers) |i| {
        const d = bind.dims(t, @intCast(i));
        n += gm.cacheBytes(&d, ctx, chunk, dt);
    }
    return n;
}

/// Lanes of `per` bytes in `free` after keeping `reserve` aside, at most `want`.
pub fn fit(free: u64, reserve: u64, per: u64, want: u32) u32 {
    if (free <= reserve or per == 0) return 0;
    return @intCast(@min(@as(u64, want), (free - reserve) / per));
}

test "capacity: Hy4-flash caches per sequence by dtype, sliding layers a ring, indexed layers the whole context" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const c = try config.Config.parse(arena.allocator(), @embedFile("fixtures/config.json"));
    const t = &c.text;
    var sliding: u32 = 0;
    for (t.attn) |a| sliding += @intFromBool(a == .sliding);
    try std.testing.expectEqual(@as(u32, 30), sliding);
    const chunk = 256;
    // a sliding layer: window 512 + chunk 256 slots whatever the context
    const ds = bind.dims(t, for (t.attn, 0..) |a, i| (if (a == .sliding) break @as(u32, @intCast(i))) else unreachable);
    try std.testing.expectEqual(gm.Shape.of(&ds, 1 << 20, chunk).slots, gm.Shape.of(&ds, 1 << 18, chunk).slots);
    try std.testing.expectEqual(@as(u64, 768), gm.Shape.of(&ds, 1 << 20, chunk).slots);
    // an indexed layer at 1M, bf16: 576 two-byte values a token, a 128-wide bf16 group key per 4 tokens
    const di = bind.dims(t, for (t.attn, 0..) |a, i| (if (a == .indexed) break @as(u32, @intCast(i))) else unreachable);
    const M: u64 = 1 << 20;
    const sh = gm.Shape.of(&di, 1 << 20, chunk);
    try std.testing.expectEqual(M, sh.slots);
    try std.testing.expectEqual(@as(u64, 260), sh.tslots);
    try std.testing.expectEqual(M / 4 * 128 * 2 + M * 576 * 2 + 260 * 128 * 4, gm.cacheBytes(&di, 1 << 20, chunk, .bf16));
    // fp8: one byte a latent value and a 4-byte scale a token
    try std.testing.expectEqual(M / 4 * 128 * 2 + M * (512 + 4 + 64 * 2) + 260 * 128 * 4, gm.cacheBytes(&di, 1 << 20, chunk, .fp8));
    const gib = @as(f64, 1 << 30);
    var line: [3]f64 = undefined;
    for ([_]u32{ 1 << 18, 1 << 19, 1 << 20 }, 0..) |ctx, j| {
        const f = seqBytes(t, ctx, chunk, .f32);
        const b = seqBytes(t, ctx, chunk, .bf16);
        const e = seqBytes(t, ctx, chunk, .fp8);
        try std.testing.expect(e < b and b < f);
        line[j] = @as(f64, @floatFromInt(b)) / gib;
        std.debug.print("hy4 capacity: {d}k tokens: f32 {d:.2} GiB, bf16 {d:.2} GiB, fp8 {d:.2} GiB a sequence\n", .{ ctx >> 10, @as(f64, @floatFromInt(f)) / gib, line[j], @as(f64, @floatFromInt(e)) / gib });
    }
    // 1M in bf16 is ~14 GiB, in fp8 ~8.5 GiB
    try std.testing.expect(line[2] > 13 and line[2] < 15.5);
    try std.testing.expect(@as(f64, @floatFromInt(seqBytes(t, 1 << 20, chunk, .fp8))) / gib < 9);
    try std.testing.expectEqual(@as(u32, 2), fit(20 << 30, 2 << 30, 8 << 30, 8));
    try std.testing.expectEqual(@as(u32, 0), fit(1 << 30, 2 << 30, 1, 8));
    try std.testing.expectEqual(@as(u32, 3), fit(100 << 30, 0, 1 << 30, 3));
}
