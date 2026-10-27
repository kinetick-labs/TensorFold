//! The positional reader returns the mapping's rows, bit for bit, however they were asked for.
const std = @import("std");
const ple = @import("cuda_ple.zig");
const fixture = @import("cuda_ple_nvfp4_test.zig").fixture;
const a = std.testing.allocator;
const io = std.testing.io;

test "NVFP4 rows read by pread equal the mapped rows, cold, ahead and after eviction" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try fixture(tmp.dir, true);
    const path = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer a.free(path);
    var g = try ple.Ngram.init(a, io, path);
    defer g.deinit();
    try std.testing.expect(g.reader == null);
    var tokens: [513]u32 = undefined;
    for (&tokens, 0..) |*v, i| v.* = @intCast((i * 11 + 3) % 100);
    var want: [513 * 32]u16 = undefined;
    var want_xs: [513]f32 = undefined;
    var want_hist = [_]i64{99};
    try g.rows(&want_hist, &tokens, &want, &want_xs);

    try g.openReader(io);
    const rd = g.reader.?;
    var got: [513 * 32]u16 = undefined;
    var got_xs: [513]f32 = undefined;
    // cold: every row from the file
    @memset(&got, 0xdead);
    var hist = [_]i64{99};
    try g.rows(&hist, &tokens, &got, &got_xs);
    try std.testing.expectEqualSlices(u16, &want, &got);
    try std.testing.expectEqualSlices(f32, &want_xs, &got_xs);
    try std.testing.expectEqualSlices(i64, &want_hist, &hist);
    const cold = rd.take();
    try std.testing.expect(cold.own > 0);
    // warm: the same rows from the cache
    @memset(&got, 0xdead);
    hist = .{99};
    try g.rows(&hist, &tokens, &got, &got_xs);
    try std.testing.expectEqualSlices(u16, &want, &got);
    const warm = rd.take();
    try std.testing.expectEqual(@as(u64, 0), warm.own);
    try std.testing.expect(warm.hits > 0);
    // ahead: a fresh reader's cache filled by a chunk's read-ahead, then gathered with no read of the gather's own
    g.reader = null;
    rd.destroy();
    try g.openReader(io);
    const rd2 = g.reader.?;
    g.aheadChunk(&[_]i64{99}, tokens[0..300]);
    g.ahead_thread.?.join();
    g.ahead_thread = null;
    rd2.drain();
    _ = rd2.take();
    @memset(&got, 0xdead);
    hist = .{99};
    try g.rows(&hist, tokens[0..300], got[0 .. 300 * 32], got_xs[0..300]);
    try std.testing.expectEqualSlices(u16, want[0 .. 300 * 32], got[0 .. 300 * 32]);
    try std.testing.expectEqual(@as(u64, 0), rd2.take().own);
    // the decode read-ahead pushes tokens with a lane's moving history
    var ahead: ple.Ahead = .{ .g = &g };
    ahead.lane(0, &[_]i64{99});
    for (tokens[300..310]) |t| ahead.push(0, t);
    ahead.flush();
    rd2.drain();
    var hist2 = [_]i64{99};
    try g.rows(&hist2, tokens[0..300], got[0 .. 300 * 32], got_xs[0..300]);
    _ = rd2.take();
    @memset(&got, 0xdead);
    try g.rows(&hist2, tokens[300..310], got[0 .. 10 * 32], got_xs[0..10]);
    try std.testing.expectEqualSlices(u16, want[300 * 32 .. 310 * 32], got[0 .. 10 * 32]);
    try std.testing.expectEqual(@as(u64, 0), rd2.take().own);
}

test "NVFP4 rows read by pread stay exact when slots collide" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try fixture(tmp.dir, true);
    const path = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer a.free(path);
    var g = try ple.Ngram.init(a, io, path);
    defer g.deinit();
    var tokens: [257]u32 = undefined;
    for (&tokens, 0..) |*v, i| v.* = @intCast((i * 7 + 1) % 100);
    var want: [257 * 32]u16 = undefined;
    var want_xs: [257]f32 = undefined;
    var h0 = [_]i64{99};
    try g.rows(&h0, &tokens, &want, &want_xs);
    // two slots for every row of the table: nearly every read evicts another
    const reader = @import("cuda_ple_reader.zig");
    var paths = [_][:0]const u8{g.files.items[0].file.path};
    const base = @intFromPtr(g.files.items[0].file.map.memory.ptr);
    var shards: [2]reader.Shard = undefined;
    for (g.tables[0..2], &shards) |t, *s| s.* = .{ .file = 0, .codes_at = @intFromPtr(t.codes.ptr) - base, .scales_at = @intFromPtr(t.scales.ptr) - base };
    g.reader = try reader.Reader.create(a, io, &paths, &shards, g.hash.starts[0..3], 8, 1, 2, 8);
    var got: [257 * 32]u16 = undefined;
    var got_xs: [257]f32 = undefined;
    for (0..3) |_| {
        var h = [_]i64{99};
        @memset(&got, 0xdead);
        try g.rows(&h, &tokens, &got, &got_xs);
        try std.testing.expectEqualSlices(u16, &want, &got);
        g.reader.?.prefetch(&.{ 0, 1, 2, 3, 4, 5, 6, 7 });
    }
}

fn ms(io_: std.Io, t0: i96) f64 {
    return @as(f64, @floatFromInt(std.Io.Clock.awake.now(io_).toNanoseconds() - t0)) / 1e6;
}

fn gather(g: *ple.Ngram, hist: *[2]i64, toks: []const u32, out: []u16, xs: []f32) !void {
    try g.rows(hist, toks, out, xs[0 .. out.len / 32]);
}

// TF_PLE_REAL_DIR=<checkpoint dir>: the real 28.7 GiB table, mapped and by pread, on random tokens (timings on stderr).
test "real checkpoint: pread rows equal the mapped rows" {
    const dir = std.c.getenv("TF_PLE_REAL_DIR") orelse return error.SkipZigTest;
    var g = try ple.Ngram.init(a, io, std.mem.span(dir));
    defer g.deinit();
    const w = g.width();
    const seed: u64 = if (std.c.getenv("TF_PLE_SEED")) |v| std.fmt.parseInt(u64, std.mem.span(v), 10) catch 42 else 42;
    var prng = std.Random.DefaultPrng.init(seed);
    const n = 3072;
    const tokens = try a.alloc(u32, 2 * n);
    defer a.free(tokens);
    for (tokens) |*t| t.* = prng.random().uintLessThan(u32, 248000);
    const want = try a.alloc(u16, 2 * n * w);
    defer a.free(want);
    const got = try a.alloc(u16, 2 * n * w);
    defer a.free(got);
    const xs = try a.alloc(f32, 2 * n * w / 32);
    defer a.free(xs);
    var hist = [_]i64{ g.hash.eos, g.hash.eos };
    var t0 = std.Io.Clock.awake.now(io).toNanoseconds();
    try gather(&g, &hist, tokens[0..n], want[0 .. n * w], xs);
    std.debug.print("\nmapped, {d} rows: {d:.1} ms\n", .{ n, ms(io, t0) });
    hist = .{ g.hash.eos, g.hash.eos };
    try gather(&g, &hist, tokens[n..], want[n * w ..], xs);
    // decode-sized, mapped: 90 tokens' rows
    const small = try a.alloc(u16, 90 * w);
    defer a.free(small);
    hist = .{ g.hash.eos, g.hash.eos };
    t0 = std.Io.Clock.awake.now(io).toNanoseconds();
    try gather(&g, &hist, tokens[100..190], small, xs);
    std.debug.print("mapped, 90 rows: {d:.2} ms\n", .{ms(io, t0)});

    try g.openReader(io);
    const rd = g.reader.?;
    hist = .{ g.hash.eos, g.hash.eos };
    t0 = std.Io.Clock.awake.now(io).toNanoseconds();
    try gather(&g, &hist, tokens[0..n], got[0 .. n * w], xs);
    std.debug.print("pread cold, {d} rows: {d:.1} ms\n", .{ n, ms(io, t0) });
    try std.testing.expectEqualSlices(u16, want[0 .. n * w], got[0 .. n * w]);
    hist = .{ g.hash.eos, g.hash.eos };
    t0 = std.Io.Clock.awake.now(io).toNanoseconds();
    try gather(&g, &hist, tokens[0..n], got[0 .. n * w], xs);
    std.debug.print("pread cached, {d} rows: {d:.1} ms\n", .{ n, ms(io, t0) });
    try std.testing.expectEqualSlices(u16, want[0 .. n * w], got[0 .. n * w]);
    // read ahead of the second chunk, then its gather
    hist = .{ tokens[n - 2], tokens[n - 1] };
    t0 = std.Io.Clock.awake.now(io).toNanoseconds();
    g.aheadChunk(&hist, tokens[n..]);
    g.ahead_thread.?.join();
    g.ahead_thread = null;
    std.debug.print("ahead of {d} rows asked: {d:.1} ms; ", .{ n, ms(io, t0) });
    rd.drain();
    std.debug.print("all read after {d:.1} ms\n", .{ms(io, t0)});
    _ = rd.take();
    hist = .{ tokens[n - 2], tokens[n - 1] };
    t0 = std.Io.Clock.awake.now(io).toNanoseconds();
    try gather(&g, &hist, tokens[n..], got[n * w ..], xs);
    std.debug.print("pread after read-ahead, {d} rows: {d:.1} ms\n", .{ n, ms(io, t0) });
    const q = rd.take();
    std.debug.print("  own reads {d}, hits {d}\n", .{ q.own, q.hits });
    // the second chunk's expected rows came after the first chunk's history: compare against a mapped gather of the same call
    const rdp = g.reader;
    g.reader = null;
    hist = .{ tokens[n - 2], tokens[n - 1] };
    try gather(&g, &hist, tokens[n..], want[n * w ..], xs);
    g.reader = rdp;
    try std.testing.expectEqualSlices(u16, want[n * w ..], got[n * w ..]);
    // a decode round: 90 rows cold by pread, then the same asked for ahead
    hist = .{ g.hash.eos, g.hash.eos };
    t0 = std.Io.Clock.awake.now(io).toNanoseconds();
    try gather(&g, &hist, tokens[100..190], small[0 .. 90 * w], xs);
    std.debug.print("pread, 90 rows (cached from the first pass): {d:.2} ms\n", .{ms(io, t0)});
    var fresh: [90]u32 = undefined;
    for (&fresh) |*t| t.* = prng.random().uintLessThan(u32, 248000);
    hist = .{ g.hash.eos, g.hash.eos };
    t0 = std.Io.Clock.awake.now(io).toNanoseconds();
    try gather(&g, &hist, &fresh, small[0 .. 90 * w], xs);
    std.debug.print("pread, 90 fresh rows, cold: {d:.2} ms\n", .{ms(io, t0)});
    var ahead: ple.Ahead = .{ .g = &g };
    var fresh2: [90]u32 = undefined;
    for (&fresh2) |*t| t.* = prng.random().uintLessThan(u32, 248000);
    ahead.lane(0, &[_]i64{ g.hash.eos, g.hash.eos });
    t0 = std.Io.Clock.awake.now(io).toNanoseconds();
    for (fresh2) |t| ahead.push(0, t);
    ahead.flush();
    rd.drain();
    std.debug.print("decode read-ahead of 90 tokens' rows, all read after {d:.2} ms\n", .{ms(io, t0)});
    hist = .{ g.hash.eos, g.hash.eos };
    t0 = std.Io.Clock.awake.now(io).toNanoseconds();
    try gather(&g, &hist, &fresh2, small[0 .. 90 * w], xs);
    std.debug.print("pread, 90 rows read ahead: {d:.2} ms\n", .{ms(io, t0)});
}
