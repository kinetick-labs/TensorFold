//! fn_row_cand (the sampled draw's candidates, TF_FLASHNEXT_SAMPLE_CAND=1) against cuda_mtp.zig's topCandidates, which a unit
//! test holds to a sort of every column: the same set of columns for rows full of ties, both zeros, both infinities, the
//! draft head's width and the vocabulary's, and a NaN row flagged and left unwritten.
const std = @import("std");
const cuda = @import("cuda");
const flash = @import("flashnext");
const check = @import("check.zig");
const rig = @import("flashnext_rig.zig");

const mtp = flash.mtp;
const Rig = rig.Rig;

fn tied(r: *Rig, row: []u16, levels: u8) void {
    if (levels == 0) {
        // a realistic row: scores around a few units wide, bf16 (the top of the row has few ties)
        for (row) |*v| {
            const g = r.rng.floatNorm(f32) * 3.0 - 1.0;
            v.* = @truncate(@as(u32, @bitCast(g)) >> 16);
        }
        return;
    }
    const specials = [_]u16{ 0x0000, 0x8000, 0x7f80, 0xff80, 0x3f80, 0xbf80 };
    for (row) |*v| {
        const x = r.rng.int(u16);
        v.* = if (x % 17 == 0) specials[x / 17 % specials.len] else @as(u16, 0x3f00) + r.rng.uintLessThan(u8, levels);
    }
}

/// Rows of `n` scores with `count` candidates each (row `bad`, when below `rows`, a NaN row), against the host reference.
fn rows(r: *Rig, mod: cuda.Module, n: usize, count: usize, levels: u8, bad: usize) !void {
    const f = try mod.function("fn_row_cand");
    const rows_n = 5;
    const stride = mtp.cand_max;
    const host = try r.a.alloc(u16, rows_n * n);
    for (0..rows_n) |i| tied(r, host[i * n ..][0..n], levels);
    if (bad < rows_n) host[bad * n + n / 3] = 0x7fc0;
    const scores = try r.dev(std.mem.sliceAsBytes(host));
    const vals = try r.out(rows_n * stride * 4);
    const cols = try r.out(rows_n * stride * 4);
    const flags = try r.out(rows_n * 4);
    var a: cuda.Args = .{};
    a.add(scores);
    a.add(@as(c_int, @intCast(n)));
    a.add(@as(c_longlong, @intCast(n)));
    a.add(@as(c_int, @intCast(count)));
    a.add(@as(c_int, @intCast(stride)));
    a.add(vals);
    a.add(cols);
    a.add(flags);
    try cuda.launch.launch(f, .{ .grid = .{ .x = rows_n }, .block = .{ .x = 1024 } }, r.s, &a);
    const got_v = try r.back(f32, vals, rows_n * stride);
    const got_c = try r.back(i32, cols, rows_n * stride);
    const got_f = try r.back(i32, flags, rows_n);
    for (0..rows_n) |i| {
        const row = host[i * n ..][0..n];
        var want_v: [mtp.cand_max]f32 = undefined;
        var want_c: [mtp.cand_max]i32 = undefined;
        const ok = mtp.topCandidates(row, count, &want_v, &want_c);
        try check.expect(ok == (i != bad) and got_f[i] == @intFromBool(i == bad), "fn_row_cand n={d} count={d} row {d}: NaN flag {d}", .{ n, count, i, got_f[i] });
        if (!ok) continue;
        const k = @min(count, n);
        // the sets (score, column) pairs, sorted by column: the kernel's order is the atomics'
        const Pair = struct { c: i32, v: f32 };
        var g: [mtp.cand_max]Pair = undefined;
        var w: [mtp.cand_max]Pair = undefined;
        for (0..k) |j| {
            g[j] = .{ .c = got_c[i * stride + j], .v = got_v[i * stride + j] };
            w[j] = .{ .c = want_c[j], .v = want_v[j] };
        }
        const less = struct {
            fn before(_: void, x: Pair, y: Pair) bool {
                return x.c < y.c;
            }
        }.before;
        std.mem.sort(Pair, g[0..k], {}, less);
        std.mem.sort(Pair, w[0..k], {}, less);
        for (g[0..k], w[0..k]) |x, y| try check.expect(x.c == y.c and @as(u32, @bitCast(x.v)) == @as(u32, @bitCast(y.v)), "fn_row_cand n={d} count={d} levels={d} row {d}: column {d} (score {e}) for column {d} (score {e})", .{ n, count, levels, i, x.c, x.v, y.c, y.v });
    }
}

const Timespec = extern struct { sec: i64, nsec: i64 };
extern "c" fn clock_gettime(clk: c_int, ts: *Timespec) c_int;

fn nowNs() u64 {
    var ts: Timespec = undefined;
    _ = clock_gettime(1, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

/// Microseconds a launch of fn_row_cand takes (back to back on one stream) for `rows_n` rows of `n` scores.
fn bench(r: *Rig, mod: cuda.Module, n: usize, rows_n: usize, count: usize, levels: u8) !void {
    const f = try mod.function("fn_row_cand");
    const host = try r.a.alloc(u16, rows_n * n);
    for (0..rows_n) |i| tied(r, host[i * n ..][0..n], levels);
    const scores = try r.dev(std.mem.sliceAsBytes(host));
    const vals = try r.out(rows_n * mtp.cand_max * 4);
    const cols = try r.out(rows_n * mtp.cand_max * 4);
    const flags = try r.out(rows_n * 4);
    var a: cuda.Args = .{};
    a.add(scores);
    a.add(@as(c_int, @intCast(n)));
    a.add(@as(c_longlong, @intCast(n)));
    a.add(@as(c_int, @intCast(count)));
    a.add(@as(c_int, @intCast(mtp.cand_max)));
    a.add(vals);
    a.add(cols);
    a.add(flags);
    const reps = 500;
    for (0..20) |_| try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(rows_n) }, .block = .{ .x = 1024 } }, r.s, &a);
    try r.s.synchronize();
    const t0 = nowNs();
    for (0..reps) |_| try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(rows_n) }, .block = .{ .x = 1024 } }, r.s, &a);
    try r.s.synchronize();
    const us = @as(f64, @floatFromInt(nowNs() - t0)) / 1000.0 / reps;
    std.debug.print("PERF fn_row_cand n={d} rows={d} count={d} levels={d}: {d:.1} us a launch\n", .{ n, rows_n, count, levels, us });
}

pub fn run(gpu: check.Gpu) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    var prng = std.Random.DefaultPrng.init(83);
    var r = try Rig.init(gpu, arena.allocator(), prng.random());
    defer r.deinit();
    var mod = try cuda.Module.load(gpu.d, cuda.kernels.flashnext_mtp);
    defer mod.unload();
    // the draft head's width (79,591), the vocabulary's (248,320), and short rows; many ties through almost none
    for ([_]usize{ 70, 1500, 79_591, 248_320 }) |n| for ([_]usize{ 1, 2, 20, 33, 64 }) |count| for ([_]u8{ 0, 1, 3, 40, 200 }) |levels| try rows(&r, mod, n, count, levels, 99);
    // a NaN row among clean ones: flagged, the others unaffected
    try rows(&r, mod, 79_591, 20, 0, 2);
    try rows(&r, mod, 79_591, 20, 40, 2);
    try rows(&r, mod, 248_320, 20, 0, 4);
    try rows(&r, mod, 248_320, 20, 40, 4);
    for ([_]u8{ 0, 3 }) |levels| {
        try bench(&r, mod, 248_320, 1, 20, levels);
        try bench(&r, mod, 248_320, 5, 20, levels);
        try bench(&r, mod, 248_320, 16, 20, levels);
        try bench(&r, mod, 79_591, 1, 20, levels);
        try bench(&r, mod, 79_591, 5, 20, levels);
    }
    check.pass("flashnext-cand: fn_row_cand's candidates are topCandidates' set for rows of 70 to 248,320 scores, 1 to 64 candidates and ties from every score equal to almost none; a NaN row is flagged", .{});
}
