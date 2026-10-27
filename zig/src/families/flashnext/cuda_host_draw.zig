//! Keyed draws on the host, several rows at once: the rows of a verify window (or of a round's windows) and the draft head's rows of
//! a batched step are independent, each drawn by `mtp.drawRowWith` from its own scores, so a few parked workers draw them side by
//! side and the draws are the ones a single thread would make, in any order.
//!
//! TF_FLASHNEXT_HOST_DRAW_THREADS=N (0, the default: one thread, the plain scans and one download a row): N workers beside the
//! engine thread, and each row's top_k found sixteen columns a compare (the same columns, in the same order).
const std = @import("std");
const mtp = @import("cuda_mtp.zig");
const ple_pool = @import("cuda_ple_pool.zig");

const unread = std.math.maxInt(u32);
var configured: std.atomic.Value(u32) = .init(unread);

/// Workers beside the engine thread at most.
pub const max_threads: usize = 31;

fn parse(text: []const u8) u32 {
    const n = std.fmt.parseInt(u32, std.mem.trim(u8, text, " \t\r\n"), 10) catch return 0;
    return @min(n, max_threads);
}

/// The worker count the environment asks for (read once); 0 draws on the engine thread alone.
pub fn threads() usize {
    var v = configured.load(.monotonic);
    if (v == unread) {
        v = if (std.c.getenv("TF_FLASHNEXT_HOST_DRAW_THREADS")) |e| parse(std.mem.span(e)) else 0;
        configured.store(v, .monotonic);
    }
    return v;
}

/// Whether a draw of `sm` over a row of `n` scores keeps its working arrays within a worker's scratch (the wider ones draw on the
/// engine thread with the heap).
pub fn fits(sm: ?mtp.Sampling, n: usize) bool {
    const s = sm orelse return true;
    if (s.temperature <= 0) return true;
    const k = if (s.top_k == 0 or s.top_k >= n) n else @as(usize, s.top_k);
    return k <= max_k;
}

const max_k: usize = 256;
/// Bytes of drawRowWith's arrays at k = max_k (vals, keys, top_v, top_c, then choose's order, scaled and probs) with room to spare.
const scratch_bytes: usize = 32 * 1024;

/// One row's draw: the row (masked by `mask` first, if any), the draw's key and, for a draft pick, the token's probability.
pub const Job = struct {
    row: []u16,
    ids: ?[]const u32 = null,
    mask: []const u32 = &.{},
    lse: f32 = 0,
    position: u64,
    sm: ?mtp.Sampling,
    pick: bool = false,
    token: u32 = 0,
    prob: f64 = 0,
    err: ?anyerror = null,

    fn run(j: *Job) void {
        var buf: [scratch_bytes]u8 align(16) = undefined;
        var fba = std.heap.FixedBufferAllocator.init(&buf);
        const a = fba.allocator();
        if (j.mask.len > 0) mtp.maskRow(j.row, j.mask);
        if (j.pick) {
            const p = mtp.drawPickWith(a, j.row, j.ids.?, j.lse, j.position, j.sm, true) catch |e| {
                j.err = e;
                return;
            };
            j.token = p.token;
            j.prob = p.prob;
        } else {
            j.token = mtp.drawRowWith(a, j.row, j.ids, j.position, j.sm, true) catch |e| {
                j.err = e;
                return;
            };
        }
    }
};

const Batch = struct {
    jobs: []Job,
    parts: usize,

    fn part(ctx: *const anyopaque, p: usize) anyerror!void {
        const b: *Batch = @ptrCast(@alignCast(@constCast(ctx)));
        var i = p;
        while (i < b.jobs.len) : (i += b.parts) b.jobs[i].run();
    }
};

var pool: std.atomic.Value(?*ple_pool.Pool) = .init(null);

/// Draws every job, the engine thread and up to threads() workers sharing them; returns when all are done. The first failed job's
/// error is returned.
pub fn run(io: std.Io, jobs: []Job) !void {
    if (jobs.len == 0) return;
    const n = threads();
    var b: Batch = .{ .jobs = jobs, .parts = @max(1, @min(jobs.len, n + 1)) };
    if (b.parts == 1) {
        try Batch.part(&b, 0);
    } else {
        const p = pool.load(.acquire) orelse blk: {
            const made = try ple_pool.Pool.create(std.heap.page_allocator, io);
            break :blk if (pool.cmpxchgStrong(null, made, .acq_rel, .acquire)) |other| other.? else made;
        };
        try p.run(b.parts, .{ .ctx = &b, .run = Batch.part });
    }
    for (jobs) |j| if (j.err) |e| return e;
}

test "jobs draw the rows a single thread draws, across worker counts" {
    const gpa = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(12);
    const n = 5000;
    const rows = try gpa.alloc(u16, 24 * n);
    defer gpa.free(rows);
    const copy = try gpa.alloc(u16, 24 * n);
    defer gpa.free(copy);
    for (rows) |*v| v.* = @as(u16, 0x3f00) + prng.random().uintLessThan(u16, 60);
    @memcpy(copy, rows);
    const ids = try gpa.alloc(u32, n);
    defer gpa.free(ids);
    for (ids, 0..) |*id, i| id.* = @intCast(3 + 7 * i);
    var bits: [n / 32 + 1]u32 = undefined;
    for (&bits) |*w| w.* = prng.random().int(u32) | prng.random().int(u32);
    for ([_]usize{ 0, 1, 3, 7 }) |t| {
        configured.store(@intCast(t), .monotonic);
        for ([_]mtp.Sampling{ .{ .seed = 5 }, .{ .seed = 9, .top_k = 64, .top_p = 0.9, .min_p = 0.02 }, .{ .seed = 2, .temperature = 0 } }) |sm| {
            var jobs: [24]Job = undefined;
            for (&jobs, 0..) |*j, r| j.* = .{ .row = rows[r * n ..][0..n], .ids = if (r % 2 == 0) ids else null, .mask = if (r % 3 == 0) &bits else &.{}, .lse = 1.5, .position = 100 + r, .sm = sm, .pick = r % 2 == 0 };
            try run(std.testing.io, &jobs);
            for (jobs, 0..) |j, r| {
                const row = copy[r * n ..][0..n];
                var plain: [n]u16 = undefined;
                @memcpy(&plain, row);
                if (r % 3 == 0) mtp.maskRow(&plain, &bits);
                if (r % 2 == 0) {
                    const want = try mtp.drawPick(gpa, &plain, ids, 1.5, 100 + r, sm);
                    try std.testing.expectEqual(want.token, j.token);
                    try std.testing.expectEqual(want.prob, j.prob);
                } else {
                    try std.testing.expectEqual(try mtp.drawRow(gpa, &plain, null, 100 + r, sm), j.token);
                }
            }
            // the rows were masked in place for the next case: restore them
            @memcpy(rows, copy);
        }
    }
    configured.store(unread, .monotonic);
}
