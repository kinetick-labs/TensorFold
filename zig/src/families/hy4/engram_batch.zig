//! Engram rows read from the sidecar in parallel. A forward call needs, for every token that owns an n-gram, one
//! FP8 row (and its bf16 scale) per hash head per Engram layer: 4 layers x 16 heads x 2 reads a token. Read one by
//! one they serialize on the disk's latency (a 256-row prompt chunk is ~32k random reads); spread over threads the
//! NVMe queue fills and the page cache serves the rest. Each request decodes its row to f32 in place.

const std = @import("std");

/// One row: `dim` e4m3 bytes at values_at + row * dim, its bf16 scale at scales_at + row * 2, decoded into `out`.
pub const Req = struct {
    fd: std.posix.fd_t,
    values_at: u64,
    scales_at: u64,
    row: u64,
    dim: u32,
    out: [*]f32,
};

pub const max_dim = 512;

/// Reads `reqs` with up to `threads` threads (the caller's own included); a failed read fails the batch.
pub fn readAll(reqs: []const Req, table: *const [256]f32, threads: u32) !void {
    if (reqs.len == 0) return;
    const want: usize = @max(1, @min(@as(usize, threads), (reqs.len + 7) / 8));
    if (want == 1) return range(reqs, table);
    var failed = std.atomic.Value(bool).init(false);
    var pool: [64]std.Thread = undefined;
    const n = @min(want, pool.len);
    const per = (reqs.len + n - 1) / n;
    var spawned: usize = 0;
    defer for (pool[0..spawned]) |t| t.join();
    for (1..n) |i| {
        const lo = i * per;
        if (lo >= reqs.len) break;
        const hi = @min(reqs.len, lo + per);
        pool[spawned] = std.Thread.spawn(.{ .stack_size = 1 << 20 }, worker, .{ reqs[lo..hi], table, &failed }) catch {
            // too many threads: this one reads the slice itself
            range(reqs[lo..hi], table) catch failed.store(true, .release);
            continue;
        };
        spawned += 1;
    }
    range(reqs[0..@min(per, reqs.len)], table) catch failed.store(true, .release);
    for (pool[0..spawned]) |t| t.join();
    spawned = 0;
    if (failed.load(.acquire)) return error.Hy4EngramRead;
}

fn worker(reqs: []const Req, table: *const [256]f32, failed: *std.atomic.Value(bool)) void {
    range(reqs, table) catch failed.store(true, .release);
}

fn range(reqs: []const Req, table: *const [256]f32) !void {
    var buf: [max_dim]u8 = undefined;
    for (reqs) |r| {
        if (r.dim > max_dim) return error.Hy4EngramRead;
        const vb = buf[0..r.dim];
        try preadAll(r.fd, vb, r.values_at + r.row * r.dim);
        var sb: [2]u8 = undefined;
        try preadAll(r.fd, &sb, r.scales_at + r.row * 2);
        const scale: f32 = @bitCast(@as(u32, std.mem.readInt(u16, &sb, .little)) << 16);
        for (r.out[0..r.dim], vb) |*o, b| o.* = table[b] * scale;
    }
}

fn preadAll(fd: std.posix.fd_t, out: []u8, at: u64) !void {
    var done: usize = 0;
    while (done < out.len) {
        const got = std.c.pread(fd, out[done..].ptr, out.len - done, @intCast(at + done));
        if (got <= 0) {
            if (got < 0 and std.c._errno().* == @intFromEnum(std.posix.E.INTR)) continue;
            return error.Hy4EngramRead;
        }
        done += @intCast(got);
    }
}

/// Threads for a batch: HY4_ENGRAM_THREADS, else 32 (an NVMe queue's worth).
pub fn threadsFromEnv() u32 {
    if (std.c.getenv("HY4_ENGRAM_THREADS")) |v| return std.fmt.parseInt(u32, std.mem.span(v), 10) catch 32;
    return 32;
}

test "engram batch: rows decoded as e4m3 x bf16 scale, the same with 1 and many threads, short reads refused" {
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const io = std.testing.io;
    const dim = 24;
    const rows = 300;
    // values then scales, at an offset
    const values_at: u64 = 64;
    const scales_at: u64 = values_at + rows * dim;
    const bytes = try a.alloc(u8, scales_at + rows * 2);
    defer a.free(bytes);
    @memset(bytes[0..values_at], 0xee);
    for (0..rows * dim) |i| bytes[values_at + i] = @intCast((i * 37 + 11) % 256);
    for (0..rows) |r| {
        const sc: f32 = 0.5 + @as(f32, @floatFromInt(r % 7));
        const u: u32 = @bitCast(sc);
        std.mem.writeInt(u16, bytes[scales_at + 2 * r ..][0..2], @intCast(u >> 16), .little);
    }
    try tmp.dir.writeFile(io, .{ .sub_path = "side.bin", .data = bytes });
    const f = try tmp.dir.openFile(io, "side.bin", .{});
    defer f.close(io);
    var table: [256]f32 = undefined;
    for (&table, 0..) |*t, i| t.* = @as(f32, @floatFromInt(i)) - 128; // any distinct values
    const n = 1000;
    const want = try a.alloc(f32, n * dim);
    defer a.free(want);
    const got = try a.alloc(f32, n * dim);
    defer a.free(got);
    const reqs = try a.alloc(Req, n);
    defer a.free(reqs);
    for (reqs, 0..) |*q, i| q.* = .{ .fd = f.handle, .values_at = values_at, .scales_at = scales_at, .row = (i * 7919) % rows, .dim = dim, .out = want[i * dim ..].ptr };
    try readAll(reqs, &table, 1);
    for (0..n) |i| {
        const row = (i * 7919) % rows;
        const sc: f32 = 0.5 + @as(f32, @floatFromInt(row % 7));
        for (0..dim) |c| try std.testing.expectEqual(table[bytes[values_at + row * dim + c]] * sc, want[i * dim + c]);
    }
    for (reqs, 0..) |*q, i| q.out = got[i * dim ..].ptr;
    try readAll(reqs, &table, 16);
    try std.testing.expectEqualSlices(f32, want, got);
    // past the end of the file
    var bad = reqs[0];
    bad.row = rows * 4;
    try std.testing.expectError(error.Hy4EngramRead, readAll(&.{bad}, &table, 1));
}
