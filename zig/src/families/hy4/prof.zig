//! HY4_PROFILE=N: where a forward step's time goes. Timing events are recorded on the engine stream after each op
//! group (`mark`); after the step the time between consecutive events is added to the group that ended there, and the
//! host-side phases (Engram reads, uploads, sampling) are timed on the clock. Every N steps of a kind (decode rows
//! only, or with prompt rows) one log line gives the mean milliseconds per step of each group, largest first. The
//! step waits for its last event, so a profiled run overlaps host and device less than a plain one.

const std = @import("std");
const cuda = @import("cuda");

pub const Op = enum { start, upload, embed, engram, attn_proj, rope_store, indexer, attention, attn_out, router, experts, shared, dense, logits };
pub const Host = enum { schedule, engram_read, sample };
pub const Kind = enum { decode, prefill };

const n_ops = std.meta.fieldNames(Op).len;
const n_host = std.meta.fieldNames(Host).len;

pub fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

pub const Prof = struct {
    s: cuda.Stream,
    ev: []cuda.Event,
    tag: []Op,
    a: std.mem.Allocator,
    n: usize = 0,
    dropped: u64 = 0,
    every: u32,
    t0: u64 = 0,
    gpu: [2][n_ops]f64 = @splat(@splat(0)),
    host: [2][n_host]f64 = @splat(@splat(0)),
    wall: [2]f64 = .{ 0, 0 },
    steps: [2]u32 = .{ 0, 0 },
    rows: [2]u64 = .{ 0, 0 },

    /// `cap` events a step at most (later marks of a step are dropped and counted).
    pub fn init(a: std.mem.Allocator, d: *const cuda.Driver, s: cuda.Stream, cap: usize, every: u32) !Prof {
        const ev = try a.alloc(cuda.Event, cap);
        errdefer a.free(ev);
        var made: usize = 0;
        errdefer for (ev[0..made]) |*e| e.deinit();
        while (made < cap) : (made += 1) ev[made] = try cuda.Event.init(d, true);
        return .{ .s = s, .ev = ev, .tag = try a.alloc(Op, cap), .a = a, .every = @max(every, 1) };
    }

    pub fn deinit(p: *Prof) void {
        for (p.ev) |*e| e.deinit();
        p.a.free(p.ev);
        p.a.free(p.tag);
    }

    /// The step begins: its first event and the wall clock.
    pub fn start(p: *Prof) !void {
        p.n = 0;
        p.t0 = nowNs();
        try p.mark(.start);
    }

    /// The op group `op` ends here on the stream.
    pub fn mark(p: *Prof, op: Op) !void {
        if (p.n == p.ev.len) {
            p.dropped += 1;
            return;
        }
        try p.ev[p.n].record(p.s);
        p.tag[p.n] = op;
        p.n += 1;
    }

    pub fn addHost(p: *Prof, h: Host, kind: Kind, ns: u64) void {
        p.host[@intFromEnum(kind)][@intFromEnum(h)] += @as(f64, @floatFromInt(ns)) / 1e6;
    }

    /// The step of `rows` rows ended: wait for its last event, add up, log every `every` steps of the kind.
    pub fn finish(p: *Prof, kind: Kind, rows: u32) !void {
        if (p.n == 0) return;
        try p.ev[p.n - 1].synchronize();
        const k = @intFromEnum(kind);
        for (1..p.n) |i| p.gpu[k][@intFromEnum(p.tag[i])] += try cuda.Event.elapsedMs(p.ev[i - 1], p.ev[i]);
        p.wall[k] += @as(f64, @floatFromInt(nowNs() - p.t0)) / 1e6;
        p.steps[k] += 1;
        p.rows[k] += rows;
        p.n = 0;
        if (p.steps[k] >= p.every) p.report(kind);
    }

    fn report(p: *Prof, kind: Kind) void {
        const k = @intFromEnum(kind);
        const steps: f64 = @floatFromInt(p.steps[k]);
        var order: [n_ops]usize = undefined;
        for (&order, 0..) |*o, i| o.* = i;
        const g = p.gpu[k];
        std.mem.sort(usize, &order, g, struct {
            fn lt(gg: [n_ops]f64, x: usize, y: usize) bool {
                return gg[x] > gg[y];
            }
        }.lt);
        var gpu_total: f64 = 0;
        for (g) |v| gpu_total += v;
        var buf: [1024]u8 = undefined;
        var w: std.Io.Writer = .fixed(&buf);
        w.print("hy4 profile {s}: {d} steps, {d:.1} rows a step, wall {d:.2} ms a step, device {d:.2}:", .{ @tagName(kind), p.steps[k], @as(f64, @floatFromInt(p.rows[k])) / steps, p.wall[k] / steps, gpu_total / steps }) catch {};
        for (order) |i| {
            if (g[i] <= 0) continue;
            w.print(" {s} {d:.2}", .{ @tagName(@as(Op, @enumFromInt(i))), g[i] / steps }) catch {};
        }
        w.writeAll("; host") catch {};
        for (p.host[k], 0..) |v, i| w.print(" {s} {d:.2}", .{ @tagName(@as(Host, @enumFromInt(i))), v / steps }) catch {};
        if (p.dropped > 0) w.print("; {d} marks dropped", .{p.dropped}) catch {};
        std.log.info("{s}", .{w.buffered()});
        p.gpu[k] = @splat(0);
        p.host[k] = @splat(0);
        p.wall[k] = 0;
        p.steps[k] = 0;
        p.rows[k] = 0;
    }
};

test "prof: op and host names stay printable and the time unit is milliseconds" {
    try std.testing.expect(nowNs() > 0);
    try std.testing.expectEqualStrings("attention", @tagName(Op.attention));
    try std.testing.expectEqual(@as(usize, 3), n_host);
}
