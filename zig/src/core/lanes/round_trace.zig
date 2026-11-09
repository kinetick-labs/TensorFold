//! Rounds around held streams (runner fires), one JSON line each: N before a hold, N after its resume; read-only.
const std = @import("std");
const Engine = @import("engine.zig").Engine;
const Stream = @import("stream.zig").Stream;

/// A backend's counters, read before and after each step; a record carries each one's change over its step.
pub const Extras = struct { ctx: *anyopaque, names: []const []const u8, read: *const fn (ctx: *anyopaque, out: []u64) void };

const max_extras = 16;
const max_streams = 64;
const max_ring = 64;

const Kind = enum { forced, drafted, plain };

/// What the trace last saw of a stream: its counters before the step, and where it stands around its fires.
const Seen = struct {
    s: ?*const Stream = null, // compared, never read: the engine frees a stream once it leaves `live`
    rounds: u64 = 0,
    drafted: u64 = 0,
    accepted: u64 = 0,
    emitted: usize = 0,
    forced: bool = false,
    awaiting: bool = false,
    hold_ns: u64 = 0,
    since: i64 = -1, // rounds since its last resume (-1: none yet)
    post: u32 = 0, // rounds after the resume still to write
};

const Rec = struct {
    step: u64,
    t_ns: u64,
    stream: u32,
    since: i64,
    kind: Kind,
    drafted: u64,
    kept: u64,
    got: usize,
    held_next: u32,
    wall_ns: u64,
    gap_ns: u64,
    streams: usize,
    x: [max_extras]u64,
};

var fd: c_int = -1;
var around: u32 = 0;
var extras: ?Extras = null;
var seen: [max_streams]Seen = @splat(.{});
var ring: [max_ring]Rec = undefined;
var ring_len: usize = 0;
var ring_at: usize = 0;
var t_open: u64 = 0;
var step_start: u64 = 0;
var last_end: u64 = 0;
var x_before: [max_extras]u64 = @splat(0);
var buf: [4096]u8 = undefined;

/// Writes to `out` (an open file descriptor): `rounds` rounds before each hold and after each resume (most 64).
pub fn open(out: c_int, rounds: u32, x: ?Extras) void {
    fd = out;
    around = @min(rounds, max_ring);
    extras = if (x) |v| if (v.names.len <= max_extras) v else null else null;
    t_open = now();
    last_end = t_open;
    var w = std.Io.Writer.fixed(&buf);
    w.print("{{\"rt\":\"open\",\"rounds\":{d},\"extras\":[", .{around}) catch return;
    if (extras) |v| for (v.names, 0..) |n, i| w.print("{s}\"{s}\"", .{ if (i > 0) "," else "", n }) catch return;
    w.writeAll("]}\n") catch return;
    put(w.buffered());
}

/// Off on Windows at compile time: the trace writes through libc there is none of.
pub inline fn on() bool {
    return @import("builtin").os.tag != .windows and fd >= 0;
}

fn now() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

fn put(s: []const u8) void {
    var at: usize = 0;
    while (at < s.len) {
        const n = std.c.write(fd, s[at..].ptr, s.len - at);
        if (n <= 0) return;
        at += @intCast(n);
    }
}

fn readExtras(out: []u64) void {
    if (extras) |v| v.read(v.ctx, out[0..v.names.len]);
}

fn live(e: *const Engine, p: *const Stream) bool {
    for (e.live.items) |s| if (s == p) return true;
    return false;
}

/// Frees entries of streams no longer live (pointers compared only): a new stream at a freed address starts fresh.
fn forget(e: *const Engine) void {
    for (&seen) |*x| if (x.s) |p| if (!live(e, p)) {
        x.* = .{};
    };
}

/// The live stream's entry (a new one takes an entry no live stream holds; null when all are taken).
fn entry(e: *const Engine, s: *const Stream) ?*Seen {
    var free: ?*Seen = null;
    for (&seen) |*x| {
        if (x.s == s) return x;
        if (free == null and (x.s == null or !live(e, x.s.?))) free = x;
    }
    const x = free orelse return null;
    x.* = .{ .s = s, .rounds = s.rounds, .drafted = s.drafted, .accepted = s.accepted, .emitted = s.emitted().len };
    return x;
}

/// Before a step: the clock, the backend's counters, each live stream's; a stream resumed since its hold is noted.
pub fn before(e: *const Engine) void {
    step_start = now();
    readExtras(&x_before);
    forget(e);
    for (e.live.items) |s| {
        const x = entry(e, s) orelse continue;
        if (x.awaiting and !s.awaiting) {
            var w = std.Io.Writer.fixed(&buf);
            w.print("{{\"rt\":\"resume\",\"stream\":{d},\"t_us\":{d},\"held_us\":{d},\"forced\":{d}}}\n", .{ index(x), us(step_start - t_open), us(step_start - x.hold_ns), s.force.items.len }) catch {};
            put(w.buffered());
            x.since = 0;
            x.post = around;
        }
        x.awaiting = s.awaiting;
        x.rounds = s.rounds;
        x.drafted = s.drafted;
        x.accepted = s.accepted;
        x.emitted = s.emitted().len;
        x.forced = s.force.items.len > 0;
    }
}

/// After a step: each stream that ran a round, written when it is near a fire, else kept as the next fire's before.
pub fn after(e: *const Engine) void {
    const t = now();
    var x_after: [max_extras]u64 = @splat(0);
    readExtras(&x_after);
    defer for (e.live.items) |s| {
        // a finished stream leaves `live` in this step's sweep, then is freed: its entry goes now, while it is live
        if (s.finished) if (find(s)) |x| {
            x.* = .{};
        };
    };
    for (e.live.items) |s| {
        const x = entry(e, s) orelse continue;
        if (s.rounds == x.rounds) continue;
        if (x.since >= 0) x.since += 1;
        var r: Rec = .{ .step = e.steps, .t_ns = t - t_open, .stream = index(x), .since = x.since, .kind = if (x.forced) .forced else if (s.drafted > x.drafted) .drafted else .plain, .drafted = s.drafted - x.drafted, .kept = s.accepted - x.accepted, .got = s.emitted().len - x.emitted, .held_next = if (s.next) |n| n.count else 0, .wall_ns = t - step_start, .gap_ns = step_start - last_end, .streams = e.live.items.len, .x = undefined };
        for (&r.x, x_after, x_before) |*d, b, a| d.* = b -% a;
        x.rounds = s.rounds;
        if (s.awaiting and !x.awaiting) {
            flush();
            write(r, "trigger");
            var w = std.Io.Writer.fixed(&buf);
            w.print("{{\"rt\":\"hold\",\"stream\":{d},\"t_us\":{d}}}\n", .{ index(x), us(t - t_open) }) catch {};
            put(w.buffered());
            x.awaiting = true;
            x.hold_ns = t;
            x.post = 0;
        } else if (x.post > 0) {
            write(r, "post");
            x.post -= 1;
        } else push(r);
    }
    last_end = now();
}

fn find(s: *const Stream) ?*Seen {
    for (&seen) |*x| if (x.s == s) return x;
    return null;
}

/// Entries held (tests).
pub fn held() usize {
    var n: usize = 0;
    for (seen) |x| n += @intFromBool(x.s != null);
    return n;
}

fn index(x: *const Seen) u32 {
    return @intCast((@intFromPtr(x) - @intFromPtr(&seen)) / @sizeOf(Seen));
}

fn us(ns: u64) u64 {
    return ns / std.time.ns_per_us;
}

fn push(r: Rec) void {
    if (around == 0) return;
    ring[(ring_at + ring_len) % max_ring] = r;
    if (ring_len < around) ring_len += 1 else ring_at = (ring_at + 1) % max_ring;
}

/// The rounds kept since the last write, as a hold's before.
fn flush() void {
    for (0..ring_len) |i| write(ring[(ring_at + i) % max_ring], "pre");
    ring_len = 0;
    ring_at = 0;
}

fn write(r: Rec, phase: []const u8) void {
    var w = std.Io.Writer.fixed(&buf);
    w.print("{{\"rt\":\"round\",\"phase\":\"{s}\",\"step\":{d},\"t_us\":{d},\"stream\":{d},\"since\":{d},\"kind\":\"{s}\",\"drafted\":{d},\"kept\":{d},\"got\":{d},\"held_next\":{d},\"wall_us\":{d},\"gap_us\":{d},\"streams\":{d}", .{ phase, r.step, us(r.t_ns), r.stream, r.since, @tagName(r.kind), r.drafted, r.kept, r.got, r.held_next, us(r.wall_ns), us(r.gap_ns), r.streams }) catch return;
    if (extras) |v| for (v.names, r.x[0..v.names.len]) |n, d| w.print(",\"{s}\":{d}", .{ n, d }) catch return;
    w.writeAll("}\n") catch return;
    put(w.buffered());
}

/// Closes the trace (tests): the descriptor is the caller's.
pub fn close() void {
    fd = -1;
    extras = null;
    seen = @splat(.{});
    ring_len = 0;
    ring_at = 0;
}
