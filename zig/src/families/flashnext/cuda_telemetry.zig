//! Draft-chain telemetry for the Flash Next CUDA engine. Off unless `TF_FLASHNEXT_TELEMETRY=1`.
//!
//! Armed once, after the startup cost measurement (`cuda_native.zig`), so that measurement is unchanged.
//! While off, callers take one false branch and do not time, allocate, or print. The depth rule's only
//! off-path cost is a null check on `lanes.depth.watch`, which stays null until `arm`.
//!
//! Grep `tf-draft`. One `tf-draft round` line per stream per round, one `tf-draft costs` line the first
//! time `depth()` runs, and a `tf-draft sum` line every 32 stream-rounds and at shutdown.
const std = @import("std");
const depth = @import("lanes").depth;

pub const max_levels = 16;
const max_streams = 64;
const sum_every: u64 = 32;

const Chain = struct {
    stream: ?*anyopaque = null,
    asked: i64 = 0,
    why: []const u8 = "-",
    stop: []const u8 = "depth",
    p: [max_levels]f64 = @splat(0),
    ms: [max_levels]f64 = @splat(-1),
    n: u8 = 0,
};

const Win = struct {
    stream: ?*anyopaque,
    acc: u16,
    spec: u16,
    verify_ms: f64 = 0,
    gap_verify: f64 = 0,
    printed: bool = false,
};

const Decision = struct {
    stream: ?*anyopaque,
    why: []const u8,
};

var enabled = false;
var armed = false;
var costs_done = false;
var dumped = false;
var last: ?std.Io.Timestamp = null;
var gap_verify_ms: f64 = 0;
var gap_draft_ms: f64 = 0;
var level_batch: f64 = 0;
var prepared = false;

var chains: [max_streams]Chain = undefined;
var n_chains: usize = 0;
var wins: [max_streams]Win = undefined;
var n_wins: usize = 0;
var decisions: [max_streams]Decision = undefined;
var n_decisions: usize = 0;

var forwards: u64 = 0;
var rounds: u64 = 0;
var tokens: u64 = 0;
var verify_sum: f64 = 0;
var draft_sum: f64 = 0;
var hit: [max_levels]u64 = @splat(0);
var reached: [max_levels]u64 = @splat(0);
var hist: [max_levels + 1]u64 = @splat(0);

var sink: u8 = 0;
var installed: depth.Watch = undefined;

pub fn on() bool {
    return enabled;
}

/// `TF_FLASHNEXT_TELEMETRY=1` and nothing else. Read once from `arm`.
pub fn enabledValue(text: ?[]const u8) bool {
    const v = text orelse return false;
    return std.mem.eql(u8, v, "1");
}

/// Install the depth-rule watch. Idempotent. Call after the startup cost measurement.
pub fn arm() void {
    if (armed) return;
    armed = true;
    const text = if (std.c.getenv("TF_FLASHNEXT_TELEMETRY")) |p| std.mem.span(p) else null;
    if (!enabledValue(text)) return;
    enabled = true;
    installed = .{ .ctx = &sink, .choose = choose, .table = table };
    depth.watch = &installed;
}

pub fn now(io: std.Io) std.Io.Timestamp {
    return std.Io.Timestamp.now(io, .awake);
}

pub fn span(a: std.Io.Timestamp, b: std.Io.Timestamp) f64 {
    const d = a.durationTo(b);
    const ms = @as(f64, @floatFromInt(d.nanoseconds)) / 1e6;
    return if (ms > 0) ms else 0;
}

pub const Gap = enum { verify, draft };

/// Host gap since the previous phase ended, and this phase's start.
pub fn markGap(kind: Gap, t: std.Io.Timestamp) void {
    if (!enabled) return;
    const g = if (last) |p| span(p, t) else 0;
    last = t;
    switch (kind) {
        .verify => gap_verify_ms = g,
        .draft => gap_draft_ms = g,
    }
}

pub fn park(t: std.Io.Timestamp) void {
    if (enabled) last = t;
}

pub fn beginVerify() void {
    if (!enabled) return;
    flush();
    n_wins = 0;
}

pub fn noteWindow(stream: ?*anyopaque, acc: usize, spec: usize) void {
    if (!enabled or n_wins == max_streams) return;
    wins[n_wins] = .{
        .stream = stream,
        .acc = @intCast(@min(acc, std.math.maxInt(u16))),
        .spec = @intCast(@min(spec, std.math.maxInt(u16))),
    };
    n_wins += 1;
}

pub fn endVerify(ms: f64, gpu_gap: ?f64) void {
    if (!enabled) return;
    if (gpu_gap) |g| gap_verify_ms = g;
    const before = rounds;
    forwards += 1;
    verify_sum += ms;
    for (wins[0..n_wins]) |*w| {
        w.verify_ms = ms;
        w.gap_verify = gap_verify_ms;
        rounds += 1;
        const out: usize = @as(usize, w.acc) + 1;
        tokens += out;
        if (out < hist.len) hist[out] += 1;
        var j: usize = 0;
        while (j < w.spec and j <= w.acc and j < max_levels) : (j += 1) {
            reached[j] += 1;
            if (j < w.acc) hit[j] += 1;
        }
    }
    if (before / sum_every != rounds / sum_every) printSum(rounds, false);
}

pub fn isPrepared() bool {
    return prepared;
}

/// A draft that failed before `finish`: drop the chain so the next round starts clean.
pub fn abandon() void {
    if (!enabled) return;
    prepared = false;
    n_chains = 0;
    level_batch = 0;
}

pub fn prepare(n: usize) void {
    if (!enabled) return;
    n_chains = @min(n, max_streams);
    level_batch = 0;
    for (chains[0..n_chains]) |*c| c.* = .{};
    prepared = true;
}

pub fn bind(i: usize, stream: ?*anyopaque, asked: i64) void {
    if (!enabled or i >= n_chains) return;
    chains[i].stream = stream;
    chains[i].asked = asked;
    chains[i].why = whyOf(stream);
}

pub fn onDraw(i: usize, p: f64, conf: f64, propose: bool, more: bool, capped: bool) void {
    if (!enabled or i >= n_chains) return;
    const c = &chains[i];
    if (c.n < max_levels) {
        c.p[c.n] = p;
        c.ms[c.n] = -1;
        c.n += 1;
    }
    c.stop = stopOf(conf, propose, more, capped);
}

pub fn endLevel(ms: f64) void {
    if (!enabled) return;
    level_batch += ms;
    for (chains[0..n_chains]) |*c| {
        if (c.n == 0) continue;
        const j = c.n - 1;
        if (c.ms[j] < 0) c.ms[j] = ms;
    }
}

pub fn finish(held: []const usize, gpu_gap: ?f64) void {
    if (!enabled) return;
    if (gpu_gap) |g| gap_draft_ms = g;
    draft_sum += level_batch;
    level_batch = 0;
    for (chains[0..n_chains], 0..) |*c, i| {
        const h = if (i < held.len) held[i] else 0;
        if (c.n == 0) c.stop = "absorb";
        const w = findWin(c.stream, i);
        var lvl: [max_levels]f64 = undefined;
        var ln: usize = 0;
        var draft_ms: f64 = 0;
        for (c.ms[0..c.n]) |m| if (m >= 0) {
            lvl[ln] = m;
            ln += 1;
            draft_ms += m;
        };
        var buf: [2048]u8 = undefined;
        const line = writeRound(&buf, .{
            .i = i,
            .asked = c.asked,
            .held = h,
            .acc = if (w) |win| win.acc else null,
            .spec = if (w) |win| win.spec else null,
            .why = c.why,
            .stop = c.stop,
            .verify_ms = if (w) |win| win.verify_ms else null,
            .draft_ms = if (c.n == 0) null else draft_ms,
            .gap_verify = if (w) |win| win.gap_verify else null,
            .gap_draft = gap_draft_ms,
            .p = c.p[0..c.n],
            .lvl = lvl[0..ln],
        });
        if (w) |win| win.printed = true;
        std.debug.print("{s}\n", .{line});
    }
    prepared = false;
    n_chains = 0;
}

pub fn shutdown() void {
    if (!enabled or dumped) return;
    dumped = true;
    flush();
    if (rounds == 0 and forwards == 0 and !costs_done) return;
    printSum(rounds, true);
}

/// Why a sampled level ended the chain. A level that may continue reports `depth` until a later draw says otherwise.
pub fn stopOf(conf: f64, propose: bool, more: bool, capped: bool) []const u8 {
    if (!propose or !more) {
        if (conf > 0) return "confidence";
        if (conf < 0) return "product";
        return "depth";
    }
    if (capped) return "depth";
    return "depth";
}

pub const View = struct {
    i: usize,
    asked: i64,
    held: usize,
    acc: ?u16 = null,
    spec: ?u16 = null,
    why: []const u8,
    stop: []const u8,
    verify_ms: ?f64 = null,
    draft_ms: ?f64 = null,
    gap_verify: ?f64 = null,
    gap_draft: ?f64 = null,
    p: []const f64 = &.{},
    lvl: []const f64 = &.{},
};

pub fn writeRound(buf: []u8, v: View) []const u8 {
    var n = put(buf, 0, "tf-draft round i={d} asked={d} held={d} ", .{ v.i, v.asked, v.held });
    n = putOptInt(buf, n, "acc", v.acc);
    if (v.acc) |a| n = put(buf, n, "out={d} ", .{@as(usize, a) + 1}) else n = put(buf, n, "out=- ", .{});
    n = putOptInt(buf, n, "win", v.spec);
    n = put(buf, n, "why={s} stop={s} ", .{ v.why, v.stop });
    n = putOptMs(buf, n, "verify_ms", v.verify_ms);
    n = putOptMs(buf, n, "draft_ms", v.draft_ms);
    n = putOptMs(buf, n, "gap_verify_ms", v.gap_verify);
    n = putOptMs(buf, n, "gap_draft_ms", v.gap_draft);
    n = putFloats(buf, n, "p", v.p);
    n = putFloats(buf, n, "lvl", v.lvl);
    if (n > 0 and buf[n - 1] == ' ') n -= 1;
    return buf[0..n];
}

fn choose(_: *anyopaque, stream: ?*anyopaque, _: i64, reason: []const u8) void {
    if (!enabled) return;
    for (decisions[0..n_decisions]) |*d| if (d.stream == stream) {
        d.why = reason;
        return;
    };
    if (n_decisions == max_streams) return;
    decisions[n_decisions] = .{ .stream = stream, .why = reason };
    n_decisions += 1;
}

fn table(_: *anyopaque, rule: *const depth.Rule) void {
    if (!enabled or costs_done) return;
    costs_done = true;
    var buf: [2048]u8 = undefined;
    var n: usize = put(&buf, 0, "tf-draft costs family=", .{});
    n = appendSlots(&buf, n, rule.family_costs.slots);
    n = put(&buf, n, " mtp_step_ms={d:.3} prior=", .{rule.mtp_step_ms});
    n = appendList(&buf, n, rule.prior[0..@min(rule.prior.len, max_levels)]);
    n = put(&buf, n, " head_stops={s} most={d} plain_guard={s}", .{
        if (rule.head_stops) "1" else "0",
        rule.most_drafts,
        if (rule.plain_guard) "1" else "0",
    });
    std.debug.print("{s}\n", .{buf[0..n]});
}

fn whyOf(stream: ?*anyopaque) []const u8 {
    for (decisions[0..n_decisions]) |d| if (d.stream == stream) return d.why;
    return "-";
}

fn findWin(stream: ?*anyopaque, i: usize) ?*Win {
    if (stream) |s| for (wins[0..n_wins]) |*w| if (!w.printed and w.stream == s) return w;
    if (stream == null and i < n_wins and !wins[i].printed) return &wins[i];
    return null;
}

fn flush() void {
    for (wins[0..n_wins]) |*w| {
        if (w.printed) continue;
        var buf: [512]u8 = undefined;
        const line = writeRound(&buf, .{
            .i = 0,
            .asked = 0,
            .held = 0,
            .acc = w.acc,
            .spec = w.spec,
            .why = "-",
            .stop = "-",
            .verify_ms = w.verify_ms,
            .gap_verify = w.gap_verify,
        });
        w.printed = true;
        std.debug.print("{s}\n", .{line});
    }
    n_wins = 0;
}

fn printSum(at: u64, stop: bool) void {
    var buf: [1024]u8 = undefined;
    var n: usize = if (stop)
        put(&buf, 0, "tf-draft sum at=stop rounds={d} ", .{at})
    else
        put(&buf, 0, "tf-draft sum at={d} rounds={d} ", .{ at, at });
    const tok = if (rounds == 0) 0 else @as(f64, @floatFromInt(tokens)) / @as(f64, @floatFromInt(rounds));
    const denom = @as(f64, @floatFromInt(@max(forwards, 1)));
    n = put(&buf, n, "tok_per_round={d:.3} ", .{tok});
    n = putRates(&buf, n);
    n = putHist(&buf, n);
    n = put(&buf, n, "ms_per_round={d:.3} verify_ms={d:.3} draft_ms={d:.3}", .{
        (verify_sum + draft_sum) / denom,
        verify_sum / denom,
        draft_sum / denom,
    });
    std.debug.print("{s}\n", .{buf[0..n]});
}

fn putRates(buf: []u8, n0: usize) usize {
    var n = put(buf, n0, "acc=", .{});
    var any = false;
    var levels: usize = 0;
    for (reached, hit) |r, h| {
        if (r == 0) break;
        const rate = @as(f64, @floatFromInt(h)) / @as(f64, @floatFromInt(r));
        n = put(buf, n, "{s}{d:.3}", .{ if (any) "," else "", rate });
        any = true;
        levels += 1;
    }
    if (!any) n = put(buf, n, "-", .{});
    n = put(buf, n, " n=", .{});
    if (levels == 0) return put(buf, n, "- ", .{});
    for (reached[0..levels], 0..) |r, i| n = put(buf, n, "{s}{d}", .{ if (i > 0) "," else "", r });
    return put(buf, n, " ", .{});
}

fn putHist(buf: []u8, n0: usize) usize {
    var n = put(buf, n0, "hist=", .{});
    var any = false;
    for (hist, 0..) |c, k| {
        if (c == 0) continue;
        n = put(buf, n, "{s}{d}:{d}", .{ if (any) "," else "", k, c });
        any = true;
    }
    if (!any) n = put(buf, n, "-", .{});
    return put(buf, n, " ", .{});
}

fn appendSlots(buf: []u8, n0: usize, slots: []const ?f64) usize {
    var n = n0;
    var any = false;
    for (slots, 0..) |s, k| {
        const v = s orelse continue;
        n = put(buf, n, "{s}{d}:{d:.3}", .{ if (any) "," else "", k, v });
        any = true;
        if (n + 32 >= buf.len) break;
    }
    if (!any) n = put(buf, n, "-", .{});
    return n;
}

fn appendList(buf: []u8, n0: usize, xs: []const f64) usize {
    if (xs.len == 0) return put(buf, n0, "-", .{});
    var n = n0;
    for (xs, 0..) |v, i| n = put(buf, n, "{s}{d:.3}", .{ if (i > 0) "," else "", v });
    return n;
}

fn put(buf: []u8, n: usize, comptime fmt: []const u8, args: anytype) usize {
    if (n >= buf.len) return buf.len;
    const wrote = std.fmt.bufPrint(buf[n..], fmt, args) catch return buf.len;
    return n + wrote.len;
}

fn putOptInt(buf: []u8, n: usize, name: []const u8, v: ?u16) usize {
    return if (v) |x| put(buf, n, "{s}={d} ", .{ name, x }) else put(buf, n, "{s}=- ", .{name});
}

fn putOptMs(buf: []u8, n: usize, name: []const u8, v: ?f64) usize {
    return if (v) |x| put(buf, n, "{s}={d:.3} ", .{ name, x }) else put(buf, n, "{s}=- ", .{name});
}

fn putFloats(buf: []u8, n0: usize, name: []const u8, xs: []const f64) usize {
    var n = put(buf, n0, "{s}=", .{name});
    if (xs.len == 0) return put(buf, n, "- ", .{});
    for (xs, 0..) |v, i| n = put(buf, n, "{s}{d:.3}", .{ if (i > 0) "," else "", v });
    return put(buf, n, " ", .{});
}

fn clear() void {
    enabled = false;
    costs_done = false;
    dumped = false;
    last = null;
    gap_verify_ms = 0;
    gap_draft_ms = 0;
    level_batch = 0;
    prepared = false;
    n_chains = 0;
    n_wins = 0;
    n_decisions = 0;
    forwards = 0;
    rounds = 0;
    tokens = 0;
    verify_sum = 0;
    draft_sum = 0;
    hit = @splat(0);
    reached = @splat(0);
    hist = @splat(0);
}

test "the flag is the string 1" {
    try std.testing.expect(enabledValue("1"));
    try std.testing.expect(!enabledValue(null));
    try std.testing.expect(!enabledValue("0"));
    try std.testing.expect(!enabledValue(""));
    try std.testing.expect(!enabledValue("11"));
}

test "stop reasons follow the confidence sign" {
    try std.testing.expectEqualStrings("confidence", stopOf(0.7, false, false, false));
    try std.testing.expectEqualStrings("product", stopOf(-0.4, true, false, false));
    try std.testing.expectEqualStrings("depth", stopOf(0, true, true, true));
    try std.testing.expectEqualStrings("depth", stopOf(-0.4, true, true, true));
}

test "a round line names the chain and the verify" {
    var buf: [512]u8 = undefined;
    const line = writeRound(&buf, .{
        .i = 0,
        .asked = 15,
        .held = 2,
        .acc = 3,
        .spec = 8,
        .why = "head_stops",
        .stop = "product",
        .verify_ms = 4.21,
        .draft_ms = 1.5,
        .gap_verify = 0.04,
        .gap_draft = 0.02,
        .p = &.{ 0.91, 0.4 },
        .lvl = &.{ 1.0, 0.5 },
    });
    try std.testing.expect(std.mem.startsWith(u8, line, "tf-draft round "));
    try std.testing.expect(std.mem.indexOf(u8, line, "asked=15") != null);
    try std.testing.expect(std.mem.indexOf(u8, line, "held=2") != null);
    try std.testing.expect(std.mem.indexOf(u8, line, "acc=3") != null);
    try std.testing.expect(std.mem.indexOf(u8, line, "out=4") != null);
    try std.testing.expect(std.mem.indexOf(u8, line, "why=head_stops") != null);
    try std.testing.expect(std.mem.indexOf(u8, line, "stop=product") != null);
    try std.testing.expect(std.mem.indexOf(u8, line, "p=0.910,0.400") != null);
    try std.testing.expect(std.mem.indexOf(u8, line, "\n") == null);
}

test "acceptance counts a hit per kept level and a miss at the break" {
    enabled = true;
    defer clear();
    clear();
    enabled = true;
    beginVerify();
    noteWindow(null, 3, 8);
    endVerify(4, null);
    try std.testing.expectEqual(@as(u64, 1), hit[0]);
    try std.testing.expectEqual(@as(u64, 1), hit[2]);
    try std.testing.expectEqual(@as(u64, 0), hit[3]);
    try std.testing.expectEqual(@as(u64, 1), reached[3]);
    try std.testing.expectEqual(@as(u64, 0), reached[4]);
    try std.testing.expectEqual(@as(u64, 4), tokens);
    try std.testing.expectEqual(@as(u64, 1), hist[4]);
    try std.testing.expectEqual(@as(u64, 1), forwards);
}

test "a product stop keeps the failing probability on the chain" {
    enabled = true;
    defer clear();
    clear();
    enabled = true;
    prepare(1);
    onDraw(0, 0.9, -0.4, true, true, false);
    endLevel(1);
    onDraw(0, 0.2, -0.4, false, false, false);
    try std.testing.expectEqualStrings("product", chains[0].stop);
    try std.testing.expectEqual(@as(u8, 2), chains[0].n);
    try std.testing.expectApproxEqAbs(@as(f64, 0.2), chains[0].p[1], 1e-12);
}
