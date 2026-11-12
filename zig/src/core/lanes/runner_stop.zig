//! The runner's digit stop in reasoning: no drawn digit right after an owned inline integer, no digit run past a cap.
const std = @import("std");
const be = @import("backend.zig");
const Stream = @import("stream.zig").Stream;
const Allocator = std.mem.Allocator;

/// A drawn token the runner replaces (an owned value is whole: a digit after it is a runaway): where, and its newline.
pub const Stop = struct { at: usize, token: u32, why: Why };

pub const Why = enum { after_value, run_cap };

/// PREFIX_DIGIT_STOP, _RUN_MAX, _HOLD_MAX (runner.zig); 0: no cap. hold_max caps runs once a stop fired in the reply.
pub const Limits = struct { after_value: bool = true, run_max: u32 = 1024, hold_max: u32 = 64 };

/// One token's bytes, as the runner's tokenizer gives them.
pub const Bytes = struct {
    ctx: ?*anyopaque = null,
    of: *const fn (ctx: ?*anyopaque, token: u32) []const u8,
};

fn digit(c: u8) bool {
    return c >= '0' and c <= '9';
}

fn leading(b: []const u8) u32 {
    var n: u32 = 0;
    while (n < b.len and digit(b[n])) n += 1;
    return n;
}

/// The digit run ending the text after `bytes` lands on a run of `run`.
fn extend(run: u32, b: []const u8) u32 {
    if (b.len == 0) return run;
    var n: u32 = 0;
    while (n < b.len and digit(b[b.len - 1 - n])) n += 1;
    return if (n == b.len) run +| n else n;
}

/// A reply's stop state, moved on by each committed token (`landed`) and asked about a round's tokens (`stop`).
pub const State = struct {
    limits: Limits = .{},
    newline: ?u32 = null, // the token put in a stopped token's place (null: the stop is off)
    looked: bool = false, // `look` asked the tokenizer for it (once: one token or none)
    lead: ?usize = null, // armed: owned tokens still to land; at 0 the next token is the model's first after the value
    run: u32 = 0, // digits ending the reasoning's committed text
    held: bool = false, // a stop fired in this reply: later runs cap at hold_max

    /// The newline token from `text` (its `encode`, as the runner's Text), asked once.
    pub fn look(st: *State, gpa: Allocator, text: anytype) Allocator.Error!void {
        if (st.looked) return;
        st.looked = true;
        const ids = try text.encode(text.ctx, gpa, "\n");
        defer gpa.free(ids);
        if (ids.len == 1) st.newline = ids[0];
    }

    /// Whether the run nears its cap with the newline not yet asked for (the runner then calls `look`).
    pub fn wants(st: *const State) bool {
        const c = st.cap();
        return !st.looked and c > 0 and st.run +| 64 > c;
    }

    /// The digit-run cap now (0: none): run_max, or hold_max once a stop fired.
    pub fn cap(st: *const State) u32 {
        const h = st.limits.hold_max;
        if (!st.held or h == 0) return st.limits.run_max;
        return if (st.limits.run_max == 0) h else @min(st.limits.run_max, h);
    }

    /// A stop this state returned was applied: the reply's later runs hold at hold_max.
    pub fn fired(st: *State) void {
        st.held = true;
    }

    /// An owned inline value is forced next: `count` tokens whose text is `value` (armed when it ends in a digit).
    pub fn arm(st: *State, gpa: Allocator, text: anytype, count: usize, value: []const u8) Allocator.Error!void {
        if (!st.limits.after_value or count == 0 or value.len == 0 or !digit(value[value.len - 1])) return;
        try st.look(gpa, text);
        if (st.newline != null) st.lead = count;
    }

    /// A committed token's bytes; outside reasoning there is no run and nothing stays armed.
    pub fn landed(st: *State, bytes: []const u8, think_open: bool) void {
        if (st.lead) |n| st.lead = if (n > 0 and think_open) n - 1 else null;
        st.run = if (think_open) extend(st.run, bytes) else 0;
    }

    /// The first of `tokens` (reasoning only; the first `forced` are forced rows) the stop replaces, or null.
    pub fn stop(st: *const State, bytes: Bytes, tokens: []const u32, forced: usize) ?Stop {
        const nl = st.newline orelse return null;
        var lead = st.lead;
        var run = st.run;
        const c = st.cap();
        for (tokens, 0..) |t, i| {
            const b = bytes.of(bytes.ctx, t);
            const owned = if (lead) |n| n > 0 else false;
            if (!owned and i >= forced and b.len > 0 and digit(b[0])) {
                if (lead != null) return .{ .at = i, .token = nl, .why = .after_value };
                if (c > 0 and run +| leading(b) > c) return .{ .at = i, .token = nl, .why = .run_cap };
            }
            if (lead) |n| lead = if (n > 0) n - 1 else null;
            run = extend(run, b);
        }
        return null;
    }

    /// Whether the next draw could be stopped: a pipelined stream reads it before feeding it.
    pub fn watching(st: *const State) bool {
        if (st.newline == null) return false;
        const c = st.cap();
        return st.lead != null or (c > 0 and st.run +| 16 > c);
    }
};

/// The stop's cut in a round's committed tokens: reasoning rows before its end, never while forced rows wait.
pub fn cut(s: *Stream, tokens: []const u32) ?Stop {
    const r = s.runner orelse return null;
    const f = r.stop orelse return null;
    if (!s.think_open or s.force.items.len > 0) return null;
    var end = tokens.len;
    for (tokens, 0..) |t, i| if (@as(i64, t) == s.think_end) {
        end = i;
        break;
    };
    return f(r.ptr, tokens[0..end], @min(s.forced_out, end));
}

/// A pipelined step's drawn token, read before it is fed while the runner watches; replaced when the stop says so.
pub fn pipelined(e: anytype, s: *Stream, feed: *be.Feed) !void {
    if (feed.* != .handle) return;
    const r = s.runner orelse return;
    const w = r.watching orelse return;
    if (r.stop == null or !s.think_open or !w(r.ptr)) return;
    const drawn = try e.readFeed(feed.*);
    const hit = r.stop.?(r.ptr, &.{drawn}, 0);
    feed.* = .{ .value = if (hit) |h| h.token else drawn };
}

test {
    _ = @import("runner_stop_test.zig");
}
