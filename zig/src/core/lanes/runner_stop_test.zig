//! The runner's digit stop (runner_stop.zig) on token text and the fake target's rounds, beside negative controls.
const std = @import("std");
const stop = @import("runner_stop.zig");
const Engine = @import("engine.zig").Engine;
const sm = @import("stream.zig");
const fake = @import("fake.zig");
const hold = @import("hold.zig");
const shared = @import("engine_test.zig");

const gpa = std.testing.allocator;
const nl: u32 = 96;

/// Ids 0-9 are their digit, 10 " x", 11 "12" (a two-digit token), 12 "ab", 13 "7)"; `nl` is the newline.
fn table(_: ?*anyopaque, t: u32) []const u8 {
    const digits = "0123456789";
    if (t < 10) return digits[t .. t + 1];
    return switch (t) {
        10 => " x",
        11 => "12",
        12 => "ab",
        13 => "7)",
        nl => "\n",
        else => "?",
    };
}
const text: stop.Bytes = .{ .of = table };

fn state(limits: stop.Limits) stop.State {
    return .{ .limits = limits, .newline = nl };
}

/// The runner's Text as `look` asks it: "\n" is `nl` (else one token per byte); `two` encodes "\n" as two tokens.
fn encNl(_: ?*anyopaque, a: std.mem.Allocator, t: []const u8) std.mem.Allocator.Error![]u32 {
    if (std.mem.eql(u8, t, "\n")) return try a.dupe(u32, &.{nl});
    const out = try a.alloc(u32, t.len);
    for (t, out) |c, *o| o.* = c;
    return out;
}
fn encTwo(_: ?*anyopaque, a: std.mem.Allocator, _: []const u8) std.mem.Allocator.Error![]u32 {
    return try a.dupe(u32, &.{ 1, 2 });
}
const Enc = struct { ctx: ?*anyopaque = null, encode: *const fn (?*anyopaque, std.mem.Allocator, []const u8) std.mem.Allocator.Error![]u32 };
const enc: Enc = .{ .encode = encNl };
const two: Enc = .{ .encode = encTwo };

fn landAll(st: *stop.State, tokens: []const u32) void {
    for (tokens) |t| st.landed(table(null, t), true);
}

test "runner digit stop: the model's digit right after an owned value is replaced, and not with the stop off" {
    var st = state(.{});
    try st.arm(gpa, enc, 3, "123");
    const hit = st.stop(text, &.{ 1, 2, 3, 4, 5 }, 3).?;
    try std.testing.expectEqual(@as(usize, 3), hit.at);
    try std.testing.expectEqual(nl, hit.token);
    try std.testing.expectEqual(stop.Why.after_value, hit.why);
    // a multi-digit token after the value is a digit too
    try std.testing.expectEqual(@as(usize, 3), st.stop(text, &.{ 1, 2, 3, 11 }, 3).?.at);
    // negative control: the same rows with the stop off land as drawn
    var off = state(.{ .after_value = false, .run_max = 0 });
    try off.arm(gpa, enc, 3, "123");
    try std.testing.expect(off.stop(text, &.{ 1, 2, 3, 4, 5 }, 3) == null);
    try std.testing.expect(!off.watching());
}

test "runner digit stop: nothing is replaced without an owned value, after a non-digit, or in the value's own rows" {
    var idle = state(.{});
    try std.testing.expect(idle.stop(text, &.{ 1, 2, 3, 4 }, 0) == null);
    try std.testing.expect(!idle.watching());
    // the model's first token after the value is not a digit: disarmed, a later digit lands
    var st = state(.{});
    try st.arm(gpa, enc, 2, "45");
    try std.testing.expect(st.watching());
    try std.testing.expect(st.stop(text, &.{ 4, 5, 10, 7, 8 }, 2) == null);
    landAll(&st, &.{ 4, 5, 10 });
    try std.testing.expect(!st.watching());
    try std.testing.expect(st.stop(text, &.{ 7, 8 }, 0) == null);
    // the value's own (forced) rows are digits and never replaced, even past the forced count the stream gives
    var own = state(.{});
    try own.arm(gpa, enc, 3, "999");
    try std.testing.expect(own.stop(text, &.{ 9, 9, 9 }, 0) == null);
    // no newline token: the stop is off
    var none: stop.State = .{};
    try none.arm(gpa, two, 1, "1");
    try std.testing.expect(none.looked and none.newline == null);
    try std.testing.expect(none.stop(text, &.{ 1, 2 }, 1) == null);
    try std.testing.expect(!none.watching());
}

test "runner digit stop: a value over several rounds is armed until its last row lands" {
    var st = state(.{});
    try st.arm(gpa, enc, 4, "1234");
    try std.testing.expect(st.stop(text, &.{ 1, 2 }, 2) == null);
    landAll(&st, &.{ 1, 2 });
    const hit = st.stop(text, &.{ 3, 4, 5 }, 2).?;
    try std.testing.expectEqual(@as(usize, 2), hit.at);
    landAll(&st, &.{ 3, 4, nl });
    try std.testing.expect(st.lead == null and st.run == 0);
}

test "runner digit stop: a value ending in a non-digit, or carrying the inline mark, does not arm" {
    var st = state(.{});
    try st.arm(gpa, enc, 2, "7)");
    try std.testing.expect(st.lead == null);
    try std.testing.expect(st.stop(text, &.{ 13, 4 }, 1) == null);
    try st.arm(gpa, enc, 3, " @engine");
    try std.testing.expect(st.lead == null);
    // outside reasoning nothing stays armed
    var closed = state(.{});
    try closed.arm(gpa, enc, 1, "5");
    closed.landed("5", false);
    try std.testing.expect(closed.lead == null and closed.run == 0);
}

test "runner digit stop: a reasoning digit run stops at the cap, and runs on with no cap" {
    var st = state(.{ .run_max = 8 });
    landAll(&st, &.{ 1, 2, 3, 4, 5, 6 });
    try std.testing.expectEqual(@as(u32, 6), st.run);
    const hit = st.stop(text, &.{ 7, 8, 9 }, 0).?;
    try std.testing.expectEqual(@as(usize, 2), hit.at);
    try std.testing.expectEqual(stop.Why.run_cap, hit.why);
    // a two-digit token past the cap
    try std.testing.expectEqual(@as(usize, 1), st.stop(text, &.{ 7, 11 }, 0).?.at);
    // forced rows are never cut, even over the cap
    try std.testing.expect(st.stop(text, &.{ 7, 8, 9 }, 3) == null);
    // a non-digit ends the run
    landAll(&st, &.{12});
    try std.testing.expectEqual(@as(u32, 0), st.run);
    landAll(&st, &.{ 13, 1 });
    try std.testing.expectEqual(@as(u32, 1), st.run);
    // negative control: no cap, the same digits land
    var off = state(.{ .run_max = 0 });
    landAll(&off, &.{ 1, 2, 3, 4, 5, 6 });
    try std.testing.expect(off.stop(text, &.{ 7, 8, 9 }, 0) == null);
}

test "runner digit stop: once a stop fired, later runs hold at hold_max; before it, and with no hold, they run on" {
    var st = state(.{ .run_max = 100, .hold_max = 5 });
    landAll(&st, &.{ 1, 2, 3, 4, 5 });
    // negative control: before any stop a sixth digit lands (the cap is still run_max)
    try std.testing.expectEqual(@as(u32, 100), st.cap());
    try std.testing.expect(st.stop(text, &.{6}, 0) == null);
    st.fired();
    try std.testing.expectEqual(@as(u32, 5), st.cap());
    const hit = st.stop(text, &.{ 6, 7 }, 0).?;
    try std.testing.expectEqual(@as(usize, 0), hit.at);
    try std.testing.expectEqual(stop.Why.run_cap, hit.why);
    try std.testing.expect(st.watching());
    // a new run after the newline holds at the short cap too, and forced rows still pass
    landAll(&st, &.{ nl, 1, 2, 3 });
    try std.testing.expectEqual(@as(usize, 2), st.stop(text, &.{ 4, 5, 6 }, 0).?.at);
    try std.testing.expect(st.stop(text, &.{ 4, 5, 6 }, 3) == null);
    // with no run cap the hold still applies once fired
    var uncapped = state(.{ .run_max = 0, .hold_max = 3 });
    uncapped.fired();
    landAll(&uncapped, &.{ 1, 2, 3 });
    try std.testing.expectEqual(stop.Why.run_cap, uncapped.stop(text, &.{4}, 0).?.why);
    // negative control: hold_max 0 keeps run_max after a stop
    var nohold = state(.{ .run_max = 100, .hold_max = 0 });
    nohold.fired();
    landAll(&nohold, &.{ 1, 2, 3, 4, 5 });
    try std.testing.expectEqual(@as(u32, 100), nohold.cap());
    try std.testing.expect(nohold.stop(text, &.{ 6, 7 }, 0) == null);
}

test "runner digit stop: the newline is asked for only near the cap or at an arm, once" {
    var st: stop.State = .{ .limits = .{ .run_max = 1024 } };
    landAll(&st, &.{ 1, 2, 3 });
    try std.testing.expect(!st.wants());
    try std.testing.expect(st.stop(text, &.{4}, 0) == null); // no newline yet: the stop is off
    var near: stop.State = .{ .limits = .{ .run_max = 100 } };
    for (0..40) |_| near.landed("7", true);
    try std.testing.expect(near.wants());
    try near.look(gpa, enc);
    try std.testing.expect(!near.wants() and near.newline.? == nl);
    for (0..60) |_| near.landed("7", true);
    try std.testing.expectEqual(stop.Why.run_cap, near.stop(text, &.{7}, 0).?.why);
}

// --- the fake target's rounds: every draw reads as the digit 7, so a fired value is followed by a runaway

fn sevens(_: ?*anyopaque, t: u32) []const u8 {
    return if (t == nl) "\n" else "7";
}

const fire_at: usize = 20;
const value = [_]u32{ 7, 8, 9 }; // the owned value's rows ("777" here)

/// A runner that fires once at `fire_at` (held, then resumed with `value`) and keeps the digit stop's state.
const Hook = struct {
    st: stop.State,
    seen: usize = 0,
    fired: bool = false,
    fires: bool = true,

    fn token(ptr: *anyopaque, t: u32, think_open: bool, can_hold: bool, _: bool) bool {
        const h: *Hook = @ptrCast(@alignCast(ptr));
        h.st.landed(sevens(null, t), think_open);
        defer h.seen += 1;
        if (!h.fires or h.fired or h.seen != fire_at or !can_hold) return false;
        h.fired = true;
        return true;
    }

    fn onStop(ptr: *anyopaque, tokens: []const u32, forced: usize) ?stop.Stop {
        const h: *Hook = @ptrCast(@alignCast(ptr));
        const hit = h.st.stop(.{ .of = sevens }, tokens, forced) orelse return null;
        h.st.fired(); // as the runner's onStop
        return hit;
    }

    fn onWatching(ptr: *anyopaque) bool {
        const h: *Hook = @ptrCast(@alignCast(ptr));
        return h.st.watching();
    }
};

const Run = struct { mtp: bool = true, drafts: bool = true, limits: stop.Limits = .{}, think_open: bool = true, stop_fn: bool = true, fires: bool = true };

/// The emitted tokens and the index of the value's first row (null: it never fired).
fn run(spec: Run) !struct { tokens: []u32, value_at: ?usize } {
    var cfg = try shared.modelWith(spec.mtp);
    defer cfg.deinit(gpa);
    cfg.forced_drafts = true;
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer engine.deinit();
    var hook: Hook = .{ .st = .{ .limits = spec.limits, .newline = nl }, .fires = spec.fires };
    const r: sm.RunnerHook = .{ .ptr = &hook, .token = Hook.token, .stop = if (spec.stop_fn) Hook.onStop else null, .watching = if (spec.stop_fn) Hook.onWatching else null };
    var s = try sm.Stream.init(gpa, .{ .id = "s", .prompt = &shared.p1, .max_new = 90, .drafts = spec.drafts, .think_end = 1000, .think_open = spec.think_open, .runner = r });
    defer s.deinit(gpa);
    try engine.addStream(&s);
    var value_at: ?usize = null;
    var rounds: usize = 0;
    while (!s.finished and rounds < 600) : (rounds += 1) {
        if (s.awaiting) {
            value_at = s.emitted().len;
            try hook.st.arm(gpa, enc, value.len, "777");
            try std.testing.expect(try hold.resumeAwaiting(&s, gpa, &value));
        }
        try engine.step();
    }
    try std.testing.expect(s.finished);
    return .{ .tokens = try gpa.dupe(u32, s.emitted()), .value_at = value_at };
}

fn longestRun(tokens: []const u32) usize {
    var best: usize = 0;
    var cur: usize = 0;
    for (tokens) |t| {
        cur = if (t == nl) 0 else cur + 1;
        best = @max(best, cur);
    }
    return best;
}

test "runner digit stop on the fake target: the draw after an owned value is a newline, MTP and pipelined rounds" {
    for ([_]bool{ true, false }) |mtp| for ([_]bool{ true, false }) |drafts| {
        const on = try run(.{ .mtp = mtp, .drafts = drafts, .limits = .{ .run_max = 0, .hold_max = 0 } });
        defer gpa.free(on.tokens);
        const off = try run(.{ .mtp = mtp, .drafts = drafts, .limits = .{ .after_value = false, .run_max = 0, .hold_max = 0 } });
        defer gpa.free(off.tokens);
        const at = on.value_at.?;
        try std.testing.expectEqual(at, off.value_at.?);
        try std.testing.expectEqualSlices(u32, &value, on.tokens[at .. at + value.len]);
        try std.testing.expectEqualSlices(u32, &value, off.tokens[at .. at + value.len]);
        // negative control: with the stop off the model's own (digit) draw follows the value
        try std.testing.expect(off.tokens[at + value.len] != nl);
        try std.testing.expectEqual(nl, on.tokens[at + value.len]);
        // everything before the value is untouched
        try std.testing.expectEqualSlices(u32, off.tokens[0 .. at + value.len], on.tokens[0 .. at + value.len]);
        // and only that one draw is replaced: no other newline the stop put in
        try std.testing.expectEqual(@as(usize, 1), std.mem.count(u32, on.tokens, &.{nl}) - std.mem.count(u32, off.tokens, &.{nl}));
    };
}

test "runner digit stop on the fake target: the run cap cuts every long reasoning digit run, none with no cap" {
    for ([_]bool{ true, false }) |mtp| {
        const on = try run(.{ .mtp = mtp, .fires = false, .limits = .{ .after_value = false, .run_max = 12 } });
        defer gpa.free(on.tokens);
        const off = try run(.{ .mtp = mtp, .fires = false, .limits = .{ .after_value = false, .run_max = 0 } });
        defer gpa.free(off.tokens);
        try std.testing.expect(longestRun(on.tokens) <= 12);
        try std.testing.expect(longestRun(off.tokens) > 12); // negative control
    }
}

test "runner digit stop on the fake target: outside reasoning the reply is the same tokens as with no stop at all" {
    for ([_]bool{ true, false }) |mtp| {
        const closed = try run(.{ .mtp = mtp, .think_open = false, .limits = .{ .run_max = 4 } });
        defer gpa.free(closed.tokens);
        const bare = try run(.{ .mtp = mtp, .think_open = false, .stop_fn = false });
        defer gpa.free(bare.tokens);
        try std.testing.expectEqualSlices(u32, bare.tokens, closed.tokens);
        // negative control: the same limits in reasoning do cut
        const open = try run(.{ .mtp = mtp, .think_open = true, .limits = .{ .run_max = 4 } });
        defer gpa.free(open.tokens);
        try std.testing.expect(!std.mem.eql(u32, bare.tokens, open.tokens));
    }
}

fn longestFrom(tokens: []const u32, from: usize) usize {
    return longestRun(tokens[@min(from, tokens.len)..]);
}

test "runner digit stop on the fake target: after the stop past a value the runaway holds at hold_max, not with no hold" {
    for ([_]bool{ true, false }) |mtp| for ([_]bool{ true, false }) |drafts| {
        const on = try run(.{ .mtp = mtp, .drafts = drafts, .limits = .{ .run_max = 0, .hold_max = 6 } });
        defer gpa.free(on.tokens);
        const off = try run(.{ .mtp = mtp, .drafts = drafts, .limits = .{ .run_max = 0, .hold_max = 0 } });
        defer gpa.free(off.tokens);
        const after = on.value_at.? + value.len;
        try std.testing.expectEqual(nl, on.tokens[after]);
        try std.testing.expectEqual(nl, off.tokens[after]); // both stop the first draw after the value
        try std.testing.expect(longestFrom(on.tokens, after) <= 6);
        try std.testing.expect(std.mem.count(u32, on.tokens[after..], &.{nl}) >= 2); // the hold cut again
        // negative control: with no hold the model's restarted run goes on past 6
        try std.testing.expect(longestFrom(off.tokens, after) > 6);
        // everything up to the value and its stop is untouched by the hold
        try std.testing.expectEqualSlices(u32, off.tokens[0 .. after + 1], on.tokens[0 .. after + 1]);
    };
}
