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

/// Ids: 0-9 digits, 10 " x", 11 "12", 12 "ab", 13 "7)", 14 " @", 15 "engine", 16 "er", 17 "?", 18 "-mike", 19 "_x".
fn table(_: ?*anyopaque, t: u32) []const u8 {
    const digits = "0123456789";
    if (t < 10) return digits[t .. t + 1];
    return switch (t) {
        10 => " x",
        11 => "12",
        12 => "ab",
        13 => "7)",
        14 => " @",
        15 => "engine",
        16 => "er",
        17 => "?",
        18 => "-mike",
        19 => "_x",
        20 => "/r",
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

test "runner digit stop: a value ending in a non-word character does not arm, nor the inline mark with the mark stop off" {
    var st = state(.{});
    try st.arm(gpa, enc, 2, "7)");
    try std.testing.expect(st.lead == null);
    try std.testing.expect(st.stop(text, &.{ 13, 4 }, 1) == null);
    var nomark = state(.{ .after_mark = false });
    try nomark.arm(gpa, enc, 3, " @engine");
    try std.testing.expect(nomark.lead == null);
    try std.testing.expect(nomark.stop(text, &.{ 14, 15, 16 }, 2) == null);
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
    st.fired(.run_cap);
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
    uncapped.fired(.run_cap);
    landAll(&uncapped, &.{ 1, 2, 3 });
    try std.testing.expectEqual(stop.Why.run_cap, uncapped.stop(text, &.{4}, 0).?.why);
    // negative control: hold_max 0 keeps run_max after a stop
    var nohold = state(.{ .run_max = 100, .hold_max = 0 });
    nohold.fired(.run_cap);
    landAll(&nohold, &.{ 1, 2, 3, 4, 5 });
    try std.testing.expectEqual(@as(u32, 100), nohold.cap());
    try std.testing.expect(nohold.stop(text, &.{ 6, 7 }, 0) == null);
}

test "runner digit stop: a word character glued to the inline mark is replaced; a space, punctuation or newline lands" {
    var st = state(.{});
    try st.arm(gpa, enc, 2, " @engine");
    try std.testing.expect(st.lead != null and st.word and st.watching());
    // the mark's own (forced) rows pass; the model's "er" right after them is the stop (@engineer)
    const hit = st.stop(text, &.{ 14, 15, 16 }, 2).?;
    try std.testing.expectEqual(@as(usize, 2), hit.at);
    try std.testing.expectEqual(stop.Why.after_mark, hit.why);
    // a glued digit, "-", "_" or "/" extends it too
    for ([_]u32{ 7, 11, 18, 19, 20 }) |t| try std.testing.expectEqual(stop.Why.after_mark, st.stop(text, &.{ 14, 15, t }, 2).?.why);
    // negative controls: what ends the mark lands as drawn
    for ([_]u32{ 17, 10, nl }) |t| try std.testing.expect(st.stop(text, &.{ 14, 15, t }, 2) == null);
    // only the model's first draw after the span is guarded: once it landed, a later letter is the model's own text
    landAll(&st, &.{ 14, 15, 17 });
    try std.testing.expect(st.lead == null and !st.watching());
    try std.testing.expect(st.stop(text, &.{16}, 0) == null);
}

test "runner digit stop: a mark stop does not hold the reply's later runs; a digit stop does" {
    var st = state(.{ .run_max = 100, .hold_max = 5 });
    try st.arm(gpa, enc, 2, " @engine");
    st.fired(st.stop(text, &.{ 14, 15, 16 }, 2).?.why);
    try std.testing.expectEqual(@as(u32, 100), st.cap());
    landAll(&st, &.{ 14, 15, nl, 1, 2, 3, 4, 5, 6 });
    try std.testing.expect(st.stop(text, &.{ 7, 8 }, 0) == null); // the model's own number goes on past 5
    // negative control: a digit stop in its place holds at 5
    st.fired(.run_cap);
    try std.testing.expectEqual(@as(u32, 5), st.cap());
    try std.testing.expectEqual(stop.Why.run_cap, st.stop(text, &.{7}, 0).?.why);
}

// --- recorded TP1 replies (tp1-digitcap.py, 2026-11-13): the model's reasoning with the runner's owned spans marked

const Fixture = struct { about: []const u8, owned: []const [2]usize, text: []const u8 };

/// A reply as tokens: one per digit (as Qwen's), a letter run as one, any other byte as one; owned span ends cut too.
const Replay = struct {
    text: []const u8,
    starts: std.ArrayList(usize) = .empty, // each token's first byte; token i is text[starts[i]..starts[i + 1]]

    fn init(t: []const u8, owned: []const [2]usize) !Replay {
        var r: Replay = .{ .text = t };
        var i: usize = 0;
        while (i < t.len) {
            try r.starts.append(gpa, i);
            if (std.ascii.isAlphabetic(t[i])) {
                i += 1;
                while (i < t.len and std.ascii.isAlphabetic(t[i]) and !edge(owned, i)) i += 1;
            } else i += 1;
        }
        try r.starts.append(gpa, t.len);
        return r;
    }
    fn edge(owned: []const [2]usize, at: usize) bool {
        for (owned) |o| if (o[0] == at or o[1] == at) return true;
        return false;
    }
    fn deinit(r: *Replay) void {
        r.starts.deinit(gpa);
    }
    fn of(ctx: ?*anyopaque, t: u32) []const u8 {
        const r: *const Replay = @ptrCast(@alignCast(ctx.?));
        if (t == replay_nl) return "\n";
        return r.text[r.starts.items[t]..r.starts.items[t + 1]];
    }
    fn token(r: *const Replay, at: usize) usize {
        return std.mem.indexOfScalar(usize, r.starts.items, at).?;
    }
};
const replay_nl: u32 = 1 << 30;

fn encReplay(_: ?*anyopaque, a: std.mem.Allocator, t: []const u8) std.mem.Allocator.Error![]u32 {
    std.debug.assert(std.mem.eql(u8, t, "\n"));
    return try a.dupe(u32, &.{replay_nl});
}

/// The recorded reply's first stop and its offset: owned spans armed and forced as the runner does, the rest drawn.
fn replay(fixture: []const u8, limits: stop.Limits) !?struct { at: usize, why: stop.Why } {
    const parsed = try std.json.parseFromSlice(Fixture, gpa, fixture, .{});
    defer parsed.deinit();
    const f = parsed.value;
    var r = try Replay.init(f.text, f.owned);
    defer r.deinit();
    const bytes: stop.Bytes = .{ .ctx = &r, .of = Replay.of };
    var st: stop.State = .{ .limits = limits };
    const e: Enc = .{ .encode = encReplay };
    var i: usize = 0;
    const n = r.starts.items.len - 1;
    while (i < n) {
        const at = r.starts.items[i];
        var span: ?[2]usize = null;
        for (f.owned) |o| if (o[0] == at) {
            span = o;
        };
        if (span) |o| {
            const end = r.token(o[1]);
            var ids: [256]u32 = undefined;
            for (i..end, 0..) |t, k| ids[k] = @intCast(t);
            try st.arm(gpa, e, end - i, f.text[o[0]..o[1]]);
            try std.testing.expect(st.stop(bytes, ids[0 .. end - i], end - i) == null); // forced rows are never cut
            for (ids[0 .. end - i]) |t| st.landed(Replay.of(&r, t), true);
            i = end;
            continue;
        }
        if (st.wants()) try st.look(gpa, e);
        if (st.stop(bytes, &.{@intCast(i)}, 0)) |hit| return .{ .at = at, .why = hit.why };
        st.landed(Replay.of(&r, @intCast(i)), true);
        i += 1;
    }
    return null;
}

const s13 = @embedFile("testdata/runner-stop-13p150-s13.json"); // failed the gate: @engineer-mike/secret-repo! then 1,024 digits
const s12 = @embedFile("testdata/runner-stop-13p150-s12.json"); // passed: marks end in '"' or a newline, one copy unmarked

test "runner digit stop on a recorded TP1 reply: the glued @engineer is stopped where base let a 1,024-digit run start" {
    // negative control: base (no mark stop) lets the served reply land whole, its 1,024-digit run included
    try std.testing.expect(try replay(s13, .{ .after_mark = false }) == null);
    // the mark stop replaces the model's first draw after the first owned span: "er" of "@engineer", before any digit
    const on = (try replay(s13, .{})).?;
    try std.testing.expectEqual(stop.Why.after_mark, on.why);
    try std.testing.expectEqual(@as(usize, 229), on.at);
    const parsed = try std.json.parseFromSlice(Fixture, gpa, s13, .{});
    defer parsed.deinit();
    try std.testing.expectEqual(on.at, parsed.value.owned[0][1]);
    try std.testing.expect(std.mem.startsWith(u8, parsed.value.text[on.at..], "er-mike/secret-repo!"));
    try std.testing.expect(std.mem.indexOf(u8, parsed.value.text, "secret-repo!8437").? > on.at); // before the run
}

test "runner digit stop on a recorded TP1 reply: a reply that passed, with a model-written copy of the value, is untouched" {
    try std.testing.expect(try replay(s12, .{}) == null);
    try std.testing.expect(try replay(s12, .{ .after_mark = false }) == null);
}

test "runner digit stop: after a mark stop the model can still write a number of the value's digits in full" {
    const parsed = try std.json.parseFromSlice(Fixture, gpa, s13, .{});
    defer parsed.deinit();
    const o = parsed.value.owned[0];
    const v = parsed.value.text[o[0] .. o[1] - " @engine".len]; // 13^150, the 168 digits the runner owned
    try std.testing.expectEqual(@as(usize, 168), v.len);
    var st = state(.{});
    try st.arm(gpa, enc, 2, " @engine");
    st.fired(st.stop(text, &.{ 14, 15, 16 }, 2).?.why);
    landAll(&st, &.{ 14, 15, nl });
    for (0..168) |k| {
        const d: u32 = v[k] - '0';
        try std.testing.expect(st.stop(text, &.{d}, 0) == null);
        landAll(&st, &.{d});
    }
    try std.testing.expectEqual(@as(u32, 168), st.run);
    // negative control: after a digit stop the same number is cut at the hold (64)
    var held = state(.{});
    held.fired(.run_cap);
    const ones: [64]u32 = @splat(1);
    landAll(&held, &ones);
    try std.testing.expectEqual(stop.Why.run_cap, held.stop(text, &.{2}, 0).?.why);
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
        h.st.fired(hit.why); // as the runner's onStop
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
