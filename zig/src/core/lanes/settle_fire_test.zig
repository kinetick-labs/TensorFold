//! Runner fires settled at their trigger (RunnerHook.settled) on the fake target: a hold's tokens with no hold
const std = @import("std");
const Engine = @import("engine.zig").Engine;
const sm = @import("stream.zig");
const fake = @import("fake.zig");
const hold = @import("hold.zig");
const shared = @import("engine_test.zig");

const gpa = std.testing.allocator;
const fires = [_]usize{ 20, 47, 75 };
const foreign = [_]u32{ 7, 8, 9 };

/// A fire's result: the target's own next `echo` draws (one the drafts guess) or `foreign`'s tokens.
const Owned = union(enum) { echo: usize, foreign };

/// How a fire's result reaches the stream: settled at its trigger, or held and resumed after `ticks` host ticks.
const Way = union(enum) { settled, held: u32 };

/// Fires at the given emitted indices; `settled` answers at the trigger itself when the way is settled.
const Hook = struct {
    seen: usize = 0,
    stream: *sm.Stream = undefined,
    owned: Owned,
    result: std.ArrayList(u32) = .empty,
    at: ?usize = null, // a settled fire's emitted count at its trigger, until the round's end is seen

    fn token(ptr: *anyopaque, _: u32, _: bool, can_hold: bool, _: bool) bool {
        const h: *Hook = @ptrCast(@alignCast(ptr));
        defer h.seen += 1;
        return can_hold and std.mem.indexOfScalar(usize, &fires, h.seen) != null;
    }

    fn settled(ptr: *anyopaque) ?[]const u32 {
        const h: *Hook = @ptrCast(@alignCast(ptr));
        compute(h) catch return null;
        h.at = h.stream.emitted().len;
        return h.result.items;
    }

    /// The fire's result after everything the stream has (its trigger is in the context already).
    fn compute(h: *Hook) !void {
        h.result.clearRetainingCapacity();
        switch (h.owned) {
            .foreign => try h.result.appendSlice(gpa, &foreign),
            .echo => |n| {
                var history: std.ArrayList(u32) = .empty;
                defer history.deinit(gpa);
                try history.appendSlice(gpa, &shared.p1);
                try history.appendSlice(gpa, h.stream.emitted());
                for (0..n) |_| try history.append(gpa, fake.next(history.items, null, history.items.len));
                try h.result.appendSlice(gpa, history.items[history.items.len - n ..]);
            },
        }
    }
};

const Replay = struct { tokens: []u32, rounds: u64 = 0, holds: u64 = 0, paused: u64 = 0, cache_ok: bool = true, kept: u64 = 0 };

fn replay(way: Way, owned: Owned) !Replay {
    var cfg = try shared.modelWith(true);
    defer cfg.deinit(gpa);
    cfg.forced_drafts = true;
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer engine.deinit();
    var hook: Hook = .{ .owned = owned };
    defer hook.result.deinit(gpa);
    const settles = way == .settled;
    var s = try sm.Stream.init(gpa, .{ .id = "s", .prompt = &shared.p1, .max_new = 100, .drafts = true, .runner = .{ .ptr = &hook, .token = Hook.token, .settled = if (settles) Hook.settled else null } });
    defer s.deinit(gpa);
    hook.stream = &s;
    try engine.addStream(&s);
    var r: Replay = .{ .tokens = &.{} };
    var waited: u32 = 0;
    const ticks: u32 = switch (way) {
        .held => |n| n,
        .settled => 1,
    };
    while (!s.finished and r.rounds < 400) {
        if (s.awaiting) {
            // a host tick: the last one of `ticks` resumes it, before the next round is planned
            waited += 1;
            if (waited >= ticks) {
                try hook.compute();
                try std.testing.expect(try hold.resumeAwaiting(&s, gpa, hook.result.items));
                waited = 0;
            } else r.paused += 1;
        }
        try engine.step();
        r.rounds += 1;
        // the result tokens a settled fire's own round kept (accepted past the trigger and the result's own)
        if (hook.at) |at| r.kept += @min(s.emitted().len - at, hook.result.items.len);
        hook.at = null;
        if (s.awaiting) r.holds += 1;
        // a hold or a settled cut: every kept token is in the target's cache but the pending one, none past it
        if (s.awaiting or s.settle == .cut) {
            const cached = target.lanes.getPtr(&s).?.history.items.len;
            r.cache_ok = r.cache_ok and cached == shared.p1.len + s.emitted().len - 1;
        }
    }
    try std.testing.expect(s.finished);
    r.tokens = try gpa.dupe(u32, s.emitted());
    return r;
}

test "settled fires: the same tokens as a hold, no hold, and the trigger round keeps the drafted result" {
    for ([_]Owned{ .{ .echo = 2 }, .{ .echo = 4 }, .foreign }) |owned| {
        const two = try replay(.{ .held = 2 }, owned);
        defer gpa.free(two.tokens);
        const one = try replay(.{ .held = 1 }, owned);
        defer gpa.free(one.tokens);
        const now = try replay(.settled, owned);
        defer gpa.free(now.tokens);
        std.debug.print("settled fires ({any}): rounds held 2 ticks {d}, 1 tick {d}, settled {d} (kept {d})\n", .{ owned, two.rounds, one.rounds, now.rounds, now.kept });
        try std.testing.expectEqualSlices(u32, two.tokens, one.tokens);
        try std.testing.expectEqualSlices(u32, one.tokens, now.tokens);
        // a hold (the negative control) waits at every fire; a settled fire never holds
        try std.testing.expectEqual(@as(u64, fires.len), one.holds);
        try std.testing.expectEqual(@as(u64, 0), now.holds);
        // two ticks pause the stream one round a fire; one tick none
        try std.testing.expectEqual(@as(u64, fires.len), two.paused);
        try std.testing.expectEqual(@as(u64, 0), one.paused);
        try std.testing.expect(now.cache_ok and one.cache_ok and two.cache_ok);
        try std.testing.expect(one.rounds < two.rounds);
        // the fake's misses hash the draft position, so other rounds move with the cuts: only kept tokens are exact
        if (owned == .echo) try std.testing.expect(now.kept > 0) else try std.testing.expectEqual(@as(u64, 0), now.kept);
    }
}
