//! Runner fires replayed on the fake target: drafts past each forced result change the rounds, never the tokens.
const std = @import("std");
const Engine = @import("engine.zig").Engine;
const sm = @import("stream.zig");
const fake = @import("fake.zig");
const hold = @import("hold.zig");
const shared = @import("engine_test.zig");
const round_trace = @import("round_trace.zig");

const gpa = std.testing.allocator;
const fires = [_]usize{ 20, 47, 75 };
const foreign = [_]u32{ 7, 8, 9 };

/// A runner hook holding the stream at the given emitted indices.
const Hook = struct {
    seen: usize = 0,
    fn token(ptr: *anyopaque, _: u32, _: bool, can_hold: bool, _: bool) bool {
        const h: *Hook = @ptrCast(@alignCast(ptr));
        defer h.seen += 1;
        return can_hold and std.mem.indexOfScalar(usize, &fires, h.seen) != null;
    }
};

/// What a fire's result is: the target's own next `echo` draws (a result the drafts guess) or `foreign`'s tokens.
const Owned = union(enum) { echo: usize, foreign };

/// Counts over one reply: its rounds, the rounds that fed a result, and the drafts those carried and kept.
const Replay = struct {
    tokens: []u32,
    rounds: u64 = 0,
    feeds: u64 = 0,
    fed_drafts: u64 = 0,
    fed_kept: u64 = 0,
    kept_before: bool = true, // at each hold every token before the trigger is in the target's cache
    trigger_drafts: u64 = 0, // drafts the trigger rounds kept before their trigger
};

fn replay(on: bool, width: ?u32, owned: Owned) !Replay {
    return replayTraced(on, width, owned, null);
}

/// The fake target's verify count as a trace's backend counter.
fn verifies(ctx: *anyopaque, out: []u64) void {
    out[0] = @as(*fake.Fake, @ptrCast(@alignCast(ctx))).rounds;
}

fn replayTraced(on: bool, width: ?u32, owned: Owned, trace: ?c_int) !Replay {
    var cfg = try shared.modelWith(true);
    defer cfg.deinit(gpa);
    cfg.forced_drafts = on;
    if (width) |w| cfg.base_width = w;
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer engine.deinit();
    if (trace) |fd| round_trace.open(fd, 4, .{ .ctx = &target, .names = &.{"verifies"}, .read = verifies });
    defer if (trace != null) round_trace.close();
    var hook: Hook = .{};
    var s = try sm.Stream.init(gpa, .{ .id = "s", .prompt = &shared.p1, .max_new = 100, .drafts = true, .runner = .{ .ptr = &hook, .token = Hook.token } });
    defer s.deinit(gpa);
    try engine.addStream(&s);
    var r: Replay = .{ .tokens = &.{} };
    var history: std.ArrayList(u32) = .empty;
    defer history.deinit(gpa);
    while (!s.finished and r.rounds < 400) {
        if (s.awaiting) {
            var result: std.ArrayList(u32) = .empty;
            defer result.deinit(gpa);
            switch (owned) {
                .foreign => try result.appendSlice(gpa, &foreign),
                .echo => |n| {
                    history.clearRetainingCapacity();
                    try history.appendSlice(gpa, &shared.p1);
                    try history.appendSlice(gpa, s.emitted());
                    for (0..n) |_| try history.append(gpa, fake.next(history.items, null, history.items.len));
                    try result.appendSlice(gpa, history.items[history.items.len - n ..]);
                },
            }
            try std.testing.expect(try hold.resumeAwaiting(&s, gpa, result.items));
        }
        const feeding = s.force.items.len > 0;
        const drafted = s.drafted;
        const accepted = s.accepted;
        const before = s.emitted().len;
        try engine.step();
        r.rounds += 1;
        if (feeding) {
            r.feeds += 1;
            r.fed_drafts += s.drafted - drafted;
            r.fed_kept += s.accepted - accepted;
        }
        if (s.awaiting) {
            const cached = target.lanes.getPtr(&s).?.history.items.len;
            r.kept_before = r.kept_before and cached == shared.p1.len + s.emitted().len - 1;
            r.trigger_drafts += s.emitted().len - before - 1;
        }
    }
    try std.testing.expect(s.finished);
    r.tokens = try gpa.dupe(u32, s.emitted());
    return r;
}

/// Each token is the target's draw over everything before it, but a fire's result, which is `owned`'s.
fn expectReplayed(out: []const u32, owned: Owned) !void {
    var h: std.ArrayList(u32) = .empty;
    defer h.deinit(gpa);
    try h.appendSlice(gpa, &shared.p1);
    var forced_until: usize = 0;
    for (out, 0..) |t, i| {
        if (i < forced_until) {
            if (owned == .foreign) try std.testing.expectEqual(foreign[i - (forced_until - foreign.len)], t);
        } else try std.testing.expectEqual(fake.next(h.items, null, h.items.len), t);
        try h.append(gpa, t);
        if (std.mem.indexOfScalar(usize, &fires, i) != null) forced_until = i + 1 + switch (owned) {
            .echo => |n| n,
            .foreign => foreign.len,
        };
    }
}

test "fire replay: drafts past forced results keep every token; a guessed result saves rounds, off is the old count" {
    // old: each reply's rounds with the engine before drafts past forced results (a9301c957)
    const cases = [_]struct { width: ?u32, owned: Owned, old: u64 }{
        .{ .width = null, .owned = .{ .echo = 2 }, .old = 34 },
        .{ .width = null, .owned = .foreign, .old = 38 },
        .{ .width = 5, .owned = .{ .echo = 6 }, .old = 37 },
        .{ .width = 5, .owned = .foreign, .old = 38 },
    };
    for (cases) |c| {
        const off = try replay(false, c.width, c.owned);
        defer gpa.free(off.tokens);
        const on = try replay(true, c.width, c.owned);
        defer gpa.free(on.tokens);
        try std.testing.expectEqualSlices(u32, off.tokens, on.tokens);
        try expectReplayed(on.tokens, c.owned);
        // off (the negative control): a round feeding a result carries no drafts, as before
        try std.testing.expectEqual(@as(u64, 0), off.fed_drafts);
        try std.testing.expectEqual(c.old, off.rounds);
        try std.testing.expect(on.fed_drafts > 0);
        try std.testing.expect(on.rounds <= off.rounds);
        if (c.owned == .echo) {
            try std.testing.expect(on.fed_kept > 0);
            try std.testing.expect(on.rounds < off.rounds);
        }
        // the trigger round keeps the drafts it accepted before the trigger: only the trigger's row runs again
        try std.testing.expect(on.kept_before and off.kept_before);
        try std.testing.expect(on.trigger_drafts > 0);
    }
}

test "fire replay traced: each fire's rounds before, its trigger, hold, resume and rounds after; the same tokens" {
    var fds: [2]c_int = undefined;
    try std.testing.expectEqual(@as(c_int, 0), std.c.pipe(&fds));
    defer _ = std.c.close(fds[0]);
    const plain = try replay(true, null, .{ .echo = 2 });
    defer gpa.free(plain.tokens);
    const traced = try replayTraced(true, null, .{ .echo = 2 }, fds[1]);
    defer gpa.free(traced.tokens);
    _ = std.c.close(fds[1]);
    try std.testing.expectEqualSlices(u32, plain.tokens, traced.tokens);
    var text: std.ArrayList(u8) = .empty;
    defer text.deinit(gpa);
    var chunk: [4096]u8 = undefined;
    while (true) {
        const n = std.c.read(fds[0], &chunk, chunk.len);
        if (n <= 0) break;
        try text.appendSlice(gpa, chunk[0..@intCast(n)]);
    }
    const names = [_][]const u8{ "open", "hold", "resume", "trigger", "pre", "post" };
    var counts: [names.len]usize = @splat(0);
    var lines = std.mem.splitScalar(u8, text.items, '\n');
    var after_resume = false;
    while (lines.next()) |line| {
        if (line.len == 0) continue;
        const p = try std.json.parseFromSlice(struct { rt: []const u8, phase: []const u8 = "", since: i64 = 0, kind: []const u8 = "", verifies: u64 = 1 }, gpa, line, .{ .ignore_unknown_fields = true });
        defer p.deinit();
        const v = p.value;
        const key = if (std.mem.eql(u8, v.rt, "round")) v.phase else v.rt;
        for (names, &counts) |n, *c| c.* += @intFromBool(std.mem.eql(u8, n, key));
        try std.testing.expectEqual(@as(u64, 1), v.verifies); // a lone stream's step: one verify
        // the first round after a resume feeds the result: since 1, forced
        if (after_resume) try std.testing.expect(v.since == 1 and std.mem.eql(u8, v.kind, "forced"));
        after_resume = std.mem.eql(u8, v.rt, "resume");
    }
    try std.testing.expectEqualSlices(usize, &.{ 1, fires.len, fires.len, fires.len }, counts[0..4]);
    try std.testing.expect(counts[4] >= fires.len and counts[4] <= fires.len * 4); // before: up to 4 a fire
    try std.testing.expect(counts[5] > 0 and counts[5] <= fires.len * 4); // after: up to 4 a fire, while the reply lasts
}
