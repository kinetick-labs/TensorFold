//! Replies held at their EOS through the lane host: the server sees the EOS, the engine idles, one resume continues.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const LaneHost = @import("lane_host.zig").LaneHost;
const Reason = api.Reason;
const Id = api.Id;
const Event = api.Event;
const Request = api.Request;

const gpa = std.testing.allocator;
const prompt = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3 };
const suffix = [_]u32{ 11, 22, 33, 44, 55, 66, 77, 88, 12, 23, 34 };

/// A request's events as the server's mailbox keeps them.
const Box = struct {
    mutex: std.Io.Mutex = .init,
    tokens: std.ArrayList(u32) = .empty,
    done: ?Reason = null,
    fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
        const b: *@This() = @ptrCast(@alignCast(ctx));
        b.mutex.lockUncancelable(std.testing.io);
        defer b.mutex.unlock(std.testing.io);
        switch (e.*) {
            .tokens => |t| b.tokens.appendSlice(gpa, t) catch {},
            .finished => |f| b.done = f.reason,
            else => {},
        }
    }
    fn count(b: *@This()) struct { n: usize, done: ?Reason } {
        b.mutex.lockUncancelable(std.testing.io);
        defer b.mutex.unlock(std.testing.io);
        return .{ .n = b.tokens.items.len, .done = b.done };
    }
    /// Waits until `n` tokens arrived (or the reply ended), up to 10 s.
    fn reach(b: *@This(), n: usize) !void {
        for (0..10_000) |_| {
            const c = b.count();
            if (c.n >= n or c.done != null) return;
            std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
        }
        return error.Timeout;
    }
    fn wait(b: *@This()) !Reason {
        for (0..10_000) |_| {
            if (b.count().done) |r| return r;
            std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
        }
        return error.Timeout;
    }
};

/// The fake target's first draw after the prompt (index 2 or later) that no earlier draw repeats: a reachable EOS.
fn eosAfter(p: []const u32) !struct { token: u32, at: usize } {
    var h: std.ArrayList(u32) = .empty;
    defer h.deinit(gpa);
    try h.appendSlice(gpa, p);
    var seen: [lanes.fake.vocab]bool = @splat(false);
    for (0..64) |i| {
        const t = lanes.fake.next(h.items, null, h.items.len);
        if (i >= 2 and !seen[t]) return .{ .token = t, .at = i };
        seen[t] = true;
        try h.append(gpa, t);
    }
    return error.NoEos;
}

/// A planner that takes each prompt as one piece and its final in the next round.
const Plan = struct {
    const Seq = struct { key: usize, n: u64, done: bool = false };
    seqs: std.ArrayList(Seq) = .empty,
    fn self(ctx: *anyopaque) *@This() {
        return @ptrCast(@alignCast(ctx));
    }
    fn beginAdmit(_: *anyopaque) void {}
    fn admit(ctx: *anyopaque, ask: api.Rounds.Ask, _: *[]const u8) api.Rounds.Verdict {
        self(ctx).seqs.append(gpa, .{ .key = ask.key, .n = ask.prompt.len }) catch return .refuse;
        return .admit;
    }
    fn admitted(_: *anyopaque) anyerror!void {}
    fn arm(_: *anyopaque, _: usize) void {}
    fn began(_: *anyopaque, _: usize, _: bool) void {}
    fn plan(ctx: *anyopaque, pieces: *std.ArrayList(api.Rounds.Piece), finals: *std.ArrayList(usize)) anyerror!void {
        pieces.clearRetainingCapacity();
        finals.clearRetainingCapacity();
        for (self(ctx).seqs.items) |*x| if (!x.done) {
            try pieces.append(gpa, .{ .key = x.key, .start = 0, .end = x.n - 1, .save = false });
            try finals.append(gpa, x.key);
            x.done = true;
        };
    }
    fn left(ctx: *anyopaque, key: usize) void {
        const p = self(ctx);
        for (p.seqs.items, 0..) |x, i| if (x.key == key) {
            _ = p.seqs.orderedRemove(i);
            return;
        };
    }
    fn after(_: *anyopaque, _: f64, _: f64) void {}
};

const End = enum { resume_twice, cancel };

/// One hold_eos request on a started host (default loop, or the round planner's): held, idle, then `end`.
fn heldReply(planner: bool, end: End) !void {
    const eos = try eosAfter(&prompt);
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, if (planner) target.pieceBackend() else target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    var pl: Plan = .{};
    defer pl.seqs.deinit(gpa);
    if (planner) host.rounds = .{ .ctx = &pl, .vtable = &.{ .begin_admit = Plan.beginAdmit, .admit = Plan.admit, .admitted = Plan.admitted, .arm = Plan.arm, .began = Plan.began, .plan = Plan.plan, .left = Plan.left, .after = Plan.after } };
    var box: Box = .{};
    defer box.tokens.deinit(gpa);
    const request: Request = .{ .prompt = &prompt, .max_tokens = 60, .eos = &.{eos.token}, .hold_eos = true };
    const e = host.engine();
    try host.start();
    defer host.stop();
    try e.submit(7, &request, .{ .ctx = &box, .event = Box.event });
    try box.reach(eos.at + 1);
    // held: the EOS reached the server, nothing finished, and the engine thread runs no rounds while it waits
    const steps = core.steps;
    std.Io.sleep(std.testing.io, .fromMilliseconds(60), .awake) catch {};
    const c = box.count();
    try std.testing.expectEqual(eos.at + 1, c.n);
    try std.testing.expect(c.done == null);
    try std.testing.expectEqual(steps, core.steps);
    switch (end) {
        .cancel => {
            e.cancel(7);
            try std.testing.expectEqual(Reason.cancelled, try box.wait());
            return;
        },
        .resume_twice => {
            try std.testing.expect(api.hold.resumeHeld(e, 7, .{ .tokens = &suffix }));
            try std.testing.expect(api.hold.resumeHeld(e, 7, .{ .tokens = &suffix })); // queued; the engine ignores it
        },
    }
    try std.testing.expectEqual(Reason.length, try box.wait());
    const out = box.tokens.items;
    try std.testing.expectEqual(@as(usize, 60), out.len);
    try std.testing.expectEqualSlices(u32, &suffix, out[eos.at + 1 .. eos.at + 1 + suffix.len]);
    var history: std.ArrayList(u32) = .empty;
    defer history.deinit(gpa);
    try history.appendSlice(gpa, &prompt);
    for (out, 0..) |t, i| {
        const forced = i > eos.at and i <= eos.at + suffix.len;
        if (!forced) try std.testing.expectEqual(lanes.fake.next(history.items, null, history.items.len), t);
        try history.append(gpa, t);
    }
}

test "compute hold: a held reply idles at its EOS and one resume continues it exactly, in the default loop and a planner's" {
    try heldReply(false, .resume_twice);
    try heldReply(true, .resume_twice);
}

test "compute hold: a held reply that is cancelled ends cancelled, in the default loop and a planner's" {
    try heldReply(false, .cancel);
    try heldReply(true, .cancel);
}
