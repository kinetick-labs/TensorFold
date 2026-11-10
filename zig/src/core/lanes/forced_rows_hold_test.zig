//! A held fire's result lands whole: a trigger on one of its forced rows never holds while later rows wait to land
const std = @import("std");
const Engine = @import("engine.zig").Engine;
const sm = @import("stream.zig");
const fake = @import("fake.zig");
const hold = @import("hold.zig");
const shared = @import("engine_test.zig");

const gpa = std.testing.allocator;
const fire = 20; // the model's trigger token (emitted index)
const result = [_]u32{ 7, 8, 9, 10, 11, 12 }; // fewer rows than a window: one forced window takes the whole tail

/// Fires at `fire` and again at `again` (an emitted index), whenever the stream may hold there.
const Hook = struct {
    seen: usize = 0,
    again: usize,

    fn token(ptr: *anyopaque, _: u32, _: bool, can_hold: bool, _: bool) bool {
        const h: *Hook = @ptrCast(@alignCast(ptr));
        defer h.seen += 1;
        return can_hold and (h.seen == fire or h.seen == h.again);
    }
};

const Run = struct { tokens: []u32, holds: u32 };

fn run(again: usize) !Run {
    var cfg = try shared.modelWith(true);
    defer cfg.deinit(gpa);
    cfg.forced_drafts = true;
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer engine.deinit();
    var hook: Hook = .{ .again = again };
    var s = try sm.Stream.init(gpa, .{ .id = "s", .prompt = &shared.p1, .max_new = 60, .drafts = true, .runner = .{ .ptr = &hook, .token = Hook.token } });
    defer s.deinit(gpa);
    try engine.addStream(&s);
    var holds: u32 = 0;
    var rounds: u32 = 0;
    while (!s.finished and rounds < 400) : (rounds += 1) {
        if (s.awaiting) {
            holds += 1;
            // the first hold resumes with the result; a hold inside it resumes with nothing (the model draws on)
            try std.testing.expect(try hold.resumeAwaiting(&s, gpa, if (holds == 1) &result else &.{}));
        }
        try engine.step();
    }
    try std.testing.expect(s.finished);
    return .{ .tokens = try gpa.dupe(u32, s.emitted()), .holds = holds };
}

test "forced rows: a trigger on one before the result's last never holds, and every row lands in order" {
    for (1..result.len) |row| {
        const r = try run(fire + row);
        defer gpa.free(r.tokens);
        try std.testing.expectEqualSlices(u32, &result, r.tokens[fire + 1 ..][0..result.len]);
        try std.testing.expectEqual(@as(u32, 1), r.holds);
    }
}

test "forced rows: a trigger on the result's last still holds, as before (nothing waits behind it)" {
    const r = try run(fire + result.len);
    defer gpa.free(r.tokens);
    try std.testing.expectEqualSlices(u32, &result, r.tokens[fire + 1 ..][0..result.len]);
    try std.testing.expectEqual(@as(u32, 2), r.holds);
}
