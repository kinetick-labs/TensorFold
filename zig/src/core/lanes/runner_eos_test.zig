//! Terminal runner holds, bounded resumes and atomic failure before any collector hand-off.
const std = @import("std");
const Stream = @import("stream.zig").Stream;
const hold = @import("hold.zig");
const Row = @import("logprob.zig").Row;
const t = std.testing;
const a = t.allocator;

const Hook = struct {
    calls: usize = 0,
    accept: bool = true,
    fn token(_: *anyopaque, _: u32, _: bool, _: bool, _: bool) bool {
        return false;
    }
    fn eos(ptr: *anyopaque, _: bool, can_hold: bool) bool {
        const h: *Hook = @ptrCast(@alignCast(ptr));
        h.calls += 1;
        const accepted = h.accept and can_hold;
        h.accept = false;
        return accepted;
    }
    fn spec(h: *Hook, max_new: u32) @import("stream.zig").Spec {
        return .{
            .id = "terminal",
            .prompt = &.{1},
            .max_new = max_new,
            .eos = &.{9},
            .runner = .{ .ptr = h, .token = token, .eos = eos },
        };
    }
};

fn drain(s: *Stream) !void {
    while (!s.finished) {
        const token = s.popForce() orelse return error.ModelDrawRequired;
        const row = [_]Row{.{ .token = token, .logprob = -1 }};
        _ = try s.commit(a, &.{token}, if (s.logprobs != null) &row else &.{});
    }
}

test "terminal EOS stays internal while held then a forced result ends with one replacement EOS" {
    var hook: Hook = .{};
    var spec = hook.spec(12);
    spec.logprobs = 0;
    var s = try Stream.init(a, spec);
    defer s.deinit(a);
    const rows = [_]Row{ .{ .token = 5, .logprob = -1 }, .{ .token = 9, .logprob = -2 } };
    try t.expectEqual(@as(usize, 2), try s.commit(a, &.{ 5, 9 }, &rows));
    try t.expect(s.held and s.awaiting and s.paused and s.runner_eos_pending and !s.finished);
    try t.expectEqualSlices(u32, &.{ 5, 9 }, s.emitted());
    try t.expectEqual(@as(usize, 2), s.rows.items.len);
    try t.expect(try hold.resumeAwaiting(&s, a, &.{ 6, 7 }));
    try t.expect(!s.held and !s.awaiting and !s.paused and !s.runner_eos_pending);
    try t.expectEqualSlices(u32, &.{ 6, 7, 9 }, s.force.items);
    try drain(&s);
    try t.expectEqualSlices(u32, &.{ 5, 9, 6, 7, 9 }, s.emitted());
    try t.expectEqual(s.emitted().len, s.rows.items.len);
    try t.expectEqual(@as(u32, 9), s.rows.items[s.rows.items.len - 1].token);
    try t.expectEqual(@import("stream.zig").Reason.stop, s.reason);
    try t.expect(!try hold.resumeAwaiting(&s, a, &.{8}));
}

test "terminal abstention and no insertion room end at the original EOS" {
    for ([_]bool{ false, true }) |no_room| {
        var hook: Hook = .{};
        var s = try Stream.init(a, hook.spec(10));
        defer s.deinit(a);
        _ = try s.commit(a, &.{ 5, 9 }, &.{});
        if (no_room) s.max_new = 3;
        try t.expect(try hold.resumeAwaiting(&s, a, if (no_room) &.{6} else &.{}));
        try t.expect(s.finished and s.reason == .stop and !s.runner_eos_pending);
        try t.expect(!s.held and !s.awaiting and !s.paused);
        try t.expectEqual(@as(usize, 0), s.force.items.len);
        try t.expectEqualSlices(u32, &.{ 5, 9 }, s.emitted());
    }
}

test "terminal result clipping reports length under max_new and answer_max" {
    for ([_]bool{ false, true }) |answer| {
        var hook: Hook = .{};
        var spec = hook.spec(if (answer) 20 else 5);
        if (answer) {
            spec.think_end = 5;
            spec.answer_max = 4;
        }
        var s = try Stream.init(a, spec);
        defer s.deinit(a);
        _ = try s.commit(a, &.{ 5, 9 }, &.{});
        try t.expect(s.runner_eos_pending);
        try t.expect(try hold.resumeAwaiting(&s, a, &.{ 6, 7, 8, 10 }));
        try t.expectEqualSlices(u32, &.{ 6, 7, 8, 9 }, s.force.items);
        try drain(&s);
        try t.expectEqualSlices(u32, &.{ 5, 9, 6, 7, 8 }, s.emitted());
        try t.expect(s.reason == .length);
    }
}

test "terminal hook is optional and external EOS holds take precedence" {
    for ([_]bool{ false, true }) |external| {
        var hook: Hook = .{};
        var spec = hook.spec(10);
        if (external) spec.hold_eos = true else spec.runner.?.eos = null;
        var s = try Stream.init(a, spec);
        defer s.deinit(a);
        _ = try s.commit(a, &.{9}, &.{});
        try t.expectEqual(@as(usize, 0), hook.calls);
        try t.expect(!s.runner_eos_pending and !s.awaiting);
        try t.expect(if (external) s.held and !s.finished else s.finished and s.reason == .stop);
    }
}

test "terminal hook does not hold past stop strings, forced tails or the length boundary" {
    const Stop = struct {
        fn check(_: *anyopaque, _: []const u32) bool {
            return true;
        }
    };
    for (0..3) |which| {
        var hook: Hook = .{};
        var spec = hook.spec(if (which == 2) 2 else 10);
        if (which == 0) spec.stop_check = .{ .ptr = &hook, .check = Stop.check };
        var s = try Stream.init(a, spec);
        defer s.deinit(a);
        if (which == 1) try s.force.append(a, 8);
        _ = try s.commit(a, &.{9}, &.{});
        try t.expect(s.finished and !s.runner_eos_pending);
        try t.expectEqual(@as(usize, 0), hook.calls);
    }
}

test "terminal resume allocation failure preserves the held EOS and can retry" {
    var hook: Hook = .{};
    var s = try Stream.init(a, hook.spec(10));
    defer s.deinit(a);
    _ = try s.commit(a, &.{9}, &.{});
    var fail = t.FailingAllocator.init(a, .{ .fail_index = 0, .resize_fail_index = 0 });
    try t.expectError(error.OutOfMemory, hold.resumeAwaiting(&s, fail.allocator(), &.{ 6, 7 }));
    try t.expect(s.held and s.awaiting and s.paused and s.runner_eos_pending and !s.finished);
    try t.expectEqual(@as(usize, 0), s.force.items.len);
    try t.expectEqualSlices(u32, &.{9}, s.emitted());
    try t.expect(try hold.resumeAwaiting(&s, a, &.{ 6, 7 }));
    try drain(&s);
    try t.expect(s.finished and s.reason == .stop);
}

test "cancelled terminal holds reject a late result" {
    var hook: Hook = .{};
    var s = try Stream.init(a, hook.spec(10));
    defer s.deinit(a);
    _ = try s.commit(a, &.{9}, &.{});
    s.finished = true;
    s.reason = .cancelled;
    try t.expect(!try hold.resumeAwaiting(&s, a, &.{6}));
    try t.expectEqualSlices(u32, &.{9}, s.emitted());
    try t.expectEqual(@as(usize, 0), s.force.items.len);
}

test "runner hook marks only an actual think transition, not a repeated token in content" {
    const Capture = struct {
        flags: [4]bool = undefined,
        n: usize = 0,
        fn token(ptr: *anyopaque, _: u32, _: bool, _: bool, ended: bool) bool {
            const c: *@This() = @ptrCast(@alignCast(ptr));
            c.flags[c.n] = ended;
            c.n += 1;
            return false;
        }
    };
    var capture: Capture = .{};
    var s = try Stream.init(a, .{
        .id = "boundary",
        .prompt = &.{1},
        .max_new = 20,
        .think_end = 8,
        .think_open = true,
        .runner = .{ .ptr = &capture, .token = Capture.token },
    });
    defer s.deinit(a);
    _ = try s.commit(a, &.{ 2, 8, 3, 8 }, &.{});
    try t.expectEqualSlices(bool, &.{ false, true, false, false }, &capture.flags);
}
