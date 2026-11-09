//! The terminal runner fence hides EOS tokens and rows until a result can be delivered before them.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const LaneHost = @import("lane_host.zig").LaneHost;
const runner = @import("lane_host_runner.zig");
const t = std.testing;
const a = t.allocator;

const Box = struct {
    tokens: std.ArrayList(u32) = .empty,
    rows: std.ArrayList(u32) = .empty,
    fn event(ptr: *anyopaque, _: api.Id, e: *const api.Event) void {
        const b: *Box = @ptrCast(@alignCast(ptr));
        switch (e.*) {
            .tokens => |tokens| b.tokens.appendSlice(a, tokens) catch unreachable,
            .logprobs => |rows| for (rows) |row| {
                b.rows.append(a, row.token) catch unreachable;
            },
            else => {},
        }
    }
};

const Hook = struct {
    armed: bool = true,
    fn token(_: *anyopaque, _: u32, _: bool, _: bool, _: bool) bool {
        return false;
    }
    fn eos(ptr: *anyopaque, _: bool, _: bool) bool {
        const h: *Hook = @ptrCast(@alignCast(ptr));
        const armed = h.armed;
        h.armed = false;
        return armed;
    }
};

test "host terminal resume flushes an unsent prefix then skips just the internal EOS and its row" {
    for ([_]bool{ false, true }) |abstain| {
        var h: LaneHost = .{ .gpa = a, .io = t.io, .core = undefined, .info_ = .{ .lanes = 1 } };
        defer h.decoded.deinit(a);
        var box: Box = .{};
        defer box.tokens.deinit(a);
        defer box.rows.deinit(a);
        var hook: Hook = .{};
        const request: api.Request = .{ .prompt = &.{1}, .max_tokens = 12 };
        var job: LaneHost.Job = .{
            .host = &h,
            .id = 1,
            .request = &request,
            .sink = .{ .ctx = &box, .event = Box.event },
            .stream = try lanes.Stream.init(a, .{
                .id = "terminal",
                .prompt = &.{1},
                .max_new = 12,
                .eos = &.{9},
                .logprobs = 0,
                .runner = .{ .ptr = &hook, .token = Hook.token, .eos = Hook.eos },
            }),
        };
        defer job.stream.deinit(a);
        const rows = [_]lanes.logprob.Row{ .{ .token = 5, .logprob = -1 }, .{ .token = 9, .logprob = -2 } };
        _ = try job.stream.commit(a, &.{ 5, 9 }, &rows);
        try t.expectEqual(@as(usize, 0), job.delivered);
        try t.expect(try runner.resumeResult(&h, &job, if (abstain) &.{} else &.{ 6, 7 }));
        try t.expectEqualSlices(u32, &.{5}, box.tokens.items);
        try t.expectEqualSlices(u32, &.{5}, box.rows.items);
        try t.expectEqual(@as(usize, if (abstain) 1 else 2), job.delivered);
        if (!abstain) while (!job.stream.finished) {
            const token = job.stream.popForce() orelse return error.ModelDrawRequired;
            const row = [_]lanes.logprob.Row{.{ .token = token, .logprob = -1 }};
            _ = try job.stream.commit(a, &.{token}, &row);
        };
        h.send(&job);
        h.send(&job);
        try t.expectEqualSlices(u32, if (abstain) &.{ 5, 9 } else &.{ 5, 6, 7, 9 }, box.tokens.items);
        try t.expectEqualSlices(u32, box.tokens.items, box.rows.items);
        try t.expect(job.stream.finished and job.stream.reason == .stop);
        try t.expect(!try runner.resumeResult(&h, &job, &.{8}));
        h.send(&job);
        try t.expectEqualSlices(u32, if (abstain) &.{ 5, 9 } else &.{ 5, 6, 7, 9 }, box.tokens.items);
    }
}

test "host send keeps a failed terminal EOS withheld while delivering its prefix rows" {
    var h: LaneHost = .{ .gpa = a, .io = t.io, .core = undefined, .info_ = .{ .lanes = 1 } };
    defer h.decoded.deinit(a);
    var box: Box = .{};
    defer box.tokens.deinit(a);
    defer box.rows.deinit(a);
    var hook: Hook = .{};
    const request: api.Request = .{ .prompt = &.{1}, .max_tokens = 12 };
    var job: LaneHost.Job = .{
        .host = &h,
        .id = 1,
        .request = &request,
        .sink = .{ .ctx = &box, .event = Box.event },
        .stream = try lanes.Stream.init(a, .{
            .id = "terminal",
            .prompt = &.{1},
            .max_new = 12,
            .eos = &.{9},
            .logprobs = 0,
            .runner = .{ .ptr = &hook, .token = Hook.token, .eos = Hook.eos },
        }),
    };
    defer job.stream.deinit(a);
    const rows = [_]lanes.logprob.Row{ .{ .token = 5, .logprob = -1 }, .{ .token = 9, .logprob = -2 } };
    _ = try job.stream.commit(a, &.{ 5, 9 }, &rows);
    var fail = t.FailingAllocator.init(a, .{ .fail_index = 0, .resize_fail_index = 0 });
    h.gpa = fail.allocator();
    try t.expectError(error.OutOfMemory, runner.resumeResult(&h, &job, &.{ 6, 7 }));
    h.gpa = a;
    job.stream.finished = true;
    job.stream.reason = .@"error";
    h.send(&job);
    try t.expectEqualSlices(u32, &.{5}, box.tokens.items);
    try t.expectEqualSlices(u32, &.{5}, box.rows.items);
    try t.expectEqual(@as(usize, 1), job.delivered);
    try t.expectEqual(@as(usize, 0), job.stream.force.items.len);
}
