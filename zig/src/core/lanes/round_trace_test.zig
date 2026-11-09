//! The round trace beside streams that finish and are freed: it reads only live streams (a rank-0 fault seen in a smoke run).
const std = @import("std");
const Engine = @import("engine.zig").Engine;
const sm = @import("stream.zig");
const fake = @import("fake.zig");
const shared = @import("engine_test.zig");
const round_trace = @import("round_trace.zig");

const gpa = std.testing.allocator;

fn stepUntil(engine: *Engine, s: *const sm.Stream) !void {
    var n: u32 = 0;
    while (!s.finished and n < 200) : (n += 1) try engine.step();
    try std.testing.expect(s.finished);
}

test "round trace: a traced stream finishes and is freed, then a new stream takes an entry; no freed stream is read" {
    var fds: [2]c_int = undefined;
    try std.testing.expectEqual(@as(c_int, 0), std.c.pipe(&fds));
    defer _ = std.c.close(fds[0]);
    defer _ = std.c.close(fds[1]);
    var cfg = try shared.modelWith(true);
    defer cfg.deinit(gpa);
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer engine.deinit();
    round_trace.open(fds[1], 4, null);
    defer round_trace.close();
    // a stream that outlives the others, so the trace keeps stepping beside the freed one
    var long = try sm.Stream.init(gpa, .{ .id = "long", .prompt = &shared.p1, .max_new = 60 });
    defer long.deinit(gpa);
    try engine.addStream(&long);
    // the first stream in a mapping of its own, unmapped once the engine is done with it: any read of it faults
    const first = try std.heap.page_allocator.create(sm.Stream);
    first.* = try sm.Stream.init(gpa, .{ .id = "first", .prompt = &shared.p1, .max_new = 6 });
    try engine.addStream(first);
    try engine.step();
    try std.testing.expectEqual(@as(usize, 2), round_trace.held());
    try stepUntil(&engine, first);
    // the step that finished it swept it from live; its entry went in that step, while it was still live
    try std.testing.expectEqual(@as(usize, 1), engine.live.items.len);
    try std.testing.expectEqual(@as(usize, 1), round_trace.held());
    first.deinit(gpa);
    std.heap.page_allocator.destroy(first);
    // a new stream looks for a free entry beside the long one: the first's memory is gone
    var second = try sm.Stream.init(gpa, .{ .id = "second", .prompt = &shared.p1, .max_new = 6 });
    defer second.deinit(gpa);
    try engine.addStream(&second);
    try engine.step();
    try std.testing.expectEqual(@as(usize, 2), round_trace.held());
    try stepUntil(&engine, &second);
    try stepUntil(&engine, &long);
    try std.testing.expectEqual(@as(usize, 0), engine.live.items.len);
    try std.testing.expectEqual(@as(usize, 0), round_trace.held());
}

test "round trace: a stream released without finishing loses its entry at the next step (pointers compared only)" {
    var fds: [2]c_int = undefined;
    try std.testing.expectEqual(@as(c_int, 0), std.c.pipe(&fds));
    defer _ = std.c.close(fds[0]);
    defer _ = std.c.close(fds[1]);
    var cfg = try shared.modelWith(true);
    defer cfg.deinit(gpa);
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer engine.deinit();
    round_trace.open(fds[1], 4, null);
    defer round_trace.close();
    var long = try sm.Stream.init(gpa, .{ .id = "long", .prompt = &shared.p1, .max_new = 30 });
    defer long.deinit(gpa);
    try engine.addStream(&long);
    const gone = try std.heap.page_allocator.create(sm.Stream);
    gone.* = try sm.Stream.init(gpa, .{ .id = "gone", .prompt = &shared.p1, .max_new = 30 });
    try engine.addStream(gone);
    try engine.step();
    try std.testing.expectEqual(@as(usize, 2), round_trace.held());
    // taken out of live by its owner (a cancel's path) and freed while still unfinished
    for (engine.live.items, 0..) |s, i| if (s == gone) {
        _ = engine.live.orderedRemove(i);
        break;
    };
    engine.release(gone);
    gone.deinit(gpa);
    std.heap.page_allocator.destroy(gone);
    try engine.step();
    try std.testing.expectEqual(@as(usize, 1), round_trace.held());
    try stepUntil(&engine, &long);
    try std.testing.expectEqual(@as(usize, 0), round_trace.held());
}
