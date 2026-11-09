//! TF_DSV41_RUNNER_SETTLE_US on reply sessions: the same owned tokens, in one tick a hold instead of one a stage.
const std = @import("std");
const rapi = @import("runner_api.zig");
const runner = @import("runner.zig");

const gpa = std.testing.allocator;

fn bytes(_: ?*anyopaque, token: u32) []const u8 {
    const all = comptime blk: {
        var t: [128]u8 = undefined;
        for (&t, 0..) |*c, i| c.* = i;
        break :blk t;
    };
    return if (token < 128) all[token..][0..1] else "";
}

fn encode(_: ?*anyopaque, a: std.mem.Allocator, text: []const u8) std.mem.Allocator.Error![]u32 {
    const out = try a.alloc(u32, text.len);
    for (text, out) |c, *t| t.* = c;
    return out;
}

/// Every cell's code is its graph program.
fn translate(_: ?*anyopaque, a: std.mem.Allocator, cell: rapi.Cell) std.mem.Allocator.Error!rapi.Translation {
    return .{ .program = .{ .kind = .graph, .bytes = try a.dupe(u8, std.mem.trim(u8, cell.code, " \n")) } };
}

/// Answers by program after `delay` polls, or once `gate` opens; a run here ends when its steps fit `quick`.
const Back = struct {
    delay: u32 = 0,
    gate: bool = true,
    jobs: std.ArrayList(struct { prog: []u8, polls: u32 }) = .empty,
    started: u32 = 0,
    here: u32 = 0,

    const answers = [_]struct { []const u8, []const u8, u64 }{
        .{ "47*83", "3901\n", 900 },
        .{ "2**100", "1267650600228229401496703205376\n", 30_000 },
        .{ "7*6", "42\n", 500 },
        .{ "3**200", "265613988875874769338781322035779626829233452653394495974574961739092490901302182994384699044001\n", 1 << 40 },
    };

    fn answer(prog: []const u8) rapi.Result {
        for (answers) |x| if (std.mem.eql(u8, x[0], prog)) return .{ .status = .ok, .out = x[1], .steps = x[2] };
        return .{ .status = .fault };
    }
    fn backend(b: *Back) rapi.Backend {
        return .{ .ctx = b, .vtable = &.{ .start = start, .poll = poll, .cancel = cancel, .release = release, .now = now } };
    }
    fn start(ctx: *anyopaque, _: u64, p: rapi.Program, _: u64) error{ OutOfMemory, Busy }!rapi.JobId {
        const b: *Back = @ptrCast(@alignCast(ctx));
        try b.jobs.append(gpa, .{ .prog = try gpa.dupe(u8, p.bytes), .polls = 0 });
        b.started += 1;
        return b.jobs.items.len - 1;
    }
    fn poll(ctx: *anyopaque, job: rapi.JobId) rapi.Poll {
        const b: *Back = @ptrCast(@alignCast(ctx));
        const j = &b.jobs.items[job];
        j.polls += 1;
        if (j.polls <= b.delay or !b.gate) return .running;
        return .{ .done = answer(j.prog) };
    }
    fn now(ctx: *anyopaque, p: rapi.Program, budget: u64, quick: u64) ?rapi.Result {
        const b: *Back = @ptrCast(@alignCast(ctx));
        const r = answer(p.bytes);
        if (r.steps > @min(budget, quick)) return null;
        b.here += 1;
        return r;
    }
    fn cancel(_: *anyopaque, _: rapi.JobId) void {}
    fn release(_: *anyopaque, _: u64) void {}
    fn deinit(b: *Back) void {
        for (b.jobs.items) |j| gpa.free(j.prog);
        b.jobs.deinit(gpa);
    }
};

const reply = "So 47*83 = 3901\nand 2**100 = 1267650600228229401496703205376\n>>> 7*6\n42\nLast 3**200 = 9\n";

/// The owned text of each hold, and the reply's ticks, starts and runs here.
const Run = struct { owned: []u8, ticks: [4]u32, started: u32, here: u32 };

/// Feeds `reply`; at a hold, waits for the cell's translation (the trigger round's rest), then ticks to its tokens.
fn run(settle_us: u32, back: *Back, open_gate_at: ?u32) !Run {
    var r: runner.Runner = .{ .gpa = gpa, .cfg = .{ .settle_us = settle_us }, .frontend = .{ .translate = translate }, .backend = back.backend(), .text = .{ .bytes = bytes, .encode = encode } };
    r.cfg.scan.inline_math = true;
    const s = (try r.session(.{})).?;
    defer s.destroy();
    const hook = s.hook();
    var owned: std.ArrayList(u8) = .empty;
    errdefer owned.deinit(gpa);
    var ticks: [4]u32 = @splat(0);
    var holds: usize = 0;
    for (reply) |ch| {
        if (!hook.token(hook.ptr, ch, false, true, false)) continue;
        const cell = s.cells.items[s.waiting.?.index];
        // settling translates an inline value at its close, on this thread
        if (settle_us > 0 and cell.form == .inline_expr) try std.testing.expect(cell.thread == null and cell.translated.load(.acquire));
        while (!cell.translated.load(.acquire)) std.Thread.yield() catch {};
        var n: u32 = 0;
        const got = while (true) {
            n += 1;
            if (open_gate_at) |at| if (n == at) {
                back.gate = true;
            };
            if (try s.tick()) |x| break x;
        };
        defer gpa.free(got);
        try std.testing.expectEqual(n, s.held_ticks);
        for (got) |t| try owned.appendSlice(gpa, bytes(null, t));
        try owned.append(gpa, '|');
        ticks[holds] = n;
        holds += 1;
    }
    try std.testing.expectEqual(@as(usize, 4), holds);
    return .{ .owned = try owned.toOwnedSlice(gpa), .ticks = ticks, .started = back.started, .here = back.here };
}

test "runner settle: the same owned tokens; off ticks a stage a round (the old counts), on one tick a hold" {
    const want = " 3901| 1267650600228229401496703205376|42\n|" ++ " " ++ Back.answers[3][1][0 .. Back.answers[3][1].len - 1] ++ "|";
    var b0: Back = .{};
    defer b0.deinit();
    const off = try run(0, &b0, null);
    defer gpa.free(off.owned);
    var b1: Back = .{};
    defer b1.deinit();
    const on = try run(3000, &b1, null);
    defer gpa.free(on.owned);
    try std.testing.expectEqualStrings(want, off.owned);
    try std.testing.expectEqualStrings(off.owned, on.owned);
    // off (the negative control): a tick to start each run, a tick to take it; every run is a job
    try std.testing.expectEqual([4]u32{ 2, 2, 2, 2 }, off.ticks);
    try std.testing.expectEqual(@as(u32, 4), off.started);
    try std.testing.expectEqual(@as(u32, 0), off.here);
    // on: runs within 3000 us of steps end here; 3**200 (past them) is a job, waited on in the same tick
    try std.testing.expectEqual([4]u32{ 1, 1, 1, 1 }, on.ticks);
    try std.testing.expectEqual(@as(u32, 1), on.started);
    try std.testing.expectEqual(@as(u32, 3), on.here);
}

test "runner settle: a job past the wait leaves the hold to a later tick, with the same tokens" {
    var b0: Back = .{ .delay = 5 };
    defer b0.deinit();
    const off = try run(0, &b0, null);
    defer gpa.free(off.owned);
    // 200 us of steps (4000) run 47*83 and 7*6 here; 2**100's job waits 200 us a tick until its gate opens at tick 3
    var b1: Back = .{ .gate = false };
    defer b1.deinit();
    const on = try run(200, &b1, 3);
    defer gpa.free(on.owned);
    try std.testing.expectEqualStrings(off.owned, on.owned);
    try std.testing.expectEqual([4]u32{ 7, 7, 7, 7 }, off.ticks);
    try std.testing.expectEqual([4]u32{ 1, 3, 1, 1 }, on.ticks);
    try std.testing.expectEqual(@as(u32, 2), on.started);
    try std.testing.expectEqual(@as(u32, 2), on.here);
}

test "runner settle: a backend without a run here still settles in one tick when its job ends in the wait" {
    var b: Back = .{ .delay = 3 };
    defer b.deinit();
    var plain = b.backend();
    const vt: rapi.Backend.VTable = .{ .start = plain.vtable.start, .poll = plain.vtable.poll, .cancel = plain.vtable.cancel, .release = plain.vtable.release };
    plain.vtable = &vt;
    try std.testing.expectEqual(null, plain.now(.{ .kind = .graph, .bytes = "7*6" }, 10, 10));
    var r: runner.Runner = .{ .gpa = gpa, .cfg = .{ .settle_us = 3000 }, .frontend = .{ .translate = translate }, .backend = plain, .text = .{ .bytes = bytes, .encode = encode } };
    r.cfg.scan.inline_math = true;
    const s = (try r.session(.{})).?;
    defer s.destroy();
    const hook = s.hook();
    var held = false;
    for ("So 7*6 =") |ch| held = hook.token(hook.ptr, ch, false, true, false);
    try std.testing.expect(held);
    const got = (try s.tick()).?;
    defer gpa.free(got);
    try std.testing.expectEqualSlices(u32, &.{ ' ', '4', '2' }, got);
    try std.testing.expectEqual(@as(u32, 1), s.held_ticks);
    try std.testing.expectEqual(@as(u32, 1), b.started);
}

test "runner settle: TF_DSV41_RUNNER_SETTLE_US parses, off (0) unless set" {
    const K = struct {
        var v: ?[]const u8 = null;
        fn get(name: []const u8) ?[]const u8 {
            if (std.mem.eql(u8, name, "TF_R")) return "1";
            if (std.mem.eql(u8, name, "TF_R_CIRCUIT")) return "/c.bin";
            return if (std.mem.eql(u8, name, "TF_R_SETTLE_US")) v else null;
        }
    };
    try std.testing.expectEqual(@as(u32, 0), (try runner.Config.parse(&K.get, "TF_R")).?.settle_us);
    K.v = "3000";
    try std.testing.expectEqual(@as(u32, 3000), (try runner.Config.parse(&K.get, "TF_R")).?.settle_us);
    K.v = "100001";
    try std.testing.expectError(error.BadKnob, runner.Config.parse(&K.get, "TF_R"));
}
