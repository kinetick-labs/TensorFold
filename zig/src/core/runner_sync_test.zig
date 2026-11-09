//! TF_DSV41_RUNNER_SYNC_GRAPH: a graph result settles at its trigger (0 ticks) with a hold's tokens; the rest hold.
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

/// Every cell's code is its graph program, but `7*6` (a calc program) and `5*5` (abstains).
fn translate(_: ?*anyopaque, a: std.mem.Allocator, cell: rapi.Cell) std.mem.Allocator.Error!rapi.Translation {
    const code = std.mem.trim(u8, cell.code, " \n");
    if (std.mem.eql(u8, code, "5*5")) return .{ .abstain = .name };
    return .{ .program = .{ .kind = if (std.mem.eql(u8, code, "7*6")) .calc else .graph, .bytes = try a.dupe(u8, code) } };
}

/// Answers by program: as a job after one poll, or here when its steps fit `quick`.
const Back = struct {
    jobs: std.ArrayList([]u8) = .empty,
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
        try b.jobs.append(gpa, try gpa.dupe(u8, p.bytes));
        b.started += 1;
        return b.jobs.items.len - 1;
    }
    fn poll(ctx: *anyopaque, job: rapi.JobId) rapi.Poll {
        const b: *Back = @ptrCast(@alignCast(ctx));
        return .{ .done = answer(b.jobs.items[job]) };
    }
    fn now(ctx: *anyopaque, p: rapi.Program, budget: u64, quick: u64) ?rapi.Result {
        const b: *Back = @ptrCast(@alignCast(ctx));
        if (p.kind != .graph) return null;
        const r = answer(p.bytes);
        if (r.steps > @min(budget, quick)) return null;
        b.here += 1;
        return r;
    }
    fn cancel(_: *anyopaque, _: rapi.JobId) void {}
    fn release(_: *anyopaque, _: u64) void {}
    fn deinit(b: *Back) void {
        for (b.jobs.items) |j| gpa.free(j);
        b.jobs.deinit(gpa);
    }
};

const reply = "So 47*83 = 3901\nand 2**100 = 1267650600228229401496703205376\nthen 7*6 = 42\nLast 3**200 = 9\nand 5*5 = 25\n";

/// Each fire's forced text ('|' after each; "-" for an abstain), and each fire's ticks (0: settled at its trigger).
const Run = struct { owned: []u8, ticks: [5]u32, started: u32, here: u32 };

/// Feeds `reply` one token at a time as a stream would: a settled fire's tokens are taken at once, a hold ticks.
fn run(sync: bool, back: *Back) !Run {
    var r: runner.Runner = .{ .gpa = gpa, .cfg = .{ .backend = .graph, .sync_graph = sync }, .frontend = .{ .translate = translate }, .backend = back.backend(), .text = .{ .bytes = bytes, .encode = encode } };
    if (sync) r.quick = .{ .translate = translate };
    r.cfg.scan.inline_math = true;
    const s = (try r.session(.{})).?;
    defer s.destroy();
    const hook = s.hook();
    try std.testing.expectEqual(sync, hook.settled != null);
    var owned: std.ArrayList(u8) = .empty;
    errdefer owned.deinit(gpa);
    var ticks: [5]u32 = @splat(0);
    var fires: usize = 0;
    var abstains_at: usize = 0; // a hold's abstain: the cell's own text
    for (reply) |ch| {
        const before = s.abstained_n;
        if (!hook.token(hook.ptr, ch, false, true, false)) {
            // settled as an abstain at the trigger: no hold, the model's own text stays
            if (s.abstained_n > before) {
                try owned.appendSlice(gpa, "-|");
                fires += 1;
            }
            continue;
        }
        if (hook.settled) |f| if (f(hook.ptr)) |ids| {
            try std.testing.expectEqual(null, s.waiting);
            for (ids) |t| try owned.appendSlice(gpa, bytes(null, t));
            try owned.append(gpa, '|');
            fires += 1;
            continue;
        };
        const cell = s.cells.items[s.waiting.?.index];
        while (!cell.translated.load(.acquire)) std.Thread.yield() catch {};
        var n: u32 = 0;
        const got = while (true) {
            n += 1;
            if (try s.tick()) |x| break x;
        };
        defer gpa.free(got);
        try std.testing.expectEqual(n, s.held_ticks);
        if (got.len == 0) {
            try owned.appendSlice(gpa, "-|");
            abstains_at += 1;
        } else {
            for (got) |t| try owned.appendSlice(gpa, bytes(null, t));
            try owned.append(gpa, '|');
        }
        ticks[fires] = n;
        fires += 1;
    }
    try std.testing.expectEqual(@as(usize, 5), fires);
    return .{ .owned = try owned.toOwnedSlice(gpa), .ticks = ticks, .started = back.started, .here = back.here };
}

test "runner sync graph: a graph result settles at its trigger in 0 ticks with the hold's tokens; the rest hold" {
    var b0: Back = .{};
    defer b0.deinit();
    const off = try run(false, &b0);
    defer gpa.free(off.owned);
    var b1: Back = .{};
    defer b1.deinit();
    const on = try run(true, &b1);
    defer gpa.free(on.owned);
    const big = Back.answers[3][1];
    const want = " 3901| 1267650600228229401496703205376| 42| " ++ big[0 .. big.len - 1] ++ "|-|";
    try std.testing.expectEqualStrings(want, off.owned);
    try std.testing.expectEqualStrings(off.owned, on.owned);
    // off (the negative control): every fire holds, a tick to start its run and one to take it; all are jobs
    try std.testing.expectEqual([5]u32{ 2, 2, 2, 2, 1 }, off.ticks);
    try std.testing.expectEqual(@as(u32, 4), off.started);
    try std.testing.expectEqual(@as(u32, 0), off.here);
    // on: 47*83 and 2**100 settle at `=` (run here); calc 7*6 and 3**200 (past the cap) hold; 5*5 abstains at `=`
    try std.testing.expectEqual([5]u32{ 0, 0, 2, 2, 0 }, on.ticks);
    try std.testing.expectEqual(@as(u32, 2), on.here);
    try std.testing.expectEqual(@as(u32, 2), on.started);
}

test "runner sync graph: TF_DSV41_RUNNER_SYNC_GRAPH parses, off unless set, and only with the graph backend" {
    const K = struct {
        var v: ?[]const u8 = null;
        var backend: ?[]const u8 = "graph";
        fn get(name: []const u8) ?[]const u8 {
            if (std.mem.eql(u8, name, "TF_R")) return "1";
            if (std.mem.eql(u8, name, "TF_R_CIRCUIT")) return "/c.bin";
            if (std.mem.eql(u8, name, "TF_R_GRAPH_BANK")) return "/b.json";
            if (std.mem.eql(u8, name, "TF_R_BACKEND")) return backend;
            return if (std.mem.eql(u8, name, "TF_R_SYNC_GRAPH")) v else null;
        }
    };
    try std.testing.expect(!(try runner.Config.parse(&K.get, "TF_R")).?.sync_graph);
    K.v = "1";
    try std.testing.expect((try runner.Config.parse(&K.get, "TF_R")).?.sync_graph);
    K.v = "2";
    try std.testing.expectError(error.BadKnob, runner.Config.parse(&K.get, "TF_R"));
    K.v = "1";
    K.backend = "circuit";
    try std.testing.expectError(error.BadKnob, runner.Config.parse(&K.get, "TF_R"));
}
