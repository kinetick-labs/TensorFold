//! The digit stop on reply sessions: owned inline values arm it in reasoning; REPL, fences, off or no reasoning do not.
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

fn translate(_: ?*anyopaque, a: std.mem.Allocator, cell: rapi.Cell) std.mem.Allocator.Error!rapi.Translation {
    return .{ .program = .{ .kind = .graph, .bytes = try a.dupe(u8, std.mem.trim(u8, cell.code, " \n")) } };
}

/// Every run ends at once: 47*83, 7*6 (REPL) and print(7*6) (a fence) by their values, anything else a fault.
const Back = struct {
    jobs: std.ArrayList([]u8) = .empty,

    fn answer(prog: []const u8) rapi.Result {
        const answers = [_][2][]const u8{ .{ "47*83", "3901\n" }, .{ "7*6", "42\n" }, .{ "print(7*6)", "42\n" } };
        for (answers) |x| if (std.mem.eql(u8, x[0], prog)) return .{ .status = .ok, .out = x[1], .steps = 10 };
        return .{ .status = .fault };
    }
    fn backend(b: *Back) rapi.Backend {
        return .{ .ctx = b, .vtable = &.{ .start = start, .poll = poll, .cancel = cancel, .release = release } };
    }
    fn start(ctx: *anyopaque, _: u64, p: rapi.Program, _: u64) error{ OutOfMemory, Busy }!rapi.JobId {
        const b: *Back = @ptrCast(@alignCast(ctx));
        try b.jobs.append(gpa, try gpa.dupe(u8, p.bytes));
        return b.jobs.items.len - 1;
    }
    fn poll(ctx: *anyopaque, job: rapi.JobId) rapi.Poll {
        const b: *Back = @ptrCast(@alignCast(ctx));
        return .{ .done = answer(b.jobs.items[job]) };
    }
    fn cancel(_: *anyopaque, _: rapi.JobId) void {}
    fn release(_: *anyopaque, _: u64) void {}
    fn deinit(b: *Back) void {
        for (b.jobs.items) |j| gpa.free(j);
        b.jobs.deinit(gpa);
    }
};

const Case = struct { lead: []const u8, think_open: bool = true, after_value: bool = true, mark: bool = false, after_mark: bool = true };

/// After the lead's hold and owned tokens: whether a drawn "5", " ", "e" or "?" is replaced, and the owned text.
const Seen = struct { digit: bool, space: bool, letter: bool, punct: bool, armed: bool, owned: []u8 };

fn run(case: Case) !Seen {
    var back: Back = .{};
    defer back.deinit();
    var r: runner.Runner = .{ .gpa = gpa, .cfg = .{}, .frontend = .{ .translate = translate }, .backend = back.backend(), .text = .{ .bytes = bytes, .encode = encode } };
    r.cfg.scan.inline_math = true;
    r.cfg.digit_stop.after_value = case.after_value;
    r.cfg.digit_stop.after_mark = case.after_mark;
    r.cfg.inline_mark = case.mark;
    const s = (try r.session(.{})).?;
    defer s.destroy();
    const hook = s.hook();
    var held = false;
    for (case.lead) |ch| if (hook.token(hook.ptr, ch, case.think_open, true, false)) {
        held = true;
        break;
    };
    try std.testing.expect(held);
    const cell = s.cells.items[s.waiting.?.index];
    while (!cell.translated.load(.acquire)) std.Thread.yield() catch {};
    const ids = for (0..64) |_| {
        if (try s.tick()) |x| break x;
    } else return error.NeverSettled;
    defer gpa.free(ids);
    const armed = s.digits.lead != null;
    // the owned tokens land as forced rows; the stop is asked about the model's next draw
    for (ids) |t| _ = hook.token(hook.ptr, t, case.think_open, false, false);
    var owned: std.ArrayList(u8) = .empty;
    for (ids) |t| try owned.appendSlice(gpa, bytes(null, t));
    return .{
        .digit = hook.stop.?(hook.ptr, &.{'5'}, 0) != null,
        .space = hook.stop.?(hook.ptr, &.{' '}, 0) != null,
        .letter = hook.stop.?(hook.ptr, &.{'e'}, 0) != null,
        .punct = hook.stop.?(hook.ptr, &.{'?'}, 0) != null,
        .armed = armed,
        .owned = try owned.toOwnedSlice(gpa),
    };
}

test "runner digit stop: an owned inline integer in reasoning stops the next digit, and not with the stop off" {
    const on = try run(.{ .lead = "So 47*83 =" });
    defer gpa.free(on.owned);
    try std.testing.expectEqualStrings(" 3901", on.owned);
    try std.testing.expect(on.armed and on.digit and !on.space);
    const off = try run(.{ .lead = "So 47*83 =", .after_value = false });
    defer gpa.free(off.owned);
    try std.testing.expectEqualStrings(on.owned, off.owned); // the owned value itself is unchanged
    try std.testing.expect(!off.armed and !off.digit); // negative control
}

test "runner digit stop: with the inline mark a letter or digit glued to its end is stopped, not a space or a question mark" {
    const on = try run(.{ .lead = "So 47*83 =", .mark = true });
    defer gpa.free(on.owned);
    try std.testing.expectEqualStrings(" 3901 @engine", on.owned);
    try std.testing.expect(on.armed and on.letter and on.digit and !on.space and !on.punct);
    // negative control: TF_R_MARK_STOP=0 (as before it) leaves the mark unguarded, the owned text the same
    const off = try run(.{ .lead = "So 47*83 =", .mark = true, .after_mark = false });
    defer gpa.free(off.owned);
    try std.testing.expectEqualStrings(on.owned, off.owned);
    try std.testing.expect(!off.armed and !off.letter and !off.digit and !off.space);
    // without the mark a plain integer still guards digits only: a letter after "3901" is the model's own
    const plain = try run(.{ .lead = "So 47*83 =" });
    defer gpa.free(plain.owned);
    try std.testing.expect(plain.armed and plain.digit and !plain.letter);
}

test "runner digit stop: REPL and fence results end in their newline tail and do not arm" {
    for ([_][]const u8{ ">>> 7*6\n", "```python\nprint(7*6)\n```\nResult:\n" }) |lead| {
        const seen = try run(.{ .lead = lead });
        defer gpa.free(seen.owned);
        try std.testing.expect(std.mem.indexOf(u8, seen.owned, "42") != null);
        try std.testing.expect(std.mem.endsWith(u8, seen.owned, "\n"));
        try std.testing.expect(!seen.armed and !seen.digit);
    }
}

test "runner digit stop: outside reasoning an owned value leaves nothing armed" {
    const seen = try run(.{ .lead = "So 47*83 =", .think_open = false });
    defer gpa.free(seen.owned);
    try std.testing.expectEqualStrings(" 3901", seen.owned);
    try std.testing.expect(!seen.digit and !seen.space);
}

test "runner digit stop: TF_R_DIGIT_STOP, _MARK_STOP, _RUN_MAX and _HOLD_MAX parse: on, on, a 1024 cap, a 64 hold unless set" {
    const K = struct {
        var set: []const [2][]const u8 = &.{};
        fn get(name: []const u8) ?[]const u8 {
            for (set) |k| if (std.mem.eql(u8, k[0], name)) return k[1];
            return null;
        }
    };
    K.set = &.{ .{ "TF_R", "1" }, .{ "TF_R_CIRCUIT", "/c.bin" } };
    const d = (try runner.Config.parse(&K.get, "TF_R")).?;
    try std.testing.expect(d.digit_stop.after_value and d.digit_stop.after_mark and d.digit_stop.run_max == 1024 and d.digit_stop.hold_max == 64);
    K.set = &.{ .{ "TF_R", "1" }, .{ "TF_R_CIRCUIT", "/c.bin" }, .{ "TF_R_MARK_STOP", "0" } };
    try std.testing.expect(!(try runner.Config.parse(&K.get, "TF_R")).?.digit_stop.after_mark);
    K.set = &.{ .{ "TF_R", "1" }, .{ "TF_R_CIRCUIT", "/c.bin" }, .{ "TF_R_MARK_STOP", "2" } };
    try std.testing.expectError(error.BadKnob, runner.Config.parse(&K.get, "TF_R"));
    K.set = &.{ .{ "TF_R", "1" }, .{ "TF_R_CIRCUIT", "/c.bin" }, .{ "TF_R_DIGIT_STOP", "0" }, .{ "TF_R_DIGIT_RUN_MAX", "0" } };
    const off = (try runner.Config.parse(&K.get, "TF_R")).?;
    try std.testing.expect(!off.digit_stop.after_value and off.digit_stop.run_max == 0);
    K.set = &.{ .{ "TF_R", "1" }, .{ "TF_R_CIRCUIT", "/c.bin" }, .{ "TF_R_DIGIT_HOLD_MAX", "0" } };
    try std.testing.expectEqual(@as(u32, 0), (try runner.Config.parse(&K.get, "TF_R")).?.digit_stop.hold_max);
    K.set = &.{ .{ "TF_R", "1" }, .{ "TF_R_CIRCUIT", "/c.bin" }, .{ "TF_R_DIGIT_HOLD_MAX", "x" } };
    try std.testing.expectError(error.BadKnob, runner.Config.parse(&K.get, "TF_R"));
    for ([_][]const u8{ "2", "x" }) |v| {
        K.set = &.{ .{ "TF_R", "1" }, .{ "TF_R_CIRCUIT", "/c.bin" }, .{ "TF_R_DIGIT_STOP", v } };
        try std.testing.expectError(error.BadKnob, runner.Config.parse(&K.get, "TF_R"));
    }
    K.set = &.{ .{ "TF_R", "1" }, .{ "TF_R_CIRCUIT", "/c.bin" }, .{ "TF_R_DIGIT_RUN_MAX", "-1" } };
    try std.testing.expectError(error.BadKnob, runner.Config.parse(&K.get, "TF_R"));
}
