//! PREFIX_COMMIT end to end on a lane host (the think closes after the owned value, the answer), and each guard.
const std = @import("std");
const rapi = @import("runner_api.zig");
const runner = @import("runner.zig");
const scan = @import("runner_scan.zig");
const answer = @import("runner_commit.zig");
const rt = @import("runner_test.zig");

const gpa = std.testing.allocator;
const lead = "Let me check.\n```python\nprint(47 * 83)\n```\nResult:\n";
const task = "What is 47 * 83? Give the exact integer.";
const end = rt.id('|'); // the think end: `|` is in no script here
const close = [_]u32{ rt.id('\n'), end, rt.id('\n') };
const ask: answer.Ask = .{ .close = &close, .task = task, .eos = 96, .max_new = 400 };

const Commit = struct { cell: u32, why: []const u8, span: []const u8, value: []const u8, field: []const u8, at: u64, token: u32, close_token: u32, ids: u32, landed: u32, closed: bool, ended: bool };
const Receipt = struct { commit: ?Commit = null, cells: []const struct { cell: u32, value: []const u8, landed: u32, ids: u32 } = &.{} };

/// One reply and its receipt with its wall-clock fields as 0 (both gpa).
fn reply(text: []const u8, o: rt.Opts) !struct { text: []u8, receipt: []u8 } {
    var back: rt.Back = .{};
    defer back.deinit();
    var m = [_]rt.Model{.{ .lead = text }};
    var sums: [1][]u8 = undefined;
    var opts = o;
    opts.summaries = &sums;
    const texts = try rt.replies(&m, opts, &back);
    defer gpa.free(texts);
    defer gpa.free(sums[0]);
    return .{ .text = texts[0], .receipt = try untimed(sums[0]) };
}

/// `receipt` with `held_ms` and `resume_ms` as 0: a loaded host's hold is longer, the reply the same.
fn untimed(receipt: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    var i: usize = 0;
    while (i < receipt.len) {
        const at = for ([_][]const u8{ "\"held_ms\":", "\"resume_ms\":" }) |key| {
            if (std.mem.startsWith(u8, receipt[i..], key)) break key.len;
        } else null;
        if (at) |n| {
            try out.appendSlice(gpa, receipt[i .. i + n]);
            try out.append(gpa, '0');
            i += n;
            while (i < receipt.len and std.ascii.isDigit(receipt[i])) i += 1;
            continue;
        }
        try out.append(gpa, receipt[i]);
        i += 1;
    }
    return out.toOwnedSlice(gpa);
}

fn parse(receipt: []const u8) !std.json.Parsed(Receipt) {
    return std.json.parseFromSlice(Receipt, gpa, receipt, .{ .ignore_unknown_fields = true, .allocate = .alloc_always });
}

test "runner commit: an owned value of the task's expression closes the think and is the answer, with its record" {
    const on = try reply(lead, .{ .commit = true, .commit_ask = ask, .think_end = end });
    defer gpa.free(on.text);
    defer gpa.free(on.receipt);
    try std.testing.expectEqualStrings(lead ++ "3901\n" ++ "\n|\n" ++ "3901", on.text);
    const p = try parse(on.receipt);
    defer p.deinit();
    const c = p.value.commit.?;
    try std.testing.expectEqual(@as(u32, 0), c.cell);
    try std.testing.expectEqualStrings("whole-task", c.why);
    try std.testing.expectEqualStrings("print(47 * 83)\n", c.span);
    try std.testing.expectEqualStrings("3901", c.value);
    try std.testing.expectEqualStrings("content", c.field);
    try std.testing.expectEqual(@as(u32, close.len + 4 + 1), c.ids);
    try std.testing.expectEqual(c.ids, c.landed);
    try std.testing.expect(c.closed and c.ended);
    // the cell's write lands, then the close, then the value: one content byte (the close's newline) before it
    try std.testing.expectEqual(@as(u32, lead.len + 5), c.close_token);
    try std.testing.expectEqual(@as(u32, lead.len + 5 + close.len), c.token);
    try std.testing.expectEqual(@as(u64, 1), c.at);
    try std.testing.expectEqual(@as(usize, 1), p.value.cells.len);
    try std.testing.expectEqualStrings("\n3901\n", p.value.cells[0].value); // from the header's newline
    try std.testing.expectEqual(p.value.cells[0].ids, p.value.cells[0].landed);
}

test "runner commit: knob off is the reply and receipt as before, with or without the request's settings" {
    const base = try reply(lead, .{ .think_end = end });
    defer gpa.free(base.text);
    defer gpa.free(base.receipt);
    try std.testing.expectEqualStrings(lead ++ "3901\nDone.", base.text);
    try std.testing.expect(std.mem.indexOf(u8, base.receipt, "\"commit\"") == null);
    for ([_]rt.Opts{
        .{ .think_end = end, .commit_ask = ask }, // settings, knob off
        .{ .think_end = end, .commit = true }, // knob on, no settings (the server's: no think close, tools, a hold)
    }) |o| {
        const got = try reply(lead, o);
        defer gpa.free(got.text);
        defer gpa.free(got.receipt);
        try std.testing.expectEqualStrings(base.text, got.text);
        try std.testing.expectEqualStrings(base.receipt, got.receipt);
    }
}

test "runner commit: each end-to-end guard keeps the reply as the knob off gives it" {
    const tight = lead.len + 5 + close.len + 4 + 1; // the cell's write and every commit token, one short of the room
    var other = ask;
    other.task = "What is 47 * 84?";
    var extra = ask;
    extra.task = "What is 47 * 83? Show 3 steps.";
    var tools = ask;
    tools.tools = true;
    var short = ask;
    short.max_new = tight;
    const cases = [_]rt.Opts{
        .{ .commit = true, .commit_ask = other, .think_end = end }, // another expression
        .{ .commit = true, .commit_ask = extra, .think_end = end }, // a number the expression leaves out
        .{ .commit = true, .commit_ask = ask }, // no think block: the reply is all answer already
        .{ .commit = true, .commit_ask = tools, .think_end = end },
        .{ .commit = true, .commit_ask = ask, .think_end = end, .cells = false }, // no records: the oracle could not check it
        .{ .commit = true, .commit_ask = short, .think_end = end, .max_tokens = tight },
    };
    for (cases) |o| {
        const off_o = blk: {
            var x = o;
            x.commit = false;
            break :blk x;
        };
        const off = try reply(lead, off_o);
        defer gpa.free(off.text);
        defer gpa.free(off.receipt);
        const got = try reply(lead, o);
        defer gpa.free(got.text);
        defer gpa.free(got.receipt);
        try std.testing.expectEqualStrings(off.text, got.text);
        try std.testing.expectEqualStrings(off.receipt, got.receipt);
    }
    // one more token of room: it fits, and fires
    short.max_new = tight + 1;
    const fit = try reply(lead, .{ .commit = true, .commit_ask = short, .think_end = end, .max_tokens = tight + 1 });
    defer gpa.free(fit.text);
    defer gpa.free(fit.receipt);
    try std.testing.expectEqualStrings(lead ++ "3901\n\n|\n3901", fit.text);
}

test "runner commit: an inline value of the task's expression, written the other way round, commits too" {
    var swapped = ask;
    swapped.task = "What is 83 × 47?";
    const got = try reply("So 47 * 83 =", .{ .commit = true, .commit_ask = swapped, .think_end = end, .inline_math = true });
    defer gpa.free(got.text);
    defer gpa.free(got.receipt);
    try std.testing.expectEqualStrings("So 47 * 83 = 3901\n|\n3901", got.text);
}

/// A session as `answer.after` reads it, every guard passing.
const Fake = struct {
    r: struct { gpa: std.mem.Allocator = gpa, cfg: struct { commit: bool = true, max_tokens: u32 = 512 } = .{}, text: ?runner.Text = .{ .bytes = bytes, .encode = encode } } = .{},
    commit: answer.State = .{ .ask = ask, .think = true },
    rec: struct { on: bool = true, tokens: u32 = 10, commit: ?answer.Record = null } = .{},
    scan: struct { state: enum { prose, fence, after } = .prose } = .{},
    forcing: ?struct { left: usize } = .{ .left = 2 },
    id: u64 = 1,

    fn bytes(_: ?*anyopaque, _: u32) []const u8 {
        return "";
    }
    fn encode(_: ?*anyopaque, a: std.mem.Allocator, text: []const u8) std.mem.Allocator.Error![]u32 {
        const out = try a.alloc(u32, text.len);
        for (text, out) |c, *t| t.* = c;
        return out;
    }
};

const FakeCell = struct { result: ?rapi.Result = .{ .status = .ok, .out = "3901\n" }, stated: ?scan.Stated = null, code: []const u8 = "print(47 * 83)\n", form: rapi.Form = .fence };

/// Whether `after` fires for this session, cell and trigger; frees what it made.
fn fires(s: *Fake, cell: FakeCell, w: scan.Trigger) !bool {
    const ids = try gpa.dupe(u32, &.{ '3', '\n' });
    const got = try answer.after(s, cell, w, 0, ids);
    defer gpa.free(got);
    if (s.rec.commit) |*c| c.deinit(gpa);
    s.rec.commit = null;
    return got.len > 2;
}

test "runner commit: every guard, one mutation at a time (each alone stops it)" {
    var base: Fake = .{};
    try std.testing.expect(try fires(&base, .{}, .{ .index = 0 }));
    try std.testing.expect(base.commit.fired);
    try std.testing.expectEqual(@as(usize, 2 + close.len + 4 + 1), base.forcing.?.left);
    // once only
    try std.testing.expect(!try fires(&base, .{}, .{ .index = 0 }));
    const Mut = struct { why: []const u8, s: Fake = .{}, cell: FakeCell = .{}, w: scan.Trigger = .{ .index = 0 } };
    var no_close = ask;
    no_close.close = &.{};
    var no_eos = ask;
    no_eos.eos = null;
    var tools = ask;
    tools.tools = true;
    var short = ask;
    short.max_new = 10 + 2 + close.len + 4 + 1; // one short
    var answer_max = ask;
    answer_max.answer_max = close.len + 4; // one short
    var extra = ask;
    extra.task = "What is 47 * 83 in base 10?";
    var two = ask;
    two.task = "What is 47 * 83 or 2 + 2?";
    var none = ask;
    none.task = "Multiply the numbers.";
    var div = ask;
    div.task = "What is 3901 divided by 47?";
    var caret = ask;
    caret.task = "What is 2^10?";
    const muts = [_]Mut{
        .{ .why = "knob off", .s = .{ .r = .{ .cfg = .{ .commit = false } } } },
        .{ .why = "no think close", .s = .{ .commit = .{ .ask = no_close, .think = true } } },
        .{ .why = "no end token", .s = .{ .commit = .{ .ask = no_eos, .think = true } } },
        .{ .why = "tools offered", .s = .{ .commit = .{ .ask = tools, .think = true } } },
        .{ .why = "think closed", .s = .{ .commit = .{ .ask = ask, .think = false } } },
        .{ .why = "records off", .s = .{ .rec = .{ .on = false } } },
        .{ .why = "no room", .s = .{ .commit = .{ .ask = short, .think = true } } },
        .{ .why = "answer quota", .s = .{ .commit = .{ .ask = answer_max, .think = true } } },
        .{ .why = "value past max_tokens", .s = .{ .r = .{ .cfg = .{ .max_tokens = 3 } } } },
        .{ .why = "inside a fence", .s = .{ .scan = .{ .state = .fence } } },
        .{ .why = "trace line", .w = .{ .index = 0, .trace = true } },
        .{ .why = "EOS hand-off", .w = .{ .index = 0, .terminal = true } },
        .{ .why = "stated cell", .cell = .{ .stated = .{ .display = "47 * 83", .divmod = false } } },
        .{ .why = "no result", .cell = .{ .result = null } },
        .{ .why = "budget stop", .cell = .{ .result = .{ .status = .budget, .out = "3901\n" } } },
        .{ .why = "fault", .cell = .{ .result = .{ .status = .fault, .out = "3901\n" } } },
        .{ .why = "float value", .cell = .{ .result = .{ .status = .ok, .out = "3901.0\n" } } },
        .{ .why = "two lines", .cell = .{ .result = .{ .status = .ok, .out = "3901\n3901\n" } } },
        .{ .why = "leading zero", .cell = .{ .result = .{ .status = .ok, .out = "03901\n" } } },
        .{ .why = "empty value", .cell = .{ .result = .{ .status = .ok, .out = "\n" } } },
        .{ .why = "another span", .cell = .{ .code = "print(47 * 84)\n" } },
        .{ .why = "a part of the task", .cell = .{ .code = "print(47)\n" } },
        .{ .why = "a call", .cell = .{ .code = "print(mul(47, 83))\n" } },
        .{ .why = "statement after print", .cell = .{ .code = "print(47 * 83)\nx = 1\n" } },
        .{ .why = "rebound name", .cell = .{ .code = "a = 47\na = 48\nprint(a * 83)\n" } },
        .{ .why = "name from a name", .cell = .{ .code = "a = 47\nb = a\nprint(a * 83)\n" } },
        .{ .why = "extra number in task", .s = .{ .commit = .{ .ask = extra, .think = true } } },
        .{ .why = "two expressions in task", .s = .{ .commit = .{ .ask = two, .think = true } } },
        .{ .why = "no expression in task", .s = .{ .commit = .{ .ask = none, .think = true } } },
        .{ .why = "floor for true division", .s = .{ .commit = .{ .ask = div, .think = true } }, .cell = .{ .code = "print(3901 // 47)\n", .result = .{ .status = .ok, .out = "83\n" } } },
        .{ .why = "Python ^ is xor", .s = .{ .commit = .{ .ask = caret, .think = true } }, .cell = .{ .code = "print(2^10)\n", .result = .{ .status = .ok, .out = "8\n" } } },
        .{ .why = "difference swapped", .s = .{ .commit = .{ .ask = blk: {
            var x = ask;
            x.task = "What is 83 - 47?";
            break :blk x;
        }, .think = true } }, .cell = .{ .code = "print(47 - 83)\n", .result = .{ .status = .ok, .out = "-36\n" } } },
    };
    for (muts) |m| {
        var s = m.s;
        if (try fires(&s, m.cell, m.w)) {
            std.debug.print("guard did not hold: {s}\n", .{m.why});
            return error.GuardDidNotHold;
        }
        try std.testing.expect(!s.commit.fired);
    }
}

test "runner commit: what fires besides the plain span (names, REPL, swaps, task spellings)" {
    const Pos = struct { task: []const u8, code: []const u8, form: rapi.Form = .fence, out: []const u8 = "3901\n" };
    const pos = [_]Pos{
        .{ .task = task, .code = "a = 47\nb = 83\nprint(a * b)\n" },
        .{ .task = task, .code = ">>> print(47*83)\n", .form = .repl },
        .{ .task = task, .code = "47*83", .form = .inline_expr },
        .{ .task = "Compute 83 \u{00d7} 47 exactly.", .code = "print(47 * 83)\n" },
        .{ .task = "What is 1,024 * 2?", .code = "print(1024 * 2)\n", .out = "2048\n" },
        .{ .task = "What is 2^10 mod 7?", .code = "print(2**10 % 7)\n", .out = "2\n" },
        .{ .task = "What is 2^10?", .code = "2^10", .form = .inline_expr, .out = "1024\n" },
        .{ .task = "What is 5 minus 7?", .code = "print(5 - 7)\n", .out = "-2\n" },
        .{ .task = "What is 47 * 83?", .code = "x = 47 * 83\nprint(x)\n" },
    };
    for (pos) |p| {
        var s: Fake = .{};
        s.commit.ask.task = p.task;
        if (!try fires(&s, .{ .code = p.code, .form = p.form, .result = .{ .status = .ok, .out = p.out } }, .{ .index = 0 })) {
            std.debug.print("did not fire: {s} | {s}\n", .{ p.task, p.code });
            return error.DidNotFire;
        }
    }
}

test "runner commit: task expressions and integers" {
    const Case = struct { []const u8, ?[]const u8 };
    for ([_]Case{
        .{ "What is 123 + 456? Give the exact integer.", "123+456" },
        .{ "Compute 12 \u{00d7} 34 exactly.", "12*34" },
        .{ "What is 7^20 mod 13?", "7**20%13" },
        .{ "What is 1,000,000 * 3?", "1000000*3" },
        .{ "What is (2 + 3) * 4?", "(2+3)*4" },
        .{ "What is 30-digit 5 * 6?", null },
        .{ "What is 1, 2 * 3?", null },
        .{ "Add 5.", null },
        .{ "Nothing here", null },
    }) |c| {
        const got = try answer.expression(gpa, c[0]);
        defer if (got) |g| gpa.free(g);
        if (c[1]) |want| try std.testing.expectEqualStrings(want, got.?) else try std.testing.expect(got == null);
    }
    for ([_][]const u8{ "0", "7", "-36", "3901" }) |v| try std.testing.expect(answer.integer(v));
    for ([_][]const u8{ "", "-", "-0", "007", "3.5", "1e3", " 1" }) |v| try std.testing.expect(!answer.integer(v));
}

test "runner commit: TF_DSV41_RUNNER_COMMIT parses, off unless set" {
    const K = struct {
        var v: ?[]const u8 = null;
        fn get(name: []const u8) ?[]const u8 {
            if (std.mem.eql(u8, name, "TF_R")) return "1";
            if (std.mem.eql(u8, name, "TF_R_CIRCUIT")) return "/c.bin";
            return if (std.mem.eql(u8, name, "TF_R_COMMIT")) v else null;
        }
    };
    try std.testing.expect(!(try runner.Config.parse(&K.get, "TF_R")).?.commit);
    K.v = "1";
    try std.testing.expect((try runner.Config.parse(&K.get, "TF_R")).?.commit);
    K.v = "2";
    try std.testing.expectError(error.BadKnob, runner.Config.parse(&K.get, "TF_R"));
}

/// Byte tokens for a session driven by hand.
fn byte(_: ?*anyopaque, t: u32) []const u8 {
    const all = comptime blk: {
        var x: [128]u8 = undefined;
        for (&x, 0..) |*c, i| c.* = i;
        break :blk x;
    };
    return if (t < 128) all[t..][0..1] else "";
}

fn bytesOf(_: ?*anyopaque, a: std.mem.Allocator, text: []const u8) std.mem.Allocator.Error![]u32 {
    const out = try a.alloc(u32, text.len);
    for (text, out) |c, *t| t.* = c;
    return out;
}

test "runner commit: after it fires the scanner sees nothing and the end is no hand-off" {
    var back: rt.Back = .{};
    defer back.deinit();
    var r: runner.Runner = .{ .gpa = gpa, .cfg = .{ .circuit = "test", .commit = true }, .frontend = .{ .translate = rt.translate }, .backend = back.backend(), .text = .{ .bytes = byte, .encode = bytesOf } };
    for ([_]bool{ false, true }) |fired| {
        const s = (try r.session(.{})).?;
        defer s.destroy();
        const hook = s.hook();
        // a closed fence: its EOS would hand off for the result (the control, before any firing)
        for ("```python\nprint(2**100)\n```\n") |c| _ = hook.token(hook.ptr, c, true, true, false);
        try std.testing.expectEqual(@as(usize, 1), s.cells.items.len);
        s.commit.fired = fired;
        try std.testing.expectEqual(!fired, hook.eos.?(hook.ptr, true, true));
    }
    const s = (try r.session(.{})).?;
    defer s.destroy();
    const hook = s.hook();
    s.commit.fired = true;
    for ("```python\nprint(2**100)\n```\nResult:\n") |c| try std.testing.expect(!hook.token(hook.ptr, c, true, true, false));
    try std.testing.expectEqual(@as(usize, 0), s.cells.items.len);
}
