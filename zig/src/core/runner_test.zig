//! The in-model runner end to end on a lane host: a scripted model writes a cell and its result header; runs own it.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const rapi = @import("runner_api.zig");
const runner = @import("runner.zig");
const LaneHost = @import("lane_host.zig").LaneHost;
const Reason = api.Reason;

const gpa = std.testing.allocator;
const EOS: u32 = 96;
const prompt = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3 };
pub const product = "121932631137021795226185032733622923332237463801111263526900";
pub const mul_cell = "Let me check.\n```python\nprint(123456789012345678901234567890 * 987654321098765432109876543210)\n```\nResult:\n";

/// Character tokens: printable ASCII 32..126 are 0..94, newline 95, EOS 96 (the fake's vocab is 97).
pub fn id(c: u8) u32 {
    return if (c == '\n') 95 else c - 32;
}

fn ids(a: std.mem.Allocator, text: []const u8) ![]u32 {
    const out = try a.alloc(u32, text.len);
    for (text, out) |c, *t| t.* = id(c);
    return out;
}

const table = blk: {
    var t: [97][]const u8 = undefined;
    for (0..95) |i| t[i] = &[_]u8{@intCast(i + 32)};
    t[95] = "\n";
    t[96] = "";
    break :blk t;
};

fn tokenBytes(_: ?*anyopaque, token: u32) []const u8 {
    return if (token < table.len) table[token] else "";
}

fn encode(_: ?*anyopaque, a: std.mem.Allocator, text: []const u8) std.mem.Allocator.Error![]u32 {
    return ids(a, text);
}

fn decode(a: std.mem.Allocator, tokens: []const u32) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    for (tokens) |t| try out.appendSlice(a, tokenBytes(null, t));
    return out.toOwnedSlice(a);
}

/// A model that writes `lead`, then its own guess on one line, then `tail` and EOS, whatever the guess line became.
pub const Model = struct {
    lead: []const u8,
    guess: []const u8 = "42",
    tail: []const u8 = "Done.",
    terminal: bool = false,

    fn next(ctx: *anyopaque, history: []const u32) u32 {
        const m: *Model = @ptrCast(@alignCast(ctx));
        var buf: [1024]u8 = undefined;
        const reply = history[prompt.len..];
        var n: usize = 0;
        for (reply) |t| for (tokenBytes(null, t)) |c| {
            buf[n] = c;
            n += 1;
        };
        const r = buf[0..n];
        if (r.len < m.lead.len) return id(m.lead[r.len]);
        if (m.terminal) return EOS;
        var rest = r[m.lead.len..];
        const nl = std.mem.indexOfScalar(u8, rest, '\n') orelse return if (rest.len < m.guess.len) id(m.guess[rest.len]) else id('\n');
        rest = rest[nl + 1 ..];
        if (std.mem.startsWith(u8, rest, "```\n")) rest = rest[4..];
        return if (rest.len < m.tail.len) id(m.tail[rest.len]) else EOS;
    }
};

/// `print(X)` and REPL `X` become calc X; code naming `undefined_` abstains.
pub fn translate(_: ?*anyopaque, a: std.mem.Allocator, cell: rapi.Cell) std.mem.Allocator.Error!rapi.Translation {
    if (std.mem.indexOf(u8, cell.code, "undefined_") != null) return .{ .abstain = .name };
    // like T5's frontend: a cell after earlier code cells abstains
    if (std.mem.eql(u8, cell.code, "2**99") and cell.prior.len != 0) return .{ .abstain = .unsupported };
    var code = std.mem.trim(u8, cell.code, " \n");
    if (std.mem.startsWith(u8, code, "print(") and std.mem.endsWith(u8, code, ")")) code = code[6 .. code.len - 1];
    return .{ .program = .{ .kind = .calc, .bytes = try a.dupe(u8, code) } };
}

/// Answers from a table after `delay` polls, or only once `gate` opens.
pub const Back = struct {
    delay: u32 = 0,
    gate: std.atomic.Value(bool) = .init(true),
    jobs: std.ArrayList(struct { prog: []u8, polls: u32 }) = .empty,
    started: u32 = 0,
    result: ?rapi.Result = null,

    const answers = [_][2][]const u8{
        .{ "123456789012345678901234567890 * 987654321098765432109876543210", product ++ "\n" },
        .{ "sum(i*i for i in range(10))", "285\n" },
        .{ "2**100", "1267650600228229401496703205376\n" },
        .{ "47 * 83", "3901" },
        .{ "1.5+2.25", "3.75" },
        .{ "2**99", "633825300114114700748351602688\n" },
    };

    pub fn backend(b: *Back) rapi.Backend {
        return .{ .ctx = b, .vtable = &.{ .start = start, .poll = poll, .cancel = cancel, .release = release } };
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
        if (j.polls <= b.delay or !b.gate.load(.acquire)) return .running;
        if (b.result) |result| return .{ .done = result };
        for (answers) |x| if (std.mem.eql(u8, x[0], j.prog)) return .{ .done = .{ .status = .ok, .out = x[1], .steps = 100 } };
        return .{ .done = .{ .status = .fault, .out = "" } };
    }
    fn cancel(_: *anyopaque, _: rapi.JobId) void {}
    fn release(_: *anyopaque, _: u64) void {}
    pub fn deinit(b: *Back) void {
        for (b.jobs.items) |j| gpa.free(j.prog);
        b.jobs.deinit(gpa);
    }
};

const Box = struct {
    mutex: std.Io.Mutex = .init,
    tokens: std.ArrayList(u32) = .empty,
    done: ?Reason = null,
    cells: [65536]u8 = undefined, // the finished event's runner receipt
    cells_len: usize = 0,
    keep: bool = false, // keep `live` (Opts.live): the caller frees it
    live: [32][]u8 = @splat(&.{}), // each record's newest Event.cells note (gpa)
    live_n: usize = 0,
    fn event(ctx: *anyopaque, _: api.Id, e: *const api.Event) void {
        const b: *@This() = @ptrCast(@alignCast(ctx));
        b.mutex.lockUncancelable(std.testing.io);
        defer b.mutex.unlock(std.testing.io);
        switch (e.*) {
            .tokens => |t| b.tokens.appendSlice(gpa, t) catch {},
            .cells => |notes| for (notes) |n| if (b.keep and n.index < b.live.len) {
                gpa.free(b.live[n.index]);
                b.live[n.index] = gpa.dupe(u8, n.json) catch &.{};
                b.live_n = @max(b.live_n, n.index + 1);
            },
            .finished => |f| {
                @memcpy(b.cells[0..f.runner.len], f.runner);
                b.cells_len = f.runner.len;
                b.done = f.reason;
            },
            else => {},
        }
    }
    fn finished(b: *@This()) bool {
        b.mutex.lockUncancelable(std.testing.io);
        defer b.mutex.unlock(std.testing.io);
        return b.done != null;
    }
    fn wait(b: *@This(), seconds: u32) !Reason {
        for (0..@as(usize, seconds) * 1000) |_| {
            if (b.finished()) return b.done.?;
            std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
        }
        return error.Timeout;
    }
};

/// `frontend`/`backend`: real ones in place of the fakes (the proof test).
pub const Opts = struct { runner: bool = true, run_cells: bool = true, delay: u32 = 0, planner: bool = false, streams: usize = 1, max_tokens: u32 = 400, frontend: ?rapi.Frontend = null, backend: ?rapi.Backend = null, wait_s: u32 = 20, exact: bool = false, inline_mark: bool = false, result_tokens: u32 = 512, inline_math: bool = false, trace: bool = false, trace_ask: runner.TraceAsk = .{}, placer: ?*runner.place.Placer = null, summaries: ?[][]u8 = null, cells: bool = true, live: ?[][]u8 = null, abstains: bool = false, commit: bool = false, commit_ask: @import("runner_commit.zig").Ask = .{}, think_end: ?u32 = null };

/// Runs `models` (one request each, concurrently) on one host; the reply texts.
pub fn replies(models: []Model, o: Opts, back: *Back) ![][]u8 {
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var router: Router = .{ .models = models };
    target.script = .{ .ctx = &router, .next = Router.next };
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 4 });
    back.delay = o.delay;
    var run: runner.Runner = .{ .gpa = gpa, .cfg = .{ .circuit = "test", .exact = o.exact, .inline_mark = o.inline_mark, .max_tokens = o.result_tokens, .cells = o.cells, .abstains = o.abstains, .commit = o.commit, .scan = .{ .inline_math = o.inline_math, .trace = o.trace } }, .frontend = o.frontend orelse .{ .translate = translate }, .backend = o.backend orelse back.backend(), .text = .{ .bytes = tokenBytes, .encode = encode }, .placer = o.placer };
    if (o.runner) host.runner = &run;
    const boxes = try gpa.alloc(Box, models.len);
    defer gpa.free(boxes);
    const prompts = try gpa.alloc([prompt.len + 1]u32, models.len);
    defer gpa.free(prompts);
    const requests = try gpa.alloc(api.Request, models.len);
    defer gpa.free(requests);
    const e = host.engine();
    try host.start();
    defer host.stop();
    for (models, 0..) |_, i| {
        boxes[i] = .{ .keep = o.live != null };
        prompts[i] = prompt ++ [_]u32{@intCast(i)}; // the router tells the streams apart by their last prompt token
        requests[i] = .{ .prompt = &prompts[i], .max_tokens = o.max_tokens, .eos = &.{EOS}, .run_cells = o.run_cells, .run_trace = o.trace_ask, .think_end = o.think_end, .run_commit = o.commit_ask };
        try e.submit(@intCast(i + 1), &requests[i], .{ .ctx = &boxes[i], .event = Box.event });
    }
    const out = try gpa.alloc([]u8, models.len);
    for (boxes, out) |*b, *text| {
        try std.testing.expectEqual(Reason.stop, try b.wait(o.wait_s));
        text.* = try decode(gpa, b.tokens.items);
        b.tokens.deinit(gpa);
    }
    for (boxes, 0..) |*b, i| {
        if (o.summaries) |s| s[i] = try gpa.dupe(u8, b.cells[0..b.cells_len]);
        if (o.live) |s| s[i] = try std.mem.join(gpa, ",", b.live[0..b.live_n]);
        for (b.live[0..b.live_n]) |n| gpa.free(n);
    }
    return out;
}

/// Each stream's script by the extra prompt token.
const Router = struct {
    models: []Model,
    fn next(ctx: *anyopaque, history: []const u32) u32 {
        const r: *Router = @ptrCast(@alignCast(ctx));
        return Model.next(&r.models[history[prompt.len]], history[1..]);
    }
};

pub fn freeAll(texts: [][]u8) void {
    for (texts) |t| gpa.free(t);
    gpa.free(texts);
}

pub fn one(lead: []const u8, o: Opts) ![]u8 {
    var back: Back = .{};
    defer back.deinit();
    var m = [_]Model{.{ .lead = lead }};
    const texts = try replies(&m, o, &back);
    defer gpa.free(texts);
    return texts[0];
}

test "runner: the model's own Result: after a multiply cell is the exact product, in one reply" {
    const got = try one(mul_cell, .{});
    defer gpa.free(got);
    try std.testing.expectEqualStrings(mul_cell ++ product ++ "\nDone.", got);
}

test "runner: a blank-delimited same-line result header owns the value before the model guess" {
    const lead = "```python\nprint(2**100)\n```\nResult: ";
    const got = try one(lead, .{});
    defer gpa.free(got);
    try std.testing.expectEqualStrings("```python\nprint(2**100)\n```\nResult:\n1267650600228229401496703205376\nDone.", got);
    const off = try one(lead, .{ .runner = false });
    defer gpa.free(off);
    try std.testing.expectEqualStrings(lead ++ "42\nDone.", off);
}

test "runner: a loop cell's result is 285; a REPL line's next line is its value" {
    const lead = "```python\nprint(sum(i*i for i in range(10)))\n```\nOutput:\n";
    const got = try one(lead, .{});
    defer gpa.free(got);
    try std.testing.expectEqualStrings(lead ++ "285\nDone.", got);
    const repl = "Check:\n>>> 2**100\n";
    const r = try one(repl, .{});
    defer gpa.free(r);
    try std.testing.expectEqualStrings(repl ++ "1267650600228229401496703205376\nDone.", r);
}

test "runner: a bare result fence with no header keeps the model's own text" {
    const lead = "```python\nprint(2**100)\n```\n```text\n";
    const got = try one(lead, .{});
    defer gpa.free(got);
    try std.testing.expectEqualStrings(lead ++ "42\nDone.", got);
}

test "runner: a run that ends late gives the same reply as one that ends early" {
    const early = try one(mul_cell, .{ .delay = 0 });
    defer gpa.free(early);
    const late = try one(mul_cell, .{ .delay = 8 });
    defer gpa.free(late);
    try std.testing.expectEqualStrings(early, late);
}

test "runner: a held stream waits while another stream decodes to its end" {
    var back: Back = .{};
    defer back.deinit();
    back.gate.store(false, .release);
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var models = [_]Model{ .{ .lead = mul_cell }, .{ .lead = "No code here.\n", .guess = "", .tail = "" } };
    var router: Router = .{ .models = &models };
    target.script = .{ .ctx = &router, .next = Router.next };
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 4 });
    var run: runner.Runner = .{ .gpa = gpa, .cfg = .{ .circuit = "test" }, .frontend = .{ .translate = translate }, .backend = back.backend(), .text = .{ .bytes = tokenBytes, .encode = encode } };
    host.runner = &run;
    var boxes = [_]Box{ .{}, .{} };
    const p0 = prompt ++ [_]u32{0};
    const p1 = prompt ++ [_]u32{1};
    const r0: api.Request = .{ .prompt = &p0, .max_tokens = 400, .eos = &.{EOS}, .run_cells = true };
    const r1: api.Request = .{ .prompt = &p1, .max_tokens = 400, .eos = &.{EOS}, .run_cells = true };
    const e = host.engine();
    try host.start();
    defer host.stop();
    try e.submit(1, &r0, .{ .ctx = &boxes[0], .event = Box.event });
    try e.submit(2, &r1, .{ .ctx = &boxes[1], .event = Box.event });
    try std.testing.expectEqual(Reason.stop, try boxes[1].wait(20));
    std.Io.sleep(std.testing.io, .fromMilliseconds(50), .awake) catch {};
    try std.testing.expect(!boxes[0].finished());
    const held = try decode(gpa, boxes[0].tokens.items);
    defer gpa.free(held);
    try std.testing.expectEqualStrings(mul_cell[0 .. mul_cell.len - 1], held); // it stopped exactly at its trigger
    back.gate.store(true, .release);
    try std.testing.expectEqual(Reason.stop, try boxes[0].wait(20));
    const got = try decode(gpa, boxes[0].tokens.items);
    defer gpa.free(got);
    try std.testing.expectEqualStrings(mul_cell ++ product ++ "\nDone.", got);
    for (&boxes) |*b| b.tokens.deinit(gpa);
}

test "runner: four concurrent replies equal their solo runs" {
    const leads = [_][]const u8{ mul_cell, "```python\nprint(sum(i*i for i in range(10)))\n```\nResult:\n", "Check:\n>>> 2**100\n", "Plain prose.\n" };
    var models: [4]Model = undefined;
    for (&models, leads) |*m, l| m.* = .{ .lead = l };
    var back: Back = .{ .delay = 3 };
    defer back.deinit();
    const together = try replies(&models, .{ .delay = 3 }, &back);
    defer freeAll(together);
    for (leads, together) |l, t| {
        const solo = try one(l, .{ .delay = 0 });
        defer gpa.free(solo);
        try std.testing.expectEqualStrings(solo, t);
    }
}

test "runner: an abstained cell, a cell with no header, and the runner off all keep the model's own text" {
    const pseudo = "```python\nprint(undefined_total)\n```\nResult:\n";
    const a = try one(pseudo, .{});
    defer gpa.free(a);
    try std.testing.expectEqualStrings(pseudo ++ "42\nDone.", a);
    const quiet = "```python\nprint(2**100)\n```\nSo that works.\n";
    const b = try one(quiet, .{});
    defer gpa.free(b);
    try std.testing.expectEqualStrings(quiet ++ "42\nDone.", b);
    inline for (.{ Opts{ .runner = false }, Opts{ .run_cells = false } }) |o| {
        const c = try one(mul_cell, o);
        defer gpa.free(c);
        try std.testing.expectEqualStrings(mul_cell ++ "42\nDone.", c);
    }
}

test "runner: an exact (calc) cell keeps the model's text unless TF_DSV41_RUNNER_EXACT is on" {
    const lead = "```calc\n2**100\n```\nResult:\n";
    const off = try one(lead, .{});
    defer gpa.free(off);
    try std.testing.expectEqualStrings(lead ++ "42\nDone.", off);
    const on = try one(lead, .{ .exact = true });
    defer gpa.free(on);
    try std.testing.expectEqualStrings(lead ++ "1267650600228229401496703205376\nDone.", on);
}

test "runner: an inline value is owned after `=`, and only with the inline form on" {
    for ([_][2][]const u8{ .{ "So 47 * 83 =", "So 47 * 83 = 3901\nDone." }, .{ "So 47 * 83 = ", "So 47 * 83 = 3901\nDone." } }) |c| {
        const got = try one(c[0], .{ .inline_math = true });
        defer gpa.free(got);
        try std.testing.expectEqualStrings(c[1], got);
    }
    const off = try one("So 47 * 83 =", .{});
    defer gpa.free(off);
    try std.testing.expectEqualStrings("So 47 * 83 =42\nDone.", off);
    const chain = try one("x = 47 * 83 =", .{ .inline_math = true });
    defer gpa.free(chain);
    try std.testing.expectEqualStrings("x = 47 * 83 =42\nDone.", chain);
}

test "runner: an inline decimal abstains in a Python-float reply; integers stay exact there" {
    const exact = try one("Exact: 1.5+2.25 =", .{ .inline_math = true });
    defer gpa.free(exact);
    try std.testing.expectEqualStrings("Exact: 1.5+2.25 = 3.75\nDone.", exact);
    const float = try one("In Python, as a float: 1.5+2.25 =", .{ .inline_math = true });
    defer gpa.free(float);
    try std.testing.expectEqualStrings("In Python, as a float: 1.5+2.25 =42\nDone.", float);
    const integer = try one("In Python, as a float: 47 * 83 =", .{ .inline_math = true });
    defer gpa.free(integer);
    try std.testing.expectEqualStrings("In Python, as a float: 47 * 83 = 3901\nDone.", integer);
}

test "runner: an inline value earlier in the reply is not notebook state for a later REPL cell" {
    const lead = "First 47 * 83 = 3901\nThen:\n>>> 2**99\n";
    const got = try one(lead, .{ .inline_math = true });
    defer gpa.free(got);
    try std.testing.expectEqualStrings(lead ++ "633825300114114700748351602688\nDone.", got);
}

test "runner: a reply's runtime block counts its cells by form, owned tokens and program kind" {
    var back: Back = .{};
    defer back.deinit();
    var m = [_]Model{ .{ .lead = mul_cell }, .{ .lead = "Check:\n```python\nprint(undefined_x)\n```\nResult:\n" } };
    var sums: [2][]u8 = undefined;
    const texts = try replies(&m, .{ .summaries = &sums }, &back);
    freeAll(texts);
    defer for (sums) |s| gpa.free(s);
    // the product and its newline: one token a character
    try std.testing.expectEqualStrings("{\"fence\":{\"closed\":1,\"owned\":1,\"abstained\":0,\"owned_tokens\":62},\"repl\":{\"closed\":0,\"owned\":0,\"abstained\":0,\"owned_tokens\":0},\"inline\":{\"closed\":0,\"owned\":0,\"abstained\":0,\"owned_tokens\":0},\"trace\":{\"closed\":0,\"owned\":0,\"abstained\":0,\"owned_tokens\":0},\"owned_by\":{\"calc\":1,\"elf\":0,\"graph\":0},\"held_ms\":", sums[0][0 .. std.mem.lastIndexOf(u8, sums[0], "\"held_ms\":").? + 10]);
    try std.testing.expect(std.mem.startsWith(u8, sums[1], "{\"fence\":{\"closed\":1,\"owned\":0,\"abstained\":1,\"owned_tokens\":0}"));
    // the owned product's tokens committed, then the model's own: its resume was timed; an abstain forces nothing
    try std.testing.expect(std.mem.indexOf(u8, sums[0], ",\"resume_ms\":") != null);
    try std.testing.expect(std.mem.endsWith(u8, sums[1], ",\"resume_ms\":0,\"cells\":[]}"));
    var off: [1][]u8 = undefined;
    var lone = [_]Model{.{ .lead = mul_cell }};
    freeAll(try replies(&lone, .{ .runner = false, .summaries = &off }, &back));
    defer gpa.free(off[0]);
    try std.testing.expectEqualStrings("", off[0]);
}

test "runner: a held stream resumes as soon as its run settles, not after the idle loop's full wait" {
    var back: Back = .{};
    defer back.deinit();
    var m = [_]Model{.{ .lead = mul_cell }};
    var sums: [1][]u8 = undefined;
    // the run is done at its 40th poll, one poll a loop turn: most come after the trigger, while the stream is held
    freeAll(try replies(&m, .{ .summaries = &sums, .delay = 40 }, &back));
    defer gpa.free(sums[0]);
    const at = std.mem.lastIndexOf(u8, sums[0], "\"held_ms\":").? + 10;
    const held = try std.fmt.parseInt(u64, sums[0][at..std.mem.indexOfScalarPos(u8, sums[0], at, ',').?], 10);
    try std.testing.expect(held < 1000);
}

test "runner: knobs parse; off unless the flag is 1 with a circuit" {
    const K = struct {
        var set: []const [2][]const u8 = &.{};
        fn get(name: []const u8) ?[]const u8 {
            for (set) |k| if (std.mem.eql(u8, k[0], name)) return k[1];
            return null;
        }
    };
    K.set = &.{};
    try std.testing.expectEqual(null, try runner.Config.parse(&K.get, "TF_R"));
    K.set = &.{.{ "TF_R", "1" }};
    try std.testing.expectError(error.BadKnob, runner.Config.parse(&K.get, "TF_R"));
    K.set = &.{ .{ "TF_R", "1" }, .{ "TF_R_CIRCUIT", "/c.bin" }, .{ "TF_R_FORMS", "repl" }, .{ "TF_R_SPANS", "reasoning" }, .{ "TF_R_MAX_STEPS", "99" } };
    const c = (try runner.Config.parse(&K.get, "TF_R")).?;
    try std.testing.expect(!c.scan.fence and c.scan.repl and c.scan.reasoning_only and c.max_steps == 99 and !c.exact);
    try std.testing.expectEqual(@as(usize, 10522), c.compile_steps);
    try std.testing.expect(!c.inline_mark);
    for ([_][]const u8{ "0", "1" }) |value| {
        K.set = &.{ .{ "TF_R", "1" }, .{ "TF_R_CIRCUIT", "/c.bin" }, .{ "TF_R_INLINE_MARK", value } };
        try std.testing.expectEqual(std.mem.eql(u8, value, "1"), (try runner.Config.parse(&K.get, "TF_R")).?.inline_mark);
    }
    K.set = &.{ .{ "TF_R", "1" }, .{ "TF_R_CIRCUIT", "/c.bin" }, .{ "TF_R_INLINE_MARK", "2" } };
    try std.testing.expectError(error.BadKnob, runner.Config.parse(&K.get, "TF_R"));

    K.set = &.{ .{ "TF_R", "1" }, .{ "TF_R_CIRCUIT", "/c.bin" }, .{ "TF_R_COMPILE_STEPS", "32768" } };
    const wider = (try runner.Config.parse(&K.get, "TF_R")).?;
    try std.testing.expectEqual(@as(usize, 32768), wider.compile_steps);
    try std.testing.expectEqual(c.max_tokens, wider.max_tokens);
    try std.testing.expectEqual(@as(u64, 4_000_000), wider.max_steps);
    for ([_][]const u8{ "0", "32769", "-1", "18446744073709551616", "disabled" }) |value| {
        K.set = &.{ .{ "TF_R", "1" }, .{ "TF_R_CIRCUIT", "/c.bin" }, .{ "TF_R_COMPILE_STEPS", value } };
        try std.testing.expectError(error.BadKnob, runner.Config.parse(&K.get, "TF_R"));
    }
    K.set = &.{ .{ "TF_R", "1" }, .{ "TF_R_CIRCUIT", "/c.bin" }, .{ "TF_R_FORMS", "fence,repl,inline" } };
    const f = (try runner.Config.parse(&K.get, "TF_R")).?;
    try std.testing.expect(f.scan.fence and f.scan.repl and f.scan.inline_math and !c.scan.inline_math);
    K.set = &.{ .{ "TF_R", "1" }, .{ "TF_R_CIRCUIT", "/c.bin" }, .{ "TF_R_EXACT", "1" } };
    try std.testing.expect((try runner.Config.parse(&K.get, "TF_R")).?.exact);
    K.set = &.{ .{ "TF_R", "1" }, .{ "TF_R_CIRCUIT", "/c.bin" }, .{ "TF_R_EXACT", "2" } };
    try std.testing.expectError(error.BadKnob, runner.Config.parse(&K.get, "TF_R"));
    K.set = &.{ .{ "TF_R", "1" }, .{ "TF_R_CIRCUIT", "/c.bin" }, .{ "TF_R_BACKEND", "sandbox" } };
    try std.testing.expectError(error.BadKnob, runner.Config.parse(&K.get, "TF_R"));
    K.set = &.{ .{ "TF_R", "1" }, .{ "TF_R_CIRCUIT", "/c.bin" }, .{ "TF_R_BACKEND", "graph" } };
    try std.testing.expectError(error.BadKnob, runner.Config.parse(&K.get, "TF_R"));
    K.set = &.{ .{ "TF_R", "1" }, .{ "TF_R_CIRCUIT", "/c.bin" }, .{ "TF_R_BACKEND", "graph" }, .{ "TF_R_GRAPH_BANK", "/b/bank.json" }, .{ "TF_R_GUIDE", "1" } };
    const g = (try runner.Config.parse(&K.get, "TF_R")).?;
    try std.testing.expect(g.backend == .graph and std.mem.eql(u8, g.graph_bank, "/b/bank.json") and std.mem.eql(u8, g.guide, "1") and c.guide.len == 0);
    K.set = &.{ .{ "TF_R", "1" }, .{ "TF_R_CIRCUIT", "/c.bin" }, .{ "TF_R_GUIDE", "/g/a.md" } };
    try std.testing.expectEqualStrings("/g/a.md", (try runner.Config.parse(&K.get, "TF_R")).?.guide);
    K.set = &.{ .{ "TF_R", "1" }, .{ "TF_R_CIRCUIT", "/c.bin" }, .{ "TF_R_GUIDE", "0" } };
    try std.testing.expectEqualStrings("", (try runner.Config.parse(&K.get, "TF_R")).?.guide);
    try std.testing.expectEqual(@as(usize, 512), (try runner.Config.parse(&K.get, "TF_R")).?.guide_max_bytes);
    for ([_][]const u8{ "512", "586", "1024", "2048" }) |value| {
        K.set = &.{ .{ "TF_R", "1" }, .{ "TF_R_CIRCUIT", "/c.bin" }, .{ "TF_R_GUIDE_MAX_BYTES", value } };
        try std.testing.expectEqual(try std.fmt.parseInt(usize, value, 10), (try runner.Config.parse(&K.get, "TF_R")).?.guide_max_bytes);
    }
    for ([_][]const u8{ "0", "511", "2049", "-1", "bogus" }) |value| {
        K.set = &.{ .{ "TF_R", "1" }, .{ "TF_R_CIRCUIT", "/c.bin" }, .{ "TF_R_GUIDE_MAX_BYTES", value } };
        try std.testing.expectError(error.BadKnob, runner.Config.parse(&K.get, "TF_R"));
    }
}

test {
    _ = @import("runner.zig");
}

test "runner: a terminal closed fence emits its computed result before one final EOS" {
    for ([_][]const u8{ "```python\nprint(2**100)\n```", "```python\nprint(2**100)\n```\n" }) |lead| {
        var models = [_]Model{.{ .lead = lead, .terminal = true }};
        var back: Back = .{};
        defer back.deinit();
        const got = try replies(&models, .{}, &back);
        defer freeAll(got);
        const want = try std.mem.concat(gpa, u8, &.{ lead, "\nResult:\n1267650600228229401496703205376\n" });
        defer gpa.free(want);
        try std.testing.expectEqualStrings(want, got[0]);
    }
}

test "runner: terminal refusal off and exhausted budget preserve the original reply" {
    for ([_]Opts{ .{}, .{ .runner = false }, .{ .max_tokens = 36 } }, 0..) |opts, i| {
        const lead = if (i == 0) "```python\nprint(undefined_)\n```" else "```python\nprint(2**100)\n```";
        var models = [_]Model{.{ .lead = lead, .terminal = true }};
        var back: Back = .{};
        defer back.deinit();
        var o = opts;
        if (i == 2) o.max_tokens = @intCast(lead.len + 1);
        const got = try replies(&models, o, &back);
        defer freeAll(got);
        try std.testing.expectEqualStrings(lead, got[0]);
    }
}

test {
    _ = @import("runner_handoff_native_test.zig");
    _ = @import("runner_teardown_test.zig");
}

test "runner: inline provenance tags only successful owned inline values" {
    const t = std.testing;
    for ([_][]const u8{ "So 47 * 83 =", "So 47 * 83 = " }) |lead| {
        const got = try one(lead, .{ .inline_math = true, .inline_mark = true });
        defer gpa.free(got);
        try t.expectEqualStrings("So 47 * 83 = 3901 @engine\nDone.", got);
        const off = try one(lead, .{ .inline_math = true });
        defer gpa.free(off);
        try t.expectEqualStrings("So 47 * 83 = 3901\nDone.", off);
    }
    for ([_][]const u8{ "No computation here.", "So 99 * 13 =", "So 47 * undefined_x =" }) |lead| {
        const got = try one(lead, .{ .inline_math = true, .inline_mark = true });
        defer gpa.free(got);
        const off = try one(lead, .{ .inline_math = true });
        defer gpa.free(off);
        try t.expectEqualStrings(off, got);
        try t.expect(std.mem.indexOf(u8, got, @import("runner_output.zig").suffix) == null);
    }
    const cell = "```python\nprint(2**100)\n```\nResult:\n";
    const marked = try one(cell, .{ .inline_mark = true });
    defer gpa.free(marked);
    const plain = try one(cell, .{});
    defer gpa.free(plain);
    try t.expectEqualStrings(plain, marked);
    const disabled = try one("So 47 * 83 =", .{ .inline_math = true, .inline_mark = true, .runner = false });
    defer gpa.free(disabled);
    try t.expectEqualStrings("So 47 * 83 =42\nDone.", disabled);
}

test "runner: inline provenance refuses a payload that cannot fit its complete marker" {
    const got = try one("So 47 * 83 =", .{ .inline_math = true, .inline_mark = true, .result_tokens = 5 });
    defer gpa.free(got);
    try std.testing.expectEqualStrings("So 47 * 83 =42\nDone.", got);
}

test "runner: inline provenance never tags an empty multiline failed or refused result" {
    for ([_]rapi.Result{
        .{ .status = .ok, .out = "\n" },
        .{ .status = .ok, .out = " \n" },
        .{ .status = .ok, .out = "3901\n\n" },
        .{ .status = .ok, .out = "3\n4\n" },
        .{ .status = .fault, .out = "3901" },
        .{ .status = .budget, .out = "3901" },
        .{ .status = .abstain, .out = "3901" },
    }) |res| {
        var back: Back = .{ .result = res };
        defer back.deinit();
        var models = [_]Model{.{ .lead = "So 47 * 83 =" }};
        const got = try replies(&models, .{ .inline_math = true, .inline_mark = true }, &back);
        defer gpa.free(got);
        defer gpa.free(got[0]);
        try std.testing.expectEqualStrings("So 47 * 83 =42\nDone.", got[0]);
    }
}
