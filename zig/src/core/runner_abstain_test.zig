//! Abstain records end to end on a lane host (each reason forced, records on/off byte-identical) and on a fake session.
const std = @import("std");
const rapi = @import("runner_api.zig");
const scan = @import("runner_scan.zig");
const records = @import("runner_receipt.zig");
const abstain = @import("runner_abstain.zig");
const rt = @import("runner_test.zig");

const gpa = std.testing.allocator;
const t = std.testing;

const Ab = struct { cell: u32, form: []const u8, code: []const u8, display: ?[]const u8, prior: []const []const u8, source_field: ?[]const u8, source_at: ?u64, source_end: ?u64, source_close_at: ?u64, program_kind: []const u8, backend_route: []const u8, stage: []const u8, reason: []const u8, error_name: ?[]const u8, resume_field: []const u8, resume_at: u64, resume_token: u32, round: u32, next_decision_token: u32, reply_tokens: u32, owned_tokens: u32, steps: u64, held_ms: u64 };
const Count = struct { closed: u32, owned: u32, abstained: u32 };
const Receipt = struct { fence: Count, repl: Count, @"inline": Count, trace: Count, abstain_schema: u32 = 0, abstains: []const Ab = &.{}, abstains_complete: bool = false, abstains_dropped: u32 = 99 };

fn parse(receipt: []const u8) !std.json.Parsed(Receipt) {
    return std.json.parseFromSlice(Receipt, gpa, receipt, .{ .ignore_unknown_fields = true, .allocate = .alloc_always });
}

const mul_lead = "Need multiply A=987654 B=137. Need exact product.\nLet's do long multiplication.\n";
const fence_lead = "```python\nprint(2**100)\n```\nResult:\n";

/// `receipt` with its wall-clock fields (held_ms, resume_ms) zeroed: everything else must match byte for byte.
fn untimed(receipt: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    var i: usize = 0;
    while (i < receipt.len) {
        const key = for ([_][]const u8{ "\"held_ms\":", "\"resume_ms\":" }) |k| {
            if (std.mem.startsWith(u8, receipt[i..], k)) break k;
        } else null;
        if (key) |k| {
            try out.appendSlice(gpa, k);
            try out.append(gpa, '0');
            i += k.len;
            while (i < receipt.len and std.ascii.isDigit(receipt[i])) i += 1;
        } else {
            try out.append(gpa, receipt[i]);
            i += 1;
        }
    }
    return out.toOwnedSlice(gpa);
}

/// One forced abstain: the reply, the runner's options and the fake run's result; what its record must say.
const Case = struct { lead: []const u8, guess: []const u8 = "42", o: rt.Opts = .{}, result: ?rapi.Result = null, reason: []const u8, stage: []const u8, error_name: ?[]const u8 = null, kind: []const u8 = "calc", source: ?[]const u8 = null };

const cases = [_]Case{
    .{ .lead = "```python\nprint(undefined_total)\n```\nResult:\n", .reason = "name", .stage = "frontend", .error_name = "name", .kind = "none", .source = "print(undefined_total)\n" },
    .{ .lead = "```calc\n2**100\n```\nResult:\n", .reason = "language", .stage = "frontend", .error_name = "language", .kind = "none", .source = "2**100\n" },
    .{ .lead = "In Python, as a float: 1.5+2.25 =", .o = .{ .inline_math = true }, .reason = "context", .stage = "context", .error_name = "context", .kind = "none", .source = "1.5+2.25" },
    .{ .lead = "So 99 * 13 =", .o = .{ .inline_math = true }, .reason = "fault", .stage = "execute", .error_name = "fault", .source = "99 * 13" },
    .{ .lead = "So 47 * 83 =", .o = .{ .inline_math = true }, .result = .{ .status = .budget, .out = "3901" }, .reason = "budget", .stage = "execute", .error_name = "budget" },
    .{ .lead = "So 47 * 83 =", .o = .{ .inline_math = true }, .result = .{ .status = .abstain, .abstain = .too_big }, .reason = "backend_abstain", .stage = "execute", .error_name = "too_big" },
    .{ .lead = fence_lead, .result = .{ .status = .ok, .out = "" }, .reason = "empty_output", .stage = "own", .source = "print(2**100)\n" },
    .{ .lead = "So 47 * 83 =", .o = .{ .inline_math = true }, .result = .{ .status = .ok, .out = "3\n4\n" }, .reason = "multiline_output", .stage = "own" },
    .{ .lead = "So 47 * 83 =", .o = .{ .inline_math = true, .inline_mark = true }, .result = .{ .status = .ok, .out = " \n" }, .reason = "mark_value_shape", .stage = "own" },
    .{ .lead = "So 47 * 83 =", .o = .{ .inline_math = true, .inline_mark = true, .result_tokens = 5 }, .reason = "token_cap", .stage = "own" },
    .{ .lead = mul_lead, .guess = "ok", .o = .{ .trace = true }, .result = .{ .status = .ok, .out = "1\n2\n" }, .reason = "trace_shape", .stage = "own" },
    .{ .lead = mul_lead, .guess = "ok", .o = .{ .trace = true, .result_tokens = 5 }, .result = .{ .status = .ok, .out = "135308598\n" }, .reason = "token_cap", .stage = "own" },
};

/// The reply and receipt of `c` with abstain records on or off.
fn run(c: Case, on: bool) !struct { text: []u8, receipt: []u8 } {
    var back: rt.Back = .{ .result = c.result };
    defer back.deinit();
    var m = [_]rt.Model{.{ .lead = c.lead, .guess = c.guess }};
    var sums: [1][]u8 = undefined;
    var o = c.o;
    o.summaries = &sums;
    o.abstains = on;
    const texts = try rt.replies(&m, o, &back);
    defer gpa.free(texts);
    return .{ .text = texts[0], .receipt = sums[0] };
}

test "runner abstain records: each reason forced on a lane host, with its source, stage and decision point" {
    for (cases) |c| {
        const on = try run(c, true);
        defer gpa.free(on.text);
        defer gpa.free(on.receipt);
        const p = try parse(on.receipt);
        defer p.deinit();
        const r = p.value;
        errdefer std.debug.print("case {s} / {s}: {s}\n", .{ c.lead, c.reason, on.receipt });
        // the model's own text stands: its guess right after the decision, nothing owned
        try t.expect(std.mem.endsWith(u8, on.text, "\nDone."));
        try t.expectEqualStrings(c.guess, on.text[on.text.len - 6 - c.guess.len ..][0..c.guess.len]);
        try t.expectEqual(1, r.abstain_schema);
        try t.expect(r.abstains_complete);
        try t.expectEqual(0, r.abstains_dropped);
        try t.expectEqual(1, r.abstains.len);
        // conservation: one record per abstained cell over every form's counter
        try t.expectEqual(r.abstains.len, r.fence.abstained + r.repl.abstained + r.@"inline".abstained + r.trace.abstained);
        const a = r.abstains[0];
        try t.expectEqualStrings(c.reason, a.reason);
        try t.expectEqualStrings(c.stage, a.stage);
        if (c.error_name) |e| try t.expectEqualStrings(e, a.error_name.?);
        try t.expectEqualStrings(c.kind, a.program_kind);
        try t.expectEqualStrings(if (std.mem.eql(u8, c.kind, "none")) "none" else "circuit", a.backend_route);
        try t.expectEqual(0, a.owned_tokens);
        // one token a character: the decision is the trigger's end, where the model's own continuation begins
        try t.expectEqualStrings("content", a.resume_field);
        try t.expectEqual(a.resume_at, a.resume_token);
        try t.expect(a.resume_at <= c.lead.len and a.resume_at + 2 >= c.lead.len);
        try t.expectEqualStrings(c.lead[0..a.resume_at], on.text[0..a.resume_at]);
        try t.expectEqual(on.text.len + 1, a.reply_tokens); // the EOS is a committed token too
        try t.expectEqual(a.reply_tokens, a.next_decision_token);
        try t.expectEqualStrings("content", a.source_field.?);
        try t.expect(a.source_close_at.? <= a.resume_at);
        if (c.source) |want| {
            try t.expectEqualStrings(want, on.text[a.source_at.?..a.source_end.?]);
            try t.expect(std.mem.eql(u8, want, a.code) or std.mem.eql(u8, want, std.mem.trim(u8, a.code, " ")));
        }
        if (a.source_at) |at| try t.expectEqualStrings(std.mem.trim(u8, a.code, " \t\r\n"), std.mem.trim(u8, on.text[at..a.source_end.?], " \t\r\n"));
        if (std.mem.eql(u8, a.form, "trace")) try t.expect(a.display != null) else try t.expect(a.display == null);
        // records off: the same tokens, and the receipt is the one before abstain records, byte for byte
        const off = try run(c, false);
        defer gpa.free(off.text);
        defer gpa.free(off.receipt);
        try t.expectEqualStrings(off.text, on.text);
        try t.expect(std.mem.indexOf(u8, off.receipt, "\"abstains") == null and std.mem.indexOf(u8, off.receipt, "abstain_") == null);
        const on_u = try untimed(on.receipt);
        defer gpa.free(on_u);
        const off_u = try untimed(off.receipt);
        defer gpa.free(off_u);
        try t.expect(std.mem.startsWith(u8, on_u, off_u[0 .. off_u.len - 1]));
        try t.expect(std.mem.startsWith(u8, on_u[off_u.len - 1 ..], ",\"abstain_schema\":1,\"abstains\":["));
    }
}

test "runner abstain records: the reason of every frontend tag, a lost start, and the run's own abstain tag" {
    const Fake = struct { translation: rapi.Translation, result: ?rapi.Result = null };
    for (std.enums.values(rapi.Abstain)) |tag| {
        const why = abstain.reason(Fake{ .translation = .{ .abstain = tag } }, null);
        try t.expectEqualStrings(@tagName(tag), @tagName(why[0]));
        try t.expectEqualStrings(@tagName(tag), why[1].?);
        try t.expectEqualStrings(switch (tag) {
            .context => "context",
            .markup => "own", // the run's screen gives it (runner_binding.Family.never), not a frontend
            else => "frontend",
        }, abstain.stage(why[0]));
    }
    const prog: rapi.Translation = .{ .program = .{ .kind = .graph, .bytes = "" } };
    const lost = abstain.reason(Fake{ .translation = prog }, null);
    try t.expectEqual(abstain.Reason.out_of_memory, lost[0]);
    try t.expectEqualStrings("dispatch", abstain.stage(lost[0]));
    const bare = abstain.reason(Fake{ .translation = prog, .result = .{ .status = .abstain } }, null);
    try t.expectEqual(abstain.Reason.backend_abstain, bare[0]);
    try t.expect(bare[1] == null);
    const marked = abstain.reason(Fake{ .translation = prog, .result = .{ .status = .abstain, .abstain = .markup } }, null);
    try t.expectEqual(abstain.Reason.markup, marked[0]);
    try t.expectEqualStrings("markup", marked[1].?);
    try t.expectEqual(abstain.Reason.token_cap, abstain.reason(Fake{ .translation = prog, .result = .{ .status = .ok, .out = "1" } }, .token_cap)[0]);
}

/// The fields records.write and abstain.told read off a runner.Session and its cells.
const FakeCell = struct {
    form: rapi.Form = .fence,
    stated: ?scan.Stated = null,
    told: enum { none, owned, abstained } = .none,
    abstained: bool = false,
    owned_tokens: u32 = 0,
    translation: rapi.Translation = .{ .program = .{ .kind = .graph, .bytes = "" } },
    result: ?rapi.Result = .{ .status = .fault, .steps = 7 },
    lang: []const u8 = "python",
    code: []const u8 = "x = 1\n",
};

const FakeSession = struct {
    r: struct { gpa: std.mem.Allocator = gpa } = .{},
    cells: struct { items: []const *FakeCell },
    held_ms: u64 = 0,
    resume_ms: u64 = 0,
    abstained_n: u32 = 0,
    rec: records.Log = .{ .abstains_on = true },
    fn feed(s: *FakeSession, text: []const u8) !void {
        for (text) |c| try s.rec.token(gpa, c, &[_]u8{c}, false);
    }
};

test "runner abstain records: mixed decisions, the source's last copy, a normalized source, the next decision, completeness" {
    var c0: FakeCell = .{ .code = "y = 2\n" };
    var c1: FakeCell = .{ .form = .repl, .code = "y\nz\n" };
    var c2: FakeCell = .{ .code = "y = 2\n", .translation = .{ .abstain = .syntax }, .result = null };
    var s: FakeSession = .{ .cells = .{ .items = &.{ &c0, &c1, &c2 } } };
    defer s.rec.deinit(gpa);
    try s.feed("```python\ny = 2\n```\n");
    s.rec.closed(gpa, 0);
    try s.feed("Out:");
    try s.rec.owned(gpa, 0, &.{ 'O', 'K' });
    c0.told = .owned;
    try s.feed("OK\n>>> y\n>>> z\n");
    s.rec.closed(gpa, 1);
    abstain.told(&s, &c1, 1, 3, null); // fault: the run's own status
    try s.feed("5\nAgain:\n```python\ny = 2\n```\n");
    s.rec.closed(gpa, 2);
    s.rec.rounds = 4;
    abstain.told(&s, &c2, 2, 0, null);
    try s.feed("tail");
    const got = records.write(&s);
    const p = try parse(got);
    defer p.deinit();
    const r = p.value;
    try t.expectEqual(2, s.abstained_n);
    try t.expect(c1.told == .abstained and c1.abstained and c2.told == .abstained);
    try t.expectEqual(2, r.abstains.len);
    const a1 = r.abstains[0];
    try t.expectEqualStrings("fault", a1.reason);
    try t.expectEqual(7, a1.steps);
    try t.expectEqual(3, a1.held_ms);
    try t.expect(a1.source_at == null); // a REPL line's code is not its bytes in the reply (the prompt is cut)
    try t.expectEqual(1, a1.prior.len);
    try t.expectEqualStrings("y = 2\n", a1.prior[0]);
    try t.expectEqual(a1.resume_token, a1.resume_at);
    try t.expectEqual(r.abstains[1].resume_token, a1.next_decision_token); // the next decision: the later abstain
    const a2 = r.abstains[1];
    const text = "```python\ny = 2\n```\nOut:OK\n>>> y\n>>> z\n5\nAgain:\n```python\ny = 2\n```\ntail";
    try t.expectEqualStrings("syntax", a2.reason);
    try t.expectEqualStrings("none", a2.program_kind);
    try t.expectEqual(std.mem.lastIndexOf(u8, text[0..a2.source_close_at.?], "y = 2\n").?, a2.source_at.?); // its own copy
    try t.expect(a2.source_at.? > 10);
    try t.expectEqual(4, a2.round);
    try t.expectEqual(text.len, a2.next_decision_token);
    try t.expectEqual(text.len, a2.reply_tokens);
    try t.expectEqual(2, a2.prior.len);
    try t.expect(r.abstains_complete);
    // a record lost to memory: the receipt says the census is incomplete
    s.rec.dropped = 1;
    const lossy = try parse(records.write(&s));
    defer lossy.deinit();
    try t.expect(!lossy.value.abstains_complete);
    try t.expectEqual(1, lossy.value.abstains_dropped);
    // abstain records off: nothing kept, nothing written, the receipt as before
    var c3: FakeCell = .{};
    var off: FakeSession = .{ .cells = .{ .items = &.{&c3} }, .rec = .{} };
    defer off.rec.deinit(gpa);
    try off.feed("abc");
    off.rec.closed(gpa, 0);
    abstain.told(&off, &c3, 0, 0, null);
    try t.expectEqual(0, off.rec.text[1].items.len);
    try t.expectEqual(0, off.rec.abstains.items.len);
    try t.expectEqual(1, off.abstained_n);
    try t.expect(std.mem.endsWith(u8, records.write(&off), "\"cells\":[]}"));
}

test "runner abstain records: PREFIX_ABSTAINS is off unless 1, and a session records abstains only with cells on" {
    const K = struct {
        var set: []const [2][]const u8 = &.{};
        fn get(name: []const u8) ?[]const u8 {
            for (set) |kv| if (std.mem.eql(u8, kv[0], name)) return kv[1];
            return null;
        }
    };
    const runner = @import("runner.zig");
    K.set = &.{ .{ "TF_R", "1" }, .{ "TF_R_CIRCUIT", "/c.bin" } };
    try t.expect(!(try runner.Config.parse(&K.get, "TF_R")).?.abstains);
    K.set = &.{ .{ "TF_R", "1" }, .{ "TF_R_CIRCUIT", "/c.bin" }, .{ "TF_R_ABSTAINS", "1" } };
    try t.expect((try runner.Config.parse(&K.get, "TF_R")).?.abstains);
    K.set = &.{ .{ "TF_R", "1" }, .{ "TF_R_CIRCUIT", "/c.bin" }, .{ "TF_R_ABSTAINS", "2" } };
    try t.expectError(error.BadKnob, runner.Config.parse(&K.get, "TF_R"));
    // cells off: no records of either kind, whatever ABSTAINS says
    const c: Case = .{ .lead = "```python\nprint(undefined_total)\n```\nResult:\n", .o = .{ .cells = false }, .reason = "name", .stage = "frontend" };
    const off = try run(c, true);
    defer gpa.free(off.text);
    defer gpa.free(off.receipt);
    try t.expect(std.mem.indexOf(u8, off.receipt, "\"abstains") == null and std.mem.indexOf(u8, off.receipt, "\"cells\"") == null);
}
