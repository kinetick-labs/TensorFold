//! TF_DSV41_RUNNER_STATED (S1 R4 d_mirror rank 1a): whole statements form trace had no shape for, on the live lines.
const std = @import("std");
const api = @import("runner_api.zig");
const graph = @import("runner_graph.zig");
const route = @import("runner_graph_route.zig");
const runner = @import("runner.zig");
const trace = @import("runner_trace.zig");
const gpa = std.testing.allocator;

/// The last line's stated computation as display and run, after every line went through one tracer (null: none).
fn expect(whole: bool, lines: []const []const u8, want: ?[2][]const u8) !void {
    var t = trace.Tracer.init(gpa);
    defer t.deinit();
    t.whole = whole;
    var got: ?trace.Found = null;
    for (lines) |l| got = try t.line(l);
    if (want) |w| {
        const f = got orelse {
            std.debug.print("nothing stated (whole {}): {s}\n", .{ whole, lines[lines.len - 1] });
            return error.TestExpectedStated;
        };
        try std.testing.expectEqualStrings(w[0], f.display);
        try std.testing.expectEqualStrings(w[1], f.run);
    } else if (got) |f| {
        std.debug.print("stated (whole {}): {s} from {s}\n", .{ whole, f.display, lines[lines.len - 1] });
        return error.TestUnexpectedStated;
    }
}

const product = "We need answer exact product. Need compute. Need use Python? We can calculate. Need final JSON only? User says explain briefly if useful, then end with FINAL_JSON. Need ensure exact. Let's compute carefully. 123,456,789,012 * 987,654,321,098.";
const product_b = [_][]const u8{ "Let A = 98,765,432,109", "B = 12,345,678,901", "B = 12,000,000,000 + 345,678,901? Or use pattern.", "A * B = 98,765,432,109 * 12,345,678,901." };
const gcd_a = "We need gcd(123456789012345, 987654321012345).";
const gcd_b = "Let's compute carefully. Need gcd(1122334455667788, 9988776655443322).";
const hex_b = "Hex = 0x123456789ABCDEF012345.";
// S2 R5 (s2-trace-ab5): the replies name the gcd in prose, the literals written after names
const gcd_of_a = "We need gcd of A=123456789012345, B=987654321012345.";
const gcd_of_b = "We need gcd of A=1122334455667788 and B=9988776655443322.";
const gcd_of_c = "Need final with explanation and final JSON. Need ensure number. Let's compute gcd of A=1122334455667788, B=9988776655443322.";
const gcd_of_d = "Need the greatest common divisor of 1,122,334,455,667,788 and 9,988,776,655,443,322 exactly.";

/// The live rank-1a lines with the knob: display, then the source the cell runs (hex as written: the circuit converts).
const live = [_]struct { lines: []const []const u8, want: [2][]const u8 }{
    .{ .lines = &.{product}, .want = .{ "123456789012*987654321098", "123456789012*987654321098" } },
    .{ .lines = &product_b, .want = .{ "98765432109*12345678901", "98765432109*12345678901" } },
    .{ .lines = &.{gcd_a}, .want = .{ "gcd(123456789012345, 987654321012345)", "from math import gcd\nprint(gcd(123456789012345, 987654321012345))" } },
    .{ .lines = &.{gcd_b}, .want = .{ "gcd(1122334455667788, 9988776655443322)", "from math import gcd\nprint(gcd(1122334455667788, 9988776655443322))" } },
    .{ .lines = &.{hex_b}, .want = .{ "0x123456789ABCDEF012345", "0x123456789ABCDEF012345" } },
    .{ .lines = &.{gcd_of_a}, .want = .{ "gcd(123456789012345, 987654321012345)", "from math import gcd\nprint(gcd(123456789012345, 987654321012345))" } },
    .{ .lines = &.{gcd_of_b}, .want = .{ "gcd(1122334455667788, 9988776655443322)", "from math import gcd\nprint(gcd(1122334455667788, 9988776655443322))" } },
    .{ .lines = &.{gcd_of_c}, .want = .{ "gcd(1122334455667788, 9988776655443322)", "from math import gcd\nprint(gcd(1122334455667788, 9988776655443322))" } },
    .{ .lines = &.{gcd_of_d}, .want = .{ "gcd(1122334455667788, 9988776655443322)", "from math import gcd\nprint(gcd(1122334455667788, 9988776655443322))" } },
};

test "runner trace stated: the R4 rank-1a lines are stated with TF_DSV41_RUNNER_STATED and not without it" {
    for (live) |c| {
        try expect(true, c.lines, c.want);
        try expect(false, c.lines, null);
    }
    // already form trace's: the knob changes nothing (live mod-a, mod-b)
    for ([_]bool{ false, true }) |w| {
        try expect(w, &.{"We need answer compute 11^97 mod 1000000007. Need use Python maybe."}, .{ "11^97 mod 1000000007", "print(pow(11, 97, 1000000007))" });
        try expect(w, &.{"Compute 13^127 mod 998244353. Let's calculate with modular arithmetic."}, .{ "13^127 mod 998244353", "print(pow(13, 127, 998244353))" });
    }
}

test "runner trace stated: a restatement that is not literally its names, a list, a short or open hex and gcd of three refuse" {
    const defs = product_b[0..2];
    for ([_][]const u8{
        "A * B = 98,765,432,109 * 12,345,678,902.", // a typo: which one the model means is unknown
        "A + B = 98,765,432,109 * 12,345,678,901.",
        "If A * B = 98,765,432,109 * 12,345,678,901.",
        "A * B = 98,765,432,109 * 12,345,678,901?",
        "A * B == 98,765,432,109 * 12,345,678,901.",
    }) |line| try expect(true, &.{ defs[0], defs[1], line }, null);
    // the brackets of a name's own sum stay: (A + B) * C is not 7 + 3 * 5
    try expect(true, &.{ "Let A = 7", "B = 3", "C = 5", "(A + B) * C = 7 + 3 * 5." }, null);
    try expect(true, &.{"A * B = 98,765,432,109 * 12,345,678,901."}, null); // no definitions
    for ([_][]const u8{
        "Need gcd(12, 18, 30).",                     "Need gcd(12,345, 18).",
        "Digits: 1,234, 5,678, 9,012.",              "1,234,5678 * 2.",
        "Hex = 0x123456789.",                        "Hex = 0x123456789ABCDEF012345?",
        "Hex = 0x123456789ABCDEF012345 + 1 is big.", "Hex = 0x123456789ABCDEF012345G.",
        "Low part 0x123456789ABCDEF012345.",         "Hex =0x123456789ABCDEF012345x.",
    }) |line| try expect(true, &.{line}, null);
}

test "runner trace stated: gcd in prose takes two literals only: names, expressions, a third number or a claim refuse" {
    for ([_][]const u8{
        "Need gcd of A and B.",                                 "Need gcd of A=123456789012 and B.",
        "Need gcd of 123456789012 and A.",                      "Need gcd of two large numbers.",
        "Need gcd of 12, 18 and 30.",                           "Need gcd of 12, 18, 30.",
        "Need gcd of 12 and 18 and 30.",                        "Need gcd of 12,345, 18.",
        "Need gcd of 123456789012 and 18, then lcm.",           "Need gcd of 123456789012 and 18 is 6.",
        "Need gcd of 2^64 and 10^9+7.",                         "Need gcd of 2^64 and 123456789012.",
        "Need gcd of 123456789012 and 2*3.",                    "Need gcd of 0x1F and 123456789012.",
        "Need gcd of 123456789012 and 1.5.",                    "Need gcd of -12 and 123456789012.",
        "Need gcd of 012 and 123456789012.",                    "Need gcd of 123456789012x and 18.",
        "Need lcm of 123456789012 and 18.",                     "Need gcd of 123456789012 and 654321?",
        "Need gcd of about 123456789012 and 18.",               "Need gcd of 123456789012 and 18 maybe.",
        "Need gcd of coefficients 13,717,421 and 109,739,369.", "Need gcd of 123456789012 or 18.",
        "Need greatest common divisor 123456789012 and 18.",
    }) |line| try expect(true, &.{line}, null);
}

pub fn python(code: []const u8) ![]u8 {
    const src = if (std.mem.startsWith(u8, code, "from ")) try gpa.dupe(u8, code) else try std.fmt.allocPrint(gpa, "print({s})", .{code});
    defer gpa.free(src);
    const r = try std.process.run(gpa, std.testing.io, .{ .argv = &.{ "/usr/bin/python3", "-I", "-S", "-c", src }, .stdout_limit = .limited(4096), .stderr_limit = .limited(4096) });
    defer gpa.free(r.stderr);
    errdefer gpa.free(r.stdout);
    try std.testing.expect(r.term.success());
    return r.stdout;
}

pub const Bank = struct {
    bank: graph.Bank,
    backend: api.Backend,
    job: u64 = 0,

    pub fn translate(b: *Bank, a: std.mem.Allocator, code: []const u8) ![]const u8 {
        var routed: route.Routed = .{ .inner = @import("runner_frontend.zig").frontend(), .bank = &b.bank };
        const f = routed.frontend();
        const tr = try f.translate(f.ctx, a, .{ .form = .repl, .lang = "python", .code = code });
        if (tr != .program or tr.program.kind != .graph) {
            std.debug.print("not on the bank: {s}\n", .{code});
            if (tr == .program) a.free(tr.program.bytes);
            return error.TestExpectedGraph;
        }
        return tr.program.bytes;
    }

    pub fn run(b: *Bank, bytes: []const u8) ![]u8 {
        b.job += 1;
        const job = try b.backend.start(b.job, .{ .kind = .graph, .bytes = bytes }, 1 << 32);
        const res = while (true) switch (b.backend.poll(job)) {
            .running => std.Thread.yield() catch {},
            .done => |r| break r,
        };
        defer b.backend.release(b.job);
        try std.testing.expectEqual(api.Status.ok, res.status);
        return gpa.dupe(u8, res.out);
    }
};

pub fn openBank() !?*Bank {
    const path = std.c.getenv("TF_RUNNER_GRAPH_BANK") orelse return null;
    const b = try gpa.create(Bank);
    errdefer gpa.destroy(b);
    b.* = .{ .bank = try graph.Bank.load(gpa, std.mem.span(path)), .backend = undefined };
    b.backend = try graph.backend(gpa, &b.bank, 2);
    return b;
}

pub fn closeBank(b: *Bank) void {
    graph.destroy(b.backend);
    b.bank.deinit();
    gpa.destroy(b);
}

test "runner proof: trace stated: each live statement runs on the bank and prints CPython's value (TF_RUNNER_GRAPH_BANK)" {
    const b = try openBank() orelse return error.SkipZigTest;
    defer closeBank(b);
    for (live) |c| {
        const want = try python(c.want[1]);
        defer gpa.free(want);
        const bytes = try b.translate(gpa, c.want[1]);
        defer gpa.free(bytes);
        const out = try b.run(bytes);
        defer gpa.free(out);
        try std.testing.expectEqualStrings(want, out);
    }
}

test "runner proof: trace stated: the circuit owns gcd and the restated product: a mutated input prints CPython's value (TF_RUNNER_GRAPH_BANK)" {
    const b = try openBank() orelse return error.SkipZigTest;
    defer closeBank(b);
    for ([_][4][]const u8{
        .{ live[2].want[1], "123456789012345", "123456789012340", "from math import gcd\nprint(gcd(123456789012340, 987654321012345))" },
        .{ live[1].want[1], "98765432109", "98765432119", "98765432119*12345678901" },
        .{ live[6].want[1], "9988776655443322", "9988776655443323", "from math import gcd\nprint(gcd(1122334455667788, 9988776655443323))" },
    }) |m| {
        const bytes = try b.translate(gpa, m[0]);
        defer gpa.free(bytes);
        const from = try std.fmt.allocPrint(gpa, "\"{s}\"", .{m[1]});
        defer gpa.free(from);
        const to = try std.fmt.allocPrint(gpa, "\"{s}\"", .{m[2]});
        defer gpa.free(to);
        if (std.mem.count(u8, bytes, from) != 1) {
            std.debug.print("input {s} not one packet value: {s}\n", .{ m[1], bytes });
            return error.TestExpectedInput;
        }
        const mutated = try std.mem.replaceOwned(u8, gpa, bytes, from, to);
        defer gpa.free(mutated);
        const want = try python(m[3]);
        defer gpa.free(want);
        const out = try b.run(mutated);
        defer gpa.free(out);
        if (!std.mem.eql(u8, want, out)) std.debug.print("host-owned: {s} -> {s}: '{s}', CPython '{s}'\n", .{ m[1], m[2], out, want });
        try std.testing.expectEqualStrings(want, out);
    }
}

test "runner trace stated: TF_DSV41_RUNNER_STATED parses, off unless set, and only with form trace" {
    const K = struct {
        var v: ?[]const u8 = null;
        var forms: ?[]const u8 = null;
        fn get(name: []const u8) ?[]const u8 {
            if (std.mem.eql(u8, name, "TF_R")) return "1";
            if (std.mem.eql(u8, name, "TF_R_CIRCUIT")) return "/c.bin";
            if (std.mem.eql(u8, name, "TF_R_FORMS")) return forms;
            return if (std.mem.eql(u8, name, "TF_R_STATED")) v else null;
        }
    };
    try std.testing.expect(!(try runner.Config.parse(&K.get, "TF_R")).?.scan.trace_whole);
    K.v = "1";
    try std.testing.expectError(error.BadKnob, runner.Config.parse(&K.get, "TF_R"));
    K.forms = "fence,repl,inline,trace";
    try std.testing.expect((try runner.Config.parse(&K.get, "TF_R")).?.scan.trace_whole);
    K.v = "2";
    try std.testing.expectError(error.BadKnob, runner.Config.parse(&K.get, "TF_R"));
}
