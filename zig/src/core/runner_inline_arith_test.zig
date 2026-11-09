//! Inline arithmetic with thousands commas, brackets and a whole `gcd(a, b)`: reply lines, cells, CPython values.
const std = @import("std");
const graph = @import("runner_graph.zig");
const route = @import("runner_graph_route.zig");
const frontend = @import("runner_frontend.zig");
const scan = @import("runner_scan.zig");
const t = std.testing;

/// The line up to the model's `=`, the cell code, and CPython's value of it (`^` read as `**`).
const Golden = struct { []const u8, []const u8, []const u8 };

/// Live lines (earlier live smoke and window replies, where the model then wrote this value), then variations.
pub const goldens = [_]Golden{
    .{ "next: 288,419,716 - 624,605,146 need borrow from previous limb (338,327,950). Borrow 1e9: (1,288,419,716 -624,605,146)=", "(1288419716 -624605146)", "663814570" },
    .{ "64,800^2 =", "64800^2", "4199040000" },
    .{ "285,000,000^2 =", "285000000^2", "81225000000000000" },
    .{ "We had 136,418,000^2 =", "136418000^2", "18609870724000000" },
    .{ "12,000,000^2=", "12000000^2", "144000000000000" },
    .{ "Compute 5,764,801^2 =", "5764801^2", "33232930569601" },
    .{ "We computed 234,567,890*12,345,679 =", "234567890*12345679", "2895899873647310" },
    .{ "876,543,210*234 =", "876543210*234", "205111111140" },
    .{ "2*69,663,000*30 =", "2*69663000*30", "4179780000" },
    .{ "18,364,758,544,493,064,720 *3 =", "18364758544493064720 *3", "55094275633479194160" },
    .{ "Compute 70,272,836,576 * 200,000,000 =", "70272836576 * 200000000", "14054567315200000000" },
    .{ "Add to *12,000,000: 2,814,814,680,000,000 + 81,085,193,647,310 =", "2814814680000000 + 81085193647310", "2895899873647310" },
    .{ "M^2: 789,971,041 + 1,950,922,114 =", "789971041 + 1950922114", "2740893155" },
    .{ "M^3: 975,461,057 + 2 =", "975461057 + 2", "975461059" },
    .{ "Add low: 507,835,392 + 985,229,328 =", "507835392 + 985229328", "1493064720" },
    .{ "We need 3^400 mod 10^18. Since gcd(3,10)=", "gcd(3,10)", "1" },
    .{ "Output: gcd(462, 1071) =", "gcd(462, 1071)", "21" },
    // bn-011's bracketed squares, other operators, signs, and gcd edges
    .{ "(147,000+73)^2=", "(147000+73)^2", "21630467329" },
    .{ "Square: (179,000+457)^2 = ", "(179000+457)^2", "32204814849" },
    .{ "2*(1,000+24) =", "2*(1000+24)", "2048" },
    .{ "(1,000 − 7) × 3 =", "(1000 − 7) × 3", "2979" },
    .{ "-(2,000+5) =", "-(2000+5)", "-2005" },
    .{ "49,659 − 29,876 =", "49659 − 29876", "19783" },
    .{ "1,024 × 1,024 =", "1024 × 1024", "1048576" },
    .{ "1,234 + 5 =", "1234 + 5", "1239" },
    .{ "Now subtract 3^67 =", "3^67", "92709463147897837085761925410587" },
    .{ "(10,000,000,000,000,000,000+123,456,789)^2-10,000,000,000,000,000,000^2 =", "(10000000000000000000+123456789)^2-10000000000000000000^2", "2469135780015241578750190521" },
    .{ "So gcd(0,0) =", "gcd(0,0)", "0" },
    .{ "gcd(1234567890, 9876543210) =", "gcd(1234567890, 9876543210)", "90" },
    .{ "Since: gcd(0, 24) =", "gcd(0, 24)", "24" },
};

/// Lines no cell may open; the model keeps writing.
const refused = [_][]const u8{
    // prose, decimal and list commas
    "3, 4 and 5 =",
    "1, 234 + 5 =",
    "3,5 * 2 =",
    "12,34 + 5 =",
    "1,23 + 5 =",
    "0,500 * 2 =",
    "1.234,567 * 2 =",
    "1,2345 + 1 =",
    "1,234,5678 + 1 =",
    ",000+5 =",
    "1,000, +5 =",
    "1673, +10 =",
    // floats with commas: CPython's value would be a float
    "44,253,970,636,272 / 16 =",
    "1,234.5 * 2 =",
    "1,000 ÷ 8 =",
    // tuple, list, set or call context
    "(1,000, 2,000) =",
    "[1,000 + 2 =",
    "{1,000*2 =",
    "f(1,000+2) =",
    "print((1,000+2) =",
    "point (3,141 + 1 =",
    "max(1,000, 2) =",
    "x = (1,000+2) =",
    // calls that are not a whole literal gcd
    "lcm(4, 6) =",
    "f(3) + 1 =",
    "φ(2^18)=",
    "Since φ(5^18)=",
    "√(16) =",
    "gcd(4, 6, 8) =",
    "gcd(-4, 6) =",
    "gcd(04, 6) =",
    "gcd(4,6)^2 =",
    "2 * gcd(4, 6) =",
    "math.gcd(4, 6) =",
    "xgcd(4, 6) =",
    "\\gcd(4, 6) =",
    "gcd(1,000, 5) =",
    "gcd(4 , 6) =",
    "gcd(a, 6) =",
    "gcd(4,  6) =",
    "x mod gcd(4, 6) =",
    "Add gcd(4, 6) =",
    "gcd(4, 6) = 2",
    // unbalanced brackets
    "(147,000+73 =",
    "147,000+73) =",
    "((1,000+2) =",
    "1,000+2)) =",
    // the right operand of a TeX or worded operator (SparkDash e89f8d1 reason-09 and arith-27 on the compute default)
    "- Jamal: \\(4 \\times 49 - 11 =",
    "1. \\(221 \\bmod 4331 + 1000 =",
    "9. \\(15314 \\bmod 8563 + 1000 =",
    "Target floor: \\(4 \\times 3 + 6 =",
    "\\(68 \\div 2 + 10 =",
    "\\(0.5^2 \\times 0.06^2 =",
    "Surface area of cube with side 2: \\(6 \\cdot 2^2 =",
    "x \\cdot 3 + 4 =",
    "4 times 49 - 11 =",
    // a running total the model holds
    "Add 11,529,600 +1 =",
    "Minus 2,386,000 +25 =",
    "subtract: 1,000 - 5 =",
    "Plus 3 + 4 =",
};

/// The cell a line closes at its `=` with inline math on, or null.
fn cellOf(a: std.mem.Allocator, line: []const u8, cfg: scan.Config) !?[]u8 {
    var s: scan.Scanner = .{ .cfg = cfg };
    defer s.deinit(a);
    const e = try s.feed(a, line, true);
    const c = e.closed orelse return null;
    if (c.form != .inline_expr or e.trigger == null or e.trigger.?.inline_space == null) return null;
    return try a.dupe(u8, c.code);
}

test "runner scan: inline arithmetic with commas, brackets and gcd opens cells with Python-readable code" {
    for (goldens) |g| {
        const got = try cellOf(t.allocator, g[0], .{ .inline_math = true }) orelse {
            std.debug.print("no cell: {s}\n", .{g[0]});
            return error.TestExpectedInline;
        };
        defer t.allocator.free(got);
        try t.expectEqualStrings(g[1], got);
    }
    for (refused) |line| if (try cellOf(t.allocator, line, .{ .inline_math = true })) |got| {
        defer t.allocator.free(got);
        std.debug.print("opened, must not: {s} -> {s}\n", .{ line, got });
        return error.TestUnexpectedInline;
    };
}

test "runner scan: inline arithmetic is off with its knob; a plain expression is not" {
    const a = t.allocator;
    const off: scan.Config = .{ .inline_math = true, .inline_arith = false };
    for ([_][]const u8{ "(147,000+73)^2=", "Since gcd(3,10)=", "1,234 + 5 =" }) |line| {
        try t.expectEqual(@as(?[]u8, null), try cellOf(a, line, off));
    }
    const plain = (try cellOf(a, "(147000+73)^2=", off)).?;
    defer a.free(plain);
    try t.expectEqualStrings("(147000+73)^2", plain);
    try t.expectEqual(@as(?[]u8, null), try cellOf(a, "(147,000+73)^2=", .{}));
}

fn bankPath() ![]const u8 {
    const p = std.c.getenv("TF_RUNNER_GRAPH_BANK") orelse return error.SkipZigTest;
    return std.mem.span(p);
}

test "runner proof: inline arithmetic lines scan to cells that run on the graph bank and print CPython's value (TF_RUNNER_GRAPH_BANK)" {
    var bank = try graph.Bank.load(t.allocator, try bankPath());
    defer bank.deinit();
    var context: frontend.Context = .{ .io = t.io, .compiler = .{ .clang = "/nonexistent-inline-arith-clang" } };
    var routed: route.Routed = .{ .inner = context.frontend(), .bank = &bank };
    const f = routed.frontend();
    for (goldens) |g| {
        // end to end: the scanner's own cell for the reply line, not the golden's expected code
        const code = try cellOf(t.allocator, g[0], .{ .inline_math = true }) orelse return error.TestExpectedInline;
        defer t.allocator.free(code);
        const result = try f.translate(f.ctx, t.allocator, .{ .form = .inline_expr, .lang = "python", .code = code });
        defer if (result == .program) t.allocator.free(result.program.bytes);
        if (result != .program or result.program.kind != .graph) {
            std.debug.print("not on the graph: {s}: {any}\n", .{ code, result });
            return error.TestExpectedGraph;
        }
        var machine = try @import("runner_graph_machine.zig").Machine.boot(t.allocator, &bank, result.program.bytes);
        defer machine.deinit();
        var out: std.ArrayList(u8) = .empty;
        defer out.deinit(t.allocator);
        const end = try machine.run(4_000_000, &out, null);
        try t.expect(end.end == .halted);
        std.testing.expectEqualStrings(g[2], std.mem.trimEnd(u8, out.items, "\n")) catch |e| {
            std.debug.print("inline arith value: {s}\n", .{code});
            return e;
        };
    }
}
