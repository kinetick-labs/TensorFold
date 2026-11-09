//! Form trace's opt-in check (runner_trace_check): residues after the owned line, refusals, and the line unchanged off.
const std = @import("std");
const trace = @import("runner_trace.zig");

const gpa = std.testing.allocator;
const default = "[exact, computed by the engine: {}]";

/// The owned line with the check on must be the line with it off, then `check` (S2 round 7's Python reference).
fn expectCheck(display: []const u8, divmod: bool, out: []const u8, check: []const u8) !void {
    const off = (try trace.owned(gpa, default, display, divmod, out, false)).?;
    defer gpa.free(off);
    const on = (try trace.owned(gpa, default, display, divmod, out, true)).?;
    defer gpa.free(on);
    const want = try std.mem.concat(gpa, u8, &.{ off[0 .. off.len - 1], check, "\n" });
    defer gpa.free(want);
    try std.testing.expectEqualStrings(want, on);
}

/// No check for this line: on and off are the same bytes.
fn expectNone(display: []const u8, divmod: bool, out: []const u8) !void {
    const off = (try trace.owned(gpa, default, display, divmod, out, false)).?;
    defer gpa.free(off);
    const on = (try trace.owned(gpa, default, display, divmod, out, true)).?;
    defer gpa.free(on);
    try std.testing.expectEqualStrings(off, on);
}

test "runner trace check: products, sums, powers and brackets show each literal's residue and the value's" {
    try expectCheck("123456789012*987654321098", false, "121932631136585886175176\n", " check: 3*8 ≡ 6 (mod 9), 012*098 ≡ 176 (mod 1000)");
    try expectCheck("320255973501901*65536 + 61185", false, "20988295479420645121\n", " check: 7*7 + 3 ≡ 7 (mod 9), 901*536 + 185 ≡ 121 (mod 1000)");
    try expectCheck("12345678901^2", false, "152415787526596567801\n", " check: 1^2 ≡ 1 (mod 9), 901^2 ≡ 801 (mod 1000)");
    try expectCheck("61185*(4000000000 + 294967296)", false, "262787574005760\n", " check: 3*(4 + 0) ≡ 3 (mod 9), 185*(000 + 296) ≡ 760 (mod 1000)");
    try expectCheck("987654 * 137", false, "135308598\n", " check: 3 * 2 ≡ 6 (mod 9), 654 * 137 ≡ 598 (mod 1000)");
    try expectCheck("(98765 - 4321)*(1000003)", false, "94444283332\n", " check: (8 - 1)*(4) ≡ 1 (mod 9), (765 - 321)*(003) ≡ 332 (mod 1000)");
    // exponents stay as written
    try expectCheck("7*11^3 + 2", false, "9319\n", " check: 7*2^3 + 2 ≡ 4 (mod 9), 7*11^3 + 2 ≡ 319 (mod 1000)");
}

test "runner trace check: divmod shows quotient times divisor plus remainder against the dividend" {
    try expectCheck("314159265358979323846264338327950288419716939937510 ÷ 271828182845904523536", true, "(1155727349790921717910215407028, 23663814570294126502)\n", " check: 0*3 + 4 ≡ 4 (mod 9), 028*536 + 502 ≡ 510 (mod 1000)");
    try expectCheck("(123456789 + 1) ÷ 97", true, "(1272750, 40)\n", " check: 6*7 + 4 ≡ 1 (mod 9), 750*97 + 40 ≡ 790 (mod 1000)");
}

test "runner trace check: hex, mod, named problems, long or wide expressions and negatives take no check" {
    try expectNone("0xFEDCBA9876543210*2^64 + 0x0123456789ABCDEF", false, "1\n");
    try expectNone("7^123 mod 1000000007", false, "937329259\n");
    try expectNone("gcd(12, 18)", false, "6\n");
    try expectNone("isqrt(99)", false, "9\n");
    try expectNone("edit_distance(\"ab\", \"a\")", false, "1\n");
    try expectNone("1*2*3*4*5", false, "120\n"); // five literals
    try expectNone("12^5", false, "248832\n"); // a power past 4
    try expectNone("2^10", false, "1024\n");
    try expectNone("5 - 100", false, "-95\n"); // a negative value
    try expectNone("-5*3", false, "-15\n"); // unary minus
    try expectNone("(10^30 - 1) ÷ 97", true, "(10309278350515463917525773195, 84)\n");
}

test "runner trace check: a value its own residues contradict is not owned with the check, and unchanged without it" {
    const wrong = "121932631136585886175177\n";
    try std.testing.expectEqual(null, try trace.owned(gpa, default, "123456789012*987654321098", false, wrong, true));
    const off = (try trace.owned(gpa, default, "123456789012*987654321098", false, wrong, false)).?;
    defer gpa.free(off);
    try std.testing.expectEqualStrings("[exact, computed by the engine: 123456789012*987654321098 = 121932631136585886175177]\n", off);
    try std.testing.expectEqual(null, try trace.owned(gpa, default, "(123456789 + 1) ÷ 97", true, "(1272750, 41)\n", true));
}

test "runner trace check: any wording's closing text comes before the check" {
    const line = (try trace.owned(gpa, "{} (exact)", "987654 * 137", false, "135308598\n", true)).?;
    defer gpa.free(line);
    try std.testing.expectEqualStrings("987654 * 137 = 135308598 (exact) check: 3 * 2 ≡ 6 (mod 9), 654 * 137 ≡ 598 (mod 1000)\n", line);
}
