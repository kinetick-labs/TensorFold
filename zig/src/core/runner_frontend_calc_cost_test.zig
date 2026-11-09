const std = @import("std");
const value = @import("runner_frontend_value.zig");
const t = std.testing;
const cap: usize = 10_522;
const unlimited = std.math.maxInt(usize);
const multiply30 = "123456789012345678901234567890*987654321098765432109876543210";

test "calc cost admits the exact boundary and rejects smaller budgets" {
    for ([_][]const u8{
        "1", "-1", "1+2", "2*3", "gcd(54,24)", "0**0", "2**1", "2**2", "2**3", multiply30,
    }) |source| {
        try t.expect(value.valid(source));
        const full = value.estimate(source, unlimited) orelse return error.UnexpectedCostRefusal;
        try t.expect(full > 0);
        try t.expectEqual(@as(?usize, full), value.estimate(source, full));
        try t.expectEqual(@as(?usize, full), value.estimate(source, full + 1));
        try t.expectEqual(@as(?usize, null), value.estimate(source, full - 1));
        try t.expectEqual(@as(?usize, null), value.estimate(source, 0));
    }
    try t.expect(value.estimate("2**3", unlimited).? > value.estimate("2**2", unlimited).?);
    try t.expect(value.estimate("2**2", unlimited).? > value.estimate("2**1", unlimited).?);
}

test "calc cost leaves headroom for the known power and multiply firmware cases" {
    for ([_][]const u8{ "2**100", "2**100-3**50", multiply30 }) |source| {
        const steps = value.estimate(source, cap) orelse return error.UnexpectedCostRefusal;
        try t.expect(steps <= cap);
    }
    try t.expect(value.estimate("2**100-3**50", unlimited).? > value.estimate("2**100", unlimited).?);
}

test "calc cost is monotone in budget and rejects composite expensive work" {
    const twice = multiply30 ++ "+" ++ multiply30;
    const thrice = twice ++ "+" ++ multiply30;
    try t.expect(value.valid(thrice));
    try t.expect(value.estimate(twice, cap) != null);
    try t.expect(value.estimate(thrice, cap) == null);
    const full = value.estimate(thrice, unlimited).?;
    for ([_]usize{ 0, 200, 1000, cap, full - 1, full, full + 1, unlimited }) |budget| {
        const result = value.estimate(thrice, budget);
        if (budget < full) try t.expect(result == null) else try t.expectEqual(@as(?usize, full), result);
    }
}

test "calc cost follows fractional reduction decimal alignment and Euclidean paths" {
    for ([_][]const u8{
        "1/3+2/3", "11/18", "1.25*2.4", "1.2+3", "7//-3", "-7%3", "gcd(0,0)", "gcd(-54,24)",
    }) |source| {
        try t.expect(value.valid(source));
        try t.expect(value.estimate(source, unlimited) != null);
    }
    try t.expect(value.estimate("1/2", unlimited).? > value.estimate("1/1", unlimited).?);
    try t.expect(value.estimate("gcd(89,55)", unlimited).? > value.estimate("gcd(99,33)", unlimited).?);
    try t.expect(value.estimate("1.2+3", unlimited).? > value.estimate("12+3", unlimited).?);
    try t.expect(value.estimate("1.2*3.4", unlimited).? > value.estimate("12*34", unlimited).?);
    const nines: [62]u8 = @splat('9');
    const ones: [61]u8 = @splat('1');
    const wide_gcd = "gcd(" ++ nines ++ "," ++ ones ++ ")";
    try t.expect(value.valid(wide_gcd));
    try t.expect(value.estimate(wide_gcd, cap) == null);
}

test "calc cost rejects invalid canonical syntax overflow and arithmetic errors" {
    const over_digits: [63]u8 = @splat('9');
    const over_source: [256]u8 = @splat('1');
    for ([_][]const u8{
        "",        ".",       " ",          "1 2",        "1+",   "2(3)", "(1",         "1)",       "unknown(1)", "gcd(1)",
        "10**100", "2**1000", "2**-1",      "1/0",        "1//0", "1%0",  "gcd(1/2,3)", "(1/2)**3", "1e10",       "1.2/3",
        "1.2%3",   "1.2+1/3", &over_digits, &over_source,
    }) |source| try t.expectEqual(@as(?usize, null), value.estimate(source, unlimited));
}
