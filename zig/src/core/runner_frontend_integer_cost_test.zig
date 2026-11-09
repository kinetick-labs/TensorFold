const std = @import("std");
const ast = @import("runner_frontend_integer_ast.zig");
const cost = @import("runner_frontend_integer_cost.zig");
const t = std.testing;
const cap: usize = 10_522;
const unlimited = std.math.maxInt(usize);

fn measured(code: []const u8, budget: usize) !?usize {
    var parser = ast.Parser{ .source = code };
    const root = try parser.parse();
    return cost.estimate(&parser, root, budget);
}

test "integer cost exact model counters and inclusive budget boundary" {
    const cases = [_][]const u8{
        "1",            "-1",   "1+2",                  "2*3",           "gcd(54,24)", "gcd(0,0)", "isqrt(0)",
        "factorial(0)", "0**0", "sum(map(int,str(0)))", "314159*271828",
    };
    for (cases) |code| {
        const full = (try measured(code, unlimited)).?;
        try t.expect(full > 0);
        try t.expectEqual(@as(?usize, full), try measured(code, full));
        try t.expectEqual(@as(?usize, null), try measured(code, full - 1));
        try t.expectEqual(@as(?usize, null), try measured(code, 0));
    }
}

test "integer cost admission is monotone in budget and does not mutate proof state" {
    var parser = ast.Parser{ .source = "pow(-2,19,-97)+isqrt(144)" };
    const root = try parser.parse();
    const before = parser;
    const full = cost.estimate(&parser, root, unlimited).?;
    for ([_]usize{ 0, 200, full / 2, full - 1, full, full + 1, unlimited }) |budget| {
        const result = cost.estimate(&parser, root, budget);
        if (budget < full) try t.expect(result == null) else try t.expectEqual(@as(?usize, full), result);
    }
    try t.expectEqual(before.count, parser.count);
    try t.expectEqualSlices(bool, &before.proved, &parser.proved);
    try t.expectEqualSlices(u8, before.source, parser.source);
    for (before.nodes[0..before.count], parser.nodes[0..parser.count]) |a, b| {
        try t.expectEqual(a.op, b.op);
        try t.expectEqual(a.l, b.l);
        try t.expectEqual(a.r, b.r);
        try t.expectEqual(a.k, b.k);
        try t.expectEqual(a.value, b.value);
        try t.expectEqualSlices(u8, a.text, b.text);
    }
}

test "integer cost charges shared DAG nodes once and ignores unreachable nodes" {
    var parser = ast.Parser{ .source = "7" };
    try parser.bind("x", try parser.parse());
    parser.source = "x+x";
    parser.pos = 0;
    const root = try parser.parse();
    const shared = cost.estimate(&parser, root, unlimited).?;
    const separate = (try measured("7+7", unlimited)).?;
    try t.expect(shared < separate);
    parser.nodes[parser.count] = .{ .op = .factorial, .value = 0 };
    parser.count += 1;
    try t.expectEqual(@as(?usize, shared), cost.estimate(&parser, root, unlimited));
    try t.expect((try measured("2**1000%1000", unlimited)) != null);
}

test "integer cost covers radix widths output formatting and signed divmod" {
    const decimal = (try measured("16", unlimited)).?;
    const hex = (try measured("0x10", unlimited)).?;
    try t.expect(hex > decimal);
    try t.expect((try measured("0b10000", unlimited)).? > decimal);
    try t.expect((try measured("1000000000", unlimited)).? > (try measured("1", unlimited)).?);
    const scalar = (try measured("4294967296//4294967295", unlimited)).?;
    const wide = (try measured("4294967296//4294967296", unlimited)).?;
    try t.expect(wide > 0 and scalar > 0);
    try t.expect((try measured("4294967296//4294967296", cap)) != null);
    try t.expect((try measured("divmod(-7,3)", unlimited)).? > (try measured("-7//3", unlimited)).?);
    try t.expect((try measured("len(str(-12345))", unlimited)) != null);
    try t.expect((try measured("sum(int(d) for d in str(12345))", unlimited)) != null);
}

test "integer cost follows bounded algorithm paths and declines costly wide work" {
    for ([_][]const u8{
        "314159*271828",     "gcd(-54,24)",           "isqrt(144)",        "factorial(5)",
        "prod(range(0,10))", "prod(range(0,-10,-1))", "prod(range(10,0))", "sum(int(d) for d in str(12345))",
    }) |code| try t.expect((try measured(code, cap)) != null);
    for ([_][]const u8{
        "factorial(48)", "isqrt(10**60)", "gcd(10**61-1,10**30-1)", "prod(range(1,21))",
    }) |code| try t.expectEqual(@as(?usize, null), try measured(code, cap));
    try t.expect((try measured("factorial(10)", unlimited)).? > (try measured("factorial(5)", unlimited)).?);
    try t.expect((try measured("gcd(89,55)", unlimited)).? > (try measured("gcd(99,33)", unlimited)).?);
    try t.expect((try measured("isqrt(144)", unlimited)).? > (try measured("isqrt(0)", unlimited)).?);
    try t.expectEqual(try measured("pow(2,100,1)", unlimited), try measured("pow(2,999,1)", unlimited));
    try t.expect((try measured("pow(2,15,97)", unlimited)).? > (try measured("pow(2,7,97)", unlimited)).?);
    try t.expect((try measured("pow(10**40+1,2,10**61)", unlimited)) != null);
}

test "integer cost zero absorption preserves the full range length precheck" {
    const shorter = (try measured("prod(range(0,1000))", unlimited)).?;
    const longer = (try measured("prod(range(0,9999))", unlimited)).?;
    try t.expectEqual(shorter, longer);
    try t.expect((try measured("prod(range(0,-9999,-1))", cap)) != null);
    var parser = ast.Parser{ .source = "prod(range(0,10000))" };
    const root = try parser.parse();
    const stop = parser.nodes[root].r;
    parser.nodes[stop].value = 10001;
    parser.nodes[stop].text = "10001";
    try t.expectEqual(@as(?usize, null), cost.estimate(&parser, root, unlimited));
}

test "integer cost refuses unproved invalid and inconsistent graph state" {
    var parser = ast.Parser{ .source = "isqrt(144)" };
    const root = try parser.parse();
    const original = parser;
    parser.proved[root] = false;
    try t.expectEqual(@as(?usize, null), cost.estimate(&parser, root, unlimited));
    parser = original;
    parser.nodes[root].l = root;
    try t.expectEqual(@as(?usize, null), cost.estimate(&parser, root, unlimited));
    parser = original;
    parser.nodes[root].l = 64;
    try t.expectEqual(@as(?usize, null), cost.estimate(&parser, root, unlimited));
    parser = original;
    parser.nodes[root].value += 1;
    try t.expectEqual(@as(?usize, null), cost.estimate(&parser, root, unlimited));
    parser = original;
    parser.nodes[root].r = 1000;
    parser.nodes[root].k = 1000;
    try t.expectEqual(cost.estimate(&original, root, unlimited), cost.estimate(&parser, root, unlimited));
    parser = original;
    parser.count = 65;
    try t.expectEqual(@as(?usize, null), cost.estimate(&parser, root, unlimited));
    parser.count = 0;
    try t.expectEqual(@as(?usize, null), cost.estimate(&parser, 0, unlimited));
    try t.expectEqual(@as(?usize, null), cost.estimate(&original, original.count, unlimited));
}

test "integer cost handles the final mask bit and rejects malformed ternary and tuple children" {
    var last = ast.Parser{ .source = "", .count = 64 };
    last.nodes[63] = .{ .op = .literal, .value = 1, .text = "1" };
    try t.expectEqual(try measured("1", unlimited), cost.estimate(&last, 63, unlimited));
    var ternary = ast.Parser{ .source = "pow(2,3,5)" };
    const power = try ternary.parse();
    ternary.nodes[power].k = ternary.count;
    try t.expectEqual(@as(?usize, null), cost.estimate(&ternary, power, unlimited));
    var tuple = ast.Parser{ .source = "7//3+1" };
    const root = try tuple.parse();
    const division = tuple.nodes[root].l;
    tuple.nodes[division].op = .divmod;
    try t.expectEqual(@as(?usize, null), cost.estimate(&tuple, root, unlimited));
}

test "integer cost refuses forged arithmetic failures and bad literal encodings" {
    const cases = [_]struct { code: []const u8, slot: enum { l, r, k }, value: i512, text: []const u8 }{
        .{ .code = "factorial(0)", .slot = .l, .value = -1, .text = "1" },
        .{ .code = "isqrt(1)", .slot = .l, .value = -1, .text = "1" },
        .{ .code = "7//3", .slot = .r, .value = 0, .text = "0" },
        .{ .code = "pow(2,1,3)", .slot = .r, .value = -1, .text = "1" },
        .{ .code = "pow(2,1,3)", .slot = .k, .value = 0, .text = "0" },
        .{ .code = "prod(range(1,5))", .slot = .k, .value = 0, .text = "0" },
        .{ .code = "sum(map(int,str(1)))", .slot = .l, .value = -1, .text = "1" },
    };
    for (cases) |case| {
        var parser = ast.Parser{ .source = case.code };
        const root = try parser.parse();
        const n = parser.nodes[root];
        const index = switch (case.slot) {
            .l => n.l,
            .r => n.r,
            .k => n.k,
        };
        parser.nodes[index].value = case.value;
        parser.nodes[index].text = case.text;
        try t.expectEqual(@as(?usize, null), cost.estimate(&parser, root, unlimited));
    }
    var literal = ast.Parser{ .source = "1" };
    const root = try literal.parse();
    literal.nodes[root].base = 3;
    try t.expectEqual(@as(?usize, null), cost.estimate(&literal, root, unlimited));
    literal.nodes[root].base = 10;
    literal.nodes[root].text = "2";
    try t.expectEqual(@as(?usize, null), cost.estimate(&literal, root, unlimited));
    literal.nodes[root].text = "g";
    try t.expectEqual(@as(?usize, null), cost.estimate(&literal, root, unlimited));
    literal.nodes[root].text = "";
    try t.expectEqual(@as(?usize, null), cost.estimate(&literal, root, unlimited));
    literal.nodes[root].text = "1";
    literal.nodes[root].value = ast.limit;
    try t.expectEqual(@as(?usize, null), cost.estimate(&literal, root, unlimited));
}

test "integer lowering preserves source APIs and applies optional cost admission before emission" {
    const integer = @import("runner_frontend_integer.zig");
    const python = @import("runner_frontend_integer_python.zig");
    const Lower = *const fn (std.mem.Allocator, []const u8, usize) std.mem.Allocator.Error!?integer.Lowered;
    for ([_]Lower{ integer.lowerWithCost, python.lowerWithCost }) |lowerWithCost| {
        const admitted = (try lowerWithCost(t.allocator, "314159*271828", cap)).?;
        defer t.allocator.free(admitted.source);
        try t.expectEqual(try measured("314159*271828", unlimited), admitted.estimated_steps);
        const bypass = (try lowerWithCost(t.allocator, "314159*271828", 0)).?;
        defer t.allocator.free(bypass.source);
        try t.expectEqual(@as(?usize, null), bypass.estimated_steps);
        try t.expectEqualSlices(u8, admitted.source, bypass.source);
        try t.expect((try lowerWithCost(t.allocator, "314159*271828", admitted.estimated_steps.? - 1)) == null);
    }
    const plain = (try integer.lower(t.allocator, "factorial(48)")).?;
    defer t.allocator.free(plain);
    const statement = "from math import factorial\nprint(factorial(48))";
    const source = (try python.lower(t.allocator, statement)).?;
    defer t.allocator.free(source);
    try t.expectEqualSlices(u8, plain, source);
    try t.expect((try python.lowerWithCost(t.allocator, statement, cap)) == null);
    const bypass = (try python.lowerWithCost(t.allocator, statement, 0)).?;
    defer t.allocator.free(bypass.source);
    try t.expectEqual(@as(?usize, null), bypass.estimated_steps);
    const wide = (try python.lowerWithCost(t.allocator, "a=314159\nb=271828\nprint(a*b)", cap)).?;
    defer t.allocator.free(wide.source);
    try t.expectEqual(try measured("314159*271828", unlimited), wide.estimated_steps);
}
