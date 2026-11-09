const std = @import("std");
const python = @import("runner_frontend_integer_python.zig");

test "configured multiple wide prints execute immutable snapshots and preserve exact lines" {
    const t = std.testing;
    var env = try t.environ.createMap(t.allocator);
    defer env.deinit();
    const clang = env.get("TF_RUNNER_CLANG") orelse return error.SkipZigTest;
    const root = env.get("TF_RUNNER_QWENPUTER") orelse return error.SkipZigTest;
    const cases = [_][2][]const u8{
        .{ "x=7\nprint(x)\nx=11\nprint(x)", "7\n11\n" },
        .{ "print(divmod(-7,3))\nprint(5*9)", "(-3, 2)\n45\n" },
        .{
            "a=11232463670103386525\nb=732667528837627997\nc=242246186388029\nprint(a-b*129+c)\nprint(b*129)",
            "-83281405303764237059\n94514111220054011613\n",
        },
    };
    for (cases) |case| {
        const bytes = (try python.compile(t.allocator, t.io, case[0], .{
            .clang = clang,
            .linker = env.get("TF_RUNNER_LLD"),
            .step_budget = 0,
        })) orelse return error.UnexpectedAbstention;
        defer t.allocator.free(bytes);
        try @import("runner_frontend_reference_test.zig").check(root, env.get("TF_RUNNER_REFERENCE_PYTHON") orelse "/usr/bin/python3", .{ .kind = .elf, .bytes = bytes }, case[1]);
    }
}

test "assignment scripts preserve original wide operands in calc input" {
    const cases = [_][2][]const u8{
        .{
            "a=140857552301908510213574088100\nb=690019271788334891377792747413\nprint(a+b)",
            "(140857552301908510213574088100+690019271788334891377792747413)",
        },
        .{
            "a=831680992073052982499463120866\nb=548277608446792630955630629392\nprint(a*b)",
            "(831680992073052982499463120866*548277608446792630955630629392)",
        },
        .{ "a=7\nb=a\na=a+2\nprint(a*b)", "((7+2)*7)" },
        .{ "a=-7\nb=3\nprint(a//b)", "((-7)//3)" },
        .{ "a=7\nb=-3\nprint(a%b)", "(7%(-3))" },
        .{ "a=-7\nb=-3\nprint(a//b)", "((-7)//(-3))" },
        .{ "print(-2**2)", "(-(2**2))" },
        .{ "print(2**114 - 3**67)", "((2**114)-(3**67))" },
        .{ "import math\na=48\nprint(math.gcd(a,18))", "gcd(48,18)" },
        .{ "from math import gcd\nprint(gcd(48,18))", "gcd(48,18)" },
        .{ "unused=2*3\na=11 # original operand\nprint(a+7)\n", "(11+7)" },
    };
    for (cases) |case| {
        const result = (try python.lowerCalc(std.testing.allocator, case[0])).?;
        defer std.testing.allocator.free(result);
        try std.testing.expectEqualStrings(case[1], result);
    }
}

test "calc script validation refuses unused failures and non integer semantics" {
    const cases = [_][]const u8{
        "x=1//0\nprint(3)",
        "x=10**62\nprint(3)",
        "x=3\nx=1%0\nprint(3)",
        "x=math.gcd(48,18)\nprint(3)",
        "x=gcd(48,18)\nprint(3)",
        "from math import *\nprint(3)",
        "import math as m\nprint(3)",
        "import os\nprint(3)",
        "print=2\nprint(3)",
        "int=2\nprint(3)",
        "gcd=2\nprint(3)",
        "x=input()\nprint(3)",
        "x=1.25\nprint(x)",
        "x=1/2\nprint(x)",
        "from fractions import Fraction\nprint(Fraction(1,2))",
        "from decimal import Decimal\nprint(Decimal('2'))",
        "x=int('1.2')\nprint(x)",
        "x=int('12',3)\nprint(x)",
        "print(3)\nx=1//0",
        "x=2\nprint(x)\nprint(x)",
        "x=2\nx*2",
        "x=0xff\nprint(x+1)",
        "x=0b101\nprint(x+1)",
        "x=int('ff',16)\nprint(x)",
        "x=2\nprint(pow(x,10,7))",
        "x=2\nprint(x**10%7)",
        "x=3\nprint(1+(2+(3+(4+(5+(6+x))))))",
        "x=1\nx=x+x\nx=x+x\nx=x+x\nx=x+x\nx=x+x\nx=x+x\nx=x+x\nprint(x)",
        "x=99999999999999999999999999999999999999999999999999999999999999\nprint(x*0)",
    };
    for (cases) |code| {
        const result = try python.lowerCalc(std.testing.allocator, code);
        defer if (result) |bytes| std.testing.allocator.free(bytes);
        try std.testing.expect(result == null);
    }
}

test "firmware refusal leaves wide runtime lowering available" {
    const code = "a=0x100\nb=0x20\nprint(a*b)";
    try std.testing.expectEqual(@as(?[]u8, null), try python.lowerCalc(std.testing.allocator, code));
    const source = (try python.lower(std.testing.allocator, code)).?;
    defer std.testing.allocator.free(source);
    try std.testing.expect(std.mem.indexOf(u8, source, "tf_run(tf_nodes,") != null);
    try std.testing.expect(std.mem.indexOf(u8, source, "\"100\"") != null);
}

test "multiple wide prints retain runtime DAGs and charge the total deadline" {
    const a = std.testing.allocator;
    const code = "a=11232463670103386525\nb=732667528837627997\nc=242246186388029\n" ++
        "print(a-b*129+c)\nprint(b*129)\n";
    try std.testing.expectEqual(@as(?[]u8, null), try python.lowerCalc(a, code));
    const result = (try python.lowerWithCost(a, code, std.math.maxInt(usize))).?;
    defer a.free(result.source);
    try std.testing.expect(std.mem.indexOf(u8, result.source, "\"11232463670103386525\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, result.source, "\"732667528837627997\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, result.source, "\"242246186388029\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, result.source, "\"83281405303764237059\"") == null);
    try std.testing.expect(std.mem.indexOf(u8, result.source, "tf_roots") != null);
    try std.testing.expect(std.mem.indexOf(u8, result.source, "tf_nodes_1") == null);
    const total = result.estimated_steps.?;
    const exact = (try python.lowerWithCost(a, code, total)).?;
    defer a.free(exact.source);
    try std.testing.expectEqual(@as(?usize, total), exact.estimated_steps);
    try std.testing.expect((try python.lowerWithCost(a, code, total - 1)) == null);
    const single = (try python.lowerWithCost(a, "b=732667528837627997\nprint(b*129)", total)).?;
    defer a.free(single.source);
    try std.testing.expect(single.estimated_steps.? < total);
}

test "multiple prints snapshot bindings and validate the complete script" {
    const a = std.testing.allocator;
    const source = (try python.lower(a, "x=7\nprint(x)\nx=11\nprint(x)")).?;
    defer a.free(source);
    try std.testing.expect(std.mem.indexOf(u8, source, "tf_roots") != null);
    try std.testing.expect(std.mem.indexOf(u8, source, "\"7\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, source, "\"11\"") != null);
    for ([_][]const u8{
        "x=7\nprint(x)\nx=1//0\nprint(3)",
        "print(1)\nprint(2)\nx=10**62",
        "print(1)\nimport os\nprint(2)",
        "print(1)\nprint=3\nprint(2)",
        "print(1)\nprint(2.5)",
        "1\nprint(2)",
        "1\nx=2\nprint(x)",
        "print(1)\nprint(2)\nprint(3)\nprint(4)\nprint(5)\nprint(6)\nprint(7)\nprint(8)\nprint(9)",
    }) |code| try std.testing.expectEqual(@as(?[]u8, null), try python.lower(a, code));
}
