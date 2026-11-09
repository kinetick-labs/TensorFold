const std = @import("std");
const fraction = @import("runner_frontend_fraction.zig");
const compiler = @import("runner_frontend_program_compile.zig");
const reference = @import("runner_frontend_reference_test.zig");
const t = std.testing;
const Case = struct { code: []const u8, expected: []const u8 };

const frozen = [_]Case{
    .{
        .code = "a=-23777868243178464\nb=-41832496\nprint(a//b, a%b)\nprint(a/b)\nprint(divmod(a,b))\n",
        .expected = "568406634 0\n568406634.0\n(568406634, 0)\n",
    },
    .{
        .code = "from fractions import Fraction\nn=44253970636272\nprint(n//48, n%48)\nprint(Fraction(n,-48))\nprint(n/-48)\n",
        .expected = "921957721589 0\n-921957721589\n-921957721589.0\n",
    },
    .{
        .code = "from fractions import Fraction\nn=-160371104\nd=932\nf=Fraction(n,d)\nprint(f, float(f))\nprint(n//d, n%d)\nimport math\nprint(math.gcd(abs(n),d))\n",
        .expected = "-172072 -172072.0\n-172072 0\n932\n",
    },
    .{
        .code = "from fractions import Fraction\nn=32685677580\nd=-4826\nf=Fraction(n,d)\nprint(f, f.numerator, f.denominator)\nprint(n/d)\nimport math\nprint(math.gcd(n,4826))\n",
        .expected = "-6772830 -6772830 1\n-6772830.0\n4826\n",
    },
    .{
        .code = "from fractions import Fraction\nn=-334567426983174\nd=-8613\nprint(n//d, n%d)\nprint(Fraction(n,d))\nprint(divmod(n,d))\n",
        .expected = "38844470798 0\n38844470798\n(38844470798, 0)\n",
    },
    .{
        .code = "from fractions import Fraction\nn=392764052; d=116\nprint(n//d, n%d)\nprint(Fraction(n,d))\nprint(n/d)",
        .expected = "3385897 0\n3385897\n3385897.0\n",
    },
};

const neighbors = [_]Case{
    .{
        .code = "a=-144\nb=-12\nprint(a//b, a%b)\nprint(a/b)\nprint(divmod(a,b))",
        .expected = "12 0\n12.0\n(12, 0)\n",
    },
    .{
        .code = "from fractions import Fraction\nn=96\nprint(n//48, n%48)\nprint(Fraction(n,-48))\nprint(n/-48)",
        .expected = "2 0\n-2\n-2.0\n",
    },
    .{
        .code = "from fractions import Fraction\nn=-144\nd=12\nf=Fraction(n,d)\nprint(f,float(f))\nprint(n//d,n%d)\nimport math\nprint(math.gcd(abs(n),d))",
        .expected = "-12 -12.0\n-12 0\n12\n",
    },
    .{
        .code = "from fractions import Fraction\nn=144\nd=-12\nf=Fraction(n,d)\nprint(f,f.numerator,f.denominator)\nprint(n/d)\nimport math\nprint(math.gcd(n,12))",
        .expected = "-12 -12 1\n-12.0\n12\n",
    },
    .{
        .code = "from fractions import Fraction\nn=-144\nd=-12\nprint(n//d,n%d)\nprint(Fraction(n,d))\nprint(divmod(n,d))",
        .expected = "12 0\n12\n(12, 0)\n",
    },
    .{
        .code = "from fractions import Fraction\nn=144; d=12\nprint(n//d,n%d)\nprint(Fraction(n,d))\nprint(n/d)",
        .expected = "12 0\n12\n12.0\n",
    },
    .{
        .code = "n=72; d=6\nx=(n+72)/(d*2)\nprint(x,n==72)",
        .expected = "12.0 True\n",
    },
    .{
        .code = "from fractions import Fraction\nf=Fraction(0,-7)\nprint(f,f.numerator,f.denominator)",
        .expected = "0 0 1\n",
    },
    .{
        .code = "from fractions import Fraction\nf=Fraction(999999999999999,1)\nprint(f,float(f))",
        .expected = "999999999999999 999999999999999.0\n",
    },
    .{
        .code = "n=144; d=12; q,r=divmod(n,d)\nprint(q,r,n/d)",
        .expected = "12 0 12.0\n",
    },
};

test "Fraction division admits frozen scripts and edited operands without a host compiler" {
    for (frozen ++ neighbors) |case| {
        const lowered = (try fraction.lowerWithCost(t.allocator, case.code, 0)) orelse {
            std.debug.print("Fraction lower abstained: {s}\n", .{case.code});
            return error.UnexpectedAbstention;
        };
        defer t.allocator.free(lowered.source);
        try t.expect(lowered.estimated_steps == null);
    }
}

test "Fraction division emits original literals and runtime quotient and denominator operations" {
    const lowered = (try fraction.lowerWithCost(t.allocator, frozen[3].code, 0)).?;
    defer t.allocator.free(lowered.source);
    const nodes = lowered.source[(std.mem.indexOf(u8, lowered.source, "static const struct tf_node tf_nodes[]") orelse
        return error.MissingRuntimeNodes)..];
    try t.expect(std.mem.indexOf(u8, nodes, "\"32685677580\"") != null);
    try t.expect(std.mem.indexOf(u8, nodes, "\"4826\"") != null);
    try t.expect(std.mem.indexOf(u8, nodes, "\"6772830\"") == null);
    try t.expect(std.mem.indexOf(u8, nodes, "\"1\"") == null);
    try t.expect(std.mem.indexOf(u8, nodes, "{5,") != null);
}

test "Fraction division refuses unsupported types exceptions effects and float domains" {
    for ([_][]const u8{
        "print(7/2)",
        "print(1/0)",
        "print(0/1)",
        "print(0/-1)",
        "print(1000000000000000/1)",
        "print(-1000000000000000/1)",
        "print(12/3+1)",
        "print(12/3*2)",
        "print(12/3/2)",
        "print(12/(3/1))",
        "print(Fraction(12,3))",
        "from fractions import Fraction\nprint(Fraction(7,2))",
        "from fractions import Fraction\nprint(Fraction(1,0))",
        "from fractions import Fraction\nprint(Fraction(1000000000000000,1))",
        "from fractions import Fraction\nf=Fraction(0,-1)\nprint(float(f))",
        "from fractions import Fraction\nprint(float(12))",
        "from fractions import Fraction\nf=Fraction(12,3)\nprint(f+1)",
        "from fractions import Fraction\nf=Fraction(12,3)\nf=Fraction(18,3)\nprint(f)",
        "from fractions import Fraction\nf=Fraction(12,3)\ng=f\nprint(g)",
        "from fractions import Fraction\nf=Fraction(12,3)\ng=(f)\nprint(g)",
        "from fractions import Fraction\nf=Fraction(12,3)\nf.numerator=0\nprint(f)",
        "from fractions import Fraction\nFraction=0\nprint(12/3)",
        "float=0\nprint(12/3)",
        "abs=0\nprint(12/3)",
        "__builtins__=0\nprint(12/3)",
        "from fractions import Fraction as F\nprint(12/3)",
        "from fractions import *\nprint(12/3)",
        "import fractions\nprint(12/3)",
        "print(12/3)\nimport os",
        "print(12/3)\nfrom math import *",
        "print(12/3)\nimport math as m",
        "print(12/3)\nunused=1//0",
        "print(12/3)\nunused=10**62",
        "print(12/3)\nfrom fractions import Fraction\nunused=Fraction(7,2)",
        "print(12/3)\nprint(math.gcd(12,3))\nimport math",
        "print(12/3)\nprint(gcd(12,3))",
        "print(12/3)\nimport math\nprint(math.gcd(abs(divmod(12,3)),3))",
        "print(12/3)\nimport math\nprint(math.gcd(divmod(12,3),3))",
        "x=12/3\nprint(x+1)",
        "n=12\nn=18\nprint(n/3)",
        "n=12\nn,r=divmod(18,3)\nprint(n/3)",
        "n=12\nq,q=divmod(n,3)\nprint(n/3)",
        "print(12/3,end='')",
        "print(12/3)\nraise ValueError()",
    }) |code| {
        const result = try fraction.lowerWithCost(t.allocator, code, 0);
        defer if (result) |lowered| t.allocator.free(lowered.source);
        if (result != null) std.debug.print("Fraction lower unexpectedly accepted: {s}\n", .{code});
        try t.expect(result == null);
    }
}

test "Fraction division cost includes formatting and enforces an inclusive budget" {
    const code = "from fractions import Fraction\nn=144; d=-12; f=Fraction(n,d)\nprint(f,float(f),f.denominator)";
    const full = (try fraction.lowerWithCost(t.allocator, code, std.math.maxInt(usize))).?;
    defer t.allocator.free(full.source);
    const steps = full.estimated_steps.?;
    const exact = (try fraction.lowerWithCost(t.allocator, code, steps)).?;
    defer t.allocator.free(exact.source);
    try t.expectEqual(steps, exact.estimated_steps.?);
    const short = try fraction.lowerWithCost(t.allocator, code, steps - 1);
    defer if (short) |lowered| t.allocator.free(lowered.source);
    try t.expect(short == null);
}

test "Fraction frozen scripts and neighbors execute as opaque RV32 arithmetic with exact formatting" {
    var env = try t.environ.createMap(t.allocator);
    defer env.deinit();
    const clang = env.get("TF_RUNNER_CLANG") orelse return error.SkipZigTest;
    const root = env.get("TF_RUNNER_QWENPUTER") orelse return error.SkipZigTest;
    for (frozen ++ neighbors) |case| {
        const lowered = (try fraction.lowerWithCost(t.allocator, case.code, 0)).?;
        defer t.allocator.free(lowered.source);
        const elf = (try compiler.compile(t.allocator, t.io, lowered.source, .{
            .clang = clang,
            .linker = env.get("TF_RUNNER_LLD"),
            .step_budget = 0,
            .opaque_inputs = true,
            .optimize = .O2,
        })) orelse return error.UnexpectedAbstention;
        defer t.allocator.free(elf);
        try reference.check(root, env.get("TF_RUNNER_REFERENCE_PYTHON") orelse "/usr/bin/python3", .{
            .kind = .elf,
            .bytes = elf,
        }, case.expected);
    }
}

test "Fraction comment bytes preserve Python syntax refusal" {
    try t.expectEqual(@as(?@import("runner_frontend_integer.zig").Lowered, null), try fraction.lowerWithCost(t.allocator, "print(12/3) #\x00", 10522));
}
