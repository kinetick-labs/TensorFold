const std = @import("std");
const python = @import("runner_frontend_integer_python.zig");
const frontend = @import("runner_frontend.zig");
const reference = @import("runner_frontend_reference_test.zig");
const optimization = @import("runner_frontend_optimization_test.zig");
const t = std.testing;

pub const dividend = "314159265358979323846264338327950288419716939937510";
pub const divisor = "271828182845904523536";
pub const inputs = "a=" ++ dividend ++ "\nb=" ++ divisor ++ "\n";
pub const verification = inputs ++ "q,r=divmod(a,b)\nprint(q)\nprint(r)\nprint(q*b+r==a)";
pub const expected = "1155727349790921717910215407028\n23663814570294126502\nTrue\n";

const fixtures = [_][2][]const u8{
    .{ verification, expected },
    .{ "a=0x100000000;b=0x10001;q,r=divmod(a,b);print(q,r,q*b+r==a)", "65535 1 True\n" },
    .{ "a=0x7fff00000000;b=0x8000ffff;q,r=divmod(a,b);print(q,r,q*b+r==a)", "65532 327676 True\n" },
    .{ "a=0x800000000000ffff;b=0x800000000001;q,r=divmod(a,b);print(q,r,q*b+r==a)", "65535 140737488355328 True\n" },
    .{ "print(1);print(2)", "1\n2\n" },
    .{ inputs ++ "q,r=divmod(a,b)\nprint(q,r)", "1155727349790921717910215407028 23663814570294126502\n" },
    .{ "a=-7; b=3; q,r=divmod(a,b); print(q); print(r); print(q*b+r==a)", "-3\n2\nTrue\n" },
    .{ "q,r=divmod(7,-3); print(q,r); print(q*(-3)+r==7)", "-3 -2\nTrue\n" },
    .{ "q,r=divmod(-7,-3); print(q,r,q*(-3)+r==-7)", "2 -1 True\n" },
    .{ "q,q=divmod(7,3); print(q)", "1\n" },
    .{ "x=7; print(x); x=11; print(x)", "7\n11\n" },
    .{ "x=3==3; print(x); print(x+1)", "True\n2\n" },
    .{ "a=100000000000000000000; b=a+1; print(a<b,a>=b,a!=b)", "True False True\n" },
    .{ "import math; a=48; b=18; print(math.gcd(a,b))", "6\n" },
    .{ "x=2; y=3\nprint(x+y); # trailing separator/comment\n", "5\n" },
};

const refused = [_][]const u8{
    "print(+(1==1))",
    "print(len(str(1==1)))",
    "print(sum(map(int,str(1==1))))",
    "x=1==1;print(len(str(x)))",
    "q,r=divmod(7,0); print(q,r)",
    "q,r=divmod(7.0,3); print(q,r)",
    "q,r=divmod(7,3.0); print(q,r)",
    "q,r,s=divmod(7,3); print(q,r)",
    "q,r=(7,3); print(q,r)",
    "q,r=divmod(7,3)[0]; print(q,r)",
    "q,r=divmod(7,3); print(q,r,sep=':')",
    "q,r=divmod(7,3); print(q and r)",
    "q,r=divmod(7,3); print(0<q<3)",
    "q,r=divmod(7,3); print(q); x=1//0",
    "q,r=divmod(7,3); print(q); assert False",
    "q,r=divmod(10**62,3); print(q,r)",
    ";print(1)",
    "x=2;;print(x)",
    "x=2; ;print(x)",
    "x=2;\n print(x)",
    "x=2; y=int('3;4');print(x+y)",
    "print(1); import os; print(2)",
};

test "whole integer tool scripts retain source DAGs and fit the unchanged cost cap" {
    for (fixtures) |fixture| {
        const result = (try python.lowerWithCost(t.allocator, fixture[0], 10_522)) orelse {
            std.debug.print("script cost abstained: {s}\n", .{fixture[0]});
            return error.UnexpectedAbstention;
        };
        defer t.allocator.free(result.source);
        try t.expect(result.estimated_steps.? <= 10_522);
        try t.expect(std.mem.indexOf(u8, result.source, "tf_run") != null);
    }
    const source = (try python.lower(t.allocator, verification)).?;
    defer t.allocator.free(source);
    try t.expect(std.mem.indexOf(u8, source, dividend) != null);
    try t.expect(std.mem.indexOf(u8, source, divisor) != null);
    try t.expect(std.mem.indexOf(u8, source, "1155727349790921717910215407028") == null);
    try t.expect(std.mem.indexOf(u8, source, "23663814570294126502") == null);
}

test "whole script validation refuses malformed unpacking effects floats and chains" {
    for (refused) |code| {
        const result = try python.lower(t.allocator, code);
        defer if (result) |s| t.allocator.free(s);
        try t.expect(result == null);
    }
}

test "configured whole scripts execute grouped and boolean prints exactly like CPython" {
    var env = try t.environ.createMap(t.allocator);
    defer env.deinit();
    const clang = env.get("TF_RUNNER_CLANG") orelse return error.SkipZigTest;
    const root = env.get("TF_RUNNER_QWENPUTER") orelse return error.SkipZigTest;
    var context: frontend.Context = .{ .io = t.io, .compiler = .{
        .clang = clang,
        .linker = env.get("TF_RUNNER_LLD"),
    } };
    const f = context.frontend();
    for (fixtures) |fixture| {
        const translated = try f.translate(f.ctx, t.allocator, .{ .form = .fence, .lang = "python", .code = fixture[0] });
        if (translated != .program) {
            std.debug.print("script dispatch abstained: {s}\n", .{fixture[0]});
            return error.UnexpectedAbstention;
        }
        defer t.allocator.free(translated.program.bytes);
        try reference.check(root, env.get("TF_RUNNER_REFERENCE_PYTHON") orelse "/usr/bin/python3", translated.program, fixture[1]);
        if (translated.program.kind == .elf) try optimization.assertNoOutputInRodata(translated.program.bytes, fixture[1]);
    }
}

test "configured whole division script passes optimized work and rodata folding guards" {
    var env = try t.environ.createMap(t.allocator);
    defer env.deinit();
    if (env.get("TF_RUNNER_CLANG") == null or env.get("TF_RUNNER_QWENPUTER") == null)
        return error.SkipZigTest;
    const source = (try python.lower(t.allocator, verification)).?;
    defer t.allocator.free(source);
    try optimization.checkOptimization(&env, source, expected, true);
}

test "configured Boolean coercion preserves CPython semantics through generic fallback" {
    var env = try t.environ.createMap(t.allocator);
    defer env.deinit();
    const clang = env.get("TF_RUNNER_CLANG") orelse return error.SkipZigTest;
    _ = env.get("TF_RUNNER_QWENPUTER") orelse return error.SkipZigTest;
    var context: frontend.Context = .{ .io = t.io, .compiler = .{
        .clang = clang,
        .linker = env.get("TF_RUNNER_LLD"),
    } };
    const f = context.frontend();
    const result = try f.translate(f.ctx, t.allocator, .{
        .form = .fence,
        .lang = "python",
        .code = "print(+(1==1))",
    });
    defer if (result == .program) t.allocator.free(result.program.bytes);
    try t.expect(result == .program);
    const compat = @import("runner_frontend_collections_compat_test.zig");
    try compat.checkExact(&env, result.program, "print(+(1==1))");
    for ([_][]const u8{
        "print(+True)",
        "print(-True)",
        "print(len(str(1==1)))",
        "x=1==1;print(len(str(x)))",
        "print(sum(map(int,str(1==1))))",
    }) |code| {
        try compat.checkRejectionOrExact(&env, &f, .{ .form = .fence, .lang = "python", .code = code });
    }
}
