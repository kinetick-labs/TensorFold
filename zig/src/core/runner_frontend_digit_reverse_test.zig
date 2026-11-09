const std = @import("std");
const ast = @import("runner_frontend_python_ast.zig");
const expr = @import("runner_frontend_python_expr.zig");
const blocks = @import("runner_frontend_python_blocks.zig");
const compiler = @import("runner_frontend_program_compile.zig");
const frontend = @import("runner_frontend.zig");
const t = std.testing;

pub const fixtures = [_]struct { code: []const u8, expected: []const u8 }{
    .{
        .code =
        \\def check(n):
        \\    rev = int(str(n)[::-1])
        \\    return n == 2 * rev - 1
        \\print(check(70))
        \\print(check(13))
        \\print(check(0))
        ,
        .expected = "False\nFalse\nFalse\n",
    },
    .{
        .code =
        \\print(int(str(70)[::-1]), int(str(1200)[::-1]), int(str(0)[::-1]))
        \\print(int(str(1463847412)[::-1]), int(str(2000000000)[::-1]))
        ,
        .expected = "7 21 0\n2147483641 2\n",
    },
    .{
        .code =
        \\def reverse(n):
        \\    return int(str(n)[::-1])
        \\def check(n):
        \\    return n == 2 * reverse(n) - 1
        \\for n in [0, 1, 10, 73, 1200]:
        \\    print(reverse(n), check(n))
        \\print(reverse(2147447412))
        ,
        .expected = "0 False\n1 True\n1 False\n37 True\n21 False\n2147447412\n",
    },
    .{
        .code =
        \\def total(xs):
        \\    return int(str(sum(n for n in xs if n > 0))[::-1])
        \\print(total([10, -1, 2]))
        \\print(sum(int(str(n)[::-1]) for n in [10, 1200, 30]))
        \\print(int(str(+True)[::-1]), int(str(-0)[::-1]))
        ,
        .expected = "21\n25\n1 0\n",
    },
};

const refused = [_][]const u8{
    "print(int(str(-70)[::-1]))",
    "n = -1\nunused = int(str(n)[::-1])\nprint(7)",
    "print(int(str(True)[::-1]))",
    "print(int(str(False)[::-1]))",
    "n = True\nunused = int(str(n)[::-1])\nprint(7)",
    "print(int(str(1 == 1)[::-1]))",
    "print(int(str(1 and True)[::-1]))",
    "print(int(str([True][0])[::-1]))",
    "def f(n):\n    return int(str(n)[::-1])\nprint(f(True))",
    "def f(n):\n    return int(str(n)[::-1])\nunused = f(-70)\nprint(7)",
    "print(int(str(2147483647)[::-1]))",
    "print(int(str(1000000009)[::-1]))",
    "unused = int(str(2147483647)[::-1])\nprint(7)",
    "print(int(str(2147483648)[::-1]))",
    "print(int(str(70)[::1]))",
    "print(int(str(70)[::-2]))",
    "print(int(str(70)[::-10]))",
    "print(int(str(70)[1::-1]))",
    "print(int(str(70)[:1:-1]))",
    "print(int(str(70)[::-1][0]))",
    "print(int(str(70)[::-1], 10))",
    "print(int(str(70, 10)[::-1]))",
    "print(int(str(object=70)[::-1]))",
    "print(int(str(unknown(70))[::-1]))",
    "print(int(str([70])[::-1]))",
    "print(int(str('70')[::-1]))",
    "print(int('70'[::-1]))",
    "print(int(70))",
    "print(str(70))",
    "int = 1\nprint(int(str(70)[::-1]))",
    "str = 1\nprint(int(str(70)[::-1]))",
    "def int(n):\n    return n\nprint(int(str(70)[::-1]))",
    "def str(n):\n    return n\nprint(int(str(70)[::-1]))",
    "def f(int):\n    return int(str(70)[::-1])\nprint(f(1))",
    "def f(str):\n    return int(str(70)[::-1])\nprint(f(1))",
};

test "digit reversal parses as one integer unary intrinsic with the original operand" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    var p: ast.Program = .{ .a = arena.allocator() };
    const id = try expr.parse(&p, "int ( str (70 + 2) [ : : - 1 ] )");
    const node = p.nodes.items[id];
    try t.expectEqual(.reverse_digits, node.tag);
    try t.expectEqual(.add, p.nodes.items[node.lhs].tag);
    try t.expectEqual(@as(i64, 70), p.nodes.items[p.nodes.items[node.lhs].lhs].value);
    const boolean = try expr.parse(&p, "int(str(True)[::-1])");
    try t.expectEqual(.boolean, p.nodes.items[p.nodes.items[boolean].lhs].tag);
    try t.expectError(error.Unsupported, expr.boundedDepth(&p, id, 7));
}

test "digit reversal lowering retains volatile inputs and runtime div mod rather than answers" {
    for (fixtures) |fixture| {
        const source = (try blocks.lowerPython(t.allocator, fixture.code)) orelse return error.UnexpectedAbstention;
        defer t.allocator.free(source);
        try t.expect(compiler.safeFormats(source));
        try t.expect(std.mem.indexOf(u8, source, "static volatile int py_inputs[]") != null);
        try t.expect(std.mem.indexOf(u8, source, "py_hold(py_reverse_digits(py_hold(") != null);
        try t.expect(std.mem.indexOf(u8, source, "r=r*10+n%10; n/=10;") != null);
        try t.expect(std.mem.indexOf(u8, source, "2147483641") == null);
    }
}

test "digit reversal proof refuses invalid conversions bounds and custom builtin bindings" {
    for (refused) |code| {
        const source = try blocks.lowerPython(t.allocator, code);
        defer if (source) |owned| t.allocator.free(owned);
        if (source != null) std.debug.print("unexpected digit reversal admission: {s}\n", .{code});
        try t.expect(source == null);
    }
}

test "digit reversal charges for each processed digit even when trailing zeroes disappear" {
    const short = (try blocks.lowerWithCost(t.allocator, "print(int(str(1)[::-1]))")) orelse
        return error.UnexpectedAbstention;
    defer t.allocator.free(short.source);
    const long = (try blocks.lowerWithCost(t.allocator, "print(int(str(1000000000)[::-1]))")) orelse
        return error.UnexpectedAbstention;
    defer t.allocator.free(long.source);
    try t.expect(long.estimated_steps > short.estimated_steps);
}

test "configured digit reversal frontend matches CPython and the RV32 reference" {
    var env = try t.environ.createMap(t.allocator);
    defer env.deinit();
    const clang = env.get("TF_RUNNER_CLANG") orelse return error.SkipZigTest;
    const root = env.get("TF_RUNNER_QWENPUTER") orelse return error.SkipZigTest;
    const python = env.get("TF_RUNNER_REFERENCE_PYTHON") orelse "/usr/bin/python3";
    var context: frontend.Context = .{ .io = t.io, .compiler = .{
        .clang = clang,
        .linker = env.get("TF_RUNNER_LLD"),
        .opaque_inputs = true,
        .step_budget = 0,
    } };
    const front = context.frontend();
    for (fixtures) |fixture| {
        const oracle = try std.process.run(t.allocator, t.io, .{
            .argv = &.{ python, "-I", "-c", fixture.code },
            .stdout_limit = .limited(4096),
            .stderr_limit = .limited(4096),
            .timeout = .{ .duration = .{ .raw = .fromSeconds(10), .clock = .awake } },
        });
        defer t.allocator.free(oracle.stdout);
        defer t.allocator.free(oracle.stderr);
        try t.expect(oracle.term.success());
        try t.expectEqualStrings(fixture.expected, oracle.stdout);
        const result = try front.translate(front.ctx, t.allocator, .{ .form = .fence, .lang = "python", .code = fixture.code });
        defer if (result == .program) t.allocator.free(result.program.bytes);
        try t.expect(result == .program);
        try t.expectEqual(.elf, result.program.kind);
        try @import("runner_frontend_reference_test.zig").check(root, python, result.program, oracle.stdout);
    }
    for (refused) |code| {
        try @import("runner_frontend_collections_compat_test.zig").checkRejectionOrExact(
            &env,
            &front,
            .{ .form = .fence, .lang = "python", .code = code },
        );
    }
}
