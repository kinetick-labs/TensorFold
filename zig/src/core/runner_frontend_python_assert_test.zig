const std = @import("std");
const blocks = @import("runner_frontend_python_blocks.zig");
const compiler = @import("runner_frontend_program_compile.zig");
const frontend = @import("runner_frontend.zig");
const t = std.testing;

pub const fixtures = [_]struct { code: []const u8, expected: []const u8 }{
    .{
        .code =
        \\def tetrahedral_number(n):
        \\    return n * (n + 1) * (n + 2) // 6
        \\
        \\assert tetrahedral_number(5) == 35
        \\print(tetrahedral_number(5))
        ,
        .expected = "35\n",
    },
    .{
        .code =
        \\def centered_hexagonal_number(n):
        \\    return 3 * n * (n - 1) + 1
        \\
        \\assert centered_hexagonal_number(10) == 271
        \\print(centered_hexagonal_number(10))
        ,
        .expected = "271\n",
    },
    .{
        .code =
        \\def pos_count(numbers):
        \\    return sum(1 for number in numbers if number > 0)
        \\
        \\assert pos_count([1, -2, 3, -4]) == 2
        \\print(pos_count([1, -2, 3, -4]))
        ,
        .expected = "2\n",
    },
    .{
        .code = "def f(n):\n    assert n > 0\n    return n + 1\nassert f(3) == 4\nprint(f(3))",
        .expected = "4\n",
    },
    .{
        .code = "assert True\nif False:\n    assert False\nprint(True)",
        .expected = "True\n",
    },
    .{
        .code = "assert sum(i for i in range(5, -3, -2) if i > 0) == 9\nprint(sum(i for i in range(5, -3, -2) if i > 0))",
        .expected = "9\n",
    },
    .{
        .code = "numbers = [-2,0,3,4]\nassert sum(n for n in numbers if n > 0) == 7\nprint(sum(n for n in numbers if n > 0))",
        .expected = "7\n",
    },
    .{
        .code = "def f(xs):\n    return sum(x for x in xs if x > 0)\nassert f([]) == 0\nprint(f([]))",
        .expected = "0\n",
    },
    .{
        .code = "n = 99\nassert sum(n for n in [1,2,3] if n > 1) == 5\nprint(n)",
        .expected = "99\n",
    },
    .{
        .code = "def f(n):\n    xs = [n, 2]\n    return sum(x for x in xs if x > 0)\nassert f(3) == 5\nprint(f(3))",
        .expected = "5\n",
    },
};

const refused = [_][]const u8{
    "def tetrahedral_number(n):\n    return n * (n + 1) * (n + 2) // 6\nassert tetrahedral_number(5) == 36\nprint(tetrahedral_number(5))",
    "def centered_hexagonal_number(n):\n    return 3 * n * (n - 1) + 2\nassert centered_hexagonal_number(10) == 271\nprint(centered_hexagonal_number(10))",
    "print(123)\nassert False\nprint(456)",
    "def f(n):\n    assert n > 0\n    return n\nprint(f(-1))",
    "def f(n):\n    if n > 0:\n        assert False\n    return n\nprint(f(1))",
    "for n in range(3):\n    assert n < 2\nprint(123)",
    "def f(n):\n    return n // 0\nassert f(1) == 1\nprint(123)",
    "assert missing\nprint(123)",
    "assert 1, 'unsupported message'\nprint(123)",
    "assert 'text'\nprint(123)",
    "assert 1.0\nprint(123)",
    "def pos_count(numbers):\n    return sum(1 for number in numbers if number > 0)\nassert pos_count([1, -2, 3, -4]) == 3\nprint(2)",
    "def f(xs):\n    xs.append(4)\n    return sum(x for x in xs if x > 0)\nprint(f([1]))",
    "xs = [1]\nys = xs\nprint(sum(x for x in ys if x > 0))",
    "print(sum(x for x in range(0, 3, 0) if x > 0))",
    "print(sum(x for x in range(10001) if x > 0))",
    "print(sum(1for x in [1,2] if x > 0))",
    "print(sum(x for x in unknown if x > 0))",
    "print(sum(x for x in [1.5] if x > 0))",
    "print(sum(x for x in ['1'] if x > 0))",
    "print(sum(x for x in [[1]] if x > 0))",
    "print(sum(x for x in [2147483647, 1] if x > 0))",
    "print(sum(x for x in [1,2] if x > 0))\nprint(x)",
    "def f(xs):\n    return sum(x for x in xs if x > 0)\nprint(f(3))",
    "def f(xs):\n    xs = [4]\n    return sum(x for x in xs)\nprint(f([1]))",
};

test "assert helpers retain source conditions and opaque scalar operations" {
    for (fixtures) |fixture| {
        const result = (try blocks.lowerWithCost(t.allocator, fixture.code)) orelse return error.UnexpectedAbstention;
        defer t.allocator.free(result.source);
        try t.expect(std.mem.indexOf(u8, result.source, "if(!(") != null);
        try t.expect(std.mem.indexOf(u8, result.source, "))py_assert_fail();") != null);
        try t.expect(std.mem.indexOf(u8, result.source, "py_hold(") != null);
        try t.expect(std.mem.indexOf(u8, result.source, "static volatile int py_inputs[]") != null);
        try t.expect(result.estimated_steps > 200);
        try t.expect(result.estimated_steps <= 10_522);
        try t.expect(compiler.safeFormats(result.source));
    }
}

test "mutatedassertmustrefuse and failing bodymustnotadmit whole cell" {
    for (refused) |code| {
        const source = try blocks.lowerPython(t.allocator, code);
        defer if (source) |s| t.allocator.free(s);
        try t.expect(source == null);
    }
}

test "assertions and filtered iteration add cost without replacing source" {
    const plain = (try blocks.lowerWithCost(t.allocator, "print(7)")) orelse return error.UnexpectedAbstention;
    defer t.allocator.free(plain.source);
    const asserted = (try blocks.lowerWithCost(t.allocator, "assert 7 == 7\nprint(7)")) orelse return error.UnexpectedAbstention;
    defer t.allocator.free(asserted.source);
    try t.expect(asserted.estimated_steps > plain.estimated_steps);
    const longer = (try blocks.lowerWithCost(t.allocator, "assert sum(i for i in range(20) if i > 0) == 190\nprint(7)")) orelse return error.UnexpectedAbstention;
    defer t.allocator.free(longer.source);
    try t.expect(longer.estimated_steps > asserted.estimated_steps);
    try t.expect(std.mem.indexOf(u8, longer.source, "for(") != null);
    try t.expect(std.mem.indexOf(u8, longer.source, "py_add(py_sum") != null);
}

const reference =
    \\import sys
    \\from pathlib import Path
    \\sys.path.insert(0, str(Path(sys.argv[1]) / 'experimental/riscv'))
    \\from chipformer.rv import elf, ref
    \\m = ref.Machine.from_program(elf.program(sys.argv[2]))
    \\tokens = 1
    \\while not m.halted and tokens < 4000000:
    \\    tokens += len(m.record())
    \\assert tokens <= 4000000 and m.halted and m.fault is None, (tokens, m.fault)
    \\assert m.exit_code == int(sys.argv[3]), m.exit_code
    \\sys.stdout.buffer.write(bytes(m.out))
;

fn checkReference(env: *const std.process.Environ.Map, bytes: []const u8, expected: []const u8, exit: []const u8) !void {
    const root = env.get("TF_RUNNER_QWENPUTER") orelse return error.SkipZigTest;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const file = try tmp.dir.createFile(t.io, "assert.elf", .{});
    defer file.close(t.io);
    try file.writeStreamingAll(t.io, bytes);
    const path = try tmp.dir.realPathFileAlloc(t.io, "assert.elf", t.allocator);
    defer t.allocator.free(path);
    const result = try std.process.run(t.allocator, t.io, .{
        .argv = &.{ env.get("TF_RUNNER_REFERENCE_PYTHON") orelse "/usr/bin/python3", "-c", reference, root, path, exit },
        .stdout_limit = .limited(4096),
        .stderr_limit = .limited(16384),
        .timeout = .{ .duration = .{ .raw = .fromSeconds(30), .clock = .awake } },
    });
    defer t.allocator.free(result.stdout);
    defer t.allocator.free(result.stderr);
    if (!result.term.success()) {
        std.debug.print("assert RV32 reference failed: {s}\n", .{result.stderr});
        return error.ReferenceFailed;
    }
    try t.expectEqualStrings(expected, result.stdout);
}

test "configured assertion cells match exact CPython stdout through fence dispatch" {
    var env = try t.environ.createMap(t.allocator);
    defer env.deinit();
    const clang = env.get("TF_RUNNER_CLANG") orelse return error.SkipZigTest;
    var context: frontend.Context = .{ .io = t.io, .compiler = .{
        .clang = clang,
        .linker = env.get("TF_RUNNER_LLD"),
        .opaque_inputs = true,
    } };
    const front = context.frontend();
    for (fixtures) |fixture| {
        const result = try front.translate(front.ctx, t.allocator, .{ .form = .fence, .lang = "python", .code = fixture.code });
        try t.expect(result == .program);
        defer t.allocator.free(result.program.bytes);
        try t.expect(result.program.kind == .elf or result.program.kind == .calc);
        const oracle = try std.process.run(t.allocator, t.io, .{
            .argv = &.{ env.get("TF_RUNNER_REFERENCE_PYTHON") orelse "/usr/bin/python3", "-I", "-c", fixture.code },
            .stdout_limit = .limited(4096),
            .stderr_limit = .limited(4096),
            .timeout = .{ .duration = .{ .raw = .fromSeconds(2), .clock = .awake } },
        });
        defer t.allocator.free(oracle.stdout);
        defer t.allocator.free(oracle.stderr);
        try t.expect(oracle.term.success());
        try t.expectEqualStrings(fixture.expected, oracle.stdout);
        try @import("runner_frontend_reference_test.zig").check(
            env.get("TF_RUNNER_QWENPUTER") orelse return error.SkipZigTest,
            env.get("TF_RUNNER_REFERENCE_PYTHON") orelse "/usr/bin/python3",
            result.program,
            oracle.stdout,
        );
    }
    for (refused) |code| {
        try @import("runner_frontend_collections_compat_test.zig").checkRejectionOrExact(
            &env,
            &front,
            .{ .form = .fence, .lang = "python", .code = code },
        );
    }
}

test "configured mutated emitted assertion exits before printing in global and helper bodies" {
    var env = try t.environ.createMap(t.allocator);
    defer env.deinit();
    const clang = env.get("TF_RUNNER_CLANG") orelse return error.SkipZigTest;
    const cases = [_][]const u8{
        "assert 7 == 7\nprint(123)",
        "def f(n):\n    assert n == 7\n    return n\nprint(f(7))",
    };
    for (cases) |code| {
        const source = (try blocks.lowerPython(t.allocator, code)) orelse return error.UnexpectedAbstention;
        defer t.allocator.free(source);
        const comparison = std.mem.lastIndexOf(u8, source, "py_eq(") orelse return error.MissingRuntimeCondition;
        @memcpy(source[comparison .. comparison + 5], "py_ne");
        const bytes = (try compiler.compile(t.allocator, t.io, source, .{
            .clang = clang,
            .linker = env.get("TF_RUNNER_LLD"),
            .opaque_inputs = true,
            .step_budget = 0,
        })) orelse return error.CompilationFailed;
        defer t.allocator.free(bytes);
        try checkReference(&env, bytes, "", "1");
    }
}
