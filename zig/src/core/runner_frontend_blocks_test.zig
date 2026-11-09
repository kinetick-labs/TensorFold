const std = @import("std");
const blocks = @import("runner_frontend_python_blocks.zig");
const compiler = @import("runner_frontend_program_compile.zig");
const frontend = @import("runner_frontend.zig");
const t = std.testing;

pub const prime =
    \\def is_prime(n):
    \\    if n < 2:
    \\        return False
    \\    if n % 2 == 0:
    \\        return n == 2
    \\    d = 3
    \\    while d * d <= n:
    \\        if n % d == 0:
    \\            return False
    \\        d += 2
    \\    return True
    \\primes = []
    \\n = 2
    \\while len(primes) < 100:
    \\    if is_prime(n):
    \\        primes.append(n)
    \\    n += 1
    \\print("100th prime:", primes[-1])
    \\print("Sum of first 100 primes:", sum(primes))
;

pub const factorial =
    \\def factorial(n):
    \\    if n < 0:
    \\        raise ValueError("factorial is undefined for negative values")
    \\    result = 1
    \\    for i in range(2, n + 1):
    \\        result *= i
    \\    return result
    \\for n in [0, 1, 5, 12]:
    \\    print(f"{n}! = {factorial(n)}")
;

const bounded_lcm =
    \\from math import gcd
    \\def lcm(a, b):
    \\    return a // gcd(a, b) * b
    \\n = 1
    \\for k in range(1, 6):
    \\    n = lcm(n, k)
    \\print("LCM:", n)
    \\print("Divisible:", all(n % d == 0 for d in range(1, 6)))
    \\m = 1
    \\while not all(m % d == 0 for d in range(1, 6)):
    \\    m += 1
    \\print("Brute force:", m)
;

const overlimit_lcm =
    \\from math import gcd
    \\def lcm(a, b):
    \\    return a // gcd(a, b) * b
    \\n = 1
    \\for k in range(1, 21):
    \\    n = lcm(n, k)
    \\print("LCM:", n)
    \\print("Divisible:", all(n % d == 0 for d in range(1, 21)))
    \\m = 1
    \\while not all(m % d == 0 for d in range(1, 21)):
    \\    m += 1
    \\print("Brute force:", m)
;

const examples = [_]struct { code: []const u8, expected: []const u8 }{
    .{ .code = "print(1,2)", .expected = "1 2\n" },
    .{ .code = prime, .expected = "100th prime: 541\nSum of first 100 primes: 24133\n" },
    .{ .code = factorial, .expected = "0! = 1\n1! = 1\n5! = 120\n12! = 479001600\n" },
    .{ .code = bounded_lcm, .expected = "LCM: 60\nDivisible: True\nBrute force: 60\n" },
    .{
        .code = "a = 5\ndef f(x):\n    a = x + 2\n    return a\nprint(f(3), a, -7 // 3, -7 % 3, True, not 1)\n",
        .expected = "5 5 -3 2 True False\n",
    },
    .{
        .code = "print(\"quote: \\\" and newline:\\n and 100% __ asm ?? %:\", f\"{{{1 + 2}}}\")\n",
        .expected = "quote: \" and newline:\n and 100% __ asm ?? %: {3}\n",
    },
};

test "structured lowering retains prime calls loops append and full translation unit" {
    for (examples) |example| {
        const source = (try blocks.lowerPython(t.allocator, example.code)) orelse return error.UnexpectedAbstention;
        defer t.allocator.free(source);
        try t.expect(std.mem.indexOf(u8, source, "int main(void)") != null);
        try t.expect(compiler.safeFormats(source));
    }
    const source = (try blocks.lowerPython(t.allocator, prime)) orelse return error.UnexpectedAbstention;
    defer t.allocator.free(source);
    try t.expect(std.mem.indexOf(u8, source, "while(") != null);
    try t.expect(std.mem.indexOf(u8, source, "py_f0(") != null);
    try t.expect(std.mem.indexOf(u8, source, "py_append(") != null);
    try t.expect(std.mem.indexOf(u8, source, "24133") == null);
    try t.expect(std.mem.indexOf(u8, source, "541") == null);
}

test "structured proof refuses whole cells on unsupported semantics and bounds" {
    const rejected = [_][]const u8{
        "print(missing)",
        "print(1.5)",
        "from decimal import Decimal\nprint(Decimal('1.0'))",
        "len = 3\nprint(len)",
        "abs = 3\nprint(abs)",
        "x = 1\nx + = 2\nprint(x)",
        "x = 4\nx // = 2\nprint(x)",
        "def print(n):\n    return n\nprint(1)",
        "def f(n):\n    return f(n)\nprint(f(1))",
        "def f(n):\n    return g(n)\ndef g(n):\n    return f(n)\nprint(f(1))",
        "x = 1\nwhile x:\n    x += 0\nprint(x)",
        "x = 2147483647\nx += 1\nprint(x)",
        "x = 1\nprint(x // 0)",
        "a = []\nfor n in range(129):\n    a.append(n)\nprint(len(a))",
        "a = [1]\nb = a\nprint(sum(b))",
        "a = []\nprint(a[-1])",
        "def f(n):\n    if n:\n        a = 1\n    return a\nprint(f(0))",
        "def f(n):\n    if n:\n        return 2\nprint(f(0))",
        "def f(n):\n    print(n)\n    return n\nprint(f(1))",
        "def f(n):\n    a.append(n)\n    return n\na = []\nprint(f(1))",
        "def f(n):\n    return n\nf = 1\nprint(f)",
        "print(open('file'))",
        "print(__import__('os'))",
        "raise ValueError('visited')",
        "raise RuntimeError('unsupported')",
        "if False:\n    print(open('file'))\nprint(1)",
        "print(gcd(4, 2))",
        "print(f'{1:03}')",
        "x = 2\nprint(x ** 3)",
        "def f(n):\n    return n\nprint(all(f() for f in range(2)))",
        overlimit_lcm,
    };
    for (rejected) |code| {
        const source = try blocks.lowerPython(t.allocator, code);
        defer if (source) |owned| t.allocator.free(owned);
        if (source != null) std.debug.print("unexpected structured admission: {s}\n", .{code});
        try t.expect(source == null);
    }
}

test "structured proof handles lexical generator and function scope without name leakage" {
    const accepted = [_][]const u8{
        "x = 7\ndef f(n):\n    return x + n\nprint(f(2))",
        "print(all(d > 0 for d in range(1, 4)))",
        "a = [1, True, 3]\nprint(sum(a), a[-2], len(a))",
        "x = 0\nfor n in range(5, 0, -2):\n    x += n\nprint(x)",
        "if False:\n    raise ValueError('unvisited')\nprint(1)",
        "x = 0\nfor n in range(10000):\n    x += 1\nprint(x)",
        "print((False and 5), (False or True), +True, -True)",
        "print('label .append and = : #', 3)",
        "def f(n):\n    if n < 2: return n\n    return n + 1\nprint(f(3))",
    };
    for (accepted) |code| {
        const source = (try blocks.lowerPython(t.allocator, code)) orelse return error.UnexpectedAbstention;
        defer t.allocator.free(source);
    }
    const source = try blocks.lowerPython(t.allocator, "print(all(d > 0 for d in range(1, 4)))\nprint(d)");
    defer if (source) |owned| t.allocator.free(owned);
    try t.expect(source == null);
}

const reference =
    \\import sys
    \\from pathlib import Path
    \\sys.path.insert(0, str(Path(sys.argv[1]) / 'experimental/riscv'))
    \\from chipformer.rv import elf, ref
    \\machine = ref.Machine.from_program(elf.program(sys.argv[2]))
    \\tokens = 1
    \\while not machine.halted and tokens < 4000000:
    \\    tokens += len(machine.record())
    \\assert tokens <= 4000000, tokens
    \\assert machine.halted and machine.fault is None and machine.exit_code == 0, (
    \\    machine.halted, machine.fault, machine.exit_code)
    \\sys.stdout.buffer.write(bytes(machine.out))
;

fn checkReference(env: *const std.process.Environ.Map, bytes: []const u8, expected: []const u8) !void {
    const root = env.get("TF_RUNNER_QWENPUTER") orelse return error.SkipZigTest;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const file = try tmp.dir.createFile(t.io, "blocks.elf", .{});
    defer file.close(t.io);
    try file.writeStreamingAll(t.io, bytes);
    const path = try tmp.dir.realPathFileAlloc(t.io, "blocks.elf", t.allocator);
    defer t.allocator.free(path);
    const result = try std.process.run(t.allocator, t.io, .{
        .argv = &.{ env.get("TF_RUNNER_REFERENCE_PYTHON") orelse "/usr/bin/python3", "-c", reference, root, path },
        .stdout_limit = .limited(4096),
        .stderr_limit = .limited(16384),
        .timeout = .{ .duration = .{ .raw = .fromSeconds(30), .clock = .awake } },
    });
    defer t.allocator.free(result.stdout);
    defer t.allocator.free(result.stderr);
    if (!result.term.success()) {
        std.debug.print("structured RV32 reference failed: {s}\n", .{result.stderr});
        return error.ReferenceFailed;
    }
    try t.expectEqualStrings(expected, result.stdout);
}

fn config(env: *const std.process.Environ.Map) !compiler.Config {
    const clang = env.get("TF_RUNNER_CLANG") orelse return error.SkipZigTest;
    return .{ .clang = clang, .linker = env.get("TF_RUNNER_LLD"), .stack_limit = 4096, .opaque_inputs = true, .step_budget = 0 };
}

test "configured structured cells execute retained operations on the RV32 reference" {
    var env = try t.environ.createMap(t.allocator);
    defer env.deinit();
    const settings = try config(&env);
    for (examples) |example| {
        const source = (try blocks.lowerPython(t.allocator, example.code)) orelse return error.UnexpectedAbstention;
        defer t.allocator.free(source);
        const bytes = (try compiler.compile(t.allocator, t.io, source, settings)) orelse return error.CompilationFailed;
        defer t.allocator.free(bytes);
        try t.expect(compiler.validElf(bytes));
        if (env.get("TF_RUNNER_QWENPUTER") != null) try checkReference(&env, bytes, example.expected);
    }
}

test "configured public frontend dispatches actual prime and factorial cells" {
    var env = try t.environ.createMap(t.allocator);
    defer env.deinit();
    var context: frontend.Context = .{ .io = t.io, .compiler = try config(&env) };
    const entry = context.frontend();
    for (examples[0..3]) |example| {
        const result = try entry.translate(entry.ctx, t.allocator, .{ .form = .repl, .lang = "python", .code = example.code });
        try t.expect(result == .program);
        defer t.allocator.free(result.program.bytes);
        try t.expectEqual(.elf, result.program.kind);
        if (env.get("TF_RUNNER_QWENPUTER") != null) try checkReference(&env, result.program.bytes, example.expected);
    }
}

const scope_cases = [_]struct { code: []const u8, expected: []const u8 }{
    .{
        .code =
        \\d = 99
        \\def probe(d):
        \\    ok = all(d > 0 for d in range(1, 4))
        \\    return ok and d == 7
        \\print(probe(7), probe(8), d)
        ,
        .expected = "True False 99\n",
    },
    .{
        .code =
        \\def fail(n):
        \\    return 1 // n
        \\print(False and fail(0), True or fail(0), not (False and fail(0)))
        \\print(False and (1 // 0), True or (1 // 0), not (True or (1 // 0)))
        \\print(0 and (1 // 0), 7 or (1 // 0), True and 5, False or 3, not 0, not 7)
        \\print(True and False, False or True, not not True)
        ,
        .expected = "False True True\nFalse True False\n0 7 5 3 True False\nFalse True True\n",
    },
    .{
        .code =
        \\d = 91
        \\def nested(d):
        \\    ok = all(all(d < 3 for d in range(1, d)) for d in range(1, 4))
        \\    return ok and d == 7
        \\print(nested(7), nested(8), d)
        \\print(all(all(d < 3 for d in range(1, d)) for d in range(1, 4)), d)
        ,
        .expected = "True False 91\nTrue 91\n",
    },
};

test "configured RV32 preserves generator scopes and boolean short circuit values" {
    var env = try t.environ.createMap(t.allocator);
    defer env.deinit();
    const settings = try config(&env);
    if (env.get("TF_RUNNER_QWENPUTER") == null) return error.SkipZigTest;
    for (scope_cases) |example| {
        const source = (try blocks.lowerPython(t.allocator, example.code)) orelse return error.UnexpectedAbstention;
        defer t.allocator.free(source);
        const bytes = (try compiler.compile(t.allocator, t.io, source, settings)) orelse return error.CompilationFailed;
        defer t.allocator.free(bytes);
        try checkReference(&env, bytes, example.expected);
    }
}

test "configured RV32 nested singleton loops preserve twenty seven full global lists" {
    var env = try t.environ.createMap(t.allocator);
    defer env.deinit();
    const settings = try config(&env);
    if (env.get("TF_RUNNER_QWENPUTER") == null) return error.SkipZigTest;
    var code: std.ArrayList(u8) = .empty;
    defer code.deinit(t.allocator);
    for (0..27) |i| try code.print(t.allocator, "a{d} = []\n", .{i});
    try code.appendSlice(t.allocator, "for n in range(128):\n");
    for (0..27) |i| try code.print(t.allocator, "    a{d}.append(1)\n", .{i});
    try code.appendSlice(t.allocator,
        \\def f():
        \\    for v in [0]:
        \\        for v in [0]:
        \\            for v in [0]:
        \\                for v in [0]:
        \\                    for v in [0]:
        \\                        for v in [0]:
        \\                            return 0
        \\print(f(),
    );
    for (0..27) |i| {
        if (i != 0) try code.appendSlice(t.allocator, ", ");
        try code.print(t.allocator, "sum(a{d})", .{i});
    }
    try code.appendSlice(t.allocator, ")\n");
    var expected: std.ArrayList(u8) = .empty;
    defer expected.deinit(t.allocator);
    try expected.appendSlice(t.allocator, "0");
    for (0..27) |_| try expected.appendSlice(t.allocator, " 128");
    try expected.appendSlice(t.allocator, "\n");
    const source = (try blocks.lowerPython(t.allocator, code.items)) orelse return error.UnexpectedAbstention;
    defer t.allocator.free(source);
    const bytes = (try compiler.compile(t.allocator, t.io, source, settings)) orelse return error.CompilationFailed;
    defer t.allocator.free(bytes);
    try checkReference(&env, bytes, expected.items);
}

test "configured compiler stack guard refuses a real oversized frame with disabled guard countercheck" {
    var env = try t.environ.createMap(t.allocator);
    defer env.deinit();
    var settings = try config(&env);
    const source = "volatile unsigned int at=4999; int main(void) { volatile char buf[5000]; buf[at] = 0; return buf[at]; }\n";
    const guarded = try compiler.compile(t.allocator, t.io, source, settings);
    defer if (guarded) |bytes| t.allocator.free(bytes);
    try t.expect(guarded == null);
    settings.stack_limit = 0;
    const unguarded = (try compiler.compile(t.allocator, t.io, source, settings)) orelse return error.CompilationFailed;
    defer t.allocator.free(unguarded);
    try t.expect(compiler.validElf(unguarded));
}
