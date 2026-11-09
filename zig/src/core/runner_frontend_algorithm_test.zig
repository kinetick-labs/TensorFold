const std = @import("std");
const ast = @import("runner_frontend_integer_ast.zig");
const integer = @import("runner_frontend_integer.zig");
const compiler = @import("runner_frontend_program_compile.zig");
const t = std.testing;

const Case = struct { code: []const u8, expected: []const u8, op: ast.Op };
const cases = [_]Case{
    .{ .code = "gcd(54,24)", .expected = "6\n", .op = .gcd },
    .{ .code = "math.gcd(-54,24)", .expected = "6\n", .op = .gcd },
    .{ .code = "gcd(0,0)", .expected = "0\n", .op = .gcd },
    .{ .code = "gcd(0,-24)", .expected = "24\n", .op = .gcd },
    .{ .code = "gcd(-54,-24)", .expected = "6\n", .op = .gcd },
    .{ .code = "math.gcd(10**61-1,10**30-1)", .expected = "9\n", .op = .gcd },
    .{ .code = "isqrt(0)", .expected = "0\n", .op = .isqrt },
    .{ .code = "math.isqrt(1)", .expected = "1\n", .op = .isqrt },
    .{ .code = "isqrt(2)", .expected = "1\n", .op = .isqrt },
    .{ .code = "isqrt(144)", .expected = "12\n", .op = .isqrt },
    .{ .code = "isqrt(143)", .expected = "11\n", .op = .isqrt },
    .{ .code = "isqrt(145)", .expected = "12\n", .op = .isqrt },
    .{ .code = "math.isqrt(10**60-1)", .expected = "999999999999999999999999999999\n", .op = .isqrt },
    .{ .code = "isqrt(10**60)", .expected = "1000000000000000000000000000000\n", .op = .isqrt },
    .{ .code = "isqrt(10**60+1)", .expected = "1000000000000000000000000000000\n", .op = .isqrt },
    .{ .code = "factorial(0)", .expected = "1\n", .op = .factorial },
    .{ .code = "math.factorial(1)", .expected = "1\n", .op = .factorial },
    .{ .code = "factorial(10)", .expected = "3628800\n", .op = .factorial },
    .{ .code = "factorial(48)", .expected = "12413915592536072670862289047373375038521486354677760000000000\n", .op = .factorial },
    .{ .code = "prod(range(1,11))", .expected = "3628800\n", .op = .range_product },
    .{ .code = "prod(range(10))", .expected = "0\n", .op = .range_product },
    .{ .code = "prod(range(0))", .expected = "1\n", .op = .range_product },
    .{ .code = "math.prod(range(10,0,-2))", .expected = "3840\n", .op = .range_product },
    .{ .code = "prod(range(-5,0,2))", .expected = "-15\n", .op = .range_product },
    .{ .code = "prod(range(-4,0))", .expected = "24\n", .op = .range_product },
    .{ .code = "prod(range(0,10000))", .expected = "0\n", .op = .range_product },
    .{ .code = "prod(range(0,-10000,-1))", .expected = "0\n", .op = .range_product },
    .{ .code = "prod(range(1,1))", .expected = "1\n", .op = .range_product },
    .{ .code = "prod(range(0,10,-1))", .expected = "1\n", .op = .range_product },
    .{ .code = "prod(range(10,0))", .expected = "1\n", .op = .range_product },
    .{ .code = "prod(range(10**61,10**61+1))", .expected = "10000000000000000000000000000000000000000000000000000000000000\n", .op = .range_product },
    .{ .code = "sum(map(int,str(0)))", .expected = "0\n", .op = .digit_sum },
    .{ .code = "sum(map(int,str(123456789)))", .expected = "45\n", .op = .digit_sum },
    .{ .code = "sum(int(d) for d in str(123456789))", .expected = "45\n", .op = .digit_sum },
    .{ .code = "sum(int(digit) for digit in str(10**61-1))", .expected = "549\n", .op = .digit_sum },
    .{ .code = "sum ( map ( int , str ( 9001 ) ) )", .expected = "10\n", .op = .digit_sum },
};

const rejected = [_][]const u8{
    "isqrt(-1)",                                     "factorial(-1)",               "factorial(49)",                     "factorial(10**61)",
    "gcd(10**62,1)",                                 "gcd(1)",                      "gcd(1,2,3)",                        "math.lcm(2,3)",
    "isqrt(1,2)",                                    "math..gcd(1,2)",              "mathgcd(1,2)",                      "prod(range())",
    "sum(int(__debug__) for __debug__ in str(123))", "prod(range(0,10001))",        "prod(range(0,-10001,-1))",          "prod(range(1,5,0))",
    "prod(range(1,50))",                             "prod([1,2,3])",               "prod(range(1,5),start=2)",          "sum(map(int,str(-123)))",
    "sum(int(d) for d in str(-123))",                "digit_sum(123)",              "sum(int(d) for e in str(123))",     "sum(int(d) for d in str(abs(-123)))",
    "sum(int(d)ford in str(123))",                   "sum(int(d) for dinstr(123))", "sum(int(for) for for in str(123))", "sum(int(int) for int in str(123))",
    "sum(int(d) for d in str(123) if d)",            "sum(map(int,str(10**62)))",   "sum(map(float,str(123)))",          "sum(map(int,str(123)),1)",
};

test "algorithm AST proves exact Python forms and rejects invalid or unbounded operations" {
    for (cases) |case| {
        var parser = ast.Parser{ .source = case.code };
        const root = try parser.parse();
        try t.expectEqual(case.op, parser.nodes[root].op);
        const expected = try std.fmt.parseInt(i512, std.mem.trim(u8, case.expected, "\n"), 10);
        try t.expectEqual(expected, parser.nodes[root].value);
    }
    for (rejected) |code| {
        var parser = ast.Parser{ .source = code };
        try t.expectError(error.Unsupported, parser.parse());
    }
}

test "algorithm emitter retains operations and visits only reachable children with the correct arity" {
    for (cases) |case| {
        const source = (try integer.lower(t.allocator, case.code)) orelse return error.UnexpectedAbstention;
        defer t.allocator.free(source);
        try t.expect(std.mem.indexOf(u8, source, "tf_run(tf_nodes,") != null);
    }
    const rows = [_][2][]const u8{
        .{ "isqrt(144)", "{0,0,0,10,\"144\"},\n{12,0,0,0,0},\n};\nint main(void) { return tf_run(tf_nodes, 2); }" },
        .{ "gcd(54,24)", "{11,0,1,0,0},\n};\nint main(void) { return tf_run(tf_nodes, 3); }" },
        .{ "prod(range(1,5))", "{14,0,1,2,0},\n};\nint main(void) { return tf_run(tf_nodes, 4); }" },
        .{ "isqrt(2**1000%1000)", "{8,0,1,2,0},\n{12,3,0,0,0},\n};\nint main(void) { return tf_run(tf_nodes, 5); }" },
    };
    for (rows) |row| {
        const source = (try integer.lower(t.allocator, row[0])) orelse return error.UnexpectedAbstention;
        defer t.allocator.free(source);
        try t.expect(std.mem.indexOf(u8, source, row[1]) != null);
    }
    for (rejected) |code| {
        const source = try integer.lower(t.allocator, code);
        defer if (source) |owned| t.allocator.free(owned);
        try t.expect(source == null);
    }
    try t.expectEqual(@as(u8, 10), @backingInt(ast.Op.divmod));
    try t.expectEqual(@as(u8, 11), @backingInt(ast.Op.gcd));
    try t.expectEqual(@as(u8, 15), @backingInt(ast.Op.digit_sum));
}

test "persistent AST bindings retain prior roots across reassignment and prune dead assignments" {
    var parser = ast.Parser{ .source = "314159" };
    const first = try parser.parse();
    try parser.bind("a", first);
    parser.source = "271828";
    parser.pos = 0;
    const second = try parser.parse();
    try parser.bind("b", second);
    parser.source = "a*b";
    parser.pos = 0;
    const product = try parser.parse();
    try t.expectEqual(@as(i512, 85397212652), parser.nodes[product].value);
    try t.expectEqual(first, parser.nodes[product].l);
    try t.expectEqual(second, parser.nodes[product].r);
    try parser.bind("a", product);
    parser.source = "a+1";
    parser.pos = 0;
    const incremented = try parser.parse();
    try t.expectEqual(product, parser.nodes[incremented].l);
    try parser.bind("a", incremented);
    parser.source = "99999";
    parser.pos = 0;
    try parser.bind("unused", try parser.parse());
    const source = try integer.lowerParser(t.allocator, &parser, incremented);
    defer t.allocator.free(source);
    const table_start = std.mem.indexOf(u8, source, "static const struct tf_node tf_nodes[]") orelse return error.MissingTable;
    const table = source[table_start..];
    try t.expect(std.mem.indexOf(u8, table, "99999") == null);
    try t.expect(std.mem.indexOf(u8, table, "85397212652") == null);
    try t.expect(std.mem.indexOf(u8, table, "{4,0,1,0,0}") != null);
    try t.expectEqual(first, parser.nodes[product].l);
    parser.source = "missing+1";
    parser.pos = 0;
    try t.expectError(error.Unsupported, parser.parse());
    parser.source = "isqrt(-1)";
    parser.pos = 0;
    try t.expectError(error.Unsupported, parser.parse());
}

test "math import policy distinguishes module and function bindings" {
    const names = [_][]const u8{ "gcd(54,24)", "isqrt(144)", "factorial(10)", "prod(range(1,5))" };
    for (names, 0..) |name, index| {
        const qualified = try std.fmt.allocPrint(t.allocator, "math.{s}", .{name});
        defer t.allocator.free(qualified);
        for ([_]bool{ false, true }) |module| {
            var parser = ast.Parser{ .source = qualified, .require_math_import = true, .math_module = module };
            if (module) {
                _ = try parser.parse();
            } else try t.expectError(error.Unsupported, parser.parse());
        }
        for ([_]u8{ 0, 1, 2, 4, 8, 15 }) |imports| {
            var parser = ast.Parser{ .source = name, .require_math_import = true, .math_imports = imports };
            if (imports & (@as(u8, 1) << @intCast(index)) != 0) {
                _ = try parser.parse();
            } else try t.expectError(error.Unsupported, parser.parse());
        }
    }
}

fn configured(env: *const std.process.Environ.Map) !compiler.Config {
    const clang = env.get("TF_RUNNER_CLANG") orelse {
        std.debug.print("SKIP algorithm ELF: set TF_RUNNER_CLANG and optional TF_RUNNER_LLD\n", .{});
        return error.SkipZigTest;
    };
    return .{ .clang = clang, .linker = env.get("TF_RUNNER_LLD"), .optimize = .O2, .opaque_inputs = true, .step_budget = 0 };
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
    \\assert machine.halted and machine.fault is None and machine.exit_code == int(sys.argv[3]), (
    \\    machine.halted, machine.fault, machine.exit_code)
    \\sys.stdout.buffer.write(bytes(machine.out))
;

fn check(env: *const std.process.Environ.Map, bytes: []const u8, expected: []const u8, exit: []const u8) !void {
    const root = env.get("TF_RUNNER_QWENPUTER") orelse return error.SkipZigTest;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const file = try tmp.dir.createFile(t.io, "algorithm.elf", .{});
    defer file.close(t.io);
    try file.writeStreamingAll(t.io, bytes);
    const path = try tmp.dir.realPathFileAlloc(t.io, "algorithm.elf", t.allocator);
    defer t.allocator.free(path);
    const result = try std.process.run(t.allocator, t.io, .{
        .argv = &.{ env.get("TF_RUNNER_REFERENCE_PYTHON") orelse "/usr/bin/python3", "-c", reference, root, path, exit },
        .stdout_limit = .limited(4096),
        .stderr_limit = .limited(16384),
        .timeout = .{ .duration = .{ .raw = .fromSeconds(60), .clock = .awake } },
    });
    defer t.allocator.free(result.stdout);
    defer t.allocator.free(result.stderr);
    if (!result.term.success()) {
        std.debug.print("algorithm RV32 reference failed: {s}\n", .{result.stderr});
        return error.ReferenceFailed;
    }
    try t.expectEqualStrings(expected, result.stdout);
}

test "configured algorithm ELF executes actual RV32 arithmetic" {
    var env = try t.environ.createMap(t.allocator);
    defer env.deinit();
    const config = try configured(&env);
    for (cases) |case| {
        const bytes = (try integer.compile(t.allocator, t.io, case.code, config)) orelse return error.CompilationFailed;
        defer t.allocator.free(bytes);
        try t.expect(compiler.validElf(bytes));
        if (env.get("TF_RUNNER_QWENPUTER") != null) try check(&env, bytes, case.expected, "0");
    }
}

test "configured RV32 executes a persistent wide assignment graph" {
    var env = try t.environ.createMap(t.allocator);
    defer env.deinit();
    const config = try configured(&env);
    var parser = ast.Parser{ .source = "314159" };
    try parser.bind("a", try parser.parse());
    parser.source = "271828";
    parser.pos = 0;
    try parser.bind("b", try parser.parse());
    parser.source = "a*b";
    parser.pos = 0;
    const root = try parser.parse();
    const source = try integer.lowerParser(t.allocator, &parser, root);
    defer t.allocator.free(source);
    const bytes = (try compiler.compile(t.allocator, t.io, source, config)) orelse return error.CompilationFailed;
    defer t.allocator.free(bytes);
    try t.expect(compiler.validElf(bytes));
    if (env.get("TF_RUNNER_QWENPUTER") != null) try check(&env, bytes, "85397212652\n", "0");
}

fn mutate(source: []const u8, before: []const u8, after: []const u8) ![]u8 {
    const index = std.mem.indexOf(u8, source, before) orelse return error.MissingMutationTarget;
    return std.fmt.allocPrint(t.allocator, "{s}{s}{s}", .{ source[0..index], after, source[index + before.len ..] });
}

test "configured RV32 mutations change the proved answer or fail before output" {
    var env = try t.environ.createMap(t.allocator);
    defer env.deinit();
    const config = try configured(&env);
    if (env.get("TF_RUNNER_QWENPUTER") == null) return error.SkipZigTest;
    const controls = [_][5][]const u8{
        .{ "gcd(54,24)", "{11,0,1,0,0}", "{4,0,1,0,0}", "1296\n", "0" },
        .{ "factorial(48)", "{0,0,0,10,\"48\"}", "{0,0,0,10,\"49\"}", "", "1" },
        .{ "isqrt(144)", "{0,0,0,10,\"144\"}", "{0,0,0,10,\"-144\"}", "", "1" },
        .{ "prod(range(1,5))", "{14,0,1,2,0}", "{14,0,1,1,0}", "1\n", "0" },
        .{ "prod(range(0,5))", "{14,0,1,2,0}", "{14,0,1,0,0}", "", "1" },
        .{ "prod(range(0,10000))", "{0,0,0,10,\"10000\"}", "{0,0,0,10,\"10001\"}", "", "1" },
        .{ "sum(map(int,str(123)))", "{0,0,0,10,\"123\"}", "{0,0,0,10,\"-123\"}", "", "1" },
    };
    for (controls) |control| {
        const source = (try integer.lower(t.allocator, control[0])) orelse return error.UnexpectedAbstention;
        defer t.allocator.free(source);
        const changed = try mutate(source, control[1], control[2]);
        defer t.allocator.free(changed);
        const bytes = (try compiler.compile(t.allocator, t.io, changed, config)) orelse return error.CompilationFailed;
        defer t.allocator.free(bytes);
        try check(&env, bytes, control[3], control[4]);
    }
}
