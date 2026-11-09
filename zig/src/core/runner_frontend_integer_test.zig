const std = @import("std");
const api = @import("runner_api.zig");
const f = @import("runner_frontend.zig");
const t = std.testing;

const Case = struct {
    code: []const u8,
    expected: []const u8,
    form: api.Form = .repl,
};

fn repeat(comptime c: u8, comptime n: usize) [n]u8 {
    return @splat(c);
}

const ten31 = "1" ++ repeat('0', 31);
const ten61 = "1" ++ repeat('0', 61);
const ten62 = "1" ++ repeat('0', 62);

const arithmetic = [_]Case{
    .{ .code = "123456789012345*987654321098765", .expected = "121932631137021071359549253925\n" },
    .{ .code = "print(-123456789012345*987654321098765)", .expected = "-121932631137021071359549253925\n", .form = .fence },
    .{ .code = "10**30+7", .expected = "1000000000000000000000000000007\n" },
    .{ .code = "7-10**30", .expected = "-999999999999999999999999999993\n" },
    .{ .code = "-(10**30)", .expected = "-1000000000000000000000000000000\n" },
    .{ .code = "+7", .expected = "7\n" },
    .{ .code = "(-7)*(-3)", .expected = "21\n" },
    .{ .code = "-2**2", .expected = "-4\n" },
    .{ .code = "(-2)**3", .expected = "-8\n" },
    .{ .code = "2+3*4", .expected = "14\n" },
    .{ .code = "(2+3)*4", .expected = "20\n" },
    .{ .code = "(10**61+1)//10**30", .expected = ten31 ++ "\n" },
    .{ .code = "(10**61+1)%10**30", .expected = "1\n" },
    .{ .code = repeat('9', 62) ++ "//10", .expected = repeat('9', 61) ++ "\n" },
    .{ .code = repeat('9', 62) ++ "%10", .expected = "9\n" },
    .{ .code = "7//3", .expected = "2\n" },
    .{ .code = "-7//3", .expected = "-3\n" },
    .{ .code = "7//-3", .expected = "-3\n" },
    .{ .code = "-7//-3", .expected = "2\n" },
    .{ .code = "-7%3", .expected = "2\n" },
    .{ .code = "7%-3", .expected = "-2\n" },
    .{ .code = "-7%-3", .expected = "-1\n" },
    .{ .code = "7%3", .expected = "1\n" },
    .{ .code = "divmod(7,3)", .expected = "(2, 1)\n" },
    .{ .code = "divmod(-7,3)", .expected = "(-3, 2)\n" },
    .{ .code = "print(divmod(7,-3))", .expected = "(-3, -2)\n", .form = .fence },
    .{ .code = "divmod(-7,-3)", .expected = "(2, -1)\n" },
    .{ .code = "divmod(10**61+1,10**30)", .expected = "(" ++ ten31 ++ ", 1)\n" },
    .{ .code = "divmod(-10**61,10**30)", .expected = "(-" ++ ten31 ++ ", 0)\n" },
};

const modular = [_]Case{
    .{ .code = "pow(2,1000,1000)", .expected = "376\n" },
    .{ .code = "print(pow(2,1000000,3))", .expected = "1\n", .form = .fence },
    .{ .code = "pow(2,1000000,-3)", .expected = "-2\n" },
    .{ .code = "pow(-2,7,5)", .expected = "2\n" },
    .{ .code = "pow(7,0,97)", .expected = "1\n" },
    .{ .code = "pow(7,0,1)", .expected = "0\n" },
    .{ .code = "pow(10**40,2,10**61)", .expected = "0\n" },
    .{ .code = "2**1000%1000", .expected = "376\n" },
    .{ .code = "(2**1000000)%3", .expected = "1\n" },
    .{ .code = "2**1000000%-3", .expected = "-2\n" },
    .{ .code = "(10**40)**2%10**61", .expected = "0\n" },
    .{ .code = "pow(10**40+1,2,10**61)", .expected = "2" ++ repeat('0', 39) ++ "1\n" },
};

const conversions = [_]Case{
    .{ .code = "0xff+0b101", .expected = "260\n" },
    .{ .code = "0x10000000000000000", .expected = "18446744073709551616\n" },
    .{ .code = "0b1" ++ repeat('0', 64), .expected = "18446744073709551616\n" },
    .{ .code = "int('ff',16)", .expected = "255\n" },
    .{ .code = "int('101',2)", .expected = "5\n" },
    .{ .code = "print(int('10000000000000000',16))", .expected = "18446744073709551616\n", .form = .fence },
    .{ .code = "int('1" ++ repeat('0', 64) ++ "',2)", .expected = "18446744073709551616\n" },
    .{ .code = "len(str(0))", .expected = "1\n" },
    .{ .code = "len(str(10**30))", .expected = "31\n" },
    .{ .code = "len(str(-10**30))", .expected = "32\n" },
};

const boundaries = [_]Case{
    .{ .code = ten61, .expected = ten61 ++ "\n" },
    .{ .code = "10**61", .expected = ten61 ++ "\n" },
    .{ .code = "10**30*10**31", .expected = ten61 ++ "\n" },
    .{ .code = "10**61+1", .expected = "1" ++ repeat('0', 60) ++ "1\n" },
    .{ .code = "-" ++ repeat('9', 62), .expected = "-" ++ repeat('9', 62) ++ "\n" },
};

const inline_cases = [_]Case{
    .{ .code = "7%3", .expected = "1\n", .form = .inline_expr },
    .{ .code = "7%(-3)", .expected = "-2\n", .form = .inline_expr },
    .{ .code = "(-7)%(-3)", .expected = "-1\n", .form = .inline_expr },
    .{ .code = "50%*3", .expected = "1.5\n", .form = .inline_expr },
    .{ .code = "25%+5%", .expected = "0.3\n", .form = .inline_expr },
    .{ .code = "0.125*2", .expected = "0.25\n", .form = .inline_expr },
    .{ .code = "1/3+2/3", .expected = "1\n", .form = .inline_expr },
    .{ .code = "11/18", .expected = "11/18\n", .form = .inline_expr },
    .{ .code = "gcd(48,18)", .expected = "6\n", .form = .inline_expr },
};

fn configured(clang: []const u8, linker: ?[]const u8) f.Context {
    return .{ .io = t.io, .compiler = .{ .clang = clang, .linker = linker, .step_budget = 0 } };
}

fn translate(context: *f.Context, case: Case) !api.Program {
    const front = context.frontend();
    const result = try front.translate(front.ctx, t.allocator, .{
        .form = case.form,
        .lang = "python",
        .code = case.code,
    });
    if (result == .abstain) {
        std.debug.print("integer frontend abstained: {s}\n", .{case.code});
        return error.UnexpectedAbstention;
    }
    return result.program;
}

fn accepts(context: *f.Context, cases: []const Case) !void {
    for (cases) |case| {
        const program = try translate(context, case);
        defer t.allocator.free(program.bytes);
    }
}

fn abstains(context: *f.Context, cell: api.Cell) !void {
    const front = context.frontend();
    const result = try front.translate(front.ctx, t.allocator, cell);
    if (result == .program) {
        defer t.allocator.free(result.program.bytes);
        std.debug.print("integer frontend unexpectedly accepted: {s}\n", .{cell.code});
    }
    try t.expect(result == .abstain);
}

fn requiredClang(env: *const std.process.Environ.Map) ![]const u8 {
    return env.get("TF_RUNNER_CLANG") orelse {
        std.debug.print("SKIP integer ELF admission: set TF_RUNNER_CLANG and optional TF_RUNNER_LLD\n", .{});
        return error.SkipZigTest;
    };
}

test "integer graphs retain supported arithmetic without requiring a host toolchain" {
    for (arithmetic ++ modular ++ conversions ++ boundaries) |case| {
        if (std.mem.eql(u8, case.code, "+7")) continue;
        const source = try @import("runner_frontend_integer.zig").lower(t.allocator, case.code);
        try t.expect(source != null);
        defer t.allocator.free(source.?);
        try t.expect(std.mem.indexOf(u8, source.?, "tf_run(tf_nodes,") != null);
    }
    for ([_][]const u8{ "10**62", "10**31*10**31", "divmod(1,0)", "pow(2,-1,3)", "pow(2,3,0)", "len(str(10**62))", "1.25*2", "divmod(7,3)+1" }) |code| {
        const source = try @import("runner_frontend_integer.zig").lower(t.allocator, code);
        defer if (source) |owned| t.allocator.free(owned);
        try t.expect(source == null);
    }
}

test "integer frontend accepts wide arithmetic and Python signed divmod" {
    var env = try t.environ.createMap(t.allocator);
    defer env.deinit();
    var context = configured(try requiredClang(&env), env.get("TF_RUNNER_LLD"));
    try accepts(&context, &arithmetic);
}

test "integer frontend accepts modular powers whose unreduced result exceeds its bound" {
    var env = try t.environ.createMap(t.allocator);
    defer env.deinit();
    var context = configured(try requiredClang(&env), env.get("TF_RUNNER_LLD"));
    try accepts(&context, &modular);
}

test "integer frontend accepts binary hex and decimal string lengths" {
    var env = try t.environ.createMap(t.allocator);
    defer env.deinit();
    var context = configured(try requiredClang(&env), env.get("TF_RUNNER_LLD"));
    try accepts(&context, &conversions);
}

test "integer frontend enforces the 62 digit input and result boundary" {
    var env = try t.environ.createMap(t.allocator);
    defer env.deinit();
    var context = configured(try requiredClang(&env), env.get("TF_RUNNER_LLD"));
    try accepts(&context, &boundaries);
    for ([_]api.Form{ .inline_expr, .repl, .fence }) |form| {
        for ([_][]const u8{
            ten62, "-" ++ ten62, "10**62", "10**31*10**31", "10**61*10", "0x1" ++ repeat('0', 52),
        }) |code| try abstains(&context, .{ .form = form, .lang = "python", .code = code });
    }
}

test "integer frontend refuses unsafe ambiguous noninteger and incomplete Python code" {
    var env = try t.environ.createMap(t.allocator);
    defer env.deinit();
    var context = configured(env.get("TF_RUNNER_CLANG") orelse "/usr/bin/clang", env.get("TF_RUNNER_LLD"));
    const bad = [_][]const u8{
        "",                             "1+",                     "(2+3",        "2**",          "divmod(7,)", "pow(2,3,)",                 "int('ff',)",
        "missing+1",                    "print(missing)",         "import os",   "1.25*2",       "1e3+1",      "1/2",                       "6/3",
        "1//0",                         "1%0",                    "divmod(1,0)", "pow(2,3,0)",   "2**-1",      "pow(2,-1,3)",               "1,2",
        "(1,2)",                        "2*3,18",                 "1<2<3",       "1+2=3",        "1==1==1",    "__import__('os').getcwd()", "(1).__class__",
        "int('__import__(\"os\")',16)", "int('ff',16).__class__", "int('2',2)",  "int('gg',16)",
    };
    for ([_]api.Form{ .repl, .fence }) |form| {
        for (bad) |code| try abstains(&context, .{ .form = form, .lang = "python", .code = code });
    }
    for ([_][]const u8{
        "1+",                        "(2+3",                         "missing+1", "1,2", "1<2<3", "1+2=3", "2**-1", "1%0",
        "__import__('os').getcwd()", "int('__import__(\"os\")',16)",
    }) |code| try abstains(&context, .{ .form = .inline_expr, .lang = "python", .code = code });
}

test "configured integer frontend prints tuple and list representations exactly like CPython" {
    var env = try t.environ.createMap(t.allocator);
    defer env.deinit();
    var context = configured(try requiredClang(&env), env.get("TF_RUNNER_LLD"));
    for ([_]api.Form{ .repl, .fence }) |form| {
        for ([_][]const u8{ "print((1,2))", "print([1,2])" }) |code| {
            const program = try translate(&context, .{ .form = form, .code = code, .expected = "" });
            defer t.allocator.free(program.bytes);
            try @import("runner_frontend_collections_compat_test.zig").checkExact(&env, program, code);
        }
    }
}

test "integer frontend refuses prior cells that can shadow integer builtins" {
    var env = try t.environ.createMap(t.allocator);
    defer env.deinit();
    var context = configured(env.get("TF_RUNNER_CLANG") orelse "/usr/bin/clang", env.get("TF_RUNNER_LLD"));
    for ([_]api.Form{ .repl, .fence }) |form| {
        for ([_][2][]const u8{
            .{ "pow = lambda a,b,c: 0", "pow(2,1000,1000)" },
            .{ "divmod = lambda a,b: (0,0)", "divmod(-7,3)" },
            .{ "int = lambda a,b: 0", "int('ff',16)" },
            .{ "str = lambda a: ''", "len(str(-10**30))" },
            .{ "print = lambda a: None", "print(7*3)" },
        }) |case| try abstains(&context, .{
            .form = form,
            .lang = "python",
            .code = case[1],
            .prior = &.{case[0]},
        });
    }
}

test "integer frontend preserves inline percentages and decimals with EXACT on or off" {
    var env = try t.environ.createMap(t.allocator);
    defer env.deinit();
    var context = configured(try requiredClang(&env), env.get("TF_RUNNER_LLD"));
    try accepts(&context, &inline_cases);
    for ([_]bool{ true, false }) |exact| {
        context.exact_inline = exact;
        for ([_][]const u8{ "7%-3", "-7%-3", "25%+1" }) |code|
            try abstains(&context, .{ .form = .inline_expr, .lang = "python", .code = code });
    }
    context.exact_inline = false;
    context.exact_tags = false;
    try accepts(&context, inline_cases[0..3]);
    for (arithmetic ++ modular ++ conversions ++ boundaries) |case| {
        var inline_case = case;
        if (std.mem.startsWith(u8, case.code, "print(")) continue;
        if (std.mem.startsWith(u8, case.code, "+")) continue;
        if (std.mem.indexOf(u8, case.code, "%-") != null) continue;
        inline_case.form = .inline_expr;
        const program = try translate(&context, inline_case);
        defer t.allocator.free(program.bytes);
    }
    for ([_][]const u8{ "1/2", "6/3", "0.125*2", "1.0+2", "50%*3", "25%+5%", "11/18" }) |code|
        try abstains(&context, .{ .form = .inline_expr, .lang = "python", .code = code });
}

const reference =
    \\import sys
    \\from pathlib import Path
    \\root = Path(sys.argv[1])
    \\sys.path.insert(0, str(root / 'experimental/riscv'))
    \\from chipformer.rv import calc, elf, ref
    \\if sys.argv[2] == 'calc':
    \\    machine = calc.machine(sys.argv[3], root / 'experimental/riscv/tests/data/rv/calc.elf')
    \\else:
    \\    machine = ref.Machine.from_program(elf.program(sys.argv[3]))
    \\machine.run(5000000)
    \\assert machine.halted and machine.fault is None and machine.exit_code == 0, (
    \\    machine.halted, machine.fault, machine.exit_code)
    \\sys.stdout.buffer.write(bytes(machine.out))
;

fn check(root: []const u8, python: []const u8, program: api.Program, expected: []const u8) !void {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var path: ?[:0]u8 = null;
    defer if (path) |value| t.allocator.free(value);
    var input = program.bytes;
    if (program.kind == .elf) {
        const file = try tmp.dir.createFile(t.io, "integer.elf", .{});
        defer file.close(t.io);
        try file.writeStreamingAll(t.io, program.bytes);
        path = try tmp.dir.realPathFileAlloc(t.io, "integer.elf", t.allocator);
        input = path.?;
    }
    const result = try std.process.run(t.allocator, t.io, .{
        .argv = &.{ python, "-c", reference, root, @tagName(program.kind), input },
        .stdout_limit = .limited(4096),
        .stderr_limit = .limited(16384),
        .timeout = .{ .duration = .{ .raw = .fromSeconds(30), .clock = .awake } },
    });
    defer t.allocator.free(result.stdout);
    defer t.allocator.free(result.stderr);
    if (!result.term.success()) {
        std.debug.print("integer RV32 reference failed: {s}\n", .{result.stderr});
        return error.ReferenceFailed;
    }
    try t.expectEqualStrings(expected, result.stdout);
}

test "integer frontend programs produce exact output in the real RV32 reference" {
    var env = try t.environ.createMap(t.allocator);
    defer env.deinit();
    const root = env.get("TF_RUNNER_QWENPUTER") orelse {
        std.debug.print("SKIP integer RV32 reference: set TF_RUNNER_QWENPUTER to its source checkout\n", .{});
        return error.SkipZigTest;
    };
    var context = configured(env.get("TF_RUNNER_CLANG") orelse "/usr/bin/clang", env.get("TF_RUNNER_LLD"));
    const python = env.get("TF_RUNNER_REFERENCE_PYTHON") orelse "/usr/bin/python3";
    for (arithmetic ++ modular ++ conversions ++ boundaries ++ inline_cases) |case| {
        const program = try translate(&context, case);
        defer t.allocator.free(program.bytes);
        try check(root, python, program, case.expected);
    }
    context.exact_inline = false;
    context.exact_tags = false;
    for ([_]Case{
        .{ .code = "7%3", .expected = "1\n", .form = .inline_expr },
        .{ .code = "7%(-3)", .expected = "-2\n", .form = .inline_expr },
        .{ .code = "(-7)%(-3)", .expected = "-1\n", .form = .inline_expr },
        .{ .code = "pow(2,1000,1000)", .expected = "376\n", .form = .inline_expr },
        .{ .code = "divmod(-7,3)", .expected = "(-3, 2)\n", .form = .inline_expr },
        .{ .code = "len(str(-10**30))", .expected = "32\n", .form = .inline_expr },
    }) |case| {
        const program = try translate(&context, case);
        defer t.allocator.free(program.bytes);
        try check(root, python, program, case.expected);
    }
}
