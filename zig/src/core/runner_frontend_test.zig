const std = @import("std");
const f = @import("runner_frontend.zig");
const t = std.testing;

test {
    _ = @import("runner_frontend_integer_test.zig");
    _ = @import("runner_frontend_integer_python.zig");
    _ = @import("runner_frontend_algorithm_test.zig");
    _ = @import("runner_frontend_cwide_test.zig");
    _ = @import("runner_frontend_blocks_test.zig");
    _ = @import("runner_frontend_c_cost_test.zig");
    _ = @import("runner_frontend_c_inputs_test.zig");
    _ = @import("runner_frontend_integer_cost_test.zig");
    _ = @import("runner_frontend_budget_test.zig");
    _ = @import("runner_frontend_optimization_test.zig");
    _ = @import("runner_frontend_calc_cost_test.zig");
    _ = @import("runner_frontend_script_test.zig");
    _ = @import("runner_frontend_python_assert_test.zig");
    _ = @import("runner_frontend_tool_script_test.zig");
    _ = @import("runner_frontend_decimal_test.zig");
    _ = @import("runner_frontend_fraction_test.zig");
    _ = @import("runner_frontend_digit_reverse_test.zig");
    _ = @import("runner_frontend_elf_cache_test.zig");
}

test "complete integer assignment requests use firmware with original inputs and no compiler" {
    var context: f.Context = .{ .io = t.io, .compiler = .{ .clang = "/nonexistent-frontend-test-clang" } };
    const front = context.frontend();
    const code = "a=264278028996235349740985471434\nb=771565915835769397012014492734\nprint(a+b)";
    const result = try front.translate(front.ctx, t.allocator, .{ .form = .fence, .lang = "python", .code = code });
    try t.expect(result == .program);
    defer t.allocator.free(result.program.bytes);
    try t.expectEqual(.calc, result.program.kind);
    try t.expectEqualStrings("(264278028996235349740985471434+771565915835769397012014492734)", result.program.bytes);
    try t.expect(std.mem.indexOf(u8, result.program.bytes, "1035843944832004746752999964168") == null);
    for ([_][]const u8{
        "a=1//0\nprint(3)", "a=10**62\nprint(3)", "import os\nprint(3)",
        "a=1.25\nprint(a)", "a=1/2\nprint(a)",    "a=2\nprint(a)\nassert False",
    }) |bad| {
        const refused = try front.translate(front.ctx, t.allocator, .{ .form = .fence, .lang = "python", .code = bad });
        defer if (refused == .program) t.allocator.free(refused.program.bytes);
        try t.expect(refused == .abstain);
    }
}

test "frontend accepts model multiply and REPL power without computing the answer" {
    var exact: f.Context = .{ .io = t.io, .exact_tags = true };
    const cases = [_][3][]const u8{
        .{ "python", "print(123456789012345678901234567890 * 987654321098765432109876543210)", "123456789012345678901234567890*987654321098765432109876543210" },
        .{ "py", "2**100", "2**100" },
        .{ "python3", "7 // -3", "7//-3" },
        .{ "exact", "1/3 + 2/3", "1/3+2/3" },
        .{ "calc", "1.25*2.4", "1.25*2.4" },
    };
    for (cases) |case| {
        const front = if (std.mem.eql(u8, case[0], "calc") or std.mem.eql(u8, case[0], "exact"))
            exact.frontend()
        else
            f.frontend();
        const result = try front.translate(front.ctx, t.allocator, .{ .form = .repl, .lang = case[0], .code = case[1] });
        try t.expect(result == .program);
        defer t.allocator.free(result.program.bytes);
        try t.expectEqual(.calc, result.program.kind);
        try t.expectEqualStrings(case[2], result.program.bytes);
    }
}

test "frontend declines unsupported code and Python output mismatches" {
    var env = try t.environ.createMap(t.allocator);
    defer env.deinit();
    var context: f.Context = .{ .io = t.io, .compiler = .{
        .clang = env.get("TF_RUNNER_CLANG") orelse "/usr/bin/clang",
        .linker = env.get("TF_RUNNER_LLD"),
    } };
    const front = context.frontend();
    const cases = [_][2][]const u8{
        .{ "javascript", "2*3" },                  .{ "python", "print(missing)" },  .{ "python", "import os" },
        .{ "python", "print(1/3)" },               .{ "python", "print(1.25*2.4)" }, .{ "python", "gcd(2,4)" },
        .{ "python", "print(2); print(missing)" }, .{ "python", "10**100" },         .{ "python", "1//0" },
        .{ "exact", "1/3" },                       .{ "calc", "1.25*2.4" },
    };
    for (cases) |case| {
        const result = try front.translate(front.ctx, t.allocator, .{ .form = .fence, .lang = case[0], .code = case[1] });
        defer if (result == .program) t.allocator.free(result.program.bytes);
        if (result == .program) std.debug.print("unexpectedly accepted {s}: {s}\n", .{ case[0], case[1] });
        try t.expect(result == .abstain);
    }
}

test "frontend allocator failure is visible to its caller" {
    const front = f.frontend();
    try t.expectError(error.OutOfMemory, front.translate(null, std.testing.failing_allocator, .{
        .form = .repl,
        .lang = "python",
        .code = "2**100",
    }));
}

test {
    _ = @import("runner_frontend_calc.zig");
    _ = @import("runner_frontend_program.zig");
    _ = @import("runner_frontend_reference_test.zig");
    _ = @import("runner_frontend_coverage_test.zig");
    _ = @import("runner_frontend_notation_test.zig");
    _ = @import("runner_frontend_boot.zig");
}

test "real compiler emits circuit ELF for loops and C with negative controls" {
    var env = try t.environ.createMap(t.allocator);
    defer env.deinit();
    const clang = env.get("TF_RUNNER_CLANG") orelse {
        std.debug.print("SKIP RV32 toolchain: set TF_RUNNER_CLANG and optionally TF_RUNNER_LLD\n", .{});
        return error.SkipZigTest;
    };
    var context: f.Context = .{
        .io = t.io,
        .compiler = .{ .clang = clang, .linker = env.get("TF_RUNNER_LLD") },
    };
    const front = context.frontend();
    const cases = [_][4][]const u8{
        .{ "python", "print(2); print(3)", "prints.elf", "2\n3\n" },
        .{ "python", "print(sum(i*i for i in range(10)))", "squares.elf", "285\n" },
        .{ "python", "print(sum(i+i for i in range(10)))", "addition.elf", "90\n" },
        .{ "python", "s = 0\nfor i in range(10):\n    s += i*i\nprint(s)", "loop.elf", "285\n" },
        .{ "python", "n = 123456789\ns = 0\nwhile n > 0:\n    s += n % 10\n    n //= 10\nprint(s)", "digits.elf", "45\n" },
        .{ "python", "a = 462\nb = 1071\nwhile b:\n    a, b = b, a % b\nprint(a)", "gcd-loop.elf", "21\n" },
        .{ "c", "#include <stdio.h>\nint main(void) { int s=0; for(int i=0;i<10;i++) s+=i*i; printf(\"%d\\n\",s); return 0; }", "c.elf", "285\n" },
        .{ "c", "int main(void) { puts(\"circuit\"); return 0; }", "puts.elf", "circuit\n" },
        .{ "c", "int s=0; for(int i=0;i<10;i++) s+=i*i; printf(\"%d\\n\",s);", "snippet.elf", "285\n" },
    };
    for (cases) |case| {
        const result = try front.translate(front.ctx, t.allocator, .{ .form = .fence, .lang = case[0], .code = case[1] });
        try t.expect(result == .program);
        defer t.allocator.free(result.program.bytes);
        try t.expectEqual(.elf, result.program.kind);
        const repeated = try front.translate(front.ctx, t.allocator, .{ .form = .fence, .lang = case[0], .code = case[1] });
        try t.expect(repeated == .program);
        defer t.allocator.free(repeated.program.bytes);
        try t.expectEqualSlices(u8, result.program.bytes, repeated.program.bytes);
        try t.expect(@import("runner_frontend_program.zig").compiler.validElf(result.program.bytes));
        if (env.get("TF_RUNNER_QWENPUTER")) |root|
            try @import("runner_frontend_reference_test.zig").check(root, env.get("TF_RUNNER_REFERENCE_PYTHON") orelse "/usr/bin/python3", result.program, case[3]);
        if (env.get("TF_RUNNER_FRONTEND_ARTIFACTS")) |path| {
            const dir = try std.Io.Dir.cwd().openDir(t.io, path, .{});
            defer dir.close(t.io);
            const file = try dir.createFile(t.io, case[2], .{});
            defer file.close(t.io);
            try file.writeStreamingAll(t.io, result.program.bytes);
        }
    }
    const missing = try front.translate(front.ctx, t.allocator, .{ .form = .fence, .lang = "c", .code = "int main(void) { unknown(); }" });
    try t.expect(missing == .abstain);
    context.compiler.clang = "/nonexistent-tensorfold-clang";
    const unavailable = try front.translate(front.ctx, t.allocator, .{
        .form = .repl,
        .lang = "python",
        .code = "print(sum(i*i for i in range(10)))",
    });
    try t.expect(unavailable == .abstain);
}

test "prior cells cannot silently shadow Python builtins" {
    const front = f.frontend();
    const result = try front.translate(null, t.allocator, .{
        .form = .repl,
        .lang = "python",
        .code = "print(2*3)",
        .prior = &.{"print = lambda x: None"},
    });
    try t.expect(result == .abstain);
}

test "independent arithmetic cells accept safe earlier cells and refuse hidden builtin rebinding" {
    const front = f.frontend();
    for ([_][]const u8{ "314159 * 271828", "n = 7", "print(3)" }) |prior| {
        const result = try front.translate(null, t.allocator, .{
            .form = .repl,
            .lang = "python",
            .code = "2**100 - 3**50",
            .prior = &.{prior},
        });
        try t.expect(result == .program);
        defer t.allocator.free(result.program.bytes);
        try t.expectEqual(.calc, result.program.kind);
    }
    for ([_][]const u8{ "__builtins__ = 0", "print = 1", "len = 1", "import builtins", "import os" }) |prior| {
        const result = try front.translate(null, t.allocator, .{
            .form = .repl,
            .lang = "python",
            .code = "print(314159*271828)",
            .prior = &.{prior},
        });
        try t.expect(result == .abstain);
    }
    const dependent = try front.translate(null, t.allocator, .{
        .form = .repl,
        .lang = "python",
        .code = "print(n+1)",
        .prior = &.{"n = 7"},
    });
    try t.expect(dependent == .abstain);
}

test "inline typographic operators normalize without changing Python syntax or tuple semantics" {
    const front = f.frontend();
    const result = try front.translate(null, t.allocator, .{ .form = .inline_expr, .lang = "python", .code = "47 × 83 − 1" });
    try t.expect(result == .program);
    defer t.allocator.free(result.program.bytes);
    try t.expectEqualStrings("((47*83)-1)", result.program.bytes);
    for ([_][]const u8{ "47 × 83", "47 − 83" }) |code| {
        try t.expect((try front.translate(null, t.allocator, .{ .form = .fence, .lang = "python", .code = code })) == .abstain);
    }
    for ([_][]const u8{ "1,000 + 2", "2*3,18", "unknown(2,4)" }) |code| {
        try t.expect((try front.translate(null, t.allocator, .{ .form = .inline_expr, .lang = "python", .code = code })) == .abstain);
    }
}

comptime {
    _ = @import("runner_frontend_collections_test.zig");
}
