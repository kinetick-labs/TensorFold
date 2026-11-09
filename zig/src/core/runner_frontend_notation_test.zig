const std = @import("std");
const f = @import("runner_frontend.zig");
const t = std.testing;

test "inline exact notation prints finite decimals or natural fractions through firmware" {
    var env = try t.environ.createMap(t.allocator);
    defer env.deinit();
    const front = f.frontend();
    const cases = [_][2][]const u8{
        .{ "(5*(0.125-0))", "0.625\n" },
        .{ "0-0.125", "-0.125\n" },
        .{ "(0.125+0)-0.25", "-0.125\n" },
        .{ "11/18", "11/18\n" },
        .{ "1/2", "1/2\n" },
        .{ "1/3+2/3", "1\n" },
        .{ "18.75*10", "187.5\n" },
        .{ "0.125*0.048", "0.006\n" },
        .{ "1.18 × 0.82", "0.9676\n" },
        .{ "889.35*0.075", "66.70125\n" },
        .{ "889.35 - 66.70125", "822.64875\n" },
        .{ "822.64875*1.06", "872.007675\n" },
        .{ "2668.05 /40", "66.70125\n" },
        .{ "32905.95 /40", "822.64875\n" },
        .{ "1.0/8", "0.125\n" },
        .{ "-1.0/8", "-0.125\n" },
        .{ ".5 ÷ 2", "0.25\n" },
        .{ "1.2/0.3", "4\n" },
        .{ "1.2/(0.3+0.1)", "3\n" },
        .{ "0.1/(0.2+0.2)", "0.25\n" },
        .{ "0.1+1/2", "0.6\n" },
        .{ "1/3*0.3", "0.1\n" },
        .{ "(1/3)*3*0.5", "0.5\n" },
        .{ "1.2**2", "1.44\n" },
        .{ "2**3 × 0.5", "4\n" },
        .{ "2**100", "1267650600228229401496703205376\n" },
        .{ "7.5% × 889.35", "66.70125\n" },
        .{ "20% + 5%", "0.25\n" },
        .{ "100 × 7.5%", "7.5\n" },
        .{ "(2+3)%", "0.05\n" },
        .{ "-7.5%", "-0.075\n" },
        .{ "50%/2", "0.25\n" },
        .{ "7//-3", "-3\n" },
        .{ "7÷-3", "-7/3\n" },
        .{ "(2+3)/8", "5/8\n" },
        .{ "2^10", "1024\n" },
        .{ "2^3^2", "512\n" },
        .{ "2^100 - 3^50", "1267649882330241709644114435127\n" },
    };
    for (cases) |case| {
        const result = try front.translate(null, t.allocator, .{ .form = .inline_expr, .lang = "python", .code = case[0] });
        if (result == .abstain) std.debug.print("inline abstained: {s}\n", .{case[0]});
        try t.expect(result == .program);
        defer t.allocator.free(result.program.bytes);
        try t.expectEqual(.calc, result.program.kind);
        if (env.get("TF_RUNNER_QWENPUTER")) |root|
            try @import("runner_frontend_reference_test.zig").check(root, "/usr/bin/python3", result.program, case[1]);
    }
}

test "inline math declines incomplete ambiguous repeating and unsafe notation" {
    var env = try t.environ.createMap(t.allocator);
    defer env.deinit();
    var context: f.Context = .{ .io = t.io, .compiler = .{
        .clang = env.get("TF_RUNNER_CLANG") orelse "/usr/bin/clang",
        .linker = env.get("TF_RUNNER_LLD"),
    } };
    const front = context.frontend();
    const bad = [_][]const u8{
        "1,000+2",    "2*3,18",  "1+2=3",           "1=1=1",   "1/",      "(2+3",    "1+",    "...",       "2.",
        "print(1/2)", ">>> 1/2", "gcd(48,missing)", "20%%",    "+1/3",
        "×2",
        "2\n3",       "3/0",     "1.0/3",           "0.1+1/3", "0.01//2", "10**100", "2**-1", "0.1/(2-2)", "10^100",
        "2^-1",       "2^",      "^2",
    };
    for (bad) |source| {
        const result = try front.translate(front.ctx, t.allocator, .{ .form = .inline_expr, .lang = "python", .code = source });
        if (result == .program) {
            defer t.allocator.free(result.program.bytes);
            std.debug.print("inline unexpectedly accepted: {s}\n", .{source});
        }
        try t.expect(result == .abstain);
    }
}

test "inline defaults and explicit override never change Python code cell semantics" {
    var context: f.Context = .{ .io = t.io, .exact_tags = true, .exact_inline = false };
    const front = context.frontend();
    for ([_][]const u8{ "1/2", "6/3", "1.25*2", "20%", "1÷2" }) |source| {
        try t.expect((try front.translate(front.ctx, t.allocator, .{
            .form = .inline_expr,
            .lang = "python",
            .code = source,
        })) == .abstain);
    }
    const integer = try front.translate(front.ctx, t.allocator, .{ .form = .inline_expr, .lang = "python", .code = "7×3" });
    try t.expect(integer == .program);
    defer t.allocator.free(integer.program.bytes);
    context.exact_inline = true;
    for ([_]f.Cell{
        .{ .form = .fence, .lang = "python", .code = "print(1/3)" },
        .{ .form = .repl, .lang = "python", .code = "1.25*2.4" },
        .{ .form = .fence, .lang = "python", .code = "gcd(48,18)" },
        .{ .form = .repl, .lang = "python", .code = "7.5%" },
    }) |cell| try t.expect((try front.translate(front.ctx, t.allocator, cell)) == .abstain);
    const tagged = try front.translate(front.ctx, t.allocator, .{ .form = .fence, .lang = "exact", .code = "1/3" });
    try t.expect(tagged == .program);
    defer t.allocator.free(tagged.program.bytes);
    const independent = try front.translate(front.ctx, t.allocator, .{
        .form = .inline_expr,
        .lang = "python",
        .code = "11/18",
        .prior = &.{"print = lambda x: None"},
    });
    try t.expect(independent == .program);
    defer t.allocator.free(independent.program.bytes);
}

test "inline `^` is a power like `**`; Python code cells never read it as one" {
    var context: f.Context = .{ .io = t.io, .exact_tags = true, .exact_inline = true };
    const front = context.frontend();
    for ([_][2][]const u8{ .{ "2^10", "2**10" }, .{ "2^100 - 3^50", "2**100 - 3**50" }, .{ "(3+4)^2", "(3+4)**2" } }) |pair| {
        const caret = try front.translate(front.ctx, t.allocator, .{ .form = .inline_expr, .lang = "python", .code = pair[0] });
        if (caret == .abstain) std.debug.print("inline abstained: {s}\n", .{pair[0]});
        try t.expect(caret == .program);
        defer t.allocator.free(caret.program.bytes);
        const star = try front.translate(front.ctx, t.allocator, .{ .form = .inline_expr, .lang = "python", .code = pair[1] });
        try t.expect(star == .program);
        defer t.allocator.free(star.program.bytes);
        try t.expectEqualSlices(u8, star.program.bytes, caret.program.bytes);
    }
    for ([_]f.Cell{
        .{ .form = .fence, .lang = "python", .code = "print(2^10)" },
        .{ .form = .repl, .lang = "python", .code = "2^10" },
        .{ .form = .repl, .lang = "python", .code = "6 ^ 3" },
    }) |cell| {
        const power = try front.translate(front.ctx, t.allocator, .{ .form = cell.form, .lang = "python", .code = "2**10" });
        defer if (power == .program) t.allocator.free(power.program.bytes);
        const got = try front.translate(front.ctx, t.allocator, cell);
        if (got == .program) {
            defer t.allocator.free(got.program.bytes);
            try t.expect(power != .program or !std.mem.eql(u8, power.program.bytes, got.program.bytes));
        }
    }
    context.exact_inline = false;
    try t.expect((try front.translate(front.ctx, t.allocator, .{ .form = .inline_expr, .lang = "python", .code = "2^10" })) == .abstain);
}

test "firmware type checks operate on intermediate values" {
    const calc = @import("runner_frontend_calc.zig");
    for ([_][]const u8{ "((266805*1000)//40)*0.00001", "(2**3)*0.5", "((1/3)*3)*0.5" }) |source| {
        const result = (try calc.lower(t.allocator, source)).?;
        defer t.allocator.free(result);
    }
    for ([_][]const u8{ "1.2/2", "1.2//2", "(1/3)*0.5", "(1/3)+0.5", "1.0**2", "0.125-0", "0-0.125", "(0.125+0)-0.25" }) |source| {
        try t.expectEqual(@as(?[]u8, null), try calc.lower(t.allocator, source));
    }
}
