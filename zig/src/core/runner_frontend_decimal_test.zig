const std = @import("std");
const decimal = @import("runner_frontend_decimal.zig");
const frontend = @import("runner_frontend.zig");
const reference = @import("runner_frontend_reference_test.zig");
const guard = @import("runner_frontend_optimization_test.zig");
const t = std.testing;
const prefix = "from decimal import Decimal\n";
const fixtures = [_][2][]const u8{
    .{ "print(Decimal('-27.201')*Decimal('4952356'))", "-134709035.556\n" },
    .{ "print(Decimal('562.1') * Decimal('7.691'))", "4323.1111\n" },
    .{ "print(Decimal('-540262.6')*Decimal('703773'))", "-380222230789.8\n" },
    .{ "a=Decimal('1.20'); b=Decimal('2.0'); print(a*b)", "2.400\n" },
    .{ "a=Decimal('0.00'); b=Decimal('2.0'); print(a*b)", "0.000\n" },
    .{ "print(Decimal('.5')*2)", "1.0\n" },
    .{ "print(-1*0*Decimal('1'))", "0\n" },
    .{ "print(Decimal('-0.0')*Decimal('-2'))", "0.0\n" },
    .{ "a=Decimal('1.25');print(a);a=a*2;print(a)", "1.25\n2.50\n" },
};

test "Decimal coefficients keep original runtime operations and exact default-context scale" {
    for (fixtures) |fixture| {
        const code = try std.fmt.allocPrint(t.allocator, "{s}{s}", .{ prefix, fixture[0] });
        defer t.allocator.free(code);
        const result = (try decimal.lowerWithCost(t.allocator, code, 10522)).?;
        defer t.allocator.free(result.source);
        try t.expect(result.estimated_steps.? <= 10522);
        try t.expect(std.mem.indexOf(u8, result.source, "tf_run_formatted") != null);
    }
    const source = (try decimal.lowerWithCost(t.allocator, prefix ++ fixtures[0][0], 10522)).?;
    defer t.allocator.free(source.source);
    try t.expect(std.mem.indexOf(u8, source.source, "134709035556") == null);
    try t.expect(std.mem.indexOf(u8, source.source, "27201") != null);
    try t.expect(std.mem.indexOf(u8, source.source, "4952356") != null);
}

test "Decimal refuses rounding contexts scientific notation negative zero and unknown effects" {
    for ([_][]const u8{
        "print(Decimal('12345678901234567890123456789')*2)",
        "print(Decimal('9999999999999999999999999999')*9)",
        "print(Decimal('1e2')*2)",
        "print(Decimal('0.0000001')*2)",
        "print(Decimal('2')) #\x00",
        "print(Decimal('NaN'))",
        "print(Decimal('Infinity'))",
        "print(Decimal('-0.00'))",
        "print(Decimal('-1')*-0)",
        "getcontext().prec=9;print(Decimal('1')*2)",
        "print(Decimal('1')/3)",
        "print(Decimal('1')+Decimal('2'))",
        "Decimal=3;print(Decimal('1'))",
        "x=Decimal('bad');print(Decimal('2'))",
        "print(Decimal(1.25))",
        "print(Decimal('1')*True)",
    }) |body| {
        const code = try std.fmt.allocPrint(t.allocator, "{s}{s}", .{ prefix, body });
        defer t.allocator.free(code);
        const result = try decimal.lowerWithCost(t.allocator, code, 10522);
        defer if (result) |x| t.allocator.free(x.source);
        try t.expect(result == null);
    }
}

test "configured Decimal programs match CPython stdout on the RV32 reference" {
    var env = try t.environ.createMap(t.allocator);
    defer env.deinit();
    const clang = env.get("TF_RUNNER_CLANG") orelse return error.SkipZigTest;
    const root = env.get("TF_RUNNER_QWENPUTER") orelse return error.SkipZigTest;
    var ctx: frontend.Context = .{ .io = t.io, .compiler = .{ .clang = clang, .linker = env.get("TF_RUNNER_LLD") } };
    const f = ctx.frontend();
    for (fixtures) |fixture| {
        const code = try std.fmt.allocPrint(t.allocator, "{s}{s}", .{ prefix, fixture[0] });
        defer t.allocator.free(code);
        const result = try f.translate(f.ctx, t.allocator, .{ .form = .fence, .lang = "python", .code = code });
        try t.expect(result == .program);
        defer t.allocator.free(result.program.bytes);
        try reference.check(root, "/usr/bin/python3", result.program, fixture[1]);
        try guard.assertNoOutputInRodata(result.program.bytes, fixture[1]);
    }
}
