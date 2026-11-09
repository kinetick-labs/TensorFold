const std = @import("std");
const graph = @import("runner_graph.zig");
const route = @import("runner_graph_route.zig");
const frontend = @import("runner_frontend.zig");
const math = @import("runner_frontend_inline_integer.zig");
const t = std.testing;

pub const cases = [_][3][]const u8{
    .{ "7**123%1000000007", "pow(7,123,1000000007)", "937329259\n" },
    .{ "3**400%10**18", "pow(3,400,10**18)", "192443283334088001\n" },
    .{ "0xFEDCBA9876543210*2**64+0x0123456789ABCDEF", "0xFEDCBA9876543210*2**64+0x0123456789ABCDEF", "338770000845734292516042252062085074415\n" },
};

fn bankPath() ![]const u8 {
    const p = std.c.getenv("TF_RUNNER_GRAPH_BANK") orelse return error.SkipZigTest;
    return std.mem.span(p);
}

fn observe(bank: *graph.Bank, source: []const u8, expected: []const u8, form: @import("runner_api.zig").Form) !void {
    var context: frontend.Context = .{ .io = t.io, .compiler = .{ .clang = "/nonexistent-regression-clang" } };
    var routed: route.Routed = .{ .inner = context.frontend(), .bank = bank };
    const f = routed.frontend();
    const result = try f.translate(f.ctx, t.allocator, .{ .form = form, .lang = "python", .code = source });
    defer if (result == .program) t.allocator.free(result.program.bytes);
    if (result == .abstain) std.debug.print("unexpected refusal {t}: {s}\n", .{ form, source });
    try t.expect(result == .program);
    try t.expect(result.program.kind == .graph);
    var machine = try @import("runner_graph_machine.zig").Machine.boot(t.allocator, bank, result.program.bytes);
    defer machine.deinit();
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(t.allocator);
    const end = try machine.run(4_000_000, &out, null);
    try t.expect(end.end == .halted);
    try t.expectEqualStrings(expected, out.items);
}

test "runner proof: S2 modular powers and hex inline statements retain ownership without clang" {
    var bank = try graph.Bank.load(t.allocator, try bankPath());
    defer bank.deinit();
    for (cases) |fixture| {
        const code = try std.fmt.allocPrint(t.allocator, "print({s})", .{fixture[1]});
        defer t.allocator.free(code);
        const oracle = try std.process.run(t.allocator, t.io, .{ .argv = &.{ "/usr/bin/python3", "-I", "-S", "-c", code }, .stdout_limit = .limited(4096), .stderr_limit = .limited(4096) });
        defer t.allocator.free(oracle.stdout);
        defer t.allocator.free(oracle.stderr);
        try t.expect(oracle.term.success());
        try t.expectEqualStrings(fixture[2], oracle.stdout);
        try observe(&bank, fixture[0], oracle.stdout, .inline_expr);
        try observe(&bank, fixture[0], oracle.stdout, .repl);
    }
    try observe(&bank, "7^123 mod 1000000007", cases[0][2], .inline_expr);
    try observe(&bank, "3^400 mod 10^18", cases[1][2], .inline_expr);
    try observe(&bank, "0xFEDCBA9876543210*2^64+0x0123456789ABCDEF", cases[2][2], .inline_expr);
    for ([_][]const u8{
        "We need answer compute 7^123 mod 1000000007. Need calculate accurately. Need provide maybe concise. Let's compute carefully. Mod p=1,000,000,007 prime. Need exponentiation.\n",
        "Need compute 3^400 mod 10^18.\n",
        "Let A = 0xFEDCBA9876543210, B=0x0123456789ABCDEF. N = A*2^64 + B.\n",
    }, 0..) |line, i| {
        var scanner: @import("runner_scan.zig").Scanner = .{ .cfg = .{
            .trace = true,
            .inline_math = false,
            .fence = false,
            .repl = false,
        } };
        defer scanner.deinit(t.allocator);
        const event = try scanner.feed(t.allocator, line, true);
        try t.expect(event.closed != null);
        const cell = event.closed.?;
        try t.expect(cell.trace != null and cell.form == .repl);
        try observe(&bank, cell.code, cases[i][2], cell.form);
    }
}

test "runner proof: modular notation keeps code XOR floats decimals and malformed expressions abstaining" {
    var bank = try graph.Bank.load(t.allocator, try bankPath());
    defer bank.deinit();
    for ([_][]const u8{
        "7.0**123%1000000007", "3**400%1e18", "3**400%(10/2)",        "3**400%0",       "3**-1%10",
        "0x",                  "0xGG",        "0xFEDCBA9876543210,1", "7^123 mod",      "7^123 mod 10=1",
        "7^123 mod 10=",       "50%+5",       "2*50%3",               "3**400%10**100", "0x1\n+2",
        "2^",                  "print(2^3)",
    }) |source| {
        const bytes = try math.lower(t.allocator, &bank, source, true);
        defer if (bytes) |b| t.allocator.free(b);
        try t.expect(bytes == null);
    }
    for ([_][]const u8{ "7^123%1000000007", "3^400 mod 10^18" }) |source|
        try t.expectEqual(@as(?[]u8, null), try math.lower(t.allocator, &bank, source, false));
    const wrong = (try math.lower(t.allocator, &bank, "7^124 mod 1000000007", true)).?;
    defer t.allocator.free(wrong);
    var machine = try @import("runner_graph_machine.zig").Machine.boot(t.allocator, &bank, wrong);
    defer machine.deinit();
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(t.allocator);
    _ = try machine.run(4_000_000, &out, null);
    try t.expect(!std.mem.eql(u8, out.items, cases[0][2]));
}

test "runner proof: inline integer recovery cannot call notebook-shadowed builtins" {
    var bank = try graph.Bank.load(t.allocator, try bankPath());
    defer bank.deinit();
    for ([_][]const u8{ "pow = lambda *args: 42", "int = lambda *args: 42", "__builtins__ = {}" }) |prior| {
        try t.expectEqual(@as(?[]u8, null), try @import("runner_graph_program.zig").direct(t.allocator, &bank, .{
            .form = .inline_expr,
            .lang = "python",
            .code = "pow(7,123,1000000007)",
            .prior = &.{prior},
        }, .unsupported));
        try t.expectEqual(@as(?[]u8, null), try @import("runner_graph_program.zig").direct(t.allocator, &bank, .{
            .form = .repl,
            .lang = "python",
            .code = "int('ff',16)",
            .prior = &.{prior},
        }, .unsupported));
    }
}

test "runner proof: routed code caret and float-context division keep their language semantics" {
    var bank = try graph.Bank.load(t.allocator, try bankPath());
    defer bank.deinit();
    var context: frontend.Context = .{ .io = t.io, .exact_inline = false, .compiler = .{ .clang = "/nonexistent-regression-clang" } };
    var routed: route.Routed = .{ .inner = context.frontend(), .bank = &bank };
    const f = routed.frontend();
    for ([_]@import("runner_api.zig").Cell{
        .{ .form = .repl, .lang = "python", .code = "2^3" },
        .{ .form = .repl, .lang = "python", .code = "7^123 mod 1000000007" },
        .{ .form = .repl, .lang = "python", .code = "1.25*2" },
        .{ .form = .repl, .lang = "python", .code = "1/3" },
        .{ .form = .fence, .lang = "python", .code = "print(2^3)" },
        .{ .form = .fence, .lang = "python", .code = "print(7.0/3)" },
        .{ .form = .inline_expr, .lang = "python", .code = "0.5+1" },
        .{ .form = .inline_expr, .lang = "python", .code = "1/3" },
        .{ .form = .inline_expr, .lang = "python", .code = ">>> 2^3" },
        .{ .form = .inline_expr, .lang = "python", .code = ">>> 0xff" },
    }) |cell| {
        const result = try f.translate(f.ctx, t.allocator, cell);
        defer if (result == .program) t.allocator.free(result.program.bytes);
        try t.expect(result == .abstain);
    }
}
