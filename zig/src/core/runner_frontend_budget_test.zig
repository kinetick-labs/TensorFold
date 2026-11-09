const std = @import("std");
const frontend = @import("runner_frontend.zig");
const compiler = @import("runner_frontend_program_compile.zig");
const program = @import("runner_frontend_program.zig");
const t = std.testing;

pub const observed_gcd =
    \\int gcd(int a, int b) {
    \\    while (b != 0) { int temp = b; b = a % b; a = temp; }
    \\    return a;
    \\}
    \\int main(void) {
    \\    int a = 462; int b = 1071;
    \\    printf("gcd(%d, %d) = %d\n", a, b, gcd(a, b)); return 0;
    \\}
;

test "saved C Euclid swap order retains default admission and rejects noncontracting mutations" {
    var env = try t.environ.createMap(t.allocator);
    defer env.deinit();
    const clang = env.get("TF_RUNNER_CLANG") orelse return error.SkipZigTest;
    const config: compiler.Config = .{ .clang = clang, .linker = env.get("TF_RUNNER_LLD") };
    const bytes = (try compiler.compile(t.allocator, t.io, observed_gcd, config)) orelse
        return error.UnexpectedAbstention;
    defer t.allocator.free(bytes);
    for ([_][]const u8{ "a + b", "a % a + b" }) |replacement| {
        const changed = try std.mem.replaceOwned(u8, t.allocator, observed_gcd, "a % b", replacement);
        defer t.allocator.free(changed);
        const refused = try compiler.compile(t.allocator, t.io, changed, config);
        defer if (refused) |elf| t.allocator.free(elf);
        try t.expect(refused == null);
    }
    try @import("runner_frontend_reference_test.zig").check(env.get("TF_RUNNER_QWENPUTER") orelse
        return error.SkipZigTest, env.get("TF_RUNNER_REFERENCE_PYTHON") orelse "/usr/bin/python3", .{ .kind = .elf, .bytes = bytes }, "gcd(462, 1071) = 21\n");
}

test "estimated cell steps reject long integer walks before compilation" {
    const short = (try program.lowerWithCost(t.allocator, "n = 0\nwhile n < 5:\n    n += 1\nprint(n)")) orelse return error.UnexpectedAbstention;
    defer t.allocator.free(short.source);
    const long = (try program.lowerWithCost(t.allocator, "n = 0\nwhile n < 10000:\n    n += 1\nprint(n)")) orelse return error.UnexpectedAbstention;
    defer t.allocator.free(long.source);
    try t.expect(short.estimated_steps <= 10522);
    try t.expect(long.estimated_steps > 10522);
}

test "configured public frontend applies the step budget with a disabled policy control" {
    var env = try t.environ.createMap(t.allocator);
    defer env.deinit();
    const clang = env.get("TF_RUNNER_CLANG") orelse return error.SkipZigTest;
    var context: frontend.Context = .{ .io = t.io, .compiler = .{ .clang = clang, .linker = env.get("TF_RUNNER_LLD") } };
    const cells = [_][]const u8{
        "n = 0\nwhile n < 10000:\n    n += 1\nprint(n)",
        "int main(void){int n=0;for(int i=0;i<10000;++i)n+=i;printf(\"%d\\n\",n);return 0;}",
    };
    for (cells, 0..) |code, index| {
        const cell: frontend.Cell = .{ .lang = if (index == 0) "python" else "c", .form = .fence, .code = code };
        const f = context.frontend();
        const refused = try f.translate(f.ctx, t.allocator, cell);
        if (refused == .program) t.allocator.free(refused.program.bytes);
        try t.expect(refused == .abstain);
        context.compiler.step_budget = 0;
        const admitted = try f.translate(f.ctx, t.allocator, cell);
        try t.expect(admitted == .program);
        t.allocator.free(admitted.program.bytes);
        context.compiler.step_budget = 10522;
    }
    const source = "int main(void){printf(\"%d\\n\",2*3);return 0;}";
    const small = (try compiler.compile(t.allocator, t.io, source, context.compiler)) orelse return error.UnexpectedAbstention;
    defer t.allocator.free(small);
    try t.expect(compiler.validElf(small));
    context.compiler.estimated_steps = 10523;
    try t.expectEqual(@as(?[]u8, null), try compiler.compile(t.allocator, t.io, source, context.compiler));
    context.compiler.estimated_steps = null;
    context.compiler.deadline = std.Io.Clock.awake.now(t.io).subDuration(.fromNanoseconds(1));
    try t.expectEqual(@as(?[]u8, null), try compiler.compile(t.allocator, t.io, source, context.compiler));
    const f = context.frontend();
    const expired = try f.translate(f.ctx, t.allocator, .{ .form = .fence, .lang = "c", .code = source });
    if (expired == .program) t.allocator.free(expired.program.bytes);
    try t.expect(expired == .abstain);
}

test "restoration bytes are budgeted independently of short source text" {
    var env = try t.environ.createMap(t.allocator);
    defer env.deinit();
    const clang = env.get("TF_RUNNER_CLANG") orelse return error.SkipZigTest;
    const config: compiler.Config = .{ .clang = clang, .linker = env.get("TF_RUNNER_LLD") };
    const copied = "static char data[4096]={1};int main(void){puts(\"ok\");return 0;}";
    const refused = try compiler.compile(t.allocator, t.io, copied, config);
    defer if (refused) |bytes| t.allocator.free(bytes);
    try t.expect(refused == null);
    const literal = "static char data[4096]=\"x\";int main(void){puts(\"ok\");return 0;}";
    const bytes = (try compiler.compile(t.allocator, t.io, literal, config)) orelse return error.UnexpectedAbstention;
    defer t.allocator.free(bytes);
    try t.expect(compiler.validElf(bytes));
    try @import("runner_frontend_reference_test.zig").check(env.get("TF_RUNNER_QWENPUTER") orelse return error.SkipZigTest, env.get("TF_RUNNER_REFERENCE_PYTHON") orelse "/usr/bin/python3", .{ .kind = .elf, .bytes = bytes }, "ok\n");
}

test "firmware and inline forms share the per-cell admission budget" {
    var context: frontend.Context = .{ .io = t.io, .compiler = .{ .step_budget = 1, .clang = "/nonexistent-t5-clang" } };
    for ([_]frontend.Cell{
        .{ .form = .fence, .lang = "python", .code = "print(2*3)" },
        .{ .form = .inline_expr, .lang = "python", .code = "2*3" },
    }) |cell| {
        const f = context.frontend();
        const refused = try f.translate(f.ctx, t.allocator, cell);
        if (refused == .program) t.allocator.free(refused.program.bytes);
        try t.expect(refused == .abstain);
        context.compiler.step_budget = 0;
        const admitted = try f.translate(f.ctx, t.allocator, cell);
        try t.expect(admitted == .program);
        defer t.allocator.free(admitted.program.bytes);
        try t.expectEqual(.calc, admitted.program.kind);
        context.compiler.step_budget = 1;
    }
}
