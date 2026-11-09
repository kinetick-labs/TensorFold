const std = @import("std");
const f = @import("runner_frontend.zig");
const t = std.testing;

// These are authored controls, separate from the observed census denominator.
test "integer statement programs match CPython and the RV32 reference" {
    var env = try t.environ.createMap(t.allocator);
    defer env.deinit();
    const clang = env.get("TF_RUNNER_CLANG") orelse return error.SkipZigTest;
    var context: f.Context = .{ .io = t.io, .compiler = .{ .clang = clang, .linker = env.get("TF_RUNNER_LLD") } };
    const front = context.frontend();
    const cases = [_][2][]const u8{
        .{ "x = 123\ny = 456\nz = x*y\nprint(z)", "56088\n" },
        .{ "print(sum([2, -7, 3*4]))", "7\n" },
        .{ "print(sum([i*i for i in range(10)]))", "285\n" },
        .{ "xs = [2, -7, 3*4]\nprint(sum(xs))", "7\n" },
        .{ "xs = []\nprint(sum(xs))", "0\n" },
        .{ "n = 12\nf = 1\nfor i in range(1, n+1):\n    f *= i\nprint(f)", "479001600\n" },
        .{ "a, b = 0, 1\nfor i in range(20):\n    a, b = b, a+b\nprint(a)", "6765\n" },
        .{ "a, b = 0, 1\nfor i in range(20):\n    a = a+b\n    b = a\nprint(a)", "524288\n" },
        .{ "x = -7\ny = 3\nprint(x//y)", "-3\n" },
        .{ "x = 7\ny = -3\nprint(x//y)", "-3\n" },
        .{ "x = -7\ny = -3\nprint(x//y)", "2\n" },
        .{ "x = -7\ny = 3\nprint(x%y)", "2\n" },
        .{ "x = 7\ny = -3\nprint(x%y)", "-2\n" },
        .{ "x = -7\ny = -3\nprint(x%y)", "-1\n" },
        .{ "a = 2\na **= 3\nprint(-a**2)", "-64\n" },
        .{ "n = 10\ns = 0\nfor i in range(n):\n    s += i*i\nprint(s)", "285\n" },
        .{ "s = 0\nfor i in range(10):\n    s += i\n    i = 100\nprint(s)", "45\n" },
        .{ "s = 0\nfor i in range(10):\n    s += i\n    i = 100\nprint(i)", "100\n" },
        .{ "i = 9\nfor i in range(0):\n    i = 100\nprint(i)", "9\n" },
        .{ "s = 0\nfor i in range(9, -1, -1):\n    s += i\nprint(s)", "45\n" },
        .{ "a = 2\nb = 3\na, b = b, a\nprint(a*10+b)", "32\n" },
        .{ "# integer work: no floats/division.\nn = 10\n\ns = 0 # accumulator\nfor i in range(n):\n    s += i*i # square\nprint(s)\n", "285\n" },
        .{ "print(sum(i//-3 for i in range(-7, 8)))", "-5\n" },
        .{ "print(sum(i%-3 for i in range(-7, 8)))", "-15\n" },
        .{ "print(sum(i//3 for i in range(-7, 8)))", "-5\n" },
        .{ "print(sum(i%3 for i in range(-7, 8)))", "15\n" },
    };
    for (cases) |case| {
        const result = try front.translate(front.ctx, t.allocator, .{ .form = .fence, .lang = "python", .code = case[0] });
        if (result == .abstain) std.debug.print("unexpected abstention: {s}\n", .{case[0]});
        try t.expect(result == .program);
        defer t.allocator.free(result.program.bytes);
        const oracle = try std.process.run(t.allocator, t.io, .{
            .argv = &.{ "/usr/bin/python3", "-I", "-c", case[0] },
            .stdout_limit = .limited(4096),
            .stderr_limit = .limited(4096),
            .timeout = .{ .duration = .{ .raw = .fromSeconds(2), .clock = .awake } },
        });
        defer t.allocator.free(oracle.stdout);
        defer t.allocator.free(oracle.stderr);
        try t.expect(oracle.term.success());
        try t.expectEqualStrings(case[1], oracle.stdout);
        if (env.get("TF_RUNNER_QWENPUTER")) |root|
            try @import("runner_frontend_reference_test.zig").check(root, "/usr/bin/python3", result.program, oracle.stdout);
    }
}
