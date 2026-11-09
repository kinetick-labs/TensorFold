const std = @import("std");
const api = @import("runner_api.zig");
const t = std.testing;

const reference =
    \\import sys
    \\from pathlib import Path
    \\root = Path(sys.argv[1])
    \\sys.path.insert(0, str(root / 'experimental/riscv'))
    \\from chipformer.rv import calc, elf, ref
    \\if sys.argv[2] == 'calc':
    \\    m = calc.machine(sys.argv[3], root / 'experimental/riscv/tests/data/rv/calc.elf')
    \\else:
    \\    m = ref.Machine.from_program(elf.program(sys.argv[3]))
    \\tokens = 1
    \\while not m.halted and tokens < 4000000:
    \\    tokens += len(m.record())
    \\assert tokens <= 4000000, tokens
    \\assert m.halted and m.fault is None and m.exit_code == 0, (m.halted, m.fault, m.exit_code)
    \\sys.stdout.buffer.write(bytes(m.out))
;

pub fn check(root: []const u8, python: []const u8, program: api.Program, expected: []const u8) !void {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var path: ?[:0]u8 = null;
    defer if (path) |p| t.allocator.free(p);
    var input = program.bytes;
    if (program.kind == .elf) {
        const file = try tmp.dir.createFile(t.io, "program.elf", .{});
        defer file.close(t.io);
        try file.writeStreamingAll(t.io, program.bytes);
        path = try tmp.dir.realPathFileAlloc(t.io, "program.elf", t.allocator);
        input = path.?;
    }
    const result = try std.process.run(t.allocator, t.io, .{
        .argv = &.{ python, "-c", reference, root, @tagName(program.kind), input },
        .stdout_limit = .limited(4096),
        .stderr_limit = .limited(16384),
        .timeout = .{ .duration = .{ .raw = .fromSeconds(10), .clock = .awake } },
    });
    defer t.allocator.free(result.stdout);
    defer t.allocator.free(result.stderr);
    if (!result.term.success()) {
        std.debug.print("RV32 reference failed: {s}\n", .{result.stderr});
        return error.ReferenceFailed;
    }
    try t.expectEqualStrings(expected, result.stdout);
}

test "accepted arithmetic executes in the firmware reference with exact output" {
    var env = try t.environ.createMap(t.allocator);
    defer env.deinit();
    const root = env.get("TF_RUNNER_QWENPUTER") orelse {
        std.debug.print("SKIP firmware reference: set TF_RUNNER_QWENPUTER to its source checkout\n", .{});
        return error.SkipZigTest;
    };
    var context: @import("runner_frontend.zig").Context = .{ .io = t.io, .exact_tags = true };
    const frontend = context.frontend();
    const cases = [_][3][]const u8{
        .{ "python", "print(123456789012345*987654321098765)", "121932631137021071359549253925\n" },
        .{ "python", "print(123456789012345678901234567890*987654321098765432109876543210)", "121932631137021795226185032733622923332237463801111263526900\n" },
        .{ "python", "2**100", "1267650600228229401496703205376\n" },
        .{ "python", "-7//3", "-3\n" },
        .{ "python", "-7%3", "2\n" },
        .{ "exact", "1/3+2/3", "1\n" },
        .{ "exact", "1.25*2.4", "3\n" },
        .{ "exact", "123456789012345678901234567890.12345678901234567890123456789012", "123456789012345678901234567890.12345678901234567890123456789012\n" },
        .{ "exact", "gcd(48,18)", "6\n" },
    };
    for (cases) |case| {
        const result = try frontend.translate(frontend.ctx, t.allocator, .{ .form = .repl, .lang = case[0], .code = case[1] });
        try t.expect(result == .program);
        defer t.allocator.free(result.program.bytes);
        try check(root, env.get("TF_RUNNER_REFERENCE_PYTHON") orelse "/usr/bin/python3", result.program, case[2]);
    }
}
