const std = @import("std");
const api = @import("runner_api.zig");
const reference = @import("runner_frontend_reference_test.zig");
const t = std.testing;

// Call only for the fixed, finite test fixtures; never for arbitrary frontend input.
pub fn checkExact(env: *const std.process.Environ.Map, program: api.Program, code: []const u8) !void {
    const python = env.get("TF_RUNNER_REFERENCE_PYTHON") orelse "/usr/bin/python3";
    const oracle = try std.process.run(t.allocator, t.io, .{
        .argv = &.{ python, "-I", "-c", code },
        .stdout_limit = .limited(16384),
        .stderr_limit = .limited(16384),
        .timeout = .{ .duration = .{ .raw = .fromSeconds(2), .clock = .awake } },
    });
    defer t.allocator.free(oracle.stdout);
    defer t.allocator.free(oracle.stderr);
    if (!oracle.term.success()) {
        std.debug.print("frontend admitted a failing CPython fixture: {s}\n{s}\n", .{ code, oracle.stderr });
        return error.InvalidPythonAdmission;
    }
    const root = env.get("TF_RUNNER_QWENPUTER") orelse return error.SkipZigTest;
    try reference.check(root, python, program, oracle.stdout);
}

pub fn checkRejectionOrExact(env: *const std.process.Environ.Map, front: *const api.Frontend, cell: api.Cell) !void {
    const result = try front.translate(front.ctx, t.allocator, cell);
    defer if (result == .program) t.allocator.free(result.program.bytes);
    if (result == .program) try checkExact(env, result.program, cell.code);
}
