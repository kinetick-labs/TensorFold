//! T2's R1 circuit evaluator host tests and the CPU steps/s bench.
const std = @import("std");

pub fn tests(b: *std.Build, step: *std.Build.Step) void {
    const host = b.graph.host;
    const module = b.createModule(.{ .root_source_file = b.path("zig/src/core/runner_circuit_test.zig"), .target = host, .optimize = .ReleaseSafe, .link_libc = true });
    const run = b.addRunArtifact(b.addTest(.{ .root_module = module }));
    b.step("test-runner-circuit-host", "R1 circuit loader, exact CPU evaluator, RAM op, machine and backend").dependOn(&run.step);
    step.dependOn(&run.step);
    const bench = b.addExecutable(.{ .name = "runner-circuit-bench", .root_module = b.createModule(.{ .root_source_file = b.path("zig/src/core/runner_circuit_bench.zig"), .target = host, .optimize = .ReleaseFast, .link_libc = true }) });
    const run_bench = b.addRunArtifact(bench);
    const circuit = b.option([]const u8, "runner-circuit", "Circuit file for bench-runner-circuit");
    const expr = b.option([]const u8, "runner-expr", "Calc expression for bench-runner-circuit") orelse "123456789012345678901234567890*987654321098765432109876543210";
    if (circuit) |path| run_bench.addArgs(&.{ path, expr });
    b.step("bench-runner-circuit", "CPU circuit steps/s: -Drunner-circuit=<file> [-Drunner-expr=<expression>]").dependOn(&run_bench.step);
}
