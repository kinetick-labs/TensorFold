//! The compute lane's sandbox host tests: the sandbox executable, then jobs through Landlock, seccomp and limits.
const std = @import("std");

pub fn steps(b: *std.Build, host: std.Build.ResolvedTarget, step: *std.Build.Step) void {
    if (host.result.os.tag != .linux) return;
    const sandbox_exe = b.addExecutable(.{ .name = "compute-sandbox", .root_module = b.createModule(.{ .root_source_file = b.path("zig/src/core/compute_sandbox_main.zig"), .target = host, .optimize = .Debug, .link_libc = true }) });
    const options = b.addOptions();
    options.addOptionPath("sandbox_exe", sandbox_exe.getEmittedBin());
    const tests = b.createModule(.{ .root_source_file = b.path("zig/src/core/compute_lane_test.zig"), .target = host, .optimize = .Debug, .link_libc = true });
    tests.addOptions("compute_lane_options", options);
    const run = b.addRunArtifact(b.addTest(.{ .root_module = tests }));
    b.step("test-compute-lane-host", "Sandboxed compute lane jobs: Landlock, seccomp, limits and result text").dependOn(&run.step);
    step.dependOn(&run.step);
}
