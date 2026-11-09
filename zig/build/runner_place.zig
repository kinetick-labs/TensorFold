//! The in-model runner's thread placement microbench (host only); its tests run in test-runner-host.
const std = @import("std");

pub fn steps(b: *std.Build) void {
    const exe = b.addExecutable(.{ .name = "runner-place-bench", .root_module = b.createModule(.{ .root_source_file = b.path("zig/src/core/runner_place_bench.zig"), .target = b.graph.host, .optimize = .ReleaseFast, .link_libc = true }) });
    b.step("bench-runner-place", "Runner at full load beside an engine host-loop proxy: idle, placed, inherit (TF_RUNNER_CIRCUIT: multiply)").dependOn(&b.addInstallArtifact(exe, .{}).step);
}
