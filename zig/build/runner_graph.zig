//! The runner's graph bank host tests (test-runner-graph-host, in test-cuda-host).
const std = @import("std");

pub fn tests(b: *std.Build, all: *std.Build.Step) void {
    const unit = b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("zig/src/core/runner_graph_test.zig"), .target = b.graph.host, .optimize = .ReleaseSafe, .link_libc = true }) });
    const step = b.step("test-runner-graph-host", "Graph bank packets, programs, tables and route.Dispatch, host only");
    step.dependOn(&b.addRunArtifact(unit).step);
    all.dependOn(step);
}
