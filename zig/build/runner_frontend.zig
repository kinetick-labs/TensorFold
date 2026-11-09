const std = @import("std");

pub fn tests(b: *std.Build, all: *std.Build.Step) void {
    const unit = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/core/runner_frontend_test.zig"),
        .target = b.graph.host,
        .optimize = .Debug,
        .link_libc = true,
    }) });
    const run = b.addRunArtifact(unit);
    const step = b.step("test-runner-frontend-host", "Model code to calc firmware or RV32 circuit ELF, host only");
    step.dependOn(&run.step);
    all.dependOn(step);
}
