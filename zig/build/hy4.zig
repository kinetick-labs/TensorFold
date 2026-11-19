//! Host tests of the Hy4-flash family: config, Engram hash and picture grid. Std-only, no GPU.

const std = @import("std");

/// `test-hy4-host` under `parent`.
pub fn wire(b: *std.Build, parent: ?*std.Build.Step) void {
    const tests = b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("zig/src/families/hy4/root.zig"), .target = b.graph.host }) });
    const step = b.step("test-hy4-host", "Hy4-flash config, Engram hash ids and picture grid against the reference");
    step.dependOn(&b.addRunArtifact(tests).step);
    if (parent) |p| p.dependOn(step);
}
