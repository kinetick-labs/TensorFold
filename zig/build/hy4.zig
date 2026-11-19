//! Host tests of the Hy4-flash family: config, Engram hash, picture grid and preprocessing, reply markup, the CPU
//! layer reference. No GPU.

const std = @import("std");

/// `test-hy4-host` under `parent`.
pub fn wire(b: *std.Build, parent: ?*std.Build.Step) void {
    const resample = b.createModule(.{ .root_source_file = b.path("zig/src/core/picture/resample.zig"), .target = b.graph.host });
    const root = b.createModule(.{ .root_source_file = b.path("zig/src/families/hy4/root.zig"), .target = b.graph.host, .imports = &.{.{ .name = "picture_resample", .module = resample }} });
    const tests = b.addTest(.{ .root_module = root });
    const step = b.step("test-hy4-host", "Hy4-flash config, Engram hashes, pictures, reply markup and the layer reference against transformers");
    step.dependOn(&b.addRunArtifact(tests).step);
    if (parent) |p| p.dependOn(step);
}
