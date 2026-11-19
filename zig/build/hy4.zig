//! Host tests of the Hy4-flash family: config, Engram hash, picture grid and preprocessing, reply markup, the CPU
//! layer reference. No GPU.

const std = @import("std");

/// `test-hy4-host` under `parent`.
pub fn wire(b: *std.Build, parent: ?*std.Build.Step) void {
    const resample = b.createModule(.{ .root_source_file = b.path("zig/src/core/picture/resample.zig"), .target = b.graph.host });
    const act_ref = b.createModule(.{ .root_source_file = b.path("zig/src/core/act_ref.zig"), .target = b.graph.host });
    const root = b.createModule(.{ .root_source_file = b.path("zig/src/families/hy4/root.zig"), .target = b.graph.host, .imports = &.{ .{ .name = "picture_resample", .module = resample }, .{ .name = "act_ref", .module = act_ref } } });
    const tests = b.addTest(.{ .root_module = root });
    const step = b.step("test-hy4-host", "Hy4-flash config, Engram hashes, pictures, reply markup, the layer, Engram and vision references against transformers");
    step.dependOn(&b.addRunArtifact(tests).step);
    if (parent) |p| p.dependOn(step);
}

/// `test-hy4-cuda-host`: the CUDA loader's host tests (layouts, weight descriptors) over the CUDA runtime and core
/// modules, without a GPU; under `all` (test-cuda-host).
pub fn cudaHost(b: *std.Build, cuda: *std.Build.Module, core: *std.Build.Module, all: *std.Build.Step) void {
    const m = b.createModule(.{ .root_source_file = b.path("zig/src/families/hy4/cuda.zig"), .target = b.graph.host, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core } } });
    const run = b.addRunArtifact(b.addTest(.{ .root_module = m }));
    b.step("test-hy4-cuda-host", "Hy4 CUDA loader: arena layout and EXL3 weight descriptors from the pack plan, no GPU").dependOn(&run.step);
    all.dependOn(&run.step);
}
