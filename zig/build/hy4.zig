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

/// `test-hy4-gpu` (runs here) and `install-hy4-gpu-test` (zig-out/bin/hy4-gpu-test, for -Dhy4-gpu-target, e.g.
/// aarch64-linux-gnu, to copy to a Spark): the Hy4 kernels against their CPU references on a GPU. The kernels come as
/// PTX at run time (HY4_HC_PTX: `nvcc -ptx zig/kernels/cuda/hy4/hc.cu`); without it the tests skip.
pub fn gpuTests(b: *std.Build) void {
    const triple = b.option([]const u8, "hy4-gpu-target", "Target of install-hy4-gpu-test (default: this host)");
    const target = if (triple) |t| b.resolveTargetQuery(std.Target.Query.parse(.{ .arch_os_abi = t }) catch @panic("bad -Dhy4-gpu-target")) else b.graph.host;
    const cuda = @import("cuda_runtime.zig").runtime(b, target, .ReleaseSafe, &.{});
    const m = b.createModule(.{ .root_source_file = b.path("zig/src/families/hy4/cuda_hc_test.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true, .imports = &.{.{ .name = "cuda", .module = cuda }} });
    const exe = b.addTest(.{ .root_module = m, .name = "hy4-gpu-test" });
    b.step("test-hy4-gpu", "Hy4 CUDA kernels against their CPU references on this machine's GPU (HY4_HC_PTX)").dependOn(&b.addRunArtifact(exe).step);
    b.step("install-hy4-gpu-test", "Install the Hy4 GPU kernel test binary (see -Dhy4-gpu-target)").dependOn(&b.addInstallArtifact(exe, .{}).step);
}
