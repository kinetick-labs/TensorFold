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
/// aarch64-linux-gnu, to copy to a Spark): the Hy4 kernels against their CPU references on a GPU, and the window-only whole-pack load (HY4_LOAD_PACK). The kernels come as
/// PTX at run time (HY4_HC_PTX: `nvcc -ptx zig/kernels/cuda/hy4/hc.cu`); without it the tests skip.
pub fn gpuTests(b: *std.Build) void {
    const triple = b.option([]const u8, "hy4-gpu-target", "Target of install-hy4-gpu-test (default: this host)");
    const target = if (triple) |t| b.resolveTargetQuery(std.Target.Query.parse(.{ .arch_os_abi = t }) catch @panic("bad -Dhy4-gpu-target")) else b.graph.host;
    const cuda = @import("cuda_runtime.zig").runtime(b, target, .ReleaseSafe, &.{});
    // core as zig/build/cuda.zig's family() makes it, for the checkpoint the whole-pack load reads
    const tokenizer = b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true });
    const living = b.createModule(.{ .root_source_file = b.path("zig/src/living/living.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true });
    const core = b.createModule(.{ .root_source_file = b.path("zig/src/core/root.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true });
    core.addImport("gpu_memory", @import("gpu_memory.zig").module(b, target));
    core.addImport("tokenizer", tokenizer);
    core.addImport("living", living);
    // vision_ref's act_ref through core, which owns the file
    const act_ref = b.createModule(.{ .root_source_file = b.path("zig/src/families/hy4/act_ref_core.zig"), .target = target, .optimize = .ReleaseSafe, .imports = &.{.{ .name = "core", .module = core }} });
    const m = b.createModule(.{ .root_source_file = b.path("zig/src/families/hy4/cuda_gpu_test.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core }, .{ .name = "act_ref", .module = act_ref } } });
    const exe = b.addTest(.{ .root_module = m, .name = "hy4-gpu-test" });
    b.step("test-hy4-gpu", "Hy4 CUDA kernels against their CPU references on this machine's GPU (HY4_HC_PTX)").dependOn(&b.addRunArtifact(exe).step);
    b.step("install-hy4-gpu-test", "Install the Hy4 GPU kernel test binary (see -Dhy4-gpu-target)").dependOn(&b.addInstallArtifact(exe, .{}).step);
    _ = forwardImage(b);
}

var forward_image: ?std.Build.LazyPath = null;
var forward_image_done = false;

/// zig/kernels/cuda/hy4/forward.cu (hc.cu included) as one fatbin: -Dhy4-nvcc compiles it (sm_121 SASS by -Dhy4-sm
/// plus compute_90 PTX for any newer GPU), -Dhy4-fatbin names a prebuilt one; `hy4-fatbin` installs it as
/// fatbin/hy4_forward.fatbin. Null with neither.
pub fn forwardImage(b: *std.Build) ?std.Build.LazyPath {
    if (forward_image_done) return forward_image;
    forward_image_done = true;
    const nvcc = b.option([]const u8, "hy4-nvcc", "nvcc that compiles the Hy4 forward kernels (zig/kernels/cuda/hy4/forward.cu)");
    const prebuilt = b.option([]const u8, "hy4-fatbin", "prebuilt fatbin of the Hy4 forward kernels (instead of -Dhy4-nvcc)");
    const sm = b.option([]const u8, "hy4-sm", "SASS target of the Hy4 forward kernels (default 121)") orelse "121";
    const out: std.Build.LazyPath = if (prebuilt) |file| b.graph.cwdRelativePath(file) else if (nvcc) |tool| blk: {
        const run = b.addSystemCommand(&.{ tool, "-fatbin", "-O3", "-std=c++17" });
        run.addArg(b.fmt("-gencode=arch=compute_{s},code=sm_{s}", .{ sm, sm }));
        run.addArg("-gencode=arch=compute_90,code=compute_90");
        run.addArg("-o");
        const o = run.addOutputFileArg("hy4_forward.fatbin");
        run.addFileArg(b.path("zig/kernels/cuda/hy4/forward.cu"));
        run.addFileInput(b.path("zig/kernels/cuda/hy4/hc.cu"));
        break :blk o;
    } else return null;
    b.step("hy4-fatbin", "Compile the Hy4 forward kernels (-Dhy4-nvcc) into fatbin/hy4_forward.fatbin").dependOn(&b.addInstallFile(out, "fatbin/hy4_forward.fatbin").step);
    forward_image = out;
    return out;
}

/// `hy4-serve`: tensorfold-hy4, the native server over the Hy4-flash engine (families/hy4/serve_engine.zig), on the
/// CUDA runtime `cuda` (which needs its images: -Dnvcc or -Dfatbins) and the forward image (-Dhy4-nvcc or -Dhy4-fatbin).
pub fn serve(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module, api: *std.Build.Module, tokenizer: *std.Build.Module, living: *std.Build.Module, build_options: *std.Build.Step.Options, runtime_images: bool, strip: ?bool) void {
    const step = b.step("hy4-serve", "Install tensorfold-hy4 (the native server with the Hy4-flash CUDA engine)");
    if (!runtime_images) return step.dependOn(&b.addFail("hy4-serve needs the CUDA runtime's images: -Dnvcc=PATH or -Dfatbins=DIR").step);
    const image = forwardImage(b) orelse return step.dependOn(&b.addFail("hy4-serve needs -Dhy4-nvcc=PATH or -Dhy4-fatbin=FILE").step);
    const resample = b.createModule(.{ .root_source_file = b.path("zig/src/families/hy4/resample_core.zig"), .target = target, .optimize = optimize, .imports = &.{.{ .name = "core", .module = core }} });
    const engine = b.createModule(.{
        .root_source_file = b.path("zig/src/families/hy4/serve_engine.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "engine_api", .module = api }, .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core }, .{ .name = "lanes", .module = lanes }, .{ .name = "picture_resample", .module = resample } },
    });
    engine.addAnonymousImport("hy4_forward_fatbin", .{ .root_source_file = image });
    const cuda_build = @import("cuda.zig");
    const template = b.createModule(.{ .root_source_file = b.path("zig/src/core/template/template.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true });
    const exe = b.addExecutable(.{ .name = "tensorfold-hy4", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/server/main.zig"),
        .target = target,
        .optimize = .ReleaseSafe,
        .link_libc = true,
        .imports = &.{ .{ .name = "core", .module = core }, .{ .name = "engine_api", .module = api }, .{ .name = "tokenizer", .module = tokenizer }, .{ .name = "template", .module = template }, .{ .name = "native_engines", .module = engine }, .{ .name = "living", .module = living }, .{ .name = "checkpoint_cli", .module = cuda_build.checkpointCli(b, target, .ReleaseSafe, null, engine, core) } },
    }) });
    exe.root_module.addOptions("build_options", build_options);
    _ = @import("server.zig").addTo(b, exe.root_module, target, .ReleaseSafe);
    exe.root_module.strip = strip;
    step.dependOn(&b.addInstallArtifact(exe, .{}).step);
}
