//! The image-input module (vision/qwen_image.zig), shared by the engine API and the CUDA test runner.
const std = @import("std");

pub fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/vision/qwen_image.zig"), .target = target, .optimize = optimize, .link_libc = true });
}

/// CPU Flash-Next image preprocess (smart_resize / AA bicubic / patchify); no GPU.
pub fn hostTests(b: *std.Build, test_step: *std.Build.Step) void {
    const host = b.graph.host;
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{
        .root_module = b.createModule(.{ .root_source_file = b.path("zig/src/vision/qwen_preprocess.zig"), .target = host, .optimize = .Debug }),
    })).step);
}
