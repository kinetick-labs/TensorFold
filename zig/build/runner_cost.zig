//! The in-model runner's host benches: real replies through a reply session (cost per token; a hold's parts), no GPU.
const std = @import("std");

pub fn steps(b: *std.Build) void {
    const target = b.graph.host;
    const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = target, .optimize = .ReleaseFast, .link_libc = true });
    const core_tok = b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = target, .optimize = .ReleaseFast });
    const tok = b.createModule(.{ .root_source_file = b.path("zig/src/families/deepseek_v41/serve/tokenizer.zig"), .target = target, .optimize = .ReleaseFast, .imports = &.{.{ .name = "tokenizer", .module = core_tok }} });
    const root = b.createModule(.{ .root_source_file = b.path("zig/src/core/runner_cost_bench.zig"), .target = target, .optimize = .ReleaseFast, .link_libc = true, .imports = &.{ .{ .name = "lanes", .module = lanes }, .{ .name = "dsv41_tokenizer", .module = tok } } });
    const exe = b.addExecutable(.{ .name = "runner-cost-bench", .root_module = root });
    b.step("bench-runner-cost", "Runner host cost per token and per fire on real replies (tokenizer.json, replies.jsonl)").dependOn(&b.addInstallArtifact(exe, .{}).step);
    const hold = b.createModule(.{ .root_source_file = b.path("zig/src/core/runner_hold_bench.zig"), .target = target, .optimize = .ReleaseFast, .link_libc = true, .imports = &.{ .{ .name = "lanes", .module = lanes }, .{ .name = "dsv41_tokenizer", .module = tok } } });
    const hold_exe = b.addExecutable(.{ .name = "runner-hold-bench", .root_module = hold });
    b.step("bench-runner-hold", "A runner hold's parts through the served runner (TF_RUNNER* knobs; tokenizer.json, replies.jsonl)").dependOn(&b.addInstallArtifact(hold_exe, .{}).step);
}
