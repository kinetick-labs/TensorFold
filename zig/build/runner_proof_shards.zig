//! One compiled proof population, executed in serial singleton server processes.
const std = @import("std");

pub fn steps(b: *std.Build, api_fast: *std.Build.Module) void {
    const tests = b.addTest(.{
        .root_module = api_fast,
        .filters = &.{"runner proof:"},
        .test_runner = .{ .path = b.path("zig/runner_proof_shard_runner.zig"), .mode = .server },
    });
    const all = b.step("test-runner-proof-host", "Multiply and loop replies on the real frontend and CPU circuit (TF_RUNNER_CIRCUIT)");
    const catalog = b.addRunArtifact(tests);
    catalog.setName("runner proof catalog");
    catalog.stdio = .inherit;
    catalog.addArg("--proof-catalog");
    catalog.has_side_effects = true;
    b.step("test-runner-proof-list", "List the compiled proof population without executing tests").dependOn(&catalog.step);
    var previous: *std.Build.Step = &catalog.step;
    for (0..256) |index| {
        const run = b.addRunArtifact(tests);
        run.setName(b.fmt("runner proof shard {d}", .{index}));
        run.addArg(b.fmt("--proof-shard-index={d}", .{index}));
        run.has_side_effects = true;
        run.step.dependOn(previous);
        previous = &run.step;
    }
    all.dependOn(previous);
}
