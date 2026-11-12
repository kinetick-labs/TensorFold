//! CPU checks for halt ordering, with the real lane host and fake target.
const std = @import("std");

pub fn steps(b: *std.Build) void {
    const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = b.graph.host, .optimize = .debug, .link_libc = true });
    const api = b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = b.graph.host, .optimize = .debug, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes }} });
    const interrupted = b.createModule(.{ .root_source_file = b.path("zig/src/core/lane_host_interrupted_test.zig"), .target = b.graph.host, .optimize = .debug, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes }} });
    b.step("test-lane-interrupted-host", "Interrupted prompt retention and clean retries without devices").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = interrupted, .filters = &.{"interrupted prefill:"} })).step);
    const safe_lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = b.graph.host, .optimize = .safe, .link_libc = true });
    const strand = b.createModule(.{ .root_source_file = b.path("zig/src/core/lane_host_strand_test.zig"), .target = b.graph.host, .optimize = .safe, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = safe_lanes }} });
    b.step("test-lane-strand-host", "Refused, cancelled and failed requests give back every lane, page and slot without devices").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = strand, .filters = &.{"lane strand:"} })).step);
    const halt = b.addTest(.{ .root_module = api, .filters = &.{ "halt (", "halt fails" } });
    b.step("test-lane-halt-host", "Halt ordering with a fake backend and no GPU dependencies").dependOn(&b.addRunArtifact(halt).step);
    b.step("test-lane-lone-host", "Existing lone-driver handoff and reclaim with a fake backend").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = api, .filters = &.{"lone driver"} })).step);
    b.step("test-lane-api-host", "Shared lane host and engine API CPU regressions").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = api })).step);
    b.step("test-conversation-cache-host", "Returning conversation eviction and queued-resume regressions without devices").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = api, .filters = &.{ "newest cache:", "queued cache:" } })).step);
    b.step("test-cache-progress-host", "One-state follow-ups and new conversations keep their next endpoint").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = api, .filters = &.{"queued cache: progress"} })).step);
    b.step("test-snapshot-index-host", "Prompt-cache lookup index against the whole-store scan, without devices").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = api, .filters = &.{"snapshot index:"} })).step);
    b.step("test-rows-prefetch-host", "Disk states read ahead for queued requests: equality, cancellation, fallbacks").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = api, .filters = &.{"prefetch:"} })).step);
    b.step("test-embedding-lane-host", "Prompt-only embedding jobs on the actual lane worker without devices").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = api, .filters = &.{"embedding lane:"} })).step);
    b.step("test-prefill-service-host", "The opt-in prompt-row budget beside decoding streams on the lane host with a fake backend").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = api, .filters = &.{"prefill service:"} })).step);
    const hold = b.step("test-compute-hold-host", "Replies held at an EOS for a server tool's result, then resumed or ended, without devices");
    hold.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = lanes, .filters = &.{"compute hold:"} })).step);
    hold.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = api, .filters = &.{"compute hold:"} })).step);
    b.step("test-runner-abstain-host", "Abstain records: each abstained cell's source, reason and decision point").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = api, .filters = &.{"runner abstain records:"} })).step);
    b.step("test-settle-fire-host", "PREFIX_SYNC_GRAPH on the fake target: fires settled at their trigger never hold").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = lanes, .filters = &.{"settled fires:"} })).step);
    b.step("test-forced-rows-host", "A held fire's forced result lands whole on the fake target: no hold inside it").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = lanes, .filters = &.{"forced rows:"} })).step);
    b.step("test-budget-hold-host", "The thinking budget beside a held tool result on the fake target: the result lands whole, then the close").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = lanes, .filters = &.{"budget hold:"} })).step);
    const digit_stop = b.step("test-digit-stop-host", "The runner's digit stop after owned inline values: its rules, the fake target's rounds, sessions");
    digit_stop.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = lanes, .filters = &.{"runner digit stop"} })).step);
    digit_stop.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = api, .filters = &.{"runner digit stop"} })).step);
    b.step("test-runner-host", "The in-model runner: scanner, held results and delivery with a fake circuit backend").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = api, .filters = &.{"runner"} })).step);
    // proof 2 on the real circuit (TF_RUNNER_CIRCUIT): optimized, since a debug evaluator takes minutes a run
    const lanes_fast = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = b.graph.host, .optimize = .ReleaseSafe, .link_libc = true });
    const api_fast = b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = b.graph.host, .optimize = .ReleaseSafe, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes_fast }} });
    @import("runner_proof_shards.zig").steps(b, api_fast);
    const copies = b.step("test-root-copy-host", "Root copy precedence, drafter absorption and real worker checks without GPU");
    copies.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = lanes, .filters = &.{"root copy:"} })).step);
    copies.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = api, .filters = &.{"root copy:"} })).step);
}
