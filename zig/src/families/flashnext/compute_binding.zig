//! Qwen3.8 Flash-Next's (`qwen4_exp`) binding of the compute lane: the generic TF_COMPUTE_LANE* names, the python tool.
const std = @import("std");
const lane = @import("engine_api").hold.compute_lane;

/// The native CUDA host boots it when the family declares `compute` (native/cuda.zig).
pub const binding: lane.Binding = .{};

var knobs: []const [2][]const u8 = &.{};
fn lookup(name: []const u8) ?[]const u8 {
    for (knobs) |kv| if (std.mem.eql(u8, kv[0], name)) return kv[1];
    return null;
}

test "Flash Next compute lane: the generic names only, the python tool" {
    knobs = &.{};
    try std.testing.expectEqual(null, try binding.parse(&lookup));
    knobs = &.{.{ "TF_DSV41_COMPUTE_LANE", "1" }}; // DeepSeek-V4.1's alias is not Flash-Next's
    try std.testing.expectEqual(null, try binding.parse(&lookup));
    knobs = &.{ .{ "TF_COMPUTE_LANE", "1" }, .{ "TF_COMPUTE_LANE_WORKERS", "2" } };
    try std.testing.expectEqual(@as(u8, 2), (try binding.parse(&lookup)).?.workers);
    knobs = &.{ .{ "TF_COMPUTE_LANE", "1" }, .{ "TF_COMPUTE_LANE_DEFAULT", "maybe" } };
    try std.testing.expectError(error.BadKnob, binding.parse(&lookup));
    try std.testing.expectEqualStrings("python", binding.tool.name);
    try std.testing.expectEqualStrings("code_interpreter", binding.tool.opt_in);
    try std.testing.expectEqual(@as(i64, 1024), binding.tool.min_tokens);
}
