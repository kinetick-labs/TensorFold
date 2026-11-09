//! A family's binding of the compute lane: the knob names it boots from and the built-in tool its requests see.
const std = @import("std");
const lane = @import("compute_lane.zig");
const log = std.log.scoped(.compute_lane);

/// The built-in tool as the server offers it.
pub const Tool = struct {
    name: []const u8 = "python", // the function the model calls
    opt_in: []const u8 = "code_interpreter", // the tool type a request lists to ask for it
    min_tokens: i64 = 1024, // under this max_tokens the compute default offers it to no one unasked
};

pub const Binding = struct {
    prefix: []const u8 = "TF_COMPUTE_LANE", // PREFIX=1 enables; PREFIX_DEFAULT, PREFIX_WORKERS, ... (compute_lane.Config)
    alias: ?[]const u8 = null, // the family's own older names for the same knobs: a boot sets one or the other
    tool: Tool = .{},

    /// The lane's config from the names the boot set; knobs under both the prefix and the alias are error.BadKnob.
    pub fn parse(comptime b: Binding, get: *const fn ([]const u8) ?[]const u8) error{BadKnob}!?lane.Config {
        const old = b.alias orelse return lane.Config.parse(get, b.prefix);
        if (!anySet(get, old)) return lane.Config.parse(get, b.prefix);
        if (anySet(get, b.prefix)) {
            log.warn("set the compute lane under {s}* or {s}*, not both", .{ b.prefix, old });
            return error.BadKnob;
        }
        return lane.Config.parse(get, old);
    }
};

fn anySet(get: *const fn ([]const u8) ?[]const u8, comptime prefix: []const u8) bool {
    inline for (lane.Config.knobs) |k| if (get(prefix ++ k)) |v| if (v.len > 0) return true;
    return false;
}

/// DeepSeek-V4.1: the generic names, or TF_DSV41_COMPUTE_LANE* as before.
pub const dsv41: Binding = .{ .alias = "TF_DSV41_COMPUTE_LANE" };
/// Qwen3.8 Flash-Next (`qwen4_exp`): the generic names only. Its serve engine does not read it yet.
pub const flashnext: Binding = .{};
