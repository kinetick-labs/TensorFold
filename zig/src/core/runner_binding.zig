//! A family's binding of the in-model runner: the knob prefix it boots from, generic TF_RUNNER* or its own older names.
const std = @import("std");
const runner = @import("runner.zig");
const log = std.log.scoped(.runner);

pub const Binding = struct {
    prefix: []const u8 = "TF_RUNNER", // PREFIX=1 enables; PREFIX_CIRCUIT, PREFIX_FORMS, ... (runner.Config.parse)
    alias: ?[]const u8 = null, // the family's own older names for the same knobs: a boot switches one or the other on

    /// The runner's config under the prefix whose switch the boot set; both switches set is error.BadKnob.
    pub fn parse(comptime b: Binding, get: *const fn ([]const u8) ?[]const u8) error{BadKnob}!?runner.Config {
        // by the switch alone: TF_RUNNER_CIRCUIT, _CLANG and _LLD also name tools for tests and image checks
        const old = b.alias orelse return runner.Config.parse(get, b.prefix);
        if (!set(get(old))) return runner.Config.parse(get, b.prefix);
        if (set(get(b.prefix))) {
            log.warn("turn the in-model runner on with {s} or {s}, not both", .{ b.prefix, old });
            return error.BadKnob;
        }
        return runner.Config.parse(get, old);
    }

    /// The prefix a boot's knobs are read under: the alias when its switch is set, else the generic one.
    pub fn active(comptime b: Binding, get: *const fn ([]const u8) ?[]const u8) []const u8 {
        const old = b.alias orelse return b.prefix;
        return if (set(get(old))) old else b.prefix;
    }
};

fn set(v: ?[]const u8) bool {
    return if (v) |s| s.len > 0 else false;
}

/// DeepSeek-V4.1: the generic names, or TF_DSV41_RUNNER* as before.
pub const dsv41: Binding = .{ .alias = "TF_DSV41_RUNNER" };
/// Qwen3.8 Flash-Next (`qwen4_exp`): the generic names only. Its serve engine does not read it yet.
pub const flashnext: Binding = .{};

const t = std.testing;

const Env = struct {
    var pairs: []const [2][]const u8 = &.{};
    fn get(name: []const u8) ?[]const u8 {
        for (pairs) |p| if (std.mem.eql(u8, p[0], name)) return p[1];
        return null;
    }
};

test "runner binding: the generic names or the family's alias, never both switches" {
    Env.pairs = &.{ .{ "TF_DSV41_RUNNER", "1" }, .{ "TF_DSV41_RUNNER_CIRCUIT", "/c.tfpc" }, .{ "TF_RUNNER_CIRCUIT", "/test.tfpc" } };
    const old = (try dsv41.parse(&Env.get)).?;
    try t.expectEqualStrings("/c.tfpc", old.circuit);
    try t.expectEqualStrings("TF_DSV41_RUNNER", dsv41.active(&Env.get));
    Env.pairs = &.{ .{ "TF_RUNNER", "1" }, .{ "TF_RUNNER_CIRCUIT", "/g.tfpc" }, .{ "TF_RUNNER_FORMS", "inline" } };
    const generic = (try dsv41.parse(&Env.get)).?;
    try t.expectEqualStrings("/g.tfpc", generic.circuit);
    try t.expect(generic.scan.inline_math and !generic.scan.fence);
    try t.expectEqualStrings("TF_RUNNER", dsv41.active(&Env.get));
    try t.expectEqualStrings("/g.tfpc", (try flashnext.parse(&Env.get)).?.circuit);
    Env.pairs = &.{ .{ "TF_RUNNER", "1" }, .{ "TF_DSV41_RUNNER", "1" }, .{ "TF_RUNNER_CIRCUIT", "/g.tfpc" } };
    try t.expectError(error.BadKnob, dsv41.parse(&Env.get));
    Env.pairs = &.{.{ "TF_RUNNER_CIRCUIT", "/test.tfpc" }};
    try t.expect((try dsv41.parse(&Env.get)) == null);
    try t.expect((try flashnext.parse(&Env.get)) == null);
}
