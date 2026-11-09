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

    /// PREFIX_TRACE=1 (stderr) or a path: rounds around each runner fire (lanes round_trace); PREFIX_TRACE_ROUNDS: 12.
    pub fn openRoundTrace(comptime b: Binding, get: *const fn ([]const u8) ?[]const u8) error{BadKnob}!void {
        if (b.alias) |old| if (set(get(old))) return roundTrace(old, get);
        return roundTrace(b.prefix, get);
    }
};

fn roundTrace(comptime p: []const u8, get: *const fn ([]const u8) ?[]const u8) error{BadKnob}!void {
    const path = get(p ++ "_TRACE") orelse return;
    if (path.len == 0 or std.mem.eql(u8, path, "0")) return;
    const rounds = if (get(p ++ "_TRACE_ROUNDS")) |r| std.fmt.parseInt(u32, r, 10) catch {
        log.warn("{s}_TRACE_ROUNDS={s}: not a count of rounds", .{ p, r });
        return error.BadKnob;
    } else 12;
    var z: [std.fs.max_path_bytes:0]u8 = undefined;
    const fd: c_int = if (std.mem.eql(u8, path, "1")) 2 else if (path.len < z.len) blk: {
        @memcpy(z[0..path.len], path);
        z[path.len] = 0;
        break :blk std.c.open(z[0..path.len :0], .{ .ACCMODE = .WRONLY, .CREAT = true, .APPEND = true }, @as(std.c.mode_t, 0o644));
    } else -1;
    if (fd < 0) return log.warn("{s}_TRACE: cannot open {s}", .{ p, path });
    @import("lanes").round_trace.open(fd, rounds, null);
    log.info("in-model runner: round trace, {d} rounds around each fire to {s}", .{ rounds, if (fd == 2) "stderr" else path });
}

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

test "runner binding: the round trace stays off unless asked, and a bad round count is refused" {
    Env.pairs = &.{.{ "TF_DSV41_RUNNER", "1" }};
    try dsv41.openRoundTrace(&Env.get);
    Env.pairs = &.{ .{ "TF_DSV41_RUNNER", "1" }, .{ "TF_DSV41_RUNNER_TRACE", "0" } };
    try dsv41.openRoundTrace(&Env.get);
    try t.expect(!@import("lanes").round_trace.on());
    Env.pairs = &.{ .{ "TF_DSV41_RUNNER", "1" }, .{ "TF_DSV41_RUNNER_TRACE", "1" }, .{ "TF_DSV41_RUNNER_TRACE_ROUNDS", "x" } };
    try t.expectError(error.BadKnob, dsv41.openRoundTrace(&Env.get));
    Env.pairs = &.{ .{ "TF_RUNNER", "1" }, .{ "TF_DSV41_RUNNER_TRACE_ROUNDS", "x" } };
    try dsv41.openRoundTrace(&Env.get);
    try t.expect(!@import("lanes").round_trace.on());
}
