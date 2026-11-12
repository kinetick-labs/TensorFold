//! The in-model runner's knob parsing (core/runner.zig's Config): a number in range, or a warning and error.BadKnob.
const std = @import("std");
const log = std.log.scoped(.runner);

pub fn knob(comptime T: type, get: *const fn ([]const u8) ?[]const u8, comptime name: []const u8, default: T, min: T, max: T) error{BadKnob}!T {
    const text = get(name) orelse return default;
    if (text.len == 0) return default;
    const v = std.fmt.parseInt(T, text, 10) catch return bad(name, text);
    if (v < min or v > max) return bad(name, text);
    return v;
}

pub fn bad(name: []const u8, value: []const u8) error{BadKnob} {
    log.warn("{s}={s} is not a valid value", .{ name, value });
    return error.BadKnob;
}
