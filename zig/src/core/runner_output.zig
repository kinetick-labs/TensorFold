//! Encodes a settled run's output for its inline, fence or REPL delivery position.
const std = @import("std");
const api = @import("runner_api.zig");
const scan = @import("runner_scan.zig");
const Allocator = std.mem.Allocator;

pub const suffix = " @engine";
pub const Policy = struct { max_tokens: u32, inline_mark: bool = false };
pub const Text = struct {
    ctx: ?*anyopaque = null,
    bytes: *const fn (ctx: ?*anyopaque, token: u32) []const u8,
    encode: *const fn (ctx: ?*anyopaque, a: Allocator, text: []const u8) Allocator.Error![]u32,
};

pub fn value(out: []const u8) []const u8 {
    return if (std.mem.endsWith(u8, out, "\n")) out[0 .. out.len - 1] else out;
}

pub fn owned(gpa: Allocator, tx: Text, res: api.Result, w: scan.Trigger, policy: Policy) Allocator.Error![]u32 {
    var body: std.ArrayList(u8) = .empty;
    defer body.deinit(gpa);
    const out = value(res.out);
    if (w.inline_space) |space| {
        // only the value after `=`; past the cap it is the model's own text again
        try body.appendSlice(gpa, if (space) " " else "");
        try body.appendSlice(gpa, out);
        if (policy.inline_mark) try body.appendSlice(gpa, suffix);
        const ids = try tx.encode(tx.ctx, gpa, body.items);
        if (ids.len <= policy.max_tokens) return ids;
        gpa.free(ids);
        return try gpa.alloc(u32, 0);
    }
    if (w.terminal) try body.appendSlice(gpa, "\nResult:\n") else if (w.newline_before)
        try body.append(gpa, '\n');
    try body.appendSlice(gpa, out);
    switch (res.status) {
        .ok, .abstain => {},
        .budget => try body.print(gpa, "\n[stopped after {d} steps]", .{res.steps}),
        .fault => try body.appendSlice(gpa, "\n[machine fault]"),
    }
    const ids = try tx.encode(tx.ctx, gpa, body.items);
    defer gpa.free(ids);
    const tail = try tx.encode(tx.ctx, gpa, if (w.fenced) "\n```\n" else "\n");
    defer gpa.free(tail);
    const note = if (ids.len > policy.max_tokens) try tx.encode(tx.ctx, gpa, "\n[output cut]") else try gpa.alloc(u32, 0);
    defer gpa.free(note);
    const keep = ids[0..@min(ids.len, policy.max_tokens)];
    return std.mem.concat(gpa, u32, &.{ keep, note, tail });
}

/// The model sees the suffix; the computation scanner sees only the result's original bytes.
pub fn scanning(remaining: *?usize, bytes: []const u8) []const u8 {
    if (remaining.*) |*left| {
        const keep = @min(left.*, bytes.len);
        left.* -= keep;
        return bytes[0..keep];
    }
    return bytes;
}
test "runner: inline provenance scanner view excludes only forced metadata" {
    const a = std.testing.allocator;
    var scanner: scan.Scanner = .{ .cfg = .{ .inline_math = true } };
    defer scanner.deinit(a);
    _ = try scanner.feed(a, "47 * 83 =", true);
    var remaining: ?usize = 5;
    for ([_][]const u8{ " 39", "01 @", "engine" }) |chunk| {
        const kept = scanning(&remaining, chunk);
        if (kept.len > 0) _ = try scanner.feed(a, kept, true);
    }
    try std.testing.expectEqualStrings("47 * 83 = 3901", scanner.line.items);
    var model: ?usize = null;
    try std.testing.expectEqualStrings(" @engine", scanning(&model, " @engine"));
}
