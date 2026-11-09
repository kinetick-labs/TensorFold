//! The in-model runner's tokenizer: every token's bytes once at start, and plain encoding of owned result text.
const std = @import("std");
const api = @import("engine_api");
const model_text = @import("model_text.zig");
const Allocator = std.mem.Allocator;

const Table = struct { text: model_text.Text, bytes: [][]const u8 };

/// Gives `r` the server's tokenizer (allocated from `arena`); without token bytes the runner stays off.
pub fn attach(arena: Allocator, text: model_text.Text, r: *api.runner.Runner) !void {
    if (text.vtable.token_bytes == null) return std.log.warn("in-model runner off: this tokenizer cannot give token bytes", .{});
    const t = try arena.create(Table);
    t.* = .{ .text = text, .bytes = try arena.alloc([]const u8, text.vocabSize()) };
    // a token with no bytes of its own (an added token) is never part of a cell's code
    for (t.bytes, 0..) |*b, i| b.* = (try text.tokenBytes(arena, @intCast(i))) orelse "";
    r.text = .{ .ctx = t, .bytes = bytes, .encode = encode };
}

fn bytes(ctx: ?*anyopaque, token: u32) []const u8 {
    const t: *const Table = @ptrCast(@alignCast(ctx.?));
    return if (token < t.bytes.len) t.bytes[token] else "";
}

/// Plain text: a result spelling a special token stays text, never the token.
fn encode(ctx: ?*anyopaque, a: Allocator, text: []const u8) Allocator.Error![]u32 {
    const t: *const Table = @ptrCast(@alignCast(ctx.?));
    return t.text.encodePlain(a, text) catch |e| switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        else => error.OutOfMemory, // the reply fails rather than owning text it could not encode
    };
}
