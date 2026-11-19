//! Hy4-flash's reply markup onto the GLM forms the server already parses. Hy4 writes `<think:opensource>`,
//! `<tool_calls:opensource><tool_call:opensource>name<arg_key:opensource>k</arg_key:opensource><arg_value:opensource>v
//! </arg_value:opensource></tool_call:opensource></tool_calls:opensource>` (single, non-special tokens 120029-120043,
//! so they reach the reply as text). Dropping the `:opensource` suffix and the `<tool_calls>` wrapper gives GLM-4.5's
//! adjacent `<tool_call>name<arg_key>` form, which the GLM tool parser and its streaming cut already handle.

const std = @import("std");

/// Hy4 tag, then what the GLM reply path expects in its place.
pub const tags = [_][2][]const u8{
    .{ "<think:opensource>", "<think>" },
    .{ "</think:opensource>", "</think>" },
    .{ "<tool_calls:opensource>", "" },
    .{ "</tool_calls:opensource>", "" },
    .{ "<tool_call:opensource>", "<tool_call>" },
    .{ "</tool_call:opensource>", "</tool_call>" },
    .{ "<arg_key:opensource>", "<arg_key>" },
    .{ "</arg_key:opensource>", "</arg_key>" },
    .{ "<arg_value:opensource>", "<arg_value>" },
    .{ "</arg_value:opensource>", "</arg_value>" },
};

/// The template's `reasoning_effort` for a request: it accepts `high` and `no_think` only and raises on anything
/// else, so thinking on (or unsaid) is `high` and thinking off is `no_think`.
pub fn reasoningEffort(thinking: ?bool) []const u8 {
    return if (thinking orelse true) "high" else "no_think";
}

/// Whether `text` could still grow into a tag: the longest suffix that is a proper prefix of one.
fn heldBack(text: []const u8) usize {
    const start = std.mem.lastIndexOfScalar(u8, text, '<') orelse return 0;
    const tail = text[start..];
    for (tags) |t| if (tail.len < t[0].len and std.mem.startsWith(u8, t[0], tail)) return tail.len;
    return 0;
}

/// Rewrites a whole reply.
pub fn toGlm(a: std.mem.Allocator, text: []const u8) std.mem.Allocator.Error![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    try rewrite(a, text, &out);
    return out.toOwnedSlice(a);
}

fn rewrite(a: std.mem.Allocator, text: []const u8, out: *std.ArrayList(u8)) std.mem.Allocator.Error!void {
    var i: usize = 0;
    outer: while (i < text.len) {
        if (text[i] == '<') for (tags) |t| if (std.mem.startsWith(u8, text[i..], t[0])) {
            try out.appendSlice(a, t[1]);
            i += t[0].len;
            continue :outer;
        };
        try out.append(a, text[i]);
        i += 1;
    }
}

/// The streaming form: each piece of decoded text in, the rewritten text that is final out; a piece that ends in
/// the start of a tag is held until the next one decides it.
pub const Stream = struct {
    pending: std.ArrayList(u8) = .empty,

    pub fn deinit(s: *Stream, a: std.mem.Allocator) void {
        s.pending.deinit(a);
    }

    pub fn feed(s: *Stream, a: std.mem.Allocator, piece: []const u8, out: *std.ArrayList(u8)) std.mem.Allocator.Error!void {
        try s.pending.appendSlice(a, piece);
        const keep = heldBack(s.pending.items);
        const ready = s.pending.items.len - keep;
        try rewrite(a, s.pending.items[0..ready], out);
        std.mem.copyForwards(u8, s.pending.items[0..keep], s.pending.items[ready..]);
        s.pending.shrinkRetainingCapacity(keep);
    }

    /// The end of the reply: a held partial tag is plain text after all.
    pub fn finish(s: *Stream, a: std.mem.Allocator, out: *std.ArrayList(u8)) std.mem.Allocator.Error!void {
        try out.appendSlice(a, s.pending.items);
        s.pending.clearRetainingCapacity();
    }
};
