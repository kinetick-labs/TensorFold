//! Hy4-flash's reply markup as the GLM forms the server parses (families/hy4/reply.zig has the same table and its
//! streaming tests). Hy4 writes `<think:opensource>` and `<tool_call:opensource>name<arg_key:opensource>...` as
//! single non-special tokens (120029-120043); a decode always holds whole tokens, so rewriting each decoded text is
//! exact. Without the `:opensource` suffixes and the `<tool_calls>` wrapper the calls are GLM-4.5's adjacent
//! `<tool_call>name<arg_key>` form, which the GLM tool parser and its streaming cut already read.
const std = @import("std");

/// Hy4 tag, then the GLM text in its place.
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

/// The Hy4 tag a GLM form stands for (`</think>` -> `</think:opensource>`), so a lookup by the GLM name finds the
/// token the model writes; null for any other piece.
pub fn tagFor(glm: []const u8) ?[]const u8 {
    if (glm.len == 0) return null;
    for (tags) |t| if (std.mem.eql(u8, t[1], glm)) return t[0];
    return null;
}

/// `text` with every Hy4 tag replaced; the same slice when it has none.
pub fn rewrite(a: std.mem.Allocator, text: []u8) std.mem.Allocator.Error![]u8 {
    if (std.mem.indexOf(u8, text, ":opensource>") == null) return text;
    var out: std.ArrayList(u8) = try .initCapacity(a, text.len);
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
    return out.toOwnedSlice(a);
}

test "Hy4 reply: decoded tags become GLM's forms, other text is left as it is" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const hy4 = "<think:opensource>plan</think:opensource>ok<tool_calls:opensource><tool_call:opensource>get_weather<arg_key:opensource>city</arg_key:opensource><arg_value:opensource>Paris</arg_value:opensource></tool_call:opensource></tool_calls:opensource>";
    try std.testing.expectEqualStrings("<think>plan</think>ok<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>", try rewrite(a, try a.dupe(u8, hy4)));
    const plain = try a.dupe(u8, "a < b and <opensource> stays");
    try std.testing.expectEqual(plain.ptr, (try rewrite(a, plain)).ptr);
    try std.testing.expectEqualStrings("</think:opensource>", tagFor("</think>").?);
    try std.testing.expect(tagFor("<|im_end|>") == null and tagFor("") == null);
}
