//! Hy4's own tokenizer through HfText: decoded replies come out in GLM's markup and `</think>` finds the token the
//! model closes its thinking with. On demand: HY4_PACK_DIR (or any directory with the release's tokenizer files).
const std = @import("std");
const hf_text = @import("hf_text.zig");

test "Hy4 reply: the release tokenizer's tags decode as GLM markup and </think> is </think:opensource>" {
    const dir = std.testing.environ.getPosix("HY4_PACK_DIR") orelse return error.SkipZigTest;
    const gpa = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var problem: []const u8 = "";
    const t = hf_text.HfText.load(gpa, std.testing.io, dir, null, a, &problem) catch |e| {
        std.debug.print("{s}\n", .{problem});
        return e;
    };
    defer t.deinit();
    try std.testing.expect(t.hy4_reply);
    const text = t.text();
    try std.testing.expectEqual(@as(?u32, 120030), text.tokenId("</think>"));
    try std.testing.expectEqual(@as(?u32, 120033), text.tokenId("<tool_call>"));
    const name = try text.encode(a, "get_weather", false);
    const key = try text.encode(a, "city", false);
    const value = try text.encode(a, "Paris", false);
    const ids = try std.mem.concat(a, u32, &.{ &.{ 120029, 120030, 120035, 120033 }, name, &.{120040}, key, &.{ 120041, 120042 }, value, &.{ 120043, 120034, 120036 } });
    try std.testing.expectEqualStrings("<think></think><tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>", try text.decode(a, ids));
}
