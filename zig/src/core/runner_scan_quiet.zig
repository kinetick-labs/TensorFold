//! A family's quiet spans: the text from an opener to its closer (a tool call, a tool's result) is never a cell's.
const std = @import("std");

/// Opener and closer pairs, each at most `max` bytes (runner_binding.Family.quiet).
pub const Spans = []const [2][]const u8;
pub const max = 32;

/// Where the reply stands: the last bytes seen (markers may span tokens), and the span it is inside.
pub const Quiet = struct {
    tail: [max]u8 = undefined,
    len: u8 = 0,
    in: ?usize = null, // the span whose closer ends the quiet

    /// Whether byte `b` is quiet: inside a span, or its closer's; an opener's own bytes before it were prose.
    pub fn skip(q: *Quiet, spans: Spans, b: u8) bool {
        if (spans.len == 0) return false;
        if (q.len == max) {
            std.mem.copyForwards(u8, q.tail[0 .. max - 1], q.tail[1..max]);
            q.len -= 1;
        }
        q.tail[q.len] = b;
        q.len += 1;
        const seen = q.tail[0..q.len];
        if (q.in) |i| {
            if (std.mem.endsWith(u8, seen, spans[i][1])) q.in = null;
            return true;
        }
        for (spans, 0..) |s, i| if (std.mem.endsWith(u8, seen, s[0])) {
            q.in = i;
            return true;
        };
        return false;
    }
};

const t = std.testing;

fn prose(q: *Quiet, spans: Spans, text: []const u8, out: *std.ArrayList(u8)) !void {
    for (text) |b| if (!q.skip(spans, b)) try out.append(t.allocator, b);
}

test "runner quiet: a span's body and closer never reach the scanner, across token boundaries" {
    const spans: Spans = &.{ .{ "<tool_call>", "</tool_call>" }, .{ "<tool_response>", "</tool_response>" } };
    var q: Quiet = .{};
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(t.allocator);
    for ([_][]const u8{ "x = 2*3 =\n<tool", "_call>\n<function=f>\n<parameter=a>\n7*8 =\n</parameter>\n</function>\n</tool", "_call>\nok\n" }) |piece|
        try prose(&q, spans, piece, &out);
    try t.expectEqualStrings("x = 2*3 =\n<tool_call\nok\n", out.items);
    out.clearRetainingCapacity();
    try prose(&q, spans, "<tool_response>\n{\"r\": \"1+1 =\"}\n</tool_response>after", &out);
    try t.expectEqualStrings("<tool_responseafter", out.items);
    var none: Quiet = .{};
    try t.expect(!none.skip(&.{}, '<'));
}
