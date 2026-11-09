//! Owned bytes through each call format's strip points: Qwen's JSON and XML calls (Flash-Next) and DSML (DSV41).
const std = @import("std");
const json = @import("json");
const reply_text = @import("reply_text.zig");
const reply_owned = @import("reply_owned.zig");
const tool_parse = @import("tool_parse.zig");
const tool_stream = @import("tool_stream.zig");
const Owned = reply_owned.Owned;
const t = std.testing;

const prose = "The value is 2^66 = 73786976294838206464 \n";
const answer = "73786976294838206464 "; // the cell's write, its trailing blank the engine's too
const qwen_json = "<tool_call>\n{\"name\": \"f\", \"arguments\": {\"x\": \"1\"}}\n</tool_call>";
const qwen_xml = "<tool_call>\n<function=f>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>";
const dsml = "<\u{ff5c}DSML\u{ff5c}tool_calls>\n<\u{ff5c}DSML\u{ff5c}invoke name=\"f\">\n<\u{ff5c}DSML\u{ff5c}parameter name=\"x\" " ++
    "string=\"true\">1</\u{ff5c}DSML\u{ff5c}parameter>\n</\u{ff5c}DSML\u{ff5c}invoke>\n</\u{ff5c}DSML\u{ff5c}tool_calls>";
const formats = [_][]const u8{ qwen_json, qwen_xml, dsml };
const tools_json =
    \\[{"type":"function","function":{"name":"f","parameters":{"type":"object","properties":{"x":{"type":"string"}}}}}]
;

fn span() Owned {
    const lo = std.mem.indexOf(u8, prose, answer).?;
    return .{ .lo = lo, .hi = lo + answer.len };
}

test "reply owned: every call format strips its prose up to the owned bytes, never into them" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const o = reply_text.shared_openers;
    const kept = prose[0..span().hi];
    inline for (formats) |call| {
        const text = prose ++ call;
        try t.expectEqual(@as(?usize, prose.len), try reply_owned.callStart(a, text, o));
        try t.expectEqualStrings(kept, try reply_owned.finishedProse(a, text, o, span()));
        try t.expectEqualStrings(kept, try reply_owned.cutProse(a, text[0 .. text.len - 5], o, span()));
        // nothing owned: the strips as before, the trailing blank gone
        try t.expectEqualStrings(try reply_text.finishedProse(a, text), try reply_owned.finishedProse(a, text, o, null));
        try t.expectEqualStrings(try reply_text.cutProse(a, text), try reply_owned.cutProse(a, text, o, null));
        try t.expectEqualStrings(kept[0 .. kept.len - 1], try reply_owned.finishedProse(a, text, o, null));
        const tools = (try json.parse(a, tools_json)).ok.array;
        const end = try tool_stream.endProseOwned(a, text, text, tools, "stop", true, o, span());
        try t.expectEqualStrings(kept, end);
        try t.expectEqualStrings(try tool_stream.endProse(a, text, text, tools, "stop", false), try tool_stream.endProseOwned(a, text, text, tools, "stop", false, o, null));
    }
}

test "reply owned: the parser keeps owned bytes and gives them in the content's offsets, for each format" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, tools_json)).ok.array;
    inline for (formats) |call| {
        const text = "  " ++ prose ++ call;
        const owned = Owned.shift(span(), 2);
        const p = try tool_parse.parseOwned(a, text, tools, null, owned);
        try t.expectEqual(@as(usize, 1), p.calls.?.len);
        try t.expectEqualStrings(prose[0..span().hi], p.content);
        try t.expectEqual(span(), p.owned.?);
        const plain = try tool_parse.parse(a, text, tools, null);
        try t.expectEqualStrings(std.mem.trimEnd(u8, prose, " \n"), plain.content);
        try t.expectEqual(null, plain.owned);
    }
}

test "reply owned: a strip never splits a code point; a span past the text is not its offsets" {
    // the span ends inside U+3000 (a blank): the trim keeps the whole code point, not its first byte
    try t.expectEqualStrings("ab\u{3000}", reply_owned.strip("ab\u{3000}", .{ .lo = 0, .hi = 3 }));
    try t.expectEqualStrings("ab", reply_owned.strip("ab\u{3000}", null));
    try t.expectEqualStrings(" ab", reply_owned.strip("  ab", .{ .lo = 1, .hi = 4 })); // an owned leading blank
    try t.expectEqualStrings("ab", reply_owned.strip("ab  ", .{ .lo = 0, .hi = 40 }));
    try t.expectEqual(Owned{ .lo = 1, .hi = 9 }, Owned.join(.{ .lo = 4, .hi = 9 }, .{ .lo = 1, .hi = 2 }).?);
    try t.expectEqual(null, Owned.shift(null, 3));
}

fn record(a: std.mem.Allocator, field: []const u8, token: usize, landed: usize) ![]const u8 {
    return std.fmt.allocPrint(a, "{{\"cell\":0,\"field\":\"{s}\",\"token\":{d},\"landed\":{d}}}", .{ field, token, landed });
}

test "reply owned: a cell write's tokens become its bytes in the content, before the format's first call" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const o = reply_text.shared_openers;
    inline for (formats) |call| {
        const pieces = [_][]const u8{ "plan", "</think>", "[2^66 = ", "73786976294838206464]", "\n", call };
        const raw = try std.mem.concat(a, u8, &pieces);
        const content = raw["plan</think>".len..];
        const write = try record(a, "content", 7, 3);
        try t.expectEqual(Owned{ .lo = 0, .hi = 30 }, (try reply_owned.of(a, &.{write}, 5, &pieces, raw, content, o)).?);
        // another segment's write (base past it), a reasoning write, an empty one, one past the tokens: none
        try t.expectEqual(null, try reply_owned.of(a, &.{write}, 8, &pieces, raw, content, o));
        try t.expectEqual(null, try reply_owned.of(a, &.{try record(a, "reasoning", 7, 3)}, 5, &pieces, raw, content, o));
        try t.expectEqual(null, try reply_owned.of(a, &.{try record(a, "content", 7, 0)}, 5, &pieces, raw, content, o));
        try t.expectEqual(null, try reply_owned.of(a, &.{try record(a, "content", 12, 1)}, 5, &pieces, raw, content, o));
        // a write in the call block owns nothing the prose keeps
        try t.expectEqual(null, try reply_owned.of(a, &.{try record(a, "content", 10, 1)}, 5, &pieces, raw, content, o));
        // content the server rewrote (not raw's bytes) owns nothing: offsets would not be its own
        const copy = try a.dupe(u8, content);
        try t.expectEqual(null, try reply_owned.of(a, &.{write}, 5, &pieces, raw, copy, o));
    }
}
