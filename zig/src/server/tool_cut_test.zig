//! A reply that max_tokens cuts inside a tool call: the whole reply gives what the stream gave (#1238).
const std = @import("std");
const api = @import("engine_api");
const pre = @import("stream_preflight_test.zig");
const TestText = pre.TestText;
const CallEngine = pre.CallEngine;
const sendWith = pre.sendWith;

const head = "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]," ++
    "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"write_file\",\"parameters\":{\"type\":\"object\"," ++
    "\"properties\":{\"path\":{\"type\":\"string\"},\"content\":{\"type\":\"string\"},\"limit\":{\"type\":\"integer\"}}}}}]";
const opened = "<tool_call>\n<function=write_file>\n<parameter=path>\nnotes.txt\n</parameter>\n<parameter=content>\nThe history of";
const rest = " the bicycle\n</parameter>\n</function>\n</tool_call>";
const glm_path = "<tool_call>write_file<arg_key>path</arg_key><arg_value>notes.txt</arg_value>";
const glm_opened = glm_path ++ "<arg_key>content</arg_key><arg_value>The history of";
const glm_rest = " the bicycle</arg_value></tool_call>";

/// ``cut`` then ``rest``, max_tokens at ``cut``'s end: whole and streamed replies agree (null ``content``: unchecked).
fn expectCut(cut: []const u8, calls: usize, arguments: []const u8, content: ?[]const u8, extra: []const u8) !void {
    return expectCutBefore(rest, cut, calls, arguments, content, extra);
}

/// ``expectCut`` of a GLM reply (``NAME<arg_key>K</arg_key><arg_value>V</arg_value>``) that goes on with ``glm_rest``.
fn expectGlmCut(cut: []const u8, calls: usize, arguments: []const u8, content: ?[]const u8, extra: []const u8) !void {
    return expectCutBefore(glm_rest, cut, calls, arguments, content, extra);
}

fn expectCutBefore(after: []const u8, cut: []const u8, calls: usize, arguments: []const u8, content: ?[]const u8, extra: []const u8) !void {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tokens = try TestText.encode(undefined, a, try std.mem.concat(a, u8, &.{ cut, after }), false);
    const body = try std.fmt.allocPrint(a, "{s},\"max_tokens\":{d}{s}", .{ head, cut.len, extra });
    for ([_]bool{ false, true }) |batched| {
        var backend: CallEngine = .{ .tokens = tokens, .batched = batched };
        const whole = try sendWith(backend.engine(), .chat, try std.mem.concat(a, u8, &.{ body, ",\"stream\":false}" }), .{});
        const streamed = try sendWith(backend.engine(), .chat, try std.mem.concat(a, u8, &.{ body, ",\"stream\":true}" }), .{});
        try std.testing.expectEqual(@as(?u16, 200), whole.status);
        try std.testing.expect(streamed.opened and streamed.status == null);
        for ([_]@TypeOf(whole){ whole, streamed }) |seen| {
            try std.testing.expectEqual(calls, seen.calls);
            try std.testing.expectEqualStrings(arguments, seen.arguments[0..seen.arguments_len]);
            if (content) |c| try std.testing.expectEqualStrings(c, seen.content());
            if (std.mem.indexOf(u8, seen.content(), "<function=") != null) return error.CallMarkupInContent;
            try std.testing.expectEqual(.length, seen.finish.?);
        }
    }
}

test "a call max_tokens cuts is the whole reply's tool call, cut, as the stream's is" {
    const args = "{\"path\":\"notes.txt\",\"content\":\"The history of";
    try expectCut(opened, 1, args, "", "");
    try expectCut("Writing it.\n" ++ opened, 1, args, "Writing it.", "");
    // a whole call first: both calls, the second cut
    const first = "<tool_call>\n<function=write_file>\n<parameter=path>\na.txt\n</parameter>\n</function>\n</tool_call>\n";
    try expectCut(first ++ opened, 2, "{\"path\":\"a.txt\"}" ++ args, "", "");
    // cut between parameters, and after the function closed but before the envelope did
    try expectCut("<tool_call>\n<function=write_file>\n<parameter=path>\nnotes.txt\n</parameter>\n", 1, "{\"path\":\"notes.txt\"", "", "");
    try expectCut("<tool_call>\n<function=write_file>\n<parameter=path>\nnotes.txt\n</parameter>\n</function>\n", 1, "{\"path\":\"notes.txt\"}", "", "");
}

test "the one-call policy and a cut before any call is named keep their replies" {
    // parallel_tool_calls false hides the cut call in both, as before
    try expectCut("Writing it.\n" ++ opened, 0, "", null, ",\"parallel_tool_calls\":false");
    // an envelope with no function named yet is no call in either reply
    try expectCut("<tool_call>\n", 0, "", "<tool_call>\n", "");
}

test "a GLM call max_tokens cuts is the whole reply's tool call, cut, as an XML call is (#1298)" {
    const args = "{\"path\":\"notes.txt\",\"content\":\"The history of";
    try expectGlmCut(glm_opened, 1, args, "", "");
    try expectGlmCut("Writing it.\n" ++ glm_opened, 1, args, "Writing it.", "");
    // the issue's reply: one argument, its value cut
    try expectGlmCut("<tool_call>write_file<arg_key>content</arg_key><arg_value>The keeper of", 1, "{\"content\":\"The keeper of", "", "");
    // a whole call first: both calls, the second cut
    const first = "<tool_call>write_file<arg_key>path</arg_key><arg_value>a.txt</arg_value></tool_call>\n";
    try expectGlmCut(first ++ glm_opened, 2, "{\"path\":\"a.txt\"}" ++ args, "", "");
    // cut after a value, after a key, inside a value's opener or a key, and inside a value's closer
    inline for (.{ "", "<arg_key>content</arg_key>", "<arg_key>content</arg_key><arg_val", "<arg_key>cont" }) |tail|
        try expectGlmCut(glm_path ++ tail, 1, "{\"path\":\"notes.txt\"", "", "");
    try expectGlmCut(glm_opened ++ "</arg_val", 1, args, "", "");
    // a value its schema types is JSON, whole or cut
    const limit = "<tool_call>write_file<arg_key>limit</arg_key><arg_value>12";
    try expectGlmCut(limit, 1, "{\"limit\":12", "", "");
    try expectGlmCut(limit ++ "</arg_value><arg_key>content</arg_key><arg_value>The history of", 1, "{\"limit\":12,\"content\":\"The history of", "", "");
}

test "a cut GLM call keeps the one-call policy, and one not yet named or not offered stays text" {
    try expectGlmCut("Writing it.\n" ++ glm_opened, 0, "", null, ",\"parallel_tool_calls\":false");
    try expectGlmCut("<tool_call>write_file", 0, "", "<tool_call>write_file", "");
    const unoffered = "<tool_call>launch<arg_key>x</arg_key><arg_value>1";
    try expectGlmCut(unoffered, 0, "", unoffered, "");
}

test "a JSON call max_tokens cuts is the whole reply's tool call, cut, as an XML call is" {
    const named = "<tool_call>\n{\"name\": \"write_file\", \"arguments\": ";
    const args = "{\"path\": \"notes.txt\", \"content\": \"The history of";
    try expectCutBefore(" the bicycle\"}}\n</tool_call>", named ++ args, 1, args, "", "");
    try expectCutBefore(" the bicycle\"}}\n</tool_call>", "Writing it.\n" ++ named ++ args, 1, args, "Writing it.", "");
    // a whole call first: both calls, the second cut
    const first = "<tool_call>\n{\"name\": \"write_file\", \"arguments\": {\"path\": \"a.txt\"}}\n</tool_call>\n";
    const both = "{\"path\":\"a.txt\"}" ++ args;
    try expectCutBefore(" the bicycle\"}}\n</tool_call>", first ++ named ++ args, 2, both, "", "");
    // arguments whole, the payload or only the envelope not yet closed: spelled as the closed call spells them
    try expectCutBefore("}\n</tool_call>", named ++ "{\"path\": \"a.txt\"}", 1, "{\"path\":\"a.txt\"}", "", "");
    try expectCutBefore("\n</tool_call>", named ++ "{\"path\": \"a.txt\"}}", 1, "{\"path\":\"a.txt\"}", "", "");
}

test "a cut JSON call reads parameters, and arguments written before the name" {
    const params = "<tool_call>\n{\"name\": \"write_file\", \"parameters\": {\"path\": \"no";
    try expectCutBefore("tes.txt\"}}\n</tool_call>", params, 1, "{\"path\": \"no", "", "");
    const first = "<tool_call>\n{\"arguments\": {\"path\": \"a.txt\"}, \"name\": \"write_file\"";
    try expectCutBefore("}\n</tool_call>", first, 1, "{\"path\":\"a.txt\"}", "", "");
    // the arguments' own "name" is not the call's: cut before the call's, the reply is text
    const unnamed = "<tool_call>\n{\"arguments\": {\"name\": \"write_file\", \"path\": \"a";
    try expectCutBefore(".txt\"}, \"name\": \"write_file\"}\n</tool_call>", unnamed, 0, "", unnamed, "");
}

test "a cut JSON call keeps the one-call policy, and one not yet named or not offered stays text" {
    const cut = "Writing it.\n<tool_call>\n{\"name\": \"write_file\", \"arguments\": {\"path\": \"no";
    try expectCutBefore("tes.txt\"}}\n</tool_call>", cut, 0, "", null, ",\"parallel_tool_calls\":false");
    try expectCutBefore("e\": \"write_file\"}\n</tool_call>", "<tool_call>\n{\"nam", 0, "", "<tool_call>\n{\"nam", "");
    const unoffered = "<tool_call>\n{\"name\": \"launch\", \"arguments\": {\"x\": 1";
    try expectCutBefore("}}\n</tool_call>", unoffered, 0, "", unoffered, "");
}

test "a chunked XML cut drops only the pending framing newline" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const before = "<tool_call>\n<function=write_file>\n<parameter=path>\nnotes.txt\n</parameter>\n<parameter=content>";
    for ([_][]const u8{ "", "\n" }) |framing| {
        const cut = try std.mem.concat(a, u8, &.{ before, framing, "The histor" });
        try expectCut(cut, 1, "{\"path\":\"notes.txt\",\"content\":\"The histor", "", "");
    }
    try expectCut(before ++ "\n\nThe histor", 1, "{\"path\":\"notes.txt\",\"content\":\"\\nThe histor", "", "");
}

test "tool cut: an XML cut inside a value's closer drops the closer's bytes, as a GLM or DSML cut does" {
    const args = "{\"path\":\"notes.txt\",\"content\":\"The history of";
    inline for (.{ "\n</", "\n</param", "\n</parameter", "</par", "\n<", "\n" }) |tail| try expectCut(opened ++ tail, 1, args, "", "");
    // bytes the value wrote stay, and a value its schema types is JSON up to the closer
    try expectCut(opened ++ " a<b", 1, args ++ " a<b", "", "");
    try expectCut(opened ++ "\n\n</", 1, args ++ "\\n", "", "");
    try expectCut("<tool_call>\n<function=write_file>\n<parameter=limit>\n12\n</", 1, "{\"limit\":12", "", "");
}

/// The receipt's answer tokens as written, one engine event per token, then ``after``: the bytes it never reached.
const PieceEngine = struct {
    pieces: []const []const u8,
    after: []const u8,

    fn engine(e: *@This()) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }

    fn info(_: *anyopaque) api.Info {
        return .{ .context_window = 4096 };
    }

    fn submit(ctx: *anyopaque, id: api.Id, _: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const e: *PieceEngine = @ptrCast(@alignCast(ctx));
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        var ids: [64]u32 = undefined;
        for (e.pieces) |piece| send(sink, id, piece, &ids);
        send(sink, id, e.after, &ids);
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .stop } });
    }

    fn send(sink: api.Sink, id: api.Id, piece: []const u8, ids: *[64]u32) void {
        for (piece, ids[0..piece.len]) |byte, *t| t.* = byte;
        sink.event(sink.ctx, id, &.{ .tokens = ids[0..piece.len] });
    }

    fn cancel(_: *anyopaque, _: api.Id) void {}

    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }

    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

// agentshape2 mixed12k-s0-t1's answer as 188d9c63c cut it at max_tokens 384 (done line sha=cf988c151b1a): its tokens
const receipt_ids = [_]u32{ 248058, 198, 27, 1628, 21402, 956, 29, 198, 27, 15704, 28, 5454, 29, 198, 24751, 976, 18997, 471, 4120, 198, 510, 15704, 29, 198, 510, 1628, 29, 198, 248059, 198, 248058, 198, 27, 1628, 21402, 956, 29, 198, 27, 15704, 28, 5454, 29, 198, 4577, 471, 4120, 593, 1715, 220, 17, 17566, 3464, 18710, 1321, 1627, 328, 2083, 593, 1715, 1, 198, 510 };
// ...as the Flash-Next tokenizer decodes each, the reply's tokens 321..383
const receipt = [_][]const u8{
    "<tool_call>", "\n", "<",  "function", "=b", "ash", ">", "\n", "<", "parameter", "=", "command", ">", "\n", "pwd", " &&", " ls", " -", "la",   "\n", "</", "parameter", ">",   "\n",    "</",  "function", ">",   "\n", "</tool_call>", "\n",
    "<tool_call>", "\n", "<",  "function", "=b", "ash", ">", "\n", "<", "parameter", "=", "command", ">", "\n", "ls",  " -",  "la",  " /", "work", " ",  "2",  ">/",        "dev", "/null", " ||", " echo",    " \"", "no", " /",           "work",
    "\"",          "\n", "</",
};
comptime {
    if (receipt.len != receipt_ids.len) @compileError("one piece per receipt token");
}
const bash = "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"bash\",\"parameters\":{\"type\":\"object\"," ++
    "\"properties\":{\"command\":{\"type\":\"string\"}}}}}]";

/// ``pieces`` then ``after``, max_tokens ``max``: whole and streamed replies give ``calls``, ``arguments``, ``finish``.
fn expectPieces(pieces: []const []const u8, after: []const u8, max: usize, calls: usize, arguments: []const u8, finish: []const u8) !void {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const body = try std.fmt.allocPrint(a, "{{\"model\":\"test-model\",\"messages\":[{{\"role\":\"user\",\"content\":\"x\"}}],{s},\"max_tokens\":{d}", .{ bash, max });
    var backend: PieceEngine = .{ .pieces = pieces, .after = after };
    const whole = try sendWith(backend.engine(), .chat, try std.mem.concat(a, u8, &.{ body, ",\"stream\":false}" }), .{});
    const streamed = try sendWith(backend.engine(), .chat, try std.mem.concat(a, u8, &.{ body, ",\"stream\":true}" }), .{});
    try std.testing.expectEqual(@as(?u16, 200), whole.status);
    for ([_]@TypeOf(whole){ whole, streamed }) |seen| {
        try std.testing.expectEqual(calls, seen.calls);
        try std.testing.expectEqualStrings(arguments, seen.arguments[0..seen.arguments_len]);
        try std.testing.expectEqualStrings(finish, @tagName(seen.finish.?));
    }
}

fn byteLen(pieces: []const []const u8) usize {
    var n: usize = 0;
    for (pieces) |p| n += p.len;
    return n;
}

const first_call = "{\"command\":\"pwd && ls -la\"}";

test "tool cut: the agentshape2 receipt's cut call ends at the bytes the value wrote, with finish length" {
    // 188d9c63c sent ``...echo \"no /work\"\n</``: the framing newline and the closer's first token
    try expectPieces(&receipt, "parameter>\n</function>\n</tool_call>", byteLen(&receipt), 2, first_call ++ "{\"command\":\"ls -la /work 2>/dev/null || echo \\\"no /work\\\"", "length");
}

test "tool cut: complete calls stay byte for byte: the receipt's calls closed, and values that hold closer-like bytes" {
    const closed = "parameter>\n</function>\n</tool_call>";
    try expectPieces(&receipt, closed, 1024, 2, first_call ++ "{\"command\":\"ls -la /work 2>/dev/null || echo \\\"no /work\\\"\"}", "tool_calls");
    // a value whose own bytes end like a closer's start, or in newlines before its framing one, keeps them
    const values = [_]struct { []const u8, []const u8 }{
        .{ "echo \"</\"\n", "echo \\\"</\\\"\\n" },
        .{ "a <", "a <" },
        .{ "x\n\n", "x\\n\\n" },
        .{ "  \n", "  \\n" },
    };
    for (values) |v| {
        const call = [_][]const u8{ "<tool_call>\n<function=bash>\n<parameter=command>\n", v[0], "\n</parameter>\n</function>\n</tool_call>" };
        var buf: [96]u8 = undefined;
        const want = try std.fmt.bufPrint(&buf, "{{\"command\":\"{s}\"}}", .{v[1]});
        try expectPieces(&call, "", 1024, 1, want, "tool_calls");
    }
}
