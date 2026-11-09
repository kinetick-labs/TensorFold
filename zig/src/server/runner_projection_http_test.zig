//! Abstain anchors through the OpenAI HTTP handler: raw runner bytes projected into the final fields.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const openai = @import("openai.zig");
const projection = @import("runner_projection.zig");
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
const model_text = @import("model_text.zig");
const Allocator = std.mem.Allocator;

const close_id: u32 = 300;
const eos_id: u32 = 301;

/// One byte a token below 256 (255 decodes lossily to U+FFFD, its token bytes stay FF); 300 is `</think>`, 301 EOS.
pub const Bytes = struct {
    pub fn text(t: *Bytes) model_text.Text {
        return .{ .ctx = t, .vtable = &.{ .encode = encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocab, .eos_ids = eos, .render = render, .template_source = template, .token_bytes = tokenBytes } };
    }
    fn encode(_: *anyopaque, a: Allocator, input: []const u8, _: bool) model_text.Error![]u32 {
        const ids = try a.alloc(u32, input.len);
        for (input, ids) |byte, *id| id.* = byte;
        return ids;
    }
    fn decode(_: *anyopaque, a: Allocator, ids: []const u32) Allocator.Error![]u8 {
        var out: std.ArrayList(u8) = .empty;
        for (ids) |id| if (id == close_id) try out.appendSlice(a, "</think>") else if (id == 255) try out.appendSlice(a, "\u{fffd}") else if (id < 256) try out.append(a, @intCast(id));
        return out.items;
    }
    fn tokenBytes(_: *anyopaque, a: Allocator, id: u32) Allocator.Error!?[]u8 {
        if (id == close_id) return try a.dupe(u8, "</think>");
        return if (id < 256) try a.dupe(u8, &.{@intCast(id)}) else null;
    }
    fn tokenId(_: *anyopaque, piece: []const u8) ?u32 {
        return if (std.mem.eql(u8, piece, "</think>")) close_id else null;
    }
    fn tokenString(_: *anyopaque, a: Allocator, id: u32) ![]u8 {
        return std.fmt.allocPrint(a, "{d}", .{id});
    }
    fn vocab(_: *anyopaque) u32 {
        return 302;
    }
    fn eos(_: *anyopaque) []const u32 {
        return &.{eos_id};
    }
    fn render(_: *anyopaque, a: Allocator, _: json.Value, _: model_text.RenderOptions, _: *[]const u8) model_text.Error!model_text.Rendered {
        return .{ .text = try a.dupe(u8, "prompt") };
    }
    fn template(_: *anyopaque) []const u8 {
        return "";
    }
};

/// Gives its tokens in one round, then ends (stop) with its runner receipt.
const Script = struct {
    tokens: []const u32,
    receipt: []const u8,
    fn engine(s: *Script) api.Engine {
        return .{ .ctx = s, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(_: *anyopaque) api.Info {
        return .{ .context_window = 8192 };
    }
    fn submit(ctx: *anyopaque, id: api.Id, _: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const s: *Script = @ptrCast(@alignCast(ctx));
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        sink.event(sink.ctx, id, &.{ .tokens = s.tokens });
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .stop, .runner = s.receipt } });
    }
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

const Reply = struct {
    a: Allocator,
    status: ?u16 = null,
    whole: ?json.Value = null,
    events: std.ArrayList(json.Value) = .empty,
    fn out(r: *Reply) openai.Out {
        return .{ .ctx = r, .vt = &.{ .open = open, .event = event, .reply = reply } };
    }
    fn open(_: *anyopaque, _: usize) error{Closed}!void {}
    fn event(ctx: *anyopaque, value: ?json.Value) error{Closed}!void {
        const r: *Reply = @ptrCast(@alignCast(ctx));
        if (value) |v| r.events.append(r.a, v) catch return error.Closed;
    }
    fn reply(ctx: *anyopaque, status_code: u16, value: json.Value) void {
        const r: *Reply = @ptrCast(@alignCast(ctx));
        r.status = status_code;
        r.whole = value;
    }
};

/// The client's fields (whole, or the stream's deltas added up), the receipt's runner block, and any tool calls.
const Got = struct { reasoning: []const u8, content: []const u8, runner: json.Value, calls: bool = false };

/// A thinking chat reply of `reasoning`, the close token and `content`, through openai.run.
fn post(a: Allocator, reasoning: []const u8, content: []const u8, receipt: []const u8, stream: bool) !Got {
    return postWith(a, reasoning, content, receipt, stream, "");
}

/// As `post`; `extra`: more request fields, each led by a comma (tools).
fn postWith(a: Allocator, reasoning: []const u8, content: []const u8, receipt: []const u8, stream: bool, extra: []const u8) !Got {
    var ids: std.ArrayList(u32) = .empty;
    for (reasoning) |b| try ids.append(a, b);
    try ids.append(a, close_id);
    for (content) |b| try ids.append(a, b);
    try ids.append(a, eos_id);
    var s: Script = .{ .tokens = ids.items, .receipt = receipt };
    var bytes: Bytes = .{};
    const srv = try Server.init(std.testing.allocator, std.testing.io, s.engine(), bytes.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .use_drafts = false }, null);
    defer srv.deinit();
    const text = try std.fmt.allocPrint(a, "{{\"model\":\"m\",\"messages\":[{{\"role\":\"user\",\"content\":\"hi\"}}],\"max_tokens\":500,\"stream\":{}{s}}}", .{ stream, extra });
    var conn = Conn{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a };
    var response = Reply{ .a = a };
    openai.run(srv, a, response.out(), .{ .conn = &conn }, true, (try json.parse(a, text)).ok);
    if (!stream) {
        const m = response.whole.?.get("choices").?.array[0].get("message").?;
        return .{ .reasoning = json.strOr(m.get("reasoning_content")), .content = json.strOr(m.get("content")), .runner = response.whole.?.get("tensorfold").?.get("runner").?, .calls = m.get("tool_calls") != null };
    }
    var r: std.ArrayList(u8) = .empty;
    var c: std.ArrayList(u8) = .empty;
    var runner: ?json.Value = null;
    var calls = false;
    for (response.events.items) |v| {
        if (v.get("tensorfold")) |t| if (t.get("runner")) |x| if (x.get("live") == null) {
            runner = x;
        };
        const cs = (v.get("choices") orelse continue).array;
        if (cs.len == 0) continue;
        const d = cs[0].get("delta") orelse continue;
        calls = calls or d.get("tool_calls") != null;
        try r.appendSlice(a, json.strOr(d.get("reasoning_content")));
        try c.appendSlice(a, json.strOr(d.get("content")));
    }
    return .{ .reasoning = r.items, .content = c.items, .runner = runner orelse return error.MissingReceipt, .calls = calls };
}

fn num(v: json.Value, key: []const u8) ?i64 {
    return (v.get(key) orelse return null).int64();
}

/// Each abstain's final [source_at, source_end) holds its source in the field the client has; raw_* keeps the engine's.
fn expectAbstain(got: Got, i: usize, source: []const u8, raw: [4]?i64, final: [4]?i64) !void {
    const x = got.runner.get("abstains").?.array[i];
    for ([_][]const u8{ "source_at", "source_end", "source_close_at", "resume_at" }, raw, final) |key, want_raw, want| {
        try std.testing.expectEqual(want, num(x, key));
        var name: [32]u8 = undefined;
        try std.testing.expectEqual(want_raw, num(x, try std.fmt.bufPrint(&name, "raw_{s}", .{key})));
    }
    if (source.len == 0) return;
    const field = if (std.mem.eql(u8, x.strField("source_field").?, "content")) got.content else got.reasoning;
    try std.testing.expectEqualStrings(source, field[@intCast(final[0].?)..@intCast(final[1].?)]);
}

fn abstain(a: Allocator, field: []const u8, source: [3]?usize, resume_field: []const u8, resume_at: usize) ![]const u8 {
    var o: std.ArrayList(u8) = .empty;
    try o.print(a, "{{\"cell\":0,\"form\":\"fence\",\"source_field\":\"{s}\"", .{field});
    for ([_][]const u8{ "source_at", "source_end", "source_close_at" }, source) |key, v|
        if (v) |n| try o.print(a, ",\"{s}\":{d}", .{ key, n }) else try o.print(a, ",\"{s}\":null", .{key});
    try o.print(a, ",\"resume_field\":\"{s}\",\"resume_at\":{d}}}", .{ resume_field, resume_at });
    return o.items;
}

// T5's counterexample: the runner finds `print(undefined_total)\n` at raw content byte 12, the client at byte 10.
const t5_content = "\n\n```python\nprint(undefined_total)\n```\nResult:\n";

test "runner projection: HTTP abstain anchors after the think close's newlines are final content bytes, raw kept" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const receipt = try std.fmt.allocPrint(a, "{{\"cells\":[{{\"cell\":1,\"field\":\"content\",\"at\":12}}],\"abstain_schema\":1,\"abstains\":[{s},{s}]}}", .{
        try abstain(a, "content", .{ 12, 35, 39 }, "content", 47),
        try abstain(a, "content", .{ null, null, null }, "content", 1), // resumed inside the dropped newlines
    });
    for ([_]bool{ false, true }) |stream| {
        const got = try post(a, "plan", t5_content, receipt, stream);
        try std.testing.expectEqualStrings("plan", got.reasoning);
        try std.testing.expectEqualStrings(t5_content[2..], got.content);
        try expectAbstain(got, 0, "print(undefined_total)\n", .{ 12, 35, 39, 47 }, .{ 10, 33, 37, 45 });
        try expectAbstain(got, 1, "", .{ null, null, null, 1 }, .{ null, null, null, null });
        try std.testing.expectEqual(10, num(got.runner.get("cells").?.array[0], "final_at"));
        const content = got.runner.get("field_projection").?.get("content").?;
        try std.testing.expectEqual(47, num(content, "raw_bytes"));
        try std.testing.expectEqual(2, num(content, "dropped_lead"));
        try std.testing.expectEqual(45, num(content, "final_bytes"));
    }
}

test "runner projection: HTTP reasoning anchors follow the whole reply's strip, and the stream's deltas alike" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const reasoning = " \n x\n```python\nprint(4)\n```\n";
    const receipt = try std.fmt.allocPrint(a, "{{\"cells\":[],\"abstains\":[{s}]}}", .{try abstain(a, "reasoning", .{ 15, 24, 28 }, "reasoning", 28)});
    const whole = try post(a, reasoning, "ok", receipt, false);
    try std.testing.expectEqualStrings(reasoning[3 .. reasoning.len - 1], whole.reasoning);
    // the close and resume bytes sit in the stripped trailing newline: no final byte, only the raw one
    try expectAbstain(whole, 0, "print(4)\n", .{ 15, 24, 28, 28 }, .{ 12, 21, null, null });
    // the stream sends the reasoning stripped as the whole reply does: the map follows the bytes it sent
    const streamed = try post(a, reasoning, "ok", receipt, true);
    try std.testing.expectEqualStrings(reasoning[3 .. reasoning.len - 1], streamed.reasoning);
    try expectAbstain(streamed, 0, "print(4)\n", .{ 15, 24, 28, 28 }, .{ 12, 21, null, null });
}

test "runner projection: HTTP a receipt without abstain records keeps its bytes" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const receipt = "{\"cells\":[{\"cell\":0,\"field\":\"content\",\"at\":12}]}";
    for ([_]bool{ false, true }) |stream| {
        const got = try post(a, "plan", t5_content, receipt, stream);
        try std.testing.expectEqualStrings(try json.stringify(a, (try json.parse(a, receipt)).ok, .{}), try json.stringify(a, got.runner, .{}));
    }
}

// T5's r10 fixtures (T5-ORACLE-R10-REVIEW.md): the map is taken after tool parsing, on the fields the client gets
const tool_text = "<tool_call>{\"name\":\"f\",\"arguments\":{}}</tool_call>\n```python\nprint(undefined_total)\n```\n";
const tools = ",\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"f\",\"parameters\":{\"type\":\"object\",\"properties\":{}}}}]";

test "runner projection: HTTP a tool envelope cut from the content moves the anchors to the content sent, whole and streamed" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const source = "print(undefined_total)\n";
    const at = std.mem.indexOf(u8, tool_text, source).?;
    const receipt = try std.fmt.allocPrint(a, "{{\"cells\":[],\"abstains\":[{s}]}}", .{try abstain(a, "content", .{ at, at + source.len, tool_text.len }, "content", tool_text.len)});
    for ([_]bool{ false, true }) |stream| {
        const got = try postWith(a, "plan", tool_text, receipt, stream, tools);
        try std.testing.expect(got.calls); // the tool path ran: the envelope became a call
        const actual = std.mem.indexOf(u8, got.content, source).?;
        const x = got.runner.get("abstains").?.array[0];
        try std.testing.expectEqual(@as(?i64, @as(i64, @intCast(actual))), num(x, "source_at"));
        try std.testing.expectEqual(@as(?i64, @as(i64, @intCast(at))), num(x, "raw_source_at"));
    }
}

test "runner projection: HTTP a lossy decode (FF as U+FFFD) publishes no map for that field, its raw anchors kept" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const text = "\xff\n```python\nprint(undefined_total)\n```\n";
    const at = std.mem.indexOf(u8, text, "print").?;
    const receipt = try std.fmt.allocPrint(a, "{{\"cells\":[],\"abstains\":[{s}]}}", .{try abstain(a, "content", .{ at, at + 23, text.len }, "content", text.len)});
    for ([_]bool{ false, true }) |stream| {
        const got = try post(a, "plan", text, receipt, stream);
        try std.testing.expect(std.mem.startsWith(u8, got.content, "\u{fffd}"));
        try std.testing.expect(got.runner.get("field_projection").?.get("content").? == .null);
        try std.testing.expect(got.runner.get("field_projection").?.get("reasoning").? != .null);
        try expectAbstain(got, 0, "", .{ @as(i64, @intCast(at)), @as(i64, @intCast(at + 23)), @as(i64, @intCast(text.len)), @as(i64, @intCast(text.len)) }, .{ null, null, null, null });
    }
}

test "runner projection: HTTP a raw anchor past the rebuilt raw bytes leaves its field unmapped" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const receipt = try std.fmt.allocPrint(a, "{{\"cells\":[],\"abstains\":[{s}]}}", .{try abstain(a, "content", .{ 12, 35, 39 }, "content", 1000)});
    const got = try post(a, "plan", t5_content, receipt, false);
    try std.testing.expect(got.runner.get("field_projection").?.get("content").? == .null);
    try expectAbstain(got, 0, "", .{ 12, 35, 39, 1000 }, .{ null, null, null, null });
}

test "runner projection: HTTP a run found only inside a deleted tool argument is never mapped, whole and streamed" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // T5: the content is the FF's U+FFFD (raw byte 0); a literal U+FFFD in the cut argument (raw byte 42) is no origin
    const text = "\xff<tool_call>{\"name\":\"f\",\"arguments\":{\"x\":\"\u{fffd}\"}}</tool_call>";
    const receipt = try std.fmt.allocPrint(a, "{{\"cells\":[],\"abstains\":[{s}]}}", .{try abstain(a, "content", .{ 0, 1, text.len }, "content", text.len)});
    for ([_]bool{ false, true }) |stream| {
        const got = try postWith(a, "plan", text, receipt, stream, tools);
        try std.testing.expect(got.calls);
        try std.testing.expectEqualStrings("\u{fffd}", got.content);
        try std.testing.expect(got.runner.get("field_projection").?.get("content").? == .null);
    }
}

test "runner projection: HTTP content the tool parse joined across an envelope is never mapped onto the argument" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const text = "a<tool_call>{\"name\":\"f\",\"arguments\":{\"x\":\"ab\"}}</tool_call>b"; // "ab" only in the cut
    const receipt = try std.fmt.allocPrint(a, "{{\"cells\":[],\"abstains\":[{s}]}}", .{try abstain(a, "content", .{ 0, 1, text.len }, "content", text.len)});
    for ([_]bool{ false, true }) |stream| {
        const got = try postWith(a, "plan", text, receipt, stream, tools);
        try std.testing.expect(got.calls);
        if (!stream) try std.testing.expectEqualStrings("ab", got.content);
        try std.testing.expect(got.runner.get("field_projection").?.get("content").? == .null);
    }
}

test "runner projection: HTTP raw invalid UTF-8 publishes no map even inside a deleted envelope, whole and streamed" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const text = "<tool_call>{\"name\":\"f\",\"arguments\":{\"x\":\"\xff\"}}</tool_call>\nok";
    const at = std.mem.indexOf(u8, text, "ok").?;
    const receipt = try std.fmt.allocPrint(a, "{{\"cells\":[],\"abstains\":[{s}]}}", .{try abstain(a, "content", .{ at, at + 2, text.len }, "content", text.len)});
    for ([_]bool{ false, true }) |stream| {
        const got = try postWith(a, "plan", text, receipt, stream, tools);
        try std.testing.expect(got.calls);
        try std.testing.expectEqualStrings("ok", got.content);
        try std.testing.expect(got.runner.get("field_projection").?.get("content").? == .null);
        try std.testing.expect(got.runner.get("field_projection").?.get("reasoning").? != .null);
    }
}

test "runner projection: HTTP a non-null raw anchor past i64 (2^63) leaves its field unmapped; a null one does not" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const big = try std.fmt.allocPrint(a, "{{\"cells\":[],\"abstains\":[{s}]}}", .{try abstain(a, "content", .{ 12, 35, @as(usize, 1) << 63 }, "content", 47)});
    for ([_]bool{ false, true }) |stream| {
        const got = try post(a, "plan", t5_content, big, stream);
        try std.testing.expect(got.runner.get("field_projection").?.get("content").? == .null);
    }
    const none = try std.fmt.allocPrint(a, "{{\"cells\":[],\"abstains\":[{s}]}}", .{try abstain(a, "content", .{ null, null, null }, "content", 39)});
    const got = try post(a, "plan", t5_content, none, false);
    try std.testing.expect(got.runner.get("field_projection").?.get("content").? != .null);
}

test "runner projection: a run is mapped only with blanks and whole envelopes outside it, and only when one run fits" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try std.testing.expectEqual(@as(?projection.Map, null), try projection.mapOf(a, "xab", "ab", "")); // kept text cut
    try std.testing.expectEqual(@as(?projection.Map, null), try projection.mapOf(a, "\n\n", "\n", "")); // two runs fit
    const env = "<tool_call>{}</tool_call>";
    const kept = "x " ++ env ++ " y"; // an envelope the parse kept sits inside the run
    try std.testing.expectEqual(@as(?projection.Map, .{ .raw = kept.len + 1, .lead = 1, .kept = kept.len }), try projection.mapOf(a, "\n" ++ kept, kept, ""));
    try std.testing.expectEqual(@as(?projection.Map, .{ .raw = env.len + 3, .lead = env.len + 1, .kept = 2 }), try projection.mapOf(a, env ++ "\nab", "ab", ""));
    try std.testing.expectEqual(@as(?projection.Map, .{ .raw = 11, .lead = 9, .kept = 2 }), try projection.mapOf(a, "<think>\n\nab", "ab", "<think>"));
    const arg = "<tool_call>{\"x\":\"ab\"}</tool_call>"; // the run again inside a cut argument is no second origin
    try std.testing.expectEqual(@as(?projection.Map, .{ .raw = arg.len + 2, .lead = 0, .kept = 2 }), try projection.mapOf(a, "ab" ++ arg, "ab", ""));
}

test "runner projection: an unfinished envelope (to EOF) proves no deletion, though kept whole in the run it maps" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const open = "<\u{ff5c}DSML\u{ff5c}tool_calls><\u{ff5c}DSML\u{ff5c}invoke name=\"f\"><\u{ff5c}DSML\u{ff5c}parameter name=\"x\" string=\"true\">KEEP";
    try std.testing.expectEqual(@as(?projection.Map, null), try projection.mapOf(a, open, "", "")); // T5's direct case
    try std.testing.expectEqual(@as(?projection.Map, null), try projection.mapOf(a, "ab\n" ++ open, "ab", ""));
    try std.testing.expectEqual(@as(?projection.Map, .{ .raw = open.len + 1, .lead = 1, .kept = open.len }), try projection.mapOf(a, "\n" ++ open, open, ""));
}

/// Replies of `n` calls then `f`: every call holds an `f` the final `f` must not map into.
const Many = struct {
    raws: [2][]const u8,
    fn run(m: *const Many, i: usize) anyerror!void {
        var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
        defer arena.deinit();
        const got = (try projection.mapOf(arena.allocator(), m.raws[i], "f", "")).?;
        try std.testing.expectEqual(m.raws[i].len - 1, got.lead);
    }
};

test "runner projection: a final run among many envelopes is placed in one pass, not again per match" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const env = "<tool_call>{\"name\":\"f\",\"arguments\":{}}</tool_call>";
    var m: Many = .{ .raws = undefined };
    for (&m.raws, [_]usize{ 625, 5000 }) |*raw, n| {
        const r = try a.alloc(u8, n * env.len + 1);
        for (0..n) |i| @memcpy(r[i * env.len ..][0..env.len], env);
        r[r.len - 1] = 'f';
        raw.* = r;
    }
    const f = try @import("linear_time.zig").fastest(5, @as(*const Many, &m), Many.run);
    // 8x the envelopes: within 24x the work (or 20 ms), not 64x
    try std.testing.expect(f.ns[1] <= 20 * std.time.ns_per_ms or f.work[1] <= 24 * f.work[0]);
}

test "runner projection: a final cut inside a multibyte blank or code point is never mapped, on either side" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cases = [_][2][]const u8{ // raw, final; no blank takes four bytes: U+1F600 stands in at that width
        .{ "\xc2\xa0", "\xa0" },               .{ "\xc2\xa0", "\xc2" }, // T5's NBSP case, and its other side
        .{ "\u{3000}", "\x80\x80" },           .{ "\u{3000}", "\xe3\x80" },
        .{ "\u{1f600}", "\x98\x80" },          .{ "\u{1f600}", "\xf0\x9f" },
        .{ "\u{3000}x\u{3000}", "\x80x\xe3" }, .{ "x\u{a0}", "x\xc2" },
        .{ "\u{a0}x", "\xa0x" },
    };
    for (cases) |c| try std.testing.expectEqual(@as(?projection.Map, null), try projection.mapOf(a, c[0], c[1], ""));
    try std.testing.expectEqual(@as(?projection.Map, .{ .raw = 7, .lead = 3, .kept = 1 }), try projection.mapOf(a, "\u{3000}x\u{3000}", "x", ""));
}

/// T5's slice-based reference (T5-PROJECTION-R4-REVIEW.md): each candidate's prefix and suffix checked afresh.
fn refDeleted(raw: []const u8, envs: []const @import("tool_parse.zig").Envelope, from: usize, to: usize) bool {
    var p = from;
    for (envs) |e| {
        if (e.end <= p or e.start >= to) continue;
        if (e.start < p or e.end > to or !e.complete) return false;
        if (@import("reply_text.zig").pyStrip(raw[p..e.start]).len > 0) return false;
        p = e.end;
    }
    return @import("reply_text.zig").pyStrip(raw[p..to]).len == 0;
}

fn refMap(a: Allocator, raw: []const u8, final: []const u8, open: []const u8) !?projection.Map {
    if (!std.unicode.utf8ValidateSlice(raw)) return null;
    const body = if (open.len > 0 and std.mem.startsWith(u8, raw, open)) open.len else 0;
    const envs = try @import("tool_parse.zig").envelopes(a, raw);
    if (final.len == 0) return if (refDeleted(raw, envs, body, raw.len)) .{ .raw = raw.len, .lead = raw.len, .kept = 0 } else null;
    var found: ?projection.Map = null;
    var from = body;
    while (std.mem.indexOfPos(u8, raw, from, final)) |lead| : (from = lead + 1) {
        if (!refDeleted(raw, envs, body, lead) or !refDeleted(raw, envs, lead + final.len, raw.len)) continue;
        if (found != null) return null;
        found = .{ .raw = raw.len, .lead = lead, .kept = final.len };
    }
    return found;
}

test "runner projection: the one-pass map equals the slice reference over blanks, envelopes and split code points" {
    const pieces = [_][]const u8{ " ", "x", "y", "<think>", "<tool_call>f</tool_call>", "<\u{ff5c}DSML\u{ff5c}tool_calls>", "\u{2003}", "\u{a0}", "\u{3000}", "\u{1f600}" };
    var checked: usize = 0;
    for (pieces) |x| for (pieces) |y| for (pieces) |z| {
        var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        const raw = try std.mem.concat(a, u8, &.{ x, y, z });
        const finals = [_][]const u8{ "", raw, @import("reply_text.zig").pyStrip(raw), "x", "y", "f", x, y, z, raw[x.len..], raw[1..], raw[0 .. raw.len - 1] };
        for (finals) |final| for ([_][]const u8{ "", "<think>" }) |open| {
            try std.testing.expectEqualDeep(try refMap(a, raw, final, open), try projection.mapOf(a, raw, final, open));
            checked += 1;
        };
    };
    try std.testing.expectEqual(10 * 10 * 10 * 12 * 2, checked);
}
