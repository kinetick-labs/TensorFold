//! The compute default where a reply cannot run a call: no built-in tool, and never an unrun call (SparkDash QA).
const std = @import("std");
const json = @import("json");
const api = @import("engine_api");
const HfText = @import("hf_text.zig").HfText;
const tf = @import("control_tokens_fixture.zig");
const h = @import("compute_lane_http_test.zig");
const turns = @import("compute_turns.zig");
const lane = api.hold.compute_lane;
const t = std.testing;
const Allocator = std.mem.Allocator;

const python = "/usr/bin/python3";

/// An engine that keeps the reply's budget: at most `max_tokens` tokens (forced ones count), then `length`.
const Script = struct {
    first: []const u32,
    answer: []const u32,
    sink: ?api.Sink = null,
    max: u32 = 0,
    sent: u32 = 0,
    hold: bool = false,
    prompt: []const u32 = &.{},
    a: Allocator,

    fn engine(s: *@This()) api.Engine {
        return .{ .ctx = s, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory, .resume_held = resumeHeld } };
    }
    fn info(_: *anyopaque) api.Info {
        return .{ .context_window = 32768 };
    }
    fn send(s: *@This(), id: api.Id, tokens: []const u32) bool {
        const n: u32 = @intCast(@min(tokens.len, s.max - s.sent));
        if (n > 0) s.sink.?.event(s.sink.?.ctx, id, &.{ .tokens = tokens[0..n] });
        s.sent += n;
        if (n == tokens.len) return true;
        s.sink.?.event(s.sink.?.ctx, id, &.{ .finished = .{ .reason = .length } });
        return false;
    }
    fn submit(ctx: *anyopaque, id: api.Id, request: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const s: *@This() = @ptrCast(@alignCast(ctx));
        s.* = .{ .a = s.a, .first = s.first, .answer = s.answer, .sink = sink, .max = request.max_tokens, .hold = request.hold_eos, .prompt = s.a.dupe(u32, request.prompt) catch unreachable };
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        if (s.send(id, s.first) and !request.hold_eos) sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .stop } });
    }
    fn resumeHeld(ctx: *anyopaque, id: api.Id, r: api.hold.Resume) void {
        const s: *@This() = @ptrCast(@alignCast(ctx));
        if (r.tokens.len == 0) return s.sink.?.event(s.sink.?.ctx, id, &.{ .finished = .{ .reason = .stop } });
        if (s.send(id, r.tokens)) _ = s.send(id, s.answer);
    }
    fn cancel(ctx: *anyopaque, id: api.Id) void {
        const s: *@This() = @ptrCast(@alignCast(ctx));
        s.sink.?.event(s.sink.?.ctx, id, &.{ .finished = .{ .reason = .cancelled } });
    }
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

/// What the client read, streamed or whole: content, tool call names and the finish reason.
const Seen = struct {
    a: Allocator,
    code: u16 = 0,
    content: std.ArrayList(u8) = .empty,
    calls: usize = 0,
    finish: []const u8 = "",
    fn out(r: *@This()) @import("openai.zig").Out {
        return .{ .ctx = r, .vt = &.{ .open = open, .event = event, .reply = reply } };
    }
    fn open(ctx: *anyopaque, _: usize) error{Closed}!void {
        const r: *@This() = @ptrCast(@alignCast(ctx));
        r.code = 200;
    }
    fn choice(r: *@This(), c: json.Value, key: []const u8) error{Closed}!void {
        if (c.strField("finish_reason")) |f| r.finish = f;
        const m = c.get(key) orelse return;
        if (m.strField("content")) |s| r.content.appendSlice(r.a, s) catch return error.Closed;
        if (m.get("tool_calls")) |list| if (list == .array) {
            r.calls += list.array.len;
        };
    }
    fn event(ctx: *anyopaque, payload: ?json.Value) error{Closed}!void {
        const r: *@This() = @ptrCast(@alignCast(ctx));
        const ch = (payload orelse return).get("choices") orelse return;
        if (ch == .array and ch.array.len > 0) try r.choice(ch.array[0], "delta");
    }
    fn reply(ctx: *anyopaque, code: u16, v: json.Value) void {
        const r: *@This() = @ptrCast(@alignCast(ctx));
        r.code = code;
        const ch = v.get("choices") orelse return;
        if (ch == .array and ch.array.len > 0) r.choice(ch.array[0], "message") catch {};
    }
};

const Ask = struct { default: bool = true, call: bool = true, max_tokens: i64, stream: bool, fields: []const u8 = "{}", think: bool = false };
const Run = struct { seen: Seen, prompt: []const u32, held: bool, call_len: usize };

/// One chat whose model calls python with `pad` bytes of comment (or answers), against the budget-keeping engine.
fn chat(a: Allocator, pad: usize, ask: Ask) !Run {
    if (@import("builtin").os.tag != .linux) return error.SkipZigTest; // the lane's executor runs on Linux only
    std.Io.Dir.accessAbsolute(t.io, python, .{}) catch return error.SkipZigTest;
    var tmp = try h.fixture(a);
    defer tmp.deinit();
    var problem: []const u8 = "";
    const hf = try HfText.load(a, t.io, tmp.path, null, a, &problem);
    defer hf.deinit();
    const text = hf.text();
    const history = try tf.messages(a, "What is 6*7?");
    const asked = try text.renderIds(a, history, .{ .add_generation_prompt = true, .enable_thinking = ask.think }, &problem);
    const comment = try a.alloc(u8, pad);
    @memset(comment, 'x');
    const call_turn = try h.obj(a, try std.fmt.allocPrint(a,
        \\{{"role":"assistant","content":"","tool_calls":[{{"id":"c1","type":"function","function":{{"name":"python","arguments":"{{\"code\":\"print(6*7) # {s}\"}}"}}}}]}}
    , .{comment}));
    const called = try text.renderIds(a, try h.appendTo(a, history, &.{call_turn}), .{ .add_generation_prompt = false, .enable_thinking = ask.think }, &problem);
    // thinking, the answer closes <think> (the tool-result turn reopens it)
    const close_id = [_]u32{text.tokenId("</think>").?};
    const close: []const u32 = if (ask.think) &close_id else &.{};
    const answer = try std.mem.concat(a, u32, &.{ close, try text.encodePlain(a, "It is 42."), &.{called[called.len - 1]} });
    const script = try a.create(Script);
    script.* = .{ .a = a, .first = if (ask.call) called[asked.len..] else answer, .answer = answer };
    const srv = try @import("server.zig").Server.init(a, t.io, script.engine(), text, .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false }, null);
    defer srv.deinit();
    const cfg: lane.Config = .{ .python = python, .default = ask.default };
    srv.info.compute = cfg;
    srv.compute = lane.Pool.init(cfg, &.{ "/bin/sh", "-c", "exec \"$0\" -I -B -u -c \"$1\"", python, lane.runner.runner });
    var seen: Seen = .{ .a = a };
    var conn = try @import("http_conn.zig").Conn.init(a, -1, "fixture");
    defer conn.deinit();
    const body = (try h.obj(a, ask.fields)).object;
    try body.put(a, "model", .{ .string = "m" });
    try body.put(a, "max_tokens", try json.intValue(a, ask.max_tokens));
    try body.put(a, "messages", history);
    try body.put(a, "stream", .{ .bool = ask.stream });
    if (ask.think) try body.put(a, "chat_template_kwargs", try h.obj(a, "{\"enable_thinking\":true}"));
    @import("openai.zig").run(srv, a, seen.out(), .{ .conn = &conn }, true, .{ .object = body });
    return .{ .seen = seen, .prompt = script.prompt, .held = script.hold, .call_len = called.len - asked.len };
}

test "compute lane: the default offers no built-in tool unthinking, to a 64-token reply or with stop strings" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]bool{ false, true }) |stream| {
        const off = try chat(a, 0, .{ .default = false, .call = false, .max_tokens = 64, .stream = stream });
        const qa = try chat(a, 0, .{ .call = false, .max_tokens = 64, .stream = stream });
        try t.expectEqualSlices(u32, off.prompt, qa.prompt); // the default off's prompt: no tool
        try t.expect(!qa.held);
        try t.expectEqualStrings("It is 42.", qa.seen.content.items);
        // thinking off, room or not: the prose before a call would be the visible answer
        const plain = try chat(a, 0, .{ .call = false, .max_tokens = 4096, .stream = stream });
        try t.expectEqualSlices(u32, off.prompt, plain.prompt);
        try t.expect(!plain.held);
        const t_off = try chat(a, 0, .{ .default = false, .call = false, .max_tokens = 64, .stream = stream, .think = true });
        const stop = try chat(a, 0, .{ .call = false, .max_tokens = 2048, .stream = stream, .think = true, .fields = "{\"stop\":[\"###\"]}" });
        try t.expect(!stop.held);
        // thinking under the floor or with stop strings: the default off's prompt too
        const t_on = try chat(a, 0, .{ .call = false, .max_tokens = 64, .stream = stream, .think = true });
        try t.expect(!t_on.held);
        try t.expectEqualSlices(u32, t_off.prompt, t_on.prompt);
        try t.expectEqualSlices(u32, t_off.prompt, stop.prompt);
        // at the floor, thinking, the built-in is offered and its call runs
        const ran = try chat(a, 0, .{ .max_tokens = turns.default_min_tokens, .stream = stream, .think = true });
        try t.expect(ran.held);
        try t.expectEqualStrings("It is 42.", ran.seen.content.items);
    }
}

test "compute lane: a built-in call the default could not run never reaches the client; the reply ends at length" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const pad = 1100; // the call alone nears the floor, so a budget past the floor still cannot hold its answer
    const len: i64 = @intCast((try chat(a, pad, .{ .max_tokens = 4096, .stream = false, .think = true })).call_len);
    // no room for an answer, no room for the result, and max_tokens inside the call itself
    for ([_]i64{ len + 8, len + 20, len - 20 }) |max| for ([_]bool{ false, true }) |stream| {
        try t.expect(max >= turns.default_min_tokens);
        const r = try chat(a, pad, .{ .max_tokens = max, .stream = stream, .think = true });
        errdefer std.debug.print("max_tokens {d} stream {}: code {d} finish {s} calls {d} content [{s}]\n", .{ max, stream, r.seen.code, r.seen.finish, r.seen.calls, r.seen.content.items });
        try t.expectEqual(@as(u16, 200), r.seen.code);
        try t.expect(r.held);
        try t.expectEqual(@as(usize, 0), r.seen.calls);
        try t.expectEqualStrings("length", r.seen.finish);
        try t.expect(std.mem.indexOf(u8, r.seen.content.items, "DSML") == null);
    };
}
