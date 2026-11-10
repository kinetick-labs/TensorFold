//! The compute lane on a template closing turns with `<|im_end|>\n` (Qwen3.x, Qwen3.8-Flash-Next): the reply goes on.
const std = @import("std");
const json = @import("json");
const HfText = @import("hf_text.zig").HfText;
const tf = @import("control_tokens_fixture.zig");
const h = @import("compute_lane_http_test.zig");
const lane = @import("engine_api").hold.compute_lane;
const t = std.testing;

const template = @embedFile("fixtures/dory/chat_template.jinja");
const python = "/usr/bin/python3";
const call_text = "<tool_call>\n<function=python>\n<parameter=code>\nprint(6*7)\n</parameter>\n</function>\n</tool_call>\n";

/// A streamed reply's deltas, joined: what a streaming client shows as content and as reasoning.
const Streamed = struct {
    a: std.mem.Allocator,
    content: std.ArrayList(u8) = .empty,
    reasoning: std.ArrayList(u8) = .empty,
    code: u16 = 0,
    fn out(r: *@This()) @import("openai.zig").Out {
        return .{ .ctx = r, .vt = &.{ .open = open, .event = event, .reply = reply } };
    }
    fn open(_: *anyopaque, _: usize) error{Closed}!void {}
    fn event(ctx: *anyopaque, v: ?json.Value) error{Closed}!void {
        const r: *@This() = @ptrCast(@alignCast(ctx));
        const choices = (v orelse return).get("choices") orelse return;
        for (choices.array) |c| if (c.get("delta")) |d| {
            if (d.strField("content")) |x| r.content.appendSlice(r.a, x) catch return error.Closed;
            if (d.strField("reasoning_content")) |x| r.reasoning.appendSlice(r.a, x) catch return error.Closed;
        };
    }
    fn reply(ctx: *anyopaque, code: u16, _: json.Value) void {
        const r: *@This() = @ptrCast(@alignCast(ctx));
        r.code = code;
    }
};

const Run = struct {
    reply: h.Reply,
    streamed: *Streamed,
    script: *h.Script,
    text: @import("model_text.zig").Text,
    hf: *HfText,
    srv: *@import("server.zig").Server,
    conn: *@import("http_conn.zig").Conn,

    fn deinit(r: *Run) void {
        r.conn.deinit();
        r.srv.deinit();
        r.hf.deinit();
    }
};

const Opt = struct { thinking: bool, stream: bool = false, answer: []const u8, then: []const u8 = "", max_tokens: i64 = 400 };

/// One lane chat on the dory template (markers are text, as in Qwen): a held call segment, `answer`, then `then`.
fn run(a: std.mem.Allocator, dir_out: *std.testing.TmpDir, o: Opt) !Run {
    const thinking, const stream = .{ o.thinking, o.stream };
    const tok = try h.obj(a, @embedFile("fixtures/control_tokens/mini_tokenizer.json"));
    for (tok.get("added_tokens").?.array) |added| {
        const name = added.strField("content").?;
        if (!std.mem.startsWith(u8, name, "<|")) try added.object.put(a, "special", .{ .bool = false });
    }
    try tok.object.put(a, "pre_tokenizer", try h.obj(a, "{\"type\":\"ByteLevel\",\"add_prefix_space\":false,\"trim_offsets\":false,\"use_regex\":false}"));
    try dir_out.dir.writeFile(t.io, .{ .sub_path = "tokenizer.json", .data = try json.stringify(a, tok, .{}) });
    try dir_out.dir.writeFile(t.io, .{ .sub_path = "config.json", .data = "{\"model_type\":\"dory\",\"vocab_size\":2048}" });
    try dir_out.dir.writeFile(t.io, .{ .sub_path = "tokenizer_config.json", .data = "{\"eos_token\":\"<|im_end|>\"}" });
    try dir_out.dir.writeFile(t.io, .{ .sub_path = "chat_template.jinja", .data = template });
    const dir = try dir_out.dir.realPathFileAlloc(t.io, ".", a);
    var problem: []const u8 = "";
    const hf = try HfText.load(a, t.io, dir, null, a, &problem);
    const text = hf.text();
    const eos = text.tokenId("<|im_end|>").?;
    const call = try std.mem.concat(a, u32, &.{ try text.encodePlain(a, call_text), &.{eos} });
    const answer = try std.mem.concat(a, u32, &.{ try text.encodePlain(a, o.answer), &.{eos} });
    const then = try std.mem.concat(a, u32, &.{ try text.encodePlain(a, o.then), &.{eos} });
    const script = try a.create(h.Script);
    script.* = .{ .a = a, .call = call, .answer = answer, .answers = if (o.then.len > 0) try a.dupe([]const u32, &.{ answer, then }) else &.{} };
    const srv = try @import("server.zig").Server.init(a, t.io, script.engine(), text, .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = thinking, .use_drafts = false }, null);
    const cfg: lane.Config = .{ .python = python };
    srv.info.compute = cfg;
    srv.compute = lane.Pool.init(cfg, &.{ "/bin/sh", "-c", "exec \"$0\" -I -B -u -c \"$1\"", python, lane.runner.runner });
    var reply: h.Reply = .{ .a = a };
    const conn = try a.create(@import("http_conn.zig").Conn);
    conn.* = try @import("http_conn.zig").Conn.init(a, -1, "fixture");
    const body = try json.newObject(a);
    try body.put(a, "model", .{ .string = "m" });
    try body.put(a, "max_tokens", try json.intValue(a, o.max_tokens));
    try body.put(a, "messages", try tf.messages(a, "What is 6*7?"));
    try body.put(a, "tools", try h.obj(a, "[{\"type\":\"code_interpreter\"}]"));
    if (thinking) try body.put(a, "chat_template_kwargs", try h.obj(a, "{\"enable_thinking\":true}"));
    const streamed = try a.create(Streamed);
    streamed.* = .{ .a = a };
    if (stream) try body.put(a, "stream", .{ .bool = true });
    @import("openai.zig").run(srv, a, if (stream) streamed.out() else reply.out(), .{ .conn = conn }, true, .{ .object = body });
    return .{ .reply = reply, .streamed = streamed, .script = script, .text = text, .hf = hf, .srv = srv, .conn = conn };
}

fn skipUnlessLinux() !void {
    if (@import("builtin").os.tag != .linux) return error.SkipZigTest; // the lane's executor runs on Linux only
    std.Io.Dir.accessAbsolute(t.io, python, .{}) catch return error.SkipZigTest;
}

test "compute lane: a ChatML turn close after the held EOS goes in with the result turn, not instead of it" {
    try skipUnlessLinux();
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var r = try run(a, &tmp, .{ .thinking = false, .answer = "It is 42." });
    defer r.deinit();
    const said = try json.stringify(a, r.reply.body, .{});
    errdefer std.debug.print("COMPUTE_LANE_CHATML_REPLY {s}\n", .{said});
    try t.expectEqual(@as(u16, 200), r.reply.code);
    // the result went in (not an empty resume ending the reply at the call), then the answer's EOS ended it
    try t.expectEqual(@as(usize, 2), r.script.resumes.items.len);
    try t.expectEqual(@as(usize, 0), r.script.resumes.items[1].len);
    // the forced rows: the template's own close after the EOS, the result turn, the next generation prompt
    try t.expectEqualStrings("\n<|im_start|>user\n<tool_response>\n42\n</tool_response>\n<|im_end|>\n<|im_start|>assistant\n<think></think>", try r.text.decode(a, r.script.resumes.items[0]));
    const choice = r.reply.body.get("choices").?.array[0];
    try t.expectEqualStrings("stop", choice.get("finish_reason").?.string);
    try t.expect(std.mem.indexOf(u8, choice.get("message").?.strField("content").?, "It is 42.") != null);
    const report = r.reply.body.get("tensorfold").?.get("compute").?.get("calls").?.array[0];
    try t.expectEqualStrings("42", report.strField("output").?);
    try t.expect(report.get("inserted").?.bool);
}

/// The reply's content and reasoning, non-streamed.
fn message(r: *const Run) struct { content: []const u8, reasoning: []const u8 } {
    const m = r.reply.body.get("choices").?.array[0].get("message").?;
    return .{ .content = m.strField("content") orelse "", .reasoning = m.strField("reasoning_content") orelse "" };
}

test "compute lane: an answer left inside the think block the result turn reopened gets the next turn, thinking off" {
    try skipUnlessLinux();
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    // thinking on: the model stops inside the reopened think block, then answers in the turn the template opens next
    var r = try run(a, &tmp, .{ .thinking = true, .answer = "Bo has 42.", .then = "Answer: 42" });
    defer r.deinit();
    errdefer std.debug.print("COMPUTE_LANE_CLOSE_TURN {s}\n", .{json.stringify(a, r.reply.body, .{}) catch ""});
    try t.expectEqual(@as(u16, 200), r.reply.code);
    try t.expectEqual(@as(usize, 3), r.script.resumes.items.len);
    try t.expect(std.mem.endsWith(u8, try r.text.decode(a, r.script.resumes.items[0]), "<|im_start|>assistant\n<think>\n"));
    // the template's close after the EOS, then its generation prompt with thinking off: the think block closed
    try t.expectEqualStrings("\n<|im_start|>assistant\n<think></think>", try r.text.decode(a, r.script.resumes.items[1]));
    try t.expectEqual(@as(usize, 0), r.script.resumes.items[2].len);
    const choice = r.reply.body.get("choices").?.array[0];
    try t.expectEqualStrings("stop", choice.get("finish_reason").?.string);
    try t.expectEqualStrings("Answer: 42", std.mem.trim(u8, message(&r).content, " \n"));
    try t.expect(std.mem.indexOf(u8, message(&r).reasoning, "Bo has 42.") != null);
}

test "compute lane: streamed, the next turn's answer is the content and the unclosed text stays reasoning" {
    try skipUnlessLinux();
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var r = try run(a, &tmp, .{ .thinking = true, .stream = true, .answer = "Bo has 42.", .then = "Answer: 42" });
    defer r.deinit();
    errdefer std.debug.print("COMPUTE_LANE_CLOSE_STREAM content [{s}] reasoning [{s}]\n", .{ r.streamed.content.items, r.streamed.reasoning.items });
    try t.expectEqual(@as(usize, 3), r.script.resumes.items.len);
    try t.expectEqualStrings("Answer: 42", std.mem.trim(u8, r.streamed.content.items, " \n"));
    try t.expect(std.mem.indexOf(u8, r.streamed.reasoning.items, "Bo has 42.") != null);
}

test "compute lane: with no room for that turn, the unclosed text is the content" {
    try skipUnlessLinux();
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var roomy = t.tmpDir(.{});
    defer roomy.cleanup();
    var first = try run(a, &roomy, .{ .thinking = true, .answer = "Bo has 42.", .then = "Answer: 42" });
    defer first.deinit();
    const rs = first.script.resumes.items;
    // one token short of the close turn and its answer room (compute_segments.answer_room, 16); the result still fits
    const used = first.script.call.len + rs[0].len + first.script.answer.len;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var r = try run(a, &tmp, .{ .thinking = true, .answer = "Bo has 42.", .then = "Answer: 42", .max_tokens = @intCast(used + rs[1].len + 16 - 1) });
    defer r.deinit();
    errdefer std.debug.print("COMPUTE_LANE_NO_ROOM {s}\n", .{json.stringify(a, r.reply.body, .{}) catch ""});
    try t.expectEqual(@as(usize, 2), r.script.resumes.items.len);
    try t.expectEqual(@as(usize, 0), r.script.resumes.items[1].len);
    try t.expect(std.mem.indexOf(u8, message(&r).content, "Bo has 42.") != null);
}
