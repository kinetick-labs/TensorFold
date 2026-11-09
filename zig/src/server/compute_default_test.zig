//! The compute default (TF_DSV41_COMPUTE_LANE_DEFAULT=1) through the chat route: thinking chats get the built-in tool.
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
const bash_tool =
    \\{"type":"function","function":{"name":"bash","description":"Run a shell command","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}}
;

/// What one chat through the route did: its status, body, the prompt the engine got and the resumes it saw.
const Run = struct { code: u16, body: json.Value, prompt: []const u32, held: bool, resumes: usize };

/// The model's first segment: a python call, one with no code, a bash call, or a plain answer; then "It is 42.".
const Model = enum { python_call, no_code, bash_call, answer };

const Ask = struct { default: bool, tools: ?[]const u8 = null, choice: ?[]const u8 = null, model: Model = .answer, max_calls: u32 = 8, think: bool = true };

fn chat(a: Allocator, ask: Ask) !Run {
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
    const call_turn = try h.obj(a, switch (ask.model) {
        .python_call =>
        \\{"role":"assistant","content":"","tool_calls":[{"id":"c1","type":"function","function":{"name":"python","arguments":"{\"code\":\"print(6*7)\"}"}}]}
        ,
        .no_code =>
        \\{"role":"assistant","content":"","tool_calls":[{"id":"c1","type":"function","function":{"name":"python","arguments":"{}"}}]}
        ,
        .bash_call =>
        \\{"role":"assistant","content":"","tool_calls":[{"id":"c1","type":"function","function":{"name":"bash","arguments":"{\"command\":\"ls\"}"}}]}
        ,
        .answer =>
        \\{"role":"assistant","content":"It is 42."}
        ,
    });
    const called = try text.renderIds(a, try h.appendTo(a, history, &.{call_turn}), .{ .add_generation_prompt = false, .enable_thinking = ask.think }, &problem);
    const eos = called[called.len - 1];
    // thinking, the tool-result turn reopens <think>: the answer closes it first, as the call turn did
    const close_id = [_]u32{text.tokenId("</think>").?};
    const close: []const u32 = if (ask.think) &close_id else &.{};
    const answer = try std.mem.concat(a, u32, &.{ close, try text.encodePlain(a, "It is 42."), &.{eos} });
    const script = try a.create(h.Script);
    script.* = .{ .a = a, .call = called[asked.len..], .answer = answer };
    const srv = try @import("server.zig").Server.init(a, t.io, script.engine(), text, .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false }, null);
    defer srv.deinit();
    const cfg: lane.Config = .{ .python = python, .default = ask.default, .max_calls = ask.max_calls };
    srv.info.compute = cfg;
    srv.compute = lane.Pool.init(cfg, &.{ "/bin/sh", "-c", "exec \"$0\" -I -B -u -c \"$1\"", python, lane.runner.runner });
    var reply: h.Reply = .{ .a = a };
    var conn = try @import("http_conn.zig").Conn.init(a, -1, "fixture");
    defer conn.deinit();
    const body = try json.newObject(a);
    try body.put(a, "model", .{ .string = "m" });
    try body.put(a, "max_tokens", try json.intValue(a, 2048)); // past turns.default_min_tokens
    try body.put(a, "messages", history);
    if (ask.tools) |tools| try body.put(a, "tools", try h.obj(a, tools));
    if (ask.choice) |c| try body.put(a, "tool_choice", try h.obj(a, c));
    if (ask.think) try body.put(a, "chat_template_kwargs", try h.obj(a, "{\"enable_thinking\":true}"));
    @import("openai.zig").run(srv, a, reply.out(), .{ .conn = &conn }, true, .{ .object = body });
    errdefer std.debug.print("COMPUTE_DEFAULT_REPLY {s}\n", .{json.stringify(a, reply.body, .{}) catch ""});
    return .{ .code = reply.code, .body = reply.body, .prompt = script.prompt, .held = script.held_asked, .resumes = script.resumes.items.len };
}

fn executed(r: Run) ?json.Value {
    const extras = r.body.get("tensorfold") orelse return null;
    return extras.get("compute_executed");
}

fn count(r: Run, key: []const u8) !i64 {
    return (executed(r) orelse return error.NoCounter).get(key).?.int64().?;
}

test "compute lane: the default gives a chat with no tools the built-in tool, runs its call and counts it" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const r = try chat(a, .{ .default = true, .model = .python_call });
    try t.expectEqual(@as(u16, 200), r.code);
    try t.expect(r.held);
    try t.expectEqual(@as(usize, 2), r.resumes); // the result forced in, then the answer's empty resume
    try t.expectEqual(@as(i64, 1), try count(r, "calls"));
    try t.expectEqual(@as(i64, 0), try count(r, "runner_cells"));
    try t.expectEqual(@as(i64, 1), try count(r, "total"));
    // the prompt is the one an explicit opt-in renders: the measured layout, declared once
    const opted = try chat(a, .{ .default = false, .tools = "[{\"type\":\"code_interpreter\"}]", .model = .python_call });
    try t.expectEqualSlices(u32, opted.prompt, r.prompt);
}

test "compute lane: the default offers no built-in tool to a chat with thinking off: its prompt is the default off's" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const r = try chat(a, .{ .default = true, .think = false });
    const off = try chat(a, .{ .default = false, .think = false });
    try t.expectEqual(@as(u16, 200), r.code);
    try t.expectEqualSlices(u32, off.prompt, r.prompt);
    try t.expect(!r.held);
    try t.expectEqualStrings("It is 42.", r.body.get("choices").?.array[0].get("message").?.strField("content").?);
    // a client's own tools stay as they are
    const tools = "[" ++ bash_tool ++ "]";
    try t.expectEqualSlices(u32, (try chat(a, .{ .default = false, .tools = tools, .think = false })).prompt, (try chat(a, .{ .default = true, .tools = tools, .think = false })).prompt);
}

test "compute lane: under the default a chat that runs nothing counts 0" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const r = try chat(arena.allocator(), .{ .default = true });
    try t.expectEqual(@as(u16, 200), r.code);
    try t.expect(r.held);
    try t.expectEqual(@as(i64, 0), try count(r, "calls"));
    try t.expectEqual(@as(i64, 0), try count(r, "total"));
}

test "compute lane: under the default a client's own tools stay, the built-in joins once, their calls go back" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = "[" ++ bash_tool ++ "]";
    const r = try chat(a, .{ .default = true, .tools = tools });
    const opted = try chat(a, .{ .default = false, .tools = "[" ++ bash_tool ++ ",{\"type\":\"code_interpreter\"}]" });
    try t.expectEqualSlices(u32, opted.prompt, r.prompt);
    try t.expect(r.held);
    // the client's call ends the reply for the client: nothing runs in the engine
    const c = try chat(a, .{ .default = true, .tools = tools, .model = .bash_call });
    try t.expectEqual(@as(u16, 200), c.code);
    const choice = c.body.get("choices").?.array[0];
    try t.expectEqualStrings("tool_calls", choice.get("finish_reason").?.string);
    const call = choice.get("message").?.get("tool_calls").?.array[0];
    try t.expectEqualStrings("bash", call.get("function").?.strField("name").?);
    try t.expectEqual(@as(i64, 0), try count(c, "total"));
}

test "compute lane: under the default a client's own code_interpreter or built-in copy declares it once" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const none = try chat(a, .{ .default = true });
    const twice = try chat(a, .{ .default = true, .tools = "[{\"type\":\"code_interpreter\"},{\"type\":\"code_interpreter\"}]" });
    const copy = try chat(a, .{ .default = true, .tools = "[" ++ turns.definition ++ ",{\"type\":\"code_interpreter\"}]" });
    try t.expectEqual(@as(u16, 200), twice.code);
    try t.expectEqual(@as(u16, 200), copy.code);
    try t.expectEqualSlices(u32, none.prompt, twice.prompt);
    try t.expectEqualSlices(u32, none.prompt, copy.prompt);
    // its python call runs in the engine and counts
    const ran = try chat(a, .{ .default = true, .tools = "[{\"type\":\"code_interpreter\"}]", .model = .python_call });
    try t.expectEqual(@as(u16, 200), ran.code);
    try t.expectEqual(@as(usize, 2), ran.resumes);
    try t.expectEqual(@as(i64, 1), try count(ran, "total"));
    // code_interpreter next to the client's own python function refuses as it does with the default off
    const both = "[{\"type\":\"code_interpreter\"}," ++ client_python ++ "]";
    try t.expect((try chat(a, .{ .default = true, .tools = both })).code >= 400);
    try t.expect((try chat(a, .{ .default = false, .tools = both })).code >= 400);
}

const client_python =
    \\{"type":"function","function":{"name":"python","description":"Run Python on my machine","parameters":{"type":"object","properties":{"code":{"type":"string"}},"required":["code"]}}}
;

test "compute lane: under the default a client's own python function keeps the name, its call goes back" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = "[" ++ client_python ++ "]";
    const r = try chat(a, .{ .default = true, .tools = tools, .model = .python_call });
    try t.expectEqual(@as(u16, 200), r.code);
    try t.expect(!r.held); // no built-in tool: nothing holds, nothing runs in the sandbox
    const choice = r.body.get("choices").?.array[0];
    try t.expectEqualStrings("tool_calls", choice.get("finish_reason").?.string);
    const call = choice.get("message").?.get("tool_calls").?.array[0];
    try t.expectEqualStrings("python", call.get("function").?.strField("name").?);
    try t.expect(std.mem.indexOf(u8, call.get("function").?.strField("arguments").?, "print(6*7)") != null);
    try t.expectEqual(@as(i64, 0), try count(r, "total"));
    // the prompt is the client's own, as with the default off
    const off = try chat(a, .{ .default = false, .tools = tools, .model = .python_call });
    try t.expectEqualSlices(u32, off.prompt, r.prompt);
}

test "compute lane: the executed counter skips a call with no code and a call past the cap" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const malformed = try chat(a, .{ .default = true, .model = .no_code });
    try t.expectEqual(@as(u16, 200), malformed.code);
    try t.expectEqual(@as(usize, 1), malformed.body.get("tensorfold").?.get("compute").?.get("calls").?.array.len);
    try t.expectEqual(@as(i64, 0), try count(malformed, "calls"));
    try t.expectEqual(@as(i64, 0), try count(malformed, "total"));
    const capped = try chat(a, .{ .default = true, .model = .python_call, .max_calls = 0 });
    try t.expectEqual(@as(u16, 200), capped.code);
    try t.expectEqual(@as(usize, 1), capped.body.get("tensorfold").?.get("compute").?.get("calls").?.array.len);
    try t.expectEqual(@as(i64, 0), try count(capped, "total"));
}

test "compute lane: under the default tool_choice none hides the client's tools but not the built-in" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const r = try chat(a, .{ .default = true, .tools = "[" ++ bash_tool ++ "]", .choice = "\"none\"" });
    const opted = try chat(a, .{ .default = false, .tools = "[{\"type\":\"code_interpreter\"}]" });
    try t.expectEqual(@as(u16, 200), r.code);
    try t.expectEqualSlices(u32, opted.prompt, r.prompt);
    try t.expect(r.held);
}

test "compute lane: with the default off a chat is as before: no built-in tool, no hold, no counter" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = try h.fixture(a);
    defer tmp.deinit();
    var problem: []const u8 = "";
    const hf = try HfText.load(a, t.io, tmp.path, null, a, &problem);
    defer hf.deinit();
    const plain = try hf.text().renderIds(a, try tf.messages(a, "What is 6*7?"), .{ .add_generation_prompt = true }, &problem);
    const r = try chat(a, .{ .default = false, .think = false });
    try t.expectEqual(@as(u16, 200), r.code);
    try t.expectEqualSlices(u32, plain, r.prompt);
    try t.expect(!r.held);
    try t.expect(executed(r) == null);
    const opted = try chat(a, .{ .default = false, .tools = "[{\"type\":\"code_interpreter\"}]", .model = .python_call });
    try t.expect(opted.held and executed(opted) == null);
    const tools = try chat(a, .{ .default = false, .tools = "[" ++ bash_tool ++ "]" });
    try t.expect(!tools.held and executed(tools) == null);
}

test "compute lane: the executed counter adds built-in calls run and the runner cells the engine wrote" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // three reports, two of them run: the counter takes the runs, not the reports
    const runtime = try h.obj(a,
        \\{"compute":{"calls":[{"name":"python"},{"name":"python"},{"name":"python"}]},"runner":{"fence":{"closed":3,"owned":2,"abstained":1,"owned_tokens":9},"repl":{"closed":0,"owned":0,"abstained":0,"owned_tokens":0},"inline":{"closed":1,"owned":1,"abstained":0,"owned_tokens":2},"trace":{"closed":0,"owned":0,"abstained":0,"owned_tokens":0}}}
    );
    try turns.countExecuted(a, runtime, 2);
    const c = runtime.get("compute_executed").?;
    try t.expectEqual(@as(i64, 2), c.get("calls").?.int64().?);
    try t.expectEqual(@as(i64, 3), c.get("runner_cells").?.int64().?);
    try t.expectEqual(@as(i64, 5), c.get("total").?.int64().?);
    const empty = try h.obj(a, "{}");
    try turns.countExecuted(a, empty, 0);
    try t.expectEqual(@as(i64, 0), empty.get("compute_executed").?.get("total").?.int64().?);
    const cell = try h.obj(a,
        \\{"runner":{"fence":{"closed":1,"owned":1,"abstained":0,"owned_tokens":3}}}
    );
    try turns.countExecuted(a, cell, 0);
    try t.expectEqual(@as(i64, 1), cell.get("compute_executed").?.get("total").?.int64().?);
}
