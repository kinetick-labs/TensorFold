//! The compute lane on a template closing turns with `<|im_end|>\n` (Qwen3.x, Qwen3.8-Flash-Next): the reply goes on.
const std = @import("std");
const json = @import("json");
const HfText = @import("hf_text.zig").HfText;
const tf = @import("control_tokens_fixture.zig");
const h = @import("compute_lane_http_test.zig");
const lane = @import("engine_api").hold.compute_lane;
const t = std.testing;

const template = @embedFile("fixtures/dory/chat_template.jinja");

test "compute lane: a ChatML turn close after the held EOS goes in with the result turn, not instead of it" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    if (@import("builtin").os.tag != .linux) return error.SkipZigTest; // the lane's executor runs on Linux only
    const python = "/usr/bin/python3";
    std.Io.Dir.accessAbsolute(t.io, python, .{}) catch return error.SkipZigTest;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    // the byte-level mini vocabulary; the call and think markers are text, as in a Qwen tokenizer
    const tok = try h.obj(a, @embedFile("fixtures/control_tokens/mini_tokenizer.json"));
    for (tok.get("added_tokens").?.array) |added| {
        const name = added.strField("content").?;
        if (!std.mem.startsWith(u8, name, "<|")) try added.object.put(a, "special", .{ .bool = false });
    }
    try tok.object.put(a, "pre_tokenizer", try h.obj(a, "{\"type\":\"ByteLevel\",\"add_prefix_space\":false,\"trim_offsets\":false,\"use_regex\":false}"));
    try tmp.dir.writeFile(t.io, .{ .sub_path = "tokenizer.json", .data = try json.stringify(a, tok, .{}) });
    try tmp.dir.writeFile(t.io, .{ .sub_path = "config.json", .data = "{\"model_type\":\"dory\",\"vocab_size\":2048}" });
    try tmp.dir.writeFile(t.io, .{ .sub_path = "tokenizer_config.json", .data = "{\"eos_token\":\"<|im_end|>\"}" });
    try tmp.dir.writeFile(t.io, .{ .sub_path = "chat_template.jinja", .data = template });
    const dir = try tmp.dir.realPathFileAlloc(t.io, ".", a);
    var problem: []const u8 = "";
    const hf = try HfText.load(a, t.io, dir, null, a, &problem);
    defer hf.deinit();
    const text = hf.text();
    const eos = text.tokenId("<|im_end|>").?;
    // the model's segments: a Qwen XML call held at its EOS, then the answer
    const call_text = "<tool_call>\n<function=python>\n<parameter=code>\nprint(6*7)\n</parameter>\n</function>\n</tool_call>\n";
    const call = try std.mem.concat(a, u32, &.{ try text.encodePlain(a, call_text), &.{eos} });
    const answer = try std.mem.concat(a, u32, &.{ try text.encodePlain(a, "It is 42."), &.{eos} });
    var script: h.Script = .{ .a = a, .call = call, .answer = answer };
    const srv = try @import("server.zig").Server.init(a, t.io, script.engine(), text, .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false }, null);
    defer srv.deinit();
    const cfg: lane.Config = .{ .python = python };
    srv.info.compute = cfg;
    srv.compute = lane.Pool.init(cfg, &.{ "/bin/sh", "-c", "exec \"$0\" -I -B -u -c \"$1\"", python, lane.runner.runner });
    var reply: h.Reply = .{ .a = a };
    var conn = try @import("http_conn.zig").Conn.init(a, -1, "fixture");
    defer conn.deinit();
    const body = try json.newObject(a);
    try body.put(a, "model", .{ .string = "m" });
    try body.put(a, "max_tokens", try json.intValue(a, 400));
    try body.put(a, "messages", try tf.messages(a, "What is 6*7?"));
    try body.put(a, "tools", try h.obj(a, "[{\"type\":\"code_interpreter\"}]"));
    @import("openai.zig").run(srv, a, reply.out(), .{ .conn = &conn }, true, .{ .object = body });
    const said = try json.stringify(a, reply.body, .{});
    errdefer std.debug.print("COMPUTE_LANE_CHATML_REPLY {s}\n", .{said});
    try t.expectEqual(@as(u16, 200), reply.code);
    // the result went in (not an empty resume ending the reply at the call), then the answer's EOS ended it
    try t.expectEqual(@as(usize, 2), script.resumes.items.len);
    try t.expectEqual(@as(usize, 0), script.resumes.items[1].len);
    // the forced rows: the template's own close after the EOS, the result turn, the next generation prompt
    try t.expectEqualStrings("\n<|im_start|>user\n<tool_response>\n42\n</tool_response>\n<|im_end|>\n<|im_start|>assistant\n<think></think>", try text.decode(a, script.resumes.items[0]));
    const choice = reply.body.get("choices").?.array[0];
    try t.expectEqualStrings("stop", choice.get("finish_reason").?.string);
    try t.expect(std.mem.indexOf(u8, choice.get("message").?.strField("content").?, "It is 42.") != null);
    const report = reply.body.get("tensorfold").?.get("compute").?.get("calls").?.array[0];
    try t.expectEqualStrings("42", report.strField("output").?);
    try t.expect(report.get("inserted").?.bool);
}
