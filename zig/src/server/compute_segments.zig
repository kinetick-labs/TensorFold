//! A lane reply's segments: at each held EOS its built-in calls run, their result turn is forced in, it goes on.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const chat = @import("chat.zig");
const reply_text = @import("reply_text.zig");
const tool_parse = @import("tool_parse.zig");
const reply_owned = @import("reply_owned.zig");
const tool_stream = @import("tool_stream.zig");
const prompt_mod = @import("prompt.zig");
const turns = @import("compute_turns.zig");
const log = @import("log.zig");
const Generation = @import("chat_generation.zig").Generation;
const Server = @import("server.zig").Server;
const lane = api.hold.compute_lane;
const Value = json.Value;
const Cx = @import("errors.zig").Cx;
const Allocator = std.mem.Allocator;

/// Tokens the reply keeps free after an inserted result, so the model can still answer.
const answer_room = 16;

/// What the earlier segments of a reply said and did, merged into its last segment's reply.
const Merge = struct {
    content: std.ArrayList(u8) = .empty,
    reasoning: std.ArrayList(u8) = .empty,
    completion: usize = 0,
    reasoning_tokens: usize = 0,
    ids: std.ArrayList(u32) = .empty, // the engine's whole continuation: segments and inserted results
    inserted: usize = 0,
    calls: std.ArrayList(Value) = .empty, // one report a call
    ran: usize = 0, // the calls the sandbox ran (Reply.compute_ran)
    owned: ?reply_owned.Owned = null, // the segments' engine-owned bytes in `content`
    history: std.ArrayList(Value) = .empty, // the request's messages, then each segment's call turn and results
};

/// Continues a reply whose first segment `first` ended held at its EOS, until the engine ends it.
pub fn rest(first: *Generation, cx: *Cx, p: chat.Prepared, filter: *turns.Filter, gone: anytype) chat.Failure!chat.Reply {
    const a = cx.a;
    const srv = first.srv;
    const cfg = srv.info.compute.?;
    var m: Merge = .{};
    if (p.input.messages == .array) try m.history.appendSlice(a, p.input.messages.array);
    var g = first;
    var run_calls: u32 = 0;
    while (g.held) {
        const used = m.ids.items.len + g.collected.items.len;
        const answer = try segmentAnswer(g);
        const calls = builtInOnly(answer.calls, srv.info.compute_tool.name);
        var plan: ?[]const u32 = null;
        var assistant: Value = .null;
        var reports: []Value = &.{};
        if (calls.len > 0 and used + answer_room < p.request.max_tokens) {
            assistant = try assistantTurn(a, answer.prose, answer.reasoning, calls);
            reports = try runAll(a, srv, cfg, calls, &run_calls, &m.ran, gone);
            plan = try suffix(srv, cx, p, &m.history, assistant, calls, reports);
            if (plan) |s| if (used + s.len + answer_room > p.request.max_tokens) {
                plan = null;
            };
        }
        const s = plan orelse {
            // no call to run, a client's call, or no room: the reply ends at this EOS as it would without the lane
            _ = api.hold.resumeHeld(srv.engine, g.id, .{ .tokens = &.{} });
            g.held = false;
            try g.loop(gone);
            for (reports) |r| {
                try r.object.put(a, "inserted", .{ .bool = false });
                try m.calls.append(a, r);
            }
            return merge(a, &m, try finish(g, cx, p), g.collected.items);
        };
        _ = api.hold.resumeHeld(srv.engine, g.id, .{ .tokens = s, .think_open = p.thinking });
        const seg = try finish(g, cx, p);
        try note(a, &m, g, seg, s);
        try m.history.append(a, assistant);
        for (calls, reports) |c, r| try m.history.append(a, try toolTurn(a, c, r.get("output").?.string));
        for (reports) |r| {
            try r.object.put(a, "inserted", .{ .bool = true });
            try m.calls.append(a, r);
            if (g.sink != null) filter.direct(try reportDelta(a, r)) catch return error.Cancelled;
        }
        g = try next(g, s, p.request.max_tokens - (used + s.len));
        g.base = m.ids.items.len; // the runner records' token indices count every segment and result before it
        filter.nextSegment();
        try g.loop(gone);
    }
    return merge(a, &m, try finish(g, cx, p), g.collected.items);
}

fn finish(g: *Generation, cx: *Cx, p: chat.Prepared) chat.Failure!chat.Reply {
    return g.finish(cx, p.prompt_len, p.received, p.thinking, p.reasoning, p.effort, p.sampled, p.drafts);
}

/// The segment after an inserted result: the same reply, its first tokens the result's, its own text and calls.
fn next(g: *Generation, inserted: []const u32, left: usize) chat.Failure!*Generation {
    const n = try g.a.create(Generation);
    n.* = .{ .srv = g.srv, .a = g.a, .box = g.box, .id = g.id, .reply_id = g.reply_id, .seed = g.seed, .sink = g.sink, .thinking = g.thinking, .chat = g.chat, .stops = g.stops, .ignore_eos = g.ignore_eos, .max_tokens = @intCast(left), .single = g.single, .tools = g.tools, .token_ids = g.token_ids, .loop_stop = g.loop_stop, .hold = true, .skip = inserted, .consumed = g.consumed, .chunk_index = g.chunk_index, .cells_sent = g.cells_sent, .cells_sig = g.cells_sig, .first_ns = g.first_ns, .wrote_ns = g.wrote_ns, .deadline_ns = g.deadline_ns, .deadline_hit = g.deadline_hit, .lane_own = g.lane_own };
    if (g.calls != null) n.calls = try tool_stream.Streamer.init(g.a, g.tools);
    return n;
}

/// The segment's calls when every one is the built-in tool's (a client's call ends the reply for its client).
fn builtInOnly(calls: ?[]Value, n: []const u8) []Value {
    const all = calls orelse return &.{};
    for (all) |c| if (!turns.isCallTo(c, n)) return &.{};
    return all;
}

const Answer = struct { prose: []const u8, reasoning: ?[]const u8, calls: ?[]Value };

/// The segment's answer as `Generation.finish` reads it, then its calls parsed out of the answer.
fn segmentAnswer(g: *Generation) chat.Failure!Answer {
    const tokens = reply_text.stripTrailing(g.collected.items, g.srv.eos);
    const read = try g.readText(try g.srv.text.decode(g.a, tokens), tokens);
    const parsed = try tool_parse.parse(g.a, read.content, g.tools, null);
    return .{ .prose = parsed.content, .reasoning = read.reasoning, .calls = parsed.calls };
}

/// Runs each call in order (results in call order); past the call cap, a note; `executed` counts calls the sandbox ran.
fn runAll(a: Allocator, srv: *Server, cfg: lane.Config, calls: []const Value, ran: *u32, executed: *usize, gone: anytype) chat.Failure![]Value {
    const Watch = struct {
        gone: @TypeOf(gone),
        fn left(ctx: *anyopaque) bool {
            const w: *@This() = @ptrCast(@alignCast(ctx));
            return w.gone.check();
        }
    };
    var watch: Watch = .{ .gone = gone };
    const out = try a.alloc(Value, calls.len);
    for (calls, out) |c, *slot| {
        const code = try codeOf(a, c);
        var o: lane.Outcome = .{ .status = .unavailable, .stderr = "the call has no string argument \"code\"" };
        var body: []const u8 = undefined;
        if (ran.* >= cfg.max_calls) {
            body = "[call limit reached: answer without running more code]";
        } else {
            if (code) |src| o = try srv.compute.?.run(a, src, .{ .ctx = &watch, .gone = Watch.left });
            if (o.status == .cancelled) return error.Cancelled;
            if (code != null and o.status != .unavailable) executed.* += 1;
            body = try lane.runner.body(a, o, cfg.max_ms, cfg.max_output_bytes);
            ran.* += 1;
        }
        slot.* = try report(a, srv.info.compute_tool.name, c, code orelse "", o, try capTokens(a, srv, body, cfg.max_result_tokens));
    }
    return out;
}

/// The call's "code" argument, its arguments a JSON object or a JSON string; null when missing or not a string.
fn codeOf(a: Allocator, call: Value) Allocator.Error!?[]const u8 {
    const f = call.get("function") orelse return null;
    const args = f.get("arguments") orelse return null;
    const obj: Value = if (args == .string) switch (try json.parseText(a, args.string)) {
        .ok => |v| v,
        .err => return null,
    } else args;
    const code = obj.strField("code") orelse return null;
    return if (std.mem.trim(u8, code, " \t\r\n").len > 0) code else null;
}

/// The result text cut to `max` tokens (decoded back at a character boundary), with a note when cut.
fn capTokens(a: Allocator, srv: *Server, body: []const u8, max: u32) Allocator.Error![]const u8 {
    const ids = srv.text.encodePlain(a, body) catch return body;
    if (ids.len <= max) return body;
    var n: usize = max;
    while (n > 0) : (n -= 1) {
        const text = try srv.text.decode(a, ids[0..n]);
        if (std.unicode.utf8ValidateSlice(text)) return std.fmt.allocPrint(a, "{s}\n[output truncated at {d} tokens]", .{ text, max });
    }
    return std.fmt.allocPrint(a, "[output truncated at {d} tokens]", .{max});
}

fn report(a: Allocator, n: []const u8, call: Value, code: []const u8, o: lane.Outcome, output: []const u8) Allocator.Error!Value {
    const r = try json.newObject(a);
    try r.put(a, "name", .{ .string = n });
    try r.put(a, "id", call.get("id") orelse .null);
    try r.put(a, "code", .{ .string = code });
    try r.put(a, "status", .{ .string = @tagName(o.status) });
    try r.put(a, "exit_code", if (o.exit_code) |x| try json.intValue(a, x) else .null);
    try r.put(a, "exec_ms", .{ .float = @as(f64, @floatFromInt(o.exec_ns)) / 1e6 });
    try r.put(a, "output", .{ .string = output });
    return .{ .object = r };
}

fn reportDelta(a: Allocator, r: Value) Allocator.Error!Value {
    const d = try json.newObject(a);
    try d.put(a, "compute", r);
    return .{ .object = d };
}

/// The assistant turn the segment wrote, as a client would send it back: its prose, reasoning and calls.
fn assistantTurn(a: Allocator, prose: []const u8, reasoning: ?[]const u8, calls: []Value) Allocator.Error!Value {
    const o = try json.newObject(a);
    try o.put(a, "role", .{ .string = "assistant" });
    try o.put(a, "content", .{ .string = prose });
    if (reasoning) |r| try o.put(a, "reasoning_content", .{ .string = r });
    try o.put(a, "tool_calls", .{ .array = calls });
    return .{ .object = o };
}

fn toolTurn(a: Allocator, call: Value, output: []const u8) Allocator.Error!Value {
    const o = try json.newObject(a);
    try o.put(a, "role", .{ .string = "tool" });
    try o.put(a, "tool_call_id", call.get("id") orelse .null);
    try o.put(a, "content", .{ .string = output });
    return .{ .object = o };
}

/// The template's tokens after the held EOS: render(history + call turn + results) past render(history + call turn).
fn suffix(srv: *Server, cx: *Cx, p: chat.Prepared, history: *const std.ArrayList(Value), assistant: Value, calls: []const Value, reports: []const Value) chat.Failure!?[]const u32 {
    const a = cx.a;
    var with: std.ArrayList(Value) = .empty;
    try with.appendSlice(a, history.items);
    try with.append(a, assistant);
    const before = prompt_mod.renderIds(srv, cx, .{ .array = try a.dupe(Value, with.items) }, p.input.tools, p.thinking, p.effort, false) catch |e| return lost(e);
    for (calls, reports) |c, r| try with.append(a, try toolTurn(a, c, r.get("output").?.string));
    const after = prompt_mod.renderIds(srv, cx, .{ .array = with.items }, p.input.tools, p.thinking, p.effort, true) catch |e| return lost(e);
    // the history must end at the model's own EOS and stand unchanged under the results: else no exact insert exists
    const ends = before.len > 0 and std.mem.indexOfScalar(u32, srv.eos, before[before.len - 1]) != null;
    if (!ends or after.len <= before.len or !std.mem.eql(u32, after[0..before.len], before)) {
        log.line("warning: compute lane: this template's tool-result turn does not extend the call turn; the reply ends at the call", .{});
        return null;
    }
    return after[before.len..];
}

fn lost(e: anyerror) chat.Failure!?[]const u32 {
    if (e == error.OutOfMemory) return error.OutOfMemory;
    log.line("warning: compute lane: the tool-result turn did not render ({t}); the reply ends at the call", .{e});
    return null;
}

/// A finished held segment into the merge: its words, its tokens and the result inserted after it.
fn note(a: Allocator, m: *Merge, g: *const Generation, seg: chat.Reply, inserted: []const u32) Allocator.Error!void {
    const parsed = try tool_parse.parseOwned(a, seg.content, g.tools, null, seg.owned);
    m.owned = reply_owned.Owned.join(m.owned, reply_owned.Owned.shift(parsed.owned, m.content.items.len));
    try m.content.appendSlice(a, parsed.content);
    if (seg.reasoning) |r| try m.reasoning.appendSlice(a, r);
    m.completion += seg.completion_tokens;
    m.reasoning_tokens += seg.reasoning_tokens;
    try m.ids.appendSlice(a, g.collected.items);
    try m.ids.appendSlice(a, inserted);
    m.inserted += inserted.len;
}

/// The last segment's reply with the earlier segments' words, counts, calls and tokens in front.
fn merge(a: Allocator, m: *Merge, last: chat.Reply, last_ids: []const u32) chat.Failure!chat.Reply {
    var r = last;
    if (m.content.items.len > 0) r.content = try std.mem.concat(a, u8, &.{ m.content.items, last.content });
    r.owned = reply_owned.Owned.join(m.owned, reply_owned.Owned.shift(last.owned, m.content.items.len));
    if (m.reasoning.items.len > 0) r.reasoning = try std.mem.concat(a, u8, &.{ m.reasoning.items, last.reasoning orelse "" });
    r.completion_tokens += m.completion;
    r.reasoning_tokens += m.reasoning_tokens;
    r.compute_ran = m.ran;
    if (m.calls.items.len == 0 and m.inserted == 0) return r;
    const rt = r.runtime.object;
    const tail = if (rt.get("token_ids")) |t| t.array else &[_]Value{};
    const full = try a.alloc(Value, m.ids.items.len + tail.len);
    for (m.ids.items, full[0..m.ids.items.len]) |t, *slot| slot.* = try json.intValue(a, t);
    @memcpy(full[m.ids.items.len..], tail);
    if (rt.get("token_ids") != null) try rt.put(a, "token_ids", .{ .array = full });
    // the shas hash the same whole continuation the ids list, inserted results included
    const all = try std.mem.concat(a, u32, &.{ m.ids.items, last_ids });
    const tokens = @import("tokens.zig");
    try rt.put(a, "token_sha", .{ .string = try a.dupe(u8, &tokens.tokenSha(all)) });
    try rt.put(a, "token_sha256", .{ .string = try a.dupe(u8, &tokens.tokenSha256(all)) });
    const c = try json.newObject(a);
    try c.put(a, "calls", .{ .array = m.calls.items });
    try c.put(a, "inserted_tokens", try json.intValue(a, m.inserted));
    try c.put(a, "prefix_tokens", try json.intValue(a, m.ids.items.len));
    try rt.put(a, "compute", .{ .object = c });
    return r;
}
