//! The compute lane's request side: the built-in `python` tool a request opts into, arming, and the client's view.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const errors = @import("errors.zig");
const chat = @import("chat.zig");
const tool_specs = @import("tool_specs.zig");
const Server = @import("server.zig").Server;
const lane = api.hold.compute_lane;
const Value = json.Value;
const Cx = errors.Cx;
const Allocator = std.mem.Allocator;

/// The default binding's tool name and opt-in type (a family's: srv.info.compute_tool), and our description.
pub const name = "python";
pub const opt_in = "code_interpreter";
pub const description = "Run Python 3 code in a sandbox with no network, no files and no installs, and get its stdout and " ++
    "stderr. print() what you need; the value of a final bare expression is printed too.";
const head =
    \\{"type":"function","function":{"name":"
;
const tail =
    \\","description":"
++ description ++
    \\","parameters":{"type":"object","properties":{"code":{"type":"string","description":"Python 3 source to run"}},"required":["code"]}}}
;
pub const definition = head ++ name ++ tail;

/// The built-in tool's definition under the binding's name `n`.
fn definitionOf(a: Allocator, n: []const u8) Allocator.Error![]const u8 {
    return std.mem.concat(a, u8, &.{ head, n, tail });
}

/// The server's sandbox pool for an engine that serves the lane; boot fails when an opted-in sandbox cannot run.
pub fn open(srv: *Server, cfg: lane.Config) !lane.Pool {
    if (srv.engine.vtable.resume_held == null) {
        std.log.err("compute lane: this engine cannot hold a reply for a tool result", .{});
        return error.ComputeLaneUnsupported;
    }
    var pool = lane.Pool.init(cfg, &.{ "/proc/self/exe", "__compute_sandbox" });
    var arena: std.heap.ArenaAllocator = .init(srv.gpa);
    defer arena.deinit();
    pool.selfTest(arena.allocator()) catch |e| {
        std.log.err("compute lane: the sandbox self-test failed ({t}); unset the compute lane flag or fix the sandbox", .{e});
        return error.ComputeLaneUnavailable;
    };
    std.log.info("compute lane: {s} in a sandbox, {d} at a time, {d} ms, {d} MiB, results up to {d} tokens, {d} calls a reply", .{ cfg.python, cfg.workers, cfg.max_ms, cfg.mem_mb, cfg.max_result_tokens, cfg.max_calls });
    if (cfg.default) std.log.info("compute default: every thinking chat gets the built-in tool, whatever it sends", .{});
    return pool;
}

/// Whether thinking chats get the built-in tool, whatever they send (the compute default).
pub fn forced(srv: *const Server) bool {
    return srv.compute != null and srv.info.compute.?.default;
}

/// The default binding's max_tokens under which the default offers no built-in tool unasked (Tool.min_tokens).
pub const default_min_tokens = 1024;

/// What a request asks, known before it renders, that decides whether its reply can hold for the lane (`arms`).
pub const Fit = struct { max_tokens: i64, thinking: bool = true, stop: bool = false, logprobs: bool = false, structured: bool = false };

/// Whether a reply that asked `f` thinks, holds at its EOS and has `min` tokens for a call, its result and an answer.
fn fits(f: Fit, min: i64) bool {
    // unthinking, the prose before a call is the visible answer and the result's answer runs on into it
    return f.thinking and !f.stop and !f.logprobs and !f.structured and f.max_tokens >= min;
}

/// A chat body's tools (`tools`) and `lane_own` into `input`, whose max_tokens is read; `logprobs`: rows asked.
pub fn requestTools(srv: *Server, cx: *Cx, body: Value, input: *chat.Input, logprobs: bool) errors.Refused!void {
    var structured = false;
    for (@import("grammar.zig").fields) |k| structured = structured or json.truthyField(body, k);
    // chat.zig's limit: max_tokens, or the server's default for 0 and unset
    const budget = if (input.max_tokens) |m| (if (m != 0) m else srv.config.default_max_tokens) else srv.config.default_max_tokens;
    const thinking = (try @import("fields.zig").thinkingFields(cx, body, srv.effort_levels)).enable orelse srv.config.enable_thinking;
    const fit: Fit = .{ .max_tokens = budget, .thinking = thinking, .stop = json.truthyField(body, "stop"), .logprobs = logprobs or json.truthyField(body, "prompt_logprobs"), .structured = structured };
    input.tools = try tools(srv, cx, body.get("tools"), body.get("tool_choice"), fit);
    input.lane_own = ownedByServer(srv, body.get("tools"), input.tools);
}

/// The request's tools, `{"type": "code_interpreter"}` as the built-in tool (the default's: every chat that `fit`s).
pub fn tools(srv: *Server, cx: *Cx, raw: ?Value, choice: ?Value, fit: Fit) errors.Refused![]Value {
    const always = forced(srv);
    const tool = srv.info.compute_tool;
    const list = raw orelse Value{ .array = &.{} };
    if (list != .array and !always) return active(srv, cx, raw, choice);
    var rest: std.ArrayList(Value) = .empty;
    var listed = false;
    if (list == .array) for (list.array) |t| {
        const kind = if (t == .object) t.strField("type") orelse "" else "";
        if (std.mem.eql(u8, kind, tool.opt_in)) listed = true else try rest.append(cx.a, t);
    };
    if (!listed and (!always or !fits(fit, tool.min_tokens))) return active(srv, cx, raw, choice);
    if (srv.compute == null) return cx.other(try std.fmt.allocPrint(cx.a, "{s} is not enabled on this server", .{tool.opt_in}));
    var specs = try active(srv, cx, if (list == .array) .{ .array = rest.items } else raw, choice);
    // a client's own copy of the built-in definition is the same tool: one declaration, not a refusal (default only)
    if (always) specs = try withoutBuiltIn(cx.a, specs, tool.name);
    for (specs) |t| if (std.mem.eql(u8, toolName(t), tool.name)) {
        // a call names only its tool: the default leaves `python` to the client that offers it
        if (always and !listed) return specs;
        return cx.other(try std.fmt.allocPrint(cx.a, "a request with {s} cannot also offer a function named {s}", .{ tool.opt_in, tool.name }));
    };
    // only an automatic choice offers it ("required" and a named function do not); the default also offers it on "none"
    if (choice) |c| if (!(c == .null or (c == .string and std.mem.eql(u8, c.string, "auto")) or
        (always and c == .string and std.mem.eql(u8, c.string, "none")))) return specs;
    const built = switch (try json.parse(cx.a, try definitionOf(cx.a, tool.name))) {
        .ok => |v| v,
        .err => unreachable,
    };
    return std.mem.concat(cx.a, Value, &.{ specs, &.{built} });
}

/// The runner receipt's cell forms: each counts the cells the engine wrote a value for (none without the runner).
const forms = [_][]const u8{ "fence", "repl", "inline", "trace" };

/// `runtime.compute_executed` under the compute default: built-in calls run (`ran`) plus runner cells written, 0: none.
pub fn countExecuted(a: Allocator, runtime: Value, ran: usize) Allocator.Error!void {
    if (runtime != .object) return;
    const calls: i64 = @intCast(ran);
    var cells: i64 = 0;
    if (runtime.get("runner")) |r| for (forms) |form| if (r.get(form)) |f| if (f.get("owned")) |n| {
        cells += n.int64() orelse 0;
    };
    const o = try json.newObject(a);
    try o.put(a, "calls", try json.intValue(a, calls));
    try o.put(a, "runner_cells", try json.intValue(a, cells));
    try o.put(a, "total", try json.intValue(a, calls + cells));
    try runtime.object.put(a, "compute_executed", .{ .object = o });
}

fn withoutBuiltIn(a: Allocator, specs: []Value, n: []const u8) Allocator.Error![]Value {
    var out: std.ArrayList(Value) = .empty;
    for (specs) |t| if (!isBuiltInAs(t, n)) try out.append(a, t);
    return out.items;
}

/// The client's own tools as the server reads them: DeepSeek's native encoder checks its tool schema first.
fn active(srv: *Server, cx: *Cx, raw: ?Value, choice: ?Value) errors.Refused![]Value {
    if (srv.text.native_deepseek) return @import("deepseek_request.zig").active(cx, raw, choice);
    return tool_specs.active(cx, raw, choice);
}

fn toolName(t: Value) []const u8 {
    const f = t.get("function") orelse return "";
    return f.strField("name") orelse "";
}

/// Whether `t` is the default binding's built-in tool, as `tools` added it.
pub fn isBuiltIn(t: Value) bool {
    return isBuiltInAs(t, name);
}

/// Whether `t` is the built-in tool (its name `n` and our description), as `tools` added it.
pub fn isBuiltInAs(t: Value, n: []const u8) bool {
    const f = t.get("function") orelse return false;
    return std.mem.eql(u8, f.strField("name") orelse "", n) and std.mem.eql(u8, f.strField("description") orelse "", description);
}

/// Whether `specs` hold the default's own built-in tool (no code_interpreter listed): its calls never reach the client.
pub fn ownedByServer(srv: *const Server, raw: ?Value, specs: []const Value) bool {
    if (!forced(srv)) return false;
    if (raw) |list| if (list == .array) for (list.array) |t| {
        if (t == .object and std.mem.eql(u8, t.strField("type") orelse "", srv.info.compute_tool.opt_in)) return false;
    };
    for (specs) |t| if (isBuiltInAs(t, srv.info.compute_tool.name)) return true;
    return false;
}

/// A reply's calls for its client (`own`: the default's tool `n`, whose unrun calls end the reply at length, unseen).
pub fn ownCalls(a: Allocator, own: bool, n: []const u8, reply: *chat.Reply, found: ?[]Value, prose: []const u8) Allocator.Error!?[]Value {
    if (!own) return found;
    const all: []const Value = found orelse &.{};
    var kept: std.ArrayList(Value) = .empty;
    for (all) |c| if (!isCallTo(c, n)) try kept.append(a, c);
    const dropped = kept.items.len < all.len;
    // a call the lane could not run (no room left), or one max_tokens cut: the budget ended the reply
    if (dropped or (found == null and std.mem.eql(u8, reply.finish_reason, "length"))) {
        reply.content = try @import("reply_text.zig").cutProse(a, prose);
        reply.finish_reason = "length";
    }
    return if (kept.items.len > 0) kept.items else null;
}

/// Whether a call names the default binding's built-in tool.
pub fn isBuiltInCall(call: Value) bool {
    return isCallTo(call, name);
}

/// Whether a call names the built-in tool `n`.
pub fn isCallTo(call: Value, n: []const u8) bool {
    return std.mem.eql(u8, toolName(call), n);
}

/// Whether a prepared request holds at its EOS for the lane: offered the tool, nothing that forced rows would break.
pub fn arms(srv: *const Server, input: chat.Input, r: *const api.Request) bool {
    if (srv.compute == null or r.eos.len == 0 or r.background or r.max_tokens == 0) return false;
    if (srv.text.vtable.format == .kimi) return false; // its native reader keeps calls apart from the text
    if (r.structure != null or r.call != null or r.logprobs != null or r.prompt_logprobs != null) return false;
    for (input.tools) |t| if (isBuiltInAs(t, srv.info.compute_tool.name)) return true;
    return false;
}

/// A lane reply as the client streams it: built-in calls stay hidden; client calls number on across segments.
pub const Filter = struct {
    a: Allocator,
    inner: chat.Sink,
    tool: []const u8 = name, // the built-in tool's name (srv.info.compute_tool)
    base: i64 = 0, // client calls earlier segments showed
    seen: i64 = 0, // client calls this segment showed
    map: std.AutoHashMapUnmanaged(i64, ?i64) = .empty, // this segment's call index -> the client's (null: built-in)

    pub fn sink(f: *Filter) chat.Sink {
        return .{ .ctx = f, .call = call, .begin = if (f.inner.begin != null) begin else null };
    }

    /// A resumed segment numbers its calls from 0 again.
    pub fn nextSegment(f: *Filter) void {
        f.base += f.seen;
        f.seen = 0;
        f.map.clearRetainingCapacity();
    }

    /// A delta straight to the client (the lane's own reports).
    pub fn direct(f: *Filter, delta: Value) error{Closed}!void {
        return f.inner.call(f.inner.ctx, delta);
    }

    fn begin(ctx: *anyopaque, prompt_len: usize) error{Closed}!void {
        const f: *Filter = @ptrCast(@alignCast(ctx));
        return f.inner.begin.?(f.inner.ctx, prompt_len);
    }

    fn call(ctx: *anyopaque, delta: Value) error{Closed}!void {
        const f: *Filter = @ptrCast(@alignCast(ctx));
        const list = if (delta == .object) delta.get("tool_calls") else null;
        if (list == null or list.? != .array) return f.inner.call(f.inner.ctx, delta);
        var kept: std.ArrayList(Value) = .empty;
        for (list.?.array) |one| {
            const at = (one.get("index") orelse continue).int64() orelse continue;
            if (one.get("function")) |fun| if (fun.strField("name")) |n| if (!f.map.contains(at)) {
                const client: ?i64 = if (std.mem.eql(u8, n, f.tool)) null else f.base + f.seen;
                if (client != null) f.seen += 1;
                f.map.put(f.a, at, client) catch return error.Closed;
            };
            const to = (f.map.get(at) orelse continue) orelse continue;
            if (one != .object) continue;
            one.object.put(f.a, "index", json.intValue(f.a, to) catch return error.Closed) catch return error.Closed;
            kept.append(f.a, one) catch return error.Closed;
        }
        if (kept.items.len == 0) return;
        delta.object.put(f.a, "tool_calls", .{ .array = kept.items }) catch return error.Closed;
        return f.inner.call(f.inner.ctx, delta);
    }
};

test "compute lane: the filter drops built-in call deltas and numbers client calls on across segments" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const Seen = struct {
        out: std.ArrayList([]const u8) = .empty,
        a: Allocator,
        fn call(ctx: *anyopaque, delta: Value) error{Closed}!void {
            const s: *@This() = @ptrCast(@alignCast(ctx));
            s.out.append(s.a, json.stringify(s.a, delta, .{}) catch return error.Closed) catch return error.Closed;
        }
    };
    var seen: Seen = .{ .a = a };
    var f: Filter = .{ .a = a, .inner = .{ .ctx = &seen, .call = Seen.call } };
    const s = f.sink();
    const deltas = [_][]const u8{
        \\{"tool_calls":[{"index":0,"id":"a","type":"function","function":{"name":"python","arguments":""}}]}
        ,
        \\{"tool_calls":[{"index":0,"function":{"arguments":"print(1)"}}]}
        ,
        \\{"tool_calls":[{"index":1,"id":"b","type":"function","function":{"name":"lookup","arguments":"{}"}}]}
        ,
    };
    for (deltas) |d| try s.call(s.ctx, (try json.parse(a, d)).ok);
    try s.call(s.ctx, .{ .string = "prose" });
    f.nextSegment();
    try s.call(s.ctx, (try json.parse(a, deltas[2])).ok);
    try std.testing.expectEqual(@as(usize, 3), seen.out.items.len);
    try std.testing.expect(std.mem.indexOf(u8, seen.out.items[0], "\"index\": 0") != null);
    try std.testing.expect(std.mem.indexOf(u8, seen.out.items[0], "lookup") != null);
    try std.testing.expectEqualStrings("\"prose\"", seen.out.items[1]);
    try std.testing.expect(std.mem.indexOf(u8, seen.out.items[2], "\"index\": 1") != null);
}

test "compute lane: the built-in definition parses and is recognized; another function named python is not" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const built = (try json.parse(arena.allocator(), definition)).ok;
    try std.testing.expect(isBuiltIn(built));
    const other = (try json.parse(arena.allocator(), "{\"type\":\"function\",\"function\":{\"name\":\"python\"}}")).ok;
    try std.testing.expect(!isBuiltIn(other));
}

test "compute lane: a binding's tool name renames the built-in definition; the default's bytes are as before" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try std.testing.expectEqualStrings(definition, try definitionOf(a, name));
    try std.testing.expect(std.mem.startsWith(u8, definition, "{\"type\":\"function\",\"function\":{\"name\":\"python\",\"description\":\"Run Python 3 code"));
    const t: lane.Tool = .{};
    try std.testing.expectEqualStrings(name, t.name);
    try std.testing.expectEqualStrings(opt_in, t.opt_in);
    try std.testing.expectEqual(@as(i64, default_min_tokens), t.min_tokens);
    const run = (try json.parse(a, try definitionOf(a, "run_code"))).ok;
    try std.testing.expect(isBuiltInAs(run, "run_code") and !isBuiltIn(run));
    const call = try json.parse(a, "{\"function\":{\"name\":\"run_code\",\"arguments\":\"{}\"}}");
    try std.testing.expect(isCallTo(call.ok, "run_code") and !isBuiltInCall(call.ok));
}
