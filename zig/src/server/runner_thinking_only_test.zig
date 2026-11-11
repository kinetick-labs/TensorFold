//! The runner with spans `reasoning` through openai.run: a non-thinking chat gets no session, cells or changed bytes.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const openai = @import("openai.zig");
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
const Bytes = @import("runner_projection_http_test.zig").Bytes;
const model_text = @import("model_text.zig");
const Allocator = std.mem.Allocator;
const t = std.testing;

const close_id: u32 = 300;
const eos_id: u32 = 301;

/// The runner's scanner, reached through the session it opens (Session.scan).
const Scanner = @FieldType(std.meta.Child(std.meta.Child(@typeInfo(@typeInfo(@TypeOf(api.runner.Runner.session)).@"fn".return_type.?).error_union.payload)), "scan");

/// Gives its tokens in one round; with run_cells it scans them as a lane's session does and owns each trigger's sum.
const Script = struct {
    tokens: []const u32,
    runner: ?*api.runner.Runner = null,
    run_cells: bool = false,
    think_end: bool = false,
    closed: u32 = 0,
    owned: u32 = 0,
    fn engine(s: *Script) api.Engine {
        return .{ .ctx = s, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(ctx: *anyopaque) api.Info {
        const s: *Script = @ptrCast(@alignCast(ctx));
        return .{ .context_window = 8192, .runner = s.runner };
    }
    fn submit(ctx: *anyopaque, id: api.Id, r: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const s: *Script = @ptrCast(@alignCast(ctx));
        s.run_cells = r.run_cells;
        s.think_end = r.think_end != null;
        var out: std.ArrayList(u32) = .empty;
        defer out.deinit(t.allocator);
        s.play(r, &out) catch return error.MemoryPressure;
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        sink.event(sink.ctx, id, &.{ .tokens = out.items });
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .stop } });
    }
    fn play(s: *Script, r: *const api.Request, out: *std.ArrayList(u32)) Allocator.Error!void {
        const a = t.allocator;
        var codes: std.ArrayList([]u8) = .empty;
        defer {
            for (codes.items) |c| a.free(c);
            codes.deinit(a);
        }
        var scan: Scanner = .{ .cfg = if (s.runner) |rn| rn.cfg.scan else .{} };
        defer scan.deinit(a);
        var think = r.think_end != null;
        var skip = false;
        for (s.tokens) |tok| {
            if (skip and tok != '\n') continue;
            skip = false;
            try out.append(a, tok);
            if (!r.run_cells) continue;
            if (tok == close_id) {
                think = false;
                continue;
            }
            if (tok >= 256) continue;
            const ev = try scan.feed(a, &.{@intCast(tok)}, think);
            if (ev.closed) |c| {
                s.closed += 1;
                try codes.append(a, try a.dupe(u8, c.code));
            }
            const w = ev.trigger orelse continue;
            s.owned += 1;
            var buf: [24]u8 = undefined;
            const value = std.fmt.bufPrint(&buf, "{s}{d}", .{ if (w.inline_space orelse false) " " else "", sum(codes.items[w.index]) }) catch unreachable;
            for (value) |b| try out.append(a, b);
            skip = true;
        }
    }
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

/// `a + b` (a cell's code, spaces and all): the value the runner would own.
fn sum(code: []const u8) i64 {
    var total: i64 = 0;
    var it = std.mem.tokenizeAny(u8, code, "+ \n");
    while (it.next()) |n| total += std.fmt.parseInt(i64, n, 10) catch 0;
    return total;
}

/// Bytes' text with a template that has a thinking switch, so a chat may turn thinking off (server.zig's check).
fn switchable(b: *Bytes) model_text.Text {
    const S = struct {
        var vt: @typeInfo(@FieldType(model_text.Text, "vtable")).pointer.child = undefined;
        fn source(_: *anyopaque) []const u8 {
            return "{% if enable_thinking %}<think>{% endif %}";
        }
    };
    const base = b.text();
    S.vt = base.vtable.*;
    S.vt.template_source = S.source;
    return .{ .ctx = base.ctx, .vtable = &S.vt };
}

const Got = struct { reasoning: []const u8, content: []const u8 };

/// One chat through openai.run: `reasoning`, the think close and `content` when `thinking`, else `content` alone.
fn chat(a: Allocator, s: *Script, thinking: bool, reasoning: []const u8, content: []const u8, stream: bool) !Got {
    var ids: std.ArrayList(u32) = .empty;
    if (thinking) {
        for (reasoning) |b| try ids.append(a, b);
        try ids.append(a, close_id);
    }
    for (content) |b| try ids.append(a, b);
    try ids.append(a, eos_id);
    s.tokens = ids.items;
    var bytes: Bytes = .{};
    const srv = try Server.init(t.allocator, t.io, s.engine(), switchable(&bytes), .{ .served_name = "m", .model_ids = &.{"m"}, .use_drafts = false }, null);
    defer srv.deinit();
    const text = try std.fmt.allocPrint(a, "{{\"model\":\"m\",\"messages\":[{{\"role\":\"user\",\"content\":\"hi\"}}],\"max_tokens\":500,\"stream\":{},\"chat_template_kwargs\":{{\"enable_thinking\":{}}}}}", .{ stream, thinking });
    var conn = Conn{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a };
    var reply: Reply = .{ .a = a };
    openai.run(srv, a, reply.out(), .{ .conn = &conn }, true, (try json.parse(a, text)).ok);
    if (!stream) {
        const whole = reply.whole orelse return error.NoReply;
        const choices = whole.get("choices") orelse {
            std.debug.print("refused: {s}\n", .{try json.stringify(a, whole, .{})});
            return error.Refused;
        };
        const m = choices.array[0].get("message").?;
        return .{ .reasoning = json.strOr(m.get("reasoning_content")), .content = json.strOr(m.get("content")) };
    }
    var r: std.ArrayList(u8) = .empty;
    var c: std.ArrayList(u8) = .empty;
    for (reply.events.items) |v| {
        const cs = (v.get("choices") orelse continue).array;
        if (cs.len == 0) continue;
        const d = cs[0].get("delta") orelse continue;
        try r.appendSlice(a, json.strOr(d.get("reasoning_content")));
        try c.appendSlice(a, json.strOr(d.get("content")));
    }
    return .{ .reasoning = r.items, .content = c.items };
}

const Reply = struct {
    a: Allocator,
    whole: ?json.Value = null,
    events: std.ArrayList(json.Value) = .empty,
    fn out(r: *Reply) openai.Out {
        return .{ .ctx = r, .vt = &.{ .open = open, .event = event, .reply = replied } };
    }
    fn open(_: *anyopaque, _: usize) error{Closed}!void {}
    fn event(ctx: *anyopaque, value: ?json.Value) error{Closed}!void {
        const r: *Reply = @ptrCast(@alignCast(ctx));
        if (value) |v| r.events.append(r.a, v) catch return error.Closed;
    }
    fn replied(ctx: *anyopaque, _: u16, value: json.Value) void {
        const r: *Reply = @ptrCast(@alignCast(ctx));
        r.whole = value;
    }
};

const Env = struct {
    var pairs: []const [2][]const u8 = &.{};
    fn get(name: []const u8) ?[]const u8 {
        for (pairs) |p| if (std.mem.eql(u8, p[0], name)) return p[1];
        return null;
    }
};

/// The Flash-Next profile's runner forms with the spans `reasoning` (its family default: runner_binding.zig).
fn reasoningRunner() !api.runner.Runner {
    Env.pairs = &.{ .{ "TF_RUNNER", "1" }, .{ "TF_RUNNER_CIRCUIT", "/c.tfpc" }, .{ "TF_RUNNER_FORMS", "fence,repl,inline" }, .{ "TF_RUNNER_SPANS", "reasoning" } };
    return .{ .gpa = t.allocator, .cfg = (try api.runner.Config.parse(&Env.get, "TF_RUNNER")).?, .frontend = undefined, .backend = undefined };
}

// an inline sum and a fenced cell with its result header, each with the model's own (wrong) value after it
const answer = "Sum: 12 + 7 = 20\n```python\nprint(12 + 7)\n```\nResult:\n20\nDone.";

test "runner thinking only: a non-thinking chat gets no session, no cells and the runner-off server's bytes" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]bool{ false, true }) |stream| {
        var off: Script = .{ .tokens = &.{} };
        const plain = try chat(a, &off, false, "", answer, stream);
        var rn = try reasoningRunner();
        var on: Script = .{ .tokens = &.{}, .runner = &rn };
        const got = try chat(a, &on, false, "", answer, stream);
        try t.expectEqualStrings(answer, plain.content);
        try t.expectEqualStrings(plain.content, got.content);
        try t.expectEqualStrings(plain.reasoning, got.reasoning);
        try t.expect(!on.run_cells and !on.think_end);
        try t.expectEqual(@as(u32, 0), on.closed);
        try t.expectEqual(@as(u32, 0), on.owned);
    }
}

test "runner thinking only: a thinking chat keeps its session, owning in the think block and never in the answer" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]bool{ false, true }) |stream| {
        var rn = try reasoningRunner();
        var on: Script = .{ .tokens = &.{}, .runner = &rn };
        const got = try chat(a, &on, true, "First 12 + 7 = 20\nthen check.", answer, stream);
        try t.expect(on.run_cells and on.think_end);
        try t.expectEqualStrings("First 12 + 7 = 19\nthen check.", got.reasoning);
        try t.expectEqualStrings(answer, got.content);
        try t.expectEqual(@as(u32, 1), on.owned);
    }
}
