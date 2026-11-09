//! The runner's live records (Generation.liveCells): a stream the client leaves mid-write keeps a record per write.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const server = @import("server.zig");
const chat = @import("chat.zig");
const model_text = @import("model_text.zig");
const Cx = @import("errors.zig").Cx;
const Note = @typeInfo(@FieldType(api.Event, "cells")).pointer.child;

const gpa = std.testing.allocator;

/// One byte a token.
const Bytes = struct {
    fn text(t: *Bytes) model_text.Text {
        return .{ .ctx = t, .vtable = &.{ .encode = encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocab, .eos_ids = eos, .render = render, .template_source = template } };
    }
    fn encode(_: *anyopaque, a: std.mem.Allocator, input: []const u8, _: bool) model_text.Error![]u32 {
        const ids = try a.alloc(u32, input.len);
        for (input, ids) |byte, *id| id.* = byte;
        return ids;
    }
    fn decode(_: *anyopaque, a: std.mem.Allocator, ids: []const u32) std.mem.Allocator.Error![]u8 {
        const out = try a.alloc(u8, ids.len);
        for (out, ids) |*v, id| v.* = @intCast(id);
        return out;
    }
    fn tokenId(_: *anyopaque, _: []const u8) ?u32 {
        return null;
    }
    fn tokenString(_: *anyopaque, a: std.mem.Allocator, id: u32) ![]u8 {
        return std.fmt.allocPrint(a, "{d}", .{id});
    }
    fn vocab(_: *anyopaque) u32 {
        return 256;
    }
    fn eos(_: *anyopaque) []const u32 {
        return &.{};
    }
    fn render(_: *anyopaque, a: std.mem.Allocator, _: json.Value, _: model_text.RenderOptions, _: *[]const u8) model_text.Error!model_text.Rendered {
        return .{ .text = try a.dupe(u8, "prompt") };
    }
    fn template(_: *anyopaque) []const u8 {
        return "";
    }
};

/// What the engine gives in one round: the records as they stand, then the tokens that land them (LaneHost.send).
const Step = struct { notes: []const Note = &.{}, tokens: []const u8 };

const a0 = "{\"cell\":0,\"form\":\"inline\",\"value\":\"\",\"field\":\"content\",\"at\":9,\"ids\":2,\"landed\":0}";
const a1 = "{\"cell\":0,\"form\":\"inline\",\"value\":\" 6\",\"field\":\"content\",\"at\":9,\"ids\":2,\"landed\":2}";
const b0 = "{\"cell\":1,\"form\":\"inline\",\"value\":\"\",\"field\":\"content\",\"at\":22,\"ids\":3,\"landed\":0}";
const b1 = "{\"cell\":1,\"form\":\"inline\",\"value\":\" 2\",\"field\":\"content\",\"at\":22,\"ids\":3,\"landed\":1}";
const b2 = "{\"cell\":1,\"form\":\"inline\",\"value\":\" 2!\",\"field\":\"content\",\"at\":22,\"ids\":3,\"landed\":2}";

/// `x = 2*3 = 6. y = 4*5 = 2` then a `!` the stream holds back (stop `!!`): the client leaves inside the second write.
const steps = [_]Step{
    .{ .notes = &.{.{ .index = 0, .content = true, .end = 9, .json = a0 }}, .tokens = "x = 2*3 =" },
    .{ .notes = &.{.{ .index = 0, .content = true, .end = 11, .json = a1 }}, .tokens = " 6" },
    .{ .notes = &.{.{ .index = 1, .content = true, .end = 22, .json = b0 }}, .tokens = ". y = 4*5 =" },
    .{ .notes = &.{.{ .index = 1, .content = true, .end = 24, .json = b1 }}, .tokens = " 2" },
    .{ .notes = &.{.{ .index = 1, .content = true, .end = 25, .json = b2 }}, .tokens = "!" },
};

/// An engine that gives a round each time the request's thread checks on its client, then sees the client leave.
const Script = struct {
    records: bool,
    sink: ?api.Sink = null,
    id: api.Id = 0,
    next: usize = 0,
    fn engine(s: *Script) api.Engine {
        return .{ .ctx = s, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(_: *anyopaque) api.Info {
        return .{ .context_window = 8192 };
    }
    fn submit(ctx: *anyopaque, id: api.Id, _: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const s: *Script = @ptrCast(@alignCast(ctx));
        s.sink = sink;
        s.id = id;
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        s.round();
    }
    fn round(s: *Script) void {
        const step = steps[s.next];
        s.next += 1;
        const sink = s.sink.?;
        if (s.records) sink.event(sink.ctx, s.id, &.{ .cells = step.notes });
        var tokens: [16]u32 = undefined;
        for (step.tokens, 0..) |b, i| tokens[i] = b;
        sink.event(sink.ctx, s.id, &.{ .tokens = tokens[0..step.tokens.len] });
    }
    fn cancel(ctx: *anyopaque, _: api.Id) void {
        const s: *Script = @ptrCast(@alignCast(ctx));
        const sink = s.sink.?;
        sink.event(sink.ctx, s.id, &.{ .finished = .{ .reason = .cancelled } });
    }
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

const Gone = struct {
    s: *Script,
    checks: *usize,
    pub fn check(g: @This()) bool {
        if (g.s.sink == null) return false; // chat.prepare looks before the engine has the request
        g.checks.* += 1;
        if (g.s.next < steps.len) {
            g.s.round();
            return false;
        }
        return g.checks.* > steps.len + 1; // one more look after the last round, then the client is gone
    }
};

/// The client: the text it has, every other delta as JSON, and the newest records with any it got ahead of its bytes.
const Client = struct {
    a: std.mem.Allocator,
    text: std.ArrayList(u8) = .empty,
    plain: std.ArrayList([]const u8) = .empty,
    last: ?json.Value = null,
    sent: usize = 0,
    ahead: usize = 0,
    fn call(ctx: *anyopaque, delta: json.Value) error{Closed}!void {
        const c: *Client = @ptrCast(@alignCast(ctx));
        if (delta == .object) if (delta.get("tensorfold")) |tf| {
            const run = tf.get("runner") orelse return error.Closed;
            for ((run.get("cells") orelse return error.Closed).array) |cell| {
                const at: usize = @intCast(cell.get("at").?.int64().?);
                const len = cell.strField("value").?.len;
                if (at + len > c.text.items.len) c.ahead += 1;
            }
            c.last = run;
            c.sent += 1;
            return;
        };
        c.plain.append(c.a, json.stringify(c.a, delta, .{}) catch return error.Closed) catch return error.Closed;
        if (delta == .string) c.text.appendSlice(c.a, delta.string) catch return error.Closed;
    }
    fn sink(c: *Client) chat.Sink {
        return .{ .ctx = c, .call = call };
    }
};

fn stream(a: std.mem.Allocator, records: bool) !Client {
    var script: Script = .{ .records = records };
    var bytes: Bytes = .{};
    const srv = try server.Server.init(gpa, std.testing.io, script.engine(), bytes.text(), .{ .served_name = "test", .model_ids = &.{"test"}, .enable_thinking = false }, null);
    defer srv.deinit();
    var client: Client = .{ .a = a };
    var checks: usize = 0;
    var cx: Cx = .{ .a = a };
    const fields = try json.newObject(a);
    try fields.put(a, "stop", .{ .string = "!!" });
    const input: chat.Input = .{ .id = "chatcmpl-live", .prompt = .{ .text = "prompt" }, .max_tokens = 100, .fields = .{ .object = fields } };
    try std.testing.expectError(error.Cancelled, chat.run(srv, &cx, input, client.sink(), Gone{ .s = &script, .checks = &checks }));
    try std.testing.expectEqual(steps.len, script.next);
    return client;
}

test "runner live records: a client that leaves inside a write has a record for each write it was shown" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const on = try stream(a, true);
    try std.testing.expectEqualStrings("x = 2*3 = 6. y = 4*5 = 2", on.text.items);
    try std.testing.expectEqual(0, on.ahead); // never a record whose bytes the client did not yet have
    try std.testing.expectEqual(4, on.sent); // a0; a1; a1 b0; a1 b1 (not b2: its `!` is held back)
    const last = on.last.?;
    try std.testing.expect(last.get("live").?.bool);
    const want = try json.parse(a, "[" ++ a1 ++ "," ++ b1 ++ "]");
    try std.testing.expectEqualStrings(try json.stringify(a, want.ok, .{}), try json.stringify(a, last.get("cells").?, .{}));
    // records off: no records chunk, and every other delta byte for byte the same
    const off = try stream(a, false);
    try std.testing.expectEqual(0, off.sent);
    try std.testing.expectEqual(on.plain.items.len, off.plain.items.len);
    for (on.plain.items, off.plain.items) |x, y| try std.testing.expectEqualStrings(x, y);
}
