//! PREFIX_COMMIT's request settings (Request.run_commit): the think close, the task's text and the end token.
const std = @import("std");
const json = @import("json");
const Value = json.Value;
const api = @import("engine_api");
const turns = @import("compute_turns.zig");
const Allocator = std.mem.Allocator;

/// A request's answer-commit settings; empty (it never fires) without the knob, a think close or an EOS.
pub fn of(a: Allocator, runner: ?*api.runner.Runner, close: []const u32, msgs: Value, tools: []const Value, r: *const api.Request) Allocator.Error!@TypeOf(r.run_commit) {
    const rn = runner orelse return .{};
    if (!rn.cfg.commit or !r.run_cells or close.len == 0 or r.eos.len == 0 or r.structure != null) return .{};
    return .{ .close = close, .task = try lastUser(a, msgs), .eos = r.eos[0], .max_new = r.max_tokens, .answer_max = r.answer_max, .tools = client(tools) };
}

/// A tool the client runs (not the lane's built-in, whose hold at the commit's EOS finds no call and ends the reply).
fn client(tools: []const Value) bool {
    for (tools) |t| if (!turns.isBuiltIn(t)) return true;
    return false;
}

/// The last user message's text: its content string, or its text parts joined ("": none).
fn lastUser(a: Allocator, msgs: Value) Allocator.Error![]const u8 {
    if (msgs != .array) return "";
    var i = msgs.array.len;
    const m = while (i > 0) {
        i -= 1;
        const role = msgs.array[i].get("role") orelse continue;
        if (role == .string and std.mem.eql(u8, role.string, "user")) break msgs.array[i];
    } else return "";
    const c = m.get("content") orelse return "";
    if (c == .string) return c.string;
    if (c != .array) return "";
    var buf: std.ArrayList(u8) = .empty;
    for (c.array) |part| {
        if (part != .object) continue;
        const kind = part.get("type") orelse continue;
        if (kind != .string or !std.mem.eql(u8, kind.string, "text")) continue;
        const t = part.get("text") orelse continue;
        if (t == .string) try buf.appendSlice(a, t.string);
    }
    return buf.items;
}

test "runner commit ask: the knob, the last user message's text and the settings that keep it off" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const first = try json.newObject(a);
    try first.put(a, "role", .{ .string = "user" });
    try first.put(a, "content", .{ .string = "What is 1 + 2?" });
    const reply = try json.newObject(a);
    try reply.put(a, "role", .{ .string = "assistant" });
    try reply.put(a, "content", .{ .string = "3" });
    const part = try json.newObject(a);
    try part.put(a, "type", .{ .string = "text" });
    try part.put(a, "text", .{ .string = "What is 47 * 83?" });
    var parts = [_]Value{.{ .object = part }};
    const last = try json.newObject(a);
    try last.put(a, "role", .{ .string = "user" });
    try last.put(a, "content", .{ .array = &parts });
    var chat = [_]Value{ .{ .object = first }, .{ .object = reply }, .{ .object = last } };
    try std.testing.expectEqualStrings("What is 47 * 83?", try lastUser(a, .{ .array = &chat }));
    try std.testing.expectEqualStrings("What is 1 + 2?", try lastUser(a, .{ .array = chat[0..2] }));
    try std.testing.expectEqualStrings("", try lastUser(a, .{ .array = chat[1..2] }));
    var rn: api.runner.Runner = undefined;
    rn.cfg = .{ .commit = true };
    const close = [_]u32{ 10, 11, 12 };
    const eos = [_]u32{7};
    var r: api.Request = .{ .prompt = &.{}, .max_tokens = 100, .eos = &eos, .run_cells = true };
    const on = try of(a, &rn, &close, .{ .array = &chat }, &.{}, &r);
    try std.testing.expectEqualSlices(u32, &close, on.close);
    try std.testing.expectEqual(@as(?u32, 7), on.eos);
    try std.testing.expectEqual(@as(u32, 100), on.max_new);
    try std.testing.expectEqualStrings("What is 47 * 83?", on.task);
    const fun = try json.newObject(a);
    try fun.put(a, "name", .{ .string = turns.name });
    try fun.put(a, "description", .{ .string = turns.description });
    const builtin = try json.newObject(a);
    try builtin.put(a, "function", .{ .object = fun });
    const theirs = try json.newObject(a);
    try theirs.put(a, "name", .{ .string = turns.name }); // their own `python`: another description
    const function = try json.newObject(a);
    try function.put(a, "function", .{ .object = theirs });
    var only = [_]Value{.{ .object = builtin }};
    var both = [_]Value{ .{ .object = builtin }, .{ .object = function } };
    // the built-in tool and its lane's hold keep it on; a client's tool turns it off
    r.hold_eos = true;
    const lane = try of(a, &rn, &close, .{ .array = &chat }, &only, &r);
    try std.testing.expect(lane.close.len > 0 and !lane.tools);
    try std.testing.expect((try of(a, &rn, &close, .{ .array = &chat }, &both, &r)).tools);
    r.hold_eos = false;
    // each of these leaves it empty: it never fires
    try std.testing.expectEqual(@as(usize, 0), (try of(a, null, &close, .{ .array = &chat }, &.{}, &r)).close.len);
    try std.testing.expectEqual(@as(usize, 0), (try of(a, &rn, &.{}, .{ .array = &chat }, &.{}, &r)).close.len);
    r.eos = &.{};
    try std.testing.expectEqual(@as(usize, 0), (try of(a, &rn, &close, .{ .array = &chat }, &.{}, &r)).close.len);
    r.eos = &eos;
    r.run_cells = false;
    try std.testing.expectEqual(@as(usize, 0), (try of(a, &rn, &close, .{ .array = &chat }, &.{}, &r)).close.len);
    r.run_cells = true;
    rn.cfg.commit = false;
    try std.testing.expectEqual(@as(usize, 0), (try of(a, &rn, &close, .{ .array = &chat }, &.{}, &r)).close.len);
}
