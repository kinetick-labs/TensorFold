//! PREFIX_GUIDE: RUNNER-GUIDE.md (T5) in a chat's system message, so the model asks for exact results early.
const std = @import("std");
const json = @import("json");
const Value = json.Value;
const TraceAsk = @import("engine_api").runner.TraceAsk;
const validTraceFormat = @import("engine_api").runner.validTraceFormat;
const maxTraceRepeat = @import("engine_api").runner.maxTraceRepeat;

/// RUNNER-GUIDE.md without its final newline (76 tokens on DeepSeek V4.1).
pub const text = "Exact computation is available. Write the full expression inline as `<expr> =`, or a Python fenced cell with `print(<expr>)` followed by a `Result:` header. The exact value will follow. Request the whole computation early; wait for the result instead of working digits, loops, carries or long division by hand. Unsupported requests leave your own continuation.";

/// The trust guide AB3's tool-full arm sent as each request's "runner_guide" (512 bytes); the compute default's guide.
pub const trust = "Think and compute in small chunks. As soon as a sub-computation is needed, write its complete expression as `<expr> =` and continue from the returned value. Do not finish a manual trace before requesting it. For a short integer program, use a Python fenced cell with print, then `Result:`. Unsupported cells leave your continuation. Verify the setup, program semantics and non-computational reasoning. Engine values after `=` or `Result:` are computed exactly; use as given; do not re-derive or re-check by hand.";

/// The most a request's own "runner_guide" may hold.
pub const max_request = 512;

/// The server's default guide, read once at boot from PREFIX_GUIDE: "1" is `text`, any other value a file.
pub const Set = struct {
    default: []const u8 = "", // "": chats get no guide unless they ask
    max_request_bytes: usize = max_request,

    pub fn load(a: std.mem.Allocator, io: std.Io, knob: []const u8) !Set {
        if (knob.len == 0) return .{};
        if (std.mem.eql(u8, knob, "1")) return .{ .default = text };
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, knob, a, .limited(64 << 10));
        return .{ .default = trim(bytes) };
    }

    pub fn refusal(s: Set, a: std.mem.Allocator) std.mem.Allocator.Error![]const u8 {
        return std.fmt.allocPrint(a, "runner_guide must be a string of at most {d} bytes", .{s.max_request_bytes});
    }

    pub fn describe(s: Set) void {
        @import("log.zig").line("runner guide: default {d} bytes, request limit {d} bytes", .{ s.default.len, s.max_request_bytes });
    }
};

/// Trailing whitespace off, so a file's final newline and a request's text read alike.
fn trim(t: []const u8) []const u8 {
    return std.mem.trimEnd(u8, t, " \t\r\n");
}

pub const Applied = struct {
    messages: Value,
    guide: ?[]const u8, // the text injected ("" none); null when the runner is off
};

/// `messages` with `asked` (a request's guide) or the default; "" adds nothing, nor does a null `set` (runner off).
pub fn apply(a: std.mem.Allocator, set: ?*const Set, asked: ?Value, messages: Value) error{ BadGuide, OutOfMemory }!Applied {
    const s = set orelse return .{ .messages = messages, .guide = null };
    const chosen = if (asked) |v| switch (v) {
        .null => s.default,
        .string => |t| if (t.len <= s.max_request_bytes) trim(t) else return error.BadGuide,
        else => return error.BadGuide,
    } else s.default;
    return .{ .messages = if (chosen.len > 0) try with(a, messages, chosen) else messages, .guide = chosen };
}

/// `messages` with `guide` after the first system message's text, or in a new system message before the rest.
pub fn with(a: std.mem.Allocator, messages: Value, guide: []const u8) std.mem.Allocator.Error!Value {
    if (messages != .array) return messages;
    const out = try a.alloc(Value, messages.array.len + 1);
    if (messages.array.len > 0) if (messages.array[0].strField("role")) |role| if (std.mem.eql(u8, role, "system")) {
        if (messages.array[0].get("content")) |content| if (content == .string) {
            const m = try json.newObject(a);
            var it = messages.array[0].object.iterator();
            while (it.next()) |e| try m.put(a, e.key_ptr.*, e.value_ptr.*);
            try m.put(a, "content", .{ .string = try std.mem.concat(a, u8, &.{ content.string, "\n\n", guide }) });
            out[0] = .{ .object = m };
            @memcpy(out[1..messages.array.len], messages.array[1..]);
            return .{ .array = out[0..messages.array.len] };
        };
    };
    const m = try json.newObject(a);
    try m.put(a, "role", .{ .string = "system" });
    try m.put(a, "content", .{ .string = guide });
    out[0] = .{ .object = m };
    @memcpy(out[1..], messages.array);
    return .{ .array = out };
}

/// A chat's "runner_trace" (false: none) and "runner_trace_format", "_check", "_repeat"; null: as booted.
pub fn traceAsk(fields: Value) error{BadTrace}!TraceAsk {
    var ask: TraceAsk = .{};
    if (fields.get("runner_trace")) |v| switch (v) {
        .null => {},
        .bool => |on| ask.off = !on,
        else => return error.BadTrace,
    };
    if (fields.get("runner_trace_format")) |v| switch (v) {
        .null => {},
        .string => |t| ask.format = if (validTraceFormat(t)) t else return error.BadTrace,
        else => return error.BadTrace,
    };
    if (fields.get("runner_trace_check")) |v| switch (v) {
        .null => {},
        .bool => |on| ask.check = on,
        else => return error.BadTrace,
    };
    if (fields.get("runner_trace_repeat")) |v| switch (v) {
        .null => {},
        .int => ask.repeat = std.math.cast(u8, v.int64() orelse return error.BadTrace) orelse return error.BadTrace,
        else => return error.BadTrace,
    };
    if (ask.repeat > maxTraceRepeat) return error.BadTrace;
    return ask;
}

test "runner guide: a chat's runner_trace and runner_trace_format, and their refusals" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const f = try json.newObject(a);
    try std.testing.expectEqualDeep(TraceAsk{}, try traceAsk(.{ .object = f }));
    try f.put(a, "runner_trace", .{ .bool = false });
    try f.put(a, "runner_trace_format", .{ .string = "[exact, computed by the engine: {}]" });
    const got = try traceAsk(.{ .object = f });
    try std.testing.expect(got.off);
    try std.testing.expectEqualStrings("[exact, computed by the engine: {}]", got.format);
    try f.put(a, "runner_trace", .{ .bool = true });
    try f.put(a, "runner_trace_format", .null);
    try std.testing.expectEqualDeep(TraceAsk{}, try traceAsk(.{ .object = f }));
    for ([_]Value{ .{ .string = "[exact]" }, .{ .string = "[{}]\n{}" }, .{ .string = "a\n{}" }, .{ .bool = true } }) |bad| {
        try f.put(a, "runner_trace_format", bad);
        try std.testing.expectError(error.BadTrace, traceAsk(.{ .object = f }));
    }
    try f.put(a, "runner_trace_format", .null);
    try f.put(a, "runner_trace", .{ .string = "off" });
    try std.testing.expectError(error.BadTrace, traceAsk(.{ .object = f }));
    try f.put(a, "runner_trace", .null);
    try f.put(a, "runner_trace_check", .{ .bool = true });
    try std.testing.expect((try traceAsk(.{ .object = f })).check);
    try f.put(a, "runner_trace_check", .{ .string = "true" });
    try std.testing.expectError(error.BadTrace, traceAsk(.{ .object = f }));
    try f.put(a, "runner_trace_check", .null);
    try f.put(a, "runner_trace_repeat", .{ .int = "1" });
    try std.testing.expectEqual(@as(u8, 1), (try traceAsk(.{ .object = f })).repeat);
    for ([_]Value{ .{ .int = "4" }, .{ .int = "-1" }, .{ .int = "300" }, .{ .string = "1" }, .{ .bool = true } }) |bad| {
        try f.put(a, "runner_trace_repeat", bad);
        try std.testing.expectError(error.BadTrace, traceAsk(.{ .object = f }));
    }
}

test "runner guide: T5's RUNNER-GUIDE.md bytes; joins a system message or leads the chat" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(text ++ "\n", &digest, .{});
    try std.testing.expectEqualStrings("b99224b5ebf6f7e9323bfc011200073211e1fc9381968e067f1a336893a8a7bb", &std.fmt.bytesToHex(digest, .lower));
    const user = try json.newObject(a);
    try user.put(a, "role", .{ .string = "user" });
    try user.put(a, "content", .{ .string = "What is 47*83?" });
    var one = [_]Value{.{ .object = user }};
    const led = try with(a, .{ .array = &one }, text);
    try std.testing.expectEqual(@as(usize, 2), led.array.len);
    try std.testing.expectEqualStrings("system", led.array[0].strField("role").?);
    try std.testing.expectEqualStrings(text, led.array[0].get("content").?.string);
    try std.testing.expectEqualStrings("user", led.array[1].strField("role").?);
    const sys = try json.newObject(a);
    try sys.put(a, "role", .{ .string = "system" });
    try sys.put(a, "content", .{ .string = "Be brief." });
    var two = [_]Value{ .{ .object = sys }, .{ .object = user } };
    const joined = try with(a, .{ .array = &two }, text);
    try std.testing.expectEqual(@as(usize, 2), joined.array.len);
    try std.testing.expectEqualStrings("Be brief.\n\n" ++ text, joined.array[0].get("content").?.string);
    try std.testing.expectEqualStrings("Be brief.", two[0].get("content").?.string); // the request is not changed
}

test "runner guide: the trust guide is AB3's tool-full text, byte for byte" {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(trust, &digest, .{});
    try std.testing.expectEqualStrings("53ed328f72e2e79076aaf9e7ce2a1da1b11f3cae7ed22b76798b15c6e59add92", &std.fmt.bytesToHex(digest, .lower));
    try std.testing.expectEqual(max_request, trust.len);
    try std.testing.expectEqualStrings(trust, trim(trust)); // a request sending it injected these same bytes
}

test "runner guide: boot default, a request's override, an empty one and the runner off" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "RUNNER-GUIDE.md", .data = text ++ "\n" });
    try tmp.dir.writeFile(io, .{ .sub_path = "terse.md", .data = "Ask for exact values.\n" });
    try tmp.dir.writeFile(io, .{ .sub_path = "empty.md", .data = "\n" });
    const root = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    const builtin = try Set.load(a, io, "1");
    try std.testing.expectEqualStrings(text, builtin.default);
    try std.testing.expectEqualStrings(text, (try Set.load(a, io, try std.fs.path.join(a, &.{ root, "RUNNER-GUIDE.md" }))).default);
    const terse = try Set.load(a, io, try std.fs.path.join(a, &.{ root, "terse.md" }));
    const none = try Set.load(a, io, "");
    try std.testing.expectEqualStrings("", (try Set.load(a, io, try std.fs.path.join(a, &.{ root, "empty.md" }))).default);
    try std.testing.expectError(error.FileNotFound, Set.load(a, io, "/nonexistent/guide.md"));

    const user = try json.newObject(a);
    try user.put(a, "role", .{ .string = "user" });
    try user.put(a, "content", .{ .string = "What is 47*83?" });
    var one = [_]Value{.{ .object = user }};
    const chat: Value = .{ .array = &one };
    const plain = try json.stringify(a, chat, .{});
    const Case = struct { set: ?*const Set, asked: ?Value, want: []const u8, guide: ?[]const u8 };
    const with_text = try json.stringify(a, try with(a, chat, text), .{});
    const with_terse = try json.stringify(a, try with(a, chat, "Ask for exact values."), .{});
    for ([_]Case{
        .{ .set = &builtin, .asked = null, .want = with_text, .guide = text }, // the boot default
        .{ .set = &builtin, .asked = .{ .string = "Ask for exact values.\n" }, .want = with_terse, .guide = "Ask for exact values." }, // override
        .{ .set = &terse, .asked = null, .want = with_terse, .guide = "Ask for exact values." }, // the same text from a boot file
        .{ .set = &none, .asked = .{ .string = "Ask for exact values." }, .want = with_terse, .guide = "Ask for exact values." },
        .{ .set = &builtin, .asked = .{ .string = "" }, .want = plain, .guide = "" }, // empty: no guide
        .{ .set = &none, .asked = null, .want = plain, .guide = "" },
        .{ .set = null, .asked = .{ .string = "Ask for exact values." }, .want = plain, .guide = null }, // runner off: ignored
    }) |c| {
        const got = try apply(a, c.set, c.asked, chat);
        try std.testing.expectEqualStrings(c.want, try json.stringify(a, got.messages, .{}));
        if (c.guide) |g| try std.testing.expectEqualStrings(g, got.guide.?) else try std.testing.expect(got.guide == null);
    }
    const long = try a.alloc(u8, max_request + 1);
    @memset(long, 'x');
    try std.testing.expectError(error.BadGuide, apply(a, &builtin, .{ .string = long }, chat));
    _ = try apply(a, &builtin, .{ .string = long[0..max_request] }, chat);
    try std.testing.expectError(error.BadGuide, apply(a, &builtin, .{ .bool = true }, chat));
}

test "runner guide: opt-in request byte boundaries preserve injection and off behavior" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const user = try json.newObject(a);
    try user.put(a, "role", .{ .string = "user" });
    try user.put(a, "content", .{ .string = "Compute 47*83." });
    var one = [_]Value{.{ .object = user }};
    const chat: Value = .{ .array = &one };
    const bytes = try a.alloc(u8, 2049);
    @memset(bytes, 'x');
    const Case = struct { limit: usize, len: usize, accept: bool };
    for ([_]Case{
        .{ .limit = 512, .len = 512, .accept = true },
        .{ .limit = 512, .len = 586, .accept = false },
        .{ .limit = 1024, .len = 586, .accept = true },
        .{ .limit = 2048, .len = 512, .accept = true },
        .{ .limit = 2048, .len = 586, .accept = true },
        .{ .limit = 2048, .len = 2048, .accept = true },
        .{ .limit = 2048, .len = 2049, .accept = false },
    }) |c| {
        const set: Set = .{ .default = "Boot guide", .max_request_bytes = c.limit };
        try std.testing.expectEqualStrings(try std.fmt.allocPrint(a, "runner_guide must be a string of at most {d} bytes", .{c.limit}), try set.refusal(a));
        const asked: Value = .{ .string = bytes[0..c.len] };
        if (c.accept) {
            const got = try apply(a, &set, asked, chat);
            try std.testing.expectEqualStrings(bytes[0..c.len], got.guide.?);
            try std.testing.expectEqualStrings(bytes[0..c.len], got.messages.array[0].strField("content").?);
            try std.testing.expectEqualStrings("Compute 47*83.", got.messages.array[1].strField("content").?);
        } else try std.testing.expectError(error.BadGuide, apply(a, &set, asked, chat));
        const off = try apply(a, null, asked, chat);
        try std.testing.expect(off.guide == null);
        try std.testing.expectEqualStrings(try json.stringify(a, chat, .{}), try json.stringify(a, off.messages, .{}));
        try std.testing.expectEqualStrings("Boot guide", (try apply(a, &set, .null, chat)).guide.?);
        try std.testing.expectEqualStrings("", (try apply(a, &set, .{ .string = "" }, chat)).guide.?);
    }
    const capped: Set = .{ .max_request_bytes = 2048 };
    @memset(bytes, ' ');
    try std.testing.expectError(error.BadGuide, apply(a, &capped, .{ .string = bytes }, chat));
    for (0..1024) |i| {
        bytes[2 * i] = 0xc3;
        bytes[2 * i + 1] = 0xa9;
    }
    bytes[2048] = 'x';
    try std.testing.expectEqual(@as(usize, 2048), (try apply(a, &capped, .{ .string = bytes[0..2048] }, chat)).guide.?.len);
    try std.testing.expectError(error.BadGuide, apply(a, &capped, .{ .string = bytes }, chat));
}
