const std = @import("std");
const ast = @import("runner_frontend_collections_ast.zig");
const v = @import("runner_frontend_collections_value.zig");
const ops = @import("runner_frontend_collections_ops.zig");
const Value = v.Value;
const Heap = v.Heap;
const Error = ast.Error;
fn is(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}
fn argc(args: []const Value, min: usize, max: usize) Error!void {
    if (args.len < min or args.len > max) return error.Unsupported;
}
pub fn call(h: *Heap, object: Value, name: []const u8, args: []const Value, keywords: []const []const u8) Error!Value {
    try h.charge(24);
    if (args.len != keywords.len) return error.Unsupported;
    for (keywords) |key| if (key.len != 0) return error.Unsupported;
    if (is(name, "append")) {
        try argc(args, 1, 1);
        if (object.kind != .list) return error.Unsupported;
        try h.append(object, args[0]);
        return .{};
    }
    if (is(name, "pop")) {
        try argc(args, 0, 1);
        if (object.kind != .list) return error.Unsupported;
        const o = object.obj.?;
        const i = try v.normalized(if (args.len == 0) -1 else try args[0].numeric(), o.items.items.len);
        try h.charge((o.items.items.len - i) * 8);
        return o.items.orderedRemove(i);
    }
    if (is(name, "reverse")) {
        try argc(args, 0, 0);
        if (object.kind != .list) return error.Unsupported;
        std.mem.reverse(Value, object.obj.?.items.items);
        try h.charge(object.obj.?.items.items.len * 8);
        return .{};
    }
    if (is(name, "copy")) {
        try argc(args, 0, 0);
        if (object.kind != .list) return error.Unsupported;
        return ops.collect(h, object, false);
    }
    if (is(name, "get") or is(name, "setdefault")) {
        try argc(args, 1, 2);
        if (object.kind != .dict) return error.Unsupported;
        if (try h.contains(object, args[0])) return h.get(object, args[0]);
        const fallback = if (args.len == 2) args[1] else Value{};
        if (is(name, "setdefault")) try h.dictSet(object, args[0], fallback);
        return fallback;
    }
    if (is(name, "keys") or is(name, "items") or is(name, "values")) {
        try argc(args, 0, 0);
        if (object.kind != .dict) return error.Unsupported;
        return .{ .kind = if (is(name, "keys")) .keys else if (is(name, "items")) .items else .values, .obj = object.obj };
    }
    if (is(name, "add")) {
        try argc(args, 1, 1);
        try h.setAdd(object, args[0]);
        return .{};
    }
    if (is(name, "remove") or is(name, "discard")) {
        try argc(args, 1, 1);
        if (object.kind != .set or !v.hashable(args[0])) return error.Unsupported;
        const data = object.obj.?.items.items;
        for (data, 0..) |x, i| {
            try h.charge(12);
            if (try h.eq(x, args[0])) {
                _ = object.obj.?.items.orderedRemove(i);
                return .{};
            }
        }
        if (is(name, "remove")) return error.Unsupported;
        return .{};
    }
    if (is(name, "find") or is(name, "rfind")) {
        try argc(args, 1, 3);
        if (object.kind != .string or args[0].kind != .string) return error.Unsupported;
        const s = object.obj.?.text;
        const needle = args[0].obj.?.text;
        const n: i64 = @intCast(s.len);
        var lo: i64 = if (args.len >= 2) try args[1].numeric() else 0;
        var hi: i64 = if (args.len >= 3) try args[2].numeric() else n;
        if (lo < 0) lo += n;
        lo = @max(lo, 0);
        if (hi < 0) hi += n;
        hi = @min(n, @max(hi, 0));
        var found: i64 = -1;
        while (lo + @as(i64, @intCast(needle.len)) <= hi) : (lo += 1) {
            try h.charge(12 + needle.len * 4);
            const i: usize = @intCast(lo);
            if (std.mem.eql(u8, s[i..][0..needle.len], needle)) {
                found = lo;
                if (is(name, "find")) break;
            }
        }
        return Value.int(found);
    }
    if (is(name, "join")) {
        try argc(args, 1, 1);
        if (object.kind != .string) return error.Unsupported;
        const seq = try ops.collect(h, args[0], false);
        var text: std.ArrayList(u8) = .empty;
        for (seq.obj.?.items.items, 0..) |x, i| {
            if (x.kind != .string) return error.Unsupported;
            if (i != 0) try text.appendSlice(h.a, object.obj.?.text);
            try text.appendSlice(h.a, x.obj.?.text);
        }
        return h.string(text.items);
    }
    if (is(name, "split")) {
        try argc(args, 1, 1);
        if (object.kind != .string or args[0].kind != .string or args[0].obj.?.text.len == 0) return error.Unsupported;
        const out = try h.new(.list);
        var iterator = std.mem.splitSequence(u8, object.obj.?.text, args[0].obj.?.text);
        while (iterator.next()) |part| {
            const item = try h.string(part);
            try h.append(out, item);
        }
        return out;
    }
    return error.Unsupported;
}
