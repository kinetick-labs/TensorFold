const std = @import("std");
const ast = @import("runner_frontend_collections_ast.zig");
pub const Error = ast.Error;
pub const Kind = enum(i32) { integer = 0, boolean = 1, none = 2, string = 3, list = 4, tuple = 5, dict = 6, set = 7, range = 8, builtin = 9, items = 10, keys = 11, values = 12, enumerate = 13, zip = 14, map_str = 15 };
pub const Value = struct {
    kind: Kind = .none,
    n: i64 = 0,
    obj: ?*Object = null,
    pub fn int(n: i64) Error!Value {
        if (n < std.math.minInt(i32) or n > std.math.maxInt(i32)) return error.Unsupported;
        return .{ .kind = .integer, .n = n };
    }
    pub fn boolean(b: bool) Value {
        return .{ .kind = .boolean, .n = @intFromBool(b) };
    }
    pub fn numeric(v: Value) Error!i64 {
        if (v.kind != .integer and v.kind != .boolean) return error.Unsupported;
        return v.n;
    }
};
pub const Object = struct { items: std.ArrayList(Value) = .empty, text: []const u8 = "", capacity: usize = 0, cursor: usize = 0, exhausted: bool = false };
pub const Heap = struct {
    a: std.mem.Allocator,
    stats: *ast.Stats,
    output_bytes: usize = 0,
    proof_work: usize = 0,
    pub fn charge(h: *Heap, steps: usize) Error!void {
        h.stats.steps += steps;
        if (h.stats.steps > 200000) return error.Unsupported;
    }
    fn work(h: *Heap, n: usize) Error!void {
        h.proof_work += n;
        if (h.proof_work > 200000) return error.Unsupported;
    }
    pub fn eq(h: *Heap, a: Value, b: Value) Error!bool {
        const weight_cost = @min(try h.weight(a), try h.weight(b));
        try h.charge(@min(200001, 16 + weight_cost * 8));
        return equal(a, b, 0);
    }
    pub fn order(h: *Heap, a: Value, b: Value) Error!std.math.Order {
        try h.charge(@min(200001, 16 + @min(try h.weight(a), try h.weight(b)) * 12));
        return compare(a, b, 0);
    }
    fn weight(h: *Heap, v: Value) Error!usize {
        var memo: [96]Weight = undefined;
        var n: usize = 0;
        return h.weightInner(v, &memo, &n, 0);
    }
    fn weightInner(h: *Heap, v: Value, memo: *[96]Weight, count: *usize, depth: usize) Error!usize {
        try h.work(1);
        if (depth > 8) return error.Unsupported;
        if (v.kind == .string) return v.obj.?.text.len + 1;
        const o = v.obj orelse return 1;
        for (memo[0..count.*]) |old| if (old.obj == o) return old.weight;
        if (count.* == memo.len) return error.Unsupported;
        const index = count.*;
        count.* += 1;
        memo[index] = .{ .obj = o, .weight = 200001 };
        var total: usize = 1;
        for (o.items.items) |x| total = @min(200001, total + try h.weightInner(x, memo, count, depth + 1));
        if (v.kind == .dict or v.kind == .set) total = @min(200001, total * @max(1, try length(v)));
        memo[index].weight = total;
        return total;
    }
    fn reaches(h: *Heap, value: Value, target: *Object) Error!bool {
        var seen: [96]*Object = undefined;
        var count: usize = 0;
        return h.reachInner(value, target, &seen, &count, 0);
    }
    fn reachInner(h: *Heap, value: Value, target: *Object, seen: *[96]*Object, count: *usize, depth: usize) Error!bool {
        try h.work(1);
        if (value.obj == null or value.kind == .string) return false;
        const o = value.obj.?;
        if (o == target) return true;
        for (seen[0..count.*]) |old| if (old == o) return false;
        if (depth > 8 or count.* == seen.len) return error.Unsupported;
        seen[count.*] = o;
        count.* += 1;
        for (o.items.items) |x| if (try h.reachInner(x, target, seen, count, depth + 1)) return true;
        return false;
    }
    pub fn new(h: *Heap, kind: Kind) Error!Value {
        if (h.stats.objects == 96) return error.Unsupported;
        h.stats.objects += 1;
        try h.charge(24);
        const object = try h.a.create(Object);
        object.* = .{};
        return .{ .kind = kind, .obj = object };
    }
    pub fn string(h: *Heap, text: []const u8) Error!Value {
        if (text.len > 128 or h.stats.chars + text.len > 2048) return error.Unsupported;
        for (text) |c| if (c >= 128) return error.Unsupported;
        const v = try h.new(.string);
        v.obj.?.text = try h.a.dupe(u8, text);
        h.stats.chars += text.len;
        try h.charge(text.len * 4);
        return v;
    }
    pub fn reserve(h: *Heap, v: Value, wanted: usize) Error!void {
        const o = v.obj orelse return error.Unsupported;
        const limit: usize = if (v.kind == .dict) 128 else if (v.kind == .set) 64 else 128;
        if (wanted > limit) return error.Unsupported;
        const width: usize = if (v.kind == .dict) 2 else 1;
        const units = (wanted + width - 1) / width;
        if (units <= o.capacity) return;
        var cap: usize = if (o.capacity == 0) 4 else o.capacity;
        while (cap < units) cap *= 2;
        if (h.stats.cells + cap * width > 1536) return error.Unsupported;
        h.stats.cells += cap * width;
        try h.charge(16 + o.items.items.len * 4);
        try o.items.ensureTotalCapacity(h.a, cap * width);
        o.capacity = cap;
    }
    pub fn append(h: *Heap, v: Value, item: Value) Error!void {
        if (v.kind != .list and v.kind != .tuple and v.kind != .range and v.kind != .enumerate and v.kind != .zip and v.kind != .map_str)
            return error.Unsupported;
        if (try h.reaches(item, v.obj.?)) return error.Unsupported;
        try h.reserve(v, v.obj.?.items.items.len + 1);
        try v.obj.?.items.append(h.a, item);
        try h.charge(12);
    }
    pub fn dictSet(h: *Heap, v: Value, key: Value, item: Value) Error!void {
        if (v.kind != .dict or !hashable(key) or try h.reaches(item, v.obj.?)) return error.Unsupported;
        const o = v.obj.?;
        var i: usize = 0;
        while (i < o.items.items.len) : (i += 2) {
            try h.charge(12);
            if (try h.eq(o.items.items[i], key)) {
                o.items.items[i + 1] = item;
                return;
            }
        }
        try h.reserve(v, o.items.items.len + 2);
        try o.items.append(h.a, key);
        try o.items.append(h.a, item);
    }
    pub fn setAdd(h: *Heap, v: Value, key: Value) Error!void {
        if (v.kind != .set or !hashable(key)) return error.Unsupported;
        if (try h.contains(v, key)) return;
        try h.reserve(v, v.obj.?.items.items.len + 1);
        try v.obj.?.items.append(h.a, key);
    }
    pub fn contains(h: *Heap, v: Value, key: Value) Error!bool {
        if (v.kind == .string) {
            if (key.kind != .string) return error.Unsupported;
            try h.charge(v.obj.?.text.len * (key.obj.?.text.len + 1) * 4);
            return std.mem.indexOf(u8, v.obj.?.text, key.obj.?.text) != null;
        }
        if (v.kind != .list and v.kind != .tuple and v.kind != .range and v.kind != .dict and v.kind != .set and v.kind != .keys)
            return error.Unsupported;
        if ((v.kind == .dict or v.kind == .set or v.kind == .keys) and !hashable(key)) return error.Unsupported;
        const stride: usize = if (v.kind == .dict or v.kind == .keys) 2 else 1;
        var i: usize = 0;
        while (i < v.obj.?.items.items.len) : (i += stride) {
            try h.charge(12);
            if (try h.eq(v.obj.?.items.items[i], key)) return true;
        }
        return false;
    }
    pub fn get(h: *Heap, v: Value, key: Value) Error!Value {
        try h.charge(16);
        if (v.kind == .dict) {
            if (!hashable(key)) return error.Unsupported;
            var i: usize = 0;
            while (i < v.obj.?.items.items.len) : (i += 2) {
                try h.charge(12);
                if (try h.eq(v.obj.?.items.items[i], key)) return v.obj.?.items.items[i + 1];
            }
            return error.Unsupported;
        }
        const index = try normalized(try key.numeric(), try length(v));
        if (v.kind == .string) return h.string(v.obj.?.text[index .. index + 1]);
        if (v.kind != .list and v.kind != .tuple and v.kind != .range) return error.Unsupported;
        return v.obj.?.items.items[index];
    }
    pub fn set(h: *Heap, v: Value, key: Value, item: Value) Error!void {
        if (v.kind == .dict) return h.dictSet(v, key, item);
        if (v.kind != .list or try h.reaches(item, v.obj.?)) return error.Unsupported;
        const i = try normalized(try key.numeric(), try length(v));
        v.obj.?.items.items[i] = item;
        try h.charge(16);
    }
    pub fn output(h: *Heap, v: Value, repr: bool, depth: usize) Error!void {
        if (depth > 8) return error.Unsupported;
        var bytes: usize = 0;
        switch (v.kind) {
            .integer => {
                var b: [16]u8 = undefined;
                bytes = (std.fmt.bufPrint(&b, "{d}", .{v.n}) catch return error.Unsupported).len;
            },
            .boolean => bytes = if (v.n != 0) 4 else 5,
            .none => bytes = 4,
            .string => {
                bytes = v.obj.?.text.len;
                if (repr) {
                    bytes += 2;
                    for (v.obj.?.text) |c| if (c < 32 or c == 127 or c == '\\' or c == '\'' or c == '"') {
                        bytes += 3;
                    };
                }
            },
            .list, .tuple, .dict => {
                const items = v.obj.?.items.items;
                bytes = 2 + items.len * 2;
                for (items) |item| try h.output(item, true, depth + 1);
            },
            else => return error.Unsupported,
        }
        h.output_bytes += bytes;
        if (h.output_bytes > 8192) return error.Unsupported;
        try h.charge(bytes * 8);
    }
};
pub fn hashable(v: Value) bool {
    return v.kind == .integer or v.kind == .boolean or v.kind == .string;
}
pub fn truth(v: Value) Error!bool {
    return switch (v.kind) {
        .none => false,
        .integer, .boolean => v.n != 0,
        .string => v.obj.?.text.len != 0,
        .list, .tuple, .dict, .set, .range, .items, .keys, .values => (try length(v)) != 0,
        .enumerate, .zip, .map_str, .builtin => true,
    };
}
pub fn length(v: Value) Error!usize {
    return switch (v.kind) {
        .string => v.obj.?.text.len,
        .list, .tuple, .set, .range => v.obj.?.items.items.len,
        .dict, .items, .keys, .values => v.obj.?.items.items.len / 2,
        else => error.Unsupported,
    };
}
pub fn normalized(i: i64, len: usize) Error!usize {
    const at = if (i < 0) i + @as(i64, @intCast(len)) else i;
    if (at < 0 or at >= len) return error.Unsupported;
    return @intCast(at);
}
const Weight = struct { obj: *Object, weight: usize };
pub fn equal(a: Value, b: Value, depth: usize) Error!bool {
    if (depth > 8) return error.Unsupported;
    if (@backingInt(a.kind) >= 10 or @backingInt(b.kind) >= 10) return error.Unsupported;
    if ((a.kind == .integer or a.kind == .boolean) and (b.kind == .integer or b.kind == .boolean)) return a.n == b.n;
    if (a.kind != b.kind) return false;
    return switch (a.kind) {
        .none => true,
        .integer, .boolean, .builtin => a.n == b.n,
        .string => std.mem.eql(u8, a.obj.?.text, b.obj.?.text),
        .list, .tuple, .range => blk: {
            const x = a.obj.?.items.items;
            const y = b.obj.?.items.items;
            if (x.len != y.len) break :blk false;
            for (x, y) |l, r| if (!try equal(l, r, depth + 1)) break :blk false;
            break :blk true;
        },
        .dict, .set => blk: {
            const x = a.obj.?.items.items;
            const y = b.obj.?.items.items;
            if (x.len != y.len) break :blk false;
            const stride: usize = if (a.kind == .dict) 2 else 1;
            var i: usize = 0;
            while (i < x.len) : (i += stride) {
                var j: usize = 0;
                while (j < y.len and !try equal(x[i], y[j], depth + 1)) : (j += stride) {}
                if (j == y.len or (stride == 2 and !try equal(x[i + 1], y[j + 1], depth + 1))) break :blk false;
            }
            break :blk true;
        },
        else => error.Unsupported,
    };
}
pub fn compare(a: Value, b: Value, depth: usize) Error!std.math.Order {
    if (depth > 8) return error.Unsupported;
    if ((a.kind == .integer or a.kind == .boolean) and (b.kind == .integer or b.kind == .boolean)) return std.math.order(a.n, b.n);
    if (a.kind != b.kind) return error.Unsupported;
    if (a.kind == .string) return std.mem.order(u8, a.obj.?.text, b.obj.?.text);
    if (a.kind != .list and a.kind != .tuple) return error.Unsupported;
    const x = a.obj.?.items.items;
    const y = b.obj.?.items.items;
    for (x[0..@min(x.len, y.len)], y[0..@min(x.len, y.len)]) |l, r| {
        if (!try equal(l, r, depth + 1)) return compare(l, r, depth + 1);
    }
    return std.math.order(x.len, y.len);
}
