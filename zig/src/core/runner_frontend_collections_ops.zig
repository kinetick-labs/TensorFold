const std = @import("std");
const ast = @import("runner_frontend_collections_ast.zig");
const val = @import("runner_frontend_collections_value.zig");
const Value = val.Value;
const Heap = val.Heap;
const Error = ast.Error;

pub fn unary(h: *Heap, tag: ast.Tag, v: Value) Error!Value {
    try h.charge(8);
    if (tag == .logical_not) return Value.boolean(!try val.truth(v));
    const n = try v.numeric();
    return Value.int(if (tag == .neg) -n else if (tag == .pos) n else return error.Unsupported);
}
pub fn binary(h: *Heap, tag: ast.Tag, a: Value, b: Value) Error!Value {
    try h.charge(16);
    if (tag == .eq or tag == .ne) {
        const same = try h.eq(a, b);
        return Value.boolean(if (tag == .eq) same else !same);
    }
    if (tag == .is_identity or tag == .is_not) {
        if (a.kind != .none and b.kind != .none and !((a.kind == .list or a.kind == .dict or a.kind == .set) and
            (b.kind == .list or b.kind == .dict or b.kind == .set))) return error.Unsupported;
        const same = a.kind == b.kind and (a.kind == .none or a.obj == b.obj);
        return Value.boolean(if (tag == .is_identity) same else !same);
    }
    if (tag == .contains or tag == .not_contains) {
        const yes = try h.contains(b, a);
        return Value.boolean(if (tag == .contains) yes else !yes);
    }
    if (tag == .lt or tag == .le or tag == .gt or tag == .ge) {
        const order = try h.order(a, b);
        return Value.boolean(switch (tag) {
            .lt => order == .lt,
            .le => order != .gt,
            .gt => order == .gt,
            .ge => order != .lt,
            else => unreachable,
        });
    }
    if (tag == .add and a.kind == .string and b.kind == .string) {
        const text = try std.mem.concat(h.a, u8, &.{ a.obj.?.text, b.obj.?.text });
        return h.string(text);
    }
    if (tag == .add and a.kind == b.kind and (a.kind == .list or a.kind == .tuple)) {
        const out = try h.new(a.kind);
        for (a.obj.?.items.items) |x| try h.append(out, x);
        for (b.obj.?.items.items) |x| try h.append(out, x);
        return out;
    }
    if (tag == .mul and (a.kind == .string or a.kind == .list or a.kind == .tuple)) return repeat(h, a, try b.numeric());
    if (tag == .mul and (b.kind == .string or b.kind == .list or b.kind == .tuple)) return repeat(h, b, try a.numeric());
    const x = try a.numeric();
    const y = try b.numeric();
    return Value.int(switch (tag) {
        .add => x + y,
        .sub => x - y,
        .mul => x * y,
        .floor => if (y == 0 or (x == std.math.minInt(i32) and y == -1)) return error.Unsupported else @divFloor(x, y),
        .modulo => if (y == 0 or (x == std.math.minInt(i32) and y == -1)) return error.Unsupported else @mod(x, y),
        else => return error.Unsupported,
    });
}
pub fn repeat(h: *Heap, v: Value, n: i64) Error!Value {
    const count: usize = @intCast(@max(0, n));
    const len = try val.length(v);
    if (count > 128 or len * count > 128) return error.Unsupported;
    if (v.kind == .string) {
        const text = try h.a.alloc(u8, len * count);
        for (0..count) |i| @memcpy(text[i * len ..][0..len], v.obj.?.text);
        return h.string(text);
    }
    const out = try h.new(v.kind);
    for (0..count) |_| for (v.obj.?.items.items) |x| {
        try h.append(out, x);
    };
    return out;
}
pub fn inplace(h: *Heap, tag: ast.Tag, left: Value, right: Value) Error!Value {
    if (left.kind != .list or (tag != .add and tag != .mul)) return binary(h, tag, left, right);
    if (tag == .add) {
        if (right.kind != .list and right.kind != .tuple) return error.Unsupported;
        const items = try collect(h, right, false);
        const copy = try h.a.dupe(Value, items.obj.?.items.items);
        for (copy) |v| try h.append(left, v);
    } else {
        const n = try right.numeric();
        const original = try h.a.dupe(Value, left.obj.?.items.items);
        if (n > 128 or original.len * @as(usize, @intCast(@max(n, 0))) > 128) return error.Unsupported;
        if (n <= 0) left.obj.?.items.clearRetainingCapacity() else {
            var k: i64 = 1;
            while (k < n) : (k += 1) for (original) |v| {
                try h.append(left, v);
            };
        }
    }
    return left;
}
pub fn slice(h: *Heap, v: Value, lo: Value, hi: Value, step_value: Value) Error!Value {
    if (v.kind != .string and v.kind != .list and v.kind != .tuple) return error.Unsupported;
    const len: i64 = @intCast(try val.length(v));
    const step = if (step_value.kind == .none) 1 else try step_value.numeric();
    if (step == 0) return error.Unsupported;
    const lower: i64 = if (step > 0) 0 else -1;
    const upper: i64 = if (step > 0) len else len - 1;
    const start = if (lo.kind == .none) (if (step > 0) @as(i64, 0) else len - 1) else @min(upper, @max(lower, adjust(try lo.numeric(), len)));
    const stop = if (hi.kind == .none) (if (step > 0) len else @as(i64, -1)) else @min(upper, @max(lower, adjust(try hi.numeric(), len)));
    var bytes: [128]u8 = undefined;
    var n: usize = 0;
    const out = if (v.kind == .string) Value{} else try h.new(v.kind);
    var i = start;
    while (if (step > 0) i < stop else i > stop) : (i += step) {
        try h.charge(12);
        if (v.kind == .string) {
            bytes[n] = v.obj.?.text[@intCast(i)];
            n += 1;
        } else try h.append(out, v.obj.?.items.items[@intCast(i)]);
    }
    return if (v.kind == .string) h.string(bytes[0..n]) else out;
}
fn adjust(i: i64, len: i64) i64 {
    return if (i < 0) i + len else i;
}
pub fn indexable(v: Value) bool {
    return v.kind == .string or v.kind == .list or v.kind == .tuple or v.kind == .dict or v.kind == .range or
        v.kind == .items or v.kind == .keys or v.kind == .values;
}
pub fn at(h: *Heap, v: Value, i: usize) Error!Value {
    if (i >= try val.length(v)) return error.Unsupported;
    try h.charge(8);
    if (v.kind == .string) return h.string(v.obj.?.text[i .. i + 1]);
    if (v.kind == .dict or v.kind == .keys) return v.obj.?.items.items[2 * i];
    if (v.kind == .values) return v.obj.?.items.items[2 * i + 1];
    if (v.kind == .items) {
        const out = try h.new(.tuple);
        try h.append(out, v.obj.?.items.items[2 * i]);
        try h.append(out, v.obj.?.items.items[2 * i + 1]);
        return out;
    }
    if (!indexable(v)) return error.Unsupported;
    return v.obj.?.items.items[i];
}
pub fn next(h: *Heap, v: Value, cursor: *usize) Error!?Value {
    if (indexable(v)) {
        if (cursor.* >= try val.length(v)) return null;
        const value = try at(h, v, cursor.*);
        cursor.* += 1;
        return value;
    }
    if (v.kind != .enumerate and v.kind != .zip and v.kind != .map_str) return error.Unsupported;
    const o = v.obj.?;
    const args = o.items.items;
    const i = o.cursor;
    if (o.exhausted) return null;
    if (v.kind == .zip and args.len == 0) {
        o.exhausted = true;
        return null;
    }
    for (if (v.kind == .enumerate) args[0..1] else args) |src| if (i >= try val.length(src)) {
        o.exhausted = true;
        return null;
    };
    o.cursor += 1;
    if (v.kind == .map_str) return try string(h, try at(h, args[0], i));
    const result = try h.new(.tuple);
    if (v.kind == .enumerate) {
        try h.append(result, try Value.int((try args[1].numeric()) + @as(i64, @intCast(i))));
        try h.append(result, try at(h, args[0], i));
    } else {
        for (args) |src| {
            try h.append(result, try at(h, src, i));
        }
    }
    return result;
}
pub fn collect(h: *Heap, v: Value, allow_set: bool) Error!Value {
    const out = try h.new(.list);
    if (v.kind == .set and allow_set) {
        for (v.obj.?.items.items) |x| try h.append(out, x);
    } else {
        var cursor: usize = 0;
        while (try next(h, v, &cursor)) |x| try h.append(out, x);
    }
    return out;
}
pub fn string(h: *Heap, v: Value) Error!Value {
    if (v.kind == .string) return v;
    var buffer: [16]u8 = undefined;
    const text = switch (v.kind) {
        .integer => std.fmt.bufPrint(&buffer, "{d}", .{v.n}) catch return error.Unsupported,
        .boolean => if (v.n != 0) "True" else "False",
        .none => "None",
        else => return error.Unsupported,
    };
    return h.string(text);
}
