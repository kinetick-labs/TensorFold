const std = @import("std");
const v = @import("runner_frontend_collections_value.zig");
const ops = @import("runner_frontend_collections_ops.zig");
const ast = @import("runner_frontend_collections_ast.zig");
const Value = v.Value;
const Heap = v.Heap;
const Error = ast.Error;
fn is(name: []const u8, wanted: []const u8) bool {
    return std.mem.eql(u8, name, wanted);
}
fn argc(args: []const Value, min: usize, max: usize) Error!void {
    if (args.len < min or args.len > max) return error.Unsupported;
}
fn iterator(h: *Heap, kind: v.Kind, args: []const Value) Error!Value {
    if (kind == .enumerate) {
        if (!iteratorInput(args[0])) return error.Unsupported;
    } else {
        for (args) |x| {
            if (!iteratorInput(x)) return error.Unsupported;
        }
    }
    if (h.stats.cells + args.len > 1536) return error.Unsupported;
    const out = try h.new(kind);
    h.stats.cells += args.len;
    try out.obj.?.items.appendSlice(h.a, args);
    return out;
}
pub fn call(h: *Heap, name: []const u8, args: []const Value, keywords: []const []const u8) Error!Value {
    try h.charge(24);
    if (args.len > 3 or keywords.len != args.len) return error.Unsupported;
    if (is(name, "sorted")) return sorted(h, args, keywords);
    for (keywords) |key| if (key.len != 0) return error.Unsupported;
    if (is(name, "len")) {
        try argc(args, 1, 1);
        return Value.int(@intCast(try v.length(args[0])));
    }
    if (is(name, "bool")) {
        try argc(args, 0, 1);
        return Value.boolean(args.len != 0 and try v.truth(args[0]));
    }
    if (is(name, "str")) {
        try argc(args, 0, 1);
        return if (args.len == 0) h.string("") else ops.string(h, args[0]);
    }
    if (is(name, "int")) {
        try argc(args, 0, 1);
        if (args.len == 0) return Value.int(0);
        if (args[0].kind != .string) return Value.int(try args[0].numeric());
        const s = std.mem.trim(u8, args[0].obj.?.text, " \t\n");
        if (s.len == 0) return error.Unsupported;
        var i: usize = @intFromBool(s[0] == '-' or s[0] == '+');
        if (i == s.len) return error.Unsupported;
        while (i < s.len) : (i += 1) if (!std.ascii.isDigit(s[i])) return error.Unsupported;
        return Value.int(std.fmt.parseInt(i64, s, 10) catch return error.Unsupported);
    }
    if (is(name, "abs")) {
        try argc(args, 1, 1);
        const n = try args[0].numeric();
        return Value.int(if (n < 0) -n else n);
    }
    if (is(name, "range")) {
        try argc(args, 1, 3);
        const start = if (args.len > 1) try args[0].numeric() else 0;
        const stop = try args[@intFromBool(args.len > 1)].numeric();
        const step = if (args.len == 3) try args[2].numeric() else 1;
        if (step == 0) return error.Unsupported;
        const out = try h.new(.range);
        var i = start;
        while (if (step > 0) i < stop else i > stop) : (i += step) try h.append(out, try Value.int(i));
        return out;
    }
    if (is(name, "list") or is(name, "tuple") or is(name, "set")) {
        try argc(args, 0, 1);
        const kind: v.Kind = if (is(name, "list")) .list else if (is(name, "tuple")) .tuple else .set;
        const out = try h.new(kind);
        if (args.len == 0) return out;
        var cursor: usize = 0;
        while (try ops.next(h, args[0], &cursor)) |x| {
            if (kind == .set) try h.setAdd(out, x) else try h.append(out, x);
        }
        return out;
    }
    if (is(name, "dict")) {
        try argc(args, 0, 1);
        const out = try h.new(.dict);
        if (args.len == 0) return out;
        if (args[0].kind == .dict) {
            const data = args[0].obj.?.items.items;
            var i: usize = 0;
            while (i < data.len) : (i += 2) try h.dictSet(out, data[i], data[i + 1]);
        } else {
            var cursor: usize = 0;
            while (try ops.next(h, args[0], &cursor)) |pair| {
                if (try v.length(pair) != 2) return error.Unsupported;
                const key = try ops.at(h, pair, 0);
                const value = try ops.at(h, pair, 1);
                try h.dictSet(out, key, value);
            }
        }
        return out;
    }
    if (is(name, "enumerate")) {
        try argc(args, 1, 2);
        const start = if (args.len == 2) args[1] else try Value.int(0);
        _ = try start.numeric();
        return iterator(h, .enumerate, &.{ args[0], start });
    }
    if (is(name, "zip")) {
        try argc(args, 0, 3);
        return iterator(h, .zip, args);
    }
    if (is(name, "map")) {
        try argc(args, 2, 2);
        if (args[0].kind != .builtin or args[0].n != 2) return error.Unsupported;
        return iterator(h, .map_str, args[1..]);
    }
    if (is(name, "sum")) {
        try argc(args, 1, 2);
        var sum = if (args.len == 2) args[1] else try Value.int(0);
        _ = try sum.numeric();
        var cursor: usize = 0;
        while (try ops.next(h, args[0], &cursor)) |x| {
            _ = try x.numeric();
            sum = try ops.binary(h, .add, sum, x);
        }
        return sum;
    }
    if (is(name, "min") or is(name, "max")) {
        try argc(args, 1, 3);
        var values = args;
        if (args.len == 1) values = (try ops.collect(h, args[0], false)).obj.?.items.items;
        if (values.len == 0) return error.Unsupported;
        var best = values[0];
        for (values[1..]) |x| {
            try h.charge(16);
            if (try h.order(x, best) == (if (is(name, "min")) std.math.Order.lt else .gt)) best = x;
        }
        return best;
    }
    if (is(name, "any") or is(name, "all")) {
        try argc(args, 1, 1);
        var cursor: usize = 0;
        while (try ops.next(h, args[0], &cursor)) |x| {
            try h.charge(8);
            if (try v.truth(x) == is(name, "any")) return Value.boolean(is(name, "any"));
        }
        return Value.boolean(is(name, "all"));
    }
    return error.Unsupported;
}
fn sorted(h: *Heap, args: []const Value, keywords: []const []const u8) Error!Value {
    if (args.len == 0 or args.len > 3 or keywords[0].len != 0) return error.Unsupported;
    var key: Value = .{};
    var reverse = false;
    for (args[1..], keywords[1..]) |arg, keyword| {
        if (is(keyword, "key")) {
            if (arg.kind != .none and (arg.kind != .builtin or arg.n != 1)) return error.Unsupported;
            key = arg;
        } else if (is(keyword, "reverse")) reverse = (try arg.numeric()) != 0 else return error.Unsupported;
    }
    const out = try ops.collect(h, args[0], true);
    const data = out.obj.?.items.items;
    var keys: [128]Value = undefined;
    for (data, 0..) |item, i| keys[i] = if (key.kind == .builtin) try call(h, "sum", &.{item}, &.{""}) else item;
    for (1..@max(1, data.len)) |i| {
        const item = data[i];
        const k = keys[i];
        var j = i;
        while (j > 0) {
            const prior = data[j - 1];
            const pk = keys[j - 1];
            try h.charge(20);
            const order = try h.order(pk, k);
            if (if (reverse) order != .lt else order != .gt) break;
            data[j] = prior;
            keys[j] = pk;
            j -= 1;
        }
        data[j] = item;
        keys[j] = k;
    }
    return out;
}

fn iteratorInput(value: Value) bool {
    return value.kind == .string or value.kind == .list or value.kind == .tuple or value.kind == .range;
}
