const std = @import("std");
const ast = @import("runner_frontend_collections_ast.zig");
const val = @import("runner_frontend_collections_value.zig");
const ops = @import("runner_frontend_collections_ops.zig");
const builtins = @import("runner_frontend_collections_builtins.zig");
const methods = @import("runner_frontend_collections_methods.zig");
const emitter = @import("runner_frontend_collections_emit.zig");
const Error = ast.Error;
const Value = val.Value;
const Frame = struct { scope: *const ast.Scope, values: [64]?Value = @splat(null) };
const Binding = struct { name: []const u8, value: Value, previous: ?*const Binding = null };
const Flow = union(enum) { next, ret: Value, break_loop, continue_loop };

pub fn prove(p: *ast.Program) Error!void {
    p.stats = .{};
    if (p.functions.items.len > 16) return error.Unsupported;
    var edges: [16]u16 = @splat(0);
    try validate(p, p.body, null, &edges, 0);
    for (p.functions.items, 0..) |f, i| {
        if (f.params.len > 8) return error.Unsupported;
        try validate(p, f.body, i, &edges, 0);
    }
    for (p.functions.items, 0..) |_, i| try acyclic(&edges, i, 0);
    var globals: Frame = .{ .scope = &p.globals };
    var r: Runtime = .{ .p = p, .globals = &globals, .heap = .{ .a = p.a, .stats = &p.stats } };
    for (p.nodes.items, 0..) |n, i| if (n.tag == .string) {
        r.literals[i] = try r.heap.string(n.text);
    };
    if (p.globals.find("__name__")) |i| globals.values[i] = try r.heap.string("__main__");
    if (try r.block(&globals, p.body, null) != .next or p.stats.prints == 0) return error.Unsupported;
}
fn acyclic(edges: *const [16]u16, id: usize, seen: u16) Error!void {
    const bit = @as(u16, 1) << @intCast(id);
    if (seen & bit != 0) return error.Unsupported;
    for (0..16) |i| if (edges[id] & (@as(u16, 1) << @intCast(i)) != 0) try acyclic(edges, i, seen | bit);
}
fn validate(p: *const ast.Program, body: []const ast.Stmt, function: ?usize, edges: *[16]u16, depth: usize) Error!void {
    if (depth > 24) return error.Unsupported;
    for (body) |s| {
        if (s.expr) |id| try node(p, id, function, edges, depth + 1);
        for (s.targets) |id| try node(p, id, function, edges, depth + 1);
        try validate(p, s.body, function, edges, depth + 1);
        try validate(p, s.otherwise, function, edges, depth + 1);
    }
}
fn node(p: *const ast.Program, id: usize, function: ?usize, edges: *[16]u16, depth: usize) Error!void {
    if (depth > 64 or id >= p.nodes.items.len) return error.Unsupported;
    const n = p.nodes.items[id];
    if (n.tag == .call) {
        if (p.functionId(n.text)) |callee| {
            if (n.args.len != p.functions.items[callee].params.len) return error.Unsupported;
            for (n.keywords) |k| if (k.len != 0) return error.Unsupported;
            if (function) |caller| edges[caller] |= @as(u16, 1) << @intCast(callee);
        } else if (emitter.lookup(&emitter.builtins, n.text) == null and !std.mem.eql(u8, n.text, "map")) return error.Unsupported;
    }
    if (n.tag == .method and emitter.lookup(&emitter.methods, n.text) == null) return error.Unsupported;
    switch (n.tag) {
        .integer, .boolean, .none, .string, .name => {},
        .list, .tuple, .dict, .set, .call => for (n.args) |arg| try node(p, arg, function, edges, depth + 1),
        .method => {
            try node(p, n.a, function, edges, depth + 1);
            for (n.args) |arg| try node(p, arg, function, edges, depth + 1);
        },
        .slice => {
            try node(p, n.a, function, edges, depth + 1);
            for (n.args) |arg| try node(p, arg, function, edges, depth + 1);
        },
        .neg, .pos, .logical_not => try node(p, n.a, function, edges, depth + 1),
        else => {
            try node(p, n.a, function, edges, depth + 1);
            try node(p, n.b, function, edges, depth + 1);
            if (n.c) |c| try node(p, c, function, edges, depth + 1);
        },
    }
}
const Runtime = struct {
    p: *ast.Program,
    globals: *Frame,
    heap: val.Heap,
    literals: [1024]?Value = @splat(null),
    depth: usize = 0,
    calls: u16 = 0,
    defined_functions: u16 = 0,
    eval_depth: usize = 0,
    fn name(r: *Runtime, f: *Frame, text: []const u8, binding: ?*const Binding) Error!Value {
        var b = binding;
        while (b) |item| : (b = item.previous) if (std.mem.eql(u8, item.name, text)) return item.value;
        if (f.scope.find(text)) |i| return f.values[i] orelse error.Unsupported;
        if (r.p.globals.find(text)) |i| return r.globals.values[i] orelse error.Unsupported;
        if (std.mem.eql(u8, text, "sum")) return .{ .kind = .builtin, .n = 1 };
        if (std.mem.eql(u8, text, "str")) return .{ .kind = .builtin, .n = 2 };
        return error.Unsupported;
    }
    fn expr(r: *Runtime, f: *Frame, id: usize, b: ?*const Binding) Error!Value {
        if (r.eval_depth > 64) return error.Unsupported;
        r.eval_depth += 1;
        defer r.eval_depth -= 1;
        try r.heap.charge(6);
        const n = r.p.nodes.items[id];
        switch (n.tag) {
            .integer => return Value.int(n.value),
            .boolean => return Value.boolean(n.value != 0),
            .none => return .{},
            .string => return r.literals[id] orelse error.Unsupported,
            .name => return r.name(f, n.text, b),
            .list, .tuple, .dict, .set => {
                const kind: val.Kind = switch (n.tag) {
                    .list => .list,
                    .tuple => .tuple,
                    .dict => .dict,
                    else => .set,
                };
                const out = try r.heap.new(kind);
                var i: usize = 0;
                while (i < n.args.len) : (i += 1) {
                    const value = try r.expr(f, n.args[i], b);
                    if (kind == .dict) {
                        i += 1;
                        if (i == n.args.len) return error.Unsupported;
                        const item = try r.expr(f, n.args[i], b);
                        try r.heap.dictSet(out, value, item);
                    } else if (kind == .set) try r.heap.setAdd(out, value) else try r.heap.append(out, value);
                }
                return out;
            },
            .neg, .pos, .logical_not => return ops.unary(&r.heap, n.tag, try r.expr(f, n.a, b)),
            .logical_and, .logical_or => {
                const x = try r.expr(f, n.a, b);
                if (try val.truth(x) == (n.tag == .logical_or)) return x;
                return r.expr(f, n.b, b);
            },
            .conditional => return r.expr(f, if (try val.truth(try r.expr(f, n.b, b))) n.a else n.c.?, b),
            .index => {
                const object = try r.expr(f, n.a, b);
                const key = try r.expr(f, n.b, b);
                return r.heap.get(object, key);
            },
            .slice => {
                if (n.args.len != 3) return error.Unsupported;
                const object = try r.expr(f, n.a, b);
                const lo = try r.expr(f, n.args[0], b);
                const hi = try r.expr(f, n.args[1], b);
                const step = try r.expr(f, n.args[2], b);
                return ops.slice(&r.heap, object, lo, hi, step);
            },
            .comprehension => return r.comprehension(f, n, b, null),
            .call, .method => {
                if (n.tag == .call and n.args.len == 1 and
                    (std.mem.eql(u8, n.text, "any") or std.mem.eql(u8, n.text, "all")))
                {
                    const gen = r.p.nodes.items[n.args[0]];
                    if (gen.tag == .comprehension and gen.value == 1) return r.comprehension(f, gen, b, std.mem.eql(u8, n.text, "all"));
                }
                const base = if (n.tag == .method) try r.expr(f, n.a, b) else Value{};
                if (n.args.len > 8) return error.Unsupported;
                var args: [8]Value = undefined;
                for (n.args, 0..) |arg, i| args[i] = try r.expr(f, arg, b);
                if (n.tag == .method) return methods.call(&r.heap, base, n.text, args[0..n.args.len], n.keywords);
                if (r.p.functionId(n.text)) |callee| return r.call(callee, args[0..n.args.len]);
                return builtins.call(&r.heap, n.text, args[0..n.args.len], n.keywords);
            },
            else => {
                const x = try r.expr(f, n.a, b);
                const y = try r.expr(f, n.b, b);
                return ops.binary(&r.heap, n.tag, x, y);
            },
        }
    }
    fn comprehension(r: *Runtime, f: *Frame, n: ast.Node, b: ?*const Binding, all: ?bool) Error!Value {
        const seq = try r.expr(f, n.b, b);
        const expected = dictSize(seq);
        const out = if (all == null) try r.heap.new(.list) else Value{};
        var cursor: usize = 0;
        while (true) {
            try guard(seq, expected);
            const x = try ops.next(&r.heap, seq, &cursor) orelse break;
            const binding: Binding = .{ .name = n.text, .value = x, .previous = b };
            if (n.c) |filter| if (!try val.truth(try r.expr(f, filter, &binding))) continue;
            const value = try r.expr(f, n.a, &binding);
            if (all) |a| {
                if (try val.truth(value) != a) return Value.boolean(!a);
            } else try r.heap.append(out, value);
        }
        return if (all) |a| Value.boolean(a) else out;
    }
    fn call(r: *Runtime, id: usize, args: []const Value) Error!Value {
        const bit = @as(u16, 1) << @intCast(id);
        if (r.depth >= 8 or r.calls & bit != 0 or r.defined_functions & bit == 0) return error.Unsupported;
        const function = r.p.functions.items[id];
        if (args.len != function.params.len) return error.Unsupported;
        var frame: Frame = .{ .scope = &function.scope };
        for (function.params, args) |name_, value| frame.values[function.scope.find(name_).?] = value;
        r.depth += 1;
        r.calls |= bit;
        defer {
            r.depth -= 1;
            r.calls &= ~bit;
        }
        r.p.stats.depth = @max(r.p.stats.depth, r.depth);
        try r.heap.charge(32);
        return switch (try r.block(&frame, function.body, null)) {
            .next => Value{},
            .ret => |v| v,
            else => error.Unsupported,
        };
    }
    fn assign(r: *Runtime, f: *Frame, target: usize, value: Value, b: ?*const Binding) Error!void {
        const n = r.p.nodes.items[target];
        switch (n.tag) {
            .name => {
                const i = f.scope.find(n.text) orelse return error.Unsupported;
                f.values[i] = value;
            },
            .tuple => {
                const seq = try ops.collect(&r.heap, value, false);
                const items = seq.obj.?.items.items;
                if (items.len != n.args.len) return error.Unsupported;
                for (n.args, items) |sub, item| try r.assign(f, sub, item, b);
            },
            .index => {
                const base = try r.expr(f, n.a, b);
                const key = try r.expr(f, n.b, b);
                try r.heap.set(base, key, value);
            },
            else => return error.Unsupported,
        }
    }
    fn block(r: *Runtime, f: *Frame, body: []const ast.Stmt, b: ?*const Binding) Error!Flow {
        for (body, 0..) |s, statement_index| {
            try r.heap.charge(10);
            switch (s.tag) {
                .assign => {
                    const value = try r.expr(f, s.expr.?, b);
                    for (s.targets) |target| try r.assign(f, target, value, b);
                },
                .augassign => {
                    const n = r.p.nodes.items[s.targets[0]];
                    const base = if (n.tag == .index) try r.expr(f, n.a, b) else Value{};
                    const key = if (n.tag == .index) try r.expr(f, n.b, b) else Value{};
                    const left = if (n.tag == .index) try r.heap.get(base, key) else try r.expr(f, s.targets[0], b);
                    const right = try r.expr(f, s.expr.?, b);
                    const result = try ops.inplace(&r.heap, s.op, left, right);
                    if (n.tag == .index) try r.heap.set(base, key, result) else try r.assign(f, s.targets[0], result, b);
                },
                .expression => {
                    const result = try r.expr(f, s.expr.?, b);
                    if (body.ptr == r.p.body.ptr and statement_index + 1 == body.len and result.kind != .none)
                        return error.Unsupported;
                },
                .print => {
                    var values: [16]Value = undefined;
                    if (s.targets.len > values.len) return error.Unsupported;
                    for (s.targets, 0..) |arg, i| values[i] = try r.expr(f, arg, b);
                    for (values[0..s.targets.len]) |value| try r.heap.output(value, false, 0);
                    r.heap.output_bytes += @max(1, s.targets.len);
                    if (r.heap.output_bytes > 8192) return error.Unsupported;
                    r.p.stats.prints += 1;
                    try r.heap.charge(s.targets.len * 8 + 8);
                },
                .assertion => if (!try val.truth(try r.expr(f, s.expr.?, b))) return error.Unsupported,
                .conditional => {
                    const flow = try r.block(f, if (try val.truth(try r.expr(f, s.expr.?, b))) s.body else s.otherwise, b);
                    if (flow != .next) return flow;
                },
                .while_loop => {
                    while (try val.truth(try r.expr(f, s.expr.?, b))) {
                        const flow = try r.block(f, s.body, b);
                        if (flow == .break_loop) break;
                        if (flow == .ret) return flow;
                    }
                },
                .for_loop => {
                    const iterable = try r.expr(f, s.expr.?, b);
                    const expected = dictSize(iterable);
                    var cursor: usize = 0;
                    while (true) {
                        try r.heap.charge(12);
                        try guard(iterable, expected);
                        const value = try ops.next(&r.heap, iterable, &cursor) orelse break;
                        try r.assign(f, s.targets[0], value, b);
                        const flow = try r.block(f, s.body, b);
                        if (flow == .break_loop) break;
                        if (flow == .ret) return flow;
                    }
                },
                .ret => return .{ .ret = if (s.expr) |id| try r.expr(f, id, b) else .{} },
                .break_loop => return .break_loop,
                .continue_loop => return .continue_loop,
                .function => r.defined_functions |= @as(u16, 1) << @intCast(s.function),
                .pass => {},
            }
        }
        return .next;
    }
};
fn dictSize(v: Value) ?usize {
    return if (v.kind == .dict or v.kind == .items or v.kind == .keys or v.kind == .values) v.obj.?.items.items.len else null;
}
fn guard(v: Value, expected: ?usize) Error!void {
    if (expected) |size| if (v.obj.?.items.items.len != size) return error.Unsupported;
}
