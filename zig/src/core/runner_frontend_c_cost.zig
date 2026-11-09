const std = @import("std");
const m = @import("runner_frontend_c_cost_model.zig");
const expression = @import("runner_frontend_c_cost_expr.zig");
const loops = @import("runner_frontend_c_cost_loops.zig");
const printing = @import("runner_frontend_c_cost_print.zig");
const Allocator = std.mem.Allocator;
const V = m.V;
const Error = m.Error;
const Value = m.Value;
const Env = m.Env;

// Estimates original typed C only; the caller reserves any additional opacity-copy startup cost.
pub fn estimate(a: Allocator, ast_json: []const u8, source_offset: usize, limit: usize) Allocator.Error!?usize {
    if (ast_json.len > 4 * 1024 * 1024 or limit == std.math.maxInt(usize) or !m.validateJson(ast_json)) return null;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const scratch = arena.allocator();
    const parsed = std.json.parseFromSlice(V, scratch, ast_json, .{}) catch |err| {
        if (err == error.OutOfMemory) return error.OutOfMemory;
        return null;
    };
    defer parsed.deinit();
    var pass: Pass = .{ .a = scratch, .limit = limit, .offset = source_offset };
    return pass.run(parsed.value) catch |err| {
        if (err == error.OutOfMemory) return error.OutOfMemory;
        return null;
    };
}

pub const Flow = struct {
    normal: bool = true,
    returned: ?Value = null,

    pub fn include(f: *Flow, other: Flow) Error!void {
        if (other.returned) |v| f.returned = if (f.returned) |old| try old.join(v) else v;
    }
};

const Pass = struct {
    a: Allocator,
    limit: usize,
    offset: usize,
    cost: usize = 0,
    work: usize = 0,
    snapshots: usize = 0,
    functions: std.ArrayList(V) = .empty,
    runtime: std.ArrayList(V) = .empty,
    globals: Env = .{},
    active: [32][]const u8 = undefined,
    depth: usize = 0,
    expression_depth: usize = 0,
    locked: std.ArrayList([]const u8) = .empty,

    pub fn touch(p: *Pass) Error!void {
        if (p.work >= 100_000) return error.Unsupported;
        p.work += 1;
    }
    pub fn charge(p: *Pass, n: usize) Error!void {
        p.cost += @min(n, p.limit + 1 - p.cost);
        if (p.cost > p.limit) return error.Unsupported;
    }
    pub fn clone(p: *Pass, env: *const Env) Error!Env {
        if (p.snapshots >= 1024) return error.Unsupported;
        p.snapshots += 1;
        return env.clone(p.a);
    }
    pub fn eval(p: *Pass, env: *Env, node: V) Error!Value {
        if (p.expression_depth >= 64) return error.Unsupported;
        p.expression_depth += 1;
        defer p.expression_depth -= 1;
        return expression.eval(p, env, node);
    }
    pub fn store(p: *Pass, env: *Env, target: V, value: Value) Error!void {
        const id = m.ref(target);
        if (id.len != 0) {
            for (p.locked.items) |s| if (std.mem.eql(u8, id, s)) return error.Unsupported;
            const b = env.find(id) orelse return error.Unsupported;
            if (b.global and p.depth > 1) return error.Unsupported;
            const t = b.typ orelse return error.Unsupported;
            var v = value;
            const r = try v.numeric();
            if (!t.fits(r)) v.number = t.range();
            b.value = v;
            return;
        }
        const n = m.strip(target);
        const c = m.children(n);
        if (!m.is(n, "ArraySubscriptExpr") or c.len != 2) return error.Unsupported;
        const index = try (try p.eval(env, c[1])).numeric();
        const b = env.find(m.ref(c[0])) orelse return error.Unsupported;
        if (b.indirect) return error.Unsupported;
        if (b.global and p.depth > 1) return error.Unsupported;
        for (p.locked.items) |s| if (std.mem.eql(u8, b.id, s)) return error.Unsupported;
        var array = b.value.array orelse return error.Unsupported;
        if (index.lo < 0 or index.hi >= array.count) return error.Unsupported;
        _ = try value.numeric();
        array.element = array.element_type.range();
        array.text = null;
        b.value.array = array;
        try p.charge(6);
    }
    pub fn declare(p: *Pass, env: *Env, node: V, global: bool) Error!void {
        if (!m.is(node, "VarDecl")) return error.Unsupported;
        if (!global and std.mem.eql(u8, m.text(m.field(node, "storageClass")), "static")) return error.Unsupported;
        const raw = m.typ(node);
        const c = m.children(node);
        const id = m.text(m.field(node, "id"));
        if (id.len == 0) return error.Unsupported;
        if (m.arrayType(raw)) |t| {
            var ar: m.Array = .{
                .count = t.count,
                .element_type = t.element,
                .element = if (global) m.Range.exact(0) else t.element.range(),
                .readable = global or c.len == 1,
            };
            if (c.len > 1) return error.Unsupported;
            if (c.len == 1) {
                if (m.is(m.strip(c[0]), "StringLiteral")) {
                    ar.text = try m.string(p.a, c[0]);
                    if (t.element.bits != 8 or ar.text.?.len >= ar.count) return error.Unsupported;
                    ar.element = .{ .lo = 0, .hi = 255 };
                } else if (m.is(c[0], "InitListExpr")) {
                    const init = m.children(c[0]);
                    const filler = m.field(c[0], "array_filler");
                    if (filler != .null or init.len > t.count) return error.Unsupported;
                    var bound: ?m.Range = if (init.len < t.count) m.Range.exact(0) else null;
                    for (init) |n| {
                        const r = try (try p.eval(env, n)).numeric();
                        if (!t.element.fits(r)) return error.Unsupported;
                        bound = if (bound) |old| old.join(r) else r;
                    }
                    ar.element = bound orelse return error.Unsupported;
                } else return error.Unsupported;
            }
            try env.add(p.a, .{ .id = id, .value = .{ .array = ar }, .typ = null, .global = global });
            const bytes = t.count * @max(@as(usize, 1), t.element.bits / 8);
            try p.charge(if (global) 4 else bytes * 10 + 4);
        } else if (m.integerType(raw)) |t| {
            if (c.len > 1) return error.Unsupported;
            var value = if (c.len == 1) try p.eval(env, c[0]) else if (global) Value.integer(0) else Value{};
            if (value.number) |r| {
                if (!t.fits(r)) value.number = t.range();
            }
            try env.add(p.a, .{ .id = id, .value = value, .typ = t, .global = global });
            try p.charge(2);
        } else if (std.mem.indexOfScalar(u8, raw, '*') != null and c.len == 1) {
            const value = try p.eval(env, c[0]);
            if (value.text == null and (value.array == null or value.array.?.text == null)) return error.Unsupported;
            try env.add(p.a, .{ .id = id, .value = value, .typ = null, .global = global, .indirect = true });
            try p.charge(3);
        } else return error.Unsupported;
    }
    pub fn block(p: *Pass, env: *Env, nodes: []const V) Error!Flow {
        const mark = env.bindings.items.len;
        defer env.bindings.shrinkRetainingCapacity(mark);
        var result: Flow = .{};
        for (nodes) |node| {
            const next = try p.stmt(env, node);
            try result.include(next);
            result.normal = next.normal;
            if (!result.normal) break;
        }
        return result;
    }
    pub fn stmt(p: *Pass, env: *Env, node: V) Error!Flow {
        try p.touch();
        const c = m.children(node);
        if (m.is(node, "CompoundStmt")) return p.block(env, c);
        if (m.is(node, "DeclStmt")) {
            for (c) |n| try p.declare(env, n, false);
        } else if (m.is(node, "ReturnStmt")) {
            if (c.len > 1) return error.Unsupported;
            try p.charge(8);
            return .{ .normal = false, .returned = if (c.len == 1) try p.eval(env, c[0]) else Value.integer(0) };
        } else if (m.is(node, "IfStmt")) {
            if (c.len < 2 or c.len > 3) return error.Unsupported;
            return p.branch(env, c[0], c[1], if (c.len == 3) c[2] else .null);
        } else if (m.is(node, "ForStmt")) {
            return loops.forLoop(p, env, node);
        } else if (m.is(node, "WhileStmt")) {
            return loops.whileLoop(p, env, node);
        } else if (!m.is(node, "NullStmt")) {
            _ = try p.eval(env, node);
        }
        return .{};
    }
    fn branch(p: *Pass, env: *Env, cond: V, yes: V, no: V) Error!Flow {
        const r = try (try p.eval(env, cond)).numeric();
        try p.charge(2);
        if (r.lo == 0 and r.hi == 0) {
            expression.refine(env, cond, false);
            return if (no == .null) .{} else p.stmt(env, no);
        }
        if (r.lo > 0 or r.hi < 0) {
            expression.refine(env, cond, true);
            return p.stmt(env, yes);
        }
        var other = try p.clone(env);
        expression.refine(env, cond, true);
        expression.refine(&other, cond, false);
        const base = p.cost;
        var result = try p.stmt(env, yes);
        const yes_cost = p.cost;
        p.cost = base;
        const next = if (no == .null) Flow{} else try p.stmt(&other, no);
        p.cost = @max(p.cost, yes_cost);
        try result.include(next);
        if (result.normal and next.normal) try env.merge(&other) else if (next.normal) env.* = other;
        result.normal = result.normal or next.normal;
        return result;
    }
    pub fn conditionalValue(p: *Pass, env: *Env, c: []const V) Error!Value {
        const r = try (try p.eval(env, c[0])).numeric();
        try p.charge(2);
        if (r.lo == 0 and r.hi == 0) return p.eval(env, c[2]);
        if (r.lo > 0 or r.hi < 0) return p.eval(env, c[1]);
        var other = try p.clone(env);
        expression.refine(env, c[0], true);
        expression.refine(&other, c[0], false);
        const base = p.cost;
        const x = try p.eval(env, c[1]);
        const yes_cost = p.cost;
        p.cost = base;
        const y = try p.eval(&other, c[2]);
        p.cost = @max(p.cost, yes_cost);
        try env.merge(&other);
        return x.join(y);
    }
    pub fn call(p: *Pass, env: *Env, node: V) Error!Value {
        const c = m.children(node);
        if (c.len == 0 or c.len > 17) return error.Unsupported;
        const ref = m.field(m.strip(c[0]), "referencedDecl");
        if (!m.is(ref, "FunctionDecl")) return error.Unsupported;
        const name = m.text(m.field(ref, "name"));
        if (std.mem.eql(u8, name, "printf") and (c.len < 2 or !m.is(m.strip(c[1]), "StringLiteral")))
            return error.Unsupported;
        var args: [16]Value = undefined;
        for (c[1..], 0..) |arg, i| args[i] = try p.eval(env, arg);
        try p.charge(12);
        for (p.functions.items) |f| if (std.mem.eql(u8, name, m.text(m.field(f, "name"))))
            return p.invoke(env, f, args[0 .. c.len - 1]);
        const id = m.text(m.field(ref, "id"));
        for (p.runtime.items) |f| {
            if (!std.mem.eql(u8, id, m.text(m.field(f, "id")))) continue;
            return printing.call(p, name, args[0 .. c.len - 1]);
        }
        return error.Unsupported;
    }
    fn invoke(p: *Pass, caller: *Env, function: V, args: []const Value) Error!Value {
        const name = m.text(m.field(function, "name"));
        if (p.depth >= p.active.len) return error.Unsupported;
        for (p.active[0..p.depth]) |s| if (std.mem.eql(u8, s, name)) return error.Unsupported;
        p.active[p.depth] = name;
        p.depth += 1;
        defer p.depth -= 1;
        var env: Env = .{};
        for (caller.bindings.items) |b| if (b.global) try env.add(p.a, b);
        var n: usize = 0;
        var body: V = .null;
        for (m.children(function)) |node| {
            if (m.is(node, "ParmVarDecl")) {
                if (n >= args.len) return error.Unsupported;
                const t = m.integerType(m.typ(node));
                var value = args[n];
                if (t) |scalar| {
                    if (!scalar.fits(try value.numeric())) value.number = scalar.range();
                } else if (value.text == null and (value.array == null or value.array.?.text == null)) return error.Unsupported;
                try env.add(p.a, .{ .id = m.text(m.field(node, "id")), .value = value, .typ = t, .indirect = t == null });
                n += 1;
            } else if (m.is(node, "CompoundStmt")) body = node else return error.Unsupported;
        }
        if (n != args.len or body == .null) return error.Unsupported;
        const saved = p.locked.items.len;
        defer p.locked.shrinkRetainingCapacity(saved);
        const flow = try p.stmt(&env, body);
        if (p.depth == 1) {
            for (env.bindings.items) |b| if (b.global) {
                if (caller.find(b.id)) |target| target.value = b.value;
            };
        }
        var result = flow.returned;
        if (flow.normal) {
            if (!std.mem.eql(u8, name, "main") and !std.mem.startsWith(u8, m.typ(function), "void ("))
                return error.Unsupported;
            result = if (result) |old| try old.join(Value.integer(0)) else Value.integer(0);
        }
        return result orelse error.Unsupported;
    }
    fn run(p: *Pass, root: V) Error!usize {
        if (!m.is(root, "TranslationUnitDecl")) return error.Unsupported;
        try p.charge(200);
        var main: V = .null;
        for (m.children(root)) |node| {
            const loc = m.field(node, "loc");
            const expansion = m.field(loc, "expansionLoc");
            const raw_offset = m.field(if (expansion == .null) loc else expansion, "offset");
            if (raw_offset != .integer or raw_offset.integer < 0) continue;
            const user = @as(u128, @intCast(raw_offset.integer)) >= p.offset;
            if (m.is(node, "FunctionDecl")) {
                if (!user) {
                    try p.runtime.append(p.a, node);
                    continue;
                }
                var defined = false;
                for (m.children(node)) |child| {
                    if (m.is(child, "CompoundStmt")) defined = true;
                }
                if (!defined) continue;
                if (p.functions.items.len >= 32) return error.Unsupported;
                try p.functions.append(p.a, node);
                if (std.mem.eql(u8, m.text(m.field(node, "name")), "main")) main = node;
            } else if (user and m.is(node, "VarDecl")) {
                try p.declare(&p.globals, node, true);
            } else if (user and !m.is(node, "TypedefDecl") and !m.is(node, "EmptyDecl")) return error.Unsupported;
        }
        if (main == .null) return error.Unsupported;
        _ = try p.invoke(&p.globals, main, &.{});
        return p.cost;
    }
};
