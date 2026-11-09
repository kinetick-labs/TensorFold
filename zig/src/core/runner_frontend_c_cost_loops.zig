const std = @import("std");
const m = @import("runner_frontend_c_cost_model.zig");
const expression = @import("runner_frontend_c_cost_expr.zig");
const Flow = @import("runner_frontend_c_cost.zig").Flow;
const V = m.V;
const Error = m.Error;
const Env = m.Env;

pub fn forLoop(p: anytype, env: *Env, node: V) Error!Flow {
    const c = m.children(node);
    if (c.len != 5 or m.field(c[1], "kind") != .null) return error.Unsupported;
    const mark = env.bindings.items.len;
    defer env.bindings.shrinkRetainingCapacity(mark);
    if (m.field(c[0], "kind") != .null) {
        const init = try p.stmt(env, c[0]);
        if (!init.normal or init.returned != null) return error.Unsupported;
    }
    const step = try increment(c[3]);
    const count = try tripCount(p, env, c[2], step);
    return run(p, env, c[2], c[3], c[4], null, count);
}

pub fn whileLoop(p: anytype, env: *Env, node: V) Error!Flow {
    const c = m.children(node);
    if (c.len != 2) return error.Unsupported;
    if (try euclid(p, env, c[0], c[1])) |flow| return flow;
    if (!m.is(c[1], "CompoundStmt")) {
        const step = try increment(c[1]);
        const count = try tripCount(p, env, c[0], step);
        return run(p, env, c[0], c[1], .null, &.{}, count);
    }
    const body = m.children(c[1]);
    if (body.len == 0) return error.Unsupported;
    const inc = body[body.len - 1];
    const step = try increment(inc);
    const count = try tripCount(p, env, c[0], step);
    return run(p, env, c[0], inc, .null, body[0 .. body.len - 1], count);
}

const Step = struct { id: []const u8, delta: i128 };

fn increment(raw: V) Error!Step {
    const n = m.strip(raw);
    const c = m.children(n);
    if (c.len == 1 and m.is(n, "UnaryOperator") and (m.op(n, "++") or m.op(n, "--"))) {
        const id = m.ref(c[0]);
        if (id.len == 0) return error.Unsupported;
        return .{ .id = id, .delta = if (m.op(n, "++")) 1 else -1 };
    }
    if (c.len == 2 and m.is(n, "CompoundAssignOperator") and (m.op(n, "+=") or m.op(n, "-="))) {
        const id = m.ref(c[0]);
        const value = m.strip(c[1]);
        if (id.len == 0 or !m.is(value, "IntegerLiteral")) return error.Unsupported;
        var delta = try m.integer(value);
        if (m.op(n, "-=")) delta = -delta;
        if (delta == 0) return error.Unsupported;
        return .{ .id = id, .delta = delta };
    }
    if (c.len == 2 and m.is(n, "BinaryOperator") and m.op(n, "=")) {
        const id = m.ref(c[0]);
        const rhs = m.strip(c[1]);
        const terms = m.children(rhs);
        if (id.len == 0 or !m.is(rhs, "BinaryOperator") or terms.len != 2 or
            !std.mem.eql(u8, id, m.ref(terms[0])) or (!m.op(rhs, "+") and !m.op(rhs, "-")))
            return error.Unsupported;
        const value = m.strip(terms[1]);
        if (!m.is(value, "IntegerLiteral")) return error.Unsupported;
        var delta = try m.integer(value);
        if (m.op(rhs, "-")) delta = -delta;
        if (delta == 0) return error.Unsupported;
        return .{ .id = id, .delta = delta };
    }
    return error.Unsupported;
}

fn tripCount(p: anytype, env: *Env, raw: V, step: Step) Error!usize {
    const n = m.strip(raw);
    const c = m.children(n);
    const binding = env.find(step.id) orelse return error.Unsupported;
    const start = try binding.value.numeric();
    if (start.lo != start.hi) return error.Unsupported;
    if (m.is(n, "ArraySubscriptExpr") and c.len == 2 and step.delta == 1 and
        std.mem.eql(u8, m.ref(c[1]), step.id))
    {
        const base = try p.eval(env, c[0]);
        const text = base.text orelse if (base.array) |ar| ar.text orelse return error.Unsupported else return error.Unsupported;
        if (start.lo < 0 or start.hi > text.len) return error.Unsupported;
        const count = text.len - @as(usize, @intCast(start.lo));
        if (count > p.limit) return error.Unsupported;
        return count;
    }
    if (!m.is(n, "BinaryOperator") or c.len != 2 or !std.mem.eql(u8, expression.safeRef(env, c[0]), step.id))
        return error.Unsupported;
    try pure(p, c[1], 0);
    const stop = try (try p.eval(env, c[1])).numeric();
    var count: i128 = 0;
    if (step.delta > 0 and (m.op(n, "<") or m.op(n, "<="))) {
        const last = stop.hi - @intFromBool(m.op(n, "<"));
        if (last >= start.lo) count = @divFloor(last - start.lo, step.delta) + 1;
    } else if (step.delta < 0 and (m.op(n, ">") or m.op(n, ">="))) {
        const last = stop.lo + @intFromBool(m.op(n, ">"));
        if (start.hi >= last) count = @divFloor(start.hi - last, -step.delta) + 1;
    } else return error.Unsupported;
    if (count > p.limit) return error.Unsupported;
    const t = binding.typ orelse return error.Unsupported;
    const end = std.math.mul(i128, count, step.delta) catch return error.Unsupported;
    const final = std.math.add(i128, start.lo, end) catch return error.Unsupported;
    if (!t.fits(m.Range.exact(final))) return error.Unsupported;
    return @intCast(count);
}

fn run(p: anytype, env: *Env, condition: V, inc: V, body: V, slice: ?[]const V, count: usize) Error!Flow {
    var exits: ?Env = null;
    var result: Flow = .{ .normal = false };
    var iteration: usize = 0;
    while (true) {
        const r = try (try p.eval(env, condition)).numeric();
        try p.charge(2);
        const maybe_exit = r.lo <= 0 and r.hi >= 0;
        if (maybe_exit) {
            var exit = try p.clone(env);
            expression.refine(&exit, condition, false);
            if (exits) |*old| try old.merge(&exit) else exits = exit;
        }
        if (r.lo == 0 and r.hi == 0) break;
        if (iteration >= count) return error.Unsupported;
        iteration += 1;
        expression.refine(env, condition, true);
        const lock_mark = p.locked.items.len;
        try protect(p, condition, 0);
        const next = if (slice) |nodes| try p.block(env, nodes) else try p.stmt(env, body);
        p.locked.shrinkRetainingCapacity(lock_mark);
        try result.include(next);
        if (!next.normal) break;
        _ = try p.eval(env, inc);
        try p.charge(2);
    }
    if (exits) |exit| {
        env.* = exit;
        result.normal = true;
    }
    return result;
}

fn protect(p: anytype, n: V, depth: usize) Error!void {
    if (depth >= 64 or p.locked.items.len >= 256) return error.Unsupported;
    try pure(p, n, depth);
    const id = m.ref(n);
    if (id.len != 0) try p.locked.append(p.a, id);
    for (m.children(n)) |child| try protect(p, child, depth + 1);
}

fn pure(p: anytype, n: V, depth: usize) Error!void {
    try p.touch();
    if (depth >= 64 or m.is(n, "CallExpr") or m.is(n, "CompoundAssignOperator") or m.op(n, "=") or
        m.op(n, "++") or m.op(n, "--") or m.op(n, ",")) return error.Unsupported;
    for (m.children(n)) |child| try pure(p, child, depth + 1);
}

fn assignment(n: V) ?struct { id: []const u8, rhs: V } {
    const c = m.children(n);
    if (!m.is(n, "BinaryOperator") or !m.op(n, "=") or c.len != 2) return null;
    const id = m.ref(c[0]);
    if (id.len == 0) return null;
    return .{ .id = id, .rhs = c[1] };
}

fn euclid(p: anytype, env: *Env, cond: V, body: V) Error!?Flow {
    const nodes = m.children(body);
    if (!m.is(body, "CompoundStmt") or nodes.len != 3 or !m.is(nodes[0], "DeclStmt")) return null;
    const declarations = m.children(nodes[0]);
    if (declarations.len != 1 or !m.is(declarations[0], "VarDecl")) return null;
    const temp = declarations[0];
    const initializers = m.children(temp);
    if (initializers.len != 1) return null;
    const first = assignment(nodes[1]) orelse return null;
    const second = assignment(nodes[2]) orelse return null;
    const delayed = m.ref(initializers[0]).len != 0;
    const remainder_source = if (delayed) first.rhs else initializers[0];
    const rem = m.strip(remainder_source);
    const operands = m.children(rem);
    if (!m.is(rem, "BinaryOperator") or !m.op(rem, "%") or operands.len != 2) return null;
    if ((m.integerType(m.typ(rem)) orelse return error.Unsupported).bits > 32) return error.Unsupported;
    const a_id = expression.safeRef(env, operands[0]);
    const b_id = expression.safeRef(env, operands[1]);
    const t_id = m.text(m.field(temp, "id"));
    if (a_id.len == 0 or b_id.len == 0 or std.mem.eql(u8, a_id, b_id)) return null;
    const saved_b = if (delayed) initializers[0] else first.rhs;
    if (!std.mem.eql(u8, first.id, if (delayed) b_id else a_id) or
        !std.mem.eql(u8, expression.safeRef(env, saved_b), b_id) or
        !std.mem.eql(u8, second.id, if (delayed) a_id else b_id) or
        !std.mem.eql(u8, m.ref(second.rhs), t_id)) return null;
    const guard = m.strip(cond);
    var guard_id = expression.safeRef(env, cond);
    if (m.is(guard, "BinaryOperator") and m.op(guard, "!=")) {
        const c = m.children(guard);
        if (c.len != 2 or !m.is(m.strip(c[1]), "IntegerLiteral") or try m.integer(m.strip(c[1])) != 0)
            return null;
        guard_id = expression.safeRef(env, c[0]);
    }
    if (!std.mem.eql(u8, guard_id, b_id)) return null;
    const a_binding = env.find(a_id) orelse return error.Unsupported;
    const b_binding = env.find(b_id) orelse return error.Unsupported;
    const x = try a_binding.value.numeric();
    const y = try b_binding.value.numeric();
    const top = @max(x.hi, y.hi);
    if (x.lo < 0 or y.lo < 0 or top > std.math.maxInt(u32)) return error.Unsupported;
    const bound: m.Range = .{ .lo = 0, .hi = top };
    var wrapper = remainder_source;
    for (0..64) |_| {
        if (m.is(wrapper, "BinaryOperator")) break;
        const inner = m.children(wrapper);
        if (inner.len != 1) return error.Unsupported;
        if (m.is(wrapper, "ImplicitCastExpr") or m.is(wrapper, "CStyleCastExpr")) {
            if (!(m.integerType(m.typ(wrapper)) orelse return error.Unsupported).fits(bound))
                return error.Unsupported;
        } else if (!m.is(wrapper, "ParenExpr")) return error.Unsupported;
        wrapper = inner[0];
    }
    if (!std.mem.eql(u8, expression.transparentRef(second.rhs, bound), t_id) or
        !std.mem.eql(u8, expression.transparentRef(saved_b, bound), b_id) or
        !std.mem.eql(u8, expression.transparentRef(operands[0], bound), a_id) or
        !std.mem.eql(u8, expression.transparentRef(operands[1], bound), b_id)) return error.Unsupported;
    if (!(a_binding.typ orelse return error.Unsupported).fits(bound) or
        !(b_binding.typ orelse return error.Unsupported).fits(bound) or
        !(m.integerType(m.typ(temp)) orelse return error.Unsupported).fits(bound)) return error.Unsupported;
    var bits: usize = 0;
    var n = top;
    while (n > 0) : (n = @divTrunc(n, 2)) bits += 1;
    const iterations = if (y.hi == 0) 0 else 2 * bits + 1;
    if (iterations > p.limit) return error.Unsupported;
    try p.charge(12 + iterations * 45);
    try p.store(env, operands[0], .{ .number = bound });
    try p.store(env, operands[1], m.Value.integer(0));
    return Flow{};
}
