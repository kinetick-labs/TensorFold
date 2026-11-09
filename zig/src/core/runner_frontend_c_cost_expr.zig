const std = @import("std");
const m = @import("runner_frontend_c_cost_model.zig");
const V = m.V;
const Error = m.Error;
const Value = m.Value;
const Range = m.Range;
const Env = m.Env;

pub fn eval(p: anytype, env: *Env, node: V) Error!Value {
    try p.touch();
    const c = m.children(node);
    if (m.is(node, "IntegerLiteral") or m.is(node, "CharacterLiteral")) {
        try p.charge(3);
        const value = try m.integer(node);
        const typ = m.integerType(m.typ(node)) orelse return error.Unsupported;
        if (!typ.fits(Range.exact(value))) return error.Unsupported;
        return Value.integer(value);
    }
    if (m.is(node, "StringLiteral")) {
        const array = m.arrayType(m.typ(node)) orelse return error.Unsupported;
        if (array.element.bits != 8) return error.Unsupported;
        return .{ .text = try m.string(p.a, node) };
    }
    if (m.is(node, "ParenExpr") or m.is(node, "ConstantExpr")) {
        if (c.len != 1) return error.Unsupported;
        return p.eval(env, c[0]);
    }
    if (m.is(node, "ImplicitCastExpr") or m.is(node, "CStyleCastExpr")) {
        if (c.len != 1) return error.Unsupported;
        var v = try p.eval(env, c[0]);
        if (m.integerType(m.typ(node))) |t| {
            const r = try v.numeric();
            if (!t.fits(r)) v.number = t.range();
        } else {
            const target = m.pointerType(m.typ(node)) orelse return error.Unsupported;
            if (v.array) |array| {
                if (target.bits != array.element_type.bits or target.signed != array.element_type.signed or
                    target.plain_char != array.element_type.plain_char)
                    return error.Unsupported;
            } else if (v.text != null) {
                if (target.bits != 8) return error.Unsupported;
            } else return error.Unsupported;
        }
        return v;
    }
    if (m.is(node, "DeclRefExpr")) {
        const b = env.find(m.ref(node)) orelse return error.Unsupported;
        if (b.value.number == null and b.value.array == null and b.value.text == null) return error.Unsupported;
        try p.charge(2);
        return b.value;
    }
    if (m.is(node, "UnaryExprOrTypeTraitExpr")) {
        if (!std.mem.eql(u8, m.text(m.field(node, "name")), "sizeof")) return error.Unsupported;
        const raw = if (c.len == 1) m.typ(c[0]) else m.text(m.field(m.field(node, "argType"), "qualType"));
        try p.charge(3);
        return Value.integer(@intCast(m.size(raw) orelse return error.Unsupported));
    }
    if (m.is(node, "ArraySubscriptExpr")) {
        if (c.len != 2) return error.Unsupported;
        const base = try p.eval(env, c[0]);
        const index = try (try p.eval(env, c[1])).numeric();
        try p.charge(6);
        if (base.array) |array| {
            if (!array.readable) return error.Unsupported;
            if (index.lo < 0 or index.hi >= array.count) return error.Unsupported;
            if (array.text) |s| return charRange(s, index);
            return .{ .number = array.element };
        }
        if (base.text) |s| {
            if (index.lo < 0 or index.hi > s.len) return error.Unsupported;
            return charRange(s, index);
        }
        return error.Unsupported;
    }
    if (m.is(node, "CallExpr")) return p.call(env, node);
    if (m.is(node, "ConditionalOperator")) {
        if (c.len != 3) return error.Unsupported;
        return p.conditionalValue(env, c);
    }
    if (m.is(node, "UnaryOperator")) {
        if (c.len != 1) return error.Unsupported;
        const x = try (try p.eval(env, c[0])).numeric();
        if (m.op(node, "++") or m.op(node, "--")) {
            const delta: i128 = if (m.op(node, "++")) 1 else -1;
            const r = try fit(node, .{ .lo = x.lo + delta, .hi = x.hi + delta });
            try p.store(env, c[0], .{ .number = r });
            try p.charge(3);
            const post = m.field(node, "isPostfix");
            return .{ .number = if (post == .bool and post.bool) x else r };
        }
        try p.charge(2);
        if (m.op(node, "+")) return .{ .number = x };
        if (m.op(node, "-")) return .{ .number = try fit(node, .{ .lo = -x.hi, .hi = -x.lo }) };
        if (m.op(node, "!")) return .{ .number = truth(x, true) };
        return error.Unsupported;
    }
    if (m.is(node, "BinaryOperator") or m.is(node, "CompoundAssignOperator")) {
        if (c.len != 2) return error.Unsupported;
        if (m.op(node, "=")) {
            const v = try p.eval(env, c[1]);
            try p.store(env, c[0], v);
            try p.charge(2);
            return v;
        }
        if (m.op(node, "&&") or m.op(node, "||") or m.op(node, ",")) return error.Unsupported;
        const x = try (try p.eval(env, c[0])).numeric();
        const y = try (try p.eval(env, c[1])).numeric();
        const raw = m.text(m.field(node, "opcode"));
        const compound = m.is(node, "CompoundAssignOperator");
        const operator = if (compound and raw.len > 1) raw[0 .. raw.len - 1] else raw;
        const t = m.integerType(m.typ(node)) orelse return error.Unsupported;
        try p.charge(if (t.bits > 32) 5 else 2);
        if (t.bits > 32 and (std.mem.eql(u8, operator, "/") or std.mem.eql(u8, operator, "%")))
            try p.charge(2000);
        const result: Value = .{ .number = try binary(node, operator, x, y) };
        if (compound) {
            try p.store(env, c[0], result);
            try p.charge(2);
        }
        return result;
    }
    return error.Unsupported;
}

fn charRange(s: []const u8, index: Range) Value {
    if (index.lo >= s.len) return Value.integer(0);
    return .{ .number = .{ .lo = if (index.hi < s.len) 1 else 0, .hi = 127 } };
}

fn truth(r: Range, invert: bool) Range {
    const yes = r.lo > 0 or r.hi < 0;
    const no = r.lo == 0 and r.hi == 0;
    if (yes) return Range.exact(@intFromBool(!invert));
    if (no) return Range.exact(@intFromBool(invert));
    return .{ .lo = 0, .hi = 1 };
}

fn fit(node: V, r: Range) Error!Range {
    const t = m.integerType(m.typ(node)) orelse return error.Unsupported;
    if (t.fits(r)) return r;
    if (t.signed) return error.Unsupported;
    return t.range();
}

fn binary(node: V, operator: []const u8, x: Range, y: Range) Error!Range {
    if (std.mem.eql(u8, operator, "+")) return fit(node, .{ .lo = x.lo + y.lo, .hi = x.hi + y.hi });
    if (std.mem.eql(u8, operator, "-")) return fit(node, .{ .lo = x.lo - y.hi, .hi = x.hi - y.lo });
    if (std.mem.eql(u8, operator, "*")) {
        const t = m.integerType(m.typ(node)) orelse return error.Unsupported;
        const pairs = [_][2]i128{ .{ x.lo, y.lo }, .{ x.lo, y.hi }, .{ x.hi, y.lo }, .{ x.hi, y.hi } };
        var r: Range = .{ .lo = std.math.maxInt(i128), .hi = std.math.minInt(i128) };
        for (pairs) |pair| {
            const n = std.math.mul(i128, pair[0], pair[1]) catch {
                if (t.signed) return error.Unsupported;
                return t.range();
            };
            r.lo = @min(r.lo, n);
            r.hi = @max(r.hi, n);
        }
        return fit(node, r);
    }
    if (std.mem.eql(u8, operator, "/") or std.mem.eql(u8, operator, "%")) {
        if (y.lo <= 0 and y.hi >= 0) return error.Unsupported;
        const t = m.integerType(m.typ(node)) orelse return error.Unsupported;
        if (x.lo == t.range().lo and t.signed and y.lo <= -1 and y.hi >= -1) return error.Unsupported;
        if (std.mem.eql(u8, operator, "%")) {
            const bound = @max(@abs(y.lo), @abs(y.hi)) - 1;
            return .{ .lo = if (x.lo < 0) -@as(i128, @intCast(bound)) else 0, .hi = if (x.hi > 0) @intCast(bound) else 0 };
        }
        const values = [_]i128{ @divTrunc(x.lo, y.lo), @divTrunc(x.lo, y.hi), @divTrunc(x.hi, y.lo), @divTrunc(x.hi, y.hi) };
        var r = Range.exact(values[0]);
        for (values[1..]) |n| r = r.join(Range.exact(n));
        return fit(node, r);
    }
    var yes = false;
    var no = false;
    if (std.mem.eql(u8, operator, "<")) {
        yes = x.hi < y.lo;
        no = x.lo >= y.hi;
    } else if (std.mem.eql(u8, operator, "<=")) {
        yes = x.hi <= y.lo;
        no = x.lo > y.hi;
    } else if (std.mem.eql(u8, operator, ">")) {
        yes = x.lo > y.hi;
        no = x.hi <= y.lo;
    } else if (std.mem.eql(u8, operator, ">=")) {
        yes = x.lo >= y.hi;
        no = x.hi < y.lo;
    } else if (std.mem.eql(u8, operator, "==") or std.mem.eql(u8, operator, "!=")) {
        yes = x.lo == x.hi and y.lo == y.hi and x.lo == y.lo;
        no = x.hi < y.lo or y.hi < x.lo;
        if (std.mem.eql(u8, operator, "!=")) std.mem.swap(bool, &yes, &no);
    } else return error.Unsupported;
    return if (yes) Range.exact(1) else if (no) Range.exact(0) else .{ .lo = 0, .hi = 1 };
}

pub fn refine(env: *Env, condition: V, positive: bool) void {
    const node = m.strip(condition);
    const c = m.children(node);
    if (m.is(node, "UnaryOperator") and m.op(node, "!") and c.len == 1) return refine(env, c[0], !positive);
    if (m.is(node, "DeclRefExpr")) {
        const b = env.find(safeRef(env, condition)) orelse return;
        var r = b.value.number orelse return;
        if (!positive and r.lo <= 0 and r.hi >= 0) r = Range.exact(0) else if (positive and r.lo == 0 and r.hi > 0) r.lo = 1;
        b.value.number = r;
        return;
    }
    if (!m.is(node, "BinaryOperator") or c.len != 2) return;
    const left = env.find(safeRef(env, c[0]));
    const right = env.find(safeRef(env, c[1]));
    const x = if (left) |b| b.value.number else literal(c[0]);
    const y = if (right) |b| b.value.number else literal(c[1]);
    if (x == null or y == null) return;
    var a = x.?;
    var b = y.?;
    var operator = m.text(m.field(node, "opcode"));
    if (!positive) {
        const pairs = [_][2][]const u8{ .{ "<", ">=" }, .{ "<=", ">" }, .{ ">", "<=" }, .{ ">=", "<" }, .{ "==", "!=" }, .{ "!=", "==" } };
        for (pairs) |pair| if (std.mem.eql(u8, operator, pair[0])) {
            operator = pair[1];
            break;
        };
    }
    if (std.mem.eql(u8, operator, ">") or std.mem.eql(u8, operator, ">=")) {
        const delta: i128 = @intFromBool(std.mem.eql(u8, operator, ">"));
        a.lo = @max(a.lo, b.lo + delta);
        b.hi = @min(b.hi, a.hi - delta);
    } else if (std.mem.eql(u8, operator, "<") or std.mem.eql(u8, operator, "<=")) {
        const delta: i128 = @intFromBool(std.mem.eql(u8, operator, "<"));
        a.hi = @min(a.hi, b.hi - delta);
        b.lo = @max(b.lo, a.lo + delta);
    } else if (std.mem.eql(u8, operator, "==")) {
        a = .{ .lo = @max(a.lo, b.lo), .hi = @min(a.hi, b.hi) };
        b = a;
    } else if (std.mem.eql(u8, operator, "!=")) {
        if (b.lo == b.hi) {
            if (a.lo == b.lo) a.lo += 1;
            if (a.hi == b.hi) a.hi -= 1;
        }
        if (x.?.lo == x.?.hi) {
            if (b.lo == x.?.lo) b.lo += 1;
            if (b.hi == x.?.hi) b.hi -= 1;
        }
    }
    if (a.lo <= a.hi) {
        if (left) |binding| binding.value.number = a;
    }
    if (b.lo <= b.hi) {
        if (right) |binding| binding.value.number = b;
    }
}

pub fn safeRef(env: *Env, raw: V) []const u8 {
    const id = m.ref(raw);
    const binding = env.find(id) orelse return "";
    const r = binding.value.number orelse return "";
    return transparentRef(raw, r);
}

pub fn transparentRef(raw: V, r: Range) []const u8 {
    const id = m.ref(raw);
    var n = raw;
    for (0..64) |_| {
        if (m.is(n, "DeclRefExpr")) return id;
        const c = m.children(n);
        if (c.len != 1) return "";
        if (m.is(n, "ImplicitCastExpr") or m.is(n, "CStyleCastExpr")) {
            const t = m.integerType(m.typ(n)) orelse return "";
            if (!t.fits(r)) return "";
        } else if (!m.is(n, "ParenExpr")) return "";
        n = c[0];
    }
    return "";
}

fn literal(v: V) ?Range {
    const n = m.strip(v);
    if (!m.is(n, "IntegerLiteral") and !m.is(n, "CharacterLiteral")) return null;
    const r = Range.exact(m.integer(n) catch return null);
    var wrapper = v;
    for (0..64) |_| {
        if (m.is(wrapper, "IntegerLiteral") or m.is(wrapper, "CharacterLiteral")) return r;
        const c = m.children(wrapper);
        if (c.len != 1) return null;
        if (m.is(wrapper, "ImplicitCastExpr") or m.is(wrapper, "CStyleCastExpr")) {
            const t = m.integerType(m.typ(wrapper)) orelse return null;
            if (!t.fits(r)) return null;
        } else if (!m.is(wrapper, "ParenExpr") and !m.is(wrapper, "ConstantExpr")) return null;
        wrapper = c[0];
    }
    return null;
}
