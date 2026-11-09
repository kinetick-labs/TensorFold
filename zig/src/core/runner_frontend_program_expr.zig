//! Small integer expression grammar with interval proofs for every emitted C intermediate.
const std = @import("std");
const Invalid = error{Unsupported};
pub const Bounds = struct { lo: i64, hi: i64 };
const Op = enum { literal, variable, add, sub, mul, floor, mod, neg, power };
const Node = struct { op: Op, lhs: usize = 0, rhs: usize = 0, value: i64 = 0, name: []const u8 = "" };

pub fn fits(n: i64) bool {
    return n >= std.math.minInt(i32) and n <= std.math.maxInt(i32);
}

pub const Parser = struct {
    src: []const u8,
    pos: usize = 0,
    nodes: [128]Node = undefined,
    len: usize = 0,
    depth: usize = 0,
    variable_bounds: Bounds = .{ .lo = 0, .hi = 0 },

    fn space(p: *Parser) void {
        while (p.pos < p.src.len and p.src[p.pos] == ' ') p.pos += 1;
    }

    pub fn eat(p: *Parser, token: []const u8) bool {
        p.space();
        if (!std.mem.startsWith(u8, p.src[p.pos..], token)) return false;
        const end_pos = p.pos + token.len;
        if (nameChar(token[token.len - 1]) and end_pos < p.src.len and nameChar(p.src[end_pos])) return false;
        p.pos = end_pos;
        return true;
    }

    pub fn need(p: *Parser, token: []const u8) Invalid!void {
        if (!p.eat(token)) return error.Unsupported;
    }

    pub fn end(p: *Parser) Invalid!void {
        p.space();
        if (p.pos != p.src.len) return error.Unsupported;
    }

    pub fn identifier(p: *Parser) Invalid![]const u8 {
        p.space();
        const start = p.pos;
        if (start == p.src.len or !(std.ascii.isAlphabetic(p.src[start]) or p.src[start] == '_'))
            return error.Unsupported;
        while (p.pos < p.src.len and nameChar(p.src[p.pos])) p.pos += 1;
        return p.src[start..p.pos];
    }

    pub fn integer(p: *Parser) Invalid!i64 {
        const negative = p.eat("-");
        if (!negative) _ = p.eat("+");
        p.space();
        const start = p.pos;
        while (p.pos < p.src.len and std.ascii.isDigit(p.src[p.pos])) p.pos += 1;
        if (p.pos == start or (p.pos - start > 1 and p.src[start] == '0') or
            (p.pos < p.src.len and nameChar(p.src[p.pos]))) return error.Unsupported;
        const value = std.fmt.parseInt(i64, p.src[start..p.pos], 10) catch return error.Unsupported;
        return if (negative) -value else value;
    }

    fn node(p: *Parser, n: Node) Invalid!usize {
        if (p.len == p.nodes.len) return error.Unsupported;
        const id = p.len;
        p.nodes[id] = n;
        p.len += 1;
        return id;
    }

    pub fn expression(p: *Parser) Invalid!usize {
        if (p.depth >= 32) return error.Unsupported;
        p.depth += 1;
        defer p.depth -= 1;
        var lhs = try p.product();
        while (true) {
            const op: Op = if (p.eat("+")) .add else if (p.eat("-")) .sub else break;
            lhs = try p.node(.{ .op = op, .lhs = lhs, .rhs = try p.product() });
        }
        return lhs;
    }

    fn product(p: *Parser) Invalid!usize {
        var lhs = try p.unary();
        while (true) {
            p.space();
            if (std.mem.startsWith(u8, p.src[p.pos..], "**")) break;
            const op: Op = if (p.eat("*")) .mul else if (p.eat("//")) .floor else if (p.eat("%")) .mod else break;
            lhs = try p.node(.{ .op = op, .lhs = lhs, .rhs = try p.unary() });
        }
        return lhs;
    }

    fn unary(p: *Parser) Invalid!usize {
        if (p.depth >= 32) return error.Unsupported;
        p.depth += 1;
        defer p.depth -= 1;
        if (p.eat("-")) return p.node(.{ .op = .neg, .lhs = try p.unary() });
        if (p.eat("+")) return p.unary();
        var base = try p.atom();
        if (p.eat("**")) {
            const exponent = try p.integer();
            if (exponent < 0 or exponent > 8) return error.Unsupported;
            base = try p.node(.{ .op = .power, .lhs = base, .value = exponent });
        }
        return base;
    }

    fn atom(p: *Parser) Invalid!usize {
        if (p.eat("(")) {
            const id = try p.expression();
            try p.need(")");
            return id;
        }
        p.space();
        if (p.pos < p.src.len and std.ascii.isDigit(p.src[p.pos])) {
            const value = try p.integer();
            if (!fits(value)) return error.Unsupported;
            return p.node(.{ .op = .literal, .value = value });
        }
        return p.node(.{ .op = .variable, .name = try p.identifier() });
    }

    pub fn bind(p: *Parser, variable: []const u8, interval: Bounds) Invalid!void {
        for (p.nodes[0..p.len]) |n| {
            if (n.op == .variable and !std.mem.eql(u8, variable, n.name)) return error.Unsupported;
        }
        p.variable_bounds = interval;
    }

    pub fn bounds(p: *const Parser, id: usize) Invalid!Bounds {
        const n = p.nodes[id];
        const result: Bounds = switch (n.op) {
            .literal => .{ .lo = n.value, .hi = n.value },
            .variable => p.variable_bounds,
            .neg => blk: {
                const b = try p.bounds(n.lhs);
                break :blk .{ .lo = -b.hi, .hi = -b.lo };
            },
            .add, .sub, .mul, .floor, .mod => blk: {
                const a = try p.bounds(n.lhs);
                const b = try p.bounds(n.rhs);
                break :blk switch (n.op) {
                    .add => .{ .lo = a.lo + b.lo, .hi = a.hi + b.hi },
                    .sub => .{ .lo = a.lo - b.hi, .hi = a.hi - b.lo },
                    .mul => multiply(a, b),
                    .floor, .mod => try divide(a, b, n.op == .mod),
                    else => unreachable,
                };
            },
            .power => blk: {
                const b = try p.bounds(n.lhs);
                var r: Bounds = .{ .lo = 1, .hi = 1 };
                var i: i64 = 0;
                while (i < n.value) : (i += 1) {
                    r = multiply(r, b);
                    if (!fits(r.lo) or !fits(r.hi)) return error.Unsupported;
                }
                break :blk r;
            },
        };
        if (!fits(result.lo) or !fits(result.hi)) return error.Unsupported;
        return result;
    }

    pub fn emit(p: *const Parser, a: std.mem.Allocator, out: *std.ArrayList(u8)) std.mem.Allocator.Error!void {
        try out.appendSlice(a, "static volatile int tf_literals[]={");
        var inputs: usize = 0;
        for (p.nodes[0..p.len]) |n| {
            if (n.op == .literal or n.op == .power) {
                try out.print(a, "{d},", .{n.value});
                inputs += 1;
            }
        }
        if (inputs == 0) try out.appendSlice(a, "0");
        try out.appendSlice(a, "};\n");
        inputs = 0;
        for (p.nodes[0..p.len], 0..) |n, id| {
            try out.print(a, "int tf_e{d} = ", .{id});
            switch (n.op) {
                .literal => {
                    try out.print(a, "tf_literals[{d}]", .{inputs});
                    inputs += 1;
                },
                .variable => try out.appendSlice(a, "tf_i"),
                .neg => try out.print(a, "-tf_e{d}", .{n.lhs}),
                .add, .sub, .mul => try out.print(a, "tf_e{d} {s} tf_e{d}", .{
                    n.lhs,
                    switch (n.op) {
                        .add => "+",
                        .sub => "-",
                        else => "*",
                    },
                    n.rhs,
                }),
                .floor, .mod => {
                    if (n.op == .floor) {
                        try out.print(a, "tf_e{0d} / tf_e{1d} - ((tf_e{0d} % tf_e{1d} != 0) && ((tf_e{0d} < 0) != (tf_e{1d} < 0)))", .{ n.lhs, n.rhs });
                    } else {
                        try out.print(a, "tf_e{0d} % tf_e{1d} + (((tf_e{0d} % tf_e{1d} != 0) && ((tf_e{0d} < 0) != (tf_e{1d} < 0))) ? tf_e{1d} : 0)", .{ n.lhs, n.rhs });
                    }
                },
                .power => {
                    try out.print(a, "1;\nint tf_exp{d}=tf_literals[{d}];\nfor(int tf_p{d}=0;tf_p{d}<tf_exp{d};++tf_p{d}) tf_e{d} *= tf_e{d}", .{
                        id, inputs, id, id, id, id, id, n.lhs,
                    });
                    inputs += 1;
                },
            }
            try out.print(a, ";\n__asm__ volatile(\"\" : \"+r\"(tf_e{d}));\n", .{id});
        }
    }
};

fn nameChar(c: u8) bool {
    return std.ascii.isAlphanumeric(c) or c == '_';
}

fn multiply(a: Bounds, b: Bounds) Bounds {
    const products = [_]i64{ a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi };
    return .{ .lo = std.mem.min(i64, &products), .hi = std.mem.max(i64, &products) };
}

test "integer proof catches intermediate overflow even when final value is small" {
    var p: Parser = .{ .src = "(i*1000000)*1000000 - (i*1000000)*1000000" };
    const root = try p.expression();
    try p.bind("i", .{ .lo = 0, .hi = 9 });
    try std.testing.expectError(error.Unsupported, p.bounds(root));
}

test "Python unary minus binds below exponentiation" {
    var p: Parser = .{ .src = "-i**2" };
    const root = try p.expression();
    try p.bind("i", .{ .lo = 3, .hi = 3 });
    const b = try p.bounds(root);
    try std.testing.expectEqual(@as(i64, -9), b.lo);
    try std.testing.expectEqual(@as(i64, -9), b.hi);
}

test "sum squares semantics are 285 with wrong-operator negative control" {
    var squares: Parser = .{ .src = "i*i" };
    const root = try squares.expression();
    var wrong: Parser = .{ .src = "i+i" };
    const wrong_root = try wrong.expression();
    var sum: i64 = 0;
    var wrong_sum: i64 = 0;
    for (0..10) |i| {
        const value: i64 = @intCast(i);
        const interval: Bounds = .{ .lo = value, .hi = value };
        try squares.bind("i", interval);
        try wrong.bind("i", interval);
        const result = try squares.bounds(root);
        try std.testing.expectEqual(result.lo, result.hi);
        sum += result.lo;
        wrong_sum += (try wrong.bounds(wrong_root)).lo;
    }
    try std.testing.expectEqual(@as(i64, 285), sum);
    try std.testing.expect(wrong_sum != sum);
}

fn divide(a: Bounds, b: Bounds, remainder: bool) Invalid!Bounds {
    if (b.lo <= 0 and b.hi >= 0) return error.Unsupported;
    if (a.lo == std.math.minInt(i32) and b.lo <= -1 and b.hi >= -1) return error.Unsupported;
    if (remainder) {
        if (a.lo == a.hi and b.lo == b.hi) {
            const value = a.lo - @divFloor(a.lo, b.lo) * b.lo;
            return .{ .lo = value, .hi = value };
        }
        return if (b.lo > 0) .{ .lo = 0, .hi = b.hi - 1 } else .{ .lo = b.lo + 1, .hi = 0 };
    }
    const corners = [_]i64{ @divFloor(a.lo, b.lo), @divFloor(a.lo, b.hi), @divFloor(a.hi, b.lo), @divFloor(a.hi, b.hi) };
    return .{ .lo = std.mem.min(i64, &corners), .hi = std.mem.max(i64, &corners) };
}

test "Python division and remainder preserve negative operand semantics" {
    for ([_][3]i64{ .{ -7, 3, -3 }, .{ 7, -3, -3 }, .{ -7, -3, 2 }, .{ 7, 3, 2 } }) |case| {
        const quotient = try divide(.{ .lo = case[0], .hi = case[0] }, .{ .lo = case[1], .hi = case[1] }, false);
        try std.testing.expectEqual(case[2], quotient.lo);
        const remainder = try divide(.{ .lo = case[0], .hi = case[0] }, .{ .lo = case[1], .hi = case[1] }, true);
        try std.testing.expectEqual(case[0] - case[2] * case[1], remainder.lo);
    }
    try std.testing.expectError(error.Unsupported, divide(.{ .lo = -2, .hi = 2 }, .{ .lo = -1, .hi = 1 }, false));
    try std.testing.expectError(error.Unsupported, divide(.{ .lo = -2147483648, .hi = 0 }, .{ .lo = -1, .hi = -1 }, true));
}
