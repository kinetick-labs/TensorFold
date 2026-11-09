const std = @import("std");

pub const Invalid = error{Unsupported};
pub const limit: i512 = std.math.pow(i512, 10, 62);
pub const Value = struct { n: i512, d: i512 = 1 };
pub const Kind = enum { literal, neg, add, sub, mul, div, floor, pow, percent };
pub const Node = struct {
    kind: Kind,
    value: Value,
    left: usize = 0,
    right: usize = 0,
    text: []const u8 = "",
};

pub fn magnitude(n: i512) i512 {
    return if (n < 0) -n else n;
}

pub fn checked(n: i512, d: i512) Invalid!Value {
    if (d == 0) return error.Unsupported;
    var x = magnitude(n);
    var y = magnitude(d);
    while (y != 0) {
        const rem = @mod(x, y);
        x = y;
        y = rem;
    }
    const sign: i512 = if (d < 0) -1 else 1;
    const v = Value{ .n = @divTrunc(n, x) * sign, .d = @divTrunc(d, x) * sign };
    if (magnitude(v.n) >= limit or v.d >= limit) return error.Unsupported;
    return v;
}

fn product(a: Value, b: Value) Invalid!Value {
    return checked(a.n * b.n, a.d * b.d);
}

pub const Parser = struct {
    source: []const u8,
    pos: usize = 0,
    nodes: [32]Node = undefined,
    count: usize = 0,
    decimal_context: bool = false,
    exact_syntax: bool = false,

    pub fn parse(self: *Parser) Invalid!usize {
        self.space();
        if (self.pos == self.source.len or self.source[self.pos] == '+') return error.Unsupported;
        const root = try self.expr(0);
        self.space();
        if (self.pos != self.source.len) return error.Unsupported;
        return root;
    }

    fn space(self: *Parser) void {
        while (self.pos < self.source.len and (self.source[self.pos] == ' ' or self.source[self.pos] == '\t'))
            self.pos += 1;
    }

    fn take(self: *Parser, s: []const u8) bool {
        self.space();
        if (!std.mem.startsWith(u8, self.source[self.pos..], s)) return false;
        self.pos += s.len;
        return true;
    }

    fn append(self: *Parser, node: Node) Invalid!usize {
        if (self.count == self.nodes.len) return error.Unsupported;
        const index = self.count;
        self.nodes[index] = node;
        self.count += 1;
        return index;
    }

    fn binary(self: *Parser, kind: Kind, left: usize, right: usize) Invalid!usize {
        const a = self.nodes[left].value;
        const b = self.nodes[right].value;
        const value: Value = switch (kind) {
            .add => try checked(a.n * b.d + b.n * a.d, a.d * b.d),
            .sub => try checked(a.n * b.d - b.n * a.d, a.d * b.d),
            .mul => try product(a, b),
            .div => try checked(a.n * b.d, a.d * b.n),
            .floor => blk: {
                if (a.d != 1 or b.d != 1 or b.n == 0) return error.Unsupported;
                break :blk try checked(@divFloor(a.n, b.n), 1);
            },
            .pow => blk: {
                if (b.d != 1 or b.n < 0 or b.n > 999) return error.Unsupported;
                var exponent = b.n;
                var base = a;
                var result = Value{ .n = 1 };
                while (exponent != 0) {
                    if (@mod(exponent, 2) != 0) result = try product(result, base);
                    exponent = @divTrunc(exponent, 2);
                    if (exponent != 0) base = try product(base, base);
                }
                break :blk result;
            },
            else => return error.Unsupported,
        };
        return self.append(.{ .kind = kind, .left = left, .right = right, .value = value });
    }

    fn expr(self: *Parser, depth: usize) Invalid!usize {
        if (depth > 10) return error.Unsupported;
        var left = try self.term(depth);
        while (true) {
            const kind: Kind = if (self.take("+")) .add else if (self.take("-")) .sub else break;
            left = try self.binary(kind, left, try self.term(depth));
        }
        return left;
    }

    fn term(self: *Parser, depth: usize) Invalid!usize {
        var left = try self.factor(depth);
        while (true) {
            self.space();
            if (std.mem.startsWith(u8, self.source[self.pos..], "**")) return error.Unsupported;
            const kind: Kind = if (self.take("//")) .floor else if (self.take("/")) .div else if (self.take("*")) .mul else break;
            if (kind == .div) self.exact_syntax = true;
            left = try self.binary(kind, left, try self.factor(depth));
        }
        return left;
    }

    fn factor(self: *Parser, depth: usize) Invalid!usize {
        if (depth > 10) return error.Unsupported;
        if (self.take("+")) return self.factor(depth + 1);
        if (self.take("-")) {
            const child = try self.factor(depth + 1);
            const v = self.nodes[child].value;
            return self.append(.{ .kind = .neg, .left = child, .value = .{ .n = -v.n, .d = v.d } });
        }
        var left = try self.atom(depth);
        if (self.take("%")) {
            self.decimal_context = true;
            self.exact_syntax = true;
            const v = self.nodes[left].value;
            left = try self.append(.{ .kind = .percent, .left = left, .value = try checked(v.n, v.d * 100) });
        }
        if (self.take("**")) {
            left = try self.binary(.pow, left, try self.factor(depth + 1));
        }
        return left;
    }

    fn atom(self: *Parser, depth: usize) Invalid!usize {
        if (self.take("(")) {
            const child = try self.expr(depth + 1);
            if (!self.take(")")) return error.Unsupported;
            return child;
        }
        self.space();
        const start = self.pos;
        var n: i512 = 0;
        var digits: usize = 0;
        var scale: usize = 0;
        var dot = false;
        while (self.pos < self.source.len) {
            const c = self.source[self.pos];
            if (std.ascii.isDigit(c)) {
                digits += 1;
                if (digits > 62) return error.Unsupported;
                n = n * 10 + c - '0';
                if (dot) scale += 1;
            } else if (c == '.' and !dot) {
                dot = true;
            } else break;
            self.pos += 1;
        }
        if (digits == 0 or (dot and scale == 0)) return error.Unsupported;
        if (dot) {
            self.decimal_context = true;
            self.exact_syntax = true;
        }
        return self.append(.{
            .kind = .literal,
            .text = self.source[start..self.pos],
            .value = try checked(n, std.math.pow(i512, 10, scale)),
        });
    }
};

test "inline proof respects precedence right association and percentage scale" {
    const cases = [_]struct { code: []const u8, n: i512, d: i512 = 1 }{
        .{ .code = "2**3**2", .n = 512 },
        .{ .code = "-2**2", .n = -4 },
        .{ .code = "(-2)**2", .n = 4 },
        .{ .code = "50%", .n = 1, .d = 2 },
        .{ .code = "-.5/2", .n = -1, .d = 4 },
        .{ .code = "-7//3", .n = -3 },
    };
    for (cases) |case| {
        var parser = Parser{ .source = case.code };
        const root = try parser.parse();
        try std.testing.expectEqual(case.n, parser.nodes[root].value.n);
        try std.testing.expectEqual(case.d, parser.nodes[root].value.d);
    }
}
