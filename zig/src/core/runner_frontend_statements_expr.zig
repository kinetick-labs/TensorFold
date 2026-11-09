//! Integer syntax and bounded proof state for statement lowering; no source is executed on the host.
const std = @import("std");
pub const Invalid = error{Unsupported};
pub const Op = enum { literal, variable, element, list_zero, add, sub, mul, neg, floor, modulo, power };
pub const Node = struct { op: Op, lhs: usize = 0, rhs: usize = 0, value: i64 = 0 };
pub const Symbol = struct { name: []const u8, list_len: ?usize = null };
pub const State = struct {
    defined: [32]bool = @splat(false),
    values: [32][32]i64 = @splat(@splat(0)),
    fuel: usize = 2_000_000,
    estimated_steps: usize = 200,
};

pub fn fits(n: i64) bool {
    return n >= std.math.minInt(i32) and n <= std.math.maxInt(i32);
}

pub const Graph = struct {
    nodes: [512]Node = undefined,
    len: usize = 0,
    symbols: [32]Symbol = undefined,
    symbols_len: usize = 0,

    pub fn find(g: *const Graph, name: []const u8) ?usize {
        for (g.symbols[0..g.symbols_len], 0..) |s, i| if (std.mem.eql(u8, s.name, name)) return i;
        return null;
    }

    pub fn bind(g: *Graph, name: []const u8, list_len: ?usize) Invalid!usize {
        if (!validName(name)) return error.Unsupported;
        if (g.find(name)) |id| {
            if (g.symbols[id].list_len != list_len) return error.Unsupported;
            return id;
        }
        if (g.symbols_len == g.symbols.len) return error.Unsupported;
        const id = g.symbols_len;
        g.symbols[id] = .{ .name = name, .list_len = list_len };
        g.symbols_len += 1;
        return id;
    }

    pub fn node(g: *Graph, n: Node) Invalid!usize {
        if (g.len == g.nodes.len) return error.Unsupported;
        const id = g.len;
        g.nodes[id] = n;
        g.len += 1;
        return id;
    }

    pub fn eval(g: *const Graph, id: usize, s: *State) Invalid!i64 {
        if (s.fuel == 0) return error.Unsupported;
        s.fuel -= 1;
        s.estimated_steps +|= 10;
        const n = g.nodes[id];
        const result: i64 = switch (n.op) {
            .literal => n.value,
            .variable, .element, .list_zero => blk: {
                if (!s.defined[n.lhs]) return error.Unsupported;
                break :blk if (n.op == .list_zero) 0 else s.values[n.lhs][if (n.op == .element) n.rhs else 0];
            },
            .neg => -(try g.eval(n.lhs, s)),
            else => blk: {
                const x = try g.eval(n.lhs, s);
                const y = try g.eval(n.rhs, s);
                break :blk switch (n.op) {
                    .add => x + y,
                    .sub => x - y,
                    .mul => x * y,
                    .floor, .modulo => div: {
                        if (y == 0 or (x == std.math.minInt(i32) and y == -1)) return error.Unsupported;
                        break :div if (n.op == .floor) @divFloor(x, y) else x - @divFloor(x, y) * y;
                    },
                    .power => pow: {
                        if (y < 0 or y > 31) return error.Unsupported;
                        s.estimated_steps +|= @as(usize, @intCast(y)) * 6;
                        var r: i64 = 1;
                        var i: i64 = 0;
                        while (i < y) : (i += 1) {
                            r *= x;
                            if (!fits(r)) return error.Unsupported;
                        }
                        break :pow r;
                    },
                    else => unreachable,
                };
            },
        };
        if (!fits(result)) return error.Unsupported;
        return result;
    }

    pub fn emit(g: *const Graph, a: std.mem.Allocator, out: *std.ArrayList(u8), start: usize, end: usize) !void {
        try out.print(a, "static volatile int tf_inputs{d}[]={{", .{start});
        var inputs: usize = 0;
        for (g.nodes[start..end]) |n| {
            if (n.op == .literal) {
                try out.print(a, "{d},", .{n.value});
                inputs += 1;
            }
        }
        if (inputs == 0) try out.appendSlice(a, "0");
        try out.appendSlice(a, "};\n");
        inputs = 0;
        for (g.nodes[start..end], start..) |n, id| {
            try out.print(a, "int tf_e{d} = ", .{id});
            switch (n.op) {
                .literal => {
                    try out.print(a, "tf_inputs{d}[{d}]", .{ start, inputs });
                    inputs += 1;
                },
                .variable => try out.print(a, "tf_v{d}", .{n.lhs}),
                .element => try out.print(a, "tf_v{d}[{d}]", .{ n.lhs, n.rhs }),
                .list_zero => try out.appendSlice(a, "0"),
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
                .floor, .modulo => {
                    try out.print(a, "tf_e{d} {s} tf_e{d};\n", .{ n.lhs, if (n.op == .floor) "/" else "%", n.rhs });
                    if (n.op == .floor) {
                        try out.print(a, "if (tf_e{d} % tf_e{d} != 0 && ((tf_e{d} < 0) != (tf_e{d} < 0))) --tf_e{d}", .{
                            n.lhs, n.rhs, n.lhs, n.rhs, id,
                        });
                    } else {
                        try out.print(a, "if (tf_e{d} != 0 && ((tf_e{d} < 0) != (tf_e{d} < 0))) tf_e{d} += tf_e{d}", .{
                            id, id, n.rhs, id, n.rhs,
                        });
                    }
                },
                .power => {
                    try out.print(a, "1;\nfor (int tf_p{d} = 0; tf_p{d} < tf_e{d}; ++tf_p{d}) tf_e{d} *= tf_e{d}", .{
                        id, id, n.rhs, id, id, n.lhs,
                    });
                },
            }
            try out.print(a, ";\n__asm__ volatile(\"\" : \"+r\"(tf_e{d}));\n", .{id});
        }
    }
};

pub const Parser = struct {
    src: []const u8,
    g: *Graph,
    pos: usize = 0,
    depth: usize = 0,

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

    pub fn expression(p: *Parser) Invalid!usize {
        if (p.depth >= 32) return error.Unsupported;
        p.depth += 1;
        defer p.depth -= 1;
        var lhs = try p.product();
        while (true) {
            const op: Op = if (p.eat("+")) .add else if (p.eat("-")) .sub else break;
            lhs = try p.g.node(.{ .op = op, .lhs = lhs, .rhs = try p.product() });
        }
        return lhs;
    }

    fn product(p: *Parser) Invalid!usize {
        var lhs = try p.unary();
        while (true) {
            p.space();
            if (std.mem.startsWith(u8, p.src[p.pos..], "**")) break;
            const op: Op = if (p.eat("*")) .mul else if (p.eat("//")) .floor else if (p.eat("%")) .modulo else break;
            lhs = try p.g.node(.{ .op = op, .lhs = lhs, .rhs = try p.unary() });
        }
        return lhs;
    }

    fn unary(p: *Parser) Invalid!usize {
        if (p.depth >= 32) return error.Unsupported;
        p.depth += 1;
        defer p.depth -= 1;
        if (p.eat("-")) return p.g.node(.{ .op = .neg, .lhs = try p.unary() });
        if (p.eat("+")) return p.unary();
        var root = try p.atom();
        if (p.eat("**")) root = try p.g.node(.{ .op = .power, .lhs = root, .rhs = try p.unary() });
        return root;
    }

    fn atom(p: *Parser) Invalid!usize {
        if (p.eat("(")) {
            const root = try p.expression();
            try p.need(")");
            return root;
        }
        if (p.eat("sum")) {
            try p.need("(");
            var root = try p.g.node(.{ .op = .literal });
            if (p.eat("[")) {
                var count: usize = 0;
                if (!p.eat("]")) while (true) {
                    if (count == 32) return error.Unsupported;
                    count += 1;
                    root = try p.g.node(.{ .op = .add, .lhs = root, .rhs = try p.expression() });
                    if (p.eat("]")) break;
                    try p.need(",");
                    if (p.eat("]")) break;
                };
            } else {
                const id = p.g.find(try p.identifier()) orelse return error.Unsupported;
                const count = p.g.symbols[id].list_len orelse return error.Unsupported;
                // A zero-element list still needs a defined binding at proof time.
                root = try p.g.node(.{ .op = .list_zero, .lhs = id });
                for (0..count) |i| {
                    const element = try p.g.node(.{ .op = .element, .lhs = id, .rhs = i });
                    root = try p.g.node(.{ .op = .add, .lhs = root, .rhs = element });
                }
            }
            try p.need(")");
            return root;
        }
        p.space();
        if (p.pos < p.src.len and std.ascii.isDigit(p.src[p.pos])) {
            const start = p.pos;
            while (p.pos < p.src.len and std.ascii.isDigit(p.src[p.pos])) p.pos += 1;
            const text = p.src[start..p.pos];
            if (text.len > 1 and text[0] == '0') return error.Unsupported;
            const value = std.fmt.parseInt(i64, text, 10) catch return error.Unsupported;
            if (!fits(value)) return error.Unsupported;
            return p.g.node(.{ .op = .literal, .value = value });
        }
        const id = p.g.find(try p.identifier()) orelse return error.Unsupported;
        if (p.g.symbols[id].list_len != null) return error.Unsupported;
        return p.g.node(.{ .op = .variable, .lhs = id });
    }
};

fn nameChar(c: u8) bool {
    return std.ascii.isAlphanumeric(c) or c == '_';
}

fn validName(s: []const u8) bool {
    const reserved = [_][]const u8{
        "print",        "sum",    "range",   "for",  "in",     "True",   "False", "None",     "and",  "or",    "not",    "if",        "else",
        "def",          "class",  "import",  "from", "return", "while",  "break", "continue", "pass", "yield", "lambda", "global",    "nonlocal",
        "try",          "except", "finally", "with", "as",     "assert", "del",   "raise",    "is",   "async", "await",  "elif",      "__debug__",
        "pow",          "divmod", "int",     "str",  "len",    "map",    "all",   "abs",      "math", "gcd",   "isqrt",  "factorial", "prod",
        "__builtins__",
    };
    for (reserved) |name| if (std.mem.eql(u8, name, s)) return false;
    return true;
}

test "signed quotient and remainder match floor semantics with truncation controls" {
    const cases = [_]struct { source: []const u8, expected: i64 }{
        .{ .source = "-7//3", .expected = -3 },
        .{ .source = "7//-3", .expected = -3 },
        .{ .source = "-7//-3", .expected = 2 },
        .{ .source = "7//3", .expected = 2 },
        .{ .source = "-7%3", .expected = 2 },
        .{ .source = "7%-3", .expected = -2 },
        .{ .source = "-7%-3", .expected = -1 },
        .{ .source = "7%3", .expected = 1 },
        .{ .source = "-6//3", .expected = -2 },
        .{ .source = "-6%3", .expected = 0 },
        .{ .source = "-2**2", .expected = -4 },
        .{ .source = "(-2)**2", .expected = 4 },
        .{ .source = "2**3**2", .expected = 512 },
    };
    for (cases) |case| {
        var g: Graph = .{};
        var parser: Parser = .{ .src = case.source, .g = &g };
        const root = try parser.expression();
        try parser.end();
        var state: State = .{};
        try std.testing.expectEqual(case.expected, try g.eval(root, &state));
    }
    try std.testing.expect(@divTrunc(@as(i64, -7), 3) != cases[0].expected);
    try std.testing.expect(@rem(@as(i64, -7), 3) != cases[4].expected);
}
