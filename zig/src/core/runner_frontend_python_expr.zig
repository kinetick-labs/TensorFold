const std = @import("std");
const ast = @import("runner_frontend_python_ast.zig");

pub fn parse(program: *ast.Program, source: []const u8) ast.Error!usize {
    if (!std.unicode.utf8ValidateSlice(source)) return error.Unsupported;
    var p: Parser = .{ .program = program, .source = source };
    const root = try p.expression();
    try p.end();
    try boundedDepth(program, root, 0);
    return root;
}

pub fn parseArgs(program: *ast.Program, source: []const u8) ast.Error![]const usize {
    if (!std.unicode.utf8ValidateSlice(source)) return error.Unsupported;
    var p: Parser = .{ .program = program, .source = source };
    var args: std.ArrayList(usize) = .empty;
    p.space();
    if (p.pos == p.source.len) return &.{};
    while (true) {
        if (args.items.len >= 128) return error.Unsupported;
        const root = try p.expression();
        try boundedDepth(program, root, 0);
        try args.append(program.a, root);
        if (!p.eat(",")) break;
        p.space();
        if (p.pos == p.source.len) break;
    }
    try p.end();
    return args.toOwnedSlice(program.a);
}

const Parser = struct {
    program: *ast.Program,
    source: []const u8,
    pos: usize = 0,
    depth: usize = 0,

    fn space(p: *Parser) void {
        while (p.pos < p.source.len and p.source[p.pos] == ' ') p.pos += 1;
    }

    fn eat(p: *Parser, token: []const u8) bool {
        p.space();
        if (!std.mem.startsWith(u8, p.source[p.pos..], token)) return false;
        const end_pos = p.pos + token.len;
        if (nameChar(token[token.len - 1]) and end_pos < p.source.len and nameChar(p.source[end_pos])) return false;
        p.pos = end_pos;
        return true;
    }

    fn need(p: *Parser, token: []const u8) ast.Invalid!void {
        if (!p.eat(token)) return error.Unsupported;
    }

    fn end(p: *Parser) ast.Invalid!void {
        p.space();
        if (p.pos != p.source.len) return error.Unsupported;
    }

    fn identifier(p: *Parser) ast.Invalid![]const u8 {
        p.space();
        const start = p.pos;
        if (start == p.source.len or !(std.ascii.isAlphabetic(p.source[start]) or p.source[start] == '_'))
            return error.Unsupported;
        p.pos += 1;
        while (p.pos < p.source.len and nameChar(p.source[p.pos])) p.pos += 1;
        return p.source[start..p.pos];
    }

    fn expression(p: *Parser) ast.Error!usize {
        if (p.depth >= 8) return error.Unsupported;
        p.depth += 1;
        defer p.depth -= 1;
        var root = try p.conjunction();
        while (p.eat("or")) root = try p.binary(.logical_or, root, try p.conjunction());
        return root;
    }

    fn conjunction(p: *Parser) ast.Error!usize {
        var root = try p.inversion();
        while (p.eat("and")) root = try p.binary(.logical_and, root, try p.inversion());
        return root;
    }

    fn inversion(p: *Parser) ast.Error!usize {
        if (p.eat("not")) {
            if (p.depth >= 8) return error.Unsupported;
            p.depth += 1;
            defer p.depth -= 1;
            return p.program.node(.{ .tag = .logical_not, .lhs = try p.inversion() });
        }
        var root = try p.additive();
        const tag: ast.Tag = if (p.eat("==")) .eq else if (p.eat("!=")) .ne else if (p.eat("<=")) .le else if (p.eat(">=")) .ge else if (p.eat("<")) .lt else if (p.eat(">")) .gt else return root;
        root = try p.binary(tag, root, try p.additive());
        return root;
    }

    fn additive(p: *Parser) ast.Error!usize {
        var root = try p.product();
        while (true) {
            const tag: ast.Tag = if (p.eat("+")) .add else if (p.eat("-")) .sub else break;
            root = try p.binary(tag, root, try p.product());
        }
        return root;
    }

    fn product(p: *Parser) ast.Error!usize {
        var root = try p.unary();
        while (true) {
            const tag: ast.Tag = if (p.eat("//")) .floor else if (p.eat("%")) .modulo else if (p.eat("*")) .mul else break;
            root = try p.binary(tag, root, try p.unary());
        }
        return root;
    }

    fn unary(p: *Parser) ast.Error!usize {
        if (p.eat("-")) {
            if (p.depth >= 8) return error.Unsupported;
            p.depth += 1;
            defer p.depth -= 1;
            return p.program.node(.{ .tag = .neg, .lhs = try p.unary() });
        }
        if (p.eat("+")) {
            if (p.depth >= 8) return error.Unsupported;
            p.depth += 1;
            defer p.depth -= 1;
            const zero = try p.program.node(.{ .tag = .integer });
            return p.binary(.add, zero, try p.unary());
        }
        var root = try p.atom();
        if (p.eat("[")) {
            root = try p.program.node(.{ .tag = .index, .lhs = root, .rhs = try p.expression() });
            try p.need("]");
        }
        return root;
    }

    fn binary(p: *Parser, tag: ast.Tag, lhs: usize, rhs: usize) ast.Error!usize {
        return p.program.node(.{ .tag = tag, .lhs = lhs, .rhs = rhs });
    }

    fn atom(p: *Parser) ast.Error!usize {
        if (p.eat("(")) {
            const root = try p.expression();
            try p.need(")");
            return root;
        }
        if (p.eat("[")) return p.sequence();
        if (p.eat("True")) return p.program.node(.{ .tag = .boolean, .value = 1 });
        if (p.eat("False")) return p.program.node(.{ .tag = .boolean, .value = 0 });
        p.space();
        if (p.pos >= p.source.len) return error.Unsupported;
        if (p.source[p.pos] == '\'' or p.source[p.pos] == '"') return p.string(false);
        if (p.source[p.pos] == 'f' and p.pos + 1 < p.source.len and
            (p.source[p.pos + 1] == '\'' or p.source[p.pos + 1] == '"'))
        {
            p.pos += 1;
            return p.string(true);
        }
        if (std.ascii.isDigit(p.source[p.pos])) {
            const start = p.pos;
            while (p.pos < p.source.len and std.ascii.isDigit(p.source[p.pos])) p.pos += 1;
            const text = p.source[start..p.pos];
            if (text.len > 1 and text[0] == '0') return error.Unsupported;
            const value = std.fmt.parseInt(i64, text, 10) catch return error.Unsupported;
            if (value > std.math.maxInt(i32)) return error.Unsupported;
            return p.program.node(.{ .tag = .integer, .value = value });
        }
        const name = try p.identifier();
        if (p.eat("(")) return p.call(name);
        if (!ast.validName(name)) return error.Unsupported;
        return p.program.node(.{ .tag = .name, .text = name });
    }

    fn sequence(p: *Parser) ast.Error!usize {
        var args: std.ArrayList(usize) = .empty;
        if (!p.eat("]")) while (true) {
            if (args.items.len >= 128) return error.Unsupported;
            try args.append(p.program.a, try p.expression());
            if (p.eat("]")) break;
            try p.need(",");
            if (p.eat("]")) break;
        };
        return p.program.node(.{ .tag = .list, .args = try args.toOwnedSlice(p.program.a) });
    }

    fn call(p: *Parser, name: []const u8) ast.Error!usize {
        if (std.mem.eql(u8, name, "int")) {
            try p.need("str");
            try p.need("(");
            const operand = try p.expression();
            try p.need(")");
            try p.need("[");
            try p.need(":");
            try p.need(":");
            try p.need("-");
            try p.need("1");
            try p.need("]");
            try p.need(")");
            return p.program.node(.{ .tag = .reverse_digits, .lhs = operand });
        }
        const builtins = [_][]const u8{ "len", "sum", "range", "gcd", "all", "print", "ValueError" };
        var allowed = ast.validName(name);
        for (builtins) |builtin| {
            if (std.mem.eql(u8, name, builtin)) allowed = true;
        }
        if (!allowed) return error.Unsupported;
        var args: std.ArrayList(usize) = .empty;
        if (!p.eat(")")) while (true) {
            if (args.items.len >= 128) return error.Unsupported;
            const arg = try p.expression();
            if (args.items.len == 0 and (std.mem.eql(u8, name, "all") or std.mem.eql(u8, name, "sum")) and p.eat("for")) {
                if (p.pos < 4 or p.source[p.pos - 4] != ' ') return error.Unsupported;
                const variable = try p.identifier();
                if (!ast.validName(variable)) return error.Unsupported;
                try p.need("in");
                const range = try p.expression();
                const iterable = p.program.nodes.items[range];
                const all = std.mem.eql(u8, name, "all");
                if (all and (iterable.tag != .call or !std.mem.eql(u8, iterable.text, "range")))
                    return error.Unsupported;
                const filter: ?usize = if (!all and p.eat("if")) try p.expression() else null;
                try p.need(")");
                return p.program.node(.{
                    .tag = if (all) .all_range else .sum_filtered,
                    .text = variable,
                    .lhs = arg,
                    .rhs = range,
                    .filter = filter,
                });
            }
            try args.append(p.program.a, arg);
            if (p.eat(")")) break;
            try p.need(",");
            if (p.eat(")")) break;
        };
        return p.program.node(.{ .tag = .call, .text = name, .args = try args.toOwnedSlice(p.program.a) });
    }

    fn string(p: *Parser, formatted: bool) ast.Error!usize {
        const quote = p.source[p.pos];
        p.pos += 1;
        var bytes: std.ArrayList(u8) = .empty;
        var pieces: std.ArrayList(usize) = .empty;
        while (p.pos < p.source.len) {
            const c = p.source[p.pos];
            p.pos += 1;
            if (c == quote) {
                if (!formatted) return p.program.node(.{ .tag = .string, .text = try bytes.toOwnedSlice(p.program.a) });
                try p.flush(&bytes, &pieces);
                return p.program.node(.{ .tag = .format, .args = try pieces.toOwnedSlice(p.program.a) });
            }
            if (c == '\n' or c == '\r' or c == 0) return error.Unsupported;
            if (c == '\\') {
                try bytes.append(p.program.a, try p.escape());
            } else if (formatted and (c == '{' or c == '}')) {
                if (p.pos < p.source.len and p.source[p.pos] == c) {
                    p.pos += 1;
                    try bytes.append(p.program.a, c);
                } else if (c == '{') {
                    try p.flush(&bytes, &pieces);
                    const interpolation = try p.field();
                    if (p.depth >= 8) return error.Unsupported;
                    var nested: Parser = .{ .program = p.program, .source = interpolation, .depth = p.depth };
                    const root = try nested.expression();
                    try nested.end();
                    try pieces.append(p.program.a, root);
                } else return error.Unsupported;
            } else try bytes.append(p.program.a, c);
        }
        return error.Unsupported;
    }

    fn flush(p: *Parser, bytes: *std.ArrayList(u8), pieces: *std.ArrayList(usize)) ast.Error!void {
        if (bytes.items.len == 0) return;
        const text = try bytes.toOwnedSlice(p.program.a);
        try pieces.append(p.program.a, try p.program.node(.{ .tag = .string, .text = text }));
    }

    fn escape(p: *Parser) ast.Error!u8 {
        if (p.pos >= p.source.len) return error.Unsupported;
        const c = p.source[p.pos];
        p.pos += 1;
        return switch (c) {
            '\\', '\'', '"' => c,
            'n' => '\n',
            'r' => '\r',
            't' => '\t',
            'b' => 8,
            'f' => 12,
            'v' => 11,
            'a' => 7,
            'x' => blk: {
                if (p.pos + 2 > p.source.len) return error.Unsupported;
                for (p.source[p.pos .. p.pos + 2]) |digit| if (!std.ascii.isHex(digit)) return error.Unsupported;
                const byte = std.fmt.parseInt(u8, p.source[p.pos .. p.pos + 2], 16) catch return error.Unsupported;
                if (byte == 0 or byte >= 128) return error.Unsupported;
                p.pos += 2;
                break :blk byte;
            },
            else => error.Unsupported,
        };
    }

    fn field(p: *Parser) ast.Invalid![]const u8 {
        const start = p.pos;
        var nesting: usize = 0;
        var quote: u8 = 0;
        var escaped = false;
        while (p.pos < p.source.len) : (p.pos += 1) {
            const c = p.source[p.pos];
            if (quote != 0) {
                if (escaped) escaped = false else if (c == '\\') escaped = true else if (c == quote) quote = 0;
                continue;
            }
            switch (c) {
                '\'', '"' => quote = c,
                '(', '[' => {
                    nesting += 1;
                    if (nesting > 8) return error.Unsupported;
                },
                ')', ']' => {
                    if (nesting == 0) return error.Unsupported;
                    nesting -= 1;
                },
                '}' => {
                    if (nesting != 0) return error.Unsupported;
                    const result = p.source[start..p.pos];
                    p.pos += 1;
                    return result;
                },
                '{', ':', '\n', '\r' => return error.Unsupported,
                '!' => {
                    if (p.pos + 1 >= p.source.len or p.source[p.pos + 1] != '=') return error.Unsupported;
                },
                else => {},
            }
        }
        return error.Unsupported;
    }
};

fn nameChar(c: u8) bool {
    return std.ascii.isAlphanumeric(c) or c == '_';
}

pub fn boundedDepth(program: *const ast.Program, id: usize, depth: usize) ast.Invalid!void {
    if (depth >= 8) return error.Unsupported;
    const node = program.nodes.items[id];
    switch (node.tag) {
        .integer, .boolean, .name, .string => {},
        .format, .list, .call => for (node.args) |arg| try boundedDepth(program, arg, depth + 1),
        .neg, .logical_not, .reverse_digits => try boundedDepth(program, node.lhs, depth + 1),
        else => {
            try boundedDepth(program, node.lhs, depth + 1);
            try boundedDepth(program, node.rhs, depth + 1);
            if (node.filter) |filter| try boundedDepth(program, filter, depth + 1);
        },
    }
}
