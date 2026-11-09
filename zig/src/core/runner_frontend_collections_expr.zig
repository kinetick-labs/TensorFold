const std = @import("std");
const ast = @import("runner_frontend_collections_ast.zig");

pub fn parse(p: *ast.Program, source: []const u8) ast.Error!ast.Id {
    var s: Parser = .{ .p = p, .source = source };
    const id = try s.tuple();
    s.space();
    if (s.pos != source.len) return error.Unsupported;
    try bounded(p, id, 0);
    return id;
}

pub const Parser = struct {
    p: *ast.Program,
    source: []const u8,
    pos: usize = 0,
    depth: usize = 0,

    pub fn space(s: *Parser) void {
        while (s.pos < s.source.len and std.ascii.isWhitespace(s.source[s.pos])) s.pos += 1;
    }

    pub fn eat(s: *Parser, token: []const u8) bool {
        s.space();
        if (!std.mem.startsWith(u8, s.source[s.pos..], token)) return false;
        const end = s.pos + token.len;
        if (nameChar(token[token.len - 1]) and end < s.source.len and nameChar(s.source[end])) return false;
        s.pos = end;
        return true;
    }

    pub fn need(s: *Parser, token: []const u8) ast.Error!void {
        if (!s.eat(token)) return error.Unsupported;
    }

    pub fn identifier(s: *Parser) ast.Error![]const u8 {
        s.space();
        const start = s.pos;
        if (start == s.source.len or !(std.ascii.isAlphabetic(s.source[start]) or s.source[start] == '_'))
            return error.Unsupported;
        s.pos += 1;
        while (s.pos < s.source.len and nameChar(s.source[s.pos])) s.pos += 1;
        const name = s.source[start..s.pos];
        if (!validName(name)) return error.Unsupported;
        return name;
    }

    pub fn tuple(s: *Parser) ast.Error!ast.Id {
        const first = try s.expression();
        if (!s.eat(",")) return first;
        var args: std.ArrayList(ast.Id) = .empty;
        try args.append(s.p.a, first);
        while (true) {
            s.space();
            if (s.pos == s.source.len or s.source[s.pos] == ')') break;
            try args.append(s.p.a, try s.expression());
            if (!s.eat(",")) break;
        }
        return s.p.node(.{ .tag = .tuple, .args = try args.toOwnedSlice(s.p.a) });
    }

    pub fn expression(s: *Parser) ast.Error!ast.Id {
        if (s.depth >= 24) return error.Unsupported;
        s.depth += 1;
        defer s.depth -= 1;
        const yes = try s.disjunction();
        if (!s.eat("if")) return yes;
        const condition = try s.disjunction();
        try s.need("else");
        return s.p.node(.{ .tag = .conditional, .a = yes, .b = condition, .c = try s.expression() });
    }

    fn disjunction(s: *Parser) ast.Error!ast.Id {
        var root = try s.conjunction();
        while (s.eat("or")) root = try s.binary(.logical_or, root, try s.conjunction());
        return root;
    }

    fn conjunction(s: *Parser) ast.Error!ast.Id {
        var root = try s.inversion();
        while (s.eat("and")) root = try s.binary(.logical_and, root, try s.inversion());
        return root;
    }

    fn inversion(s: *Parser) ast.Error!ast.Id {
        if (s.eat("not")) {
            if (s.depth >= 24) return error.Unsupported;
            s.depth += 1;
            defer s.depth -= 1;
            return s.p.node(.{ .tag = .logical_not, .a = try s.inversion() });
        }
        const left = try s.additive();
        const tag: ast.Tag = if (s.eat("==")) .eq else if (s.eat("!=")) .ne else if (s.eat("<=")) .le else if (s.eat(">=")) .ge else if (s.eat("<")) .lt else if (s.eat(">")) .gt else if (s.eat("in")) .contains else if (s.eat("not")) blk: {
            try s.need("in");
            break :blk .not_contains;
        } else if (s.eat("is")) (if (s.eat("not")) ast.Tag.is_not else ast.Tag.is_identity) else return left;
        return s.binary(tag, left, try s.additive());
    }

    fn additive(s: *Parser) ast.Error!ast.Id {
        var root = try s.product();
        while (true) {
            const tag: ast.Tag = if (s.eat("+")) .add else if (s.eat("-")) .sub else break;
            root = try s.binary(tag, root, try s.product());
        }
        return root;
    }

    fn product(s: *Parser) ast.Error!ast.Id {
        var root = try s.unary();
        while (true) {
            const tag: ast.Tag = if (s.eat("//")) .floor else if (s.eat("%")) .modulo else if (s.eat("*")) .mul else break;
            root = try s.binary(tag, root, try s.unary());
        }
        return root;
    }

    fn unary(s: *Parser) ast.Error!ast.Id {
        const tag: ?ast.Tag = if (s.eat("-")) .neg else if (s.eat("+")) .pos else null;
        if (tag) |op| {
            if (s.depth >= 24) return error.Unsupported;
            s.depth += 1;
            defer s.depth -= 1;
            return s.p.node(.{ .tag = op, .a = try s.unary() });
        }
        var root = try s.atom();
        while (true) {
            if (s.eat("[")) {
                const start = if (s.peek(":")) try s.p.node(.{ .tag = .none }) else try s.expression();
                if (s.eat(":")) {
                    const stop = if (s.peek(":") or s.peek("]")) try s.p.node(.{ .tag = .none }) else try s.expression();
                    var step = try s.p.node(.{ .tag = .none });
                    if (s.eat(":")) {
                        if (!s.peek("]")) step = try s.expression();
                    }
                    try s.need("]");
                    root = try s.p.node(.{ .tag = .slice, .a = root, .args = try s.p.a.dupe(ast.Id, &.{ start, stop, step }) });
                } else {
                    try s.need("]");
                    root = try s.p.node(.{ .tag = .index, .a = root, .b = start });
                }
            } else if (s.eat(".")) {
                const name = try s.identifier();
                try s.need("(");
                root = try s.call(.method, name, root);
            } else break;
        }
        return root;
    }

    fn peek(s: *Parser, token: []const u8) bool {
        const save = s.pos;
        defer s.pos = save;
        return s.eat(token);
    }

    fn binary(s: *Parser, tag: ast.Tag, a: ast.Id, b: ast.Id) ast.Error!ast.Id {
        return s.p.node(.{ .tag = tag, .a = a, .b = b });
    }

    fn atom(s: *Parser) ast.Error!ast.Id {
        if (s.eat("(")) {
            if (s.eat(")")) return s.p.node(.{ .tag = .tuple });
            const root = try s.tuple();
            try s.need(")");
            return root;
        }
        if (s.eat("[")) return s.sequence(.list, "]");
        if (s.eat("{")) return s.sequence(.set, "}");
        if (s.eat("True")) return s.p.node(.{ .tag = .boolean, .value = 1 });
        if (s.eat("False")) return s.p.node(.{ .tag = .boolean });
        if (s.eat("None")) return s.p.node(.{ .tag = .none });
        s.space();
        if (s.pos == s.source.len) return error.Unsupported;
        if (s.source[s.pos] == '\'' or s.source[s.pos] == '"') return s.string();
        if (std.ascii.isDigit(s.source[s.pos])) {
            const start = s.pos;
            while (s.pos < s.source.len and std.ascii.isDigit(s.source[s.pos])) s.pos += 1;
            const text = s.source[start..s.pos];
            if (s.pos < s.source.len and nameChar(s.source[s.pos])) return error.Unsupported;
            if (text.len > 1 and text[0] == '0') return error.Unsupported;
            const value = std.fmt.parseInt(i64, text, 10) catch return error.Unsupported;
            if (value > 2147483648) return error.Unsupported;
            return s.p.node(.{ .tag = .integer, .value = value });
        }
        const name = try s.identifier();
        if (s.eat("(")) return s.call(.call, name, 0);
        return s.p.node(.{ .tag = .name, .text = name });
    }

    fn sequence(s: *Parser, kind: ast.Tag, close: []const u8) ast.Error!ast.Id {
        var tag = kind;
        var args: std.ArrayList(ast.Id) = .empty;
        if (s.eat(close)) return s.p.node(.{ .tag = if (kind == .set) .dict else kind });
        const first = try s.expression();
        if (kind == .list and s.eat("for")) {
            const comp = try s.comprehension(first);
            try s.need(close);
            return comp;
        }
        try args.append(s.p.a, first);
        if (kind == .set and s.eat(":")) {
            tag = .dict;
            try args.append(s.p.a, try s.expression());
        }
        while (!s.eat(close)) {
            try s.need(",");
            if (s.eat(close)) break;
            try args.append(s.p.a, try s.expression());
            if (tag == .dict) {
                try s.need(":");
                try args.append(s.p.a, try s.expression());
            }
        }
        return s.p.node(.{ .tag = tag, .args = try args.toOwnedSlice(s.p.a) });
    }

    fn comprehension(s: *Parser, output: ast.Id) ast.Error!ast.Id {
        const name = try s.identifier();
        try s.need("in");
        const iterable = try s.disjunction();
        const filter: ?ast.Id = if (s.eat("if")) try s.disjunction() else null;
        return s.p.node(.{ .tag = .comprehension, .text = name, .a = output, .b = iterable, .c = filter });
    }

    fn call(s: *Parser, tag: ast.Tag, name: []const u8, base: ast.Id) ast.Error!ast.Id {
        var args: std.ArrayList(ast.Id) = .empty;
        var keywords: std.ArrayList([]const u8) = .empty;
        var named = false;
        if (!s.eat(")")) while (true) {
            const save = s.pos;
            var key: []const u8 = "";
            if (s.identifier()) |candidate| {
                if (s.eat("=") and !s.peek("=")) key = candidate else s.pos = save;
            } else |_| s.pos = save;
            if (key.len != 0) {
                for (keywords.items) |old| if (std.mem.eql(u8, old, key)) return error.Unsupported;
                named = true;
            } else if (named) return error.Unsupported;
            var arg = try s.expression();
            if (s.eat("for")) {
                if (tag != .call or args.items.len != 0 or key.len != 0 or
                    !(std.mem.eql(u8, name, "any") or std.mem.eql(u8, name, "all"))) return error.Unsupported;
                arg = try s.comprehension(arg);
                s.p.nodes.items[arg].value = 1;
                try args.append(s.p.a, arg);
                try keywords.append(s.p.a, "");
                try s.need(")");
                break;
            }
            try args.append(s.p.a, arg);
            try keywords.append(s.p.a, key);
            if (s.eat(")")) break;
            try s.need(",");
            if (s.eat(")")) break;
        };
        return s.p.node(.{ .tag = tag, .text = name, .a = base, .args = try args.toOwnedSlice(s.p.a), .keywords = try keywords.toOwnedSlice(s.p.a) });
    }

    fn string(s: *Parser) ast.Error!ast.Id {
        const quote = s.source[s.pos];
        s.pos += 1;
        var bytes: std.ArrayList(u8) = .empty;
        while (s.pos < s.source.len) {
            var c = s.source[s.pos];
            s.pos += 1;
            if (c == quote) return s.p.node(.{ .tag = .string, .text = try bytes.toOwnedSlice(s.p.a) });
            if (c < 32 or c >= 128) return error.Unsupported;
            if (c == '\\') c = try s.escape();
            if (bytes.items.len >= 128) return error.Unsupported;
            try bytes.append(s.p.a, c);
        }
        return error.Unsupported;
    }

    fn escape(s: *Parser) ast.Error!u8 {
        if (s.pos == s.source.len) return error.Unsupported;
        const c = s.source[s.pos];
        s.pos += 1;
        switch (c) {
            '\\', '\'', '"' => return c,
            'n' => return 10,
            'r' => return 13,
            't' => return 9,
            'b' => return 8,
            'f' => return 12,
            'v' => return 11,
            'a' => return 7,
            'x', 'u', 'U' => {
                const len: usize = if (c == 'x') 2 else if (c == 'u') 4 else 8;
                if (s.pos + len > s.source.len) return error.Unsupported;
                const value = std.fmt.parseInt(u32, s.source[s.pos .. s.pos + len], 16) catch return error.Unsupported;
                for (s.source[s.pos .. s.pos + len]) |digit| if (!std.ascii.isHex(digit)) return error.Unsupported;
                if (value >= 128) return error.Unsupported;
                s.pos += len;
                return @intCast(value);
            },
            '0'...'7' => {
                var value: u16 = c - '0';
                var count: usize = 1;
                while (count < 3 and s.pos < s.source.len and s.source[s.pos] >= '0' and s.source[s.pos] <= '7') : (count += 1) {
                    value = value * 8 + s.source[s.pos] - '0';
                    s.pos += 1;
                }
                if (value >= 128) return error.Unsupported;
                return @intCast(value);
            },
            else => return error.Unsupported,
        }
    }
};

fn nameChar(c: u8) bool {
    return std.ascii.isAlphanumeric(c) or c == '_';
}

pub fn validName(name: []const u8) bool {
    if (name.len == 0 or !(std.ascii.isAlphabetic(name[0]) or name[0] == '_')) return false;
    for (name) |c| if (!nameChar(c)) return false;
    const reserved = [_][]const u8{ "__debug__", "__builtins__", "False", "None", "True", "and", "as", "assert", "async", "await", "break", "class", "continue", "def", "del", "elif", "else", "except", "finally", "for", "from", "global", "if", "import", "in", "is", "lambda", "nonlocal", "not", "or", "pass", "raise", "return", "try", "while", "with", "yield" };
    for (reserved) |word| if (std.mem.eql(u8, word, name)) return false;
    return true;
}

fn bounded(p: *const ast.Program, id: ast.Id, depth: usize) ast.Error!void {
    if (depth >= 24) return error.Unsupported;
    const n = p.nodes.items[id];
    switch (n.tag) {
        .integer, .boolean, .none, .string, .name => {},
        .list, .tuple, .dict, .set, .call => {},
        .slice, .method, .neg, .pos, .logical_not => try bounded(p, n.a, depth + 1),
        else => {
            try bounded(p, n.a, depth + 1);
            try bounded(p, n.b, depth + 1);
            if (n.c) |c| try bounded(p, c, depth + 1);
        },
    }
    for (n.args) |arg| try bounded(p, arg, depth + 1);
}
