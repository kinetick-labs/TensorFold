const std = @import("std");
const ast = @import("runner_frontend_collections_ast.zig");
const expr = @import("runner_frontend_collections_expr.zig");

pub fn parse(p: *ast.Program, source: []const u8) ast.Error!void {
    if (source.len == 0 or source.len > 4096) return error.Unsupported;
    var s: Parser = .{ .p = p, .lines = try lines(p, source) };
    if (s.lines.len == 0 or s.lines[0].indent != 0) return error.Unsupported;
    p.body = try s.block(0, &p.globals, false, 0, 0);
    if (s.pos != s.lines.len) return error.Unsupported;
    for (p.nodes.items) |n| {
        if (n.tag == .name and std.mem.eql(u8, n.text, "__name__")) _ = try p.globals.bind(p.a, "__name__");
    }
    try protectedScope(p, &p.globals);
    for (p.functions.items) |*f| try protectedScope(p, &f.scope);
}

const Line = struct { indent: usize, text: []const u8 };
const Parser = struct {
    p: *ast.Program,
    lines: []const Line,
    pos: usize = 0,
    count: usize = 0,

    fn block(s: *Parser, indent: usize, scope: *ast.Scope, function: bool, loops: usize, depth: usize) ast.Error![]const ast.Stmt {
        if (depth >= 16) return error.Unsupported;
        var out: std.ArrayList(ast.Stmt) = .empty;
        while (s.pos < s.lines.len) {
            const line = s.lines[s.pos];
            if (line.indent < indent) break;
            if (line.indent != indent) return error.Unsupported;
            if (starts(line.text, "else:") or starts(line.text, "elif ")) break;
            s.pos += 1;
            try s.statements(&out, line.text, indent, scope, function, loops, depth);
        }
        if (out.items.len == 0) return error.Unsupported;
        return out.toOwnedSlice(s.p.a);
    }

    fn statements(s: *Parser, out: *std.ArrayList(ast.Stmt), text: []const u8, indent: usize, scope: *ast.Scope, function: bool, loops: usize, depth: usize) ast.Error!void {
        if (compound(text)) {
            try out.append(s.p.a, try s.statement(text, indent, scope, function, loops, depth));
            return;
        }
        var rest = text;
        while (rest.len != 0) {
            const sep = topChar(rest, ';') orelse rest.len;
            const part = trim(rest[0..sep]);
            if (part.len == 0 or compound(part)) return error.Unsupported;
            try out.append(s.p.a, try s.statement(part, indent, scope, function, loops, depth));
            if (sep == rest.len) break;
            rest = trim(rest[sep + 1 ..]);
        }
    }

    fn suite(s: *Parser, tail: []const u8, indent: usize, scope: *ast.Scope, function: bool, loops: usize, depth: usize) ast.Error![]const ast.Stmt {
        if (depth >= 16) return error.Unsupported;
        if (trim(tail).len != 0) {
            if (compound(trim(tail))) return error.Unsupported;
            var out: std.ArrayList(ast.Stmt) = .empty;
            try s.statements(&out, trim(tail), indent, scope, function, loops, depth + 1);
            return out.toOwnedSlice(s.p.a);
        }
        if (s.pos == s.lines.len or s.lines[s.pos].indent <= indent) return error.Unsupported;
        return s.block(s.lines[s.pos].indent, scope, function, loops, depth + 1);
    }

    fn conditional(s: *Parser, text: []const u8, indent: usize, scope: *ast.Scope, function: bool, loops: usize, depth: usize) ast.Error!ast.Stmt {
        if (depth >= 16) return error.Unsupported;
        const colon = topChar(text, ':') orelse return error.Unsupported;
        const condition = try expr.parse(s.p, trim(text[0..colon]));
        const body = try s.suite(text[colon + 1 ..], indent, scope, function, loops, depth);
        var otherwise: []const ast.Stmt = &.{};
        if (s.pos < s.lines.len and s.lines[s.pos].indent == indent) {
            const next = s.lines[s.pos].text;
            if (starts(next, "else:")) {
                s.pos += 1;
                otherwise = try s.suite(next[5..], indent, scope, function, loops, depth);
            } else if (starts(next, "elif ")) {
                s.pos += 1;
                const branch = try s.p.a.alloc(ast.Stmt, 1);
                branch[0] = try s.conditional(next[5..], indent, scope, function, loops, depth + 1);
                otherwise = branch;
            }
        }
        return .{ .tag = .conditional, .expr = condition, .body = body, .otherwise = otherwise };
    }

    fn statement(s: *Parser, text: []const u8, indent: usize, scope: *ast.Scope, function: bool, loops: usize, depth: usize) ast.Error!ast.Stmt {
        if (s.count >= 256) return error.Unsupported;
        s.count += 1;
        if (keyword(text, "if")) return s.conditional(text[2..], indent, scope, function, loops, depth);
        if (starts(text, "def ")) {
            if (function or depth != 0 or s.p.functions.items.len >= 16) return error.Unsupported;
            return s.definition(text[4..], indent, depth);
        }
        if (starts(text, "for ") or keyword(text, "while")) {
            const is_for = starts(text, "for ");
            const header_start: usize = if (is_for) 4 else 5;
            const header = text[header_start..];
            const colon = topChar(header, ':') orelse return error.Unsupported;
            var targets: []const ast.Id = &.{};
            const value = if (is_for) blk: {
                const sep = topWord(header[0..colon], "in") orelse return error.Unsupported;
                const target = try expr.parse(s.p, trim(header[0..sep]));
                try bindTarget(s.p, scope, target, false, 0);
                targets = try s.p.a.dupe(ast.Id, &.{target});
                break :blk try expr.parse(s.p, trim(header[sep + 2 .. colon]));
            } else try expr.parse(s.p, trim(header[0..colon]));
            const body = try s.suite(header[colon + 1 ..], indent, scope, function, loops + 1, depth);
            return .{ .tag = if (is_for) .for_loop else .while_loop, .targets = targets, .expr = value, .body = body };
        }
        if (std.mem.eql(u8, text, "break") or std.mem.eql(u8, text, "continue")) {
            if (loops == 0) return error.Unsupported;
            return .{ .tag = if (text[0] == 'b') .break_loop else .continue_loop };
        }
        if (std.mem.eql(u8, text, "pass")) return .{ .tag = .pass };
        if (keyword(text, "return")) {
            if (!function) return error.Unsupported;
            const tail = trim(text[6..]);
            return .{ .tag = .ret, .expr = if (tail.len == 0) null else try expr.parse(s.p, tail) };
        }
        if (keyword(text, "assert")) {
            const tail = trim(text[6..]);
            if (topChar(tail, ',') != null) return error.Unsupported;
            return .{ .tag = .assertion, .expr = try expr.parse(s.p, tail) };
        }
        if (assignment(text)) |eq| {
            var lhs = trim(text[0..eq]);
            const rhs = trim(text[eq + 1 ..]);
            if (lhs.len == 0 or rhs.len == 0) return error.Unsupported;
            var op: ?ast.Tag = null;
            if (std.mem.endsWith(u8, lhs, "//")) {
                lhs = trim(lhs[0 .. lhs.len - 2]);
                op = .floor;
            } else {
                op = switch (lhs[lhs.len - 1]) {
                    '+' => .add,
                    '-' => .sub,
                    '*' => .mul,
                    '%' => .modulo,
                    else => null,
                };
                if (op != null) lhs = trim(lhs[0 .. lhs.len - 1]);
            }
            var targets: std.ArrayList(ast.Id) = .empty;
            const target = try expr.parse(s.p, lhs);
            try bindTarget(s.p, scope, target, true, 0);
            if (op != null and s.p.nodes.items[target].tag != .name and s.p.nodes.items[target].tag != .index)
                return error.Unsupported;
            try targets.append(s.p.a, target);
            var rest = rhs;
            while (assignment(rest)) |next| {
                if (op != null) return error.Unsupported;
                const chained = try expr.parse(s.p, trim(rest[0..next]));
                try bindTarget(s.p, scope, chained, true, 0);
                try targets.append(s.p.a, chained);
                rest = trim(rest[next + 1 ..]);
            }
            return .{ .tag = if (op != null) .augassign else .assign, .op = op orelse .add, .targets = try targets.toOwnedSlice(s.p.a), .expr = try expr.parse(s.p, rest) };
        }
        const root = try expr.parse(s.p, text);
        const node = s.p.nodes.items[root];
        if (node.tag == .call and std.mem.eql(u8, node.text, "print")) {
            for (node.keywords) |key| if (key.len != 0) return error.Unsupported;
            return .{ .tag = .print, .targets = node.args };
        }
        return .{ .tag = .expression, .expr = root };
    }

    fn definition(s: *Parser, text: []const u8, indent: usize, depth: usize) ast.Error!ast.Stmt {
        var e: expr.Parser = .{ .p = s.p, .source = text };
        const name = try e.identifier();
        if (builtinName(name) or s.p.functionId(name) != null) return error.Unsupported;
        try e.need("(");
        var params: std.ArrayList([]const u8) = .empty;
        var scope: ast.Scope = .{};
        if (!e.eat(")")) while (true) {
            const param = try e.identifier();
            if (scope.find(param) != null or builtinName(param)) return error.Unsupported;
            _ = try scope.bind(s.p.a, param);
            try params.append(s.p.a, param);
            if (e.eat(")")) break;
            try e.need(",");
            if (e.eat(")")) break;
        };
        try e.need(":");
        const id = s.p.functions.items.len;
        try s.p.functions.append(s.p.a, .{ .name = name, .params = try params.toOwnedSlice(s.p.a) });
        const body = try s.suite(text[e.pos..], indent, &scope, true, 0, depth);
        s.p.functions.items[id].scope = scope;
        s.p.functions.items[id].body = body;
        return .{ .tag = .function, .function = id };
    }
};

fn bindTarget(p: *ast.Program, scope: *ast.Scope, id: ast.Id, index: bool, depth: usize) ast.Error!void {
    if (depth >= 24) return error.Unsupported;
    const n = p.nodes.items[id];
    switch (n.tag) {
        .name => {
            if (builtinName(n.text) or std.mem.eql(u8, n.text, "__name__")) return error.Unsupported;
            _ = try scope.bind(p.a, n.text);
        },
        .tuple, .list => {
            if (n.args.len == 0) return error.Unsupported;
            p.nodes.items[id].tag = .tuple;
            for (n.args) |arg| try bindTarget(p, scope, arg, index, depth + 1);
        },
        .index => if (!index) return error.Unsupported,
        else => return error.Unsupported,
    }
}

fn protectedScope(p: *const ast.Program, scope: *const ast.Scope) ast.Error!void {
    for (scope.names.items) |name| if (p.functionId(name) != null) return error.Unsupported;
    for (p.nodes.items) |n| {
        if (n.tag == .call and scope.find(n.text) != null) return error.Unsupported;
        if (n.tag == .comprehension and (builtinName(n.text) or p.functionId(n.text) != null)) return error.Unsupported;
    }
}

fn builtinName(name: []const u8) bool {
    const names = [_][]const u8{ "print", "len", "sum", "min", "max", "range", "enumerate", "zip", "sorted", "set", "dict", "list", "tuple", "str", "int", "bool", "any", "all", "map", "abs", "reversed" };
    for (names) |n| if (std.mem.eql(u8, n, name)) return true;
    return false;
}

fn trim(text: []const u8) []const u8 {
    return std.mem.trim(u8, text, " \t\r\n");
}

fn starts(text: []const u8, prefix: []const u8) bool {
    return std.mem.startsWith(u8, text, prefix);
}

fn keyword(text: []const u8, word: []const u8) bool {
    return starts(text, word) and (text.len == word.len or
        !(std.ascii.isAlphanumeric(text[word.len]) or text[word.len] == '_'));
}

fn compound(text: []const u8) bool {
    return keyword(text, "if") or starts(text, "for ") or keyword(text, "while") or starts(text, "def ");
}

pub fn topChar(text: []const u8, wanted: u8) ?usize {
    var nesting: usize = 0;
    var quote: u8 = 0;
    var escaped = false;
    for (text, 0..) |c, i| {
        if (quote != 0) {
            if (escaped) escaped = false else if (c == '\\') escaped = true else if (c == quote) quote = 0;
        } else if (c == '\'' or c == '"') {
            quote = c;
        } else if (nesting == 0 and c == wanted) {
            return i;
        } else if (c == '(' or c == '[' or c == '{') {
            nesting += 1;
        } else if (c == ')' or c == ']' or c == '}') {
            if (nesting == 0) return null;
            nesting -= 1;
        }
    }
    return null;
}

fn topWord(text: []const u8, word: []const u8) ?usize {
    var offset: usize = 0;
    while (offset < text.len) {
        const relative = topChar(text[offset..], word[0]) orelse return null;
        const i = offset + relative;
        if ((i == 0 or text[i - 1] == ' ') and starts(text[i..], word) and
            (i + word.len == text.len or text[i + word.len] == ' ')) return i;
        offset = i + 1;
    }
    return null;
}

fn assignment(text: []const u8) ?usize {
    var offset: usize = 0;
    while (offset < text.len) {
        const relative = topChar(text[offset..], '=') orelse return null;
        const i = offset + relative;
        if ((i == 0 or std.mem.indexOfScalar(u8, "=!<>", text[i - 1]) == null) and
            (i + 1 == text.len or text[i + 1] != '=')) return i;
        offset = i + 1;
    }
    return null;
}

fn lines(p: *ast.Program, source: []const u8) ast.Error![]const Line {
    var out: std.ArrayList(Line) = .empty;
    var text: std.ArrayList(u8) = .empty;
    var stack: [24]u8 = undefined;
    var depth: usize = 0;
    var quote: u8 = 0;
    var escaped = false;
    var comment = false;
    var beginning = true;
    var indent: usize = 0;
    for (source, 0..) |c, source_pos| {
        if (c == '\r') {
            if (source_pos + 1 < source.len and source[source_pos + 1] == '\n') continue;
            return error.Unsupported;
        }
        if (c == 0 or c >= 128 or (c < 32 and c != '\n' and c != '\t')) return error.Unsupported;
        if (comment and c != '\n') continue;
        if (c == '\n' and quote == 0) {
            comment = false;
            if (depth != 0) {
                try text.append(p.a, ' ');
            } else {
                const value = trim(text.items);
                if (value.len != 0) try out.append(p.a, .{ .indent = indent, .text = try p.a.dupe(u8, value) });
                text.clearRetainingCapacity();
                beginning = true;
                indent = 0;
            }
            continue;
        }
        if (quote != 0) {
            if (c == '\n') return error.Unsupported;
            if (escaped) escaped = false else if (c == '\\') escaped = true else if (c == quote) quote = 0;
        } else {
            if (c == '#') {
                comment = true;
                continue;
            }
            if (beginning and c == ' ') {
                indent += 1;
                continue;
            }
            if (c == '\t') return error.Unsupported;
            beginning = false;
            if (c == '\'' or c == '"') quote = c else if (c == '(' or c == '[' or c == '{') {
                if (depth == stack.len) return error.Unsupported;
                stack[depth] = c;
                depth += 1;
            } else if (c == ')' or c == ']' or c == '}') {
                if (depth == 0) return error.Unsupported;
                depth -= 1;
                const expected: u8 = switch (c) {
                    ')' => '(',
                    ']' => '[',
                    else => '{',
                };
                if (stack[depth] != expected) return error.Unsupported;
            }
        }
        try text.append(p.a, c);
    }
    if (quote != 0 or depth != 0) return error.Unsupported;
    const value = trim(text.items);
    if (value.len != 0) try out.append(p.a, .{ .indent = indent, .text = try p.a.dupe(u8, value) });
    return out.toOwnedSlice(p.a);
}

test "collections parser preserves print grouping and conditional slots" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var p: ast.Program = .{ .a = arena.allocator() };
    try parse(&p, "print(1, 2); print((1, 2)); print(3 if True else 4)");
    try std.testing.expectEqual(@as(usize, 3), p.body.len);
    try std.testing.expectEqual(@as(usize, 2), p.body[0].targets.len);
    try std.testing.expectEqual(@as(usize, 1), p.body[1].targets.len);
    try std.testing.expectEqual(ast.Tag.tuple, p.nodes.items[p.body[1].targets[0]].tag);
    const choice = p.nodes.items[p.body[2].targets[0]];
    try std.testing.expectEqual(ast.Tag.conditional, choice.tag);
    try std.testing.expectEqual(@as(i64, 3), p.nodes.items[choice.a].value);
    try std.testing.expectEqual(ast.Tag.boolean, p.nodes.items[choice.b].tag);
    try std.testing.expectEqual(@as(i64, 4), p.nodes.items[choice.c.?].value);

    const slices = [_][]const u8{ "a[:]", "a[1:3]", "a[::-1]" };
    for (slices, 0..) |source, i| {
        const slice_id = try expr.parse(&p, source);
        const slice = p.nodes.items[slice_id];
        try std.testing.expectEqual(ast.Tag.slice, slice.tag);
        try std.testing.expectEqualStrings("a", p.nodes.items[slice.a].text);
        try std.testing.expectEqual(@as(usize, 3), slice.args.len);
        if (i == 1) {
            try std.testing.expectEqual(@as(i64, 1), p.nodes.items[slice.args[0]].value);
            try std.testing.expectEqual(@as(i64, 3), p.nodes.items[slice.args[1]].value);
        } else {
            try std.testing.expectEqual(ast.Tag.none, p.nodes.items[slice.args[0]].tag);
            try std.testing.expectEqual(ast.Tag.none, p.nodes.items[slice.args[1]].tag);
        }
        try std.testing.expectEqual(if (i == 2) ast.Tag.neg else ast.Tag.none, p.nodes.items[slice.args[2]].tag);
    }
    const nested_id = try expr.parse(&p, "a[0][1:3]");
    const nested = p.nodes.items[nested_id];
    try std.testing.expectEqual(ast.Tag.slice, nested.tag);
    const indexed = p.nodes.items[nested.a];
    try std.testing.expectEqual(ast.Tag.index, indexed.tag);
    try std.testing.expectEqualStrings("a", p.nodes.items[indexed.a].text);
    const after_slice_id = try expr.parse(&p, "a[:][0]");
    const after_slice = p.nodes.items[after_slice_id];
    try std.testing.expectEqual(ast.Tag.index, after_slice.tag);
    try std.testing.expectEqual(ast.Tag.slice, p.nodes.items[after_slice.a].tag);
    const ordinary_id = try expr.parse(&p, "a[0][1]");
    const ordinary = p.nodes.items[ordinary_id];
    try std.testing.expectEqual(ast.Tag.index, ordinary.tag);
    try std.testing.expectEqual(ast.Tag.index, p.nodes.items[ordinary.a].tag);
}

test "collections parser builds source calls and parameter first recursive scopes" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var p: ast.Program = .{ .a = arena.allocator() };
    try parse(&p, "def work(xs):\n" ++
        "    a = b = 0\n" ++
        "    for i, value in enumerate(xs):\n" ++
        "        a, b = b, value\n" ++
        "    return a\n" ++
        "digits = [3, 1, 2]\n" ++
        "print(''.join(map(str, sorted(\n" ++
        "    digits, reverse=True))))\n" ++
        "print(sorted([[2], [1]], key=sum))\n" ++
        "print(any('x' in item for item in ['x', 'y']))\n");
    const scope = p.functions.items[0].scope.names.items;
    try std.testing.expectEqual(@as(usize, 5), scope.len);
    try std.testing.expectEqualStrings("xs", scope[0]);
    try std.testing.expectEqualStrings("a", scope[1]);
    try std.testing.expectEqualStrings("b", scope[2]);
    try std.testing.expectEqualStrings("i", scope[3]);
    try std.testing.expectEqualStrings("value", scope[4]);
    var found_str = false;
    var found_sum = false;
    for (p.nodes.items) |n| {
        if (n.tag == .name and std.mem.eql(u8, n.text, "str")) found_str = true;
        if (n.tag == .name and std.mem.eql(u8, n.text, "sum")) found_sum = true;
    }
    try std.testing.expect(found_str and found_sum);
}

test "collections parser rejects unsupported cells without erasing statements" {
    const cases = [_][]const u8{
        "import os\nprint(1)",
        "print(1)\nraise ValueError()",
        "def f(x=1): return x\nprint(f())",
        "def f():\n    def g(): return 1\n    return g()\nprint(f())",
        "def f():\n    nonlocal x\n    return x\nprint(f())",
        "print(1.5)",
        "a = [1]; a[:] = [2]; print(a)",
        "a = [[1]]; a[0][:] = [2]; print(a)",
        "a = [1]; a[::-1] += [2]; print(a)",
        "a = [[1]]; print(a[:, 0])",
        "print(1 < 2 < 3)",
        "print(1and True)",
        "print([x for x in [1] for y in [2]])",
        "print({x for x in [1]})",
        "print({x: x for x in [1]})",
        "print([x for x in [1] if x if True])",
        "print(1)\nfor x in [1]: pass\nelse: print(2)",
        "print(1)\nbreak",
        "print(1)\nreturn 2",
        "len = 1\nprint(len)",
        "def f(): return 1\nf = 2\nprint(f())",
        "assert False, 'failure'\nprint(1)",
        "print(1);;print(2)",
        "print([1, 2))",
        "print('unterminated)",
        "print(1)\n    print(2)",
    };
    for (cases) |source| {
        var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
        defer arena.deinit();
        var p: ast.Program = .{ .a = arena.allocator() };
        try std.testing.expectError(error.Unsupported, parse(&p, source));
    }
}

test "collections parser marks only any and all generator syntax as lazy" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var p: ast.Program = .{ .a = arena.allocator() };
    try parse(&p, "print(any(x for x in [1, 0]))\n" ++
        "print(any([x for x in [1, 0]]))\n" ++
        "print(all(x for x in [1, 0]))\n" ++
        "print(all([x for x in [1, 0]]))\n" ++
        "print([x for x in [1, 0]])\n");
    for (p.body, 0..) |stmt, i| {
        const argument = p.nodes.items[stmt.targets[0]];
        const comp = if (argument.tag == .call) p.nodes.items[argument.args[0]] else argument;
        try std.testing.expectEqual(ast.Tag.comprehension, comp.tag);
        const expected: i64 = if (i == 0 or i == 2) 1 else 0;
        try std.testing.expectEqual(expected, comp.value);
    }
}
