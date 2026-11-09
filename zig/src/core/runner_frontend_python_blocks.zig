//! Structured Python admission and source lowering; the proof never supplies the emitted program's answers.
const std = @import("std");
const ast = @import("runner_frontend_python_ast.zig");
const expr = @import("runner_frontend_python_expr.zig");
const evaluator = @import("runner_frontend_python_eval.zig");
const emitter = @import("runner_frontend_python_emit.zig");
const Allocator = std.mem.Allocator;

pub const Lowered = struct { source: []u8, estimated_steps: usize };

pub fn lowerPython(a: Allocator, source: []const u8) Allocator.Error!?[]u8 {
    const lowered = try lowerWithCost(a, source) orelse return null;
    return lowered.source;
}

// Cost admission belongs to the caller; disabling its budget preserves semantic proof and source lowering.
pub fn lowerWithCost(a: Allocator, source: []const u8) Allocator.Error!?Lowered {
    if (source.len == 0 or source.len > 4096 or std.mem.indexOfScalar(u8, source, 0) != null) return null;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var program: ast.Program = .{ .a = arena.allocator() };
    parse(&program, source) catch |err| switch (err) {
        error.Unsupported => return null,
        error.OutOfMemory => return error.OutOfMemory,
    };
    evaluator.prove(&program) catch |err| switch (err) {
        error.Unsupported => return null,
        error.OutOfMemory => return error.OutOfMemory,
    };
    return .{ .source = try emitter.emit(a, &program), .estimated_steps = program.estimated_steps };
}

const Line = struct { indent: usize, text: []const u8 };
const Parser = struct {
    p: *ast.Program,
    lines: []const Line,
    pos: usize = 0,
    count: usize = 0,

    fn block(s: *Parser, indent: usize, scope: *ast.Scope, in_function: bool, depth: usize) ast.Error![]ast.Stmt {
        if (depth > 8) return error.Unsupported;
        var out: std.ArrayList(ast.Stmt) = .empty;
        while (s.pos < s.lines.len) {
            const line = s.lines[s.pos];
            if (line.indent < indent) break;
            if (line.indent != indent) return error.Unsupported;
            if (std.mem.eql(u8, line.text, "else:") or std.mem.startsWith(u8, line.text, "elif ")) break;
            s.pos += 1;
            try out.append(s.p.a, try s.statement(line.text, indent, scope, in_function, depth));
        }
        if (out.items.len == 0) return error.Unsupported;
        return out.toOwnedSlice(s.p.a);
    }

    fn suite(s: *Parser, tail: []const u8, indent: usize, scope: *ast.Scope, in_function: bool, depth: usize) ast.Error![]ast.Stmt {
        if (depth >= 8) return error.Unsupported;
        if (trim(tail).len != 0) {
            var out: std.ArrayList(ast.Stmt) = .empty;
            var rest = trim(tail);
            while (true) {
                const sep = topChar(rest, ';') orelse rest.len;
                const text = trim(rest[0..sep]);
                if (text.len == 0 or topChar(text, ':') != null) return error.Unsupported;
                try out.append(s.p.a, try s.statement(text, indent, scope, in_function, depth + 1));
                if (sep == rest.len) break;
                rest = trim(rest[sep + 1 ..]);
                if (rest.len == 0) break;
            }
            return out.toOwnedSlice(s.p.a);
        }
        if (s.pos == s.lines.len or s.lines[s.pos].indent <= indent) return error.Unsupported;
        return s.block(s.lines[s.pos].indent, scope, in_function, depth + 1);
    }

    fn conditional(s: *Parser, text: []const u8, indent: usize, scope: *ast.Scope, in_function: bool, depth: usize) ast.Error!ast.Stmt {
        const colon = topChar(text, ':') orelse return error.Unsupported;
        const condition = try expr.parse(s.p, trim(text[0..colon]));
        const body = try s.suite(text[colon + 1 ..], indent, scope, in_function, depth);
        var otherwise: []ast.Stmt = &.{};
        if (s.pos < s.lines.len and s.lines[s.pos].indent == indent) {
            const next = s.lines[s.pos].text;
            if (std.mem.startsWith(u8, next, "else:")) {
                s.pos += 1;
                otherwise = try s.suite(next[5..], indent, scope, in_function, depth);
            } else if (std.mem.startsWith(u8, next, "elif ")) {
                s.pos += 1;
                if (depth >= 8) return error.Unsupported;
                otherwise = try s.p.a.alloc(ast.Stmt, 1);
                otherwise[0] = try s.conditional(next[5..], indent, scope, in_function, depth + 1);
            }
        }
        return .{ .tag = .conditional, .expr = condition, .body = body, .otherwise = otherwise };
    }

    fn statement(s: *Parser, text: []const u8, indent: usize, scope: *ast.Scope, in_function: bool, depth: usize) ast.Error!ast.Stmt {
        if (s.count >= 256) return error.Unsupported;
        s.count += 1;
        if (std.mem.startsWith(u8, text, "if ")) return s.conditional(text[3..], indent, scope, in_function, depth);
        if (std.mem.startsWith(u8, text, "while ")) {
            const colon = topChar(text, ':') orelse return error.Unsupported;
            return .{
                .tag = .loop,
                .expr = try expr.parse(s.p, trim(text[6..colon])),
                .body = try s.suite(text[colon + 1 ..], indent, scope, in_function, depth),
            };
        }
        if (std.mem.startsWith(u8, text, "for ")) {
            const colon = topChar(text, ':') orelse return error.Unsupported;
            const head = trim(text[4..colon]);
            const sep = std.mem.indexOf(u8, head, " in ") orelse return error.Unsupported;
            const name = trim(head[0..sep]);
            _ = try scope.bind(s.p.a, name);
            const iterable = try expr.parse(s.p, trim(head[sep + 4 ..]));
            const value = s.p.nodes.items[iterable];
            if (value.tag != .list and !(value.tag == .call and std.mem.eql(u8, value.text, "range")))
                return error.Unsupported;
            return .{
                .tag = .each,
                .name = name,
                .expr = iterable,
                .body = try s.suite(text[colon + 1 ..], indent, scope, in_function, depth),
            };
        }
        if (std.mem.startsWith(u8, text, "def ")) {
            if (in_function or indent != 0 or depth != 0) return error.Unsupported;
            const colon = topChar(text, ':') orelse return error.Unsupported;
            const head = trim(text[4..colon]);
            const open = std.mem.indexOfScalar(u8, head, '(') orelse return error.Unsupported;
            if (!std.mem.endsWith(u8, head, ")")) return error.Unsupported;
            const name = trim(head[0..open]);
            if (!ast.validName(name) or s.p.functionId(name) != null or scope.find(name) != null)
                return error.Unsupported;
            var params: std.ArrayList([]const u8) = .empty;
            var locals: ast.Scope = .{};
            const args_text = trim(head[open + 1 .. head.len - 1]);
            if (args_text.len > 0) {
                var args = std.mem.splitScalar(u8, args_text, ',');
                while (args.next()) |raw| {
                    const arg = trim(raw);
                    if (params.items.len >= 8 or locals.find(arg) != null) return error.Unsupported;
                    _ = try locals.bind(s.p.a, arg);
                    try params.append(s.p.a, arg);
                }
            }
            const id = s.p.functions.items.len;
            if (id >= 32) return error.Unsupported;
            try s.p.functions.append(s.p.a, .{ .name = name, .params = try params.toOwnedSlice(s.p.a) });
            const body = try s.suite(text[colon + 1 ..], indent, &locals, true, depth);
            s.p.functions.items[id].scope = locals;
            s.p.functions.items[id].body = body;
            return .{ .tag = .function, .function = id };
        }
        if (std.mem.startsWith(u8, text, "return ")) {
            if (!in_function) return error.Unsupported;
            return .{ .tag = .ret, .expr = try expr.parse(s.p, trim(text[7..])) };
        }
        if (std.mem.startsWith(u8, text, "assert ")) {
            return .{ .tag = .assertion, .expr = try expr.parse(s.p, trim(text[7..])) };
        }
        if (std.mem.startsWith(u8, text, "raise ")) {
            const id = try expr.parse(s.p, trim(text[6..]));
            const call = s.p.nodes.items[id];
            if (call.tag != .call or !std.mem.eql(u8, call.text, "ValueError") or call.args.len != 1 or
                s.p.nodes.items[call.args[0]].tag != .string) return error.Unsupported;
            return .{ .tag = .raise };
        }
        if (std.mem.eql(u8, text, "from math import gcd")) {
            if (in_function or indent != 0 or s.p.gcd_imported or scope.find("gcd") != null)
                return error.Unsupported;
            s.p.gcd_imported = true;
            return .{ .tag = .math_import };
        }
        const equal = assignment(text);
        if (equal) |pos| {
            var target = trim(text[0..pos]);
            var op: ?ast.Tag = null;
            if (target.len > 0) {
                op = switch (target[target.len - 1]) {
                    '+' => .add,
                    '-' => .sub,
                    '*' => .mul,
                    '%' => .modulo,
                    else => null,
                };
                if (op != null) target = trim(target[0 .. target.len - 1]);
                if (std.mem.endsWith(u8, target, "//")) {
                    op = .floor;
                    target = trim(target[0 .. target.len - 2]);
                }
            }
            if (op != null and (pos == 0 or text[pos - 1] == ' ')) return error.Unsupported;
            _ = try scope.bind(s.p.a, target);
            var id = try expr.parse(s.p, trim(text[pos + 1 ..]));
            if (op) |tag| id = try s.p.node(.{
                .tag = tag,
                .lhs = try s.p.node(.{ .tag = .name, .text = target }),
                .rhs = id,
            });
            try expr.boundedDepth(s.p, id, 0);
            return .{ .tag = .assign, .name = target, .expr = id };
        }
        if (topChar(text, '.')) |dot| {
            if (!std.mem.startsWith(u8, text[dot..], ".append")) return error.Unsupported;
            const name = trim(text[0..dot]);
            if (!ast.validName(name)) return error.Unsupported;
            const tail = trim(text[dot + 7 ..]);
            if (tail.len < 2 or tail[0] != '(' or tail[tail.len - 1] != ')') return error.Unsupported;
            return .{ .tag = .append, .name = name, .expr = try expr.parse(s.p, tail[1 .. tail.len - 1]) };
        }
        const id = try expr.parse(s.p, text);
        const call = s.p.nodes.items[id];
        if (call.tag != .call or !std.mem.eql(u8, call.text, "print")) return error.Unsupported;
        return .{ .tag = .print, .args = call.args };
    }
};

fn parse(p: *ast.Program, source: []const u8) ast.Error!void {
    var lines: std.ArrayList(Line) = .empty;
    var input = std.mem.splitScalar(u8, source, '\n');
    while (input.next()) |raw| {
        const text = std.mem.trimEnd(u8, raw, "\r ");
        if (std.mem.indexOfScalar(u8, text, '\t') != null) return error.Unsupported;
        var indent: usize = 0;
        while (indent < text.len and text[indent] == ' ') indent += 1;
        const comment = commentAt(text[indent..]) orelse text.len - indent;
        const content = trim(text[indent .. indent + comment]);
        if (content.len == 0) continue;
        if (topChar(content, ';') != null and topChar(content, ':') == null) return error.Unsupported;
        try lines.append(p.a, .{ .indent = indent, .text = content });
    }
    if (lines.items.len == 0 or lines.items[0].indent != 0) return error.Unsupported;
    var parser: Parser = .{ .p = p, .lines = lines.items };
    p.body = try parser.block(0, &p.globals, false, 0);
    if (parser.pos != lines.items.len) return error.Unsupported;
    for (p.functions.items) |f| {
        if (p.globals.find(f.name) != null) return error.Unsupported;
        for (f.scope.symbols.items) |sym| if (p.functionId(sym.name) != null) return error.Unsupported;
    }
}

fn trim(text: []const u8) []const u8 {
    return std.mem.trim(u8, text, " ");
}

fn assignment(text: []const u8) ?usize {
    var quote: u8 = 0;
    var nesting: usize = 0;
    var escaped = false;
    for (text, 0..) |c, i| {
        if (quote != 0) {
            if (escaped) escaped = false else if (c == '\\') escaped = true else if (c == quote) quote = 0;
            continue;
        }
        if (c == '\'' or c == '"') {
            quote = c;
            continue;
        }
        if (c == '(' or c == '[') {
            nesting += 1;
            continue;
        }
        if (c == ')' or c == ']') {
            nesting -|= 1;
            continue;
        }
        if (nesting == 0 and c == '=' and (i + 1 == text.len or text[i + 1] != '=') and
            (i == 0 or std.mem.indexOfScalar(u8, "=!<>", text[i - 1]) == null)) return i;
    }
    return null;
}

fn topChar(text: []const u8, target: u8) ?usize {
    var quote: u8 = 0;
    var nesting: usize = 0;
    var escaped = false;
    for (text, 0..) |c, i| {
        if (quote != 0) {
            if (escaped) escaped = false else if (c == '\\') escaped = true else if (c == quote) quote = 0;
            continue;
        }
        if (c == '\'' or c == '"') {
            quote = c;
            continue;
        }
        if (c == target and nesting == 0) return i;
        if (c == '(' or c == '[') nesting += 1;
        if (c == ')' or c == ']') nesting -|= 1;
    }
    return null;
}

fn commentAt(text: []const u8) ?usize {
    var quote: u8 = 0;
    var escaped = false;
    for (text, 0..) |c, i| {
        if (quote != 0) {
            if (escaped) escaped = false else if (c == '\\') escaped = true else if (c == quote) quote = 0;
        } else if (c == '\'' or c == '"') {
            quote = c;
        } else if (c == '#') return i;
    }
    return null;
}

test "cost metadata preserves source lowering and counts dynamic work beyond policy" {
    const t = std.testing;
    const code = "x = 0\nfor i in range(300):\n    x = x + 1\nprint(x)";
    const result = (try lowerWithCost(t.allocator, code)) orelse return error.UnexpectedAbstention;
    defer t.allocator.free(result.source);
    try t.expect(result.estimated_steps > 10_522);
    const legacy = (try lowerPython(t.allocator, code)) orelse return error.UnexpectedAbstention;
    defer t.allocator.free(legacy);
    try t.expectEqualStrings(legacy, result.source);
    try t.expect(std.mem.indexOf(u8, result.source, "for(") != null);
}

test "cost metadata includes printing digits and reachable function calls" {
    const t = std.testing;
    const cases = [_][2][]const u8{
        .{ "print(1)", "print(1234567890)" },
        .{ "print(\"a\")", "print(\"a longer string\")" },
        .{
            "def f(x):\n    return x + 1\nprint(f(0))",
            "def f(x):\n    return x + 1\nprint(f(f(f(0))))",
        },
        .{
            "from math import gcd\nprint(gcd(8, 4))",
            "from math import gcd\nprint(gcd(89, 55))",
        },
        .{
            "values = [1,2,3]\nx = sum(values)\nprint(x)",
            "values = [1,2,3]\nx = sum(values) + sum(values)\nprint(x)",
        },
    };
    for (cases) |pair| {
        const short = (try lowerWithCost(t.allocator, pair[0])) orelse return error.UnexpectedAbstention;
        defer t.allocator.free(short.source);
        const long = (try lowerWithCost(t.allocator, pair[1])) orelse return error.UnexpectedAbstention;
        defer t.allocator.free(long.source);
        try t.expect(short.estimated_steps >= 200);
        try t.expect(long.estimated_steps > short.estimated_steps);
    }
}
