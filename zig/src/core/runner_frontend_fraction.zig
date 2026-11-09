const std = @import("std");
const ast = @import("runner_frontend_integer_ast.zig");
const integer = @import("runner_frontend_integer.zig");
const statements = @import("runner_frontend_integer_statements.zig");
const cost = @import("runner_frontend_integer_cost.zig");
const Allocator = std.mem.Allocator;
const Invalid = ast.Invalid;
const Failure = Allocator.Error || Invalid;

const Kind = enum { integer, fraction, integral_float };
const Value = struct { root: usize, kind: Kind = .integer, denominator: ?usize = null };
const Binding = struct { name: []const u8, value: Value };

// Proofs admit exact integral quotients; runtime nodes retain the original operands and literal text.
pub fn lowerWithCost(a: Allocator, code: []const u8, budget: usize) Allocator.Error!?integer.Lowered {
    return checked(a, code, budget) catch |err| switch (err) {
        error.Unsupported => null,
        error.OutOfMemory => error.OutOfMemory,
    };
}

fn checked(a: Allocator, code: []const u8, budget: usize) Failure!integer.Lowered {
    var script = try parse(code);
    const roots = script.roots[0..script.output_count];
    var steps: ?usize = null;
    if (budget != 0) {
        var extra: usize = 0;
        for (script.styles[0..script.output_count]) |style| {
            if (style == 1) extra += 128;
        }
        if (extra >= budget) return error.Unsupported;
        steps = (cost.estimateMany(&script.parser, roots, budget - extra) orelse return error.Unsupported) + extra;
    }
    return .{
        .source = try integer.lowerFormatted(
            a,
            &script.parser,
            roots,
            script.endings[0..script.output_count],
            script.styles[0..script.output_count],
            script.scales[0..script.output_count],
        ),
        .estimated_steps = steps,
        .table = integer.table(&script.parser, roots, script.endings[0..script.output_count], script.styles[0..script.output_count], script.scales[0..script.output_count]),
    };
}

const Script = struct {
    parser: ast.Parser = .{ .source = "", .require_math_import = true },
    bindings: [32]Binding = undefined,
    binding_count: usize = 0,
    fraction_import: bool = false,
    special: bool = false,
    roots: [8]usize = undefined,
    endings: [8]u8 = undefined,
    styles: [8]u8 = undefined,
    scales: [8]u8 = @splat(0),
    output_count: usize = 0,

    fn lookup(self: *const Script, name: []const u8) ?Value {
        for (self.bindings[0..self.binding_count]) |binding| {
            if (std.mem.eql(u8, name, binding.name)) return binding.value;
        }
        return null;
    }

    fn assign(self: *Script, name: []const u8, rhs: []const u8) Invalid!void {
        if (statements.separator(name, ',')) |comma| {
            const first = trim(name[0..comma]);
            const second = trim(name[comma + 1 ..]);
            if (!validBinding(first) or !validBinding(second) or std.mem.eql(u8, first, second))
                return error.Unsupported;
            const pair = try self.parser.divisionPair(try self.integerExpr(rhs, 0));
            try self.bind(first, .{ .root = pair[0] });
            try self.bind(second, .{ .root = pair[1] });
            return;
        }
        const value = try self.expression(rhs, 0);
        if (self.parser.nodes[value.root].op == .divmod) return error.Unsupported;
        if (value.kind != .integer and self.lookup(unwrap(trim(rhs))) != null) return error.Unsupported;
        try self.bind(name, value);
    }

    fn bind(self: *Script, name: []const u8, value: Value) Invalid!void {
        if (!validBinding(name) or self.lookup(name) != null or self.binding_count == self.bindings.len)
            return error.Unsupported;
        self.bindings[self.binding_count] = .{ .name = name, .value = value };
        self.binding_count += 1;
        if (value.kind == .integer) try self.parser.bind(name, value.root);
    }

    fn print(self: *Script, text: []const u8) Invalid!void {
        var rest = text;
        while (true) {
            const comma = statements.separator(rest, ',');
            if (self.output_count == self.roots.len) return error.Unsupported;
            const value = try self.expression(rest[0 .. comma orelse rest.len], 0);
            const i = self.output_count;
            self.roots[i] = value.root;
            self.endings[i] = if (comma != null) ' ' else '\n';
            self.styles[i] = if (value.kind == .integral_float) 1 else 0;
            self.output_count += 1;
            if (comma) |j| rest = rest[j + 1 ..] else break;
        }
    }

    fn expression(self: *Script, raw: []const u8, depth: usize) Invalid!Value {
        if (depth > 12) return error.Unsupported;
        const text = unwrap(trim(raw));
        if (text.len == 0) return error.Unsupported;
        if (self.lookup(text)) |value| return value;
        if (call(text, "Fraction")) |args| {
            if (!self.fraction_import) return error.Unsupported;
            const comma = statements.separator(args, ',') orelse return error.Unsupported;
            const numerator = try self.integerExpr(args[0..comma], depth + 1);
            const denominator = try self.integerExpr(args[comma + 1 ..], depth + 1);
            const root = try self.quotient(numerator, denominator);
            self.special = true;
            return .{ .root = root, .kind = .fraction, .denominator = denominator };
        }
        if (call(text, "float")) |args| {
            const value = try self.expression(args, depth + 1);
            if (value.kind != .fraction) return error.Unsupported;
            try self.floatRange(value.root);
            self.special = true;
            return .{ .root = value.root, .kind = .integral_float };
        }
        if (std.mem.lastIndexOfScalar(u8, text, '.')) |dot| {
            if (self.lookup(trim(text[0..dot]))) |value| {
                if (value.kind != .fraction) return error.Unsupported;
                const attribute = trim(text[dot + 1 ..]);
                if (std.mem.eql(u8, attribute, "numerator")) return .{ .root = value.root };
                if (!std.mem.eql(u8, attribute, "denominator")) return error.Unsupported;
                const denominator = value.denominator orelse return error.Unsupported;
                const root = try self.append(.{ .op = .floor, .l = denominator, .r = denominator, .value = 1 });
                return .{ .root = root };
            }
        }
        if (singleSlash(text)) |slash| {
            const left = trim(text[0..slash]);
            const right = trim(text[slash + 1 ..]);
            if (!divisionOperand(left) or !divisionOperand(right)) return error.Unsupported;
            const n = try self.integerExpr(left, depth + 1);
            const d = try self.integerExpr(right, depth + 1);
            const root = try self.quotient(n, d);
            try self.floatRange(root);
            self.special = true;
            return .{ .root = root, .kind = .integral_float };
        }
        return .{ .root = try self.integerExpr(text, depth + 1) };
    }

    fn integerExpr(self: *Script, raw: []const u8, depth: usize) Invalid!usize {
        if (depth > 12) return error.Unsupported;
        const text = unwrap(trim(raw));
        if (call(text, "abs")) |arg| {
            const root = try self.integerExpr(arg, depth + 1);
            if (boolean(self.parser.nodes[root].op) or self.parser.nodes[root].op == .divmod)
                return error.Unsupported;
            const value = self.parser.nodes[root].value;
            return self.append(.{ .op = .gcd, .l = root, .r = root, .value = if (value < 0) -value else value });
        }
        const qualified = call(text, "math.gcd");
        const unqualified = call(text, "gcd");
        if (qualified != null or unqualified != null) {
            if (if (qualified != null) !self.parser.math_module else self.parser.math_imports & 1 == 0)
                return error.Unsupported;
            const args = qualified orelse unqualified.?;
            const comma = statements.separator(args, ',') orelse return error.Unsupported;
            const l = try self.integerExpr(args[0..comma], depth + 1);
            const r = try self.integerExpr(args[comma + 1 ..], depth + 1);
            if (self.parser.nodes[l].op == .divmod or self.parser.nodes[r].op == .divmod) return error.Unsupported;
            var x = self.parser.nodes[l].value;
            var y = self.parser.nodes[r].value;
            if (x < 0) x = -x;
            if (y < 0) y = -y;
            while (y != 0) {
                const remainder = @mod(x, y);
                x = y;
                y = remainder;
            }
            return self.append(.{ .op = .gcd, .l = l, .r = r, .value = x });
        }
        self.parser.source = text;
        self.parser.pos = 0;
        return self.parser.parse();
    }

    fn quotient(self: *Script, numerator: usize, denominator: usize) Invalid!usize {
        if (boolean(self.parser.nodes[numerator].op) or boolean(self.parser.nodes[denominator].op) or
            self.parser.nodes[numerator].op == .divmod or self.parser.nodes[denominator].op == .divmod)
            return error.Unsupported;
        const n = self.parser.nodes[numerator].value;
        const d = self.parser.nodes[denominator].value;
        if (d == 0 or @mod(n, d) != 0) return error.Unsupported;
        const q = @divFloor(n, d);
        if (q <= -1_000_000_000_000_000 or q >= 1_000_000_000_000_000) return error.Unsupported;
        return self.append(.{ .op = .floor, .l = numerator, .r = denominator, .value = q });
    }

    fn floatRange(self: *const Script, root: usize) Invalid!void {
        const value = self.parser.nodes[root].value;
        if (value == 0 or value <= -1_000_000_000_000_000 or value >= 1_000_000_000_000_000)
            return error.Unsupported;
    }

    fn append(self: *Script, node: ast.Node) Invalid!usize {
        if (self.parser.count == self.parser.capacity) return error.Unsupported;
        const root = self.parser.count;
        self.parser.nodes[root] = node;
        self.parser.proved[root] = true;
        self.parser.count += 1;
        return root;
    }
};

fn parse(code: []const u8) Invalid!Script {
    if (code.len == 0 or code.len > 4096 or std.mem.indexOfScalar(u8, code, '\r') != null)
        return error.Unsupported;
    var script: Script = .{};
    var lines = statements.Iterator{ .code = code };
    while (try lines.next()) |raw| {
        var line = std.mem.trimEnd(u8, raw.text, " \t");
        if (raw.after_semicolon) line = std.mem.trimStart(u8, line, " \t");
        if (line.len == 0) continue;
        if (line[0] == ' ' or line[0] == '\t') return error.Unsupported;
        if (std.mem.eql(u8, line, "from fractions import Fraction")) {
            script.fraction_import = true;
        } else if (std.mem.eql(u8, line, "import math")) {
            script.parser.math_module = true;
        } else if (std.mem.startsWith(u8, line, "from math import ")) {
            var names = std.mem.splitScalar(u8, line[17..], ',');
            while (names.next()) |part| {
                const name = trim(part);
                script.parser.math_imports |= if (std.mem.eql(u8, name, "gcd")) @as(u8, 1) else if (std.mem.eql(u8, name, "isqrt"))
                    @as(u8, 2)
                else if (std.mem.eql(u8, name, "factorial"))
                    @as(u8, 4)
                else if (std.mem.eql(u8, name, "prod"))
                    @as(u8, 8)
                else
                    return error.Unsupported;
            }
        } else if (call(line, "print")) |args| {
            try script.print(args);
        } else if (statements.separator(line, '=')) |equal| {
            try script.assign(trim(line[0..equal]), line[equal + 1 ..]);
        } else return error.Unsupported;
    }
    if (!script.special or script.output_count == 0) return error.Unsupported;
    return script;
}

fn trim(text: []const u8) []const u8 {
    return std.mem.trim(u8, text, " \t");
}

fn call(text: []const u8, name: []const u8) ?[]const u8 {
    if (!std.mem.startsWith(u8, text, name)) return null;
    const rest = trim(text[name.len..]);
    if (rest.len < 2 or rest[0] != '(' or close(rest) != rest.len - 1) return null;
    return rest[1 .. rest.len - 1];
}

fn close(text: []const u8) ?usize {
    var depth: usize = 0;
    var quote: u8 = 0;
    var escaped = false;
    for (text, 0..) |c, i| {
        if (quote != 0) {
            if (escaped) escaped = false else if (c == '\\') escaped = true else if (c == quote) quote = 0;
        } else if (c == '\'' or c == '"') {
            quote = c;
        } else if (c == '(') {
            depth += 1;
        } else if (c == ')') {
            if (depth == 0) return null;
            depth -= 1;
            if (depth == 0) return i;
        }
    }
    return null;
}

fn unwrap(raw: []const u8) []const u8 {
    var text = raw;
    while (text.len >= 2 and text[0] == '(' and close(text) == text.len - 1)
        text = trim(text[1 .. text.len - 1]);
    return text;
}

fn singleSlash(text: []const u8) ?usize {
    const i = statements.separator(text, '/') orelse return null;
    if ((i > 0 and text[i - 1] == '/') or (i + 1 < text.len and text[i + 1] == '/')) return null;
    if (statements.separator(text[i + 1 ..], '/') != null) return null;
    return i;
}

// Restrict true division to two atoms; accepting a surrounding sum would change Python's float arithmetic.
fn divisionOperand(raw: []const u8) bool {
    var text = trim(raw);
    if (text.len == 0) return false;
    if (text[0] == '+' or text[0] == '-') text = trim(text[1..]);
    if (text.len == 0) return false;
    for ("+-*/%<>=,&|") |token| if (statements.separator(text, token) != null) return false;
    return true;
}

fn boolean(op: ast.Op) bool {
    return switch (op) {
        .eq, .ne, .lt, .le, .gt, .ge => true,
        else => false,
    };
}

fn validBinding(name: []const u8) bool {
    if (name.len == 0 or !(std.ascii.isAlphabetic(name[0]) or name[0] == '_')) return false;
    for (name) |c| if (!(std.ascii.isAlphanumeric(c) or c == '_')) return false;
    for ([_][]const u8{
        "Fraction", "float",  "fractions", "print",  "pow",   "divmod",   "int",   "str",   "len",   "sum",       "map",          "range",    "all", "abs",    "math",
        "gcd",      "isqrt",  "factorial", "prod",   "True",  "False",    "None",  "and",   "or",    "not",       "if",           "else",     "for", "while",  "in",
        "is",       "def",    "return",    "import", "from",  "as",       "class", "del",   "yield", "lambda",    "global",       "nonlocal", "try", "except", "finally",
        "with",     "assert", "raise",     "pass",   "break", "continue", "async", "await", "elif",  "__debug__", "__builtins__",
    }) |reserved| if (std.mem.eql(u8, name, reserved)) return false;
    return true;
}
