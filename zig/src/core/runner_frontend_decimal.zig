const std = @import("std");
const ast = @import("runner_frontend_integer_ast.zig");
const integer = @import("runner_frontend_integer.zig");
const statements = @import("runner_frontend_integer_statements.zig");
const Invalid = ast.Invalid;
const Value = struct { root: usize, scale: u8 = 0, negative: bool = false, decimal: bool = false };

pub fn lowerWithCost(a: std.mem.Allocator, code: []const u8, budget: usize) std.mem.Allocator.Error!?integer.Lowered {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var p = Parser{ .a = arena.allocator(), .nodes = .{ .source = "" } };
    return p.script(a, code, budget) catch |err| switch (err) {
        error.Unsupported => null,
        error.OutOfMemory => error.OutOfMemory,
    };
}

const Parser = struct {
    a: std.mem.Allocator,
    nodes: ast.Parser,
    source: []const u8 = "",
    pos: usize = 0,
    imported: bool = false,
    names: [32][]const u8 = undefined,
    values: [32]Value = undefined,
    count: usize = 0,

    fn script(p: *Parser, a: std.mem.Allocator, code: []const u8, budget: usize) !?integer.Lowered {
        if (code.len == 0 or code.len > 4096 or std.mem.indexOfScalar(u8, code, '\r') != null) return error.Unsupported;
        var lines = statements.Iterator{ .code = code };
        var roots: [8]usize = undefined;
        var styles: [8]u8 = undefined;
        var scales: [8]u8 = undefined;
        var count: usize = 0;
        while (try lines.next()) |line| {
            var text = std.mem.trimEnd(u8, line.text, " \t");
            if (line.after_semicolon) text = std.mem.trimStart(u8, text, " \t");
            if (text.len == 0) continue;
            if (text[0] == ' ' or text[0] == '\t') return error.Unsupported;
            if (std.mem.eql(u8, text, "from decimal import Decimal")) {
                if (p.imported or p.count != 0 or count != 0) return error.Unsupported;
                p.imported = true;
            } else if (statements.separator(text, '=')) |i| {
                const name = std.mem.trim(u8, text[0..i], " \t");
                if (!@import("runner_frontend_python_ast.zig").validName(name) or std.mem.eql(u8, name, "Decimal"))
                    return error.Unsupported;
                const value = try p.parse(text[i + 1 ..]);
                var existing: ?usize = null;
                for (p.names[0..p.count], 0..) |old, j| if (std.mem.eql(u8, old, name)) {
                    existing = j;
                    break;
                };
                if (existing) |j| p.values[j] = value else {
                    if (p.count == p.names.len) return error.Unsupported;
                    p.names[p.count] = name;
                    p.values[p.count] = value;
                    p.count += 1;
                }
            } else {
                if (!std.mem.startsWith(u8, text, "print(") or text[text.len - 1] != ')' or count == roots.len)
                    return error.Unsupported;
                const value = try p.parse(text[6 .. text.len - 1]);
                if (!value.decimal) return error.Unsupported;
                if (p.nodes.nodes[value.root].value == 0 and value.negative) return error.Unsupported;
                roots[count] = value.root;
                styles[count] = 2;
                scales[count] = value.scale;
                count += 1;
            }
        }
        if (!p.imported or count == 0) return error.Unsupported;
        var steps: ?usize = null;
        if (budget != 0) {
            const extra = count * 256;
            if (extra >= budget) return error.Unsupported;
            steps = (@import("runner_frontend_integer_cost.zig").estimateMany(&p.nodes, roots[0..count], budget - extra) orelse return error.Unsupported) + extra;
        }
        const endings: [8]u8 = @splat('\n');
        const source = try integer.lowerFormatted(a, &p.nodes, roots[0..count], endings[0..count], styles[0..count], scales[0..count]);
        return .{ .source = source, .estimated_steps = steps, .table = integer.table(&p.nodes, roots[0..count], endings[0..count], styles[0..count], scales[0..count]) };
    }

    fn parse(p: *Parser, text: []const u8) !Value {
        p.source = std.mem.trim(u8, text, " \t");
        p.pos = 0;
        const value = try p.product(0);
        p.space();
        if (p.pos != p.source.len) return error.Unsupported;
        return value;
    }

    fn space(p: *Parser) void {
        while (p.pos < p.source.len and p.source[p.pos] == ' ') : (p.pos += 1) {}
    }
    fn eat(p: *Parser, token: []const u8) bool {
        p.space();
        if (!std.mem.startsWith(u8, p.source[p.pos..], token)) return false;
        p.pos += token.len;
        return true;
    }
    fn need(p: *Parser, token: []const u8) !void {
        if (!p.eat(token)) return error.Unsupported;
    }
    fn node(p: *Parser, text: []const u8) !usize {
        p.nodes.source = text;
        p.nodes.pos = 0;
        return p.nodes.parse();
    }
    fn product(p: *Parser, depth: usize) (Invalid || std.mem.Allocator.Error)!Value {
        if (depth > 8) return error.Unsupported;
        var left = try p.atom(depth + 1);
        while (p.eat("*")) {
            const right = try p.atom(depth + 1);
            if (left.scale + right.scale > 6) return error.Unsupported;
            try p.nodes.bind("_decimal_l", left.root);
            try p.nodes.bind("_decimal_r", right.root);
            const root = try p.node("_decimal_l*_decimal_r");
            var magnitude = p.nodes.nodes[root].value;
            if (magnitude < 0) magnitude = -magnitude;
            if (magnitude >= std.math.pow(i512, 10, 28)) return error.Unsupported;
            const typed_decimal = left.decimal or right.decimal;
            left = .{ .root = root, .scale = left.scale + right.scale, .negative = (left.negative != right.negative) and (typed_decimal or magnitude != 0), .decimal = typed_decimal };
        }
        return left;
    }
    fn atom(p: *Parser, depth: usize) (Invalid || std.mem.Allocator.Error)!Value {
        if (depth > 8) return error.Unsupported;
        if (p.eat("(")) {
            const value = try p.product(depth + 1);
            try p.need(")");
            return value;
        }
        if (p.eat("Decimal(")) {
            if (!p.imported) return error.Unsupported;
            p.space();
            if (p.pos == p.source.len or (p.source[p.pos] != '\'' and p.source[p.pos] != '"')) return error.Unsupported;
            const quote = p.source[p.pos];
            p.pos += 1;
            const start = p.pos;
            while (p.pos < p.source.len and p.source[p.pos] != quote) : (p.pos += 1) {}
            if (p.pos == p.source.len) return error.Unsupported;
            const text = p.source[start..p.pos];
            p.pos += 1;
            try p.need(")");
            return p.literal(text, true);
        }
        p.space();
        const start = p.pos;
        if (p.pos < p.source.len and (p.source[p.pos] == '-' or p.source[p.pos] == '+')) p.pos += 1;
        while (p.pos < p.source.len and std.ascii.isDigit(p.source[p.pos])) : (p.pos += 1) {}
        if (p.pos > start and std.ascii.isDigit(p.source[p.pos - 1])) return p.literal(p.source[start..p.pos], false);
        p.pos = start;
        while (p.pos < p.source.len and (std.ascii.isAlphanumeric(p.source[p.pos]) or p.source[p.pos] == '_')) : (p.pos += 1) {}
        const name = p.source[start..p.pos];
        for (p.names[0..p.count], p.values[0..p.count]) |old, value| if (std.mem.eql(u8, old, name)) return value;
        return error.Unsupported;
    }
    fn literal(p: *Parser, text: []const u8, decimal: bool) !Value {
        if (text.len == 0 or text.len > 64) return error.Unsupported;
        var digits: [64]u8 = undefined;
        var len: usize = 0;
        var scale: u8 = 0;
        var point = false;
        const sign = text[0] == '-' or text[0] == '+';
        for (text[@intFromBool(sign)..]) |c| {
            if (c == '.' and decimal and !point) {
                point = true;
                continue;
            }
            if (!std.ascii.isDigit(c)) return error.Unsupported;
            digits[len] = c;
            len += 1;
            if (point) scale += 1;
        }
        if (len == 0 or scale > 6) return error.Unsupported;
        var first: usize = 0;
        while (first + 1 < len and digits[first] == '0') : (first += 1) {}
        if (len - first > 28 or (!decimal and len > 1 and digits[0] == '0')) return error.Unsupported;
        const negative = text[0] == '-' and (decimal or digits[first] != '0' or len - first > 1);
        const coefficient = try std.fmt.allocPrint(p.a, "{s}{s}", .{ if (negative) @as([]const u8, "-") else "", digits[first..len] });
        return .{ .root = try p.node(coefficient), .scale = scale, .negative = negative, .decimal = decimal };
    }
};
