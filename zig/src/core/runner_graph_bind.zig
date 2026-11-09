//! Exact integer source binding onto prepared graph kernels, independent of the Python/RV32 frontend.
const std = @import("std");
const packet = @import("runner_graph_packet.zig");
const Bank = @import("runner_graph_bank.zig").Bank;
const Allocator = std.mem.Allocator;
const Failure = error{ OutOfMemory, Unsupported };

pub const Syntax = enum { math, python };
const Kind = enum { literal, add, sub, mul, div, mod, pow, powmod, divmod, gcd, isqrt, neg };
const Node = struct { kind: Kind, a: u32 = 0, b: u32 = 0, c: u32 = 0, text: []const u8 = "" };
const Bound = struct { ref: u32, digits: usize, second: ?u32 = null };

/// Only empty-context Python inline/REPL cells qualify; callers retain their ordinary refusal path.
pub fn cell(a: Allocator, bank: *const Bank, request: anytype) Allocator.Error!?[]u8 {
    if (request.prior.len != 0 or !std.mem.eql(u8, request.lang, "python")) return null;
    if (request.form == .fence) return script(a, bank, request.code);
    const syntax: Syntax = if (request.form == .inline_expr) .math else if (request.form == .repl) .python else return null;
    var source = std.mem.trim(u8, request.code, " \t");
    if (request.form == .repl and std.mem.startsWith(u8, source, ">>>")) source = std.mem.trim(u8, source[3..], " \t");
    if (try expression(a, bank, source, syntax)) |bound| return bound;
    if (syntax == .math) return @import("runner_graph_exact.zig").expression(a, bank, source);
    return if (request.form == .repl) script(a, bank, source) else null;
}

/// Parses literals and selects schedules; no expression answer, circuit weights or executable IR is built here.
pub fn expression(a: Allocator, bank: *const Bank, source: []const u8, syntax: Syntax) Allocator.Error!?[]u8 {
    if (source.len == 0 or source.len > 4096 or std.mem.indexOfAny(u8, source, "\x00\r\n") != null) return null;
    var arena_state = std.heap.ArenaAllocator.init(a);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var p: Parser = .{ .a = arena, .source = source, .syntax = syntax };
    const root = p.sum() catch |e| return if (e == error.OutOfMemory) error.OutOfMemory else null;
    p.space();
    if (p.at != source.len) return null;
    var lower: Lower = .{ .a = arena, .bank = bank, .nodes = p.nodes.items };
    const result = lower.build(root) catch |e| return if (e == error.OutOfMemory) error.OutOfMemory else null;
    _ = bank.validate(lower.values.items.len, lower.ops.items, result.ref) catch return null;
    return packet.Builder.program(a, lower.values.items, lower.ops.items, &.{.{ .ref = result.ref, .style = if (result.second != null) .tuple else .integer, .second = result.second }}) catch |e| return if (e == error.OutOfMemory) error.OutOfMemory else null;
}

/// Straight-line integer cells retain original operands, assignments, declared math intrinsics and print order.
pub fn script(a: Allocator, bank: *const Bank, source: []const u8) Allocator.Error!?[]u8 {
    if (source.len == 0 or source.len > 4096 or std.mem.indexOfAny(u8, source, "\x00\r") != null) return null;
    var arena_state = std.heap.ArenaAllocator.init(a);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var p: Parser = .{ .a = arena, .source = "", .syntax = .python };
    var roots: std.ArrayList(struct { node: u32, ending: packet.Ending }) = .empty;
    var schedule: std.ArrayList(u32) = .empty;
    const statements = splitStatements(arena, source) catch |e| return if (e == error.OutOfMemory) error.OutOfMemory else null;
    for (statements) |raw| {
        const text = std.mem.trim(u8, raw, " \t");
        if (text.len == 0) continue;
        if (std.mem.eql(u8, text, "import math")) {
            if (p.names.contains("math")) return null;
            p.math_module = true;
            continue;
        }
        if (std.mem.startsWith(u8, text, "from math import ")) {
            var names = std.mem.splitScalar(u8, text[17..], ',');
            while (names.next()) |name| {
                const n = std.mem.trim(u8, name, " \t");
                if (p.names.contains(n)) return null;
                if (std.mem.eql(u8, n, "gcd")) p.gcd_import = true else if (std.mem.eql(u8, n, "isqrt")) p.isqrt_import = true else return null;
            }
            continue;
        }
        p.at = 0;
        if (std.mem.startsWith(u8, text, "print(") and text[text.len - 1] == ')') {
            p.source = text[6 .. text.len - 1];
            while (true) {
                const root = p.sum() catch |e| return if (e == error.OutOfMemory) error.OutOfMemory else null;
                try schedule.append(arena, root);
                const more = p.take(",");
                if (roots.items.len >= 128) return null;
                try roots.append(arena, .{ .node = root, .ending = if (more) .space else .newline });
                if (!more) break;
            }
        } else {
            const equal = std.mem.indexOfScalar(u8, text, '=') orelse return null;
            const name = std.mem.trim(u8, text[0..equal], " \t");
            if (!identifier(name) or reserved(name)) return null;
            p.source = text[equal + 1 ..];
            const root = p.sum() catch |e| return if (e == error.OutOfMemory) error.OutOfMemory else null;
            try schedule.append(arena, root);
            try p.names.put(arena, name, root);
        }
        p.space();
        if (p.at != p.source.len) return null;
    }
    if (roots.items.len == 0) return null;
    var lower: Lower = .{ .a = arena, .bank = bank, .nodes = p.nodes.items };
    lower.initialize() catch |e| return if (e == error.OutOfMemory) error.OutOfMemory else null;
    for (schedule.items) |root| _ = lower.emit(root) catch |e| return if (e == error.OutOfMemory) error.OutOfMemory else null;
    var prints: std.ArrayList(packet.Print) = .empty;
    for (roots.items) |root| {
        const value = lower.emit(root.node) catch |e| return if (e == error.OutOfMemory) error.OutOfMemory else null;
        try prints.append(arena, .{ .ref = value.ref, .ending = root.ending, .style = if (value.second != null) .tuple else .integer, .second = value.second });
    }
    _ = bank.validate(lower.values.items.len, lower.ops.items, prints.items[0].ref) catch return null;
    return packet.Builder.program(a, lower.values.items, lower.ops.items, prints.items) catch |e| return if (e == error.OutOfMemory) error.OutOfMemory else null;
}

fn identifier(text: []const u8) bool {
    if (text.len == 0 or !(std.ascii.isAlphabetic(text[0]) or text[0] == '_')) return false;
    for (text) |ch| if (!(std.ascii.isAlphanumeric(ch) or ch == '_')) return false;
    return true;
}

fn splitStatements(a: Allocator, source: []const u8) Failure![]const []const u8 {
    var statements: std.ArrayList([]const u8) = .empty;
    var lines = std.mem.splitScalar(u8, source, '\n');
    while (lines.next()) |line| {
        if (std.mem.trim(u8, line, " \t").len == 0) continue;
        if (line[0] == ' ' or line[0] == '\t') return error.Unsupported;
        var pieces = std.mem.splitScalar(u8, line, ';');
        var seen = false;
        var trailing = false;
        while (pieces.next()) |piece| {
            const text = std.mem.trim(u8, piece, " \t");
            if (text.len == 0) {
                if (!seen or trailing) return error.Unsupported;
                trailing = true;
                continue;
            }
            if (trailing) return error.Unsupported;
            try statements.append(a, text);
            seen = true;
        }
    }
    return statements.items;
}

fn reserved(text: []const u8) bool {
    for ([_][]const u8{ "__debug__", "math", "pow", "divmod", "print", "gcd", "isqrt", "False", "None", "True", "and", "as", "assert", "async", "await", "break", "class", "continue", "def", "del", "elif", "else", "except", "finally", "for", "from", "global", "if", "import", "in", "is", "lambda", "nonlocal", "not", "or", "pass", "raise", "return", "try", "while", "with", "yield" }) |name|
        if (std.mem.eql(u8, text, name)) return true;
    return false;
}

const Parser = struct {
    a: Allocator,
    source: []const u8,
    syntax: Syntax,
    at: usize = 0,
    depth: usize = 0,
    nodes: std.ArrayList(Node) = .empty,
    names: std.StringHashMapUnmanaged(u32) = .empty,
    math_module: bool = false,
    gcd_import: bool = false,
    isqrt_import: bool = false,

    fn space(p: *Parser) void {
        while (p.at < p.source.len and (p.source[p.at] == ' ' or p.source[p.at] == '\t')) p.at += 1;
    }

    fn take(p: *Parser, text: []const u8) bool {
        p.space();
        if (!std.mem.startsWith(u8, p.source[p.at..], text)) return false;
        p.at += text.len;
        return true;
    }

    fn push(p: *Parser, node: Node) Failure!u32 {
        if (p.nodes.items.len >= 256) return error.Unsupported;
        try p.nodes.append(p.a, node);
        return @intCast(p.nodes.items.len - 1);
    }

    fn sum(p: *Parser) Failure!u32 {
        var lhs = try p.product();
        while (true) {
            const op: Kind = if (p.take("+")) .add else if (p.take("-")) .sub else break;
            lhs = try p.push(.{ .kind = op, .a = lhs, .b = try p.product() });
        }
        return lhs;
    }

    fn product(p: *Parser) Failure!u32 {
        var lhs = try p.unary();
        while (true) {
            p.space();
            const word_mod = p.syntax == .math and p.at > 0 and
                std.ascii.isWhitespace(p.source[p.at - 1]) and
                std.mem.startsWith(u8, p.source[p.at..], "mod") and
                p.at + 3 < p.source.len and std.ascii.isWhitespace(p.source[p.at + 3]);
            const op: Kind = if (p.take("//")) .div else if (p.take("%")) blk: {
                p.space();
                if (p.syntax == .math and p.at < p.source.len and
                    (p.source[p.at] == '+' or p.source[p.at] == '-')) return error.Unsupported;
                break :blk .mod;
            } else if (word_mod) blk: {
                p.at += 3;
                break :blk .mod;
            } else if (p.take("*")) .mul else break;
            lhs = try p.push(.{ .kind = op, .a = lhs, .b = try p.unary() });
        }
        return lhs;
    }

    fn unary(p: *Parser) Failure!u32 {
        if (p.depth >= 32) return error.Unsupported;
        p.depth += 1;
        defer p.depth -= 1;
        if (p.take("+")) return p.unary();
        if (p.take("-")) return p.push(.{ .kind = .neg, .a = try p.unary() });
        const lhs = try p.atom();
        if (p.take("**") or (p.syntax == .math and p.take("^")))
            return p.push(.{ .kind = .pow, .a = lhs, .b = try p.unary() });
        return lhs;
    }

    fn atom(p: *Parser) Failure!u32 {
        if (p.take("(")) {
            const result = try p.sum();
            if (!p.take(")")) return error.Unsupported;
            return result;
        }
        if (p.take("divmod(")) {
            const a = try p.sum();
            if (!p.take(",")) return error.Unsupported;
            const b = try p.sum();
            if (!p.take(")")) return error.Unsupported;
            return p.push(.{ .kind = .divmod, .a = a, .b = b });
        }
        if (p.take("pow(")) {
            const a = try p.sum();
            if (!p.take(",")) return error.Unsupported;
            const b = try p.sum();
            if (!p.take(",")) return error.Unsupported;
            const c = try p.sum();
            if (!p.take(")")) return error.Unsupported;
            return p.push(.{ .kind = .powmod, .a = a, .b = b, .c = c });
        }
        p.space();
        if (p.at < p.source.len and (std.ascii.isAlphabetic(p.source[p.at]) or p.source[p.at] == '_')) {
            const start = p.at;
            while (p.at < p.source.len and (std.ascii.isAlphanumeric(p.source[p.at]) or p.source[p.at] == '_' or p.source[p.at] == '.')) p.at += 1;
            const name = p.source[start..p.at];
            if (p.names.get(name)) |ref| return ref;
            const qualified = std.mem.startsWith(u8, name, "math.");
            if (qualified and !p.math_module) return error.Unsupported;
            const bare = if (qualified) name[5..] else name;
            const kind: Kind = if (std.mem.eql(u8, bare, "gcd") and (qualified or p.gcd_import or p.syntax == .math)) .gcd else if (std.mem.eql(u8, bare, "isqrt") and (qualified or p.isqrt_import or p.syntax == .math)) .isqrt else return error.Unsupported;
            if (!p.take("(")) return error.Unsupported;
            const a = try p.sum();
            var b: u32 = 0;
            if (kind == .gcd) {
                if (!p.take(",")) return error.Unsupported;
                b = try p.sum();
            }
            if (!p.take(")")) return error.Unsupported;
            return p.push(.{ .kind = kind, .a = a, .b = b });
        }
        p.space();
        const start = p.at;
        const hex = std.mem.startsWith(u8, p.source[start..], "0x") or std.mem.startsWith(u8, p.source[start..], "0X");
        if (hex) p.at += 2;
        const digits = p.at;
        while (p.at < p.source.len and (if (hex) std.ascii.isHex(p.source[p.at]) else std.ascii.isDigit(p.source[p.at]))) p.at += 1;
        if (p.at == digits or p.at - start > 512) return error.Unsupported;
        const text = p.source[start..p.at];
        if (!hex and text.len > 1 and text[0] == '0')
            for (text) |ch| if (ch != '0') return error.Unsupported;
        const value = std.fmt.parseInt(u1024, if (hex) text[2..] else text, if (hex) 16 else 10) catch return error.Unsupported;
        const decimal = try std.fmt.allocPrint(p.a, "{d}", .{value});
        if (decimal.len > packet.max_digits) return error.Unsupported;
        return p.push(.{ .kind = .literal, .text = decimal });
    }
};

const Lower = struct {
    a: Allocator,
    bank: *const Bank,
    nodes: []const Node,
    values: std.ArrayList([]const u8) = .empty,
    ops: std.ArrayList(packet.Op) = .empty,
    literals: [256]u32 = @splat(0),
    emitted: [256]?Bound = @splat(null),
    produced: usize = 0,

    fn build(l: *Lower, root: u32) Failure!Bound {
        try l.initialize();
        return l.emit(root);
    }

    fn initialize(l: *Lower) Failure!void {
        try l.values.appendSlice(l.a, &.{ "0", "1" });
        for (l.nodes, 0..) |node, i| if (node.kind == .literal) {
            if (node.text.len > 128 and l.bank.find("format", 256) == null) return error.Unsupported;
            l.literals[i] = @intCast(l.values.items.len);
            try l.values.append(l.a, node.text);
        };
    }

    fn width(digits: usize) Failure!u16 {
        for ([_]u16{ 1, 2, 4, 8, 16, 32, 62, 128 }) |w| if (digits <= w) return w;
        // Conservative bounds may saturate; checked compiled kernels still refuse actual overflow.
        return 256;
    }

    fn operation(l: *Lower, name: []const u8, args: []const u32, digits: usize) Failure!Bound {
        const w = try width(digits);
        if (l.ops.items.len >= 1024) return error.Unsupported;
        if (std.mem.eql(u8, name, "powmod") or std.mem.eql(u8, name, "gcd") or std.mem.eql(u8, name, "isqrt")) {
            const plan = try std.fmt.allocPrint(l.a, "{s}-{d}", .{ name, w });
            if (l.bank.plan(plan) == null) return error.Unsupported;
        } else if (l.bank.find(name, w) == null) return error.Unsupported;
        const refs = try l.a.dupe(u32, args);
        const ref: u32 = @intCast(l.values.items.len + l.produced);
        try l.ops.append(l.a, .{ .op = name, .width = w, .args = refs, .checked = true });
        l.produced += 1;
        return .{ .ref = ref, .digits = @min(digits, packet.max_digits) };
    }

    fn modular(l: *Lower, a: u32, b: u32, c: u32) Failure!Bound {
        const base = try l.emit(a);
        const exponent = try l.emit(b);
        const modulus = try l.emit(c);
        if (base.second != null or exponent.second != null or modulus.second != null) return error.Unsupported;
        return l.operation("powmod", &.{ base.ref, exponent.ref, modulus.ref }, @max(base.digits, @max(exponent.digits, modulus.digits)));
    }

    fn emit(l: *Lower, index: u32) Failure!Bound {
        if (l.emitted[index]) |value| return value;
        const value = try l.emitFresh(index);
        l.emitted[index] = value;
        return value;
    }

    fn emitFresh(l: *Lower, index: u32) Failure!Bound {
        const node = l.nodes[index];
        if (node.kind == .literal) return .{ .ref = l.literals[index], .digits = node.text.len };
        if (node.kind == .mod and l.nodes[node.a].kind == .pow) {
            const power = l.nodes[node.a];
            return l.modular(power.a, power.b, node.b);
        }
        if (node.kind == .powmod) return l.modular(node.a, node.b, node.c);
        const lhs = try l.emit(node.a);
        if (lhs.second != null) return error.Unsupported;
        if (node.kind == .neg) return l.operation("sub", &.{ 0, lhs.ref }, lhs.digits);
        if (node.kind == .isqrt) return l.operation("isqrt", &.{lhs.ref}, lhs.digits);
        if (node.kind == .pow) {
            const exponent = l.nodes[node.b];
            if (exponent.kind != .literal) return error.Unsupported;
            var n = std.fmt.parseInt(u16, exponent.text, 10) catch return error.Unsupported;
            if (n > 999) return error.Unsupported;
            var result: Bound = .{ .ref = 1, .digits = 1 };
            var factor = lhs;
            while (n != 0) {
                if (n & 1 != 0) result = try l.operation("mul", &.{ result.ref, factor.ref }, result.digits + factor.digits);
                n >>= 1;
                if (n != 0) factor = try l.operation("mul", &.{ factor.ref, factor.ref }, 2 * factor.digits);
            }
            return result;
        }
        const rhs = try l.emit(node.b);
        if (rhs.second != null) return error.Unsupported;
        if (node.kind == .gcd) return l.operation("gcd", &.{ lhs.ref, rhs.ref }, @max(lhs.digits, rhs.digits));
        if (node.kind == .div or node.kind == .mod or node.kind == .divmod) {
            const w = try width(@max(lhs.digits, rhs.digits));
            const plan = try std.fmt.allocPrint(l.a, "divmod-{d}", .{w});
            if (l.bank.plan(plan) == null or l.ops.items.len >= 1024) return error.Unsupported;
            const refs = try l.a.dupe(u32, &.{ lhs.ref, rhs.ref });
            const ref: u32 = @intCast(l.values.items.len + l.produced);
            try l.ops.append(l.a, .{ .op = "divmod", .args = refs, .outputs = 2, .width = w, .checked = true });
            l.produced += 2;
            return .{ .ref = ref + @intFromBool(node.kind == .mod), .digits = @max(lhs.digits, rhs.digits), .second = if (node.kind == .divmod) ref + 1 else null };
        }
        const digits = switch (node.kind) {
            .add, .sub => @max(lhs.digits, rhs.digits) + 1,
            .mul => lhs.digits + rhs.digits,
            .div, .mod => @max(lhs.digits, rhs.digits),
            else => unreachable,
        };
        return l.operation(@tagName(node.kind), &.{ lhs.ref, rhs.ref }, digits);
    }
};
