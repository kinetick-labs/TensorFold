const std = @import("std");
const ast = @import("runner_frontend_inline_ast.zig");
const calc = @import("runner_frontend_calc.zig");
const Allocator = std.mem.Allocator;
const Failure = Allocator.Error || ast.Invalid;

// Proof values select a format and admission only; emitted expressions retain source arithmetic.
pub fn lower(a: Allocator, code: []const u8, exact: bool) Allocator.Error!?[]u8 {
    if (code.len == 0 or code.len > 1024) return null;
    // `^` is a power in math notation (Python's XOR), so only exact inline math takes it
    if (!exact and std.mem.indexOfScalar(u8, code, '^') != null) return null;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const scratch = arena.allocator();
    const source = try normalize(scratch, code) orelse return null;
    var parser = ast.Parser{ .source = source };
    const root = parser.parse() catch return null;
    if (!exact and parser.exact_syntax) return null;
    const emitter = Emitter{ .a = scratch, .parser = &parser };
    if (!parser.decimal_context) {
        const canonical = emitter.canonical(root) catch |err| return failure(err);
        return calc.lower(a, canonical);
    }
    // Keep decimal arithmetic integral until the final scale; firmware can miscompare a shifted zero.
    const k = finiteScale(parser.nodes[root].value.d) orelse return null;
    const rep = emitter.represent(root) catch |err| return failure(err);
    const integer = emitter.scaledInteger(rep, k) catch |err| return failure(err);
    const output = if (k == 0) integer else blk: {
        const reciprocal = try scratch.alloc(u8, k + 2);
        reciprocal[0] = '0';
        reciprocal[1] = '.';
        @memset(reciprocal[2..], '0');
        reciprocal[k + 1] = '1';
        break :blk emitter.binary(integer, "*", reciprocal) catch |err| return failure(err);
    };
    return calc.lower(a, output);
}

fn failure(err: Failure) Allocator.Error!?[]u8 {
    return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        error.Unsupported => null,
    };
}

fn normalize(a: Allocator, code: []const u8) Allocator.Error!?[]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    for (code) |c| if (c == '\n' or c == '\r') return null;
    var pos: usize = 0;
    while (pos < code.len) {
        var replaced = false;
        const symbols = [_][]const u8{ "×", "÷", "−", "^" };
        const operators = [_][]const u8{ "*", "/", "-", "**" };
        for (symbols, operators) |symbol, op| {
            if (std.mem.startsWith(u8, code[pos..], symbol)) {
                try out.appendSlice(a, op);
                pos += symbol.len;
                replaced = true;
                break;
            }
        }
        if (!replaced) {
            try out.append(a, code[pos]);
            pos += 1;
        }
    }
    return try out.toOwnedSlice(a);
}

fn finiteScale(denominator: i512) ?usize {
    var d = denominator;
    var two: usize = 0;
    var five: usize = 0;
    while (@mod(d, 2) == 0) : (two += 1) d = @divTrunc(d, 2);
    while (@mod(d, 5) == 0) : (five += 1) d = @divTrunc(d, 5);
    const k = @max(two, five);
    return if (d == 1 and k <= 61) k else null;
}

const Rep = struct { n: []const u8, d: ?[]const u8 = null, scale: i32 = 0 };
const Emitter = struct {
    a: Allocator,
    parser: *const ast.Parser,

    fn binary(self: Emitter, left: []const u8, op: []const u8, right: []const u8) Failure![]const u8 {
        if (left.len + right.len + op.len + 2 > 255) return error.Unsupported;
        return std.fmt.allocPrint(self.a, "({s}{s}{s})", .{ left, op, right });
    }

    fn times(self: Emitter, left: []const u8, right: []const u8) Failure![]const u8 {
        if (std.mem.eql(u8, left, "1")) return right;
        if (std.mem.eql(u8, right, "1")) return left;
        return self.binary(left, "*", right);
    }

    fn denominators(self: Emitter, left: ?[]const u8, right: ?[]const u8) Failure!?[]const u8 {
        if (left == null) return right;
        if (right == null) return left;
        return try self.times(left.?, right.?);
    }

    fn ten(self: Emitter, exponent: i32) Failure![]const u8 {
        if (exponent < 0 or exponent > 61) return error.Unsupported;
        const text = try self.a.alloc(u8, @as(usize, @intCast(exponent)) + 1);
        @memset(text, '0');
        text[0] = '1';
        return text;
    }

    fn shift(self: Emitter, text: []const u8, exponent: i32) Failure![]const u8 {
        return self.times(text, try self.ten(exponent));
    }

    fn literal(self: Emitter, text: []const u8, point: bool) Failure![]const u8 {
        var out: std.ArrayList(u8) = .empty;
        if (point and text[0] == '.') try out.append(self.a, '0');
        for (text) |c| if (c != '.' or point) try out.append(self.a, c);
        var result = out.items;
        while (result.len > 1 and result[0] == '0' and result[1] != '.') result = result[1..];
        return result;
    }

    fn canonical(self: Emitter, index: usize) Failure![]const u8 {
        const node = self.parser.nodes[index];
        if (node.kind == .literal) return self.literal(node.text, true);
        const left = try self.canonical(node.left);
        if (node.kind == .neg) return std.fmt.allocPrint(self.a, "(-{s})", .{left});
        if (node.kind == .percent) return self.binary(left, "/", "100");
        const op: []const u8 = switch (node.kind) {
            .add => "+",
            .sub => "-",
            .mul => "*",
            .div => "/",
            .floor => "//",
            .pow => "**",
            else => return error.Unsupported,
        };
        return self.binary(left, op, try self.canonical(node.right));
    }

    fn scaledInteger(self: Emitter, rep: Rep, scale: usize) Failure![]const u8 {
        const delta = @as(i32, @intCast(scale)) - rep.scale;
        const n = if (delta > 0) try self.shift(rep.n, delta) else rep.n;
        const d: ?[]const u8 = if (delta < 0) try self.shift(rep.d orelse "1", -delta) else rep.d;
        return if (d) |denominator| self.binary(n, "//", denominator) else n;
    }

    fn represent(self: Emitter, index: usize) Failure!Rep {
        const node = self.parser.nodes[index];
        if (node.kind == .literal) {
            const dot = std.mem.indexOfScalar(u8, node.text, '.');
            return .{
                .n = try self.literal(node.text, false),
                .scale = if (dot) |pos| @intCast(node.text.len - pos - 1) else 0,
            };
        }
        var left = try self.represent(node.left);
        if (node.kind == .neg) {
            left.n = try std.fmt.allocPrint(self.a, "(-{s})", .{left.n});
            return left;
        }
        if (node.kind == .percent) {
            left.scale += 2;
            if (@abs(left.scale) > 124) return error.Unsupported;
            return left;
        }
        const right = try self.represent(node.right);
        var result: Rep = undefined;
        switch (node.kind) {
            .add, .sub => {
                const scale = @max(left.scale, right.scale);
                const a = try self.shift(try self.times(left.n, right.d orelse "1"), scale - left.scale);
                const b = try self.shift(try self.times(right.n, left.d orelse "1"), scale - right.scale);
                result = .{
                    .n = try self.binary(a, if (node.kind == .add) "+" else "-", b),
                    .d = try self.denominators(left.d, right.d),
                    .scale = scale,
                };
            },
            .mul => result = .{
                .n = try self.binary(left.n, "*", right.n),
                .d = try self.denominators(left.d, right.d),
                .scale = left.scale + right.scale,
            },
            .div => result = .{
                .n = try self.times(left.n, right.d orelse "1"),
                .d = try self.times(left.d orelse "1", right.n),
                .scale = left.scale - right.scale,
            },
            .floor => result = .{ .n = try self.binary(
                try self.scaledInteger(left, 0),
                "//",
                try self.scaledInteger(right, 0),
            ) },
            .pow => {
                if (right.d != null or right.scale != 0) return error.Unsupported;
                const exponent: i32 = @intCast(self.parser.nodes[node.right].value.n);
                result = .{
                    .n = try self.binary(left.n, "**", right.n),
                    .d = if (left.d) |d| try self.binary(d, "**", right.n) else null,
                    .scale = left.scale * exponent,
                };
            },
            else => return error.Unsupported,
        }
        if (@abs(result.scale) > 124) return error.Unsupported;
        return result;
    }
};

test "inline normalization and exact decimal arithmetic retain source operands" {
    const a = std.testing.allocator;
    const bytes = (try lower(a, "1.25 × 2.4", true)).?;
    defer a.free(bytes);
    try std.testing.expect(std.mem.indexOf(u8, bytes, "125") != null);
    try std.testing.expect(std.mem.indexOf(u8, bytes, "24") != null);
    const fraction = (try lower(a, "11 ÷ 18", true)).?;
    defer a.free(fraction);
    try std.testing.expectEqualStrings("(11/18)", fraction);
    const normalized = (try normalize(a, "−.5 × 2")).?;
    defer a.free(normalized);
    try std.testing.expectEqualStrings("-.5 * 2", normalized);
}

test "finite decimal division emits arithmetic rather than the host answer" {
    const a = std.testing.allocator;
    const bytes = (try lower(a, "1.0/8", true)).?;
    defer a.free(bytes);
    try std.testing.expect(std.mem.indexOf(u8, bytes, "//") != null);
    try std.testing.expect(std.mem.indexOf(u8, bytes, "0.001") != null);
    try std.testing.expect(std.mem.indexOf(u8, bytes, "0.125") == null);
}

test "malformed prose repeating decimals and exact syntax with exact off abstain" {
    const bad = [_][]const u8{
        "+2",    "2.",  "2+",    "(2",      "2,3", "x+1",  "print(2)", ">>> 2", "2=", "2#x", "2\n",
        "1.0/3", "1/0", "2**-1", "2**1000", "2%%", "2(3)", "50%3",
    };
    for (bad) |code| try std.testing.expectEqual(@as(?[]u8, null), try lower(std.testing.allocator, code, true));
    for ([_][]const u8{ "1.0", "50%", "4/2", "0.0+1" }) |code|
        try std.testing.expectEqual(@as(?[]u8, null), try lower(std.testing.allocator, code, false));
    const integer = (try lower(std.testing.allocator, "-7//3", false)).?;
    defer std.testing.allocator.free(integer);
    try std.testing.expectEqualStrings("((-7)//3)", integer);
}
