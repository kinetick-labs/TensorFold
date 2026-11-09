//! Bounded straight-line Python and single-level loops, proved before emitting runtime C operations.
const std = @import("std");
const E = @import("runner_frontend_statements_expr.zig");
const Allocator = std.mem.Allocator;
const Failure = Allocator.Error || E.Invalid;
const Kind = enum { assign, list, loop, while_loop, print };
const Condition = enum { truth, eq, ne, lt, le, gt, ge };
const Statement = struct {
    kind: Kind = .assign,
    first: usize,
    end: usize = 0,
    roots: [32]usize = undefined,
    targets: [32]usize = undefined,
    count: usize = 0,
    body_end: usize = 0,
    condition: Condition = .truth,
};

const Program = struct {
    g: E.Graph = .{},
    statements: [128]Statement = undefined,
    len: usize = 0,

    fn parse(p: *Program, code: []const u8) E.Invalid!void {
        if (code.len == 0 or code.len > 4096 or std.mem.indexOfScalar(u8, code, '\r') != null)
            return error.Unsupported;
        var lines = std.mem.splitScalar(u8, std.mem.trim(u8, code, "\n"), '\n');
        var loop: ?usize = null;
        var printed = false;
        while (lines.next()) |raw| {
            const comment = std.mem.indexOfScalar(u8, raw, '#') orelse raw.len;
            const line = std.mem.trimEnd(u8, raw[0..comment], " \t");
            if (line.len == 0) continue;
            if (p.len == p.statements.len or printed or std.mem.indexOfAny(u8, line, "\t\\") != null)
                return error.Unsupported;
            const indented = line[0] == ' ';
            if (indented) {
                if (loop == null or !std.mem.startsWith(u8, line, "    ") or line.len == 4 or line[4] == ' ')
                    return error.Unsupported;
            } else if (loop) |id| {
                if (id + 1 == p.len) return error.Unsupported;
                p.statements[id].body_end = p.len;
                loop = null;
            }
            var parser: E.Parser = .{ .src = if (indented) line[4..] else line, .g = &p.g };
            var stmt: Statement = .{ .first = p.g.len };
            if (parser.eat("for")) {
                if (indented) return error.Unsupported;
                stmt.kind = .loop;
                const name = try parser.identifier();
                try parser.need("in");
                try parser.need("range");
                try parser.need("(");
                stmt.roots[0] = try p.g.node(.{ .op = .literal });
                stmt.roots[1] = try parser.expression();
                stmt.roots[2] = try p.g.node(.{ .op = .literal, .value = 1 });
                if (parser.eat(",")) {
                    stmt.roots[0] = stmt.roots[1];
                    stmt.roots[1] = try parser.expression();
                    if (parser.eat(",")) stmt.roots[2] = try parser.expression();
                }
                try parser.need(")");
                try parser.need(":");
                stmt.targets[0] = try p.g.bind(name, null);
                loop = p.len;
            } else if (parser.eat("while")) {
                if (indented) return error.Unsupported;
                stmt.kind = .while_loop;
                stmt.roots[0] = try parser.expression();
                stmt.condition = if (parser.eat("==")) .eq else if (parser.eat("!=")) .ne else if (parser.eat("<="))
                    .le
                else if (parser.eat(">="))
                    .ge
                else if (parser.eat("<"))
                    .lt
                else if (parser.eat(">"))
                    .gt
                else
                    .truth;
                if (stmt.condition != .truth) stmt.roots[1] = try parser.expression();
                try parser.need(":");
                loop = p.len;
            } else if (parser.eat("print")) {
                if (indented) return error.Unsupported;
                stmt.kind = .print;
                try parser.need("(");
                stmt.roots[0] = try parser.expression();
                try parser.need(")");
                printed = true;
            } else {
                try p.assignment(&parser, &stmt);
            }
            try parser.end();
            stmt.end = p.g.len;
            p.statements[p.len] = stmt;
            p.len += 1;
        }
        if (!printed or loop != null) return error.Unsupported;
    }

    fn assignment(p: *Program, parser: *E.Parser, stmt: *Statement) E.Invalid!void {
        var names: [32][]const u8 = undefined;
        const paren = parser.eat("(");
        var count: usize = 1;
        names[0] = try parser.identifier();
        while (parser.eat(",")) {
            if (count == names.len) return error.Unsupported;
            names[count] = try parser.identifier();
            count += 1;
        }
        if (paren) try parser.need(")");
        const aug: ?E.Op = if (parser.eat("+=")) .add else if (parser.eat("-=")) .sub else if (parser.eat("**="))
            .power
        else if (parser.eat("*="))
            .mul
        else if (parser.eat("//="))
            .floor
        else if (parser.eat("%="))
            .modulo
        else
            null;
        if (aug == null) try parser.need("=");
        if (aug != null and (count != 1 or paren)) return error.Unsupported;
        if (count == 1 and aug == null and parser.eat("[")) {
            stmt.kind = .list;
            stmt.count = 0;
            if (!parser.eat("]")) while (true) {
                if (stmt.count == stmt.roots.len) return error.Unsupported;
                stmt.roots[stmt.count] = try parser.expression();
                stmt.count += 1;
                if (parser.eat("]")) break;
                try parser.need(",");
                if (parser.eat("]")) break;
            };
            stmt.targets[0] = try p.g.bind(names[0], stmt.count);
            return;
        }
        stmt.count = count;
        const rhs_paren = count > 1 and parser.eat("(");
        for (0..count) |i| {
            if (i != 0) try parser.need(",");
            var previous: usize = undefined;
            if (aug != null) {
                const id = p.g.find(names[0]) orelse return error.Unsupported;
                if (p.g.symbols[id].list_len != null) return error.Unsupported;
                previous = try p.g.node(.{ .op = .variable, .lhs = id });
            }
            stmt.roots[i] = try parser.expression();
            if (aug) |op| stmt.roots[i] = try p.g.node(.{ .op = op, .lhs = previous, .rhs = stmt.roots[i] });
        }
        if (rhs_paren) try parser.need(")");
        // Bind after parsing every RHS so newly introduced targets cannot define their own inputs.
        for (names[0..count], 0..) |name, i| stmt.targets[i] = try p.g.bind(name, null);
    }

    fn prove(p: *const Program, s: *E.State, begin: usize, end: usize, iterations: *usize) E.Invalid!void {
        var i = begin;
        while (i < end) : (i += 1) {
            const stmt = p.statements[i];
            switch (stmt.kind) {
                .assign, .list => {
                    var values: [32]i64 = undefined;
                    for (0..stmt.count) |j| values[j] = try p.g.eval(stmt.roots[j], s);
                    if (stmt.kind == .list) {
                        s.values[stmt.targets[0]] = @splat(0);
                        for (0..stmt.count) |j| s.values[stmt.targets[0]][j] = values[j];
                        s.defined[stmt.targets[0]] = true;
                    } else for (0..stmt.count) |j| {
                        s.values[stmt.targets[j]][0] = values[j];
                        s.defined[stmt.targets[j]] = true;
                    }
                },
                .print => _ = try p.g.eval(stmt.roots[0], s),
                .loop => {
                    const start = try p.g.eval(stmt.roots[0], s);
                    const stop = try p.g.eval(stmt.roots[1], s);
                    const step = try p.g.eval(stmt.roots[2], s);
                    if (step == 0) return error.Unsupported;
                    const distance = if (step > 0) stop - start else start - stop;
                    const stride: i64 = @intCast(@abs(step));
                    const count: usize = if (distance <= 0) 0 else @intCast(@divTrunc(distance + stride - 1, stride));
                    if (count > 10000 or iterations.* + count > 10000) return error.Unsupported;
                    iterations.* += count;
                    // The emitted induction update also runs after the last body execution.
                    if (!E.fits(start + @as(i64, @intCast(count)) * step)) return error.Unsupported;
                    var value = start;
                    for (0..count) |_| {
                        s.values[stmt.targets[0]][0] = value;
                        s.defined[stmt.targets[0]] = true;
                        try p.prove(s, i + 1, stmt.body_end, iterations);
                        value += step;
                    }
                    i = stmt.body_end - 1;
                },
                .while_loop => {
                    while (try p.condition(stmt, s)) {
                        if (iterations.* == 10000) return error.Unsupported;
                        iterations.* += 1;
                        try p.prove(s, i + 1, stmt.body_end, iterations);
                    }
                    i = stmt.body_end - 1;
                },
            }
        }
    }

    fn condition(p: *const Program, stmt: Statement, s: *E.State) E.Invalid!bool {
        const lhs = try p.g.eval(stmt.roots[0], s);
        if (stmt.condition == .truth) return lhs != 0;
        const rhs = try p.g.eval(stmt.roots[1], s);
        return switch (stmt.condition) {
            .eq => lhs == rhs,
            .ne => lhs != rhs,
            .lt => lhs < rhs,
            .le => lhs <= rhs,
            .gt => lhs > rhs,
            .ge => lhs >= rhs,
            .truth => unreachable,
        };
    }

    fn emit(p: *const Program, a: Allocator, out: *std.ArrayList(u8), begin: usize, end: usize) Allocator.Error!void {
        var i = begin;
        while (i < end) : (i += 1) {
            const stmt = p.statements[i];
            try out.appendSlice(a, "{\n");
            if (stmt.kind == .while_loop) {
                try out.appendSlice(a, "while (1) {\n");
                try p.g.emit(a, out, stmt.first, stmt.end);
                if (stmt.condition == .truth) {
                    try out.print(a, "if (!tf_e{d}) break;\n", .{stmt.roots[0]});
                } else {
                    const op = switch (stmt.condition) {
                        .eq => "==",
                        .ne => "!=",
                        .lt => "<",
                        .le => "<=",
                        .gt => ">",
                        .ge => ">=",
                        .truth => unreachable,
                    };
                    try out.print(a, "if (!(tf_e{d} {s} tf_e{d})) break;\n", .{ stmt.roots[0], op, stmt.roots[1] });
                }
                try p.emit(a, out, i + 1, stmt.body_end);
                try out.appendSlice(a, "}\n}\n");
                i = stmt.body_end - 1;
                continue;
            }
            try p.g.emit(a, out, stmt.first, stmt.end);
            switch (stmt.kind) {
                .assign => for (0..stmt.count) |j| {
                    try out.print(a, "tf_v{d} = tf_e{d};\n", .{ stmt.targets[j], stmt.roots[j] });
                },
                .list => for (0..stmt.count) |j| {
                    try out.print(a, "tf_v{d}[{d}] = tf_e{d};\n", .{ stmt.targets[0], j, stmt.roots[j] });
                },
                .print => try out.print(a, "tf_print_int(tf_e{d});\n", .{stmt.roots[0]}),
                .loop => {
                    try out.print(a, "for (int tf_r{d} = tf_e{d}; (tf_e{d} > 0 ? tf_r{d} < tf_e{d} : tf_r{d} > tf_e{d}); tf_r{d} += tf_e{d}) {{\ntf_v{d} = tf_r{d};\n", .{
                        i, stmt.roots[0], stmt.roots[2],   i, stmt.roots[1], i, stmt.roots[1],
                        i, stmt.roots[2], stmt.targets[0], i,
                    });
                    try p.emit(a, out, i + 1, stmt.body_end);
                    try out.appendSlice(a, "}\n");
                    i = stmt.body_end - 1;
                },
                .while_loop => unreachable,
            }
            try out.appendSlice(a, "}\n");
        }
    }
};

/// Returns a caller-owned C body, or null outside the proved subset; no compiler or host execution.
pub const Lowered = struct { source: []u8, estimated_steps: usize };
pub fn lowerPython(a: Allocator, code: []const u8) Allocator.Error!?[]u8 {
    const result = try lowerWithCost(a, code) orelse return null;
    return result.source;
}
pub fn lowerWithCost(a: Allocator, code: []const u8) Allocator.Error!?Lowered {
    return lower(a, code) catch |err| switch (err) {
        error.Unsupported => null,
        error.OutOfMemory => error.OutOfMemory,
    };
}

fn lower(a: Allocator, code: []const u8) Failure!Lowered {
    const p = try a.create(Program);
    defer a.destroy(p);
    p.* = .{};
    try p.parse(code);
    var state: E.State = .{};
    var iterations: usize = 0;
    try p.prove(&state, 0, p.len, &iterations);
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    for (p.g.symbols[0..p.g.symbols_len], 0..) |symbol, id| {
        if (symbol.list_len != null) {
            try out.print(a, "int tf_v{d}[32];\n", .{id});
        } else try out.print(a, "int tf_v{d};\n", .{id});
    }
    try p.emit(a, &out, 0, p.len);
    try out.appendSlice(a, "return 0;\n");
    return .{ .source = try out.toOwnedSlice(a), .estimated_steps = state.estimated_steps + iterations * 12 + 300 };
}

test "bounded factorial retains multiplication and runtime loop" {
    const a = std.testing.allocator;
    const body = (try lowerPython(a, "n = 12\nf = 1\nfor i in range(1, n+1):\n    f *= i\nprint(f)")).?;
    defer a.free(body);
    try std.testing.expect(std.mem.indexOf(u8, body, "for (int tf_r") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, " * tf_e") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "479001600") == null);
}

test "fibonacci tuple assignment proves simultaneous RHS" {
    const a = std.testing.allocator;
    const source = "a, b = 0, 1\nfor i in range(20):\n    a, b = b, a+b\nprint(a)";
    const p = try a.create(Program);
    defer a.destroy(p);
    p.* = .{};
    try p.parse(source);
    var state: E.State = .{};
    var iterations: usize = 0;
    try p.prove(&state, 0, p.len, &iterations);
    try std.testing.expectEqual(@as(i64, 6765), state.values[p.g.find("a").?][0]);
    try std.testing.expect(state.values[p.g.find("a").?][0] != 524288);
    const body = (try lowerPython(a, source)).?;
    defer a.free(body);
    try std.testing.expect(std.mem.indexOf(u8, body, "6765") == null);
}

test "sum lists and signed floor operations retain integer operations" {
    const a = std.testing.allocator;
    const good = [_][]const u8{
        "x = -7\ny = 3\nprint(x//y)",
        "x = 7\ny = -3\nprint(x%y)",
        "xs = [2, -7, 3*4]\nprint(sum(xs))",
        "print(sum([2, -7, 3*4]))",
        "xs = []\nprint(sum(xs))",
        "a = 2\na **= 3\nprint(-a**2)",
        "a = 0\nfor i in range(9, -1, -1):\n    a += i\nprint(a)",
        "a = 3\nfor i in range(0):\n    a += 1\nprint(a)",
    };
    for (good) |source| {
        const body = (try lowerPython(a, source)).?;
        defer a.free(body);
        try std.testing.expect(std.mem.indexOf(u8, body, "tf_print_int(tf_e") != null);
    }
}

test "unproved arithmetic, names and arbitrary syntax abstain" {
    const bad = [_][]const u8{
        "x = 1\nprint(y)",                                               "x = x + 1\nprint(x)",                     "print = 1\nprint(2)",                                 "sum = 2\nprint(sum([1]))",
        "range = 1\nprint(range)",                                       "import os\nprint(1)",                     "x = 1.0\nprint(x)",                                   "print(1/2)",
        "x = 1\nif x:\n    x = 2\nprint(x)",                             "print(1)\nprint(2)",                      "x = [1]\nprint(x)",                                   "print(1//0)",
        "print(1%0)",                                                    "print(2147483647+1)",                     "print(50000*50000-50000*50000)",                      "print(2**31)",
        "print(2**-1)",                                                  "print((-2147483647-1)//-1)",              "f = 1\nfor i in range(1, 14):\n    f *= i\nprint(f)", "x = 0\nfor i in range(10001):\n    x += i\nprint(x)",
        "x = 0\nfor i in range(0, 10, 0):\n    x += i\nprint(x)",        "for i in range(0):\n    x = 1\nprint(x)", "for i in range(0):\n    x = 1\nprint(i)",             "n = 1\nfor i in range(3):\n    n -= 1\n    x = 10//n\nprint(x)",
        "for i in range(3):\n    i = 2147483647\n    x = i+1\nprint(x)", "x = 1\\\nprint(x)",                       "x = 1\nfor i in range(3):\n    x //= i\nprint(x)",    "for i in range(2):\n    for j in range(2):\n        x = 1\nprint(x)",
    };
    for (bad) |source| try std.testing.expectEqual(@as(?[]u8, null), try lowerPython(std.testing.allocator, source));
}

test "blank lines and comments are harmless but continuation is outside the grammar" {
    const a = std.testing.allocator;
    const source = "# header\n\nx = 1 # initial\nfor i in range(3): # frozen bound\n    # body note\n\n    x += i # update\n\nprint(x) # final\n# trailing\n";
    const body = (try lowerPython(a, source)).?;
    defer a.free(body);
    try std.testing.expect(std.mem.indexOf(u8, body, "#") == null);
    try std.testing.expect(std.mem.indexOf(u8, body, "for (int tf_r") != null);
}

test "range and induction values are captured before body mutations" {
    const a = std.testing.allocator;
    const p = try a.create(Program);
    defer a.destroy(p);
    const sources = [_][]const u8{
        "n = 4\nx = 0\nfor i in range(n):\n    n = 0\n    x += 1\nprint(x)",
        "x = 0\nfor i in range(4):\n    i = 100\n    x += 1\nprint(x)",
        "n = 1\nx = 0\nfor i in range(0, 4, n):\n    n = 0\n    x += 1\nprint(x)",
    };
    for (sources) |source| {
        p.* = .{};
        try p.parse(source);
        var state: E.State = .{};
        var iterations: usize = 0;
        try p.prove(&state, 0, p.len, &iterations);
        try std.testing.expectEqual(@as(i64, 4), state.values[p.g.find("x").?][0]);
        const body = (try lowerPython(a, source)).?;
        defer a.free(body);
        try std.testing.expect(std.mem.indexOf(u8, body, "for (int tf_r") != null);
        try std.testing.expect(std.mem.indexOf(u8, body, " += tf_e") != null);
    }
}

test "while conditions and signed floor operations are reevaluated after every body" {
    const cases = [_]struct { code: []const u8, name: []const u8, value: i64 }{
        .{ .code = "n = 123456789\ns = 0\nwhile n > 0:\n    s += n % 10\n    n //= 10\nprint(s)", .name = "s", .value = 45 },
        .{ .code = "a = 462\nb = 1071\nwhile b:\n    a, b = b, a % b\nprint(a)", .name = "a", .value = 21 },
        .{ .code = "n = -7\ns = 0\nwhile n < 0:\n    s += n // 3\n    n += 1\nprint(s)", .name = "s", .value = -12 },
        .{ .code = "n = 0\nwhile n != 0:\n    n -= 1\nprint(n)", .name = "n", .value = 0 },
        .{ .code = "n = 0\nwhile n <= 4:\n    n += 1\nprint(n)", .name = "n", .value = 5 },
        .{ .code = "n = 3\nwhile n >= 1:\n    n -= 1\nprint(n)", .name = "n", .value = 0 },
        .{ .code = "n = 1\nwhile n == 1:\n    n = 2\nprint(n)", .name = "n", .value = 2 },
    };
    const a = std.testing.allocator;
    for (cases) |case| {
        const p = try a.create(Program);
        defer a.destroy(p);
        p.* = .{};
        try p.parse(case.code);
        var state: E.State = .{};
        var iterations: usize = 0;
        try p.prove(&state, 0, p.len, &iterations);
        try std.testing.expectEqual(case.value, state.values[p.g.find(case.name).?][0]);
        const body = (try lowerPython(a, case.code)).?;
        defer a.free(body);
        const loop = std.mem.indexOf(u8, body, "while (1)") orelse return error.MissingRuntimeLoop;
        const condition_load = std.mem.indexOf(u8, body[loop..], "= tf_v") orelse return error.MissingConditionLoad;
        const condition_check = std.mem.indexOf(u8, body[loop..], "break;") orelse return error.MissingConditionCheck;
        try std.testing.expect(condition_load < condition_check);
    }
}

test "unbounded unsafe and unproved while executions abstain" {
    for ([_][]const u8{
        "n = 1\nwhile n:\n    n = n\nprint(n)",
        "n = 0\nwhile n < 10001:\n    n += 1\nprint(n)",
        "n = 1\nwhile n > 0:\n    n //= 0\nprint(n)",
        "n = 2147483647\nwhile n:\n    n += 1\nprint(n)",
        "while 0:\n    x = 1\nprint(x)",
        "n = 1\nwhile n:\n    while n:\n        n -= 1\nprint(n)",
        "n = 1\nwhile missing:\n    n -= 1\nprint(n)",
    }) |code| try std.testing.expectEqual(@as(?[]u8, null), try lowerPython(std.testing.allocator, code));
}
