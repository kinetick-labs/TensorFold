//! Bounded Python loop lowering; accepted integer operations are exact in RV32 signed C arithmetic.
const std = @import("std");
const Allocator = std.mem.Allocator;
const Expr = @import("runner_frontend_program_expr.zig");
const statements = @import("runner_frontend_statements.zig");
pub const compiler = @import("runner_frontend_program_compile.zig");
pub const Config = compiler.Config;
pub const compile = compiler.compile;
pub const wrapBody = compiler.wrapBody;

const Range = struct { start: i64, stop: i64, step: i64, count: i64, bounds: Expr.Bounds };
const Invalid = error{Unsupported};

/// Returns an owned C main body, or abstains; this function neither compiles nor executes the source.
pub const Lowered = statements.Lowered;
pub fn lowerPython(a: Allocator, code: []const u8) Allocator.Error!?[]u8 {
    const result = try lowerWithCost(a, code) orelse return null;
    return result.source;
}
pub fn lowerWithCost(a: Allocator, code: []const u8) Allocator.Error!?Lowered {
    if (try statements.lowerWithCost(a, code)) |result| return result;
    return lower(a, code) catch |err| switch (err) {
        error.Unsupported => null,
        error.OutOfMemory => error.OutOfMemory,
    };
}

fn lower(a: Allocator, code: []const u8) (Allocator.Error || Invalid)!Lowered {
    if (code.len > 4096 or std.mem.indexOfScalar(u8, code, '\r') != null) return error.Unsupported;
    const src = std.mem.trim(u8, code, "\n");
    if (src.len == 0 or src[0] == ' ' or src[0] == '\t') return error.Unsupported;
    var p: Expr.Parser = .{ .src = src };
    var initial: i64 = 0;
    var range: Range = undefined;
    var root: usize = undefined;
    if (std.mem.indexOfScalar(u8, src, '\n') == null) {
        const printed = p.eat("print");
        if (printed) try p.need("(");
        try p.need("sum");
        try p.need("(");
        const list = p.eat("[");
        root = try p.expression();
        try p.need("for");
        const variable = try p.identifier();
        if (!validName(variable)) return error.Unsupported;
        try p.need("in");
        range = try parseRange(&p);
        if (list) try p.need("]");
        try p.need(")");
        if (printed) try p.need(")");
        try p.end();
        try p.bind(variable, range.bounds);
    } else {
        var lines = std.mem.splitScalar(u8, src, '\n');
        const init_line = lines.next() orelse return error.Unsupported;
        const for_line = lines.next() orelse return error.Unsupported;
        const body_line = lines.next() orelse return error.Unsupported;
        const print_line = lines.next() orelse return error.Unsupported;
        if (lines.next() != null) return error.Unsupported;
        var init: Expr.Parser = .{ .src = init_line };
        const accumulator = try init.identifier();
        if (!validName(accumulator)) return error.Unsupported;
        try init.need("=");
        initial = try init.integer();
        if (!Expr.fits(initial)) return error.Unsupported;
        try init.end();
        var header: Expr.Parser = .{ .src = for_line };
        try header.need("for");
        const variable = try header.identifier();
        if (!validName(variable) or std.mem.eql(u8, variable, accumulator)) return error.Unsupported;
        try header.need("in");
        range = try parseRange(&header);
        try header.need(":");
        try header.end();
        if (!std.mem.startsWith(u8, body_line, "    ") or
            (for_line.len > 0 and for_line[0] == ' ') or
            (print_line.len > 0 and print_line[0] == ' ')) return error.Unsupported;
        p = .{ .src = body_line[4..] };
        const target = try p.identifier();
        if (!std.mem.eql(u8, accumulator, target)) return error.Unsupported;
        if (!p.eat("+=")) {
            try p.need("=");
            const previous = try p.identifier();
            if (!std.mem.eql(u8, previous, accumulator)) return error.Unsupported;
            try p.need("+");
        }
        root = try p.expression();
        try p.end();
        try p.bind(variable, range.bounds);
        var output: Expr.Parser = .{ .src = print_line };
        try output.need("print");
        try output.need("(");
        const value = try output.identifier();
        if (!std.mem.eql(u8, value, accumulator)) return error.Unsupported;
        try output.need(")");
        try output.end();
    }
    const bounds = try p.bounds(root);
    if (!Expr.fits(initial + range.count * @min(bounds.lo, 0)) or
        !Expr.fits(initial + range.count * @max(bounds.hi, 0))) return error.Unsupported;
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    try out.print(a, "static volatile int tf_range[]={{ {d},{d},{d},{d} }};\nvolatile int tf_acc=tf_range[0];\nint tf_start=tf_range[1],tf_stop=tf_range[2],tf_step=tf_range[3];\nfor (int tf_i=tf_start; (tf_step>0 ? tf_i < tf_stop : tf_i > tf_stop); tf_i+=tf_step) {{\n", .{
        initial, range.start, range.stop, range.step,
    });
    try p.emit(a, &out);
    try out.print(a, "tf_acc += tf_e{d};\n}}\ntf_print_int(tf_acc);\nreturn 0;\n", .{root});
    var per_iteration: usize = 16;
    for (p.nodes[0..p.len]) |n| {
        per_iteration += 10;
        if (n.op == .power) per_iteration += @as(usize, @intCast(n.value)) * 6;
    }
    return .{ .source = try out.toOwnedSlice(a), .estimated_steps = 500 + @as(usize, @intCast(range.count)) * per_iteration };
}

fn validName(s: []const u8) bool {
    const reserved = [_][]const u8{
        "print", "sum",    "range",  "for",      "in",     "True",      "False",        "None",  "and",   "or",       "not",
        "if",    "else",   "def",    "class",    "import", "from",      "return",       "while", "break", "continue", "pass",
        "yield", "lambda", "global", "nonlocal", "try",    "except",    "finally",      "with",  "as",    "assert",   "del",
        "raise", "is",     "async",  "await",    "elif",   "__debug__", "__builtins__",
    };
    for (reserved) |name| if (std.mem.eql(u8, name, s)) return false;
    return true;
}

fn parseRange(p: *Expr.Parser) Invalid!Range {
    try p.need("range");
    try p.need("(");
    const first = try p.integer();
    var start: i64 = 0;
    var stop = first;
    var step: i64 = 1;
    if (p.eat(",")) {
        start = first;
        stop = try p.integer();
        if (p.eat(",")) step = try p.integer();
    }
    try p.need(")");
    if (@abs(start) > 1000000 or @abs(stop) > 1000000 or @abs(step) > 1000000 or step == 0)
        return error.Unsupported;
    const distance = if (step > 0) stop - start else start - stop;
    const stride: i64 = @intCast(@abs(step));
    const count = if (distance <= 0) 0 else @divTrunc(distance + stride - 1, stride);
    if (count > 10000) return error.Unsupported;
    const last = if (count == 0) start else start + (count - 1) * step;
    return .{
        .start = start,
        .stop = stop,
        .step = step,
        .count = count,
        .bounds = .{ .lo = @min(start, last), .hi = @max(start, last) },
    };
}

test "sum squares 285 is lowered as circuit loop operations" {
    const a = std.testing.allocator;
    const body = (try lower(a, "print(sum(i*i for i in range(10)))")).source;
    defer a.free(body);
    try std.testing.expect(std.mem.indexOf(u8, body, "tf_i < tf_stop") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "tf_e0 * tf_e1") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "tf_acc += tf_e2") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "285") == null);
}

test "accumulator and descending stepped ranges" {
    const a = std.testing.allocator;
    const body = (try lower(a, "total = 0\nfor i in range(10):\n    total += i**2\nprint(total)")).source;
    defer a.free(body);
    try std.testing.expect(std.mem.indexOf(u8, body, "tf_i < tf_stop") != null);
    const assigned = (try lower(a, "total = 0\nfor i in range(10):\n    total = total + i*i\nprint(total)")).source;
    defer a.free(assigned);
    try std.testing.expect(std.mem.indexOf(u8, assigned, "tf_acc += tf_e2") != null);
    const descending = (try lowerPython(a, "sum(x*x for x in range(9, -1, -1))")).?;
    defer a.free(descending);
    try std.testing.expect(std.mem.indexOf(u8, descending, "tf_i > tf_stop") != null);
}

test "unsupported code and unsafe Python integer operations abstain" {
    const bad = [_][]const u8{
        "print(sum(j*j for i in range(10)))",
        "print(sum(i/i for i in range(10)))",
        "print(sum(i**32 for i in range(10)))",
        "print(sum(i*i for i in range(100000)))",
        "print(sum(i*i for i in range(10000)))",
        "print(sum(i for i in range(10, 0, 0)))",
        "print(sum(i for i in range(n)))",
        "print(sum(i for i in range(10) if i))",
        "print(sum([i for i in range(10) if i]))",
        "print(sum([i for i in range(10)))",
        "print(sum(i for i in range(10))); print(2)",
        "import os\nprint(os.getpid())",
        "t = 0\nfor i in range(10):\n    t += unknown\nprint(t)",
        "t = 2147483647\nfor i in range(1):\n    t += 1\nprint(t)",
        "t = 0\nfor i in range(10):\n    t += i\n    print(t)",
    };
    for (bad) |src| try std.testing.expectEqual(@as(?[]u8, null), try lowerPython(std.testing.allocator, src));
}
