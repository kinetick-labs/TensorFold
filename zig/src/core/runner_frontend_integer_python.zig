const std = @import("std");
const ast = @import("runner_frontend_integer_ast.zig");
const integer = @import("runner_frontend_integer.zig");
const compiler = @import("runner_frontend_program_compile.zig");
const cost = @import("runner_frontend_integer_cost.zig");
const list = @import("runner_frontend_integer_list.zig");
const Allocator = std.mem.Allocator;
const Failure = Allocator.Error || ast.Invalid;

// Immutable bindings retain expression nodes; proof values never replace the requested runtime arithmetic.
pub fn lower(a: Allocator, code: []const u8) Allocator.Error!?[]u8 {
    const result = try lowerWithCost(a, code, 0) orelse return null;
    if (result.graph_only) {
        a.free(result.source);
        return null;
    }
    return result.source;
}

pub fn lowerWithCost(a: Allocator, code: []const u8, budget: usize) Allocator.Error!?integer.Lowered {
    return checked(a, code, budget) catch |err| switch (err) {
        error.Unsupported => null,
        error.OutOfMemory => error.OutOfMemory,
    };
}

fn checked(a: Allocator, code: []const u8, budget: usize) Failure!integer.Lowered {
    var script = try parseOutputs(code, true);
    const parser = &script.parser;
    var steps: ?usize = null;
    const roots = script.roots[0..script.count];
    const endings = script.endings[0..script.count];
    var t = integer.table(parser, roots, endings, null, null);
    @memcpy(t.brackets[0..script.count], script.brackets[0..script.count]);
    // lists, defs, asserts, loops, over 64 nodes or 8 outputs: only the graph route's table (calc, ELF keep theirs)
    if (parser.listed or parser.function_count != 0 or parser.asserted or parser.looped or parser.count > 64 or script.count > 8) {
        if (budget != 0) return error.Unsupported;
        return .{ .source = try a.alloc(u8, 0), .estimated_steps = null, .table = t, .graph_only = true };
    }
    if (budget != 0) steps = cost.estimateMany(parser, roots, budget) orelse return error.Unsupported;
    return .{ .source = try integer.lowerPrints(a, parser, roots, endings), .estimated_steps = steps, .table = t };
}

// Shares whole-script validation with the RV32 path, including unused assignments and import gates.
pub fn lowerCalc(a: Allocator, code: []const u8) Allocator.Error!?[]u8 {
    var script = parseScript(code) catch return null;
    const p = &script.parser;
    if (p.listed or p.function_count != 0 or p.asserted or p.looped or p.count > 64) return null;
    return @import("runner_frontend_integer_calc.zig").lower(a, &script.parser, script.root);
}

pub const Script = struct { parser: ast.Parser, root: usize };

pub fn parseScript(code: []const u8) ast.Invalid!Script {
    const script = try parseOutputs(code, false);
    return .{ .parser = script.parser, .root = script.roots[0] };
}

const Outputs = struct {
    parser: ast.Parser,
    roots: [32]usize = undefined,
    endings: [32]u8 = undefined, // ' ' or '\n' as print ends a value; ',' between a list's elements
    brackets: [32]u8 = @splat(0), // 1: a list opens before the value, 2: it closes after
    count: usize = 0,
};
const statements = @import("runner_frontend_integer_statements.zig");
const suite = @import("runner_frontend_integer_suite.zig");

fn parseOutputs(code: []const u8, multiple: bool) ast.Invalid!Outputs {
    if (code.len == 0 or code.len > 4096 or std.mem.indexOfScalar(u8, code, '\r') != null)
        return error.Unsupported;
    var parser: ast.Parser = .{ .source = "", .require_math_import = true, .allow_lists = true, .capacity = ast.max_nodes };
    var lines = statements.Iterator{ .code = code };
    var outputs = Outputs{ .parser = undefined };
    var expression_only = false;
    var started = false;
    while (try lines.next()) |raw| {
        var line = std.mem.trimEnd(u8, raw.text, " \t");
        if (raw.after_semicolon) line = std.mem.trimStart(u8, line, " \t");
        if (line.len == 0) continue;
        if (((!multiple or expression_only) and outputs.count != 0) or line[0] == ' ' or line[0] == '\t')
            return error.Unsupported;
        if (std.mem.startsWith(u8, line, ">>> ")) line = line[4..];
        if (std.mem.eql(u8, line, "import math")) {
            if (started) return error.Unsupported;
            parser.math_module = true;
            continue;
        }
        if (std.mem.startsWith(u8, line, "from math import ")) {
            if (started) return error.Unsupported;
            var names = std.mem.splitScalar(u8, line[17..], ',');
            while (names.next()) |part| {
                const name = std.mem.trim(u8, part, " ");
                parser.math_imports |= if (std.mem.eql(u8, name, "gcd")) @as(u8, 1) else if (std.mem.eql(u8, name, "isqrt"))
                    @as(u8, 2)
                else if (std.mem.eql(u8, name, "factorial"))
                    @as(u8, 4)
                else if (std.mem.eql(u8, name, "prod"))
                    @as(u8, 8)
                else
                    return error.Unsupported;
            }
            continue;
        }
        started = true;
        const printed = std.mem.startsWith(u8, line, "print") and line.len > 5 and
            (line[5] == '(' or line[5] == ' ' or line[5] == '\t');
        if (keyword(line, "def")) {
            try define(&parser, &lines, line);
        } else if (keyword(line, "for") or keyword(line, "while")) {
            if (!multiple) return error.Unsupported;
            const Print = struct {
                fn print(ctx: *anyopaque, p: *ast.Parser, text: []const u8) ast.Invalid!void {
                    return printRoots(p, @ptrCast(@alignCast(ctx)), text, true);
                }
            };
            try suite.module(&parser, try block(&lines, line), .{ .ctx = &outputs, .print = Print.print });
        } else if (keyword(line, "assert")) {
            try check(&parser, line["assert".len..]);
        } else if (printed) {
            line = std.mem.trimStart(u8, line[5..], " \t");
            if (line.len < 3 or line[0] != '(' or line[line.len - 1] != ')') return error.Unsupported;
            try printRoots(&parser, &outputs, line[1 .. line.len - 1], multiple);
        } else if (statements.separator(line, '=')) |equal| {
            try assignment(&parser, std.mem.trim(u8, line[0..equal], " \t"), line[equal + 1 ..]);
        } else {
            if (outputs.count != 0 or parser.binding_count != 0 or parser.function_count != 0 or parser.math_module or
                parser.math_imports != 0) return error.Unsupported;
            try printRoots(&parser, &outputs, line, false);
            expression_only = true;
        }
    }
    if (outputs.count == 0) return error.Unsupported;
    outputs.parser = parser;
    return outputs;
}

/// A compound statement from its header line through its indented suite, as one slice of the script.
fn block(lines: *statements.Iterator, header: []const u8) ast.Invalid![]const u8 {
    const start = @intFromPtr(header.ptr) - @intFromPtr(lines.code.ptr);
    var end = start + header.len;
    var indented = false;
    while (true) {
        var ahead = lines.*;
        const raw = try ahead.next() orelse break;
        const text = std.mem.trimEnd(u8, raw.text, " \t");
        const stmt = std.mem.trim(u8, text, " \t");
        // a semicolon's next statement shares its line: inside the suite after an indented one, else the header's body
        if (stmt.len != 0 and !raw.after_semicolon and text[0] != ' ' and text[0] != '\t') break;
        if (raw.after_semicolon and !indented and stmt.len != 0 and header[header.len - 1] == ':') break;
        lines.* = ahead;
        if (stmt.len == 0) continue;
        if (!raw.after_semicolon) indented = true;
        end = @intFromPtr(raw.text.ptr) - @intFromPtr(lines.code.ptr) + text.len;
    }
    return lines.code[start..end];
}

fn expression(parser: *ast.Parser, text: []const u8) ast.Invalid!usize {
    parser.source = std.mem.trim(u8, text, " \t");
    parser.pos = 0;
    return parser.parse();
}

/// A whole right side or print argument: a static int list when one parses, else an int expression.
fn side(parser: *ast.Parser, text: []const u8) ast.Invalid!ast.Parser.Operand {
    parser.source = std.mem.trim(u8, text, " \t");
    parser.pos = 0;
    if (list.ahead(parser)) {
        const m = parser.mark();
        if (list.whole(parser, parser.source)) |li| return .{ .index = li, .list = true } else |_| parser.reset(m);
    }
    return .{ .index = try parser.parse(), .list = false };
}

fn assignment(parser: *ast.Parser, names: []const u8, rhs: []const u8) ast.Invalid!void {
    if (statements.separator(names, ',')) |comma| {
        const first = std.mem.trim(u8, names[0..comma], " \t");
        const second = std.mem.trim(u8, names[comma + 1 ..], " \t");
        if (!validBinding(first) or !validBinding(second)) return error.Unsupported;
        const pair = try parser.divisionPair(try expression(parser, rhs));
        try parser.bind(first, pair[0]);
        try parser.bind(second, pair[1]);
    } else {
        if (!validBinding(names)) return error.Unsupported;
        const v = try side(parser, rhs);
        if (!v.list and parser.nodes[v.index].op == .divmod) return error.Unsupported;
        try parser.bindAs(names, v);
    }
}

fn keyword(line: []const u8, word: []const u8) bool {
    return std.mem.startsWith(u8, line, word) and line.len > word.len and
        (line[word.len] == ' ' or line[word.len] == '\t' or line[word.len] == '(');
}

/// `def name(a, b):` (`int` annotations): one `return <int expr>` (a docstring first), or a suite of int statements.
fn define(parser: *ast.Parser, lines: *statements.Iterator, header: []const u8) ast.Invalid!void {
    const open = std.mem.indexOfScalar(u8, header, '(') orelse return error.Unsupported;
    const close = std.mem.indexOfScalarPos(u8, header, open, ')') orelse return error.Unsupported;
    var f: ast.Parser.Function = .{ .name = std.mem.trim(u8, header["def".len..open], " \t"), .body = "" };
    if (!validBinding(f.name) and !unimported(parser, f.name)) return error.Unsupported;
    const params = std.mem.trim(u8, header[open + 1 .. close], " \t");
    if (params.len != 0) {
        var names = std.mem.splitScalar(u8, params, ',');
        while (names.next()) |raw| {
            var p = std.mem.trim(u8, raw, " \t");
            if (std.mem.indexOfScalar(u8, p, ':')) |colon| {
                if (!std.mem.eql(u8, std.mem.trim(u8, p[colon + 1 ..], " \t"), "int")) return error.Unsupported;
                p = std.mem.trim(u8, p[0..colon], " \t");
            }
            if (!validBinding(p) or f.arity == f.params.len) return error.Unsupported;
            f.params[f.arity] = p;
            f.arity += 1;
        }
    }
    var rest = std.mem.trim(u8, header[close + 1 ..], " \t");
    if (std.mem.startsWith(u8, rest, "->")) {
        rest = std.mem.trimStart(u8, rest[2..], " \t");
        if (!std.mem.startsWith(u8, rest, "int")) return error.Unsupported;
        rest = std.mem.trimStart(u8, rest[3..], " \t"); // `-> intx:` leaves `x:`, refused below
    }
    if (rest.len == 0 or rest[0] != ':') return error.Unsupported;
    rest = std.mem.trim(u8, rest[1..], " \t");
    if (rest.len == 0) {
        // an indented suite: one `return <expr>` (a docstring first) stays an expression body, else statements run
        const code = try block(lines, header);
        const text = code[header.len..];
        if (single(text)) |value| f.body = value else f.suite = text;
        return parser.define(f);
    }
    f.body = try returned(rest);
    // nothing else may follow on the def's own line or under it
    while (true) {
        var ahead = lines.*;
        const raw = try ahead.next() orelse break;
        if (raw.after_semicolon) return error.Unsupported; // `def f(): return x; y = 1`: a second body statement
        const text = std.mem.trimEnd(u8, raw.text, " \t");
        const stmt = std.mem.trim(u8, text, " \t");
        if (stmt.len != 0 and text[0] != ' ' and text[0] != '\t') break;
        lines.* = ahead;
        if (stmt.len != 0) return error.Unsupported;
    }
    try parser.define(f);
}

/// A suite that is one `return <expr>` without a conditional expression (a docstring may come first), else null.
fn single(text: []const u8) ?[]const u8 {
    var it = statements.Iterator{ .code = text };
    var value: ?[]const u8 = null;
    var docstring = true;
    while (it.next() catch return null) |raw| {
        const stmt = std.mem.trim(u8, raw.text, " \t");
        if (stmt.len == 0) continue;
        if (value != null) return null;
        if (docstring and stmt.len >= 2 and (stmt[0] == '"' or stmt[0] == '\'') and stmt[stmt.len - 1] == stmt[0]) {
            docstring = false;
            continue;
        }
        docstring = false;
        const v = returned(stmt) catch return null;
        if (list.top(v, "if", 0) != null) return null;
        value = v;
    }
    return value;
}

/// The expression of `return <expr>`: one value (a tuple, or a bare `return`, is not an int).
fn returned(stmt: []const u8) ast.Invalid![]const u8 {
    if (!keyword(stmt, "return")) return error.Unsupported;
    const value = std.mem.trim(u8, stmt["return".len..], " \t");
    if (value.len == 0 or statements.separator(value, ',') != null) return error.Unsupported;
    return value;
}

/// `assert <int cmp>`, proved on the outputs' exact values: true prints nothing; false refuses (the sandbox raises).
fn check(parser: *ast.Parser, text: []const u8) ast.Invalid!void {
    if (statements.separator(text, ',') != null) return error.Unsupported; // a message: refuse rather than evaluate it
    const m = parser.mark();
    const root = try expression(parser, text);
    if (!parser.boolean(root) or parser.nodes[root].value == 0) return error.Unsupported;
    parser.reset(m); // its nodes only proved it: they never print
    parser.asserted = true;
}

fn printRoots(parser: *ast.Parser, outputs: *Outputs, text: []const u8, multiple: bool) ast.Invalid!void {
    var rest = text;
    while (true) {
        const comma = statements.separator(rest, ',');
        if (comma != null and !multiple) return error.Unsupported;
        const ending: u8 = if (comma != null) ' ' else '\n';
        const v = try side(parser, rest[0 .. comma orelse rest.len]);
        if (v.list) {
            // Python's repr: [1, -2, True]; an empty list or a range object prints no value, so it refuses
            const l = parser.lists[v.index];
            if (!multiple or l.range or l.masked or l.len == 0 or outputs.count + l.len > outputs.roots.len) return error.Unsupported;
            for (parser.items[l.start..][0..l.len], 0..) |e, i| {
                outputs.roots[outputs.count] = e;
                outputs.endings[outputs.count] = if (i + 1 == l.len) ending else ',';
                outputs.brackets[outputs.count] = @as(u8, @intFromBool(i == 0)) | @as(u8, if (i + 1 == l.len) 2 else 0);
                outputs.count += 1;
            }
        } else {
            if (outputs.count == outputs.roots.len) return error.Unsupported;
            outputs.roots[outputs.count] = v.index;
            outputs.endings[outputs.count] = ending;
            outputs.count += 1;
        }
        if (comma) |i| rest = rest[i + 1 ..] else break;
    }
}

pub fn compile(a: Allocator, io: std.Io, code: []const u8, config: compiler.Config) Allocator.Error!?[]u8 {
    const result = try lowerWithCost(a, code, config.step_budget) orelse return null;
    defer a.free(result.source);
    if (result.graph_only) return null;
    var trusted = config;
    trusted.estimated_steps = result.estimated_steps;
    if (trusted.optimize != .O0) trusted.optimize = .O2;
    trusted.opaque_inputs = true;
    return compiler.compile(a, io, result.source, trusted);
}

/// math's gcd/isqrt/factorial/prod as a def's own name, where the script never imports it bare.
fn unimported(parser: *const ast.Parser, name: []const u8) bool {
    for ([_][]const u8{ "gcd", "isqrt", "factorial", "prod" }, 0..) |w, i| {
        if (std.mem.eql(u8, name, w)) return parser.math_imports & (@as(u8, 1) << @intCast(i)) == 0;
    }
    return false;
}

pub fn validBinding(name: []const u8) bool {
    if (name.len == 0 or !(std.ascii.isAlphabetic(name[0]) or name[0] == '_')) return false;
    for (name) |c| if (!(std.ascii.isAlphanumeric(c) or c == '_')) return false;
    for ([_][]const u8{
        "print",  "pow",    "divmod",    "int",       "str",          "len",     "sum",    "map",    "range", "all",   "abs",       "math",
        "gcd",    "isqrt",  "factorial", "prod",      "True",         "False",   "None",   "and",    "or",    "not",   "if",        "else",
        "for",    "while",  "in",        "is",        "def",          "return",  "import", "from",   "as",    "class", "del",       "yield",
        "lambda", "global", "nonlocal",  "try",       "except",       "finally", "with",   "assert", "raise", "pass",  "break",     "continue",
        "async",  "await",  "elif",      "__debug__", "__builtins__", "min",     "max",    "sorted", "list",  "zip",   "enumerate",
    }) |reserved| if (std.mem.eql(u8, name, reserved)) return false;
    return true;
}

test "wide assignment bindings preserve requests and validate unused expressions" {
    const a = std.testing.allocator;
    for ([_][]const u8{
        "a = 314159\nb = 271828\nprint(a * b)",
        "x = 10**40\ny = pow(x,2,97)\nprint(y)",
        "a = 7\na = a+2\nprint(a*a)",
        "import math\nn = 12345678901234567890\nprint(math.isqrt(n))",
        "from math import gcd\na = 48\nb = 18\nprint(gcd(a,b))",
    }) |code| {
        const source = (try lower(a, code)).?;
        defer a.free(source);
        try std.testing.expect(std.mem.indexOf(u8, source, "tf_run(tf_nodes,") != null);
    }
    for ([_][]const u8{
        "x = 1//0\nprint(3)",
        "x = 10**62\nprint(3)",
        "x = x+1\nprint(x)",
        "print(math.gcd(48,18))",
        "print(gcd(48,18))",
        "from math import *\nprint(3)",
        "import math as m\nprint(m.gcd(48,18))",
        "pow = 2\nprint(pow)",
        "a = divmod(7,3)\nprint(a)",
        "import os\nprint(3)",
        "x = 1.25\nprint(x)",
        "__builtins__ = 0\nprint(3)",
        "x = 3\nx*2",
    }) |code| try std.testing.expectEqual(@as(?[]u8, null), try lower(a, code));
}
