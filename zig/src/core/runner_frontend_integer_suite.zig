//! Suites in the whole-script integer lowering: def bodies and for/while loops, unrolled into nodes.
const std = @import("std");
const ast = @import("runner_frontend_integer_ast.zig");
const statements = @import("runner_frontend_integer_statements.zig");
const list = @import("runner_frontend_integer_list.zig");
const Parser = ast.Parser;
const Invalid = ast.Invalid;

// Value decisions (a trip, a while test, a base case) are 0/1 nodes gating each assignment; unroll counts are schedule.

/// Iterations one loop may unroll (a range's length, a while's true tests); past it the script refuses.
pub const max_trips = 32;

/// A module statement's `print(...)` arguments, printed as the script's own outputs.
pub const Printer = struct { ctx: *anyopaque, print: *const fn (*anyopaque, *Parser, []const u8) Invalid!void };

const Line = struct { indent: usize, text: []const u8 };
const Lines = struct {
    items: [96]Line = undefined,
    len: usize = 0,

    fn push(l: *Lines, indent: usize, text: []const u8) Invalid!void {
        if (l.len == l.items.len) return error.Unsupported;
        l.items[l.len] = .{ .indent = indent, .text = text };
        l.len += 1;
    }
};

pub const Run = struct {
    p: *Parser,
    lines: *const Lines,
    function: bool, // a def's body: locals bind in its frame, `return` ends it, print refuses (a side effect)
    printer: ?Printer = null,
    pred: ?usize = null, // the 0/1 node gating every assignment here; null: unconditional
    check: bool = false, // at the def: syntax only (each loop body once, both branches parsed), as Python compiles it
    divided: ?struct { line: usize, name: []const u8, root: usize } = null, // a digit loop's `x //= L`: its closed form
};

/// A def's suite inlined at a call: its frame holds the parameters; every name it assigns is local from the start.
pub fn call(p: *Parser, suite: []const u8) Invalid!usize {
    const lines = try split(suite);
    p.looped = true;
    try prebind(p, &lines);
    var r: Run = .{ .p = p, .lines = &lines, .function = true };
    return (try exec(&r, 0, lines.len, true)) orelse error.Unsupported;
}

/// A def's suite as Python compiles it: statement shapes and expression syntax; names resolve at each call.
pub fn check(p: *Parser, suite: []const u8) Invalid!void {
    const lines = try split(suite);
    const saved = .{ p.source, p.pos, p.count, p.item_count, p.list_count, p.check_only, p.binding_count };
    defer {
        p.source, p.pos, p.count, p.item_count, p.list_count, p.check_only, p.binding_count = saved;
    }
    p.check_only = true;
    var r: Run = .{ .p = p, .lines = &lines, .function = true, .check = true };
    _ = try exec(&r, 0, lines.len, true);
}

/// A module-level `for`/`while` statement and its suite; prints only where no value decides whether they run.
pub fn module(p: *Parser, block: []const u8, printer: Printer) Invalid!void {
    const lines = try split(block);
    p.looped = true;
    var r: Run = .{ .p = p, .lines = &lines, .function = false, .printer = printer };
    if (try exec(&r, 0, lines.len, false) != null) return error.Unsupported;
}

/// A suite's statements with their columns; a header's same-line body sits one column deeper.
fn split(code: []const u8) Invalid!Lines {
    var out: Lines = .{};
    var it = statements.Iterator{ .code = code };
    var last: usize = 0;
    while (try it.next()) |raw| {
        const text = std.mem.trimEnd(u8, raw.text, " \t");
        const body = std.mem.trimStart(u8, text, " \t");
        if (body.len == 0) continue;
        var indent = last;
        if (!raw.after_semicolon) {
            indent = 0;
            for (text[0 .. text.len - body.len]) |c| indent = if (c == '\t') (indent / 8 + 1) * 8 else indent + 1;
        }
        last = indent;
        if (header(body)) |colon| {
            try out.push(indent, body[0 .. colon + 1]);
            const rest = std.mem.trim(u8, body[colon + 1 ..], " \t");
            if (rest.len != 0) {
                last = indent + 1;
                try out.push(last, rest);
            }
        } else try out.push(indent, body);
    }
    if (out.len == 0) return error.Unsupported;
    return out;
}

/// The colon ending a compound statement's header, or null for a simple statement.
fn header(text: []const u8) ?usize {
    for ([_][]const u8{ "for", "while", "if", "elif", "else", "def", "class", "with", "try", "except", "finally" }) |w| {
        if (word(text, w)) return statements.separator(text, ':') orelse text.len - 1;
    }
    return null;
}

fn word(text: []const u8, w: []const u8) bool {
    if (!std.mem.startsWith(u8, text, w)) return false;
    return text.len == w.len or !(std.ascii.isAlphanumeric(text[w.len]) or text[w.len] == '_');
}

/// The first line past `i` at `i`'s column or left of it: the end of i's suite.
fn blockEnd(lines: *const Lines, i: usize, end: usize) usize {
    var j = i + 1;
    while (j < end and lines.items[j].indent > lines.items[i].indent) j += 1;
    return j;
}

/// Lines start..end, one block at one column; the returned root when a `return` ends it (`top`: the def's own suite).
pub fn exec(r: *Run, start: usize, end: usize, top: bool) Invalid!?usize {
    if (start >= end) return error.Unsupported;
    const column = r.lines.items[start].indent;
    var i = start;
    while (i < end) {
        const line = r.lines.items[i];
        if (line.indent != column) return error.Unsupported;
        const next = blockEnd(r.lines, i, end);
        const t = line.text;
        if (r.divided) |d| if (d.line == i) {
            try store(r, d.name, d.root);
            i = next;
            continue;
        };
        const compound = header(t) != null;
        if (compound and (next == i + 1 or t[t.len - 1] != ':')) return error.Unsupported;
        if (!compound and next != i + 1) return error.Unsupported; // an indented line under a simple statement
        if (word(t, "for")) {
            try forLoop(r, t["for".len .. t.len - 1], i + 1, next);
        } else if (word(t, "while")) {
            try whileLoop(r, t["while".len .. t.len - 1], i + 1, next);
        } else if (word(t, "if")) {
            if (!r.function or !top or r.pred != null) return error.Unsupported;
            return try early(r, t["if".len .. t.len - 1], i + 1, next, end);
        } else if (compound) {
            return error.Unsupported; // elif, else, def, class, with, try
        } else if (word(t, "return")) {
            if (!r.function or !top or r.pred != null or next != end) return error.Unsupported;
            return try value(r, t["return".len..]);
        } else if (i == start and top and r.function and t.len >= 2 and (t[0] == '"' or t[0] == '\'') and t[t.len - 1] == t[0]) {
            // a docstring
        } else if (std.mem.eql(u8, t, "pass")) {} else if (word(t, "print")) {
            const args = std.mem.trim(u8, t["print".len..], " \t");
            if (args.len < 2 or args[0] != '(' or args[args.len - 1] != ')') return error.Unsupported;
            const pr = r.printer orelse return error.Unsupported;
            if (r.pred != null or r.check) return error.Unsupported;
            try pr.print(pr.ctx, r.p, args[1 .. args.len - 1]);
        } else try assignment(r, t);
        i = next;
    }
    return null;
}

/// An expression's root: proved, or at the def syntax only.
fn expression(r: *Run, text: []const u8) Invalid!usize {
    const p = r.p;
    p.source = std.mem.trim(u8, text, " \t");
    p.pos = 0;
    if (!r.check) return p.parse();
    if (p.source.len == 0) return error.Unsupported;
    const root = try p.expr(0);
    p.space();
    if (p.pos != p.source.len) return error.Unsupported;
    return root;
}

/// A returned value: `A if C else B` chooses on the graph, else one int expression.
fn value(r: *Run, text: []const u8) Invalid!usize {
    const t = std.mem.trim(u8, text, " \t");
    if (t.len == 0 or statements.separator(t, ',') != null) return error.Unsupported;
    const at = list.top(t, "if", 0) orelse return expression(r, t);
    const otherwise = list.top(t, "else", at + 2) orelse return error.Unsupported;
    return choose(r, t[at + 2 .. otherwise], .{ .text = t[0..at] }, .{ .text = t[otherwise + 4 ..] });
}

const Branch = union(enum) { text: []const u8, rest: [2]usize };

fn branch(r: *Run, b: Branch) Invalid!usize {
    return switch (b) {
        .text => |t| value(r, t),
        .rest => |x| (try exec(r, x[0], x[1], true)) orelse error.Unsupported, // falling off the end returns None
    };
}

/// `if C: return A` then the rest of the suite: A if C else (the rest).
fn early(r: *Run, cond: []const u8, body: usize, body_end: usize, end: usize) Invalid!usize {
    const ret = r.lines.items[body].text;
    if (body_end != body + 1 or !word(ret, "return") or body_end == end) return error.Unsupported;
    if (word(r.lines.items[body_end].text, "elif") or word(r.lines.items[body_end].text, "else")) return error.Unsupported;
    return choose(r, cond, .{ .text = ret["return".len..] }, .{ .rest = .{ body_end, end } });
}

/// Bindings as they were: a branch's assignments never reach the other branch.
pub const Saved = struct {
    count: usize,
    roots: [64]usize,
    kinds: [64]bool,

    pub fn take(p: *const Parser) Saved {
        return .{ .count = p.binding_count, .roots = p.roots, .kinds = p.kinds };
    }

    pub fn put(s: *const Saved, p: *Parser) void {
        p.binding_count = s.count;
        p.roots = s.roots;
        p.kinds = s.kinds;
    }
};

/// c*A + (1-c)*B, both built: the branch C takes, then the other unless it needs recursion (the schedule's last level).
fn choose(r: *Run, cond: []const u8, a: Branch, b: Branch) Invalid!usize {
    const p = r.p;
    const c = try expression(r, cond);
    if (!p.boolean(c)) return error.Unsupported;
    const saved = Saved.take(p);
    if (r.check) {
        _ = try branch(r, a);
        saved.put(p);
        _ = try branch(r, b);
        saved.put(p);
        return c;
    }
    const yes = p.nodes[c].value != 0;
    const before = p.recursions;
    const taken = try branch(r, if (yes) a else b);
    const recursed = p.recursions != before;
    saved.put(p);
    const m = p.mark();
    const was = .{ p.no_recurse, p.cut };
    p.no_recurse, p.cut = .{ true, false };
    const other: ?usize = branch(r, if (yes) b else a) catch null;
    const cut = p.cut;
    p.no_recurse, p.cut = was;
    saved.put(p);
    const o = other orelse {
        // only a base case may stand alone: the other branch recursed past the schedule, and this one did not recurse
        if (!cut or recursed) return error.Unsupported;
        p.reset(m);
        return taken;
    };
    if (p.boolean(taken) or p.boolean(o)) return error.Unsupported; // c*True is 1, not True
    return blend(p, c, if (yes) taken else o, if (yes) o else taken);
}

/// c*a + (1-c)*b over a 0/1 node c.
fn blend(p: *Parser, c: usize, a: usize, b: usize) Invalid!usize {
    if (p.boolean(a) or p.boolean(b)) return error.Unsupported;
    const off = try node(p, .sub, try literal(p, 1), c);
    return node(p, .add, try node(p, .mul, c, a), try node(p, .mul, off, b));
}

pub fn node(p: *Parser, op: ast.Op, l: usize, rhs: usize) Invalid!usize {
    const n = try p.add(.{ .op = op, .l = l, .r = rhs });
    _ = try p.prove(n, n);
    return n;
}

/// A constant of the lowering itself (a 0/1 offset, a digit's place, an iteration's index): schedule, not input.
pub fn literal(p: *Parser, v: i512) Invalid!usize {
    return p.add(.{ .op = .literal, .value = v, .text = "" });
}

/// `T op= E` or `T = E` on one int name; under a predicate the old value stays where the test is 0.
fn assignment(r: *Run, t: []const u8) Invalid!void {
    const p = r.p;
    const eq = statements.separator(t, '=') orelse return error.Unsupported;
    var name = std.mem.trim(u8, t[0..eq], " \t");
    var op: ?ast.Op = null;
    if (std.mem.endsWith(u8, name, "//")) {
        op = .floor;
        name = name[0 .. name.len - 2];
    } else if (name.len != 0) {
        op = switch (name[name.len - 1]) {
            '+' => .add,
            '-' => .sub,
            '*' => .mul,
            '%' => .mod,
            else => null,
        };
        if (op != null) name = name[0 .. name.len - 1];
    }
    name = std.mem.trim(u8, name, " \t");
    if (!@import("runner_frontend_integer_python.zig").validBinding(name)) return error.Unsupported;
    const v = try expression(r, t[eq + 1 ..]);
    if (r.check) return;
    const old: ?usize = if (p.slot(name)) |s| (if (p.kinds[s] or p.roots[s] == ast.unbound) null else p.roots[s]) else null;
    const new = if (op) |o| try update(p, r.pred, o, old orelse return error.Unsupported, v) else if (r.pred) |c|
        try blend(p, c, v, old orelse return error.Unsupported) // a name first bound under a test may stay unbound
    else
        v;
    try store(r, name, new);
}

/// old op v; under c: x + c*v, x - c*v, x*(1 + c*(v-1)), x*(v - c*(v-1)) // v (v a literal); `%=` refuses.
fn update(p: *Parser, pred: ?usize, op: ast.Op, old: usize, v: usize) Invalid!usize {
    const c = pred orelse return node(p, op, old, v);
    const one = try literal(p, 1);
    return switch (op) {
        .add, .sub => node(p, op, old, try node(p, .mul, c, v)),
        .mul => node(p, .mul, old, try node(p, .add, one, try node(p, .mul, c, try node(p, .sub, v, one)))),
        .floor => blk: {
            if (p.nodes[v].op != .literal or p.nodes[v].value == 0) return error.Unsupported;
            const keep = try node(p, .sub, v, try node(p, .mul, c, try node(p, .sub, v, one)));
            break :blk node(p, .floor, try node(p, .mul, old, keep), v);
        },
        else => error.Unsupported,
    };
}

/// A def's locals in its frame; module names as the module binds them.
pub fn store(r: *Run, name: []const u8, root: usize) Invalid!void {
    const p = r.p;
    if (!r.function) return p.bind(name, root);
    const f = p.frame orelse return error.Unsupported;
    var i = p.binding_count;
    while (i > f.start) {
        i -= 1;
        if (std.mem.eql(u8, p.names[i], name)) {
            p.roots[i] = root;
            p.kinds[i] = false;
            return;
        }
    }
    return error.Unsupported; // prebind made every assigned name local
}

/// Every name a def's suite assigns is local for the whole body: read before it is assigned, Python raises.
fn prebind(p: *Parser, lines: *const Lines) Invalid!void {
    const f = p.frame orelse return error.Unsupported;
    for (lines.items[0..lines.len]) |line| {
        const name = assigns(line.text) orelse continue;
        const known = for (p.names[f.start..p.binding_count]) |held| {
            if (std.mem.eql(u8, held, name)) break true;
        } else false;
        if (known) continue;
        if (p.binding_count == p.names.len) return error.Unsupported;
        p.names[p.binding_count] = name;
        p.roots[p.binding_count] = ast.unbound;
        p.kinds[p.binding_count] = false;
        p.binding_count += 1;
    }
}

/// The name a line binds: a `for` target, or an assignment's (augmented or not).
pub fn assigns(t: []const u8) ?[]const u8 {
    if (word(t, "for")) {
        const at = list.top(t, "in", 3) orelse return null;
        return std.mem.trim(u8, t[3..at], " \t");
    }
    if (header(t) != null or word(t, "return")) return null;
    const eq = statements.separator(t, '=') orelse return null;
    return std.mem.trim(u8, std.mem.trimEnd(u8, std.mem.trim(u8, t[0..eq], " \t"), "+-*/%"), " \t");
}

/// `for T in range(...)` (predicated per trip) or over a literal list (its length is the script's text).
fn forLoop(r: *Run, head: []const u8, body: usize, end: usize) Invalid!void {
    const p = r.p;
    const at = list.top(head, "in", 0) orelse return error.Unsupported;
    const target = std.mem.trim(u8, head[0..at], " \t");
    if (!@import("runner_frontend_integer_python.zig").validBinding(target)) return error.Unsupported;
    const iter = std.mem.trim(u8, head[at + 2 ..], " \t");
    if (std.mem.startsWith(u8, iter, "range(") and iter[iter.len - 1] == ')') return rangeLoop(r, target, iter[6 .. iter.len - 1], body, end);
    p.source = iter;
    p.pos = 0;
    if (!r.check and !list.ahead(p)) return error.Unsupported;
    const li = try list.whole(p, iter);
    if (r.check) {
        _ = try exec(r, body, end, false);
        return;
    }
    const l = p.lists[li];
    if (!l.literal) return error.Unsupported; // a range object, filter or sort: its length is a value's decision
    var elems: [ast.max_items]usize = undefined;
    @memcpy(elems[0..l.len], p.items[l.start..][0..l.len]);
    for (elems[0..l.len]) |e| {
        try store(r, target, e);
        if (try exec(r, body, end, false) != null) return error.Unsupported;
    }
}

/// range(stop) / range(start, stop[, step]): trip k runs where start + k*step is still short of stop.
fn rangeLoop(r: *Run, target: []const u8, args_text: []const u8, body: usize, end: usize) Invalid!void {
    const p = r.p;
    var args: [3]usize = undefined;
    var n: usize = 0;
    var rest = args_text;
    while (true) {
        if (n == 3) return error.Unsupported;
        const comma = statements.separator(rest, ',');
        args[n] = try expression(r, rest[0 .. comma orelse rest.len]);
        if (!r.check and p.boolean(args[n])) return error.Unsupported;
        n += 1;
        rest = rest[(comma orelse break) + 1 ..];
    }
    if (r.check) {
        _ = try exec(r, body, end, false);
        return;
    }
    const start: ?usize = if (n == 1) null else args[0];
    const stop = if (n == 1) args[0] else args[1];
    const step: ?usize = if (n == 3) args[2] else null;
    const a: i512 = if (start) |s| p.nodes[s].value else 0;
    const b = p.nodes[stop].value;
    const s: i512 = if (step) |x| p.nodes[x].value else 1;
    if (s == 0) return error.Unsupported; // ValueError
    const span = if (s > 0) b - a else a - b;
    const mag = if (s > 0) s else -s;
    const trips: i512 = if (span <= 0) 0 else @divFloor(span + mag - 1, mag);
    if (trips > max_trips) return error.Unsupported;
    // headroom: the schedule's trips are a power of two past the script's own, so a larger input still runs exactly
    const m = p.mark();
    const saved = Saved.take(p);
    const spare = schedule(@intCast(trips));
    if (spare > trips) {
        if (trip(r, target, .{ start, stop, step }, s > 0, spare, body, end)) |_| {
            if (!heavy(p, null)) return unbind(r, target);
        } else |_| {}
        p.reset(m);
        saved.put(p);
    }
    try trip(r, target, .{ start, stop, step }, s > 0, @intCast(trips), body, end);
    try unbind(r, target); // after the loop its value is the last trip's, which a value decided
}

/// Whether a bound value outgrows headroom: its calc tree (shared nodes repeat) or the graph route's digit bound.
pub fn heavy(p: *const Parser, extra: ?usize) bool {
    var size: [ast.max_nodes]u32 = undefined;
    var digits: [ast.max_nodes]u32 = undefined;
    for (p.nodes[0..p.count], 0..) |n, i| {
        const l: u32 = if (n.op == .literal) 0 else digits[n.l];
        const r: u32 = if (n.op == .literal) 0 else digits[n.r];
        size[i] = switch (n.op) {
            .literal => 1,
            .neg, .digits, .isqrt, .factorial, .digit_sum => 1 +| size[n.l],
            else => 1 +| size[n.l] +| size[n.r] +| (if (n.op == .powmod or n.op == .range_product) size[n.k] else 0),
        };
        digits[i] = switch (n.op) {
            .literal => @intCast(std.fmt.count("{d}", .{@abs(n.value)})),
            .add, .sub => @max(l, r) + 1,
            .mul => l +| r,
            .floor => @min(l, @max(1, (l + 2) -| r)),
            .mod => r,
            .eq, .ne, .lt, .le, .gt, .ge => 5, // the table's sign trick: (d + M - 1) // M, and 1 - that
            else => l,
        };
    }
    for (p.roots[0..p.binding_count], p.kinds[0..p.binding_count]) |root, listed| {
        if (listed or root == ast.unbound or root >= p.count) continue;
        if (size[root] > 250 or digits[root] > 120) return true;
    }
    const x = extra orelse return false;
    return size[x] > 250 or digits[x] > 120;
}

/// A power of two at least `n` (at least one trip), within the budget.
pub fn schedule(n: usize) usize {
    return @min(max_trips, std.math.ceilPowerOfTwo(usize, @max(n, 1)) catch max_trips);
}

/// `count` trips of a range loop's body, each gated by its index short of stop (`up`) or past it.
fn trip(r: *Run, target: []const u8, bounds: [3]?usize, up: bool, count: usize, body: usize, end: usize) Invalid!void {
    const p = r.p;
    const start, const stop, const step = bounds;
    for (0..count) |k| {
        const kk = try literal(p, @intCast(k));
        const index = if (start) |first| (if (k == 0) first else try node(p, .add, first, if (step) |x| try node(p, .mul, kk, x) else kk)) else kk;
        const c = try node(p, if (up) .lt else .gt, index, stop.?);
        var inner = r.*;
        inner.pred = if (r.pred) |outer| try node(p, .mul, outer, c) else c;
        try store(r, target, index);
        if (try exec(&inner, body, end, false) != null) return error.Unsupported;
    }
}

/// `while C:` unrolled while C is true here; trip k runs where every test so far is 1.
fn whileLoop(r: *Run, cond: []const u8, body: usize, end: usize) Invalid!void {
    const p = r.p;
    if (r.check) {
        _ = try expression(r, cond);
        _ = try exec(r, body, end, false);
        return;
    }
    if (try @import("runner_frontend_integer_digits.zig").whileDivides(r, cond, body, end)) return;
    var active = r.pred;
    var k: usize = 0;
    while (true) : (k += 1) {
        const c = try expression(r, cond);
        if (!p.boolean(c)) return error.Unsupported;
        if (p.nodes[c].value == 0) break;
        if (k == max_trips) return error.Unsupported; // unbounded here, or past the budget
        active = if (active) |x| try node(p, .mul, x, c) else c;
        var inner = r.*;
        inner.pred = active;
        if (try exec(&inner, body, end, false) != null) return error.Unsupported;
    }
}

fn unbind(r: *Run, name: []const u8) Invalid!void {
    const s = r.p.slot(name) orelse return error.Unsupported;
    r.p.roots[s] = ast.unbound;
}

test "suite lines: columns, same-line bodies and semicolons" {
    const lines = try split("for i in range(3): s += i; t += 1\nwhile x:\n    x -= 1\n");
    try std.testing.expectEqual(@as(usize, 5), lines.len);
    try std.testing.expectEqualStrings("for i in range(3):", lines.items[0].text);
    try std.testing.expectEqual(@as(usize, 1), lines.items[1].indent);
    try std.testing.expectEqual(@as(usize, 1), lines.items[2].indent);
    try std.testing.expectEqual(@as(usize, 4), lines.items[4].indent);
}
