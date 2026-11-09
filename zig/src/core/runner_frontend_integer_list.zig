//! Static int lists in the integer lowering: literals, range, list(), sorted(), comprehensions; len/sum/min/max/[i].
const std = @import("std");
const ast = @import("runner_frontend_integer_ast.zig");
const Parser = ast.Parser;
const Invalid = ast.Invalid;
const node_cap = @typeInfo(@FieldType(Parser, "nodes")).array.len;

/// items[start..][0..len]; a range never prints here; a filtered list's 0/1 masks follow (its len is a circuit value).
pub const List = struct { start: usize = 0, len: usize = 0, range: bool = false, masked: bool = false, literal: bool = false };

/// A list whose elements all exist: a filtered one only reaches sum() and len(), on the circuit.
pub fn plain(p: *const Parser, li: usize) Invalid!List {
    const l = p.lists[li];
    if (l.masked) return error.Unsupported;
    return l;
}

/// Whether node `i` is the script's text alone: no comparison (a filter's or a min's choice) beneath it.
fn fixed(p: *const Parser, i: usize) bool {
    var seen: [node_cap]bool = @splat(false);
    return fixedAt(p, i, &seen);
}

fn fixedAt(p: *const Parser, i: usize, seen: *[node_cap]bool) bool {
    if (i >= p.count) return false;
    if (seen[i]) return true;
    seen[i] = true;
    const n = p.nodes[i];
    return switch (n.op) {
        .literal => true,
        .eq, .ne, .lt, .le, .gt, .ge => false,
        .neg, .digits, .isqrt, .factorial, .digit_sum => fixedAt(p, n.l, seen),
        .powmod, .range_product => fixedAt(p, n.l, seen) and fixedAt(p, n.r, seen) and fixedAt(p, n.k, seen),
        else => fixedAt(p, n.l, seen) and fixedAt(p, n.r, seen),
    };
}

/// `acc` or `e`, chosen on the circuit: acc + (e - acc) * (e < acc), or `>` for max; a tie keeps acc (Python's first).
fn pick(p: *Parser, acc: usize, e: usize, largest: bool) Invalid!usize {
    const better = try p.add(.{ .op = if (largest) .gt else .lt, .l = e, .r = acc });
    const step = try p.add(.{ .op = .mul, .l = try p.add(.{ .op = .sub, .l = e, .r = acc }), .r = better });
    return p.add(.{ .op = .add, .l = acc, .r = step });
}

/// min and max of a and b on the circuit: t = (b - a) * (b < a); min = a + t, max = b - t.
fn exchange(p: *Parser, a: usize, b: usize) Invalid![2]usize {
    const t = try p.add(.{ .op = .mul, .l = try p.add(.{ .op = .sub, .l = b, .r = a }), .r = try p.add(.{ .op = .lt, .l = b, .r = a }) });
    return .{ try p.add(.{ .op = .add, .l = a, .r = t }), try p.add(.{ .op = .sub, .l = b, .r = t }) };
}

/// sorted() as a compare-exchange network, each output a circuit value; its depth sets the packet's size.
fn network(p: *Parser, elems: []usize, reverse: bool) Invalid!void {
    for (elems) |e| if (p.boolean(e)) return error.Unsupported; // sorted([True, 0]) keeps its bools
    if (elems.len == 4) {
        // the depth-3 network for four (transposition's depth 4 outgrows a packet)
        for ([_][2]usize{ .{ 0, 1 }, .{ 2, 3 }, .{ 0, 2 }, .{ 1, 3 }, .{ 1, 2 } }) |c| {
            const lo, const hi = try exchange(p, elems[c[0]], elems[c[1]]);
            elems[c[0]], elems[c[1]] = if (reverse) .{ hi, lo } else .{ lo, hi };
        }
        if (p.count > network_nodes) return error.Unsupported;
        return;
    }
    // odd-even transposition: n rounds
    for (0..elems.len) |round| {
        var i = round % 2;
        while (i + 1 < elems.len) : (i += 2) {
            const lo, const hi = try exchange(p, elems[i], elems[i + 1]);
            elems[i], elems[i + 1] = if (reverse) .{ hi, lo } else .{ lo, hi };
        }
    }
    // the 64-node budget the networks were sized for (a packet's depth), though a whole script may make 512 nodes
    if (p.count > network_nodes) return error.Unsupported;
}

const network_nodes = 64;

/// Small literals the lowering itself makes (an index, a length): their text, as a literal node must carry one.
const digits = blk: {
    @setEvalBranchQuota(20000);
    var texts: [ast.max_items + 1][]const u8 = undefined;
    for (&texts, 0..) |*t, i| t.* = std.fmt.comptimePrint("{d}", .{i});
    break :blk texts;
};

fn number(p: *Parser, v: usize) Invalid!usize {
    if (v > ast.max_items) return error.Unsupported;
    return p.add(.{ .op = .literal, .value = @intCast(v), .text = digits[v] });
}

/// Whether a list expression starts here (nothing consumed): `[`, `range(`/`sorted(`/`list(`, or a name holding one.
pub fn ahead(p: *Parser) bool {
    if (!p.allow_lists) return false;
    p.space();
    const save = p.pos;
    defer p.pos = save;
    if (p.pos < p.source.len and p.source[p.pos] == '[') return true;
    for ([_][]const u8{ "range", "sorted", "list" }) |w| {
        p.pos = save;
        if (p.word(w)) {
            p.space();
            return p.pos < p.source.len and p.source[p.pos] == '(';
        }
    }
    p.pos = save;
    if (p.check_only) return false; // a def's body at its definition: a name may be anything
    const name = p.identifier() catch return false;
    p.space();
    if (p.pos < p.source.len and p.source[p.pos] == '(') return false;
    return p.lookupList(name) != null;
}

/// `text` as one whole list expression (an assignment's right side, a print argument).
pub fn whole(p: *Parser, text: []const u8) Invalid!usize {
    p.source = std.mem.trim(u8, text, " \t");
    p.pos = 0;
    const li = try parse(p, 0);
    p.space();
    if (p.pos != p.source.len) return error.Unsupported;
    return li;
}

/// A list expression here. Never a node: its length is the script's shape; its filter and order are circuit values.
pub fn parse(p: *Parser, depth: usize) Invalid!usize {
    if (depth > 12) return error.Unsupported;
    p.space();
    if (p.pos < p.source.len and p.source[p.pos] == '[') {
        const close = closing(p.source, p.pos) orelse return error.Unsupported;
        const content = p.source[p.pos + 1 .. close];
        const li = if (top(content, "for", 0) != null) try comprehension(p, content, depth) else try literal(p, content, depth);
        p.pos = close + 1;
        return li;
    }
    if (p.word("range")) return range(p, depth);
    const sorting = p.word("sorted");
    if (sorting or p.word("list")) {
        const content = try parenthesized(p);
        var arg = content;
        var reverse = false;
        // a bare generator is the only argument (its targets' commas are its own)
        const generator = top(content, "for", 0) != null;
        if (sorting and !generator) {
            // sorted(xs, reverse=True|False); key= or anything else refuses
            if (separator(content)) |comma| {
                const kw = std.mem.trim(u8, content[comma + 1 ..], " \t");
                if (std.mem.eql(u8, kw, "reverse=True")) reverse = true else if (!std.mem.eql(u8, kw, "reverse=False")) return error.Unsupported;
                arg = content[0..comma];
            }
        } else if (!generator and separator(content) != null) return error.Unsupported;
        const src = try argument(p, arg, depth);
        if (p.check_only) return src;
        const l = try plain(p, src);
        var elems: [ast.max_items]usize = undefined;
        @memcpy(elems[0..l.len], p.items[l.start..][0..l.len]);
        if (sorting) try network(p, elems[0..l.len], reverse);
        return make(p, elems[0..l.len], false);
    }
    const name = try p.identifier();
    if (p.check_only) return placeholder(p);
    return p.lookupList(name) orelse error.Unsupported;
}

/// The argument of sorted()/list()/sum()/min()/max(): a bare generator expression, or a list.
fn argument(p: *Parser, text: []const u8, depth: usize) Invalid!usize {
    const t = std.mem.trim(u8, text, " \t");
    if (top(t, "for", 0) != null) return comprehension(p, t, depth);
    return nested(p, t, depth);
}

/// `text` as one whole list expression, the parser's position kept.
fn nested(p: *Parser, text: []const u8, depth: usize) Invalid!usize {
    const saved = .{ p.source, p.pos };
    defer {
        p.source, p.pos = saved;
    }
    p.source = std.mem.trim(u8, text, " \t");
    p.pos = 0;
    const li = try parse(p, depth + 1);
    p.space();
    if (p.pos != p.source.len) return error.Unsupported;
    return li;
}

fn placeholder(p: *Parser) Invalid!usize {
    const one = try p.add(.{ .op = .literal, .value = 1, .text = "1" });
    return make(p, &.{one}, false);
}

/// A new list of these element nodes, each proved now (Python builds a list eagerly: `[1//0]` raises at once).
pub fn make(p: *Parser, elems: []const usize, is_range: bool) Invalid!usize {
    p.listed = true;
    if (p.list_count == p.lists.len or p.item_count + elems.len > p.items.len) return error.Unsupported;
    for (elems) |e| {
        if (e >= p.count or p.nodes[e].op == .divmod) return error.Unsupported; // a tuple is not an int element
        if (p.check_only) continue;
        _ = try p.prove(e, e);
    }
    @memcpy(p.items[p.item_count..][0..elems.len], elems);
    p.lists[p.list_count] = .{ .start = p.item_count, .len = elems.len, .range = is_range };
    p.item_count += elems.len;
    p.list_count += 1;
    return p.list_count - 1;
}

/// A comprehension's list with each element's 0/1 keep mask (every element proved: one a filter drops may refuse).
fn filtered(p: *Parser, elems: []const usize, masks: []const usize) Invalid!usize {
    if (p.item_count + 2 * elems.len > p.items.len) return error.Unsupported;
    const li = try make(p, elems, false);
    for (masks) |m| if (!p.check_only) {
        _ = try p.prove(m, m);
    };
    @memcpy(p.items[p.item_count..][0..masks.len], masks);
    p.item_count += masks.len;
    p.lists[li].masked = true;
    return li;
}

fn literal(p: *Parser, content: []const u8, depth: usize) Invalid!usize {
    var elems: [ast.max_items]usize = undefined;
    var n: usize = 0;
    const saved = .{ p.source, p.pos };
    defer {
        p.source, p.pos = saved;
    }
    p.source = content;
    p.pos = 0;
    p.space();
    while (p.pos < p.source.len) {
        if (n == elems.len) return error.Unsupported;
        elems[n] = try p.expr(depth + 1);
        n += 1;
        p.space();
        if (p.pos == p.source.len) break;
        try p.need(",");
        p.space();
    }
    const li = try make(p, elems[0..n], false);
    p.lists[li].literal = true; // its length is the script's text, never a value's decision
    return li;
}

/// range(n) is the literals 0..n-1; range(start, stop[, step]) is start, then each element before plus the step.
fn range(p: *Parser, depth: usize) Invalid!usize {
    var args: [3]usize = undefined;
    var n: usize = 0;
    try p.need("(");
    while (true) {
        if (n == 3) return error.Unsupported;
        args[n] = try p.expr(depth + 1);
        n += 1;
        if (p.take(")")) break;
        try p.need(",");
    }
    if (p.check_only) return placeholder(p);
    // the length is a shape: never from a circuit choice (a filter's len, a min)
    for (args[0..n]) |a| if (p.boolean(a) or p.nodes[a].op == .divmod or !fixed(p, a)) return error.Unsupported;
    const start: ?usize = if (n == 1) null else args[0];
    const stop = if (n == 1) args[0] else args[1];
    const step: ?usize = if (n == 3) args[2] else null;
    const a: i512 = if (start) |s| try p.prove(s, s) else 0;
    const b = try p.prove(stop, stop);
    const s: i512 = if (step) |x| try p.prove(x, x) else 1;
    if (s == 0) return error.Unsupported; // ValueError in Python
    const span = if (s > 0) b - a else a - b;
    const mag = if (s > 0) s else -s;
    const len: i512 = if (span <= 0) 0 else @divFloor(span + mag - 1, mag);
    if (len > ast.max_items) return error.Unsupported;
    const count: usize = @intCast(len);
    var elems: [ast.max_items]usize = undefined;
    if (start) |first| {
        if (count != 0) elems[0] = first;
        const inc = if (count > 1) step orelse try number(p, 1) else 0;
        for (1..@max(count, 1)) |k| elems[k] = try p.add(.{ .op = .add, .l = elems[k - 1], .r = inc });
    } else for (0..count) |k| {
        elems[k] = try number(p, k);
    }
    return make(p, elems[0..count], true);
}

/// `body for target in iterable [if cond]...`: one `for` clause; targets are one name, or two over zip/enumerate.
fn comprehension(p: *Parser, content: []const u8, depth: usize) Invalid!usize {
    const f = top(content, "for", 0) orelse return error.Unsupported;
    const body = std.mem.trim(u8, content[0..f], " \t");
    const in_at = top(content, "in", f + 3) orelse return error.Unsupported;
    if (top(content, "for", f + 3) != null) return error.Unsupported; // a second for clause
    var targets_text = std.mem.trim(u8, content[f + 3 .. in_at], " \t");
    if (targets_text.len >= 2 and targets_text[0] == '(' and targets_text[targets_text.len - 1] == ')')
        targets_text = std.mem.trim(u8, targets_text[1 .. targets_text.len - 1], " \t");
    var targets: [2][]const u8 = undefined;
    var arity: usize = 0;
    var names = std.mem.splitScalar(u8, targets_text, ',');
    while (names.next()) |raw| {
        if (arity == 2) return error.Unsupported;
        targets[arity] = std.mem.trim(u8, raw, " \t");
        if (!target(targets[arity])) return error.Unsupported;
        arity += 1;
    }
    if (arity == 2 and std.mem.eql(u8, targets[0], targets[1])) return error.Unsupported;
    const first_if = top(content, "if", in_at + 2);
    const iterable = std.mem.trim(u8, content[in_at + 2 .. first_if orelse content.len], " \t");
    var conds: [4][]const u8 = undefined;
    var nconds: usize = 0;
    var at = first_if;
    while (at) |i| {
        if (nconds == conds.len) return error.Unsupported;
        const next = top(content, "if", i + 2);
        conds[nconds] = std.mem.trim(u8, content[i + 2 .. next orelse content.len], " \t");
        if (conds[nconds].len == 0) return error.Unsupported;
        nconds += 1;
        at = next;
    }
    if (body.len == 0 or iterable.len == 0) return error.Unsupported;
    // the iterable: zip(a, b) and enumerate(a[, start]) give pairs; any other list gives single values
    var columns: [2]List = undefined;
    var first: ?usize = null; // enumerate's start node
    var pairs: enum { none, zip, enumerate } = .none;
    {
        const saved = .{ p.source, p.pos };
        defer {
            p.source, p.pos = saved;
        }
        p.source = iterable;
        p.pos = 0;
        if (p.word("zip")) {
            const args = try parenthesized(p);
            const comma = separator(args) orelse return error.Unsupported;
            if (separator(args[comma + 1 ..]) != null) return error.Unsupported;
            columns[0] = try plain(p, try nested(p, args[0..comma], depth));
            columns[1] = try plain(p, try nested(p, args[comma + 1 ..], depth));
            pairs = .zip;
        } else if (p.word("enumerate")) {
            const args = try parenthesized(p);
            const comma = separator(args);
            columns[1] = try plain(p, try nested(p, args[0 .. comma orelse args.len], depth));
            if (comma) |c| {
                first = try sub(p, std.mem.trim(u8, args[c + 1 ..], " \t"), depth);
                if (p.boolean(first.?) or p.nodes[first.?].op == .divmod) return error.Unsupported;
            }
            pairs = .enumerate;
        } else {
            columns[0] = try plain(p, try parse(p, depth + 1));
        }
        p.space();
        if (p.pos != p.source.len) return error.Unsupported;
    }
    if ((pairs == .none) != (arity == 1)) return error.Unsupported;
    const len = switch (pairs) {
        .none => columns[0].len,
        .zip => @min(columns[0].len, columns[1].len),
        .enumerate => columns[1].len,
    };
    var out: [ast.max_items]usize = undefined;
    var masks: [ast.max_items]usize = undefined;
    var counter: usize = 0; // enumerate's count from its start: the start, then one more each element
    const one = if (first != null and len > 1) try number(p, 1) else 0;
    for (0..len) |k| {
        var values: [2]usize = undefined;
        switch (pairs) {
            .none => values[0] = p.items[columns[0].start + k],
            .zip => values = .{ p.items[columns[0].start + k], p.items[columns[1].start + k] },
            .enumerate => {
                counter = if (first) |s| (if (k == 0) s else try p.add(.{ .op = .add, .l = counter, .r = one })) else try number(p, k);
                values = .{ counter, p.items[columns[1].start + k] };
            },
        }
        // the targets are the comprehension's own names: they shadow, and vanish after it (Python 3 scoping)
        const bound = p.binding_count;
        defer p.binding_count = bound;
        if (p.binding_count + arity > p.names.len) return error.Unsupported;
        for (targets[0..arity], values[0..arity]) |name, value| {
            p.names[p.binding_count] = name;
            p.roots[p.binding_count] = value;
            p.kinds[p.binding_count] = false;
            p.binding_count += 1;
        }
        // the conditions' product is the element's keep mask, a circuit value: the host never chooses what stays
        var mask: ?usize = null;
        for (conds[0..nconds]) |cond| {
            const c = try sub(p, cond, depth);
            if (p.nodes[c].op == .divmod) return error.Unsupported;
            // an int condition is its truth: nonzero
            const truth = if (p.boolean(c)) c else try p.add(.{ .op = .ne, .l = c, .r = try number(p, 0) });
            mask = if (mask) |m| try p.add(.{ .op = .mul, .l = m, .r = truth }) else truth;
        }
        if (k == out.len) return error.Unsupported;
        if (mask) |m| masks[k] = m;
        out[k] = try sub(p, body, depth);
    }
    if (nconds == 0) return make(p, out[0..len], false);
    return filtered(p, out[0..len], masks[0..len]);
}

/// An int expression over `text`, the parser's position kept.
fn sub(p: *Parser, text: []const u8, depth: usize) Invalid!usize {
    const saved = .{ p.source, p.pos };
    defer {
        p.source, p.pos = saved;
    }
    p.source = text;
    p.pos = 0;
    const n = try p.expr(depth + 1);
    p.space();
    if (p.pos != p.source.len) return error.Unsupported;
    return n;
}

fn target(name: []const u8) bool {
    if (name.len == 0 or !(std.ascii.isAlphabetic(name[0]) or name[0] == '_')) return false;
    for (name) |c| if (!(std.ascii.isAlphanumeric(c) or c == '_')) return false;
    for ([_][]const u8{ "for", "in", "if", "else", "and", "or", "not", "is", "lambda", "None", "True", "False", "sum", "len", "min", "max", "range", "sorted", "list", "zip", "enumerate", "print", "pow", "divmod", "int", "str", "abs", "math", "gcd", "isqrt", "factorial", "prod" }) |w|
        if (std.mem.eql(u8, name, w)) return false;
    return true;
}

/// The text inside `( ... )` at the position (after the call's name); the position moves past `)`.
fn parenthesized(p: *Parser) Invalid![]const u8 {
    p.space();
    if (p.pos >= p.source.len or p.source[p.pos] != '(') return error.Unsupported;
    const close = closing(p.source, p.pos) orelse return error.Unsupported;
    const content = p.source[p.pos + 1 .. close];
    p.pos = close + 1;
    return content;
}

// ---- int consumers, called from the parser's atoms ----

/// len(xs): the list's static length, a literal (its new nodes dropped); a filtered list's is its masks' sum.
pub fn length(p: *Parser, depth: usize) Invalid!usize {
    const content = try parenthesized(p);
    // len() of a generator is a TypeError
    if (top(content, "for", 0) != null or separator(content) != null) return error.Unsupported;
    const m = p.mark();
    const li = try nested(p, content, depth);
    const l = p.lists[li];
    if (p.check_only or !l.masked) {
        p.reset(m);
        return number(p, if (p.check_only) 1 else l.len);
    }
    // from 0: one mask alone would print as a bool
    var acc = try number(p, 0);
    for (p.items[l.start + l.len ..][0..l.len]) |mask| acc = try p.add(.{ .op = .add, .l = acc, .r = mask });
    return acc;
}

/// sum(xs) or sum(generator): 0 for none, else the elements added left to right (start= refuses).
pub fn sum(p: *Parser, depth: usize) Invalid!usize {
    const content = try parenthesized(p);
    if (top(content, "for", 0) == null and separator(content) != null) return error.Unsupported;
    const l = p.lists[try argument(p, content, depth)];
    if (l.masked) {
        // each element times its keep mask, from 0
        var acc = try number(p, 0);
        for (p.items[l.start..][0..l.len], p.items[l.start + l.len ..][0..l.len]) |e, mask|
            acc = try p.add(.{ .op = .add, .l = acc, .r = try p.add(.{ .op = .mul, .l = e, .r = mask }) });
        return acc;
    }
    if (l.len == 0) return number(p, 0);
    var acc = p.items[l.start];
    // sum([True]) is 1, an int
    if (l.len == 1 and p.boolean(acc)) return p.add(.{ .op = .add, .l = try number(p, 0), .r = acc });
    for (p.items[l.start + 1 ..][0 .. l.len - 1]) |e| acc = try p.add(.{ .op = .add, .l = acc, .r = e });
    return acc;
}

/// min/max of one list or generator, or of two or more ints: chosen on the circuit, pairwise (an empty list refuses).
pub fn extreme(p: *Parser, depth: usize, comptime largest: bool) Invalid!usize {
    p.listed = true;
    const content = try parenthesized(p);
    var elems: [ast.max_items]usize = undefined;
    var n: usize = 0;
    if (top(content, "for", 0) == null and separator(content) != null) {
        var rest = content;
        while (true) {
            if (n == elems.len) return error.Unsupported;
            const comma = separator(rest);
            // a key= or default= does not parse as an int; a list argument is a TypeError
            elems[n] = try sub(p, std.mem.trim(u8, rest[0 .. comma orelse rest.len], " \t"), depth);
            if (p.nodes[elems[n]].op == .divmod) return error.Unsupported;
            n += 1;
            if (comma) |c| rest = rest[c + 1 ..] else break;
        }
    } else {
        const l = try plain(p, try argument(p, content, depth));
        @memcpy(elems[0..l.len], p.items[l.start..][0..l.len]);
        n = l.len;
    }
    if (n == 0) return error.Unsupported; // ValueError
    for (elems[0..n]) |e| if (p.boolean(e)) return error.Unsupported; // min(True, 2) is True, a bool
    var best = elems[0];
    for (elems[1..n]) |e| best = try pick(p, best, e, largest);
    if (!p.check_only) _ = try p.prove(best, best);
    return best;
}

/// xs[i] for a proved index (negative from the end); out of range refuses (IndexError).
pub fn index(p: *Parser, li: usize, depth: usize) Invalid!usize {
    try p.need("[");
    const i = try p.expr(depth + 1);
    try p.need("]");
    if (p.check_only) return i;
    // which element is a shape: never from a circuit choice
    if (p.boolean(i) or p.nodes[i].op == .divmod or !fixed(p, i)) return error.Unsupported;
    const l = try plain(p, li);
    const v = try p.prove(i, i);
    const at = if (v < 0) v + @as(i512, @intCast(l.len)) else v;
    if (at < 0 or at >= l.len) return error.Unsupported;
    return p.items[l.start + @as(usize, @intCast(at))];
}

// ---- text scanning at bracket depth 0, outside quotes ----

/// The index of the bracket closing the one at `open`.
pub fn closing(text: []const u8, open: usize) ?usize {
    var depth: usize = 0;
    var quote: u8 = 0;
    var i = open;
    while (i < text.len) : (i += 1) {
        const c = text[i];
        if (quote != 0) {
            if (c == '\\') i += 1 else if (c == quote) quote = 0;
            continue;
        }
        switch (c) {
            '\'', '"' => quote = c,
            '(', '[', '{' => depth += 1,
            ')', ']', '}' => {
                depth -= 1;
                if (depth == 0) return i;
            },
            else => {},
        }
    }
    return null;
}

/// The first top-level comma.
fn separator(text: []const u8) ?usize {
    return @import("runner_frontend_integer_statements.zig").separator(text, ',');
}

/// The first whole word `w` at depth 0 from `from`.
pub fn top(text: []const u8, w: []const u8, from: usize) ?usize {
    var depth: usize = 0;
    var quote: u8 = 0;
    var i = from;
    while (i < text.len) : (i += 1) {
        const c = text[i];
        if (quote != 0) {
            if (c == '\\') i += 1 else if (c == quote) quote = 0;
            continue;
        }
        switch (c) {
            '\'', '"' => quote = c,
            '(', '[', '{' => depth += 1,
            ')', ']', '}' => depth -|= 1,
            else => if (depth == 0 and std.mem.startsWith(u8, text[i..], w)) {
                const before_ok = i == 0 or !(std.ascii.isAlphanumeric(text[i - 1]) or text[i - 1] == '_');
                const end = i + w.len;
                const after_ok = end == text.len or !(std.ascii.isAlphanumeric(text[end]) or text[end] == '_');
                if (before_ok and after_ok) return i;
            },
        }
    }
    return null;
}
