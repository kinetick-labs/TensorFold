//! <prefix>_BACKEND=graph: the frontend's checked integer script (its DAG and formatted outputs) as packets.
const std = @import("std");
const graph = @import("runner_graph.zig");
const route = @import("runner_graph_route.zig");
const prog = @import("runner_graph_program.zig");
const packet = @import("runner_graph_packet.zig");
const ast = @import("runner_frontend_integer_ast.zig");
const integer = @import("runner_frontend_integer.zig");
const Allocator = std.mem.Allocator;

/// Pieces a sequence may hold (a printed tuple is five: its text and its two values).
const max_pieces = 40;
/// Calc input a piece may expand to: the DAG's shared nodes repeat in each print that uses them.
const max_expr = 8192;

/// A whole Python script the frontend checks (integer, exact Fraction, fixed Decimal) as graph bytes; null: other.
pub fn translate(a: Allocator, bank: *const graph.Bank, code: []const u8) Allocator.Error!?[]u8 {
    if (std.mem.indexOfScalar(u8, code, 0) != null) return null;
    const lowerings = .{ @import("runner_frontend_integer_python.zig"), @import("runner_frontend_decimal.zig"), @import("runner_frontend_fraction.zig") };
    inline for (lowerings) |l| if (try l.lowerWithCost(a, code, 0)) |lowered| {
        a.free(lowered.source);
        return lower(a, bank, &(lowered.table orelse return null));
    };
    return null;
}

/// The outputs in order, each a piece (`I` integer, `B` bool, `F` integral float, `1`..`6` fixed scale, `L` text).
pub fn lower(a: Allocator, bank: *const graph.Bank, t: *const integer.Table) Allocator.Error!?[]u8 {
    var arena_state = std.heap.ArenaAllocator.init(a);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var e: Emit = .{ .arena = arena, .bank = bank, .t = t };
    for (t.roots[0..t.outputs], t.endings[0..t.outputs], t.styles[0..t.outputs], t.scales[0..t.outputs], t.brackets[0..t.outputs]) |root, end, style, scale, bracket| {
        const closes = bracket & 2 != 0;
        // a printed list: `[` J, each value J with `, ` J between, `]` with the print's own separator
        const outer: u8 = if (end == ' ') 'S' else if (end == '\n') 'N' else if (end == ',' and !closes) 'J' else return null;
        const sep: u8 = if (bracket != 0 or end == ',') 'J' else outer;
        if (root >= t.count or bracket > 3) return null;
        const n = t.nodes[root];
        if (bracket != 0 or end == ',') {
            if (n.op == .divmod or style != 0) return null;
            if (bracket & 1 != 0) try e.text('J', "[");
            if (!try e.value(if (boolean(n.op)) 'B' else 'I', 'J', try e.render(root))) return null;
            if (closes) try e.text(outer, "]") else try e.text('J', ", ");
            continue;
        }
        if (style != 0 and (n.op == .divmod or boolean(n.op))) return null;
        if (style > 2 or scale > 6) return null;
        if (n.op == .divmod) {
            // Python prints the pair as a tuple: (q, r)
            try e.text('J', "(");
            if (!try e.value('I', 'J', try e.expr(n.l, n.r, "//"))) return null;
            try e.text('J', ", ");
            if (!try e.value('I', 'J', try e.expr(n.l, n.r, "%"))) return null;
            try e.text(sep, ")");
            continue;
        }
        const kind: u8 = if (style == 2) (if (scale == 0) 'I' else '0' + scale) else if (style == 1) 'F' else if (boolean(n.op)) 'B' else 'I';
        if (kind == 'I' or kind == 'B') if (try e.algorithm(root)) |p| {
            try e.piece(kind, sep, p);
            continue;
        };
        if (!try e.value(kind, sep, try e.render(root))) return null;
    }
    if (e.n == 1 and e.lone) return try a.dupe(u8, e.last);
    return try std.mem.concat(a, u8, &.{ prog.seq_magic, e.out.items });
}

fn boolean(op: ast.Op) bool {
    return switch (op) {
        .eq, .ne, .lt, .le, .gt, .ge => true,
        else => false,
    };
}

const Emit = struct {
    arena: Allocator,
    bank: *const graph.Bank,
    t: *const integer.Table,
    out: std.ArrayList(u8) = .empty,
    n: usize = 0,
    lone: bool = false, // the only piece so far is one integer ending its line
    last: []const u8 = "",

    fn piece(e: *Emit, kind: u8, sep: u8, body: []const u8) Allocator.Error!void {
        try e.out.print(e.arena, "{c}{c} {s}\n", .{ kind, sep, body });
        e.lone = e.n == 0 and kind == 'I' and sep == 'N';
        e.last = body;
        e.n += 1;
    }

    fn text(e: *Emit, sep: u8, s: []const u8) Allocator.Error!void {
        try e.piece('L', sep, s);
    }

    /// A piece computing `src` on the bank; false when it does not fit a plan (or the sequence is full).
    fn value(e: *Emit, kind: u8, sep: u8, src: ?[]const u8) Allocator.Error!bool {
        const s = src orelse return false;
        if (e.n >= max_pieces) return false;
        try e.piece(kind, sep, try route.lower(e.arena, e.bank, s) orelse return false);
        return true;
    }

    /// One of T1's algorithm plans when the root calls it on literals: gcd, isqrt, factorial, pow(a, b, m).
    fn algorithm(e: *Emit, root: usize) Allocator.Error!?[]const u8 {
        const n = e.t.nodes[root];
        const algo: packet.Algorithm, const args: []const usize = switch (n.op) {
            .gcd => .{ .gcd, &.{ n.l, n.r } },
            .isqrt => .{ .isqrt, &.{n.l} },
            .factorial => .{ .factorial, &.{n.l} },
            .powmod => .{ .powmod, &.{ n.l, n.r, n.k } },
            else => return null,
        };
        var values: [3][]const u8 = undefined;
        for (args, 0..) |arg, j| {
            var i = arg;
            // gcd ignores signs: gcd(abs(x), y) is gcd(x, y), and abs(x) is the frontend's gcd(x, x)
            if (algo == .gcd and i < e.t.count and e.t.nodes[i].op == .gcd and e.t.nodes[i].l == e.t.nodes[i].r) i = e.t.nodes[i].l;
            const v = e.literal(i) orelse return null;
            // plans take magnitudes: gcd is sign-blind; the others leave a negative operand to the frontend
            if (v < 0 and algo != .gcd) return null;
            values[j] = try std.fmt.allocPrint(e.arena, "{d}", .{@abs(v)});
        }
        const bytes = graph.Builder.integer(e.arena, algo, values[0..args.len]) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return null,
        };
        // only a plan this bank prepared: a missing one would abstain at run time, after the hold began
        const at = (std.mem.indexOf(u8, bytes, "\"plan\":\"") orelse return null) + "\"plan\":\"".len;
        const end = std.mem.indexOfScalarPos(u8, bytes, at, '"') orelse return null;
        return if (e.bank.plan(bytes[at..end]) != null) bytes else null;
    }

    /// A literal operand, or the negation of one (`-12` is the literal's sign, not a computation).
    fn literal(e: *Emit, i: usize) ?i512 {
        if (i >= e.t.count) return null;
        const n = e.t.nodes[i];
        if (n.op == .literal) return n.value;
        if (n.op == .neg and n.l < e.t.count and e.t.nodes[n.l].op == .literal) return -e.t.nodes[n.l].value;
        return null;
    }

    fn expr(e: *Emit, l: usize, r: usize, op: []const u8) Allocator.Error!?[]const u8 {
        const x = try e.render(l) orelse return null;
        const y = try e.render(r) orelse return null;
        return try e.join(&.{ "(", x, op, y, ")" });
    }

    fn join(e: *Emit, parts: []const []const u8) Allocator.Error!?[]const u8 {
        const s = try std.mem.concat(e.arena, u8, parts);
        return if (s.len > max_expr) null else s;
    }

    /// Node `i` as calc input over the script's literals; null for what calc cannot say (a call, a computed exponent).
    fn render(e: *Emit, i: usize) Allocator.Error!?[]const u8 {
        if (i >= e.t.count) return null;
        const n = e.t.nodes[i];
        return switch (n.op) {
            // only a literal's value is read: every other node's value is the frontend's proof
            .literal => try std.fmt.allocPrint(e.arena, "({d})", .{n.value}),
            .neg => try e.join(&.{ "(-", try e.render(n.l) orelse return null, ")" }),
            .add => e.expr(n.l, n.r, "+"),
            .sub => e.expr(n.l, n.r, "-"),
            .mul => e.expr(n.l, n.r, "*"),
            .floor => e.expr(n.l, n.r, "//"),
            .mod => e.expr(n.l, n.r, "%"),
            .pow => {
                if (n.r >= e.t.count or e.t.nodes[n.r].op != .literal or e.t.nodes[n.r].value < 0) return null;
                return try e.join(&.{ "(", try e.render(n.l) orelse return null, ")**", try std.fmt.allocPrint(e.arena, "{d}", .{e.t.nodes[n.r].value}) });
            },
            // abs(x) is gcd(x, x): |x| = x * sign(x)
            .gcd => if (n.l == n.r) e.sign(n.l, n.l, .abs) else null,
            .eq => e.sign(n.l, n.r, .eq),
            .ne => e.sign(n.l, n.r, .ne),
            .lt => e.sign(n.l, n.r, .lt),
            .le => e.sign(n.l, n.r, .le),
            .gt => e.sign(n.l, n.r, .gt),
            .ge => e.sign(n.l, n.r, .ge),
            .powmod, .digits, .divmod, .isqrt, .factorial, .range_product, .digit_sum => null,
        };
    }

    /// l vs r as 0/1 (or |l|): with d = l - r and M = 10^D above |d|, (d+M-1)//M is 1 exactly when d > 0.
    fn sign(e: *Emit, l: usize, r: usize, how: enum { eq, ne, lt, le, gt, ge, abs }) Allocator.Error!?[]const u8 {
        const x = try e.render(l) orelse return null;
        const d = if (how == .abs) x else try e.join(&.{ "(", x, "-", try e.render(r) orelse return null, ")" }) orelse return null;
        const sh = try route.shape(e.arena, d) orelse return null;
        const digits = sh.nodes[sh.root].digits;
        if (digits > 123) return null;
        const m = try e.arena.alloc(u8, digits + 1);
        m[0] = '1';
        @memset(m[1..], '0');
        if (how == .eq or how == .ne) {
            // (d+M)%M is 0 only at d = 0 (operands non-negative): d read once, not twice as in pos + neg
            const nz = try e.join(&.{ "(((", d, "+", m, ")%", m, "+", m, "-1)//", m, ")" }) orelse return null;
            return if (how == .ne) nz else try e.join(&.{ "(1-", nz, ")" });
        }
        const pos = try e.join(&.{ "((", d, "+", m, "-1)//", m, ")" }) orelse return null;
        const neg = try e.join(&.{ "((-", d, "+", m, "-1)//", m, ")" }) orelse return null;
        return switch (how) {
            .gt => pos,
            .lt => neg,
            .ge => try e.join(&.{ "(1-", neg, ")" }),
            .le => try e.join(&.{ "(1-", pos, ")" }),
            .eq, .ne => unreachable,
            .abs => try e.join(&.{ "(", d, "*(", pos, "-", neg, "))" }),
        };
    }
};
