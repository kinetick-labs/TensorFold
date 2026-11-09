//! <prefix>_BACKEND=graph: a straight-line integer cell (assignments, divmod, prints) as graph packets.
const std = @import("std");
const graph = @import("runner_graph.zig");
const route = @import("runner_graph_route.zig");
const packet = @import("runner_graph_packet.zig");
const api = @import("runner_api.zig");
const Allocator = std.mem.Allocator;

/// A straight-line cell's prints run as packets; each piece is `<I|B><N|S> <packet>` on its own line.
pub const seq_magic = "TFGS1\n";
/// As TFGS1, and a piece may be a template (`@` body) over earlier pieces' printed values, lowered when they finish.
pub const seq_magic_v2 = "TFGS2\n";

/// Straight-line integer Python (assignments, divmod, `;`, prints) as graph bytes over literals; null: other shapes.
pub fn program(a: Allocator, bank: *const graph.Bank, code: []const u8) Allocator.Error!?[]u8 {
    var arena_state = std.heap.ArenaAllocator.init(a);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    if (code.len > 4096) return null;
    // `import math` makes `math.f` callable, `from math import f` the bare `f`
    var math_names: Calls = .{};
    const text = try arena.dupe(u8, code);
    _ = std.mem.replace(u8, code, "math.", "math_", text);
    if (std.mem.indexOfAny(u8, text, "'\"\\[]{}.:") != null) return null;
    var defs: std.StringHashMapUnmanaged([]const u8) = .empty;
    // names whose value an earlier 'I' piece printed: later pieces read it there instead of recomputing it
    var printed: std.StringHashMapUnmanaged(usize) = .empty;
    var templated = false;
    var pieces: std.ArrayList(u8) = .empty;
    var n_pieces: usize = 0;
    var lone: ?[]const u8 = null;
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |raw| {
        const line = raw[0 .. std.mem.indexOfScalar(u8, raw, '#') orelse raw.len];
        if (line.len > 0 and (line[0] == ' ' or line[0] == '\t') and std.mem.trim(u8, line, " \t\r").len > 0) return null;
        var stmts = std.mem.splitScalar(u8, line, ';');
        while (stmts.next()) |st| {
            const t = std.mem.trim(u8, st, " \t\r");
            if (t.len == 0) continue;
            if (std.mem.eql(u8, t, "import math")) {
                math_names.module = true;
                continue;
            }
            if (std.mem.startsWith(u8, t, "from math import ")) {
                var it = std.mem.splitScalar(u8, t["from math import ".len..], ',');
                while (it.next()) |f| if (!math_names.allow(std.mem.trim(u8, f, " \t"))) return null;
                continue;
            }
            if (std.mem.startsWith(u8, t, "print(") and t[t.len - 1] == ')') {
                const args = try splitArgs(arena, t[6 .. t.len - 1]) orelse return null;
                if (args.len == 0) return null;
                for (args, 0..) |arg, i| {
                    const last = i + 1 == args.len;
                    var one = try call(arena, &defs, math_names, arg) orelse try piece(arena, bank, &defs, arg) orelse return null;
                    if (one.from_piece) if (try template(arena, &defs, &printed, arg)) |tmpl| {
                        one.packet = tmpl;
                        templated = true;
                    };
                    if (n_pieces >= 16) return null;
                    try pieces.print(arena, "{c}{c} {s}\n", .{ one.kind, @as(u8, if (last) 'N' else 'S'), one.packet });
                    if (one.kind == 'I' and name(arg)) if (defs.get(arg)) |def| if (costly(def)) try printed.put(arena, arg, n_pieces);
                    lone = if (n_pieces == 0 and one.kind == 'I' and last) one.packet else null;
                    n_pieces += 1;
                }
                continue;
            }
            const eq = std.mem.indexOfScalar(u8, t, '=') orelse return null;
            if (eq + 1 < t.len and t[eq + 1] == '=') return null;
            const lhs = std.mem.trim(u8, t[0..eq], " \t()");
            const rhs = std.mem.trim(u8, t[eq + 1 ..], " \t");
            if (std.mem.indexOfScalar(u8, lhs, ',')) |comma| {
                // q, r = divmod(x, y): Python's floor quotient and remainder
                const q = std.mem.trim(u8, lhs[0..comma], " \t");
                const rem = std.mem.trim(u8, lhs[comma + 1 ..], " \t");
                if (!name(q) or !name(rem) or !std.mem.startsWith(u8, rhs, "divmod(") or rhs[rhs.len - 1] != ')') return null;
                const args = try splitArgs(arena, rhs[7 .. rhs.len - 1]) orelse return null;
                if (args.len != 2) return null;
                const x = try expand(arena, &defs, args[0]) orelse return null;
                const y = try expand(arena, &defs, args[1]) orelse return null;
                try defs.put(arena, q, try std.mem.concat(arena, u8, &.{ "(", x, "//", y, ")" }));
                try defs.put(arena, rem, try std.mem.concat(arena, u8, &.{ "(", x, "%", y, ")" }));
                _ = printed.remove(q);
                _ = printed.remove(rem);
                continue;
            }
            if (!name(lhs)) return null;
            const v = try expand(arena, &defs, rhs) orelse return null;
            // a plain literal stays bare, so it can still be an exponent
            const bare = v.len > 0 and for (v) |c| {
                if (!std.ascii.isDigit(c)) break false;
            } else true;
            try defs.put(arena, lhs, if (bare) v else try std.mem.concat(arena, u8, &.{ "(", v, ")" }));
            _ = printed.remove(lhs);
        }
    }
    if (n_pieces == 0) return null;
    if (n_pieces == 1) if (lone) |p| return try a.dupe(u8, p);
    return try std.mem.concat(a, u8, &.{ if (templated) seq_magic_v2 else seq_magic, pieces.items });
}

const Piece = struct { kind: u8, packet: []const u8, from_piece: bool = false };

/// The math functions a cell may call: `math.f` after `import math`, a bare `f` after `from math import f`.
const Calls = struct {
    module: bool = false,
    gcd: bool = false,
    isqrt: bool = false,
    factorial: bool = false,

    fn allow(c: *Calls, f: []const u8) bool {
        inline for (.{ "gcd", "isqrt", "factorial" }) |n| if (std.mem.eql(u8, f, n)) {
            @field(c, n) = true;
            return true;
        };
        return false;
    }
};

/// A print of pow(a, b, m) or math gcd, isqrt, factorial over integer literals (T1's algorithm plans); null: not one.
fn call(arena: Allocator, defs: anytype, calls: Calls, arg: []const u8) Allocator.Error!?Piece {
    const open = std.mem.indexOfScalar(u8, arg, '(') orelse return null;
    if (arg[arg.len - 1] != ')') return null;
    const f = std.mem.trim(u8, arg[0..open], " \t");
    const module = std.mem.startsWith(u8, f, "math_");
    const bare = if (module) f[5..] else f;
    const algo: packet.Algorithm, const n: usize, const ok: bool = if (std.mem.eql(u8, f, "pow"))
        .{ .powmod, 3, true }
    else if (std.mem.eql(u8, bare, "gcd"))
        .{ .gcd, 2, if (module) calls.module else calls.gcd }
    else if (std.mem.eql(u8, bare, "isqrt"))
        .{ .isqrt, 1, if (module) calls.module else calls.isqrt }
    else if (std.mem.eql(u8, bare, "factorial"))
        .{ .factorial, 1, if (module) calls.module else calls.factorial }
    else
        return null;
    if (!ok) return null;
    const args = try splitArgs(arena, arg[open + 1 .. arg.len - 1]) orelse return null;
    // pow(a, b) with two arguments is an expression, not powmod
    if (args.len != n) return null;
    const values = try arena.alloc([]const u8, n);
    for (args, values) |x, *v| {
        const e = try expand(arena, defs, x) orelse return null;
        const lit = std.mem.trim(u8, e, "()");
        if (lit.len == 0 or lit.len + 2 * @as(usize, @intFromBool(lit.len != e.len)) != e.len) return null;
        for (lit, 0..) |c, i| if (!std.ascii.isDigit(c) and !(i == 0 and c == '-' and lit.len > 1)) return null;
        v.* = lit;
    }
    return .{ .kind = 'I', .packet = graph.Builder.integer(arena, algo, values) catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return null,
    } };
}

/// One print argument: an integer expression ('I'), or one `==`/`!=` printed as True/False ('B').
fn piece(arena: Allocator, bank: *const graph.Bank, defs: anytype, arg: []const u8) Allocator.Error!?Piece {
    const cmp = std.mem.indexOf(u8, arg, "==") orelse std.mem.indexOf(u8, arg, "!=");
    if (cmp == null) {
        const e = try expand(arena, defs, arg) orelse return null;
        return .{ .kind = 'I', .packet = try route.lower(arena, bank, e) orelse return null, .from_piece = true };
    }
    const at = cmp.?;
    const l = try expand(arena, defs, arg[0..at]) orelse return null;
    const r = try expand(arena, defs, arg[at + 2 ..]) orelse return null;
    return .{ .kind = 'B', .packet = try compare(arena, bank, l, r, arg[at] == '=') orelse return null, .from_piece = true };
}

/// `l == r` (or `l != r`) over calc input as one packet printing 1 or 0; null when the bank cannot say it.
pub fn compare(arena: Allocator, bank: *const graph.Bank, l: []const u8, r: []const u8, eq: bool) Allocator.Error!?[]u8 {
    if (std.mem.indexOfAny(u8, l, "=!<>") != null or std.mem.indexOfAny(u8, r, "=!<>") != null) return null;
    // d = l - r; with M = 10^D above |d|, (d+M)%M is 0 only at d = 0, so ((d+M)%M+M-1)//M is d != 0 (d read once)
    const d = try std.mem.concat(arena, u8, &.{ "(", l, ")-(", r, ")" });
    const sh = try route.shape(arena, d) orelse return null;
    const digits = sh.nodes[sh.root].digits;
    if (digits > 123) return null;
    const m = try arena.alloc(u8, digits + 1);
    m[0] = '1';
    @memset(m[1..], '0');
    const nz = try std.mem.concat(arena, u8, &.{ "((", d, "+", m, ")%", m, "+", m, "-1)//", m });
    const e = if (eq) try std.mem.concat(arena, u8, &.{ "1-(", nz, ")" }) else nz;
    return route.lower(arena, bank, e);
}

/// `@I<expr>`, `@=<l>;<r>` or `@!<l>;<r>` with `$k` for piece k's printed value; null when `arg` reads none.
fn template(arena: Allocator, defs: anytype, printed: anytype, arg: []const u8) Allocator.Error!?[]const u8 {
    var any = false;
    const cmp = std.mem.indexOf(u8, arg, "==") orelse std.mem.indexOf(u8, arg, "!=");
    const parts: [2][]const u8 = if (cmp) |at| .{ arg[0..at], arg[at + 2 ..] } else .{ arg, "" };
    var out: std.ArrayList(u8) = .empty;
    try out.appendSlice(arena, if (cmp) |at| if (arg[at] == '=') "@=" else "@!" else "@I");
    for (parts, 0..) |part, side| {
        if (side == 1 and cmp == null) break;
        if (side == 1) try out.append(arena, ';');
        var i: usize = 0;
        while (i < part.len) {
            const c = part[i];
            if (std.ascii.isAlphabetic(c) or c == '_') {
                const start = i;
                while (i < part.len and (std.ascii.isAlphanumeric(part[i]) or part[i] == '_')) i += 1;
                if (printed.get(part[start..i])) |k| {
                    try out.print(arena, "${d}", .{k});
                    any = true;
                } else try out.appendSlice(arena, defs.get(part[start..i]) orelse return null);
                continue;
            }
            if (c != ' ' and c != '\t') try out.append(arena, c);
            i += 1;
        }
    }
    return if (any) out.items else null;
}

/// A template piece's packet with each `$k` bound to `values[k]` (decimal text); null for a bad template or index.
pub fn bind(arena: Allocator, bank: *const graph.Bank, body: []const u8, values: []const ?[]const u8) Allocator.Error!?[]u8 {
    if (body.len < 2 or body[0] != '@') return null;
    var out: std.ArrayList(u8) = .empty;
    var i: usize = 2;
    while (i < body.len) {
        if (body[i] != '$') {
            try out.append(arena, body[i]);
            i += 1;
            continue;
        }
        const start = i + 1;
        i = start;
        while (i < body.len and std.ascii.isDigit(body[i])) i += 1;
        const k = std.fmt.parseInt(usize, body[start..i], 10) catch return null;
        if (k >= values.len) return null;
        const v = values[k] orelse return null;
        try out.print(arena, "({s})", .{v});
    }
    const text = out.items;
    return switch (body[1]) {
        'I' => route.lower(arena, bank, text),
        '=', '!' => {
            const semi = std.mem.indexOfScalar(u8, text, ';') orelse return null;
            return compare(arena, bank, text[0..semi], text[semi + 1 ..], body[1] == '=');
        },
        else => null,
    };
}

/// `src` with every name replaced by its definition and blanks dropped (calc input); null for an undefined name.
fn expand(arena: Allocator, defs: anytype, src: []const u8) Allocator.Error!?[]const u8 {
    var out: std.ArrayList(u8) = .empty;
    var i: usize = 0;
    while (i < src.len) {
        const c = src[i];
        if (std.ascii.isAlphabetic(c) or c == '_') {
            const start = i;
            while (i < src.len and (std.ascii.isAlphanumeric(src[i]) or src[i] == '_')) i += 1;
            try out.appendSlice(arena, defs.get(src[start..i]) orelse return null);
            continue;
        }
        if (c != ' ' and c != '\t') try out.append(arena, c);
        i += 1;
    }
    return if (out.items.len == 0) null else out.items;
}

/// Top-level comma-separated arguments; null for unbalanced parentheses.
fn splitArgs(arena: Allocator, src: []const u8) Allocator.Error!?[]const []const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    var depth: usize = 0;
    var start: usize = 0;
    for (src, 0..) |c, i| switch (c) {
        '(' => depth += 1,
        ')' => depth = std.math.sub(usize, depth, 1) catch return null,
        ',' => if (depth == 0) {
            try out.append(arena, std.mem.trim(u8, src[start..i], " \t"));
            start = i + 1;
        },
        else => {},
    };
    if (depth != 0) return null;
    const last = std.mem.trim(u8, src[start..], " \t");
    if (last.len > 0) try out.append(arena, last) else if (out.items.len > 0) return null;
    return out.items;
}

/// A definition worth reading from an earlier piece rather than recomputing: it divides or raises to a power.
fn costly(def: []const u8) bool {
    return std.mem.indexOf(u8, def, "//") != null or std.mem.indexOfScalar(u8, def, '%') != null or std.mem.indexOf(u8, def, "**") != null;
}

fn name(s: []const u8) bool {
    if (s.len == 0 or !(std.ascii.isAlphabetic(s[0]) or s[0] == '_')) return false;
    for (s) |c| if (!(std.ascii.isAlphanumeric(c) or c == '_')) return false;
    for ([_][]const u8{ "print", "divmod", "True", "False", "None" }) |k| if (std.mem.eql(u8, s, k)) return false;
    return true;
}

/// Whether earlier cells rebind `print` or `divmod`, which this cell's meaning depends on.
pub fn shadowed(prior: []const []const u8) bool {
    for (prior) |code| for ([_][]const u8{ "print", "divmod" }) |k| {
        var at: usize = 0;
        while (std.mem.indexOfPos(u8, code, at, k)) |i| : (at = i + 1) {
            const rest = std.mem.trimStart(u8, code[i + k.len ..], " \t");
            if (rest.len > 0 and (rest[0] == '=' and (rest.len == 1 or rest[1] != '=')) or std.mem.startsWith(u8, rest, ",")) return true;
            if (i >= 4 and std.mem.eql(u8, code[i - 4 .. i], "def ")) return true;
        }
    };
    return false;
}

/// An inline or REPL expression the frontend refused, onto 128-digit plans; `%` and REPL `^` (XOR) refuse.
pub fn direct(a: Allocator, bank: *const graph.Bank, cell: api.Cell, why: api.Abstain) Allocator.Error!?[]u8 {
    if ((why != .unsupported and why != .too_big) or cell.form == .fence or !std.mem.eql(u8, cell.lang, "python")) return null;
    var src = std.mem.trim(u8, cell.code, " \t\n");
    if (cell.form != .inline_expr and std.mem.startsWith(u8, src, ">>>")) src = std.mem.trim(u8, src[3..], " \t");
    if (src.len == 0 or src.len > 4096 or shadowed(cell.prior)) return null;
    if (cell.prior.len == 0)
        if (try @import("runner_frontend_inline_integer.zig").lower(a, bank, src, cell.form == .inline_expr)) |bytes|
            return bytes;
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(a);
    var i: usize = 0;
    while (i < src.len) {
        const rest = src[i..];
        if (cell.form == .inline_expr and std.mem.startsWith(u8, rest, "×")) {
            try out.append(a, '*');
            i += "×".len;
        } else if (cell.form == .inline_expr and std.mem.startsWith(u8, rest, "−")) {
            try out.append(a, '-');
            i += "−".len;
        } else {
            switch (rest[0]) {
                ' ', '\t' => {},
                '%' => return null,
                '^' => if (cell.form == .inline_expr) try out.appendSlice(a, "**") else return null,
                else => try out.append(a, rest[0]),
            }
            i += 1;
        }
    }
    return route.lower(a, bank, out.items);
}

fn planNames(a: Allocator, src: []const u8) !?[]u8 {
    var arena_state = std.heap.ArenaAllocator.init(a);
    defer arena_state.deinit();
    const sh = try route.shape(arena_state.allocator(), src) orelse return null;
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    for (sh.nodes, sh.live, sh.plans) |n, on, plan_name| if (on and n.kind != .literal) {
        if (out.items.len != 0) try out.append(a, ' ');
        try out.appendSlice(a, plan_name);
    };
    return try out.toOwnedSlice(a);
}

test "runner graph route: pure-integer calc programs become bank plans sized from their syntax" {
    const a = std.testing.allocator;
    for ([_][2][]const u8{
        .{ "2**100-3**50", "pow-100-62 pow-50-62 sub-32" },
        .{ "47*83", "mul-4" },
        .{ "-7//2", "div-1" },
        .{ "1071%462", "mod-4" },
        .{ "-(2+3)", "add-2 sub-4" },
        .{ "(-3)**5", "pow-5-62" },
        .{ "10**61", "pow-61-62" },
        .{ "(2+3)*4-1", "add-2 mul-4 sub-4" },
        .{ "42", "" },
        .{ "10**62", "pow-62-128" },
        .{ "1000000000000000000000000000000000000000*100000000000000000000000000000", "mul-128" },
        // big-number-020: (Y+B)**2 - Y**2 as B*(Y+(Y+B)), so no value passes 128 digits
        .{ "(10**80+123456789)**2-10**160", "pow-80-128 add-128 add-128 mul-128" },
    }) |c| {
        const got = (try planNames(a, c[0])) orelse {
            std.debug.print("not routed: {s}\n", .{c[0]});
            return error.TestExpectedRoute;
        };
        defer a.free(got);
        try std.testing.expectEqualStrings(c[1], got);
    }
}

test "runner graph route: decimals, true division, computed exponents or divisors and values past 256 digits stay on calc" {
    const a = std.testing.allocator;
    for ([_][]const u8{
        "1/2",
        "1.5*2",
        "2**-1",
        "2**3**2",
        "2**(3)",
        "7//0",
        "7%(1+1)",
        "10**256", // past T1's 256-digit width (b047d48c8); a bank without it refuses at lowering
        "(10**130+123456789)**2-10**261",
        "(10**90+123456789)**3-10**270",
        "007+1",
        "x*2",
        "",
        "(1+2",
        "2**1000",
        "3^2",
    }) |src| {
        if (try planNames(a, src)) |got| {
            defer a.free(got);
            std.debug.print("routed: {s}\n", .{src});
            return error.TestUnexpectedRoute;
        }
    }
}
