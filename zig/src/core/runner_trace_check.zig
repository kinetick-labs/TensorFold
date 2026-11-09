//! The runner's `trace` form, opt-in per request (runner_trace_check): residues mod 9 and 1000 after the owned line.
const std = @import("std");
const Allocator = std.mem.Allocator;

/// Casting out nines covers every digit; mod 1000 the last three.
const moduli = [_]u64{ 9, 1000 };

/// A check the model can confirm quickly: at most this many literals, and powers at most this.
const max_literals = 4;
const max_power = 4;

/// Appends ` check: <residues> ≡ <value's> (mod 9), … (mod 1000)` if the line takes one; false: residues contradict q.
pub fn append(a: Allocator, line: *std.ArrayList(u8), display: []const u8, q: []const u8, r: ?[]const u8) Allocator.Error!bool {
    const start = line.items.len;
    try line.appendSlice(a, " check: ");
    for (moduli, 0..) |m, k| {
        if (k > 0) try line.appendSlice(a, ", ");
        const fits = if (r) |rem| try divmodPart(a, line, display, q, rem, m) else try plainPart(a, line, display, q, m);
        switch (fits) {
            .yes => try line.print(a, " (mod {d})", .{m}),
            .none => {
                line.shrinkRetainingCapacity(start);
                return true;
            },
            .contradicted => {
                line.shrinkRetainingCapacity(start);
                return false;
            },
        }
    }
    return true;
}

const Fits = enum { yes, none, contradicted };

/// `L ≡ v (mod m)`: the display with each literal (not an exponent) as its residue, and the value's residue.
fn plainPart(a: Allocator, line: *std.ArrayList(u8), display: []const u8, v: []const u8, m: u64) Allocator.Error!Fits {
    if (!checkable(display) or !digits(v)) return .none;
    const dm = eval(display, m) orelse return .none;
    if (dm != residue(v, m)) return .contradicted;
    var prev: u8 = 0;
    var i: usize = 0;
    while (i < display.len) {
        const c = display[i];
        if (!std.ascii.isDigit(c)) {
            try line.append(a, c);
            if (c != ' ') prev = c;
            i += 1;
            continue;
        }
        var j = i;
        while (j < display.len and std.ascii.isDigit(display[j])) j += 1;
        const lit = display[i..j];
        if (prev == '^') try line.appendSlice(a, lit) else try shown(a, line, lit, residue(lit, m), m);
        prev = '0';
        i = j;
    }
    try line.appendSlice(a, " ≡ ");
    try shown(a, line, v, dm, m);
    return .yes;
}

/// `q*B + r ≡ A (mod m)` for `A ÷ B = q remainder r`, each as its residue.
fn divmodPart(a: Allocator, line: *std.ArrayList(u8), display: []const u8, q: []const u8, r: []const u8, m: u64) Allocator.Error!Fits {
    const at = std.mem.indexOf(u8, display, " ÷ ") orelse return .none;
    const x = display[0..at];
    const y = display[at + " ÷ ".len ..];
    if (!checkable(x) or !checkable(y) or !digits(q) or !digits(r)) return .none;
    const xm = eval(x, m) orelse return .none;
    const ym = eval(y, m) orelse return .none;
    const qm = residue(q, m);
    const rm = residue(r, m);
    if ((qm * ym + rm) % m != xm) return .contradicted;
    try shown(a, line, q, qm, m);
    try line.append(a, '*');
    try operand(a, line, y, ym, m);
    try line.appendSlice(a, " + ");
    try shown(a, line, r, rm, m);
    try line.appendSlice(a, " ≡ ");
    try operand(a, line, x, xm, m);
    return .yes;
}

/// A residue as written: mod 1000 of a number with three or more digits keeps its last three (`012`).
fn shown(a: Allocator, line: *std.ArrayList(u8), number: []const u8, v: u64, m: u64) Allocator.Error!void {
    if (m == 1000 and number.len >= 3) try line.print(a, "{d:0>3}", .{v}) else try line.print(a, "{d}", .{v});
}

/// An operand of ÷: a lone literal (bracketed or not) shows as a literal's residue, an expression as a plain one.
fn operand(a: Allocator, line: *std.ArrayList(u8), e: []const u8, v: u64, m: u64) Allocator.Error!void {
    const lit = std.mem.trim(u8, e, "()");
    if (digits(lit) and lit.len + 2 >= e.len) return shown(a, line, lit, v, m);
    try line.print(a, "{d}", .{v});
}

/// Decimal literals with + - * ^ and brackets, at most `max_literals`, powers at most `max_power`, no unary minus.
fn checkable(d: []const u8) bool {
    var literals: usize = 0;
    var prev: u8 = 0; // the last non-space byte, '0' for a whole literal
    var i: usize = 0;
    while (i < d.len) {
        const c = d[i];
        if (std.ascii.isDigit(c)) {
            var j = i;
            while (j < d.len and std.ascii.isDigit(d[j])) j += 1;
            if (prev == '^') {
                const k = std.fmt.parseInt(u32, d[i..j], 10) catch return false;
                if (k > max_power) return false;
            } else literals += 1;
            prev = '0';
            i = j;
            continue;
        }
        if (std.mem.indexOfScalar(u8, " +-*^()", c) == null) return false;
        if (c == '-' and prev != '0' and prev != ')') return false;
        if (c != ' ') prev = c;
        i += 1;
    }
    return literals >= 1 and literals <= max_literals;
}

fn digits(s: []const u8) bool {
    if (s.len == 0) return false;
    for (s) |c| if (!std.ascii.isDigit(c)) return false;
    return true;
}

fn residue(number: []const u8, m: u64) u64 {
    var v: u64 = 0;
    for (number) |c| v = (v * 10 + (c - '0')) % m;
    return v;
}

/// `d` mod m with Python's precedence (literals reduced, exponents as written); null if malformed.
fn eval(d: []const u8, m: u64) ?u64 {
    var p: Eval = .{ .s = d, .m = m };
    const v = p.sum() orelse return null;
    return if (p.peek() == 0) v else null;
}

const Eval = struct {
    s: []const u8,
    m: u64,
    i: usize = 0,

    /// The next non-space byte, 0 at the end.
    fn peek(p: *Eval) u8 {
        while (p.i < p.s.len and p.s[p.i] == ' ') p.i += 1;
        return if (p.i < p.s.len) p.s[p.i] else 0;
    }

    fn sum(p: *Eval) ?u64 {
        var v = p.term() orelse return null;
        while (p.peek() == '+' or p.peek() == '-') {
            const plus = p.s[p.i] == '+';
            p.i += 1;
            const r = p.term() orelse return null;
            v = if (plus) (v + r) % p.m else (v + p.m - r) % p.m;
        }
        return v;
    }

    fn term(p: *Eval) ?u64 {
        var v = p.power() orelse return null;
        while (p.peek() == '*') {
            p.i += 1;
            v = v * (p.power() orelse return null) % p.m;
        }
        return v;
    }

    fn power(p: *Eval) ?u64 {
        const b = p.atom() orelse return null;
        if (p.peek() != '^') return b;
        p.i += 1;
        _ = p.peek();
        const k0 = p.i;
        while (p.i < p.s.len and std.ascii.isDigit(p.s[p.i])) p.i += 1;
        const k = std.fmt.parseInt(u32, p.s[k0..p.i], 10) catch return null;
        if (k > max_power) return null;
        var v: u64 = 1 % p.m;
        for (0..k) |_| v = v * b % p.m;
        return v;
    }

    fn atom(p: *Eval) ?u64 {
        if (p.peek() == '(') {
            p.i += 1;
            const v = p.sum() orelse return null;
            if (p.peek() != ')') return null;
            p.i += 1;
            return v;
        }
        const d0 = p.i;
        while (p.i < p.s.len and std.ascii.isDigit(p.s[p.i])) p.i += 1;
        return if (p.i > d0) residue(p.s[d0..p.i], p.m) else null;
    }
};
