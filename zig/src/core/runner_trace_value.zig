//! The runner's `trace` form: the integer value of a checked source, to tell a restated name from a reassigned one.
const std = @import("std");
const Allocator = std.mem.Allocator;
const Int = std.math.big.int.Managed;

/// The widest value evaluated: a wider source is never called the same (the name it would keep is poisoned instead).
const max_bits = 4096;

/// `x` and `y` (sources over + - * ** % /, brackets, decimal and 0x literals) are one integer; `a` should be an arena.
pub fn same(a: Allocator, x: []const u8, y: []const u8) Allocator.Error!bool {
    const u = try value(a, x) orelse return false;
    const v = try value(a, y) orelse return false;
    return u.eql(v);
}

fn value(a: Allocator, s: []const u8) Allocator.Error!?Int {
    var p: Eval = .{ .a = a, .s = s };
    const v = p.sum() catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        error.Refused => return null,
    };
    return if (p.i == s.len) v else null;
}

/// Python's precedence: sum := term (+|- term)*; term := unary (*|%|/ unary)*; unary := -unary | atom [** digits].
const Eval = struct {
    a: Allocator,
    s: []const u8,
    i: usize = 0,

    const Error = error{ OutOfMemory, Refused };

    fn fail(e: anyerror) Error {
        return if (e == error.OutOfMemory) error.OutOfMemory else error.Refused;
    }

    fn eat(p: *Eval, op: []const u8) bool {
        if (!std.mem.startsWith(u8, p.s[p.i..], op)) return false;
        p.i += op.len;
        return true;
    }

    fn wide(v: Int) Error!Int {
        return if (v.bitCountAbs() > max_bits) error.Refused else v;
    }

    fn sum(p: *Eval) Error!Int {
        var v = try p.term();
        while (p.i < p.s.len and (p.s[p.i] == '+' or p.s[p.i] == '-')) {
            const plus = p.s[p.i] == '+';
            p.i += 1;
            const r = try p.term();
            var out = Int.init(p.a) catch |e| return fail(e);
            if (plus) out.add(&v, &r) catch |e| return fail(e) else out.sub(&v, &r) catch |e| return fail(e);
            v = try wide(out);
        }
        return v;
    }

    fn term(p: *Eval) Error!Int {
        var v = try p.unary();
        while (p.i < p.s.len and (p.s[p.i] == '%' or p.s[p.i] == '/' or (p.s[p.i] == '*' and !std.mem.startsWith(u8, p.s[p.i..], "**")))) {
            const op = p.s[p.i];
            p.i += 1;
            const r = try p.unary();
            var out = Int.init(p.a) catch |e| return fail(e);
            if (op == '*') {
                out.mul(&v, &r) catch |e| return fail(e);
            } else {
                if (r.eqlZero()) return error.Refused;
                var q = Int.init(p.a) catch |e| return fail(e);
                q.divFloor(&out, &v, &r) catch |e| return fail(e); // Python's % takes the divisor's sign
                if (op == '/') {
                    if (!out.eqlZero()) return error.Refused; // a quotient only when nothing is left over
                    out.swap(&q);
                }
            }
            v = try wide(out);
        }
        return v;
    }

    fn unary(p: *Eval) Error!Int {
        if (p.eat("-")) {
            var v = try p.unary();
            v.negate();
            return v;
        }
        const base = try p.atom();
        if (!p.eat("**")) return base;
        const k0 = p.i;
        while (p.i < p.s.len and std.ascii.isDigit(p.s[p.i])) p.i += 1;
        const k = std.fmt.parseInt(u32, p.s[k0..p.i], 10) catch return error.Refused;
        if (base.bitCountAbs() > 1 and @as(u64, base.bitCountAbs()) * k > max_bits + 64) return error.Refused;
        var out = Int.init(p.a) catch |e| return fail(e);
        out.pow(&base, k) catch |e| return fail(e);
        return wide(out);
    }

    fn atom(p: *Eval) Error!Int {
        if (p.eat("(")) {
            const v = try p.sum();
            if (!p.eat(")")) return error.Refused;
            return v;
        }
        const hex = p.eat("0x");
        const d0 = p.i;
        while (p.i < p.s.len and (if (hex) std.ascii.isHex(p.s[p.i]) else std.ascii.isDigit(p.s[p.i]))) p.i += 1;
        if (p.i == d0 or p.i - d0 > max_bits / 3) return error.Refused;
        var v = Int.init(p.a) catch |e| return fail(e);
        v.setString(if (hex) 16 else 10, p.s[d0..p.i]) catch |e| return fail(e);
        return wide(v);
    }
};

test "runner trace value: a restatement in another form is the same integer; a different value, a wide or bad one is not" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try std.testing.expect(try same(a, "12345678901", "12000000000+345678901"));
    try std.testing.expect(try same(a, "(0xFEDCBA98)*2**32+(0x76543210)", "18364758544493064720"));
    try std.testing.expect(try same(a, "-7%3", "2")); // Python: the divisor's sign
    try std.testing.expect(try same(a, "-2**2", "-4"));
    try std.testing.expect(!try same(a, "280223976814164", "18364758544493064720"));
    try std.testing.expect(!try same(a, "2**100000", "2**100000")); // too wide to evaluate: poisoned, never kept
    try std.testing.expect(!try same(a, "5%0", "5"));
    try std.testing.expect(!try same(a, "12+", "12"));
    try std.testing.expect(try same(a, "(10**30-85)/97", "10309278350515463917525773195"));
    try std.testing.expect(!try same(a, "(10**30-86)/97", "10309278350515463917525773195")); // not exact: refused
    try std.testing.expect(!try same(a, "7/0", "0"));
}
