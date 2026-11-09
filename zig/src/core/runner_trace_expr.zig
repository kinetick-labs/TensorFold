//! The runner's `trace` form: integer expression text to checked Python source, display text and a width bound.
const std = @import("std");
const text = @import("runner_trace_text.zig");
const Allocator = std.mem.Allocator;

pub const inf: u32 = 1 << 20;

pub const Kind = enum { num, name, op };
pub const Tok = struct { kind: Kind, text: []const u8 };

/// A parsed operand: source, display, digit bound, operators, literal-only; `run` the frontend's source, `pw` a power.
pub const Expr = struct { source: []u8, display: []u8, bound: u32, ops: u32, constant: bool, run: []u8, pw: ?Pow = null };

pub const Pow = struct { base: []u8, k: []const u8 };

pub const Name = struct { name: []u8, expr: ?Expr };

pub fn sat(x: u32) u32 {
    return @min(x, inf);
}

/// Expression text to a checked expression, or null (not a plain integer expression, or an unknown/poisoned name).
pub fn parse(a: Allocator, names: []const Name, s: []const u8) Allocator.Error!?Expr {
    var toks = try tokenize(a, s) orelse return null;
    toks = try expandRun(a, toks) orelse return null;
    var p: Parser = .{ .names = names, .a = a, .toks = toks };
    const e = p.sum() catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        error.Refused => return null,
    };
    if (p.i != toks.len) return null;
    return e;
}

/// Numbers (plain, strict comma groups, 0x hex), names, + - * ^ ( ) mod and `...`; null on anything else.
fn tokenize(a: Allocator, s: []const u8) Allocator.Error!?[]Tok {
    var out: std.ArrayList(Tok) = .empty;
    var i: usize = 0;
    while (i < s.len) {
        const c = s[i];
        if (text.isSpace(c)) {
            i += 1;
            continue;
        }
        if (i > 0 and (std.ascii.isAlphanumeric(s[i - 1]) or s[i - 1] == ',' or s[i - 1] == '.') and std.ascii.isAlphanumeric(c)) return null;
        if (std.ascii.isDigit(c)) {
            const n = number(s, i) orelse return null;
            var digits: std.ArrayList(u8) = .empty;
            for (s[i..n]) |d| if (d != ',') try digits.append(a, d);
            if (digits.items.len > 1 and digits.items[0] == '0' and digits.items[1] != 'x') return null;
            try out.append(a, .{ .kind = .num, .text = digits.items });
            i = n;
            continue;
        }
        const nl = text.nameLen(s, i);
        if (nl > 0 and (i + nl == s.len or !text.isWord(s[i + nl]))) {
            try out.append(a, .{ .kind = .name, .text = s[i .. i + nl] });
            i += nl;
            continue;
        }
        if (std.mem.startsWith(u8, s[i..], "...") or std.mem.startsWith(u8, s[i..], "…")) {
            try out.append(a, .{ .kind = .op, .text = "..." });
            i += 3;
            continue;
        }
        if (std.mem.startsWith(u8, s[i..], "**")) {
            try out.append(a, .{ .kind = .op, .text = "^" });
            i += 2;
            continue;
        }
        const op: ?[]const u8, const w: usize = if (std.mem.indexOfScalar(u8, "+-*^()", c) != null)
            .{ s[i .. i + 1], 1 }
        else if (std.mem.startsWith(u8, s[i..], "×") or std.mem.startsWith(u8, s[i..], "·"))
            .{ "*", 2 }
        else if (std.mem.startsWith(u8, s[i..], "−"))
            .{ "-", 3 }
        else if (i + 3 <= s.len and std.ascii.eqlIgnoreCase(s[i .. i + 3], "mod") and (i + 3 == s.len or !std.ascii.isAlphabetic(s[i + 3])))
            .{ "%", 3 }
        else
            .{ null, 0 };
        try out.append(a, .{ .kind = .op, .text = op orelse return null });
        i += w;
    }
    return out.items;
}

/// The end of a number token at `i`: 0x hex, strict comma groups or plain digits, not running into a word.
fn number(s: []const u8, i: usize) ?usize {
    const stop = struct {
        fn ok(t: []const u8, j: usize, comma: bool) bool {
            if (j >= t.len) return true;
            if (text.isWord(t[j]) or (comma and t[j] == ',')) return false;
            return !(t[j] == '.' and j + 1 < t.len and std.ascii.isDigit(t[j + 1]));
        }
    };
    if (s[i] == '0' and i + 2 < s.len and s[i + 1] == 'x' and std.ascii.isHex(s[i + 2])) {
        var j = i + 2;
        while (j < s.len and std.ascii.isHex(s[j])) j += 1;
        if (j < s.len and (text.isWord(s[j]) or s[j] == ',' or s[j] == '.')) return null;
        return j;
    }
    if (s[i] != '0') {
        var j = i;
        while (j < s.len and j - i < 3 and std.ascii.isDigit(s[j])) j += 1;
        var groups: usize = 0;
        while (j + 3 < s.len + 0 and s[j] == ',' and allDigits(s[j + 1 .. j + 4])) : (groups += 1) j += 4;
        if (groups > 0 and stop.ok(s, j, true)) return j;
    }
    var j = i;
    while (j < s.len and std.ascii.isDigit(s[j])) j += 1;
    return if (stop.ok(s, j, true)) j else null;
}

fn allDigits(s: []const u8) bool {
    for (s) |c| if (!std.ascii.isDigit(c)) return false;
    return true;
}

/// `a*b*...*c` with b = a+1 and c - a <= 64 becomes the explicit product; any other `...` refuses.
fn expandRun(a: Allocator, toks: []Tok) Allocator.Error!?[]Tok {
    var dots = false;
    for (toks) |t| dots = dots or std.mem.eql(u8, t.text, "...");
    if (!dots) return toks;
    if (toks.len != 7) return null;
    const shape = [_][]const u8{ "", "*", "", "*", "...", "*", "" };
    for (toks, shape) |t, want| {
        if (want.len == 0) {
            if (t.kind != .num or t.text.len > 6 or std.mem.startsWith(u8, t.text, "0x")) return null;
        } else if (!std.mem.eql(u8, t.text, want)) return null;
    }
    const x = std.fmt.parseInt(u32, toks[0].text, 10) catch return null;
    const y = std.fmt.parseInt(u32, toks[2].text, 10) catch return null;
    const z = std.fmt.parseInt(u32, toks[6].text, 10) catch return null;
    if (y != x + 1 or z <= y or z - x > 64) return null;
    var out: std.ArrayList(Tok) = .empty;
    var n = x;
    while (n <= z) : (n += 1) {
        if (n != x) try out.append(a, .{ .kind = .op, .text = "*" });
        try out.append(a, .{ .kind = .num, .text = try std.fmt.allocPrint(a, "{d}", .{n}) });
    }
    return out.items;
}

/// sum := term (+|- term)*; term := factor (*|mod factor)*; factor := -factor | atom [^ factor]; one reading only.
const Parser = struct {
    names: []const Name,
    a: Allocator,
    toks: []const Tok,
    i: usize = 0,
    depth: u32 = 0, // brackets open at the cursor
    last_mod: bool = false, // the term just parsed ends in `mod`

    const Error = error{ OutOfMemory, Refused };

    fn peek(p: *const Parser, op: []const u8) bool {
        return p.i < p.toks.len and p.toks[p.i].kind == .op and std.mem.eql(u8, p.toks[p.i].text, op);
    }

    fn sum(p: *Parser) Error!Expr {
        var e = try p.term();
        var terms: u32 = 1;
        var has_mod = p.last_mod;
        while (p.peek("+") or p.peek("-")) {
            const op = p.toks[p.i].text;
            p.i += 1;
            const r = try p.term();
            has_mod = has_mod or p.last_mod;
            terms += 1;
            e = .{ .source = try cat(p.a, &.{ e.source, op, r.source }), .display = try cat(p.a, &.{ e.display, " ", op, " ", r.display }), .bound = sat(@max(e.bound, r.bound) + 1), .ops = e.ops + r.ops + 1, .constant = false, .run = try cat(p.a, &.{ e.run, op, r.run }) };
        }
        // `A + B mod M` reads two ways: mod only in an expression that is one term
        if (has_mod and terms > 1) return error.Refused;
        return e;
    }

    fn term(p: *Parser) Error!Expr {
        var e = try p.factor();
        var moded = false;
        while (p.peek("*") or p.peek("%")) {
            const op = p.toks[p.i].text;
            if (moded or (op[0] == '%' and p.depth > 0)) return error.Refused; // mod last, never in brackets
            p.i += 1;
            const start = p.i;
            const r = try p.factor();
            if (op[0] == '%') {
                // the modulus is a plain number, name or power chain (10^18), never a sum or a bracket
                for (p.toks[start..p.i]) |t| if (t.kind == .op and !std.mem.eql(u8, t.text, "^")) return error.Refused;
                moded = true;
                // a lone power mod m (a product carries no `pw`) runs as Python's pow(a, b, m): the same value
                const run = if (e.pw != null) try cat(p.a, &.{ "pow(", e.pw.?.base, ", ", e.pw.?.k, ", ", try tens(p.a, r.run), ")" }) else try cat(p.a, &.{ e.run, "%", r.run });
                e = .{ .source = try cat(p.a, &.{ e.source, "%", r.source }), .display = try cat(p.a, &.{ e.display, " mod ", r.display }), .bound = r.bound, .ops = e.ops + r.ops + 1, .constant = false, .run = run };
            } else {
                e = .{ .source = try cat(p.a, &.{ e.source, "*", r.source }), .display = try cat(p.a, &.{ e.display, "*", r.display }), .bound = sat(e.bound + r.bound), .ops = e.ops + r.ops + 1, .constant = false, .run = try cat(p.a, &.{ e.run, "*", r.run }) };
            }
        }
        p.last_mod = moded;
        return e;
    }

    fn factor(p: *Parser) Error!Expr {
        if (p.peek("-")) {
            p.i += 1;
            const e = try p.factor();
            return .{ .source = try cat(p.a, &.{ "-", e.source }), .display = try cat(p.a, &.{ "-", e.display }), .bound = e.bound, .ops = e.ops, .constant = false, .run = try cat(p.a, &.{ "-", e.run }) };
        }
        const base = try p.atom();
        if (!p.peek("^")) return base;
        p.i += 1;
        if (p.i >= p.toks.len or p.toks[p.i].kind != .num) return error.Refused; // a literal exponent only
        const k_text = p.toks[p.i].text;
        p.i += 1;
        if (p.peek("^")) return error.Refused; // a^b^c
        const k = std.fmt.parseInt(u32, k_text, 10) catch return error.Refused;
        const bound: u32 = if (base.constant and k <= 4096) constPowDigits(base.source, k) else sat(@as(u32, @intCast(@min(@as(u64, base.bound) * k, inf))));
        return .{ .source = try cat(p.a, &.{ base.source, "**", k_text }), .display = try cat(p.a, &.{ base.display, "^", k_text }), .bound = bound, .ops = base.ops + 1, .constant = false, .run = try cat(p.a, &.{ base.run, "**", k_text }), .pw = .{ .base = base.run, .k = k_text } };
    }

    fn atom(p: *Parser) Error!Expr {
        if (p.i >= p.toks.len) return error.Refused;
        const t = p.toks[p.i];
        p.i += 1;
        switch (t.kind) {
            .num => return .{ .source = try p.a.dupe(u8, t.text), .display = try p.a.dupe(u8, t.text), .bound = litDigits(t.text), .ops = 0, .constant = true, .run = try decimal(p.a, t.text) },
            .name => {
                const e = lookup(p.names, t.text) orelse return error.Refused;
                return .{ .source = try cat(p.a, &.{ "(", e.source, ")" }), .display = if (e.ops > 0) try cat(p.a, &.{ "(", e.display, ")" }) else e.display, .bound = e.bound, .ops = e.ops, .constant = e.constant, .run = try cat(p.a, &.{ "(", e.run, ")" }) };
            },
            .op => {
                if (!std.mem.eql(u8, t.text, "(")) return error.Refused;
                p.depth += 1;
                const e = try p.sum();
                if (!p.peek(")")) return error.Refused;
                p.depth -= 1;
                p.i += 1;
                return .{ .source = try cat(p.a, &.{ "(", e.source, ")" }), .display = try cat(p.a, &.{ "(", e.display, ")" }), .bound = e.bound, .ops = e.ops, .constant = e.constant, .run = try cat(p.a, &.{ "(", e.run, ")" }) };
            },
        }
    }
};

fn cat(a: Allocator, parts: []const []const u8) Allocator.Error![]u8 {
    return std.mem.concat(a, u8, parts);
}

/// A modulus `10**k` written out in digits (notation, like hex in decimal); any other modulus as it is.
fn tens(a: Allocator, run: []const u8) Allocator.Error![]const u8 {
    if (!std.mem.startsWith(u8, run, "10**") or run.len == 4 or !allDigits(run[4..])) return run;
    const k = std.fmt.parseInt(u8, run[4..], 10) catch return run;
    if (k > 128) return run;
    const out = try a.alloc(u8, @as(usize, k) + 1);
    out[0] = '1';
    @memset(out[1..], '0');
    return out;
}

/// A literal as the frontend gets it: hex in decimal (text plumbing, the value unchanged), else as written.
fn decimal(a: Allocator, lit: []const u8) Allocator.Error![]u8 {
    if (!std.mem.startsWith(u8, lit, "0x")) return a.dupe(u8, lit);
    var v = try std.math.big.int.Managed.init(a);
    v.setString(16, lit[2..]) catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        else => unreachable, // number() admits hex digits only
    };
    return v.toString(a, 10, .lower) catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        else => unreachable, // base 10 is valid
    };
}

/// Decimal digits of a literal (hex: of its value).
fn litDigits(lit: []const u8) u32 {
    if (!std.mem.startsWith(u8, lit, "0x")) return @intCast(@min(lit.len, inf));
    const h = lit.len - 2;
    if (h <= 32) {
        const v = std.fmt.parseInt(u128, lit[2..], 16) catch return inf;
        var d: u32 = 1;
        var x = v;
        while (x >= 10) : (x /= 10) d += 1;
        return d;
    }
    return @intCast(@min(h * 121 / 100 + 1, inf));
}

/// Digits of a constant base raised to `k`, from logarithms of the literal (its own width, not an answer).
fn constPowDigits(source: []const u8, k: u32) u32 {
    var s = source;
    while (s.len >= 2 and s[0] == '(' and s[s.len - 1] == ')') s = s[1 .. s.len - 1];
    const v: f64 = if (std.mem.startsWith(u8, s, "0x"))
        @floatFromInt(std.fmt.parseInt(u128, s[2..], 16) catch return inf)
    else
        std.fmt.parseFloat(f64, s) catch return inf;
    if (v < 2 or k == 0) return 1;
    const d = @as(f64, @floatFromInt(k)) * std.math.log10(v);
    return @intFromFloat(@min(@floor(d + 1e-9) + 1, @as(f64, inf)));
}

fn lookup(names: []const Name, n: []const u8) ?Expr {
    for (names) |m| if (std.mem.eql(u8, m.name, n)) return m.expr;
    return null;
}
