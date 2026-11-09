//! The runner's `trace` form: how one `Name = …` sighting reads, for the binding states in runner_trace.zig.
const std = @import("std");
const text = @import("runner_trace_text.zig");

/// A name's binding: not yet exact, exact, kept aside for a plain restatement, stale through a dependency, blocked.
pub const State = enum { tentative, exact, quarantined, stale, hard };

/// Which recovery rules run (T5-BINDING-RECOVERY.md); the hard guards run under every combination.
pub const Rules = struct {
    r1: bool = true, // a later plain exact declaration binds a name first seen tentatively
    r2: bool = true, // typed metadata and spaced groups prove a restatement; an unprovable one quarantines
    r3: bool = true, // an equality chain binds or restates when every expression is the same integer
    r4: bool = true, // `B mod A = …` is a compound left side, no assignment to A
};

/// The formula at the start of a right side, read with lowercase names, `/`, implicit products, pipes and 1eN.
pub const Formula = struct { end: usize, names: bool = false, div: bool = false, sci: bool = false, pipe: bool = false, decimal: bool = false, self: bool = false };

/// The longest well-formed formula at the start of `s` (`end` 0 if none); `self` if it reads the assigned name.
pub fn formula(s: []const u8, name: []const u8) Formula {
    var f: Formula = .{ .end = 0 };
    var i: usize = 0;
    var depth: usize = 0;
    while (true) {
        // an operand
        while (i < s.len and text.isSpace(s[i])) i += 1;
        while (i < s.len and (s[i] == '-' or std.mem.startsWith(u8, s[i..], "−"))) i += if (s[i] == '-') 1 else 3;
        while (i < s.len and s[i] == '(') : (i += 1) depth += 1;
        while (i < s.len and text.isSpace(s[i])) i += 1;
        const at = i;
        if (i < s.len and std.ascii.isDigit(s[i])) {
            if (s[i] == '0' and i + 2 < s.len and s[i + 1] == 'x' and std.ascii.isHex(s[i + 2])) {
                i += 2;
                while (i < s.len and std.ascii.isHex(s[i])) i += 1;
            } else {
                while (i < s.len and (std.ascii.isDigit(s[i]) or (s[i] == ',' and i + 1 < s.len and std.ascii.isDigit(s[i + 1])))) i += 1;
                if (i + 1 < s.len and s[i] == '.' and std.ascii.isDigit(s[i + 1])) {
                    f.decimal = true;
                    i += 1;
                    while (i < s.len and std.ascii.isDigit(s[i])) i += 1;
                }
                if (i + 1 < s.len and (s[i] == 'e' or s[i] == 'E') and std.ascii.isDigit(s[i + 1])) {
                    f.sci = true;
                    i += 1;
                    while (i < s.len and std.ascii.isDigit(s[i])) i += 1;
                }
            }
            // an implicit product: `8A`, `2q0`
            if (i < s.len and std.ascii.isAlphabetic(s[i])) i = ident(s, i, name, &f);
        } else if (i < s.len and std.ascii.isAlphabetic(s[i])) {
            i = ident(s, i, name, &f);
        }
        if (i == at) break; // no operand: the formula ends before the operator that wanted one
        while (true) {
            while (i < s.len and text.isSpace(s[i])) i += 1;
            if (i < s.len and s[i] == ')' and depth > 0) {
                depth -= 1;
                i += 1;
            } else break;
        }
        if (depth == 0) f.end = trimmed(s, i);
        // an operator
        while (i < s.len and text.isSpace(s[i])) i += 1;
        const ops = [_][]const u8{ "**", "+", "-", "*", "/", "^", "%", "|", "×", "÷", "·", "−" };
        var w: usize = 0;
        for (ops) |o| if (std.mem.startsWith(u8, s[i..], o)) {
            w = o.len;
            break;
        };
        if (w == 0 and text.wordAt(s, i, "mod", true)) w = 3;
        if (w == 0) break;
        if (s[i] == '/' or std.mem.startsWith(u8, s[i..], "÷")) f.div = true;
        if (s[i] == '|') f.pipe = true;
        i += w;
    }
    return f;
}

fn ident(s: []const u8, i: usize, name: []const u8, f: *Formula) usize {
    var j = i;
    while (j < s.len and text.isWord(s[j])) j += 1;
    const w = s[i..j];
    if (std.ascii.eqlIgnoreCase(w, "mod")) return i; // an operator, not a name
    f.names = true;
    if (std.mem.eql(u8, w, name)) f.self = true;
    return j;
}

fn trimmed(s: []const u8, i: usize) usize {
    return std.mem.trimEnd(u8, s[0..i], " \t").len;
}

/// `Name` here is updated in place (`N += b`, `N *= 2`, `N++`): a hard block, never an assignment the binder reads.
pub fn updateAt(s: []const u8, i: usize) ?usize {
    if (i > 0 and (text.isWord(s[i - 1]) or s[i - 1] == ',' or s[i - 1] == '.')) return null;
    const n = text.nameLen(s, i);
    if (n == 0 or (i + n < s.len and text.isWord(s[i + n]))) return null;
    var j = i + n;
    if (j + 2 <= s.len and (std.mem.eql(u8, s[j .. j + 2], "++") or std.mem.eql(u8, s[j .. j + 2], "--")) and
        (j + 2 == s.len or !(text.isWord(s[j + 2]) or s[j + 2] == '-' or s[j + 2] == '+'))) return n;
    while (j < s.len and text.isSpace(s[j])) j += 1;
    if (j + 2 <= s.len and std.mem.indexOfScalar(u8, "+-*/%^", s[j]) != null and s[j + 1] == '=' and (j + 2 == s.len or s[j + 2] != '=')) return n;
    return null;
}

/// `B mod A = …`, `B % A = …`, `x ≡ A = …`: the name ends a compound left side, so the `=` assigns nothing (R4).
pub fn compoundLeft(left: []const u8) bool {
    const l = std.mem.trimEnd(u8, left, " \t");
    if (l.len == 0) return false;
    if (l[l.len - 1] == '%') return true;
    for ([_][]const u8{ "≡", "÷", "·" }) |o| if (std.mem.endsWith(u8, l, o)) return true;
    return l.len >= 3 and text.wordAt(l, l.len - 3, "mod", true);
}

/// The `=`-separated expressions of a chain (`0x76543210 = 1985229328`); `=>`, `==`, `<=`, `>=`, `!=` do not split.
pub fn pieces(s: []const u8, out: *std.ArrayList([]const u8), a: std.mem.Allocator) std.mem.Allocator.Error!void {
    var from: usize = 0;
    for (s, 0..) |c, k| {
        if (c != '=') continue;
        if (k + 1 < s.len and (s[k + 1] == '>' or s[k + 1] == '=')) continue;
        if (k > 0 and std.mem.indexOfScalar(u8, "<>!=", s[k - 1]) != null) continue;
        try out.append(a, s[from..k]);
        from = k + 1;
    }
    try out.append(a, s[from..]);
}

/// A piece with a closing bracket that no bracket in it opens (`…=4275878552)` ending a parenthetical) dropped.
pub fn unclosed(p: []const u8) []const u8 {
    var depth: isize = 0;
    for (p) |c| depth += if (c == '(') @as(isize, 1) else if (c == ')') @as(isize, -1) else 0;
    var q = std.mem.trimEnd(u8, p, " \t");
    while (depth < 0 and q.len > 0 and q[q.len - 1] == ')') : (depth += 1) q = std.mem.trimEnd(u8, q[0 .. q.len - 1], " \t");
    return q;
}

const meta_words = [_][]const u8{ "groups", "group", "chunks", "chunk", "limbs", "limb" };

/// The value before typed layout metadata (`98,765,432,109 -> groups [98, …]`), or the piece if it has none.
pub fn metadata(p: []const u8) struct { value: []const u8, cut: bool } {
    for (0..p.len) |i| for (meta_words) |w| if (text.wordAt(p, i, w, true)) {
        var v = std.mem.trimEnd(u8, p[0..i], " \t");
        for ([_][]const u8{ "->", "=>", "→", ":" }) |arrow| if (std.mem.endsWith(u8, v, arrow)) {
            v = std.mem.trimEnd(u8, v[0 .. v.len - arrow.len], " \t");
        };
        return .{ .value = v, .cut = true };
    };
    return .{ .value = p, .cut = false };
}

/// Spaced digit groups (`271 828 183`) as digits, only when the raw line's next note counts them (`(9 digits)`).
pub fn spaced(a: std.mem.Allocator, p: []const u8, raw: []const u8) std.mem.Allocator.Error!?[]const u8 {
    const v = std.mem.trim(u8, p, " \t");
    var digits: std.ArrayList(u8) = .empty;
    var groups: usize = 0;
    var it = std.mem.splitScalar(u8, v, ' ');
    while (it.next()) |g| : (groups += 1) {
        if (g.len == 0 or g.len > 3 or (groups > 0 and g.len != 3)) return null;
        for (g) |c| if (!std.ascii.isDigit(c)) return null;
        if (groups == 0 and g[0] == '0') return null;
        try digits.appendSlice(a, g);
    }
    if (groups < 2) return null;
    var from: usize = 0;
    while (std.mem.indexOfPos(u8, raw, from, v)) |k| : (from = k + 1) {
        var j = k + v.len;
        while (j < raw.len and text.isSpace(raw[j])) j += 1;
        if (j >= raw.len or raw[j] != '(') continue;
        j += 1;
        const n0 = j;
        while (j < raw.len and std.ascii.isDigit(raw[j])) j += 1;
        const n = std.fmt.parseInt(usize, raw[n0..j], 10) catch continue;
        while (j < raw.len and text.isSpace(raw[j])) j += 1;
        if (n == digits.items.len and text.wordAt(raw, j, "digits", true)) return digits.items;
    }
    return null;
}

/// A right side running into the next definition (`A=123 B=456`) ends before it (R3), unless an operator leads it.
pub fn nextDef(txt: []const u8) usize {
    for (@min(1, txt.len)..txt.len) |k| {
        if (!text.isSpace(txt[k - 1]) or text.defAt(txt, k) == null) continue;
        const l = std.mem.trimEnd(u8, txt[0..k], " \t");
        if (l.len > 0 and std.mem.indexOfScalar(u8, "+-*/^%=(", l[l.len - 1]) == null and !std.mem.endsWith(u8, l, "×")) return k;
    }
    return txt.len;
}

/// After a leading layout list (`chunks [493,064,720, 364,758,544, 18] = 18,364,…`), where the right side goes on.
pub fn layoutList(s: []const u8, rhs: usize) ?usize {
    for (meta_words) |w| if (text.wordAt(s, rhs, w, true)) {
        var j = rhs + w.len;
        while (j < s.len and text.isSpace(s[j])) j += 1;
        if (j >= s.len or s[j] != '[') return null;
        const close = std.mem.indexOfScalarPos(u8, s, j, ']') orelse return null;
        return close + 1;
    };
    return null;
}

/// `p` (a whole formula) as a runner_trace_value source: no spaces or digit commas, `^` as `**`; null if wide.
pub fn plain(a: std.mem.Allocator, p: []const u8) std.mem.Allocator.Error!?[]u8 {
    if (p.len > 512) return null;
    var out: std.ArrayList(u8) = .empty;
    for (p, 0..) |c, i| switch (c) {
        ' ', '\t' => {},
        ',' => if (i == 0 or !std.ascii.isDigit(p[i - 1]) or i + 1 == p.len or !std.ascii.isDigit(p[i + 1])) return null,
        '^' => try out.appendSlice(a, "**"),
        else => try out.append(a, c),
    };
    return out.items;
}

/// The left side ends in the value of a definition on the same line (`A=123 B=…`), not a label like `Step3 N=` (R3).
pub fn afterValue(left: []const u8) bool {
    const l = std.mem.trimEnd(u8, left, " \t");
    var j = l.len;
    while (j > 0 and (std.ascii.isHex(l[j - 1]) or l[j - 1] == ',' or l[j - 1] == 'x')) j -= 1;
    if (j == l.len or !std.ascii.isDigit(l[j])) return false;
    const before = std.mem.trimEnd(u8, l[0..j], " \t");
    return before.len > 0 and before[before.len - 1] == '=';
}
