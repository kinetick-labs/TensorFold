//! The runner's `trace` form: sentence, definition, statement-phrase and hand-trace cue predicates over one line.
const std = @import("std");
const Allocator = std.mem.Allocator;

pub fn isWord(c: u8) bool {
    return std.ascii.isAlphanumeric(c) or c == '_';
}

pub fn isSpace(c: u8) bool {
    return c == ' ' or c == '\t' or c == '\r';
}

/// Case-insensitive `words` at `i`, separated by one or more spaces; the end index, else null.
pub fn seq(s: []const u8, i: usize, words: []const []const u8) ?usize {
    var j = i;
    for (words, 0..) |w, k| {
        if (k > 0) {
            const from = j;
            while (j < s.len and isSpace(s[j])) j += 1;
            if (j == from) return null;
        }
        if (j + w.len > s.len or !std.ascii.eqlIgnoreCase(s[j .. j + w.len], w)) return null;
        j += w.len;
    }
    return j;
}

pub fn wordAt(s: []const u8, i: usize, w: []const u8, whole: bool) bool {
    if (i > 0 and isWord(s[i - 1])) return false;
    if (i + w.len > s.len or !std.ascii.eqlIgnoreCase(s[i .. i + w.len], w)) return false;
    return !whole or i + w.len == s.len or !isWord(s[i + w.len]);
}

fn anyWord(s: []const u8, words: []const []const u8, whole: bool) bool {
    for (0..s.len) |i| for (words) |w| if (wordAt(s, i, w, whole)) return true;
    return false;
}

/// Doubt or approximation near a number: ? ~ ≈, 1.2e20, maybe/known/wait/around, approx…/estimate…/guess….
pub fn approx(s: []const u8) bool {
    if (std.mem.indexOfAny(u8, s, "?~") != null or std.mem.indexOf(u8, s, "≈") != null) return true;
    for (0..s.len) |i| {
        if (!std.ascii.isDigit(s[i])) continue;
        var j = i + 1;
        if (j < s.len and s[j] == '.') j += 1;
        while (j < s.len and std.ascii.isDigit(s[j])) j += 1;
        if (j + 1 < s.len and (s[j] == 'e' or s[j] == 'E') and std.ascii.isDigit(s[j + 1])) return true;
    }
    return anyWord(s, &.{ "maybe", "known", "wait", "around" }, true) or anyWord(s, &.{ "approx", "estimate", "guess" }, false);
}

/// A sentence with a question or a hypothesis (if/suppose/would/could/…) defines nothing.
pub fn hypothetical(s: []const u8) bool {
    if (std.mem.indexOfScalar(u8, s, '?') != null) return true;
    const t = std.mem.trimStart(u8, s, " \t");
    for ([_][]const u8{ "if", "suppose", "assume", "maybe", "wait", "perhaps" }) |w| if (wordAt(t, 0, w, true)) return true;
    return anyWord(s, &.{ "maybe", "would", "could", "should", "might", "if" }, true);
}

/// The sentence of `clean` around byte `i` (ends at ". ", "? " or "! ").
pub fn sentenceAt(clean: []const u8, i: usize) []const u8 {
    var a: usize = 0;
    for ([_][]const u8{ ". ", "? ", "! " }) |sep| {
        if (std.mem.lastIndexOf(u8, clean[0..i], sep)) |k| a = @max(a, k + 2);
    }
    var b: usize = clean.len;
    for ([_][]const u8{ ". ", "? ", "! " }) |sep| {
        if (std.mem.indexOfPos(u8, clean, i, sep)) |k| b = @min(b, k + 1);
    }
    return clean[@min(a, b)..b];
}

/// Drops digit-count notes ("(20 digits?)", "(30 nines)", "(7 groups)") and the spaces before them.
pub fn stripNotes(a: Allocator, raw: []const u8) Allocator.Error![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    var i: usize = 0;
    while (i < raw.len) {
        if (raw[i] == '(') if (std.mem.indexOfScalarPos(u8, raw, i + 1, ')')) |close| {
            const inner = raw[i + 1 .. close];
            if (std.mem.indexOfScalar(u8, inner, '(') == null and isNote(inner)) {
                while (out.items.len > 0 and isSpace(out.items[out.items.len - 1])) out.items.len -= 1;
                i = close + 1;
                continue;
            }
        };
        try out.append(a, raw[i]);
        i += 1;
    }
    return out.items;
}

fn isNote(inner: []const u8) bool {
    for (0..inner.len) |i| {
        if (wordAt(inner, i, "digit", true) or wordAt(inner, i, "digits", true)) return true;
        if (wordAt(inner, i, "group", true) or wordAt(inner, i, "groups", true)) return true;
    }
    var j: usize = 0;
    while (j < inner.len and std.ascii.isDigit(inner[j])) j += 1;
    if (j == 0) return false;
    const k = j;
    while (j < inner.len and isSpace(inner[j])) j += 1;
    return j > k and std.mem.eql(u8, inner[j..], "nines");
}

/// A hand trace starts here: a step, table or digit line, or words like long division, carry, borrow, by hand.
pub fn cueIn(s: []const u8) bool {
    var i: usize = 0;
    while (i < s.len and isSpace(s[i])) i += 1;
    for ([_][]const u8{ "so", "now" }) |w| if (seq(s, i, &.{w})) |j| if (j < s.len and isSpace(s[j])) {
        i = j;
        while (i < s.len and isSpace(s[i])) i += 1;
        break;
    };
    if (i < s.len and std.ascii.isDigit(s[i])) return true;
    if (wordAt(s, i, "step", true) or wordAt(s, i, "start", true) or (seq(s, i, &.{"first,"}) != null)) return true;
    if (i < s.len and (s[i] == 'i' or s[i] == 'I')) {
        var j = i + 1;
        while (j < s.len and isSpace(s[j])) j += 1;
        if (j < s.len and s[j] == '=') {
            j += 1;
            while (j < s.len and isSpace(s[j])) j += 1;
            if (j < s.len and s[j] == '0' and (j + 1 == s.len or !isWord(s[j + 1]))) return true;
        }
    }
    if (seq(s, i, &.{"row"})) |j| {
        var k = j;
        while (k < s.len and isSpace(s[k])) k += 1;
        if (k < s.len and s[k] == '0' and (k + 1 == s.len or !isWord(s[k + 1]))) return true;
    }
    const phrases = [_][]const []const u8{
        &.{ "long", "division" },   &.{ "partial", "product" }, &.{ "repeated", "squaring" },
        &.{ "step", "by", "step" }, &.{ "by", "hand" },
    };
    const lets = [_][]const u8{ "compute", "calculate", "do", "multiply", "perform", "use", "divide", "subtract", "add", "square" };
    for (0..s.len) |k| {
        for (phrases) |p| if (seq(s, k, p) != null) return true;
        for ([_][]const u8{ "carry", "carries" }) |w| if (wordAt(s, k, w, true)) return true;
        for ([_][]const u8{ "group", "borrow", "manual" }) |w| if (wordAt(s, k, w, false)) return true;
        if (wordAt(s, k, "base", false)) {
            var j = k + 4;
            while (j < s.len and isSpace(s[j])) j += 1;
            for ([_][]const u8{ "1000", "10^", "1e", "10**" }) |b| if (std.mem.startsWith(u8, s[j..], b)) return true;
        }
        for ([_][]const u8{ "let's", "lets" }) |l| for (lets) |v| if (seq(s, k, &.{ l, v }) != null) return true;
        for ([_][]const u8{ "carefully", "exactly", "step" }) |v| if (seq(s, k, &.{ "compute", v }) != null) return true;
        for ([_][]const u8{ "use", "perform", "do", "compute" }) |v| if (seq(s, k, &.{ "we", "can", v }) != null) return true;
    }
    return false;
}

// ---- definitions ----

pub const Def = struct { start: usize, name_end: usize, rhs: usize };

pub fn nameLen(s: []const u8, i: usize) usize {
    if (i >= s.len or !std.ascii.isUpper(s[i])) return 0;
    var j = i + 1;
    if (j < s.len and std.ascii.isAlphabetic(s[j])) j += 1;
    if (j < s.len and std.ascii.isDigit(s[j])) j += 1;
    return j - i;
}

/// `Name = ` at `i` (not inside a word, a comma group or a decimal).
pub fn defAt(s: []const u8, i: usize) ?Def {
    if (i > 0 and (isWord(s[i - 1]) or s[i - 1] == ',' or s[i - 1] == '.')) return null;
    const n = nameLen(s, i);
    if (n == 0) return null;
    var j = i + n;
    while (j < s.len and isSpace(s[j])) j += 1;
    if (j >= s.len or s[j] != '=') return null;
    j += 1;
    while (j < s.len and isSpace(s[j])) j += 1;
    return .{ .start = i, .name_end = i + n, .rhs = j };
}

/// The name starts a definition: it does not follow an operator, `=`, a closing bracket or a digit ("A * B = …").
pub fn defLeftOk(left: []const u8) bool {
    const l = std.mem.trimEnd(u8, left, " \t");
    if (l.len == 0) return true;
    const c = l[l.len - 1];
    if (std.ascii.isDigit(c) or std.mem.indexOfScalar(u8, "+-*/^=)]>", c) != null) return false;
    return !std.mem.endsWith(u8, l, "×");
}

/// `Name =` here assigns the name (a label, `Let` or a digit may lead: "Step3 N = …"); after an operator it states one.
pub fn assignLeft(left: []const u8) bool {
    const l = std.mem.trimEnd(u8, left, " \t");
    if (l.len == 0) return true;
    if (std.mem.indexOfScalar(u8, "+-*/^=", l[l.len - 1]) != null) return false;
    return !std.mem.endsWith(u8, l, "×");
}

/// The shortest right side followed by a sentence end, ", ", " (", or " by|from|and|then|is|which|so|has".
pub fn defEnd(s: []const u8, rhs: usize) ?usize {
    var k = rhs + 1;
    while (k <= s.len) : (k += 1) {
        var j = k;
        while (j < s.len and isSpace(s[j])) j += 1;
        if (j == s.len) return k;
        if ((s[j] == '.' or s[j] == ';') and (j + 1 == s.len or isSpace(s[j + 1]))) return k;
        if (s[j] == ',' and j + 1 < s.len and isSpace(s[j + 1])) return k;
        if (j > k and s[j] == '(') return k;
        if (j > k) for ([_][]const u8{ "by", "from", "and", "then", "is", "which", "so", "has" }) |w| {
            if (j + w.len <= s.len and std.mem.eql(u8, s[j .. j + w.len], w) and (j + w.len == s.len or !isWord(s[j + w.len]))) return k;
        };
    }
    return null;
}

pub fn sentenceEndsAt(s: []const u8, k: usize) bool {
    var j = k;
    while (j < s.len and isSpace(s[j])) j += 1;
    return j == s.len or ((s[j] == '.' or s[j] == ';') and (j + 1 == s.len or isSpace(s[j + 1])));
}

// ---- statements ----

pub const Head = struct {
    end: usize,
    bare: bool = false,
    pub fn less(_: void, x: Head, y: Head) bool {
        return x.end < y.end;
    }
};

/// need / we need / need to / let's / compute / calculate / evaluate / expression, then optionally a verb.
pub fn goalHeads(a: Allocator, s: []const u8, out: *std.ArrayList(Head)) Allocator.Error!void {
    const heads = [_][]const []const u8{ &.{ "we", "need", "to" }, &.{ "we", "need" }, &.{ "need", "to" }, &.{"need"}, &.{"let's"}, &.{"lets"}, &.{"compute"}, &.{"calculate"}, &.{"evaluate"}, &.{"expression"} };
    var i: usize = 0;
    while (i < s.len) {
        if (i > 0 and !(isSpace(s[i - 1]) or s[i - 1] == '.' or s[i - 1] == ':' or s[i - 1] == ';')) {
            i += 1;
            continue;
        }
        var end: ?usize = null;
        for (heads) |h| if (seq(s, i, h)) |j| if (j == s.len or !isWord(s[j])) {
            end = j;
            break;
        };
        var j = end orelse {
            i += 1;
            continue;
        };
        for ([_][]const u8{ "compute", "calculate", "evaluate", "find", "to" }) |v| {
            var k = j;
            while (k < s.len and isSpace(s[k])) k += 1;
            if (k > j and seq(s, k, &.{v}) != null and k + v.len < s.len and isSpace(s[k + v.len])) {
                j = k + v.len;
                break;
            }
        }
        const after = j;
        while (j < s.len and isSpace(s[j])) j += 1;
        if (j == after) {
            i += 1;
            continue;
        }
        try out.append(a, .{ .end = j });
        i = j;
    }
}

/// A sentence that is only an expression: a digit or `(` to a digit or `)` before ". " (`commas`: thousands groups).
pub fn bareHeads(a: Allocator, s: []const u8, commas: bool, out: *std.ArrayList(Head)) Allocator.Error!void {
    var i: usize = 0;
    while (i < s.len) : (i += 1) {
        if (!(i == 0 or (i >= 2 and s[i - 2] == '.' and s[i - 1] == ' '))) continue;
        if (!(std.ascii.isDigit(s[i]) or s[i] == '(')) continue;
        var j = i;
        while (j < s.len) {
            const c = s[j];
            if (std.ascii.isAlphanumeric(c) or std.mem.indexOfScalar(u8, "^*+-() ", c) != null) {
                j += 1;
            } else if (commas and thousandsAt(s, j)) {
                j += 1;
            } else if (std.mem.startsWith(u8, s[j..], "×")) {
                j += 2;
            } else if (std.mem.startsWith(u8, s[j..], "−")) {
                j += 3;
            } else break;
        }
        if (j > i and j < s.len and s[j] == '.' and (j + 1 == s.len or isSpace(s[j + 1])) and
            (std.ascii.isDigit(s[j - 1]) or s[j - 1] == ')')) try out.append(a, .{ .end = i, .bare = true });
    }
}

pub const Phrase = struct { kind: enum { divmod, mul, subfrom, expr, gcd, isqrt, edit }, parts: [2][2]usize = undefined, n: usize = 1, end: usize };

const fill_words = [_][]const u8{ "the", "answer", "exact", "exactly", "value", "of", "integer", "decimal", "product", "sum", "difference", "result", "floor", "huge", "number", "big" };
const trail_words = [_][]const u8{ "exactly", "exact", "accurately", "carefully", "precisely", "manually" };

/// The phrase a head introduces, up to the first sentence end (`.,;:` then a space, or the line's end; `whole`: named).
pub fn phrase(rest: []const u8, whole: bool) ?Phrase {
    var i: usize = 0;
    var n: usize = 0;
    var integral = false; // "integer" or "floor" read: a square root is isqrt's
    while (n < 4) : (n += 1) {
        var hit = false;
        for (fill_words) |w| if (seq(rest, i, &.{w})) |j| if (j < rest.len and isSpace(rest[j])) {
            integral = integral or std.mem.eql(u8, w, "integer") or std.mem.eql(u8, w, "floor");
            i = j;
            while (i < rest.len and isSpace(rest[i])) i += 1;
            hit = true;
            break;
        };
        if (!hit) break;
    }
    if (verbPhrase(rest, i, "divide", &.{"by"})) |p| return .{ .kind = .divmod, .parts = p.parts, .n = 2, .end = p.end };
    if (verbPhrase(rest, i, "multiply", &.{ "by", "and", "with" })) |p| return .{ .kind = .mul, .parts = p.parts, .n = 2, .end = p.end };
    if (multiplyDefs(rest, i)) |p| return .{ .kind = .mul, .parts = p.parts, .n = 2, .end = p.end };
    if (verbPhrase(rest, i, "subtract", &.{"from"})) |p| return .{ .kind = .subfrom, .parts = p.parts, .n = 2, .end = p.end };
    if (whole) if (gcdCall(rest, i)) |p| return .{ .kind = .gcd, .parts = p.parts, .n = 2, .end = p.end };
    if (whole) if (gcdOf(rest, i)) |p| return .{ .kind = .gcd, .parts = p.parts, .n = 2, .end = p.end };
    if (whole) if (isqrtOf(rest, i, integral)) |p| return .{ .kind = .isqrt, .parts = p.parts, .end = p.end };
    if (whole) if (editDistance(rest, i)) |p| return .{ .kind = .edit, .parts = p.parts, .n = 2, .end = p.end };
    const e = lazyEnd(rest, i) orelse return null;
    return .{ .kind = .expr, .parts = .{ .{ i, e.text }, undefined }, .end = e.end };
}

const Two = struct { parts: [2][2]usize, end: usize };

/// An optional `Name =` before an operand (case-insensitive name, as the census regex reads it).
fn skipDef(s: []const u8, i: usize) usize {
    if (i >= s.len or !std.ascii.isAlphabetic(s[i])) return i;
    var j = i + 1;
    if (j < s.len and std.ascii.isAlphabetic(s[j])) j += 1;
    if (j < s.len and std.ascii.isDigit(s[j])) j += 1;
    while (j < s.len and isSpace(s[j])) j += 1;
    if (j >= s.len or s[j] != '=') return i;
    j += 1;
    while (j < s.len and isSpace(s[j])) j += 1;
    return j;
}

/// `verb X <sep> Y`: X up to the first spaced separator word, Y up to the first sentence end.
fn verbPhrase(s: []const u8, i: usize, verb: []const u8, seps: []const []const u8) ?Two {
    var j = seq(s, i, &.{verb}) orelse return null;
    const after = j;
    while (j < s.len and isSpace(s[j])) j += 1;
    if (j == after) return null;
    const x0 = skipDef(s, j);
    var k = x0 + 1;
    while (k < s.len) : (k += 1) {
        if (!isSpace(s[k])) continue;
        var m = k;
        while (m < s.len and isSpace(s[m])) m += 1;
        for (seps) |sep| {
            if (m + sep.len < s.len and std.ascii.eqlIgnoreCase(s[m .. m + sep.len], sep) and isSpace(s[m + sep.len])) {
                var y = m + sep.len;
                while (y < s.len and isSpace(s[y])) y += 1;
                y = skipDef(s, y);
                const e = lazyEnd(s, y) orelse return null;
                return .{ .parts = .{ .{ x0, k }, .{ y, e.text } }, .end = e.end };
            }
        }
    }
    return null;
}

/// `gcd(X, Y)` then a sentence end: X and Y split at the call's one top-level comma.
fn gcdCall(s: []const u8, i: usize) ?Two {
    const open = seq(s, i, &.{"gcd("}) orelse return null;
    var depth: usize = 0;
    var comma: ?usize = null;
    var k = open;
    while (k < s.len) : (k += 1) {
        switch (s[k]) {
            '(' => depth += 1,
            ')' => {
                if (depth == 0) break;
                depth -= 1;
            },
            ',' => if (depth == 0) {
                if (comma != null) return null; // three arguments, or a thousands comma: not one reading
                comma = k;
            },
            else => {},
        }
    } else return null;
    const c = comma orelse return null;
    const e = endAt(s, k + 1) orelse return null;
    return .{ .parts = .{ .{ open, c }, .{ c + 1, k } }, .end = e };
}

/// `gcd of X and Y` / `greatest common divisor of X, Y` then a sentence end: two literals, each maybe after `Name=`.
fn gcdOf(s: []const u8, i: usize) ?Two {
    var j = seq(s, i, &.{ "gcd", "of" }) orelse seq(s, i, &.{ "greatest", "common", "divisor", "of" }) orelse return null;
    j = spaces(s, j) orelse return null;
    const x = literal(s, skipDef(s, j)) orelse return null;
    const comma = x[1] < s.len and s[x[1]] == ',';
    const sep = spaces(s, x[1] + @intFromBool(comma)) orelse return null;
    const y0 = if (comma) sep else spaces(s, seq(s, sep, &.{"and"}) orelse return null) orelse return null;
    const y = literal(s, skipDef(s, y0)) orelse return null;
    // "gcd of 12,345, 18" may be three numbers, as in gcd(12,345, 18): a comma splits only plain literals
    if (comma and (std.mem.indexOfScalar(u8, s[x[0]..x[1]], ',') != null or std.mem.indexOfScalar(u8, s[y[0]..y[1]], ',') != null)) return null;
    // a third argument after a comma ("gcd of 12, 18, 30") is not a gcd of two
    var k = y[1];
    while (k < s.len and isSpace(s[k])) k += 1;
    if (k < s.len and s[k] == ',') return null;
    const e = trailEnd(s, y[1]) orelse return null;
    return .{ .parts = .{ x, y }, .end = e };
}

/// A decimal integer literal at `i` (thousands commas, not zero-led) that no word character continues.
fn literal(s: []const u8, i: usize) ?[2]usize {
    var j = i;
    while (j < s.len and (std.ascii.isDigit(s[j]) or thousandsAt(s, j))) j += 1;
    if (j == i or (s[i] == '0' and j > i + 1) or (j < s.len and isWord(s[j]))) return null;
    return .{ i, j };
}

/// `isqrt(N)`, or `sqrt of N` / `square root of N` after "integer" or "floor" (bare, it may mean the real root).
fn isqrtOf(s: []const u8, i: usize, integral: bool) ?Two {
    var j: usize = undefined;
    var call = false;
    if (seq(s, i, &.{"isqrt("})) |k| {
        j = k;
        call = true;
    } else if (integral) {
        j = seq(s, i, &.{ "sqrt", "of" }) orelse seq(s, i, &.{ "square", "root", "of" }) orelse return null;
        j = skipDef(s, spaces(s, j) orelse return null);
    } else return null;
    const x0 = j;
    while (j < s.len and (std.ascii.isDigit(s[j]) or thousandsAt(s, j))) j += 1;
    if (j == x0 or (j < s.len and isWord(s[j]))) return null;
    const x1 = j;
    if (call) {
        if (j >= s.len or s[j] != ')') return null;
        j += 1;
    }
    const e = trailEnd(s, j) orelse return null;
    return .{ .parts = .{ .{ x0, x1 }, undefined }, .end = e };
}

/// The longest string an edit distance takes as written (the frontend refuses long ones on cost anyway).
pub const max_edit = 32;

/// `Levenshtein [distance]` or `edit distance`, [between|of|from] "A" and|to "B": quoted literals, not prose words.
fn editDistance(s: []const u8, i: usize) ?Two {
    var j = seq(s, i, &.{ "levenshtein", "distance" }) orelse seq(s, i, &.{ "edit", "distance" }) orelse
        seq(s, i, &.{"levenshtein"}) orelse return null;
    j = spaces(s, j) orelse return null;
    for ([_][]const u8{ "between", "of", "from" }) |w| if (seq(s, j, &.{w})) |k| {
        j = spaces(s, k) orelse return null;
        break;
    };
    const x = quoted(s, j) orelse return null;
    var k = spaces(s, x.end) orelse return null;
    k = seq(s, k, &.{"and"}) orelse seq(s, k, &.{"to"}) orelse return null;
    k = spaces(s, k) orelse return null;
    const y = quoted(s, k) orelse return null;
    const e = trailEnd(s, y.end) orelse return null;
    return .{ .parts = .{ x.text, y.text }, .end = e };
}

/// A quoted literal at `i` ('…', "…", ‘…’ or “…”): its contents, printable ASCII without quotes or a backslash.
fn quoted(s: []const u8, i: usize) ?struct { text: [2]usize, end: usize } {
    const marks = [_][2][]const u8{ .{ "\"", "\"" }, .{ "'", "'" }, .{ "“", "”" }, .{ "‘", "’" } };
    for (marks) |q| if (std.mem.startsWith(u8, s[i..], q[0])) {
        const from = i + q[0].len;
        const close = std.mem.indexOfPos(u8, s, from, q[1]) orelse return null;
        if (close == from or close - from > max_edit) return null;
        for (s[from..close]) |c| if (c < 0x20 or c > 0x7e or c == '"' or c == '\'' or c == '\\') return null;
        return .{ .text = .{ from, close }, .end = close + q[1].len };
    };
    return null;
}

/// Past one or more spaces at `i`; null without one.
fn spaces(s: []const u8, i: usize) ?usize {
    var j = i;
    while (j < s.len and isSpace(s[j])) j += 1;
    return if (j > i) j else null;
}

/// A sentence end at `k`, or after one trailer word ("exactly").
fn trailEnd(s: []const u8, k: usize) ?usize {
    if (endAt(s, k)) |e| return e;
    const j = spaces(s, k) orelse return null;
    for (trail_words) |w| if (seq(s, j, &.{w})) |t| if (endAt(s, t)) |e| return e;
    return null;
}

/// `multiply A=x B=y`: two definitions side by side.
fn multiplyDefs(s: []const u8, i: usize) ?Two {
    var j = seq(s, i, &.{"multiply"}) orelse return null;
    const after = j;
    while (j < s.len and isSpace(s[j])) j += 1;
    if (j == after) return null;
    const x0 = skipDef(s, j);
    if (x0 == j) return null;
    var k = x0 + 1;
    while (k < s.len) : (k += 1) {
        if (!isSpace(s[k])) continue;
        var m = k;
        while (m < s.len and isSpace(s[m])) m += 1;
        const y = skipDef(s, m);
        if (y == m) continue;
        const e = lazyEnd(s, y) orelse return null;
        return .{ .parts = .{ .{ x0, k }, .{ y, e.text } }, .end = e.end };
    }
    return null;
}

/// The shortest text from `i` (at least one byte) followed by an optional trailer word and a sentence end.
fn lazyEnd(s: []const u8, i: usize) ?struct { text: usize, end: usize } {
    var k = i + 1;
    while (k <= s.len) : (k += 1) {
        var j = k;
        while (j < s.len and isSpace(s[j])) j += 1;
        if (j > k) for (trail_words) |w| if (seq(s, j, &.{w})) |t| if (endAt(s, t)) |e| return .{ .text = k, .end = e };
        if (endAt(s, k)) |e| return .{ .text = k, .end = e };
    }
    return null;
}

fn endAt(s: []const u8, k: usize) ?usize {
    var j = k;
    while (j < s.len and isSpace(s[j])) j += 1;
    if (j == s.len) return j;
    if (std.mem.indexOfScalar(u8, ".;:,", s[j]) != null and (j + 1 == s.len or isSpace(s[j + 1]))) {
        return if (j + 1 < s.len) j + 2 else j + 1;
    }
    return null;
}

/// A thousands comma at `j`: a digit before it, exactly three digits after it.
pub fn thousandsAt(s: []const u8, j: usize) bool {
    if (j == 0 or j + 4 > s.len or s[j] != ',' or !std.ascii.isDigit(s[j - 1])) return false;
    for (s[j + 1 .. j + 4]) |c| if (!std.ascii.isDigit(c)) return false;
    return j + 4 == s.len or !std.ascii.isDigit(s[j + 4]);
}

/// "Compute D*349, D*350": a phrase ending in a comma before another candidate is a list, not one computation.
pub fn listed(rest: []const u8, ph: Phrase) bool {
    if (ph.end < 2 or rest[ph.end - 2] != ',') return false;
    const w = rest[ph.end - 2 .. @min(rest.len, ph.end + 2)];
    var j: usize = 1;
    while (j < w.len and isSpace(w[j])) j += 1;
    if (j < w.len and std.ascii.isDigit(w[j])) return true;
    if (j < w.len and std.ascii.isUpper(w[j])) {
        j += 1;
        if (j < w.len and isWord(w[j])) j += 1;
        while (j < w.len and isSpace(w[j])) j += 1;
        return j < w.len and (w[j] == '*' or w[j] == 0xC3);
    }
    return false;
}
