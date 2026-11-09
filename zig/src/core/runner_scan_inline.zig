//! Inline triggers: a whole arithmetic expression or a modular step ending a reply line at `=`, as a cell's code.
const std = @import("std");
const Allocator = std.mem.Allocator;

/// Whether `word` ends in a TeX command (`\times`, `\bmod`, `\cdot`) or `times`: an operator before the expression.
fn operatorWord(word: []const u8) bool {
    var k = word.len;
    while (k > 0 and std.ascii.isAlphabetic(word[k - 1])) k -= 1;
    if (k == word.len) return false;
    if (k > 0 and word[k - 1] == '\\') return true;
    return std.ascii.eqlIgnoreCase(word[k..], "times");
}

/// A whole arithmetic expression ending the line at `=` or `= ` (no comparison, word, gap; chain only with `chain`).
pub fn inlineExpr(line: []const u8, arith: bool, chain: bool) ?struct { expr: []const u8, space: bool, call: bool = false } {
    var t = line;
    var space = true;
    if (std.mem.endsWith(u8, t, " ")) {
        t = t[0 .. t.len - 1];
        space = false;
    }
    if (t.len < 4 or t[t.len - 1] != '=') return null;
    const before = t[0 .. t.len - 1];
    // chain (opt-in): only the text after the last `=`, when that `=` ends a statement of its own (`chained` below)
    var seg_from: usize = 0;
    if (std.mem.lastIndexOfScalar(u8, before, '=')) |k| {
        if (!chain or (k > 0 and before[k - 1] == '=')) return null;
        seg_from = k + 1;
    }
    if (std.mem.indexOfAny(u8, before, "<>!") != null) return null;
    if (arith and seg_from == 0) if (gcdCall(before)) |g| return .{ .expr = g, .space = space, .call = true };
    // live R4 d_mirror hex-a and hex: `0xABCDEF0123456789 =`, the whole conversion as one Python literal
    if (arith and seg_from == 0) if (hexLiteral(before)) |h| return .{ .expr = h, .space = space, .call = true };
    var start = before.len;
    var period_stop = false;
    while (start > seg_from) {
        const b = before[start - 1];
        // chain: `… = 5. 3*4 =`, a full stop before a space ends the sentence, not a decimal
        if (chain and b == '.' and start < before.len and (before[start] == ' ' or before[start] == '\t')) {
            period_stop = true;
            break;
        }
        if (std.ascii.isDigit(b) or std.mem.indexOfScalar(u8, " \t+-*/^().", b) != null) {
            start -= 1;
        } else if (arith and b == ',' and start >= 2 and std.ascii.isDigit(before[start - 2]) and start + 3 <= before.len and
            allDigits(before[start .. start + 3]))
        {
            start -= 1; // a thousands comma; `thousands` checks the whole number below
        } else if (start >= 2 and before[start - 2] == 0xC3 and (b == 0x97 or b == 0xB7)) {
            start -= 2; // × ÷
        } else if (start >= 3 and before[start - 3] == 0xE2 and before[start - 2] == 0x88 and b == 0x92) {
            start -= 3; // −
        } else break;
    }
    const lead = before[0..start];
    const expr = std.mem.trim(u8, before[start..], " \t");
    if (expr.len < 3 or expr.len > 120 or std.mem.indexOf(u8, expr, "..") != null) return null;
    if (lead.len > 0) {
        const stop = lead[lead.len - 1];
        // φ(2^18): a parenthesis glued to a non-ASCII letter is a function call (live bn-011 got 262144 for φ)
        if (stop >= 0x80 and before[start] == '(') return null;
        // a word glued to the expression (x3, f(2), $5) or a list (1673, +10) is not a whole expression
        if (stop == ',' or stop == '$' or stop == '%' or stop == '_' or std.ascii.isAlphanumeric(stop)) {
            if (before[start] != ' ' and before[start] != '\t') return null;
            if (stop == ',') return null;
        }
        // after a word, an unspaced "12-20" is a range ("positions 12-20 ="), not a subtraction
        const word = std.mem.trimEnd(u8, lead, " \t");
        // the modulus of `A mod M` or `A % M` is not the whole expression (live bn-011: `3^400 mod 10^18 =`)
        if (std.mem.endsWith(u8, word, "%") or (word.len >= 3 and std.ascii.eqlIgnoreCase(word[word.len - 3 ..], "mod") and
            (word.len == 3 or !std.ascii.isAlphanumeric(word[word.len - 4])))) return null;
        // nor is the right operand of a TeX or worded operator (SparkDash: `\(4 \times 49 - 11 =` got 38, `\bmod` 9563)
        if (operatorWord(word)) return null;
        // `1, 234 + 5 =`: the expression is the next item of a number list
        if (word.len >= 2 and word[word.len - 1] == ',' and std.ascii.isDigit(word[word.len - 2])) return null;
        // `Add 11,529,600 +1 =` adds to a running total the model holds; a lone operand (`subtract 3^67 =`) is fine
        if (runningOperand(std.mem.trimEnd(u8, word, ":")) and topSum(expr)) return null;
        if (word.len > 0 and std.ascii.isAlphabetic(word[word.len - 1])) {
            for (1..expr.len - 1) |k| {
                if (expr[k] == '-' and std.ascii.isDigit(expr[k - 1]) and std.ascii.isDigit(expr[k + 1])) return null;
            }
        }
    }
    const first = expr[0];
    if (!(std.ascii.isDigit(first) or first == '(' or first == '-')) return null;
    if (first == '-' and std.mem.trimEnd(u8, lead, " \t").len > 0 and !std.mem.endsWith(u8, std.mem.trimEnd(u8, lead, " \t"), ":")) return null;
    const last = expr[expr.len - 1];
    if (!(std.ascii.isDigit(last) or last == ')')) return null;
    // a comma inside an open ( [ { is a tuple, list or call argument; else every grouped number must read as thousands
    if (std.mem.indexOfScalar(u8, expr, ',') != null and (opensBracket(lead) or !thousands(expr))) return null;
    // integers only with commas, so the value is CPython's: `/` and a point would be floats there
    if (std.mem.indexOfScalar(u8, expr, ',') != null and
        (std.mem.indexOfAny(u8, expr, "./") != null or std.mem.indexOf(u8, expr, "÷") != null)) return null;
    // a cell only the chain knob opens: integers only, and its `=` statement must stand alone
    if ((seg_from > 0 or period_stop) and chainUnsafe(expr)) return null;
    if (seg_from > 0 and (!clauseBreak(before[seg_from..start]) or chainContext(lead))) return null;
    var depth: i32 = 0;
    var ops: u32 = 0;
    var prev: enum { none, num, op, open, close } = .none;
    var gap = false;
    var i: usize = 0;
    while (i < expr.len) : (i += 1) {
        const c = expr[i];
        if (c == ' ' or c == '\t') {
            gap = true;
            continue;
        }
        defer gap = false;
        if (std.ascii.isDigit(c) or c == '.' or c == ',') {
            if ((prev == .num and gap) or prev == .close) return null;
            prev = .num;
        } else if (c == '(') {
            if (prev == .num or prev == .close) return null;
            depth += 1;
            prev = .open;
        } else if (c == ')') {
            depth -= 1;
            if (depth < 0 or prev == .op or prev == .open) return null;
            prev = .close;
        } else {
            const n: usize = if (c == 0xC3) 2 else if (c == 0xE2) 3 else 1;
            const unary = c == '-' and (prev == .none or prev == .open or prev == .op);
            const power = c == '*' and i + 1 < expr.len and expr[i + 1] == '*';
            if (!unary) {
                if (prev != .num and prev != .close) return null;
                ops += 1;
            }
            i += n - 1;
            if (power) i += 1;
            prev = .op;
        }
    }
    if (depth != 0 or ops == 0) return null;
    return .{ .expr = expr, .space = space };
}

/// A modular step closing a line: `A` as the model wrote it, its literal modulus, and the space flag as inlineExpr.
pub const Mod = struct {
    left: []const u8,
    modulus: []const u8,
    space: bool,

    /// `A mod M` in plain digits for the cell (the graph binder's math syntax; `%` there would read as a percent).
    pub fn code(m: Mod, a: Allocator, out: *std.ArrayList(u8)) Allocator.Error!void {
        var i: usize = 0;
        while (i < m.left.len) : (i += 1) {
            if (std.mem.startsWith(u8, m.left[i..], "×")) {
                try out.append(a, '*');
                i += "×".len - 1;
            } else if (std.mem.startsWith(u8, m.left[i..], "−")) {
                try out.append(a, '-');
                i += "−".len - 1;
            } else if (m.left[i] != ',') try out.append(a, m.left[i]);
        }
        try out.appendSlice(a, " mod ");
        if (std.mem.indexOfScalar(u8, m.modulus, 'e')) |e| {
            try out.print(a, "({s}*10^{s})", .{ m.modulus[0..e], m.modulus[e + 1 ..] });
        } else for (m.modulus) |c| if (c != ',') try out.append(a, c);
    }
};

/// `A mod M =` or `A % M =` ending a line, after any earlier `=`: A over `* × ^ **`, M a nonzero literal; no names.
pub fn inlineMod(line: []const u8) ?Mod {
    var t = line;
    var space = true;
    if (std.mem.endsWith(u8, t, " ")) {
        t = t[0 .. t.len - 1];
        space = false;
    }
    if (t.len < 6 or t[t.len - 1] != '=') return null;
    const before = t[0 .. t.len - 1];
    if (std.mem.indexOfScalar(u8, "=<>!", before[before.len - 1]) != null) return null;
    var seg = before;
    if (std.mem.lastIndexOfScalar(u8, before, '=')) |k| {
        if (k > 0 and std.mem.indexOfScalar(u8, "=<>!", before[k - 1]) != null) return null;
        seg = before[k + 1 ..];
    }
    const s = std.mem.trimEnd(u8, seg, " \t");
    var m_start = s.len;
    while (m_start > 0 and (std.ascii.isDigit(s[m_start - 1]) or std.mem.indexOfScalar(u8, ",^*e", s[m_start - 1]) != null))
        m_start -= 1;
    const modulus = s[m_start..];
    if (!literalModulus(modulus)) return null;
    var rest = std.mem.trimEnd(u8, s[0..m_start], " \t");
    if (std.mem.endsWith(u8, rest, "%")) {
        rest = rest[0 .. rest.len - 1];
    } else if (std.mem.endsWith(u8, rest, "mod") and rest.len > 3 and (rest[rest.len - 4] == ' ' or rest[rest.len - 4] == '\t')) {
        rest = rest[0 .. rest.len - 3];
    } else return null;
    const left_text = std.mem.trimEnd(u8, rest, " \t");
    var start = left_text.len;
    while (start > 0) {
        const b = left_text[start - 1];
        if (std.ascii.isDigit(b) or std.mem.indexOfScalar(u8, " \t*^()+-,", b) != null) {
            start -= 1;
        } else if (start >= 2 and left_text[start - 2] == 0xC3 and b == 0x97) {
            start -= 2; // ×
        } else if (start >= 3 and left_text[start - 3] == 0xE2 and left_text[start - 2] == 0x88 and b == 0x92) {
            start -= 3; // −
        } else break;
    }
    const lead = left_text[0..start];
    const left = std.mem.trim(u8, left_text[start..], " \t");
    if (left.len == 0 or left.len > 120 or !grouped(left)) return null;
    if (!(std.ascii.isDigit(left[0]) or left[0] == '(')) return null;
    if (!(std.ascii.isDigit(left[left.len - 1]) or left[left.len - 1] == ')')) return null;
    if (lead.len > 0) {
        const stop = lead[lead.len - 1];
        // glued to a word or a decimal (x3, 3.5), or after an operator, comparison or symbol: not a whole operand
        if ((std.ascii.isAlphanumeric(stop) or stop == '_' or stop == '$' or stop == '.') and
            left_text[start] != ' ' and left_text[start] != '\t') return null;
        const word = std.mem.trimEnd(u8, lead, " \t");
        if (word.len > 0 and (word[word.len - 1] >= 0x80 or std.mem.indexOfScalar(u8, ",/%<>!", word[word.len - 1]) != null))
            return null;
    }
    var depth: u32 = 0;
    var prev: enum { none, num, op, open, close } = .none;
    var gap = false;
    var i: usize = 0;
    while (i < left.len) : (i += 1) {
        const c = left[i];
        if (c == ' ' or c == '\t') {
            gap = true;
            continue;
        }
        defer gap = false;
        if (std.ascii.isDigit(c) or c == ',') {
            if ((prev == .num and gap) or prev == .close) return null;
            prev = .num;
        } else if (c == '(') {
            if (prev == .num or prev == .close) return null;
            depth += 1;
            prev = .open;
        } else if (c == ')') {
            if (depth == 0 or prev == .op or prev == .open) return null;
            depth -= 1;
            prev = .close;
        } else {
            const n: usize = if (c == 0xC3) 2 else if (c == 0xE2) 3 else 1;
            // which side `mod` takes in `2 + 3 mod 7` is a convention; inside parentheses a sum is one operand
            if ((c == '+' or c == '-' or c == 0xE2) and depth == 0) return null;
            if (prev != .num and prev != .close) return null;
            if (c == '*' and i + 1 < left.len and left[i + 1] == '*') i += 1;
            i += n - 1;
            prev = .op;
        }
    }
    if (depth != 0 or prev == .op) return null;
    return .{ .left = left, .modulus = modulus, .space = space };
}

/// A stacked column at a lone `=`: `first` a literal, `second` an operator (`+ - * ×`) and a literal, as `x op y`.
pub const Stacked = struct { x: []const u8, op: u8, y: []const u8, space: bool };

/// `line` only `=` (or `= `) under `first` and `second`; `before`, the line above, may not be one of the column's own.
pub fn stacked(line: []const u8, before: []const u8, first: []const u8, second: []const u8) ?Stacked {
    const l = std.mem.trimStart(u8, line, " \t");
    if (!std.mem.eql(u8, l, "=") and !std.mem.eql(u8, l, "= ")) return null;
    // a longer column or a multiplication table: which lines the `=` sums is the model's layout, not one reading
    if (columnLiteral(before) != null or columnOperand(before) != null) return null;
    const x = columnLiteral(first) orelse return null;
    const o = columnOperand(second) orelse return null;
    return .{ .x = x, .op = o.op, .y = o.y, .space = l.len == 1 };
}

/// A line that is one plain non-negative integer: digits or thousands groups, no leading zero, at most 120 bytes.
fn columnLiteral(line: []const u8) ?[]const u8 {
    const t = std.mem.trim(u8, line, " \t\r");
    if (t.len == 0 or t.len > 120 or (t[0] == '0' and t.len > 1) or !std.ascii.isDigit(t[0])) return null;
    for (t) |c| if (!std.ascii.isDigit(c) and c != ',') return null;
    if (std.mem.indexOfScalar(u8, t, ',')) |first| if (first > 3 or !grouped(t)) return null;
    return t;
}

/// A line that is an operator, then a column literal (`+771,565,…`, `× 48`).
fn columnOperand(line: []const u8) ?struct { op: u8, y: []const u8 } {
    const t = std.mem.trimStart(u8, line, " \t");
    if (std.mem.startsWith(u8, t, "×")) return .{ .op = '*', .y = columnLiteral(t[2..]) orelse return null };
    if (t.len < 2 or std.mem.indexOfScalar(u8, "+-*", t[0]) == null) return null;
    return .{ .op = t[0], .y = columnLiteral(t[1..]) orelse return null };
}

/// Thousands groups only: each `,` has a digit before it and exactly three digits after it.
fn grouped(text: []const u8) bool {
    for (text, 0..) |c, i| if (c == ',') {
        if (i == 0 or !std.ascii.isDigit(text[i - 1]) or i + 4 > text.len) return false;
        for (text[i + 1 .. i + 4]) |d| if (!std.ascii.isDigit(d)) return false;
        if (i + 4 < text.len and std.ascii.isDigit(text[i + 4])) return false;
    };
    return true;
}

/// A `.`, `;`, `?` or `:` ending a clause between an earlier `=` and the expression: not that statement's right side.
fn clauseBreak(gap: []const u8) bool {
    for (gap, 0..) |c, i| {
        if (std.mem.indexOfScalar(u8, ".;?:", c) != null and (i + 1 == gap.len or gap[i + 1] == ' ' or gap[i + 1] == '\t'))
            return true;
    }
    return false;
}

fn powAt(e: []const u8, i: usize) ?usize {
    if (i < e.len and e[i] == '^') return 1;
    if (i + 1 < e.len and e[i] == '*' and e[i + 1] == '*') return 2;
    return null;
}

fn blank(e: []const u8, i: usize) usize {
    var j = i;
    while (j < e.len and (e[j] == ' ' or e[j] == '\t')) j += 1;
    return j;
}

/// Not CPython's integer: a float (`.`, `/`, `÷`, `2^-53`), a zero-led literal (`014`) or a power tower (`3^20^20`).
fn chainUnsafe(e: []const u8) bool {
    if (std.mem.indexOfAny(u8, e, "./") != null or std.mem.indexOf(u8, e, "÷") != null) return true;
    for (e, 0..) |c, i| {
        const after_digit = i > 0 and (std.ascii.isDigit(e[i - 1]) or e[i - 1] == ',');
        if (c == '0' and i + 1 < e.len and std.ascii.isDigit(e[i + 1]) and !after_digit) return true;
    }
    for (0..e.len) |i| {
        const w = powAt(e, i) orelse continue;
        var j = blank(e, i + w);
        if (j < e.len and e[j] == '-') return true;
        const d = j;
        while (j < e.len and std.ascii.isDigit(e[j])) j += 1;
        if (j > d and powAt(e, blank(e, j)) != null) return true;
    }
    return false;
}

fn wordChar(c: u8) bool {
    return std.ascii.isAlphanumeric(c) or c == '_';
}

/// `mod 97 =` or `% 97 =`: an earlier residue chain, where `=` means congruent.
fn residueChain(t: []const u8) bool {
    for (0..t.len) |i| {
        var j: usize = if (t[i] == '%') i + 1 else if (i + 3 <= t.len and std.ascii.eqlIgnoreCase(t[i .. i + 3], "mod") and
            (i == 0 or !wordChar(t[i - 1]))) i + 3 else continue;
        j = blank(t, j);
        const d = j;
        while (j < t.len and (std.ascii.isDigit(t[j]) or std.mem.indexOfScalar(u8, "^*e,", t[j]) != null)) j += 1;
        if (j > d and blank(t, j) < t.len and t[blank(t, j)] == '=') return true;
    }
    return false;
}

/// A name beside an earlier `=` (`a6=7^64`, `= a6^2`): variable or modular bookkeeping, not plain values.
fn nameBeside(t: []const u8) bool {
    for (t, 0..) |c, k| {
        if (c != '=') continue;
        var j = k;
        while (j > 0 and (t[j - 1] == ' ' or t[j - 1] == '\t')) j -= 1;
        var s = j;
        while (s > 0 and wordChar(t[s - 1])) s -= 1;
        if (s < j and (std.ascii.isAlphabetic(t[s]) or t[s] == '_')) return true;
        var r = blank(t, k + 1);
        if (r < t.len and t[r] == '-') r += 1;
        if (r < t.len and (std.ascii.isAlphabetic(t[r]) or t[r] == '_')) return true;
    }
    return false;
}

/// Words before the expression that make it an operand of something else (`Times 10^6 =`, `digits of 3^100 =`).
const chain_words = [_][]const u8{
    "of",         "by",     "times",   "plus", "minus", "add",  "adding", "subtract", "subtracting", "multiply",
    "multiplied", "divide", "divided", "than", "to",    "from", "with",   "into",     "over",
};

/// Whether `w` (lowercase ASCII) occurs in `text`, either case.
fn containsIgnoreCase(text: []const u8, w: []const u8) bool {
    if (text.len < w.len) return false;
    for (0..text.len - w.len + 1) |i| if (std.ascii.eqlIgnoreCase(text[i..][0..w.len], w)) return true;
    return false;
}

/// Before a chained expression: congruence (`≡`, a residue chain), names, digit counts or pieces, operand words.
fn chainContext(lead: []const u8) bool {
    if (std.mem.indexOf(u8, lead, "≡") != null or residueChain(lead) or nameBeside(lead)) return true;
    for ([_][]const u8{ "digit", "group", "concaten", "append", "length" }) |w| {
        if (containsIgnoreCase(lead, w)) return true;
    }
    const word = std.mem.trimEnd(u8, lead, " \t:");
    var s = word.len;
    while (s > 0 and std.ascii.isAlphabetic(word[s - 1])) s -= 1;
    if (s == word.len) return false;
    for (chain_words) |w| if (std.ascii.eqlIgnoreCase(word[s..], w)) return true;
    return false;
}

fn allDigits(text: []const u8) bool {
    for (text) |c| if (!std.ascii.isDigit(c)) return false;
    return text.len > 0;
}

/// Each comma number reads as thousands: 1-3 lead digits, no leading 0, a point only after the last group (not `3,5`).
fn thousands(e: []const u8) bool {
    var i: usize = 0;
    while (i < e.len) {
        if (!std.ascii.isDigit(e[i]) and e[i] != '.') {
            i += 1;
            continue;
        }
        const s = i;
        while (i < e.len and (std.ascii.isDigit(e[i]) or e[i] == '.' or e[i] == ',')) i += 1;
        const num = e[s..i];
        const last = std.mem.lastIndexOfScalar(u8, num, ',') orelse continue;
        const first = std.mem.indexOfScalar(u8, num, ',').?;
        if (first == 0 or first > 3 or num[0] == '0' or !grouped(num)) return false;
        if (std.mem.indexOfScalar(u8, num[0..last], '.') != null) return false;
    }
    return true;
}

/// An unclosed `(`, `[` or `{` in the text before an expression: its commas would be a tuple's, list's or call's.
fn opensBracket(lead: []const u8) bool {
    var depth: usize = 0;
    for (lead) |c| switch (c) {
        '(', '[', '{' => depth += 1,
        ')', ']', '}' => depth -|= 1,
        else => {},
    };
    return depth > 0;
}

/// A `+` or binary `-` outside parentheses: the expression is a sum or difference.
fn topSum(e: []const u8) bool {
    var depth: usize = 0;
    var operand = false;
    var i: usize = 0;
    while (i < e.len) : (i += 1) {
        const c = e[i];
        if (c == '(') depth += 1 else if (c == ')') depth -|= 1;
        const minus = c == '-' or std.mem.startsWith(u8, e[i..], "−");
        if (depth == 0 and (c == '+' or minus) and operand) return true;
        if (std.ascii.isDigit(c) or c == ')') operand = true else if (c != ' ' and c != '\t') operand = false;
    }
    return false;
}

/// A lead ending in an operator word (`Add`, `minus`): the expression is an operand of a value the model holds.
fn runningOperand(word: []const u8) bool {
    for ([_][]const u8{ "add", "adding", "plus", "minus", "subtract", "subtracting" }) |w| {
        if (word.len >= w.len and std.ascii.eqlIgnoreCase(word[word.len - w.len ..], w) and
            (word.len == w.len or !std.ascii.isAlphanumeric(word[word.len - w.len - 1]))) return true;
    }
    return false;
}

/// `1119 + 392`: two plain non-negative integers added, nothing else (after a join cue it may be a concatenation).
pub fn twoInts(e: []const u8) bool {
    const plus = std.mem.indexOfScalar(u8, e, '+') orelse return false;
    for ([_][]const u8{ e[0..plus], e[plus + 1 ..] }) |side| {
        const t = std.mem.trim(u8, side, " \t");
        if (t.len == 0) return false;
        for (t) |c| if (!std.ascii.isDigit(c) and c != ',') return false;
    }
    return true;
}

/// `gcd(a, b)` on plain non-negative literals ending the text, as its own phrase (not `2 * gcd(…)`, `math.gcd`).
fn gcdCall(before: []const u8) ?[]const u8 {
    const s = std.mem.trimEnd(u8, before, " \t");
    if (!std.mem.endsWith(u8, s, ")")) return null;
    const at = std.mem.lastIndexOf(u8, s, "gcd(") orelse return null;
    if (at > 0) {
        if (s[at - 1] != ' ' and s[at - 1] != '\t') return null;
        const word = std.mem.trimEnd(u8, s[0..at], " \t");
        if (word.len > 0) {
            const w = word[word.len - 1];
            // an operand of a bigger expression, of `mod` or of a running value
            if (std.ascii.isDigit(w) or w >= 0x80 or std.mem.indexOfScalar(u8, "+-*/^%(,.", w) != null) return null;
            if (std.mem.endsWith(u8, word, "mod") and (word.len == 3 or !std.ascii.isAlphanumeric(word[word.len - 4])))
                return null;
            if (runningOperand(std.mem.trimEnd(u8, word, ":"))) return null;
        }
    }
    const args = s[at + 4 .. s.len - 1];
    const comma = std.mem.indexOfScalar(u8, args, ',') orelse return null;
    const y = args[comma + 1 ..];
    for ([_][]const u8{ args[0..comma], if (std.mem.startsWith(u8, y, " ")) y[1..] else y }) |n| {
        if (n.len > 60 or !allDigits(n) or (n[0] == '0' and n.len > 1)) return null;
    }
    return s[at..];
}

/// A bare `0x…` literal of 2 to 100 hex digits ending the text, as its own word (not `#0xFF`, `-0xFF`, `mod 0xFF`).
fn hexLiteral(before: []const u8) ?[]const u8 {
    const s = std.mem.trimEnd(u8, before, " \t");
    var at = s.len;
    while (at > 0 and std.ascii.isHex(s[at - 1])) at -= 1;
    if (s.len - at < 2 or s.len - at > 100 or at < 2 or s[at - 2] != '0' or (s[at - 1] != 'x' and s[at - 1] != 'X'))
        return null;
    const from = at - 2;
    if (from > 0 and s[from - 1] != ' ' and s[from - 1] != '\t') return null;
    const word = std.mem.trimEnd(u8, s[0..from], " \t");
    if (word.len > 0) {
        const w = word[word.len - 1];
        // an operand of a bigger expression (live 0633: `0xFFFFFFFFFFFFFFFF - 0xFEDC… =`), of mod or of a total
        if (std.ascii.isDigit(w) or w >= 0x80 or std.mem.indexOfScalar(u8, "+-*/^%(,&|<>~", w) != null) return null;
        if (std.mem.endsWith(u8, word, "mod") and (word.len == 3 or !std.ascii.isAlphanumeric(word[word.len - 4])))
            return null;
        if (runningOperand(std.mem.trimEnd(u8, word, ":"))) return null;
    }
    return s[from..];
}

/// A nonzero literal modulus: digits, `b^k`, `b**k`, `dek` (1e9), or thousands groups.
fn literalModulus(m: []const u8) bool {
    // a zero or a leading zero (0, 007, 0^3, 0e5, 0,000): Python raises or reads it otherwise
    if (m.len == 0 or !std.ascii.isDigit(m[0]) or m[0] == '0') return false;
    if (std.mem.indexOfScalar(u8, m, ',') != null) {
        if (std.mem.indexOfAny(u8, m, "^*e") != null) return false;
        const first = std.mem.indexOfScalar(u8, m, ',').?;
        return first <= 3 and grouped(m);
    }
    const sep = std.mem.indexOfAny(u8, m, "^*e") orelse return true;
    const width: usize = if (std.mem.startsWith(u8, m[sep..], "**")) 2 else if (m[sep] == '*') return false else 1;
    const exponent = m[sep + width ..];
    if (exponent.len == 0 or exponent.len > 3) return false;
    for (m[0..sep]) |c| if (!std.ascii.isDigit(c)) return false;
    for (exponent) |c| if (!std.ascii.isDigit(c)) return false;
    return !(exponent[0] == '0' and exponent.len > 1);
}
