//! Answer-commit (PREFIX_COMMIT): an owned value of the task's own expression closes the think and is the answer.
const std = @import("std");
const scan = @import("runner_scan.zig");
const value = @import("runner_output.zig").value;
const Allocator = std.mem.Allocator;
const log = std.log.scoped(.runner);

/// A request's commit settings (Request.run_commit); empty `close`: never fires.
pub const Ask = struct {
    close: []const u32 = &.{}, // the think close the stream writes (server.think_close)
    task: []const u8 = "", // the last user message's text
    eos: ?u32 = null, // the token that ends the reply after the answer
    max_new: u32 = 0, // the reply's token limit
    answer_max: u32 = 0, // tokens after the think close at most (0: max_new alone)
    tools: bool = false, // the request offers tools: the answer may be a call, so never
};

/// A session's commit state: its ask, the think state of the token fed last, and whether it fired.
pub const State = struct {
    ask: Ask = .{},
    think: bool = false,
    fired: bool = false,
};

/// Feed: the token's think state kept for a trigger; true once fired (the scanner sees nothing after the answer).
pub fn skip(st: *State, think_open: bool) bool {
    st.think = think_open;
    return st.fired;
}

/// The firing's write as it lands: the close, then the value (the answer field's bytes), then the end.
pub const Record = struct {
    cell: u32,
    want: []u32, // close ++ value ++ eos (gpa)
    close_len: u32,
    value_len: u32,
    landed: u32 = 0, // of `want`, committed in order (fewer: another token came, or the reply ended)
    close_token: u32 = 0, // the reply's tokens before the close
    token: u32 = 0, // the reply's tokens before the value
    at: u64 = 0, // the content field's bytes before the value
    value: std.ArrayList(u8) = .empty,
    cut: bool = false,

    /// A committed token after the cell's own write; `content_bytes`/`tokens` count those before it.
    pub fn land(r: *Record, gpa: Allocator, id: u32, bytes: []const u8, content_bytes: u64, tokens: u32) Allocator.Error!void {
        if (r.cut or r.landed >= r.want.len) return;
        if (r.want[r.landed] != id) {
            r.cut = true; // nothing after another token lands
            return;
        }
        if (r.landed == 0) r.close_token = tokens;
        if (r.landed == r.close_len) {
            r.token = tokens;
            r.at = content_bytes;
        }
        if (r.landed >= r.close_len and r.landed < r.close_len + r.value_len) try r.value.appendSlice(gpa, bytes);
        r.landed += 1;
    }

    pub fn deinit(r: *Record, gpa: Allocator) void {
        gpa.free(r.want);
        r.value.deinit(gpa);
    }
};

/// The receipt's `"commit"` object: the cell whose span is the task's expression, its answer as landed, where.
pub fn write(o: *std.ArrayList(u8), gpa: Allocator, cells: anytype, r: Record) Allocator.Error!void {
    const receipt = @import("runner_receipt.zig");
    try o.print(gpa, ",\"commit\":{{\"cell\":{d},\"why\":\"whole-task\",", .{r.cell});
    try receipt.text(o, gpa, "span", cells[r.cell].code);
    try receipt.text(o, gpa, "value", r.value.items);
    try o.print(gpa, "\"field\":\"content\",\"at\":{d},\"token\":{d},\"close_token\":{d},\"ids\":{d},\"landed\":{d},\"closed\":{},\"ended\":{}}}", .{ r.at, r.token, r.close_token, r.close_len + r.value_len + 1, r.landed, r.landed >= r.close_len, r.landed == r.close_len + r.value_len + 1 });
}

/// `ids` (the owned cell's result, already recorded) then the commit's tokens when it fires (`ids` freed or returned).
pub fn after(s: anytype, cell: anytype, w: scan.Trigger, index: u32, ids: []u32) Allocator.Error![]u32 {
    const gpa = s.r.gpa;
    if (!s.r.cfg.commit or s.commit.fired or !try fires(gpa, s, cell, w)) return ids;
    const tx = s.r.text.?;
    const ask = s.commit.ask;
    const v = value(cell.result.?.out);
    const vid = try tx.encode(tx.ctx, gpa, v);
    defer gpa.free(vid);
    // after the think end: at most the close's own tokens, the value and the end
    if (vid.len > s.r.cfg.max_tokens or (ask.answer_max > 0 and ask.close.len + vid.len + 1 > ask.answer_max)) return ids;
    // the stream forces at most its room less one: every token must fit, or the answer would be a cut number
    if (@as(u64, s.rec.tokens) + ids.len + ask.close.len + vid.len + 2 > ask.max_new) return ids;
    const tail = try std.mem.concat(gpa, u32, &.{ ask.close, vid, &.{ask.eos.?} });
    errdefer gpa.free(tail);
    const all = try std.mem.concat(gpa, u32, &.{ ids, tail });
    errdefer gpa.free(all);
    s.rec.commit = .{ .cell = index, .want = tail, .close_len = @intCast(ask.close.len), .value_len = @intCast(vid.len) };
    gpa.free(ids);
    s.commit.fired = true;
    if (s.forcing) |*f| f.left += tail.len;
    log.info("reply {d} cell {d}: commit, the think closes and {d} value tokens are the answer", .{ s.id, index, vid.len });
    return all;
}

/// Every guard but the room: only what the oracle checks (the task's one expression as the span, an integer value).
fn fires(gpa: Allocator, s: anytype, cell: anytype, w: scan.Trigger) Allocator.Error!bool {
    const ask = s.commit.ask;
    if (ask.close.len == 0 or ask.eos == null or ask.tools or !s.rec.on) return false;
    // inside reasoning, after a closed cell: never a trace envelope, an EOS hand-off or an open fence
    if (!s.commit.think or w.trace or w.terminal or cell.stated != null or s.scan.state == .fence) return false;
    const res = cell.result orelse return false;
    if (res.status != .ok or !integer(value(res.out))) return false;
    const task = try expression(gpa, ask.task) orelse return false;
    defer gpa.free(task);
    const span = try spanOf(gpa, cell.code, cell.form) orelse return false;
    defer gpa.free(span);
    return same(task, span);
}

/// A decimal integer as Python prints one: an optional minus, digits, no leading zero.
pub fn integer(v: []const u8) bool {
    const d = if (std.mem.startsWith(u8, v, "-")) v[1..] else v;
    if (d.len == 0 or (d.len > 1 and d[0] == '0') or std.mem.eql(u8, v, "-0")) return false;
    for (d) |c| if (!std.ascii.isDigit(c)) return false;
    return true;
}

/// The one closed literal expression `text` states, normalized, when every digit in `text` is in it; else null.
pub fn expression(a: Allocator, text: []const u8) Allocator.Error!?[]u8 {
    const words = try spoken(a, text);
    defer a.free(words);
    var found: ?[]u8 = null;
    errdefer if (found) |f| a.free(f);
    var i: usize = 0;
    var digits: usize = 0; // digits outside the expression found
    while (i < words.len) {
        if (!inRun(words[i])) {
            i += 1;
            continue;
        }
        var j = i;
        while (j < words.len and inRun(words[j])) j += 1;
        const run = std.mem.trim(u8, words[i..j], " \t\r\n,");
        i = j;
        var n: usize = 0;
        for (run) |c| n += @intFromBool(std.ascii.isDigit(c));
        if (n == 0) continue;
        const e = try normalize(a, run, true) orelse {
            digits += n;
            continue;
        };
        if (found != null or !binary(e)) {
            a.free(e);
            digits += n;
            continue;
        }
        found = e;
    }
    if (digits > 0) {
        if (found) |f| a.free(f);
        return null;
    }
    return found;
}

fn inRun(c: u8) bool {
    return std.ascii.isDigit(c) or std.mem.indexOfScalar(u8, " \t+-*/%()^,_", c) != null;
}

/// `text` with the spelled operators as symbols (`divided by` `/`, `mod` `%`, `times` `*`, `×` `*`, `−` `-`, …).
fn spoken(a: Allocator, text: []const u8) Allocator.Error![]u8 {
    const subs = [_][2][]const u8{
        .{ "divided by", "/" },
        .{ "mod", "%" },
        .{ "times", "*" },
        .{ "plus", "+" },
        .{ "minus", "-" },
        .{ "\u{00d7}", "*" },
        .{ "\u{00b7}", "*" },
        .{ "\u{00f7}", "/" },
        .{ "\u{2212}", "-" },
    };
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    var i: usize = 0;
    next: while (i < text.len) {
        for (subs) |sub| if (i + sub[0].len <= text.len and std.ascii.eqlIgnoreCase(text[i .. i + sub[0].len], sub[0])) {
            const word = std.ascii.isAlphabetic(sub[0][0]);
            const before = i == 0 or !std.ascii.isAlphanumeric(text[i - 1]);
            const end = i + sub[0].len;
            if (!word or (before and (end == text.len or !std.ascii.isAlphanumeric(text[end])))) {
                try out.appendSlice(a, sub[1]);
                i = end;
                continue :next;
            }
        };
        try out.append(a, text[i]);
        i += 1;
    }
    return out.toOwnedSlice(a);
}

/// A cell's span as the expression it computes: inline as written, else its last `print(<expr>)` over literal names.
pub fn spanOf(a: Allocator, code: []const u8, form: anytype) Allocator.Error!?[]u8 {
    if (form == .inline_expr) {
        const words = try spoken(a, code);
        defer a.free(words);
        return normalize(a, words, true);
    }
    var names: std.ArrayList([2][]const u8) = .empty;
    defer names.deinit(a);
    var last: ?[]const u8 = null;
    var it = std.mem.splitScalar(u8, std.mem.trim(u8, code, " \t\r\n"), '\n');
    while (it.next()) |raw| {
        var line = std.mem.trim(u8, raw, " \t\r");
        if (std.mem.startsWith(u8, line, ">>> ")) line = line[4..];
        if (line.len == 0) continue;
        if (last != null) return null; // the printed expression is the cell's last statement
        if (assignment(line)) |as| {
            for (names.items) |n| if (std.mem.eql(u8, n[0], as[0])) return null; // rebound
            try names.append(a, as);
            continue;
        }
        last = if (std.mem.startsWith(u8, line, "print(") and line[line.len - 1] == ')') line[6 .. line.len - 1] else line;
    }
    const e = last orelse return null;
    var sub: std.ArrayList(u8) = .empty;
    defer sub.deinit(a);
    var i: usize = 0;
    while (i < e.len) {
        if (std.ascii.isAlphabetic(e[i]) or e[i] == '_') {
            var j = i;
            while (j < e.len and (std.ascii.isAlphanumeric(e[j]) or e[j] == '_')) j += 1;
            const name = e[i..j];
            const rhs = for (names.items) |n| {
                if (std.mem.eql(u8, n[0], name)) break n[1];
            } else return null; // a call or an unknown name
            try sub.print(a, "({s})", .{rhs});
            i = j;
            continue;
        }
        try sub.append(a, e[i]);
        i += 1;
    }
    return normalize(a, sub.items, false);
}

/// `name = <literal right side>` (no other name on the right).
fn assignment(line: []const u8) ?[2][]const u8 {
    const eq = std.mem.indexOfScalar(u8, line, '=') orelse return null;
    if (eq + 1 < line.len and line[eq + 1] == '=') return null;
    const name = std.mem.trim(u8, line[0..eq], " \t");
    if (name.len == 0 or !(std.ascii.isAlphabetic(name[0]) or name[0] == '_')) return null;
    for (name) |c| if (!(std.ascii.isAlphanumeric(c) or c == '_')) return null;
    const rhs = std.mem.trim(u8, line[eq + 1 ..], " \t");
    for (rhs) |c| if (std.ascii.isAlphabetic(c)) return null;
    return .{ name, rhs };
}

/// A literal-only expression in one spelling (digit groups joined, `**` powers, `(n)` as n); null if not well formed.
pub fn normalize(a: Allocator, src: []const u8, caret: bool) Allocator.Error!?[]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    var i: usize = 0;
    while (i < src.len) {
        const c = src[i];
        if (c == ' ' or c == '\t') {
            i += 1;
        } else if (std.ascii.isDigit(c)) {
            while (i < src.len) {
                if (std.ascii.isDigit(src[i])) {
                    try out.append(a, src[i]);
                    i += 1;
                } else if ((src[i] == ',' and group(src, i + 1)) or
                    (src[i] == '_' and i + 1 < src.len and std.ascii.isDigit(src[i + 1])))
                {
                    i += 1;
                } else break;
            }
        } else if (c == '^') {
            if (!caret) return fail(a, &out);
            try out.appendSlice(a, "**");
            i += 1;
        } else if (std.mem.indexOfScalar(u8, "+-*/%()", c) != null) {
            try out.append(a, c);
            i += 1;
        } else return fail(a, &out);
    }
    const s = try out.toOwnedSlice(a);
    errdefer a.free(s);
    const plain = try unbracket(a, s);
    a.free(s);
    if (!wellFormed(plain)) {
        a.free(plain);
        return null;
    }
    return plain;
}

fn fail(a: Allocator, out: *std.ArrayList(u8)) ?[]u8 {
    out.deinit(a);
    return null;
}

/// Three digits at `i`, then no further digit.
fn group(s: []const u8, i: usize) bool {
    if (i + 3 > s.len) return false;
    for (s[i .. i + 3]) |c| if (!std.ascii.isDigit(c)) return false;
    return i + 3 == s.len or !std.ascii.isDigit(s[i + 3]);
}

/// `(digits)` as `digits`, again until none is left, and brackets around the whole expression dropped.
fn unbracket(a: Allocator, s: []const u8) Allocator.Error![]u8 {
    var cur = try a.dupe(u8, s);
    while (cur.len >= 2 and cur[0] == '(' and closing(cur) == cur.len - 1) {
        const inner = try a.dupe(u8, cur[1 .. cur.len - 1]);
        a.free(cur);
        cur = inner;
    }
    while (true) {
        var i: usize = 0;
        const hit = while (i < cur.len) : (i += 1) {
            if (cur[i] != '(') continue;
            var j = i + 1;
            while (j < cur.len and std.ascii.isDigit(cur[j])) j += 1;
            if (j > i + 1 and j < cur.len and cur[j] == ')') break .{ i, j };
        } else null;
        const h = hit orelse return cur;
        const next = try std.mem.concat(a, u8, &.{ cur[0..h[0]], cur[h[0] + 1 .. h[1]], cur[h[1] + 1 ..] });
        a.free(cur);
        cur = next;
    }
}

/// The index of the bracket closing the one `e` opens with (null: unbalanced).
fn closing(e: []const u8) ?usize {
    var depth: usize = 0;
    for (e, 0..) |c, i| switch (c) {
        '(' => depth += 1,
        ')' => {
            if (depth == 0) return null;
            depth -= 1;
            if (depth == 0) return i;
        },
        else => {},
    };
    return null;
}

/// Infix over integers: unary signs, `+ - * / // % **`, balanced brackets.
fn wellFormed(s: []const u8) bool {
    var p: Parse = .{ .s = s };
    return p.expr() and p.i == s.len;
}

const Parse = struct {
    s: []const u8,
    i: usize = 0,
    depth: u32 = 0,

    fn expr(p: *Parse) bool {
        if (!p.unary()) return false;
        while (p.i < p.s.len and std.mem.indexOfScalar(u8, "+-*/%", p.s[p.i]) != null) {
            p.i += 1;
            if (p.i < p.s.len and (p.s[p.i] == '*' or p.s[p.i] == '/') and p.s[p.i - 1] == p.s[p.i]) p.i += 1;
            if (!p.unary()) return false;
        }
        return true;
    }

    fn unary(p: *Parse) bool {
        while (p.i < p.s.len and (p.s[p.i] == '-' or p.s[p.i] == '+')) p.i += 1;
        if (p.i >= p.s.len) return false;
        if (std.ascii.isDigit(p.s[p.i])) {
            while (p.i < p.s.len and std.ascii.isDigit(p.s[p.i])) p.i += 1;
            return true;
        }
        if (p.s[p.i] != '(' or p.depth > 64) return false;
        p.i += 1;
        p.depth += 1;
        if (!p.expr() or p.i >= p.s.len or p.s[p.i] != ')') return false;
        p.depth -= 1;
        p.i += 1;
        return true;
    }
};

/// A binary operator at any depth (not only a signed number).
fn binary(e: []const u8) bool {
    for (e, 0..) |c, i| if (i > 0 and std.mem.indexOfScalar(u8, "+-*/%", c) != null and
        (std.ascii.isDigit(e[i - 1]) or e[i - 1] == ')')) return true;
    return false;
}

/// The same expression, or one sum or product of two operands written the other way round.
pub fn same(task: []const u8, span: []const u8) bool {
    if (std.mem.eql(u8, task, span)) return true;
    const x = split(task) orelse return false;
    const y = split(span) orelse return false;
    return x.op == y.op and std.mem.eql(u8, x.left, y.right) and std.mem.eql(u8, x.right, y.left);
}

const Split = struct { left: []const u8, op: u8, right: []const u8 };

/// `left + right` or `left * right` when that is the only binary operator outside brackets.
fn split(e: []const u8) ?Split {
    var depth: i32 = 0;
    var at: ?usize = null;
    for (e, 0..) |c, i| {
        switch (c) {
            '(' => depth += 1,
            ')' => depth -= 1,
            '+', '-', '*', '/', '%' => if (depth == 0 and i > 0 and (std.ascii.isDigit(e[i - 1]) or e[i - 1] == ')')) {
                if (at != null) return null;
                at = i;
            },
            else => {},
        }
    }
    const k = at orelse return null;
    if (e[k] != '+' and e[k] != '*') return null;
    if (k + 1 < e.len and (e[k + 1] == '*' or e[k + 1] == '/')) return null; // `**`
    return .{ .left = e[0..k], .op = e[k], .right = e[k + 1 ..] };
}

test {
    _ = @import("runner_commit_test.zig");
}
