//! The runner's `trace` form: whole integer computations the reasoning states, and the lines where hand work starts.
const std = @import("std");
const text = @import("runner_trace_text.zig");
const expr = @import("runner_trace_expr.zig");
const value = @import("runner_trace_value.zig");
const checks = @import("runner_trace_check.zig");
const bind = @import("runner_trace_bind.zig");
const Allocator = std.mem.Allocator;
const Expr = expr.Expr;

/// The integer frontend's width: a statement whose syntax bound is wider is never a candidate.
pub const max_digits = 62;

/// A stated computation: Python source for the frontend, the expression shown to the model, divmod's re-shaping.
pub const Found = struct { source: []const u8, display: []const u8, divmod: bool = false, same_line: bool = false, run: []const u8 };

/// A name's binding state; `names` holds the same name with its expression only while exact.
const Bind = struct { state: bind.State, kept: ?Expr = null, deps: []const []const u8 = &.{} };

/// Per reply: the names the reasoning defined and the sources already owned (`keep`); one line's work (`scratch`).
pub const Tracer = struct {
    keep: std.heap.ArenaAllocator,
    scratch: std.heap.ArenaAllocator,
    names: std.ArrayList(expr.Name) = .empty, // what the parser reads: an expression only for an exact name
    binds: std.ArrayList(Bind) = .empty, // index-aligned with `names`
    rules: bind.Rules = .{}, // the recovery rules (all in the runner; tests run each alone)
    fired: std.ArrayList([]const u8) = .empty,
    stated: std.ArrayList([]const u8) = .empty, // names whose first bindable definition was looked at as a statement
    moved: std.ArrayList([]const u8) = .empty, // this line's names that left exact after its statement (scratch)
    repeat: u8 = 0, // times a displayed line may be owned again when the reasoning restates it (TraceAsk.repeat)
    whole: bool = false, // PREFIX_STATED: also comma sentences, gcd(x, y), `L = R.` restated and a hex value

    pub fn init(gpa: Allocator) Tracer {
        return .{ .keep = .init(gpa), .scratch = .init(gpa) };
    }

    pub fn deinit(t: *Tracer) void {
        t.keep.deinit();
        t.scratch.deinit();
    }

    /// A displayed line owned: a restatement (also through names) fires again only `repeat` more times.
    pub fn fire(t: *Tracer, display: []const u8) Allocator.Error!void {
        const a = t.keep.allocator();
        try t.fired.append(a, try a.dupe(u8, display));
    }

    fn firedHas(t: *const Tracer, display: []const u8) bool {
        var n: usize = 0;
        for (t.fired.items) |f| n += @intFromBool(std.mem.eql(u8, f, display));
        return n > t.repeat;
    }

    fn find(t: *const Tracer, n: []const u8) ?usize {
        for (t.names.items, 0..) |m, k| if (std.mem.eql(u8, m.name, n)) return k;
        return null;
    }

    /// Moves `n` to `state`; leaving exact makes what read `n` stale, transitively (restoring `n` restores none).
    fn set(t: *Tracer, n: []const u8, state: bind.State, kept: ?Expr, deps: []const []const u8) Allocator.Error!void {
        const a = t.keep.allocator();
        const k = t.find(n) orelse blk: {
            try t.names.append(a, .{ .name = try a.dupe(u8, n), .expr = null });
            try t.binds.append(a, .{ .state = .tentative });
            break :blk t.names.items.len - 1;
        };
        const was = t.binds.items[k].state;
        if (was == .exact and state != .exact) try t.moved.append(t.scratch.allocator(), t.names.items[k].name);
        var v = kept;
        if (v) |e| if (!(t.binds.items[k].kept != null and t.binds.items[k].kept.?.source.ptr == e.source.ptr)) {
            v = .{ .source = try a.dupe(u8, e.source), .display = try a.dupe(u8, e.display), .bound = e.bound, .ops = e.ops, .constant = e.constant, .run = try a.dupe(u8, e.run) };
        };
        const ds = try a.alloc([]const u8, deps.len);
        for (deps, ds) |d, *o| o.* = try a.dupe(u8, d);
        t.binds.items[k] = .{ .state = state, .kept = if (state == .exact or state == .quarantined) v else null, .deps = ds };
        t.names.items[k].expr = if (state == .exact) v else null;
        if (was != .exact or state == .exact) return;
        for (t.binds.items, t.names.items) |b, m| if (b.state == .exact) for (b.deps) |d| if (std.mem.eql(u8, d, n)) {
            try t.set(m.name, .stale, null, &.{});
            break;
        };
    }

    /// Binding k's dependencies with `more` added, each once.
    fn withDeps(t: *Tracer, k: usize, more: []const []const u8) Allocator.Error![]const []const u8 {
        const a = t.scratch.allocator();
        var all: std.ArrayList([]const u8) = .empty;
        try all.appendSlice(a, t.binds.items[k].deps);
        next: for (more) |d| {
            for (all.items) |x| if (std.mem.eql(u8, x, d)) continue :next;
            try all.append(a, d);
        }
        return all.items;
    }

    fn movedIn(t: *const Tracer, s: []const u8) bool {
        for (t.moved.items) |n| {
            var i: usize = 0;
            while (std.mem.indexOfPos(u8, s, i, n)) |j| : (i = j + 1) {
                if ((j == 0 or !text.isWord(s[j - 1])) and (j + n.len == s.len or !text.isWord(s[j + n.len]))) return true;
            }
        }
        return false;
    }

    fn wasStated(t: *const Tracer, n: []const u8) bool {
        for (t.stated.items) |m| if (std.mem.eql(u8, m, n)) return true;
        return false;
    }

    /// One reasoning line, definitions and requests in source order: its first statement; valid until the next call.
    pub fn line(t: *Tracer, raw: []const u8) Allocator.Error!?Found {
        _ = t.scratch.reset(.retain_capacity);
        const a = t.scratch.allocator();
        const clean = try text.stripNotes(a, raw);
        var define: ?Found = null;
        var heads: std.ArrayList(text.Head) = .empty;
        try text.goalHeads(a, clean, &heads);
        try text.bareHeads(a, clean, t.whole, &heads);
        std.mem.sort(text.Head, heads.items, {}, text.Head.less);
        var first: ?Found = null;
        var asked: []const u8 = ""; // the statement's sentence
        var last_head: usize = 0;
        var read: usize = 0; // definitions before this byte are bound: one after a request never supplies it
        t.moved = .empty;
        for (heads.items) |h| {
            last_head = @max(last_head, h.end);
            try t.definitions(clean, raw, &read, h.end, &define);
            // a name the request read moved later on its line (`Need N mod p. N=…`): neither value is the one meant
            if (first != null and t.movedIn(asked)) first = null;
            if (first != null) continue;
            t.moved = .empty;
            first = try t.statement(clean, h);
            asked = text.sentenceAt(clean, h.end);
        }
        try t.definitions(clean, raw, &read, clean.len, &define);
        if (first != null and t.movedIn(asked)) first = null;
        var tail = last_head;
        if (first == null and t.whole) if (try t.wholeShape(clean)) |w| {
            first = w.found;
            tail = w.end;
        };
        if (first) |*f| {
            f.same_line = text.cueIn(clean[tail..]);
            f.run = try printed(a, f.run);
            return f.*;
        }
        if (define) |*d| d.run = try printed(a, d.run);
        return define;
    }

    fn definitions(t: *Tracer, clean: []const u8, raw: []const u8, from: *usize, upto: usize, define: *?Found) Allocator.Error!void {
        const a = t.keep.allocator();
        var i = from.*;
        defer from.* = @max(from.*, i);
        while (i < upto) {
            // `N += b`, `N *= 2`, `N++`: an update in place moves the name for good
            if (bind.updateAt(clean, i)) |n| {
                try t.set(clean[i .. i + n], .hard, null, &.{});
                i += n;
                continue;
            }
            const d = text.defAt(clean, i) orelse {
                i += 1;
                continue;
            };
            i = d.rhs;
            // `A * B = …` and (R4) `B mod A = …` state a value; any other `Name =` assigns, even binding nothing
            if (!text.assignLeft(clean[0..d.start])) continue;
            if (t.rules.r4 and bind.compoundLeft(clean[0..d.start])) continue;
            const n = clean[d.start..d.name_end];
            try t.sight(clean, raw, d);
            // a name's first wide definition with nothing after it states it, bound or not ("N = A*2^64 + B.")
            const x = (if (text.defLeftOk(clean[0..d.start])) try t.bindable(clean, d) else null) orelse continue;
            if (t.wasStated(n)) continue;
            try t.stated.append(a, try a.dupe(u8, n));
            if (hasArith(x.e.source) and x.e.bound >= 12 and x.e.bound <= max_digits and text.sentenceEndsAt(clean, x.end) and
                !t.firedHas(x.e.display)) define.* = .{ .source = x.e.source, .display = x.e.display, .run = x.e.run };
        }
    }

    /// One `Name = …` moves the binding (T5-BINDING-RECOVERY.md): bind, restate, stay tentative, quarantine or block.
    fn sight(t: *Tracer, clean: []const u8, raw: []const u8, d: text.Def) Allocator.Error!void {
        const n = clean[d.start..d.name_end];
        const k = t.find(n);
        const st: ?bind.State = if (k) |j| t.binds.items[j].state else null;
        if (st == .hard) return;
        const list = if (t.rules.r2) bind.layoutList(clean, d.rhs) else null;
        const end = text.defEnd(clean, list orelse d.rhs) orelse clean.len;
        const txt = clean[d.rhs..if (t.rules.r3) d.rhs + bind.nextDef(clean[d.rhs..end]) else end];
        const doubt = text.approx(clean[d.rhs..charsOn(clean, end, 3)]) or std.mem.indexOfAny(u8, txt, "[]") != null or
            text.hypothetical(text.sentenceAt(clean, d.start));
        const held = st == .exact or st == .quarantined;
        const r = try t.reading(raw, n, txt, doubt, if (held) t.binds.items[k.?].kept else null);
        switch (r) {
            .hard => try t.set(n, .hard, null, &.{}),
            .soft => if (st == null)
                try t.set(n, .tentative, null, &.{})
            else if (st == .exact)
                // R2: an unprovable restatement keeps the value aside (and what it read) until a plain one restates it
                try t.set(n, if (t.rules.r2) .quarantined else .hard, t.binds.items[k.?].kept, t.binds.items[k.?].deps),
            .value => |v| {
                if (held) {
                    const kept = t.binds.items[k.?].kept.?;
                    if (!std.mem.eql(u8, v.e.source, kept.source) and !try value.same(t.scratch.allocator(), v.e.source, kept.source))
                        return t.set(n, .hard, null, &.{});
                    // a proved restatement reads its names as much as the declaration did: any of them moving moves n
                    const ds = try t.withDeps(k.?, v.deps);
                    // R2: only a plain declaration of the same value clears a quarantine
                    if (st == .quarantined and !doubt and !v.restate) return t.set(n, .exact, v.e, ds);
                    return t.set(n, st.?, kept, ds);
                }
                // a hypothesis, a question, an approximation or a layout note never makes a binding
                if (doubt or v.restate) {
                    if (st == null) try t.set(n, .tentative, null, &.{});
                    return;
                }
                // a value after a label or an operator (`Step3 N=…`) binds nothing and blocks the name, as before
                const left = clean[0..d.start];
                if (!text.defLeftOk(left) and !(t.rules.r3 and bind.afterValue(left)))
                    return if (st == null) t.set(n, .hard, null, &.{}) else {};
                // R1: the first exact binding after tentative sightings; a stale name rebinds at a new declaration
                if (st == null or st == .stale or t.rules.r1) try t.set(n, .exact, v.e, v.deps);
            },
        }
    }

    const Reading = union(enum) { hard, soft, value: struct { e: Expr, deps: []const []const u8, restate: bool } };

    /// A right side: one integer (a chain all equal, R3), a doubted or unresolved formula, or a hard block.
    fn reading(t: *Tracer, raw: []const u8, n: []const u8, txt: []const u8, doubt: bool, kept: ?Expr) Allocator.Error!Reading {
        const a = t.scratch.allocator();
        // a doubt ends the value it doubts: "12,000,000,000 + 345,678,901? Or use pattern"
        var v = txt;
        for ([_][]const u8{ "?", "~", "≈" }) |m| if (std.mem.indexOf(u8, v, m)) |j| {
            v = v[0..j];
        };
        var ps: std.ArrayList([]const u8) = .empty;
        if (t.rules.r3) try bind.pieces(v, &ps, a) else try ps.append(a, v);
        var first: ?Expr = null;
        var deps: std.ArrayList([]const u8) = .empty;
        var restate = false; // typed metadata or spaced groups: proves a value equal, never binds one
        var layout = false; // after a metadata word, unparsed pieces describe the layout (`= 192|443|…`)
        var soft = false;
        for (ps.items) |p0| {
            var p = std.mem.trim(u8, p0, " \t");
            if (t.rules.r3) p = bind.unclosed(p);
            if (p.len == 0) continue; // the chain goes on past the right side's end (`Q = (…)/97 = (…`)
            if (t.rules.r2) {
                const m = bind.metadata(p);
                if (m.cut) {
                    restate = true;
                    layout = true;
                    p = m.value;
                    if (p.len == 0) continue;
                }
            }
            const f = bind.formula(p, n);
            if (f.self or f.sci) return .hard;
            var e = try t.parse(p);
            if (e == null and t.rules.r2) if (try bind.spaced(a, p, raw)) |digits| {
                e = try t.parse(digits);
                restate = true;
            };
            // R2: an exact quotient proves a held value restated (`Q = (10^30 - 85)/97`), never binds one
            if (e == null and t.rules.r2 and f.div and f.end == p.len) if (kept) |k| if (try bind.plain(a, p)) |s| {
                if (try value.same(a, s, k.source)) {
                    e = k;
                    restate = true;
                }
            };
            const x = e orelse {
                if (layout) continue;
                // a value that opens a list (`base B=10^9:`) is a sighting; a later declaration binds (R1)
                if (p[p.len - 1] == ':' and !f.pipe) {
                    soft = true;
                    continue;
                }
                // pipes (`D=1|2` may be OR) and other notation: never read; a doubted one only quarantines a held value
                if (f.pipe and !doubt) return .hard;
                if (!doubt and (f.decimal or f.end == 0 or !(f.names or f.div))) return .hard;
                soft = true;
                continue;
            };
            if (first) |e0| {
                if (!std.mem.eql(u8, e0.source, x.source) and !try value.same(a, e0.source, x.source)) return .hard;
            } else first = x;
            // every proved piece's names are dependencies (`B=12345678901=A.` reads A as much as `B=A=…` does)
            var j: usize = 0;
            while (j < p.len) : (j += 1) {
                const l = text.nameLen(p, j);
                if (l == 0 or (j > 0 and text.isWord(p[j - 1])) or (j + l < p.len and text.isWord(p[j + l]))) continue;
                try deps.append(a, p[j .. j + l]);
                j += l;
            }
        }
        if (soft) return .soft;
        const e = first orelse return .soft;
        return .{ .value = .{ .e = e, .deps = deps.items, .restate = restate } };
    }

    /// The value a definition binds and its text's end; null if approximate, bracketed, hypothetical or unparsed.
    fn bindable(t: *Tracer, clean: []const u8, d: text.Def) Allocator.Error!?struct { e: Expr, end: usize } {
        const txt_end = text.defEnd(clean, d.rhs) orelse return null;
        const txt = clean[d.rhs..txt_end];
        if (text.approx(clean[d.rhs..charsOn(clean, txt_end, 3)])) return null;
        if (std.mem.indexOfAny(u8, txt, "[]") != null) return null;
        if (text.hypothetical(text.sentenceAt(clean, d.start))) return null;
        const e = try t.parse(txt) orelse return null;
        return .{ .e = e, .end = txt_end };
    }

    fn statement(t: *Tracer, clean: []const u8, h: text.Head) Allocator.Error!?Found {
        const a = t.scratch.allocator();
        const rest = clean[h.end..];
        const ph = text.phrase(rest, t.whole) orelse return null;
        if (text.approx(rest[0..ph.end])) return null;
        if (text.listed(rest, ph)) return null;
        if (std.mem.indexOfScalar(u8, text.sentenceAt(clean, h.end), '?') != null) return null;
        if (ph.kind == .isqrt or ph.kind == .edit) return t.named(rest, ph);
        var parts: [2]Expr = undefined;
        for (ph.parts[0..ph.n], 0..) |p, k| parts[k] = try t.parse(rest[p[0]..p[1]]) orelse return null;
        var f: Found = undefined;
        var bound: u32 = 0;
        switch (ph.kind) {
            .expr => {
                if (parts[0].ops == 0) return null;
                f = .{ .source = parts[0].source, .display = parts[0].display, .run = parts[0].run };
                bound = parts[0].bound;
            },
            .divmod => {
                f = .{ .source = try std.fmt.allocPrint(a, "divmod({s}, {s})", .{ parts[0].source, parts[1].source }), .display = try std.fmt.allocPrint(a, "{s} ÷ {s}", .{ try wrapped(a, parts[0]), try wrapped(a, parts[1]) }), .divmod = true, .run = try std.fmt.allocPrint(a, "divmod({s}, {s})", .{ parts[0].run, parts[1].run }) };
                bound = parts[0].bound;
            },
            .mul => {
                f = .{ .source = try std.fmt.allocPrint(a, "({s}) * ({s})", .{ parts[0].source, parts[1].source }), .display = try std.fmt.allocPrint(a, "{s} * {s}", .{ try wrapped(a, parts[0]), try wrapped(a, parts[1]) }), .run = try std.fmt.allocPrint(a, "({s}) * ({s})", .{ parts[0].run, parts[1].run }) };
                bound = expr.sat(parts[0].bound + parts[1].bound);
            },
            .gcd => {
                f = .{ .source = try std.fmt.allocPrint(a, "gcd({s}, {s})", .{ parts[0].source, parts[1].source }), .display = try std.fmt.allocPrint(a, "gcd({s}, {s})", .{ parts[0].display, parts[1].display }), .run = try std.fmt.allocPrint(a, "from math import gcd\nprint(gcd({s}, {s}))", .{ parts[0].run, parts[1].run }) };
                bound = @max(parts[0].bound, parts[1].bound);
            },
            .subfrom => {
                f = .{ .source = try std.fmt.allocPrint(a, "({s}) - ({s})", .{ parts[1].source, parts[0].source }), .display = try std.fmt.allocPrint(a, "{s} - {s}", .{ try wrapped(a, parts[1]), try wrapped(a, parts[0]) }), .run = try std.fmt.allocPrint(a, "({s}) - ({s})", .{ parts[1].run, parts[0].run }) };
                bound = expr.sat(@max(parts[0].bound, parts[1].bound) + 1);
            },
            .isqrt, .edit => unreachable,
        }
        if (t.firedHas(f.display) or bound > max_digits) return null;
        // a bare sentence counts only when it is plainly a computation: a power or a six-digit literal
        if (h.bare and std.mem.indexOf(u8, f.source, "**") == null and !hasDigitRun(f.source, 6)) return null;
        return f;
    }

    /// A named problem over literals (STATED): isqrt on the bank's plan, edit distance as one RV32 program.
    fn named(t: *Tracer, rest: []const u8, ph: text.Phrase) Allocator.Error!?Found {
        const a = t.scratch.allocator();
        const x = rest[ph.parts[0][0]..ph.parts[0][1]];
        const f: Found = if (ph.kind == .isqrt) blk: {
            const n = try std.mem.replaceOwned(u8, a, x, ",", "");
            // a zero-led literal is no Python integer, and the plan is as wide as the integer frontend
            if ((n.len > 1 and n[0] == '0') or n.len > max_digits) return null;
            const call = try std.fmt.allocPrint(a, "isqrt({s})", .{n});
            break :blk .{ .source = call, .display = call, .run = try std.fmt.allocPrint(a, "from math import isqrt\nprint({s})", .{call}) };
        } else blk: {
            const y = rest[ph.parts[1][0]..ph.parts[1][1]];
            const call = try std.fmt.allocPrint(a, "edit_distance(\"{s}\", \"{s}\")", .{ x, y });
            break :blk .{ .source = call, .display = call, .run = try std.fmt.allocPrint(a, edit_program, .{ x, y }) };
        };
        return if (t.firedHas(f.display)) null else f;
    }

    /// PREFIX_STATED's line shapes: `L = R.` with L over names and literally R, or a hex literal's value.
    fn wholeShape(t: *Tracer, clean: []const u8) Allocator.Error!?struct { found: Found, end: usize } {
        const a = t.scratch.allocator();
        for (clean, 0..) |c, k| {
            if (c != '=' or (k > 0 and std.mem.indexOfScalar(u8, "=<>!", clean[k - 1]) != null)) continue;
            if (k + 1 < clean.len and clean[k + 1] == '=') continue;
            const sentence = text.sentenceAt(clean, k);
            if (text.hypothetical(sentence)) continue;
            const from = @intFromPtr(sentence.ptr) - @intFromPtr(clean.ptr);
            // the left side: names and operators only (a definition or a stated literal is not a restatement)
            const left = std.mem.trim(u8, clean[from..k], " \t");
            if (left.len == 0 or std.mem.indexOfAny(u8, left, "0123456789") != null) continue;
            const l = try t.parse(left) orelse continue;
            if (l.ops == 0) continue;
            var r0 = k + 1;
            while (r0 < clean.len and text.isSpace(clean[r0])) r0 += 1;
            const rest = clean[r0..];
            const ph = text.phrase(rest, false) orelse continue;
            if (ph.kind != .expr or text.approx(rest[0..ph.end])) continue;
            const r = try t.parse(rest[ph.parts[0][0]..ph.parts[0][1]]) orelse continue;
            if (r.ops == 0 or r.bound > max_digits or t.firedHas(r.display)) continue;
            if (!std.mem.eql(u8, try literalParens(a, l.source), r.source)) continue;
            return .{ .found = .{ .source = r.source, .display = r.display, .run = r.run }, .end = r0 + ph.end };
        }
        // `Hex = 0x….` / `This is 0x….`: ten or more hex digits ending a sentence; the radix runs on the circuit
        var i: usize = 0;
        while (std.mem.indexOfPos(u8, clean, i, "0x")) |h| : (i = h + 2) {
            const lead = std.mem.trimEnd(u8, clean[0..h], " \t");
            if (h > 0 and !text.isSpace(clean[h - 1])) continue;
            if (!std.mem.endsWith(u8, lead, "=") and !(lead.len >= 2 and text.wordAt(lead, lead.len - 2, "is", true))) continue;
            var j = h + 2;
            while (j < clean.len and std.ascii.isHex(clean[j])) j += 1;
            if (j - h - 2 < 10 or j - h - 2 > 100 or (j < clean.len and text.isWord(clean[j]))) continue;
            if (!text.sentenceEndsAt(clean, j)) continue;
            const lit = clean[h..j];
            const e = try t.parse(lit) orelse continue;
            if (e.bound > max_digits or t.firedHas(lit)) continue;
            return .{ .found = .{ .source = try a.dupe(u8, lit), .display = try a.dupe(u8, lit), .run = try a.dupe(u8, lit) }, .end = j };
        }
        return null;
    }

    fn parse(t: *Tracer, s: []const u8) Allocator.Error!?Expr {
        return expr.parse(t.scratch.allocator(), t.names.items, s);
    }
};

/// Levenshtein distance (T5's census program): the graph route runs it as a DP packet (runner_graph_edit.zig).
pub const edit_program = @import("runner_graph_edit.zig").program;

/// The owned line's default wording (`{}` is `<display> = <value>`); seeded S2 round 3: 42% fewer tokens than off.
pub const default_format = "[exact, computed by the engine: {}]";

/// The most a request may ask a restated line to be owned again (runner_trace_repeat).
pub const max_repeat = 3;

/// The longest wording a request may give (runner_trace_format).
pub const max_format = 128;

/// A request's wording: one line, at most `max_format` bytes, `{}` exactly once.
pub fn validFormat(f: []const u8) bool {
    return f.len <= max_format and std.mem.count(u8, f, "{}") == 1 and std.mem.indexOfAny(u8, f, "\r\n") == null;
}

/// A request's say over form trace (chat fields "runner_trace", "runner_trace_format", ...); defaults change nothing.
pub const Ask = struct {
    off: bool = false, // no trace cells in this reply, though the server's forms have trace
    format: []const u8 = "", // the owned line, `{}` for `<expr> = <value>` (validFormat); "": default_format
    repeat: u8 = 0, // a restated, already owned line is owned again up to this many times (0..max_repeat)
    check: bool = false, // residues mod 9 and 1000 follow the line where it takes them (runner_trace_check.zig)
};

/// `format` around "<display> = <value>" then `check`'s residues, or null (not one integer or (q, r), or contradicted).
pub fn owned(a: Allocator, format: []const u8, display: []const u8, divmod: bool, out: []const u8, check: bool) Allocator.Error!?[]u8 {
    const f = if (format.len > 0) format else default_format;
    const at = std.mem.indexOf(u8, f, "{}").?;
    const v = if (std.mem.endsWith(u8, out, "\n")) out[0 .. out.len - 1] else out;
    if (v.len == 0 or std.mem.indexOfScalar(u8, v, '\n') != null) return null;
    var q = v;
    var r: ?[]const u8 = null;
    if (divmod) {
        if (v.len < 6 or v[0] != '(' or v[v.len - 1] != ')') return null;
        const inner = v[1 .. v.len - 1];
        const comma = std.mem.indexOf(u8, inner, ", ") orelse return null;
        q = inner[0..comma];
        r = inner[comma + 2 ..];
        if (!isInt(r.?)) return null;
    }
    if (!isInt(q)) return null;
    var line: std.ArrayList(u8) = .empty;
    errdefer line.deinit(a);
    try line.print(a, "{s}{s} = {s}", .{ f[0..at], display, q });
    if (r) |rem| try line.print(a, " remainder {s}", .{rem});
    try line.appendSlice(a, f[at + 2 ..]);
    if (check and !try checks.append(a, &line, display, q, r)) {
        line.deinit(a);
        return null;
    }
    try line.append(a, '\n');
    return try line.toOwnedSlice(a);
}

fn isInt(s: []const u8) bool {
    const d = if (s.len > 0 and s[0] == '-') s[1..] else s;
    if (d.len == 0) return false;
    for (d) |c| if (!std.ascii.isDigit(c)) return false;
    return true;
}

fn wrapped(a: Allocator, e: Expr) Allocator.Error![]const u8 {
    return if (e.ops > 0) std.fmt.allocPrint(a, "({s})", .{e.display}) else e.display;
}

/// A whole-statement pow(a, b, m) as `print(...)`: the form the graph bank's modular-power plan takes; else `run`.
fn printed(a: Allocator, run: []const u8) Allocator.Error![]const u8 {
    if (!std.mem.startsWith(u8, run, "pow(") or run[run.len - 1] != ')') return run;
    var depth: usize = 0;
    for (run[3 .. run.len - 1]) |c| {
        if (c == '(') depth += 1;
        if (c == ')') depth -= 1;
        if (depth == 0) return run; // pow(...) is only part of the statement
    }
    return std.mem.concat(a, u8, &.{ "print(", run, ")" });
}

/// A name's source with the brackets around each plain literal dropped: `(98765432109)*(12345678901)` reads as written.
fn literalParens(a: Allocator, source: []const u8) Allocator.Error![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    var i: usize = 0;
    while (i < source.len) : (i += 1) {
        if (source[i] == '(') {
            var j = i + 1;
            while (j < source.len and std.ascii.isDigit(source[j])) j += 1;
            if (j > i + 1 and j < source.len and source[j] == ')') {
                try out.appendSlice(a, source[i + 1 .. j]);
                i = j;
                continue;
            }
        }
        try out.append(a, source[i]);
    }
    return out.items;
}

fn hasArith(source: []const u8) bool {
    var i: usize = 0;
    while (i < source.len) : (i += 1) {
        if (source[i] == '*' and i + 1 < source.len and source[i + 1] == '*') {
            i += 1;
            continue;
        }
        if (std.mem.indexOfScalar(u8, "-+*%", source[i]) != null) return true;
    }
    return false;
}

/// The byte index `n` UTF-8 characters past `i` (the study counts characters: "≈" is one).
fn charsOn(s: []const u8, i: usize, n: usize) usize {
    var j = i;
    var left = n;
    while (j < s.len and left > 0) {
        j += 1;
        while (j < s.len and s[j] & 0xC0 == 0x80) j += 1;
        left -= 1;
    }
    return j;
}

fn hasDigitRun(s: []const u8, n: usize) bool {
    var run: usize = 0;
    for (s) |c| {
        run = if (std.ascii.isDigit(c)) run + 1 else 0;
        if (run >= n) return true;
    }
    return false;
}

/// A hand trace starts in this text (a step, table or digit line, long division, carry, borrow, by hand, ...).
pub const cue = text.cueIn;
