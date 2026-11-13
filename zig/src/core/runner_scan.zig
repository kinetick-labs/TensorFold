//! The in-model runner's scanner: closed code cells and the model's own result headers, from committed reply bytes.
const std = @import("std");
const api = @import("runner_api.zig");
const trace = @import("runner_trace.zig");
const header = @import("runner_scan_boundary.zig");
const inline_scan = @import("runner_scan_inline.zig");
const quiet = @import("runner_scan_quiet.zig");
const Allocator = std.mem.Allocator;

pub const Config = struct {
    fence: bool = true, // ```lang … ``` blocks
    repl: bool = true, // `>>> ` lines
    lookahead: u32 = 64, // tokens after a closed fence in which a result header counts
    reasoning_only: bool = false, // triggers only while the think block is open
    inline_math: bool = false, // `A op B =` at a line's end: the value after `=` (exact math semantics)
    inline_mod: bool = true, // with inline_math: `A mod M =` and `A % M =` on literals, after any earlier `=`
    inline_arith: bool = true, // with inline_math: thousands commas (`147,000+73`), a whole `gcd(a, b)` or `0xFF`
    inline_chain: bool = false, // opt-in, T5 decides: the last `<expr> =` after an earlier `=` statement
    inline_quoted: bool = true, // PREFIX_INLINE_QUOTED: `<expr> =` inside an open quotation is quoted text, not owned
    float_window: u32 = 512, // tokens after a Python-float cue in which an inline span with `.` or `/` abstains
    trace: bool = false, // a computation the reasoning states, owned as `[exact: …]` where its hand trace starts
    trace_window: u32 = 512, // tokens after a stated computation's line in which a hand-trace cue counts
    trace_repeat: u8 = 0, // a restated, already owned line is owned again up to this many times
    trace_whole: bool = false, // opt-in (PREFIX_STATED): trace also takes the whole-statement shapes
    quiet: quiet.Spans = &.{}, // the family's spans no cell is read in: tool calls, tool results (runner_binding.Family)
};

/// Trace ownerships per reply, and trace cells (stated computations) a reply may open.
const trace_owned_max = 16;
const trace_cells_max = 32;

/// Text after which a reply keeps its own decimals and fractions: Python floats, or a common denominator.
const float_cues = [_][]const u8{ "float", "Decimal(", "sum(", "round(", "0.1+0.2", "0.1 + 0.2", "lcm", "LCM", "LCD", "denominator" };
/// Text after which `a + b` may join digits rather than add them (live: `Concatenate: … 1119 + 392 = 1119392`).
const join_cues = [_][]const u8{ "oncatenat", "join", "Join" };
/// Lines after a join cue in which a bare `a + b` opening its line keeps the model's own value.
const join_lines = 2;

/// A closed cell; `lang` and `code` live in the scanner until the next `feed`.
pub const Closed = struct { index: u32, form: api.Form, lang: []const u8, code: []const u8, own_context: bool = false, quoted: bool = false, trace: ?Stated = null }; // own_context, quoted: the reply keeps its value

/// A trace cell's line text: what `[exact: … = value]` shows, and whether the value is a (quotient, remainder) pair.
pub const Stated = struct { display: []const u8, divmod: bool };

/// Cell `index`'s result begins: `fenced` closes a fence after it; `inline_space` (inline: value only) adds a space.
pub const Trigger = struct { index: u32, fenced: bool = false, inline_space: ?bool = null, trace: bool = false, newline_before: bool = false, terminal: bool = false }; // trace: `[exact: …]` line

pub const Events = struct { closed: ?Closed = null, trigger: ?Trigger = null };

const State = enum { prose, fence, after };

pub const Scanner = struct {
    cfg: Config,
    state: State = .prose,
    line: std.ArrayList(u8) = .empty,
    lang: std.ArrayList(u8) = .empty,
    code: std.ArrayList(u8) = .empty,
    cells: u32 = 0, // cells closed so far; the next one's index
    pending: u32 = 0, // the fence or REPL cell a result header refers to
    since: u32 = 0, // tokens since the pending cell closed (state .after)
    terminal_ready: bool = false,
    armed: ?Trigger = null, // a trigger this token set; later non-blank bytes in the same token spoil it
    tokens: u32 = 0, // tokens fed
    float_at: ?u32 = null, // the token of the last float cue or closed code cell
    lines: u32 = 0, // newlines fed
    join_line: ?u32 = null, // the line of the last join cue
    tail: [16]u8 = undefined, // the last bytes fed, for cues split across tokens
    tail_len: usize = 0,
    think: bool = false, // the think block was open at the last token
    hush: quiet.Quiet = .{}, // inside one of cfg.quiet's spans
    code_cells: u32 = 0, // fence and REPL cells closed so far
    tracer: ?trace.Tracer = null, // names and owned statements (form trace; made at the first reasoning line)
    trace_src: std.ArrayList(u8) = .empty, // the pending statement's Python source
    trace_disp: std.ArrayList(u8) = .empty,
    trace_armed: std.ArrayList(u8) = .empty, // the display of the trace cell armed in this token
    trace_cell: ?u32 = null, // the stated computation waiting for its cue
    trace_at: u32 = 0, // the token its line ended at
    trace_cells: u32 = 0,
    trace_owned: u32 = 0,
    above: [3]std.ArrayList(u8) = .{ .empty, .empty, .empty }, // with STATED: the last three lines, newest first

    pub fn deinit(s: *Scanner, a: Allocator) void {
        for (&s.above) |*l| l.deinit(a);
        s.line.deinit(a);
        s.lang.deinit(a);
        s.code.deinit(a);
        s.trace_src.deinit(a);
        s.trace_disp.deinit(a);
        s.trace_armed.deinit(a);
        if (s.tracer) |*t| t.deinit();
    }

    /// One committed token's bytes; `think_open`: the think block is open at this token.
    pub fn feed(s: *Scanner, a: Allocator, bytes: []const u8, think_open: bool) Allocator.Error!Events {
        var ev: Events = .{};
        s.armed = null;
        s.tokens +|= 1;
        // the think block's end has no bytes of its own, yet the answer starts on a new line
        if (s.think and !think_open and s.line.items.len > 0) {
            try s.endLine(a, &ev, true);
            try s.keepLine(a);
            s.line.clearRetainingCapacity();
        }
        s.think = think_open;
        if (s.state == .after) {
            s.since += 1;
            if (s.since > s.cfg.lookahead) s.state = .prose;
        }
        if (s.trace_cell != null and s.tokens - s.trace_at > s.cfg.trace_window) s.trace_cell = null;
        for (bytes) |b| {
            if (s.hush.skip(s.cfg.quiet, b)) continue;
            s.cue(b);
            if (s.state == .after and !std.ascii.isWhitespace(b)) s.terminal_ready = false;
            if (s.armed != null and b != '\n' and b != ' ' and b != '\t' and b != '\r') s.armed = null;
            if (b != '\n') {
                try s.line.append(a, b);
                continue;
            }
            s.lines +|= 1;
            try s.endLine(a, &ev, think_open);
            try s.keepLine(a);
            s.line.clearRetainingCapacity();
        }
        ev.trigger = s.armed;
        if (ev.trigger) |t| if (t.trace) try s.traceSpent(t.index);
        const allowed = think_open or !s.cfg.reasoning_only;
        if (ev.trigger == null and s.state == .after and allowed and header.ready(s.line.items)) {
            s.fire(false);
            ev.trigger = s.armed;
            ev.trigger.?.newline_before = true;
        }
        if (s.cfg.inline_math and allowed and s.state != .fence and ev.closed == null and ev.trigger == null and
            std.mem.indexOfScalar(u8, bytes, '=') != null)
        {
            const floats = s.float_at != null and s.tokens - s.float_at.? <= s.cfg.float_window;
            // a modular step first: inlineExpr alone would take `10^18` of `3^400 mod 10^18 =` for the whole
            if (if (s.cfg.inline_mod) inline_scan.inlineMod(s.line.items) else null) |m| {
                s.lang.clearRetainingCapacity();
                try s.lang.appendSlice(a, "python");
                s.code.clearRetainingCapacity();
                try m.code(a, &s.code);
                ev.closed = s.close(.inline_expr);
                // Python reads 1e9 as a float: a reply reasoning in floats keeps its own value
                ev.closed.?.own_context = floats and std.mem.indexOfScalar(u8, m.modulus, 'e') != null;
                ev.closed.?.quoted = s.cfg.inline_quoted and inline_scan.quoted(s.line.items, m.left);
                ev.trigger = .{ .index = ev.closed.?.index, .inline_space = m.space };
            } else if (inline_scan.inlineExpr(s.line.items, s.cfg.inline_arith, s.cfg.inline_chain)) |x| {
                s.lang.clearRetainingCapacity();
                try s.lang.appendSlice(a, "python");
                s.code.clearRetainingCapacity();
                // thousands commas go: Python would read `147,000+73` as a tuple
                if (x.call) {
                    try s.code.appendSlice(a, x.expr);
                } else for (x.expr) |c| {
                    if (c != ',') try s.code.append(a, c);
                }
                ev.closed = s.close(.inline_expr);
                // `1119 + 392 =` opening its line, just after a join cue: the model may be joining digit groups
                const lead = std.mem.trim(u8, s.line.items[0 .. @intFromPtr(x.expr.ptr) - @intFromPtr(s.line.items.ptr)], " \t");
                const joined = s.join_line != null and s.lines - s.join_line.? <= join_lines and lead.len == 0 and
                    inline_scan.twoInts(x.expr);
                ev.closed.?.own_context = joined or (floats and std.mem.indexOfAny(u8, x.expr, "./") != null);
                ev.closed.?.quoted = s.cfg.inline_quoted and inline_scan.quoted(s.line.items, x.expr);
                ev.trigger = .{ .index = ev.closed.?.index, .inline_space = x.space };
            } else if (if (s.cfg.trace_whole) inline_scan.stacked(s.line.items, s.above[2].items, s.above[1].items, s.above[0].items) else null) |c| {
                // STATED: a stacked column (`264,278,…` / `+771,565,…` / `= ?`), the `=` alone on its line
                s.lang.clearRetainingCapacity();
                try s.lang.appendSlice(a, "python");
                s.code.clearRetainingCapacity();
                for (c.x) |d| if (d != ',') try s.code.append(a, d);
                try s.code.append(a, c.op);
                for (c.y) |d| if (d != ',') try s.code.append(a, d);
                ev.closed = s.close(.inline_expr);
                // a join cue just above the column: its two lines may be digit groups to join, as for `1119 + 392 =`
                ev.closed.?.own_context = c.op == '+' and s.join_line != null and s.lines - s.join_line.? <= join_lines + 2;
                ev.trigger = .{ .index = ev.closed.?.index, .inline_space = c.space };
            }
        }
        return ev;
    }

    /// STATED keeps the last three lines for a stacked column; a line too long for one is kept as a non-number.
    fn keepLine(s: *Scanner, a: Allocator) Allocator.Error!void {
        if (!s.cfg.trace_whole) return;
        const oldest = s.above[2];
        s.above[2] = s.above[1];
        s.above[1] = s.above[0];
        s.above[0] = oldest;
        s.above[0].clearRetainingCapacity();
        const l = std.mem.trimEnd(u8, s.line.items, "\r");
        try s.above[0].appendSlice(a, if (l.len > 200) "-" else l);
    }

    pub fn finishEos(s: *Scanner, a: Allocator, think_open: bool) Allocator.Error!Events {
        var ev: Events = .{};
        s.tokens +|= 1;
        if (s.state == .fence and header.fenceClose(std.mem.trimEnd(u8, s.line.items, "\r"))) {
            try s.endLine(a, &ev, think_open);
            s.line.clearRetainingCapacity();
        }
        if (s.state == .after and s.terminal_ready and s.since < s.cfg.lookahead and
            (think_open or !s.cfg.reasoning_only))
        {
            s.fire(false);
            ev.trigger = s.armed;
            ev.trigger.?.terminal = true;
        }
        return ev;
    }

    fn endLine(s: *Scanner, a: Allocator, ev: *Events, think_open: bool) Allocator.Error!void {
        const raw = std.mem.trimEnd(u8, s.line.items, "\r");
        const allowed = think_open or !s.cfg.reasoning_only;
        switch (s.state) {
            .fence => {
                if (header.fenceClose(raw)) {
                    const prev = s.pending;
                    const had = s.code_cells > 0;
                    ev.closed = s.close(.fence);
                    // `gcc x.c -o x && ./x` runs the cell before it: a header after it names that cell
                    if (had and runsPrevious(ev.closed.?.lang, ev.closed.?.code)) s.pending = prev;
                    s.state = .after;
                    s.since = 0;
                    s.terminal_ready = true;
                } else {
                    try s.code.appendSlice(a, raw);
                    try s.code.append(a, '\n');
                }
            },
            .prose, .after => {
                if (s.state == .after and allowed) {
                    // a header line only: a bare output fence is no handoff (T6 census)
                    if (header.complete(raw)) return s.fire(false);
                }
                if (s.cfg.fence) if (header.fenceOpen(raw)) |lang| {
                    s.lang.clearRetainingCapacity();
                    for (lang) |c| try s.lang.append(a, std.ascii.toLower(c));
                    s.code.clearRetainingCapacity();
                    s.state = .fence;
                    return;
                };
                if (s.cfg.repl) if (replLine(raw)) |src| {
                    s.lang.clearRetainingCapacity();
                    try s.lang.appendSlice(a, "python");
                    s.code.clearRetainingCapacity();
                    try s.code.appendSlice(a, src);
                    ev.closed = s.close(.repl);
                    s.state = .after;
                    if (allowed) s.fire(false);
                    return;
                };
                if (s.cfg.trace and allowed) try s.traceLine(a, ev, raw);
            },
        }
    }

    /// A reasoning prose line: a pending statement's cue fires it; the line's own statement opens a trace cell.
    fn traceLine(s: *Scanner, a: Allocator, ev: *Events, raw: []const u8) Allocator.Error!void {
        if (s.tracer == null) {
            s.tracer = trace.Tracer.init(a);
            s.tracer.?.repeat = s.cfg.trace_repeat;
            s.tracer.?.whole = s.cfg.trace_whole;
        }
        const t = &s.tracer.?;
        var fired = false;
        if (s.trace_cell != null and trace.cue(raw)) fired = try s.traceArm(a);
        const found = try t.line(raw) orelse return;
        if (s.trace_cell != null and std.mem.eql(u8, found.display, s.trace_disp.items)) return;
        if (ev.closed != null or s.trace_cells >= trace_cells_max or s.trace_owned >= trace_owned_max) return;
        s.trace_src.clearRetainingCapacity();
        try s.trace_src.appendSlice(a, found.run);
        s.trace_disp.clearRetainingCapacity();
        try s.trace_disp.appendSlice(a, found.display);
        ev.closed = .{ .index = s.cells, .form = .repl, .lang = "python", .code = s.trace_src.items, .trace = .{ .display = s.trace_disp.items, .divmod = found.divmod } };
        s.trace_cell = s.cells;
        s.trace_at = s.tokens;
        s.cells += 1;
        s.trace_cells += 1;
        if (found.same_line and !fired) _ = try s.traceArm(a);
    }

    /// Arms the pending trace cell's trigger at this line's end, unless another form's trigger is armed.
    fn traceArm(s: *Scanner, a: Allocator) Allocator.Error!bool {
        if (s.armed != null) return false;
        s.armed = .{ .index = s.trace_cell.?, .trace = true };
        s.trace_armed.clearRetainingCapacity();
        try s.trace_armed.appendSlice(a, s.trace_disp.items);
        return true;
    }

    /// The armed trace trigger survived its token: that statement is owned once, and no longer pending.
    fn traceSpent(s: *Scanner, index: u32) Allocator.Error!void {
        if (s.tracer) |*t| try t.fire(s.trace_armed.items);
        if (s.trace_cell == index) s.trace_cell = null;
        s.trace_owned += 1;
    }

    fn cue(s: *Scanner, b: u8) void {
        if (s.tail_len == s.tail.len) {
            std.mem.copyForwards(u8, s.tail[0 .. s.tail.len - 1], s.tail[1..]);
            s.tail_len -= 1;
        }
        s.tail[s.tail_len] = b;
        s.tail_len += 1;
        for (float_cues) |c| if (std.mem.endsWith(u8, s.tail[0..s.tail_len], c)) {
            s.float_at = s.tokens;
        };
        for (join_cues) |c| if (std.mem.endsWith(u8, s.tail[0..s.tail_len], c)) {
            s.join_line = s.lines;
        };
    }

    fn close(s: *Scanner, form: api.Form) Closed {
        const c: Closed = .{ .index = s.cells, .form = form, .lang = s.lang.items, .code = s.code.items };
        if (form != .inline_expr) {
            s.pending = s.cells;
            s.float_at = s.tokens;
            s.code_cells += 1;
        }
        s.cells += 1;
        return c;
    }

    fn fire(s: *Scanner, fenced: bool) void {
        s.armed = .{ .index = s.pending, .fenced = fenced };
        s.state = .prose;
        s.terminal_ready = false;
    }
};

/// A shell fence whose every command compiles or runs a program (no defines, pipes or redirects).
fn runsPrevious(lang: []const u8, code: []const u8) bool {
    const shells = [_][]const u8{ "bash", "sh", "shell", "zsh", "console" };
    for (shells) |sh| {
        if (std.mem.eql(u8, lang, sh)) break;
    } else return false;
    const tools = [_][]const u8{ "gcc ", "g++ ", "cc ", "clang ", "clang++ ", "tcc ", "./", "python ", "python3 ", "node " };
    var any = false;
    var lines = std.mem.splitScalar(u8, code, '\n');
    while (lines.next()) |raw| {
        var line = std.mem.trim(u8, raw, " \t\r");
        if (std.mem.startsWith(u8, line, "$ ")) line = line[2..];
        if (line.len == 0 or line[0] == '#') continue;
        if (std.mem.indexOfAny(u8, line, "|<>;`") != null or std.mem.indexOf(u8, line, "-D") != null) return false;
        var parts = std.mem.splitSequence(u8, line, "&&");
        while (parts.next()) |part| {
            const cmd = std.mem.trim(u8, part, " \t");
            for (tools) |t| {
                if (std.mem.startsWith(u8, cmd, t)) break;
            } else return false;
            any = true;
        }
    }
    return any;
}

/// ">>> code" (one line, not opening a block): the code, else null.
fn replLine(line: []const u8) ?[]const u8 {
    const t = header.indentOk(line) orelse return null;
    if (!std.mem.startsWith(u8, t, ">>> ")) return null;
    const src = std.mem.trim(u8, t[4..], " \t");
    if (src.len == 0 or src[src.len - 1] == ':') return null;
    return src;
}

fn run(a: Allocator, s: *Scanner, tokens: []const []const u8, out: *std.ArrayList(Events)) !void {
    for (tokens) |t| try out.append(a, try s.feed(a, t, true));
}

test "runner scan: a fence closes a cell; the header after it triggers that cell, plain" {
    const a = std.testing.allocator;
    var s: Scanner = .{ .cfg = .{} };
    defer s.deinit(a);
    var ev: std.ArrayList(Events) = .empty;
    defer ev.deinit(a);
    try run(a, &s, &.{ "let me check.\n```", "Python\n", "print(2*3)\n", "```\n", "Result", ":\n", "6" }, &ev);
    try std.testing.expectEqual(@as(?Closed, null), ev.items[2].closed);
    const c = ev.items[3].closed.?;
    try std.testing.expectEqual(@as(u32, 0), c.index);
    try std.testing.expectEqual(api.Form.fence, c.form);
    try std.testing.expectEqual(Trigger{ .index = 0, .fenced = false }, ev.items[5].trigger.?);
    for (ev.items[0..5]) |e| try std.testing.expectEqual(@as(?Trigger, null), e.trigger);
}

test "runner scan: cell code and lang are kept exactly" {
    const a = std.testing.allocator;
    var s: Scanner = .{ .cfg = .{} };
    defer s.deinit(a);
    _ = try s.feed(a, "```py\nx = 3\nprint(x*x)\n", true);
    const e = try s.feed(a, "```\n", true);
    try std.testing.expectEqualStrings("py", e.closed.?.lang);
    try std.testing.expectEqualStrings("x = 3\nprint(x*x)\n", e.closed.?.code);
}

test "runner scan: a bare result fence after a cell is no trigger; nor is a header in prose without a cell" {
    const a = std.testing.allocator;
    var s: Scanner = .{ .cfg = .{} };
    defer s.deinit(a);
    try std.testing.expectEqual(@as(?Trigger, null), (try s.feed(a, "Result:\n", true)).trigger);
    _ = try s.feed(a, "```python\nprint(1)\n```\n", true);
    try std.testing.expectEqual(@as(?Trigger, null), (try s.feed(a, "```text\n", true)).trigger);
}

test "runner scan: a REPL line is a cell and its own trigger; a block opener is not" {
    const a = std.testing.allocator;
    var s: Scanner = .{ .cfg = .{} };
    defer s.deinit(a);
    const e = try s.feed(a, ">>> 2**100\n", true);
    try std.testing.expectEqualStrings("2**100", e.closed.?.code);
    try std.testing.expectEqual(api.Form.repl, e.closed.?.form);
    try std.testing.expectEqual(Trigger{ .index = 0, .fenced = false }, e.trigger.?);
    const f = try s.feed(a, ">>> for i in range(3):\n", true);
    try std.testing.expect(f.closed == null and f.trigger == null);
}

test "runner scan: no trigger past the lookahead, when the model already writes in the trigger's token, or a claim line" {
    const a = std.testing.allocator;
    var s: Scanner = .{ .cfg = .{ .lookahead = 2 } };
    defer s.deinit(a);
    _ = try s.feed(a, "```python\nprint(1)\n```\n", true);
    _ = try s.feed(a, "So", true);
    _ = try s.feed(a, " then", true);
    try std.testing.expectEqual(@as(?Trigger, null), (try s.feed(a, "\nOutput:\n", true)).trigger);
    var t: Scanner = .{ .cfg = .{} };
    defer t.deinit(a);
    _ = try t.feed(a, "```python\nprint(1)\n```\n", true);
    try std.testing.expectEqual(@as(?Trigger, null), (try t.feed(a, "Output:\n1", true)).trigger);
    try std.testing.expectEqual(@as(?Trigger, null), (try t.feed(a, "Result: we need O(n)\n", true)).trigger);
}

test "runner scan: reasoning_only ignores headers outside the think block" {
    const a = std.testing.allocator;
    var s: Scanner = .{ .cfg = .{ .reasoning_only = true } };
    defer s.deinit(a);
    _ = try s.feed(a, "```python\nprint(1)\n```\n", false);
    try std.testing.expectEqual(@as(?Trigger, null), (try s.feed(a, "Output:\n", false)).trigger);
}

fn inlineOf(line: []const u8) ?[]const u8 {
    const a = std.testing.allocator;
    var s: Scanner = .{ .cfg = .{ .inline_math = true } };
    defer s.deinit(a);
    const e = s.feed(a, line, true) catch return null;
    const t = e.trigger orelse return null;
    if (t.inline_space == null or e.closed == null or e.closed.?.form != .inline_expr) return null;
    return a.dupe(u8, e.closed.?.code) catch null;
}

test "runner scan: an inline expression at a line's end is a cell and its own trigger" {
    const a = std.testing.allocator;
    for ([_][2][]const u8{
        .{ "So 47 * 83 =", "47 * 83" },                 .{ "15*16*17 = ", "15*16*17" },
        .{ "47 × 83 =", "47 × 83" },
        .{ "Compute: -3 * (4 + 5) =", "-3 * (4 + 5)" }, .{ "2**10 =", "2**10" },
        .{ "3 * -4 =", "3 * -4" },
        .{ "43210 − 98765 + 1 =", "43210 − 98765 + 1" },
        .{ "So 9876 - 4321 =", "9876 - 4321" },         .{ "9876-4321=", "9876-4321" },
        .{ "2^100 - 3^50 =", "2^100 - 3^50" },
        .{ "≈ 2*3 =", "2*3" },
        .{ "φ: 2^18 =", "2^18" },
    }) |c| {
        const got = inlineOf(c[0]) orelse return error.TestExpectedInline;
        defer a.free(got);
        try std.testing.expectEqualStrings(c[1], got);
    }
}

test "runner scan: inline refuses chains, comparisons, words, lists and incomplete expressions" {
    for ([_][]const u8{
        "a = 3*4 =",                     "15*16 = 240*17 =", "3*4 ==",    "x <= 3*4 =",    "2x + 3 =",  "x3*4 =",   "f(2)*3 =",
        "1673, +10 =",                   "1,23 + 5 =",       "a - 3*4 =", "3 + 4 + ... =", "3 4 + 5 =", "(3 + 4 =", "3 + =",
        "42 =",                          "$5 + 3 =",         "50% * 2 =", "3 *",           "**3*4** =", "2(3+4) =", "3*4 = 12",
        "Rightmost 9 positions 12-20 =", "pages 3-5 + 1 =",
        "Since φ(2^18)=",
        "φ(5^18) =",
        "λ(10^18)=",
    }) |line| {
        if (inlineOf(line)) |got| {
            std.testing.allocator.free(got);
            std.debug.print("accepted: {s}\n", .{line});
            return error.TestUnexpectedInline;
        }
    }
}

test "runner scan: inline stays off by default, inside a fence and in the token after the value" {
    const a = std.testing.allocator;
    var s: Scanner = .{ .cfg = .{} };
    defer s.deinit(a);
    try std.testing.expect((try s.feed(a, "47 * 83 =", true)).trigger == null);
    var f: Scanner = .{ .cfg = .{ .inline_math = true } };
    defer f.deinit(a);
    _ = try f.feed(a, "```python\n", true);
    try std.testing.expect((try f.feed(a, "x = 47 * 83 =", true)).trigger == null);
    var g: Scanner = .{ .cfg = .{ .inline_math = true } };
    defer g.deinit(a);
    try std.testing.expect((try g.feed(a, "47 * 83 = 3901", true)).trigger == null);
}

test "runner scan: a Python-float reply marks inline decimals and fractions, not integers (live run-check-014)" {
    const a = std.testing.allocator;
    var s: Scanner = .{ .cfg = .{ .inline_math = true } };
    defer s.deinit(a);
    for ([_][]const u8{ "Actually Python:\n", ">>> sum([0.1]*10)\n", "0.9999999999999999\n", "I think yes. ", "Let's recall common: " }) |t|
        _ = try s.feed(a, t, true);
    const live = try s.feed(a, "0.1+0.2 =", true);
    try std.testing.expect(live.closed.?.form == .inline_expr and live.closed.?.own_context);
    _ = try s.feed(a, " 0.3\n", true);
    for ([_][]const u8{ "7/12 =", "6 * 7 =" }, [_]bool{ true, false }) |t, want| {
        const e = try s.feed(a, t, true);
        try std.testing.expectEqual(want, e.closed.?.own_context);
        _ = try s.feed(a, " 1\n", true);
    }
    var far: Scanner = .{ .cfg = .{ .inline_math = true, .float_window = 4 } };
    defer far.deinit(a);
    _ = try far.feed(a, "a flo", true);
    _ = try far.feed(a, "at\n", true); // a cue split across tokens
    try std.testing.expect((try far.feed(a, "1.5+2 =", true)).closed.?.own_context);
    _ = try far.feed(a, " 3.5\n", true);
    for (0..4) |_| _ = try far.feed(a, "ok\n", true);
    try std.testing.expect(!(try far.feed(a, "1.5+2 =", true)).closed.?.own_context);
    var plain: Scanner = .{ .cfg = .{ .inline_math = true } };
    defer plain.deinit(a);
    try std.testing.expect(!(try plain.feed(a, "Discount: 889.35*0.075 =", true)).closed.?.own_context);
}

test "runner scan: an answer's first fence opens a cell when the reasoning ended mid-line (live run-check-004)" {
    const a = std.testing.allocator;
    var s: Scanner = .{ .cfg = .{} };
    defer s.deinit(a);
    _ = try s.feed(a, "Let's craft.", true);
    _ = try s.feed(a, "", true); // the think block's end: no bytes
    for ([_][]const u8{ "```python\n", "print(2**100)\n" }) |t| _ = try s.feed(a, t, false);
    const closed = (try s.feed(a, "```\n", false)).closed.?;
    try std.testing.expectEqualStrings("python", closed.lang);
    try std.testing.expectEqualStrings("print(2**100)\n", closed.code);
    try std.testing.expectEqual(@as(u32, 0), (try s.feed(a, "Output:\n", false)).trigger.?.index);
}

test "runner scan: a header after a compile-and-run shell fence names the program cell" {
    const a = std.testing.allocator;
    for ([_][]const u8{
        "```bash\ngcc -Wall -o gcd gcd.c\n./gcd\n```\n",
        "```sh\n$ cc fib.c -o fib && ./fib\n```\n",
        "```bash\necho 42\n```\n",
        "```bash\ngcc -DN=5 x.c -o x && ./x\n```\n",
        "```bash\n./x | head\n```\n",
    }, [_]u32{ 0, 0, 1, 1, 1 }) |shell, want| {
        var s: Scanner = .{ .cfg = .{} };
        defer s.deinit(a);
        _ = try s.feed(a, "```c\nint main(void) { return 0; }\n```\n", false);
        _ = try s.feed(a, shell, false);
        try std.testing.expectEqual(want, (try s.feed(a, "Output:\n", false)).trigger.?.index);
    }
}

test "runner scan: a header after an inline cell still names the fence cell before it" {
    const a = std.testing.allocator;
    var s: Scanner = .{ .cfg = .{ .inline_math = true } };
    defer s.deinit(a);
    _ = try s.feed(a, "```python\nprint(1)\n```\n", true);
    const e = try s.feed(a, "Note 2 + 2 =", true);
    try std.testing.expectEqual(@as(u32, 1), e.trigger.?.index);
    _ = try s.feed(a, " 4\n", true);
    try std.testing.expectEqual(Trigger{ .index = 0, .fenced = false }, (try s.feed(a, "Output:\n", true)).trigger.?);
}

comptime {
    _ = @import("runner_scan_result_test.zig");
    _ = @import("runner_scan_quiet.zig");
}
