//! <prefix>_BACKEND=graph: pure-integer calc programs run as one packet of T1's bank plans (128 digits).
const std = @import("std");
const api = @import("runner_api.zig");
const graph = @import("runner_graph.zig");
const packet = @import("runner_graph_packet.zig");
const Family = packet.Family;
const Allocator = std.mem.Allocator;
const prog = @import("runner_graph_program.zig");
pub const program = prog.program;

/// One node of the program, children first; `digits` bounds its value's decimal digits from syntax alone.
const Node = struct {
    kind: enum { literal, neg, binary, pow },
    family: Family = .add,
    l: u32 = 0,
    r: u32 = 0,
    text: []const u8 = "",
    k: u16 = 0,
    digits: u32,
    span: []const u8 = "", // its source text, for structural matches
};

const max_digits: u32 = packet.max_digits;

const Invalid = error{ Unsupported, OutOfMemory };

/// Calc firmware input over integer literals, + - * // % ** and parentheses; widths are checked when lowering.
const Parser = struct {
    a: Allocator,
    src: []const u8,
    pos: usize = 0,
    nodes: std.ArrayList(Node) = .empty,
    depth: u32 = 0,

    fn peek(p: *Parser, token: []const u8) bool {
        return std.mem.startsWith(u8, p.src[p.pos..], token);
    }
    fn take(p: *Parser, token: []const u8) bool {
        if (!p.peek(token)) return false;
        p.pos += token.len;
        return true;
    }
    fn add(p: *Parser, n: Node) Invalid!u32 {
        if (p.nodes.items.len >= packet.max_ops) return error.Unsupported;
        try p.nodes.append(p.a, n);
        return @intCast(p.nodes.items.len - 1);
    }
    fn at(p: *Parser, i: u32) Node {
        return p.nodes.items[i];
    }

    fn expr(p: *Parser) Invalid!u32 {
        p.depth += 1;
        defer p.depth -= 1;
        if (p.depth > 32) return error.Unsupported;
        var l = try p.term();
        while (true) {
            const f: Family = if (p.take("+")) .add else if (p.take("-")) .sub else return l;
            l = try p.binary(f, l, try p.term());
        }
    }

    fn term(p: *Parser) Invalid!u32 {
        var l = try p.unary();
        while (true) {
            if (p.peek("**")) return error.Unsupported;
            const f: Family = if (p.take("//")) .div else if (p.peek("/")) return error.Unsupported else if (p.take("*")) .mul else if (p.take("%")) .mod else return l;
            l = try p.binary(f, l, try p.unary());
        }
    }

    fn unary(p: *Parser) Invalid!u32 {
        if (p.take("+")) return p.unary();
        if (!p.take("-")) return p.power();
        const c = try p.unary();
        const n = p.at(c);
        if (n.kind == .literal and n.text[0] != '-')
            return p.add(.{ .kind = .literal, .text = try std.mem.concat(p.a, u8, &.{ "-", n.text }), .digits = n.digits });
        // -x is 0 - x
        return p.add(.{ .kind = .neg, .r = c, .digits = n.digits });
    }

    fn power(p: *Parser) Invalid!u32 {
        const base = try p.atom();
        if (!p.take("**")) return base;
        // the exponent is control data: a plain literal, so it selects a prepared pow-<k>-<width> schedule
        const start = p.pos;
        while (p.pos < p.src.len and std.ascii.isDigit(p.src[p.pos])) p.pos += 1;
        const k = std.fmt.parseInt(u16, p.src[start..p.pos], 10) catch return error.Unsupported;
        if (k > 999 or (p.pos - start > 1 and p.src[start] == '0') or p.peek("**") or p.peek(".")) return error.Unsupported;
        const b = p.at(base);
        const digits: u32 = if (k == 0) 1 else if (b.kind == .literal) literalPowDigits(b.text, k) else @min(1 << 20, b.digits * k);
        return p.add(.{ .kind = .pow, .l = base, .k = k, .digits = digits, .span = if (b.span.len > 0) p.src[b.span.ptr - p.src.ptr .. p.pos] else "" });
    }

    fn atom(p: *Parser) Invalid!u32 {
        if (p.take("(")) {
            const e = try p.expr();
            if (!p.take(")")) return error.Unsupported;
            // `(x)` has x's value; its span keeps the parentheses, so `(x)**2` and `x` differ only by them
            return e;
        }
        const start = p.pos;
        while (p.pos < p.src.len and std.ascii.isDigit(p.src[p.pos])) p.pos += 1;
        const n = p.pos - start;
        if (n == 0 or (n > 1 and p.src[start] == '0') or p.peek(".")) return error.Unsupported;
        return p.add(.{ .kind = .literal, .text = p.src[start..p.pos], .digits = @intCast(n), .span = p.src[start..p.pos] });
    }

    fn binary(p: *Parser, f: Family, l: u32, r: u32) Invalid!u32 {
        const dl = p.at(l).digits;
        const dr = p.at(r).digits;
        if (f == .div or f == .mod) {
            // only a nonzero literal divisor: Python raises on zero, a computed divisor could be zero
            const d = p.at(r);
            if (d.kind != .literal or std.mem.eql(u8, std.mem.trimStart(u8, d.text, "-"), "0")) return error.Unsupported;
        }
        const digits = switch (f) {
            .add, .sub => @max(dl, dr) + 1,
            .mul => dl + dr,
            // |a // b| < 10^(da-db+1), plus one for the floor; never more than |a| itself
            .div => @min(dl, @max(1, (dl + 2) -| dr)),
            .mod => dr,
            .pow => unreachable,
        };
        if (f == .sub) if (try p.squareOffset(l, r)) |n| return n;
        const span = if (p.at(l).span.len > 0 and p.at(r).span.len > 0) p.src[p.at(l).span.ptr - p.src.ptr .. p.at(r).span.ptr - p.src.ptr + p.at(r).span.len] else "";
        return p.add(.{ .kind = .binary, .family = f, .l = l, .r = r, .digits = digits, .span = span });
    }

    /// (Y+B)**2 - Y**2 (or L**(2k) for Y = L**k) as B*(Y+(Y+B)), without the square's width (big-number-020).
    fn squareOffset(p: *Parser, l: u32, r: u32) Invalid!?u32 {
        const sq = p.at(l);
        if (sq.kind != .pow or sq.k != 2) return null;
        const x = p.at(sq.l);
        if (x.kind != .binary or x.family != .add) return null;
        const same = struct {
            fn f(q: *Parser, y: u32, rhs: Node) bool {
                const yn = q.at(y);
                if (rhs.kind != .pow) return false;
                if (rhs.k == 2 and yn.span.len > 0 and std.mem.eql(u8, q.at(rhs.l).span, yn.span)) return true;
                // L**(2k) against Y = L**k, over the same literal L
                const base = q.at(rhs.l);
                return yn.kind == .pow and base.kind == .literal and q.at(yn.l).kind == .literal and
                    std.mem.eql(u8, base.text, q.at(yn.l).text) and rhs.k == 2 * @as(u32, yn.k);
            }
        }.f;
        const rhs = p.at(r);
        const y, const b = if (same(p, x.l, rhs)) .{ x.l, x.r } else if (same(p, x.r, rhs)) .{ x.r, x.l } else return null;
        const yx = try p.add(.{ .kind = .binary, .family = .add, .l = y, .r = sq.l, .digits = @max(p.at(y).digits, x.digits) + 1 });
        return try p.add(.{ .kind = .binary, .family = .mul, .l = b, .r = yx, .digits = p.at(b).digits + p.at(yx).digits });
    }
};

/// Builder.binary's bucket: add/sub cover the wider operand plus a carry, mul the digit sum, div/mod the wider operand.
fn plan(a: Allocator, f: Family, dl: u32, dr: u32) error{ Unsupported, OutOfMemory }![]const u8 {
    const needed = switch (f) {
        .add, .sub => @max(dl, dr) + 1,
        .mul => dl + dr,
        .div, .mod => @max(dl, dr),
        .pow => unreachable,
    };
    if (needed > max_digits) return error.Unsupported;
    // 256: T1's b047d48c8 widths; a bank without that plan (v4, or div/mod) refuses at lowering, never here
    const width = for ([_]u16{ 1, 2, 4, 8, 16, 32, 62, 128, 256 }) |n| {
        if (n >= needed) break n;
    } else return error.Unsupported;
    return std.fmt.allocPrint(a, "{t}-{d}", .{ f, width });
}

/// Digits of |b|**k bounded from the literal's own digits: floor(k * log10|b|) + 1, rounded up for safety.
fn literalPowDigits(text: []const u8, k: u16) u32 {
    const digits = std.mem.trimStart(u8, text, "-");
    if (std.mem.eql(u8, digits, "0") or std.mem.eql(u8, digits, "1")) return 1;
    const lead = @min(digits.len, 15);
    const m = std.fmt.parseFloat(f64, digits[0..lead]) catch return 1 << 20;
    // past 15 digits the leading part rounds up, so the bound stays above the true value
    const log = std.math.log10(if (digits.len > lead) m + 1 else m) + @as(f64, @floatFromInt(digits.len - lead));
    const bound = @floor(@as(f64, @floatFromInt(k)) * log + 1e-9) + 1;
    return if (bound > 1 << 20) 1 << 20 else @intFromFloat(bound);
}

/// A prepared constant back to the decimal text that binds it (plan data such as 0 and 1, never an answer).
fn constantText(a: Allocator, v: *const packet.Value) ![]u8 {
    var buf: [packet.max_digits + 1]u8 = undefined;
    var top: usize = 0;
    var digits: [packet.max_digits]u8 = undefined;
    for (&digits, 0..) |*d, i| {
        d.* = 0;
        for (0..4) |bit| if (v[i / 31][2 + i % 31 * 4 + bit] != 0) {
            d.* |= @as(u8, 1) << @intCast(bit);
        };
        if (d.* > 9) return error.GraphLiteral;
        if (d.* != 0) top = i;
    }
    var n: usize = 0;
    if (v[0][126] != 0) {
        buf[0] = '-';
        n = 1;
    }
    var i = top + 1;
    while (i > 0) : (i -= 1) {
        buf[n] = '0' + digits[i - 1];
        n += 1;
    }
    const text = try a.dupe(u8, buf[0..n]);
    const back = try packet.literal(text);
    if (!std.mem.eql(u8, std.mem.asBytes(&back), std.mem.asBytes(v))) return error.GraphLiteral;
    return text;
}

pub const Shape = struct {
    nodes: []const Node,
    root: u32,
    live: []const bool, // reachable from root; only these are lowered and width-checked
    plans: []const []const u8, // each live operation's bank plan
};

/// Calc input `src` as nodes with their bank plans; null when any live part is not a graph route.
pub fn shape(arena: Allocator, src: []const u8) Allocator.Error!?Shape {
    if (src.len == 0 or src.len > 4096) return null;
    var p: Parser = .{ .a = arena, .src = src };
    const root = p.expr() catch |e| return if (e == error.OutOfMemory) error.OutOfMemory else null;
    if (p.pos != src.len) return null;
    const nodes = p.nodes.items;
    const live = try arena.alloc(bool, nodes.len);
    @memset(live, false);
    live[root] = true;
    var i = nodes.len;
    while (i > 0) : (i -= 1) if (live[i - 1]) switch (nodes[i - 1].kind) {
        .literal => {},
        .neg => live[nodes[i - 1].r] = true,
        .pow => live[nodes[i - 1].l] = true,
        .binary => {
            live[nodes[i - 1].l] = true;
            live[nodes[i - 1].r] = true;
        },
    };
    const plans = try arena.alloc([]const u8, nodes.len);
    for (nodes, live, plans) |n, on, *name| {
        name.* = "";
        if (!on) continue;
        if (n.digits > max_digits) return null;
        name.* = switch (n.kind) {
            .literal => continue,
            .neg => plan(arena, .sub, 1, n.digits),
            .binary => plan(arena, n.family, nodes[n.l].digits, nodes[n.r].digits),
            .pow => std.fmt.allocPrint(arena, "pow-{d}-{d}", .{ n.k, @as(u32, if (n.digits > 128) 256 else if (n.digits > 62) 128 else 62) }),
        } catch |e| return if (e == error.OutOfMemory) error.OutOfMemory else null;
    }
    return .{ .nodes = nodes, .root = root, .live = live, .plans = plans };
}

/// Calc input `src` as one graph packet inlining each operation's named plan from `bank`, else null.
pub fn lower(a: Allocator, bank: *const graph.Bank, src: []const u8) Allocator.Error!?[]u8 {
    var arena_state = std.heap.ArenaAllocator.init(a);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const sh = try shape(arena, src) orelse return null;
    const nodes = sh.nodes;
    // pass 1: every bound value first (literals, then each plan's constants), as the packet requires
    var values: std.ArrayList([]const u8) = .empty;
    const value_at = try arena.alloc(u32, nodes.len);
    const const_at = try arena.alloc(u32, nodes.len);
    const plans = try arena.alloc(?*const @import("runner_graph_bank.zig").PreparedPlan, nodes.len);
    var zero: ?u32 = null;
    for (nodes, 0..) |n, i| {
        plans[i] = null;
        if (!sh.live[i]) continue;
        if (n.kind == .literal) {
            value_at[i] = @intCast(values.items.len);
            try values.append(arena, n.text);
            continue;
        }
        const pp = bank.plan(sh.plans[i]) orelse return null;
        if (pp.algorithm != null) return null;
        if (pp.inputs != @as(u32, if (n.kind == .pow) 1 else 2)) return null;
        plans[i] = pp;
        const_at[i] = @intCast(values.items.len);
        for (pp.constants) |*c| try values.append(arena, constantText(arena, c) catch |e| return if (e == error.OutOfMemory) error.OutOfMemory else null);
        if (n.kind == .neg and zero == null) {
            zero = @intCast(values.items.len);
            try values.append(arena, "0");
        }
    }
    if (values.items.len > packet.max_values) return null;
    // pass 2: each plan's ops with its inputs, constants and outputs renumbered into the packet
    var ops: std.ArrayList(packet.Op) = .empty;
    var count: u32 = @intCast(values.items.len);
    for (nodes, 0..) |n, i| {
        const pp = plans[i] orelse continue;
        const ins: [2]u32 = switch (n.kind) {
            .literal => unreachable,
            .neg => .{ zero.?, value_at[n.r] },
            .binary => .{ value_at[n.l], value_at[n.r] },
            .pow => .{ value_at[n.l], 0 },
        };
        const m: Map = .{ .ins = ins[0..pp.inputs], .consts = const_at[i], .n_consts = @intCast(pp.constants.len), .base = count };
        for (pp.ops) |op| {
            if (ops.items.len >= packet.max_ops) return null;
            var o = op;
            o.a = m.ref(op.a);
            if (op.b) |b| o.b = m.ref(b);
            if (op.args) |args| {
                const re = try arena.alloc(u32, args.len);
                for (args, re) |x, *y| y.* = m.ref(x);
                o.args = re;
            }
            if (op.copies.len != 0) {
                const re = try arena.dupe(packet.Copy, op.copies);
                for (re) |*c| c.ref = m.ref(c.ref);
                o.copies = re;
            }
            if (op.flags) |f| o.flags = m.ref(f);
            try ops.append(arena, o);
            count += op.outputs;
        }
        value_at[i] = m.ref(pp.result);
    }
    const result = value_at[sh.root];
    _ = bank.validate(values.items.len, ops.items, result) catch return null;
    return graph.Builder.encode(a, .{ .values = values.items, .ops = ops.items, .result = result }) catch |e| switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        else => null,
    };
}

/// A plan's own references (inputs, then constants, then its outputs) to packet references.
const Map = struct {
    ins: []const u32,
    consts: u32,
    n_consts: u32,
    base: u32,

    fn ref(m: Map, r: u32) u32 {
        const n_in: u32 = @intCast(m.ins.len);
        if (r < n_in) return m.ins[r];
        if (r < n_in + m.n_consts) return m.consts + (r - n_in);
        return m.base + (r - n_in - m.n_consts);
    }
};

/// The most hex digits a bare literal cell converts: 16^100 has 121 decimal digits, under the bank's 128.
pub const max_hex = 100;

/// A bare hex literal cell (`0xFEDC… =`): Horner's rule on the bank over its digits, so the circuit does the radix.
pub fn hexLiteral(a: Allocator, bank: *const graph.Bank, cell: api.Cell) Allocator.Error!?[]u8 {
    if ((cell.form != .inline_expr and cell.form != .repl) or !std.mem.eql(u8, cell.lang, "python")) return null;
    const src = std.mem.trim(u8, cell.code, " \t");
    if (src.len < 4 or src.len > 2 + max_hex or src[0] != '0' or (src[1] != 'x' and src[1] != 'X')) return null;
    const hex = src[2..];
    for (hex) |c| if (!std.ascii.isHex(c)) return null;
    var arena_state = std.heap.ArenaAllocator.init(a);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const values = try arena.alloc([]const u8, hex.len + 1);
    values[0] = "16";
    for (hex, 1..) |c, i| values[i] = try std.fmt.allocPrint(arena, "{d}", .{std.fmt.charToDigit(c, 16) catch unreachable});
    var ops: std.ArrayList(packet.Op) = .empty;
    var acc: u32 = 1;
    for (2..values.len) |k| {
        // after k digits the value is below 16^k: at most k*1.21 + 1 decimal digits (log10 16 < 1.21)
        const w = for ([_]u8{ 1, 2, 4, 8, 16, 32, 62, 128 }) |x| {
            if (k * 121 / 100 + 1 <= x) break x;
        } else return null;
        if (bank.find("mul", w) == null or bank.find("add", w) == null) return null;
        const digit: u32 = @intCast(k);
        const product: u32 = @intCast(values.len + ops.items.len);
        try ops.append(arena, .{ .op = "mul", .width = w, .args = try arena.dupe(u32, &.{ acc, 0 }), .checked = true });
        try ops.append(arena, .{ .op = "add", .width = w, .args = try arena.dupe(u32, &.{ product, digit }), .checked = true });
        acc = product + 1;
    }
    _ = bank.validate(values.len, ops.items, acc) catch return null;
    return packet.Builder.program(a, values, ops.items, &.{.{ .ref = acc }}) catch |e| return if (e == error.OutOfMemory) error.OutOfMemory else null;
}

/// A frontend that runs `inner`, then moves a pure-integer calc program onto the graph bank; calc and ELF otherwise.
pub const Routed = struct {
    inner: api.Frontend,
    bank: *const graph.Bank,

    pub fn frontend(r: *Routed) api.Frontend {
        return .{ .ctx = r, .translate = translate };
    }

    /// The graph steps before the inner frontend (no compile): T1's binding, a straight-line program, a whole script.
    fn graphFirst(r: *const Routed, a: Allocator, cell: api.Cell) Allocator.Error!?[]u8 {
        const source_frontend = @import("runner_frontend.zig");
        if (cell.form == .inline_expr and r.inner.ctx != null and
            r.inner.translate == source_frontend.frontend().translate)
        {
            const context: *const source_frontend.Context = @ptrCast(@alignCast(r.inner.ctx.?));
            if (!context.exact_inline) return null;
        }
        if (try hexLiteral(a, r.bank, cell)) |bytes| return bytes;
        if (try @import("runner_graph_edit.zig").cell(a, r.bank, cell)) |bytes| return bytes;
        if (try graph.bindCell(a, r.bank, cell)) |bytes| return bytes;
        if (cell.form != .inline_expr and std.mem.eql(u8, cell.lang, "python") and !prog.shadowed(cell.prior))
            if (try prog.program(a, r.bank, cell.code)) |bytes| return bytes;
        // a whole script the frontend checks runs on the bank before any compile; earlier cells' names stay on its path
        if (cell.form != .inline_expr and std.mem.eql(u8, cell.lang, "python") and cell.prior.len == 0)
            if (try @import("runner_graph_table.zig").translate(a, r.bank, cell.code)) |bytes| return bytes;
        // an inline `gcd(a, b) =` (the scanner's literal call) is math.gcd's plan, as the printed Python call
        if (cell.form == .inline_expr and std.mem.startsWith(u8, cell.code, "gcd(") and cell.code.len <= 160) {
            var src: [200]u8 = undefined;
            const py = std.fmt.bufPrint(&src, "from math import gcd\nprint({s})", .{cell.code}) catch unreachable;
            if (try prog.program(a, r.bank, py)) |bytes| return bytes;
        }
        return null;
    }

    fn translate(ctx: ?*anyopaque, a: Allocator, cell: api.Cell) Allocator.Error!api.Translation {
        const r: *Routed = @ptrCast(@alignCast(ctx.?));
        // R5: a literal-only inline value (`140 mod 9`, `10^9 mod 11`) reads no earlier cell, so it runs without them
        if (cell.form == .inline_expr and cell.prior.len != 0 and closedLiteral(cell.code)) {
            var alone = cell;
            alone.prior = &.{};
            return translate(ctx, a, alone);
        }
        if (try r.graphFirst(a, cell)) |bytes| return .{ .program = .{ .kind = .graph, .bytes = bytes } };
        const t = try r.inner.translate(r.inner.ctx, a, cell);
        if (t == .abstain) if (try prog.direct(a, r.bank, cell, t.abstain)) |bytes| return .{ .program = .{ .kind = .graph, .bytes = bytes } };
        if (t != .program or t.program.kind != .calc) return t;
        const bytes = try lower(a, r.bank, t.program.bytes) orelse return t;
        a.free(t.program.bytes); // the calc program the graph replaces
        return .{ .program = .{ .kind = .graph, .bytes = bytes } };
    }
};

/// An inline value of literals only: digits, `+ - * ^`, brackets, the word `mod`; no name, call, `%`, `.`, `/` or `e`.
pub fn closedLiteral(code: []const u8) bool {
    var i: usize = 0;
    var digit = false;
    while (i < code.len) : (i += 1) {
        if (std.mem.startsWith(u8, code[i..], " mod ")) {
            i += 4;
            continue;
        }
        switch (code[i]) {
            '0'...'9' => digit = true,
            ' ', '(', ')', '+', '-', '*', '^' => {},
            else => return false,
        }
    }
    return digit;
}

/// <prefix>_SYNC_GRAPH on the engine thread: the whole route for an inline value, else its graph steps only.
pub const Quick = struct {
    routed: *Routed,

    pub fn frontend(q: *Quick) api.Frontend {
        return .{ .ctx = q, .translate = translate };
    }

    fn translate(ctx: ?*anyopaque, a: Allocator, cell: api.Cell) Allocator.Error!api.Translation {
        const q: *Quick = @ptrCast(@alignCast(ctx.?));
        if (cell.form == .inline_expr) return Routed.translate(q.routed, a, cell);
        const bytes = try q.routed.graphFirst(a, cell) orelse return .{ .abstain = .unsupported };
        return .{ .program = .{ .kind = .graph, .bytes = bytes } };
    }
};

/// Graph programs to the graph backend, calc and ELF to the circuit (runner_graph_dispatch.zig).
pub const Dispatch = @import("runner_graph_dispatch.zig").Dispatch;
