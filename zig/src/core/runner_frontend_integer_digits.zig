//! A number's digits in the integer lowering: int(str(x)[::-1]) and `while x > 0: ... x //= L`, unrolled by place.
const std = @import("std");
const ast = @import("runner_frontend_integer_ast.zig");
const suite = @import("runner_frontend_integer_suite.zig");
const Parser = ast.Parser;
const Invalid = ast.Invalid;

/// Decimal digits a reversal may unroll.
const max_digits = 30;

/// `int(str(x)[::-1])` after `int(str(`: x's digits read back, each one past the first gated by x >= 10^k.
pub fn reversed(p: *Parser, depth: usize) Invalid!usize {
    const x = try p.expr(depth + 1);
    try p.need(")");
    try p.need("[::-1]");
    try p.need(")");
    if (p.check_only) return x;
    if (p.boolean(x)) return error.Unsupported;
    const v = try p.prove(x, x);
    if (v < 0) return error.Unsupported; // int("21-") raises ValueError
    p.looped = true;
    var exact: usize = 1;
    var m = v;
    while (m >= 10) : (m = @divTrunc(m, 10)) exact += 1;
    if (exact > max_digits) return error.Unsupported;
    // headroom, as a range loop's: a power of two of places, unless the graph route could not hold it
    const spare = @min(max_digits, std.math.ceilPowerOfTwo(usize, @max(exact, 2)) catch max_digits);
    const mark = p.mark();
    if (spare > exact) {
        if (horner(p, x, spare)) |root| {
            if (!suite.heavy(p, root)) return root;
        } else |_| {}
        p.reset(mark);
    }
    return horner(p, x, exact);
}

/// r = x % 10, then r = r * (1 + 9c) + c * d for each further place: c = x >= 10^k, d = x // 10^k % 10.
fn horner(p: *Parser, x: usize, places: usize) Invalid!usize {
    const ten = try suite.literal(p, 10);
    const nine = try suite.literal(p, 9);
    const one = try suite.literal(p, 1);
    var r = try suite.node(p, .mod, x, ten);
    var place: i512 = 1;
    for (1..places) |_| {
        place *= 10;
        const pk = try suite.literal(p, place);
        const c = try suite.node(p, .ge, x, pk);
        const d = try suite.node(p, .mod, try suite.node(p, .floor, x, pk), ten);
        const grow = try suite.node(p, .add, one, try suite.node(p, .mul, c, nine));
        r = try suite.node(p, .add, try suite.node(p, .mul, r, grow), try suite.node(p, .mul, c, d));
    }
    return r;
}

/// `while x > 0:` (`x >= 1`, `x != 0`, `0 < x`) with one `x //= L`: trip k runs where x0 >= L^k, reading x0 // L^k.
pub fn whileDivides(r: *suite.Run, cond: []const u8, body: usize, end: usize) Invalid!bool {
    const p = r.p;
    // no trip's test reads another trip's state; anything else (false) is the general loop's
    if (r.pred != null) return false;
    var buf: [64]u8 = undefined;
    const test_text = compact(&buf, cond) orelse return false;
    const name = for ([_][]const u8{ ">0", ">=1", "!=0" }) |tail| {
        if (std.mem.endsWith(u8, test_text, tail)) break test_text[0 .. test_text.len - tail.len];
    } else if (std.mem.startsWith(u8, test_text, "0<")) test_text[2..] else return false;
    if (!@import("runner_frontend_integer_python.zig").validBinding(name)) return false;
    var at: ?usize = null;
    var divisor: i512 = 0;
    for (body..end) |i| {
        const line = r.lines.items[i];
        const target = suite.assigns(line.text) orelse continue;
        if (!std.mem.eql(u8, target, name)) continue;
        if (at != null or line.indent != r.lines.items[body].indent) return false;
        var lbuf: [64]u8 = undefined;
        const t = compact(&lbuf, line.text) orelse return false;
        if (!std.mem.startsWith(u8, t, name) or !std.mem.startsWith(u8, t[name.len..], "//=")) return false;
        const digits = t[name.len + 3 ..];
        if (digits.len == 0 or digits.len > 18) return false;
        for (digits) |c| if (!std.ascii.isDigit(c)) return false;
        divisor = std.fmt.parseInt(i512, digits, 10) catch return false;
        if (divisor < 2) return false;
        at = i;
    }
    const line = at orelse return false;
    const s = p.slot(name) orelse return false;
    if (p.kinds[s] or p.roots[s] == ast.unbound) return false;
    const x0 = p.roots[s];
    var v = p.nodes[x0].value;
    if (v < 0) return false; // `x != 0` would never end
    var trips: usize = 0;
    while (v > 0) : (v = @divFloor(v, divisor)) {
        trips += 1;
        if (trips > suite.max_trips) return error.Unsupported;
    }
    const mark = p.mark();
    const saved = suite.Saved.take(p);
    const spare = suite.schedule(trips);
    if (spare > trips) {
        if (unroll(r, name, x0, divisor, line, spare, body, end)) |_| {
            if (!suite.heavy(p, null)) return true;
        } else |_| {}
        p.reset(mark);
        saved.put(p);
    }
    try unroll(r, name, x0, divisor, line, trips, body, end);
    return true;
}

fn unroll(r: *suite.Run, name: []const u8, x0: usize, divisor: i512, line: usize, count: usize, body: usize, end: usize) Invalid!void {
    const p = r.p;
    var place: i512 = 1;
    for (0..count) |k| {
        const pk = try suite.literal(p, place);
        const c = try suite.node(p, .ge, x0, pk);
        const xk = if (k == 0) x0 else try suite.node(p, .floor, x0, pk);
        place = std.math.mul(i512, place, divisor) catch return error.Unsupported;
        if (place > ast.limit) return error.Unsupported;
        var inner = r.*;
        inner.pred = c;
        inner.divided = .{ .line = line, .name = name, .root = try suite.node(p, .floor, x0, try suite.literal(p, place)) };
        try suite.store(r, name, xk);
        if (try suite.exec(&inner, body, end, false) != null) return error.Unsupported;
    }
    // after the loop: 0 where it ran (the schedule covers its trips), x0 where it never did
    const one = try suite.literal(p, 1);
    const idle = try suite.node(p, .sub, one, try suite.node(p, .ge, x0, one));
    try suite.store(r, name, try suite.node(p, .mul, idle, x0));
}

/// `text` without spaces or tabs, or null past the buffer.
fn compact(buf: []u8, text: []const u8) ?[]const u8 {
    var n: usize = 0;
    for (text) |c| {
        if (c == ' ' or c == '\t') continue;
        if (n == buf.len) return null;
        buf[n] = c;
        n += 1;
    }
    return buf[0..n];
}
