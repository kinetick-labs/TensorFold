//! A whole-script integer node's value, folded once with every bound checked (integer_ast Parser.prove).
const ast = @import("runner_frontend_integer_ast.zig");
const Parser = ast.Parser;
const Invalid = ast.Invalid;
const bound = ast.bound;

pub fn prove(self: *Parser, index: usize, root: usize) Invalid!i512 {
    const n = self.nodes[index];
    if (n.op == .divmod and index != root) return error.Unsupported;
    if (self.proved[index]) return n.value;
    if (n.op == .literal) return n.value;
    const a = try self.prove(n.l, root);
    const b = switch (n.op) {
        .neg, .digits, .isqrt, .factorial, .digit_sum => 0,
        else => try self.prove(n.r, root),
    };
    const result = switch (n.op) {
        .literal => unreachable,
        .neg => -a,
        .add => a + b,
        .sub => a - b,
        .mul => a * b,
        .eq => @intFromBool(a == b),
        .ne => @intFromBool(a != b),
        .lt => @intFromBool(a < b),
        .le => @intFromBool(a <= b),
        .gt => @intFromBool(a > b),
        .ge => @intFromBool(a >= b),
        .floor, .mod, .divmod => blk: {
            if (b == 0 or (n.op == .divmod and index != root)) return error.Unsupported;
            if (n.op == .mod) break :blk @mod(a, b);
            const q = @divFloor(a, b);
            if (n.op == .divmod) _ = try bound(@mod(a, b));
            break :blk q;
        },
        .pow => try power(a, b),
        .powmod => blk: {
            const m = try self.prove(n.k, root);
            if (m == 0 or b < 0) return error.Unsupported;
            var exponent = b;
            var base = @mod(a, m);
            var result = @mod(@as(i512, 1), m);
            while (exponent != 0) {
                if (@mod(exponent, 2) != 0) result = @mod(result * base, m);
                exponent = @divTrunc(exponent, 2);
                if (exponent != 0) base = @mod(base * base, m);
            }
            break :blk result;
        },
        .gcd => blk: {
            var x = if (a < 0) -a else a;
            var y = if (b < 0) -b else b;
            while (y != 0) {
                const r = @mod(x, y);
                x = y;
                y = r;
            }
            break :blk x;
        },
        .isqrt => blk: {
            if (a < 0) return error.Unsupported;
            var x = a;
            var y = @divTrunc(x + 1, 2);
            while (y < x) {
                x = y;
                y = @divTrunc(x + @divTrunc(a, x), 2);
            }
            break :blk x;
        },
        .factorial => blk: {
            if (a < 0 or a > 48) return error.Unsupported;
            var result: i512 = 1;
            var x: i512 = 2;
            while (x <= a) : (x += 1) result = try bound(result * x);
            break :blk result;
        },
        .range_product => blk: {
            const step = try self.prove(n.k, root);
            if (step == 0) return error.Unsupported;
            var x = a;
            var count: usize = 0;
            var result: i512 = 1;
            while (if (step > 0) x < b else x > b) : (x += step) {
                if (count == 10000) return error.Unsupported;
                count += 1;
                result = try bound(result * try bound(x));
            }
            break :blk result;
        },
        .digit_sum => blk: {
            if (self.boolean(n.l) or a < 0) return error.Unsupported;
            var x = a;
            var result: i512 = 0;
            while (x != 0) : (x = @divTrunc(x, 10)) result += @mod(x, 10);
            break :blk result;
        },
        .digits => blk: {
            if (self.boolean(n.l)) return error.Unsupported;
            var magnitude = if (a < 0) -a else a;
            var count: i512 = if (a < 0) 2 else 1;
            while (magnitude >= 10) : (magnitude = @divTrunc(magnitude, 10)) count += 1;
            break :blk count;
        },
    };
    self.nodes[index].value = try bound(result);
    self.proved[index] = true;
    return result;
}

fn power(a: i512, b: i512) Invalid!i512 {
    if (b < 0) return error.Unsupported;
    var exponent = b;
    var base = a;
    var result: i512 = 1;
    while (exponent != 0) {
        if (@mod(exponent, 2) != 0) result = try bound(result * base);
        exponent = @divTrunc(exponent, 2);
        if (exponent != 0) base = try bound(base * base);
    }
    return result;
}
