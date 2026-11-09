const ast = @import("runner_frontend_integer_ast.zig");

const Refused = error{Refused};

// Source-based RV32 instruction weights need calibration against the selected compiler and optimization mode.
pub fn estimate(parser: *const ast.Parser, root: usize, budget: usize) ?usize {
    return estimateMany(parser, &.{root}, budget);
}

// One shared traversal charges arithmetic once and formatting separately for every ordered output.
pub fn estimateMany(parser: *const ast.Parser, roots: []const usize, budget: usize) ?usize {
    if (parser.count == 0 or parser.count > 64 or roots.len == 0 or roots.len > 8) return null;
    var counter = Counter{ .parser = parser, .root = 0, .budget = budget };
    counter.charge(200) catch return null;
    for (roots) |root| {
        if (root >= parser.count) return null;
        counter.root = root;
        const value = counter.visit(root) catch return null;
        const op = parser.nodes[root].op;
        counter.charge(if (isComparison(op)) 160 else formatCost(value, true) + 16) catch return null;
        if (op == .divmod) {
            const n = parser.nodes[root];
            const remainder = @mod(parser.nodes[n.l].value, parser.nodes[n.r].value);
            counter.charge(formatCost(remainder, true) + 64) catch return null;
        }
    }
    return counter.total;
}

fn isComparison(op: ast.Op) bool {
    return switch (op) {
        .eq, .ne, .lt, .le, .gt, .ge => true,
        else => false,
    };
}

const Counter = struct {
    parser: *const ast.Parser,
    root: usize,
    budget: usize,
    total: usize = 0,
    seen: u64 = 0,

    fn charge(self: *Counter, amount: usize) Refused!void {
        if (amount > self.budget - self.total) {
            self.total = self.budget +| 1;
            return error.Refused;
        }
        self.total += amount;
    }

    fn child(self: *Counter, index: usize, parent: usize) Refused!i512 {
        if (index >= parent) return error.Refused;
        if (self.parser.nodes[index].op == .divmod) return error.Refused;
        return self.visit(index);
    }

    fn visit(self: *Counter, index: usize) Refused!i512 {
        if (index >= self.parser.count or index >= 64) return error.Refused;
        const n = self.parser.nodes[index];
        try bounded(n.value);
        const mask = @as(u64, 1) << @intCast(index);
        if (self.seen & mask != 0) return n.value;
        self.seen |= mask;
        try self.charge(160);
        if (n.op == .literal) {
            if (n.text.len == 0 or n.text.len > 4096 or (n.base != 2 and n.base != 10 and n.base != 16))
                return error.Refused;
            const digit_cost: usize = if (n.base == 16) 132 else 128;
            try self.charge(if (n.base == 10) 80 + n.text.len * 8 else 48 + n.text.len * digit_cost);
            var value: i512 = 0;
            for (n.text, 0..) |c, i| {
                if (n.base == 10 and i % 4 == 0) try self.charge(40 + @max(words(value), 1) * 20);
                const digit: u8 = switch (c) {
                    '0'...'9' => c - '0',
                    'a'...'f' => c - 'a' + 10,
                    'A'...'F' => c - 'A' + 10,
                    else => return error.Refused,
                };
                if (digit >= n.base) return error.Refused;
                value = value * n.base + digit;
                try bounded(value);
            }
            if (n.value != (if (n.value < 0) -value else value)) return error.Refused;
            return n.value;
        }
        if (!self.parser.proved[index] or (n.op == .divmod and index != self.root)) return error.Refused;
        const a = try self.child(n.l, index);
        if ((n.op == .digits or n.op == .digit_sum) and isComparison(self.parser.nodes[n.l].op))
            return error.Refused;
        const b: i512 = switch (n.op) {
            .neg, .digits, .isqrt, .factorial, .digit_sum => 0,
            .add, .sub, .mul, .floor, .mod, .pow, .powmod, .divmod, .gcd, .range_product, .eq, .ne, .lt, .le, .gt, .ge => try self.child(n.r, index),
            else => return error.Refused,
        };
        const k: i512 = switch (n.op) {
            .powmod, .range_product => try self.child(n.k, index),
            else => 0,
        };
        const value: i512 = switch (n.op) {
            .neg => blk: {
                try self.charge(80);
                break :blk -a;
            },
            .add, .sub => blk: {
                try self.charge(220);
                break :blk if (n.op == .add) a + b else a - b;
            },
            .mul => blk: {
                try self.charge(multiplyCost(a, b));
                break :blk try product(a, b);
            },
            .floor, .mod, .divmod => blk: {
                if (b == 0) return error.Refused;
                var cached = false;
                if (n.op == .mod) {
                    for (self.parser.nodes[0..self.parser.count], 0..) |prior, j| {
                        if ((prior.op == .floor or prior.op == .divmod) and prior.l == n.l and prior.r == n.r and
                            self.seen & (@as(u64, 1) << @intCast(j)) != 0) cached = true;
                    }
                }
                try self.charge(if (cached) 80 else divisionCost(a, b) + 140);
                break :blk if (n.op == .mod) @mod(a, b) else @divFloor(a, b);
            },
            .eq, .ne, .lt, .le, .gt, .ge => blk: {
                try self.charge(180);
                break :blk @intFromBool(switch (n.op) {
                    .eq => a == b,
                    .ne => a != b,
                    .lt => a < b,
                    .le => a <= b,
                    .gt => a > b,
                    .ge => a >= b,
                    else => unreachable,
                });
            },
            .pow, .powmod => try self.power(a, b, if (n.op == .powmod) k else null),
            .digits => blk: {
                try self.charge(formatCost(a, false));
                break :blk @as(i512, @intCast(decimalDigits(a) + @intFromBool(a < 0)));
            },
            .gcd => try self.gcd(a, b),
            .isqrt => try self.isqrt(a),
            .factorial => try self.factorial(a),
            .range_product => try self.range(a, b, k),
            .digit_sum => try self.digitSum(a),
            else => return error.Refused,
        };
        try bounded(value);
        if (value != n.value) return error.Refused;
        return value;
    }

    fn gcd(self: *Counter, a: i512, b: i512) Refused!i512 {
        var x = positive(a);
        var y = positive(b);
        try self.charge(80);
        var count: usize = 0;
        while (y != 0) {
            if (count == 512) return error.Refused;
            count += 1;
            try self.charge(divisionCost(x, y) + 120);
            const remainder = @mod(x, y);
            x = y;
            y = remainder;
        }
        return x;
    }

    fn isqrt(self: *Counter, a: i512) Refused!i512 {
        if (a < 0) return error.Refused;
        try self.charge(160);
        if (a == 0) return 0;
        var x = @as(i512, 1) << @intCast((bits(a) + 1) / 2);
        var count: usize = 0;
        while (true) {
            if (count == 256) return error.Refused;
            count += 1;
            try self.charge(divisionCost(a, x) + 220 + smallDivisionCost(0, 8, 2) + 100);
            const y = @divTrunc(x + @divTrunc(a, x), 2);
            if (y >= x) return x;
            x = y;
        }
    }

    fn factorial(self: *Counter, a: i512) Refused!i512 {
        if (a < 0 or a > 48) return error.Refused;
        try self.charge(100);
        var result: i512 = 1;
        var factor: i512 = 2;
        while (factor <= a) : (factor += 1) {
            try self.charge(multiplyCost(result, factor) + 80);
            result = try product(result, factor);
        }
        return result;
    }

    fn range(self: *Counter, start: i512, stop: i512, step: i512) Refused!i512 {
        if (step == 0) return error.Refused;
        try self.charge(160);
        if (if (step > 0) start >= stop else start <= stop) return 1;
        const distance = if (step > 0) stop - start else start - stop;
        const stride = positive(step);
        const length = @divTrunc(distance, stride) + @intFromBool(@mod(distance, stride) != 0);
        if (length > 10000) return error.Refused;
        try self.charge(220 + divisionCost(distance, stride) + 140 + 80);
        if (@mod(distance, stride) != 0) try self.charge(220);
        var value = start;
        var result: i512 = 1;
        var count: usize = 0;
        while (if (step > 0) value < stop else value > stop) {
            if (count == 10000) return error.Refused;
            count += 1;
            try bounded(value);
            try self.charge(180 + multiplyCost(result, value));
            result = try product(result, value);
            if (result == 0) return 0;
            try self.charge(220);
            value += step;
        }
        return result;
    }

    fn digitSum(self: *Counter, a: i512) Refused!i512 {
        if (a < 0) return error.Refused;
        try self.charge(80);
        var value = a;
        var result: i512 = 0;
        while (value != 0) {
            try self.charge(smallDivisionCost(@intCast(value), 8, 10) + 64);
            result += @mod(value, 10);
            value = @divTrunc(value, 10);
        }
        return result;
    }

    fn power(self: *Counter, a: i512, exponent: i512, modulus: ?i512) Refused!i512 {
        if (exponent < 0) return error.Refused;
        try self.charge(160);
        var base = a;
        var result: i512 = 1;
        var m: i512 = 0;
        if (modulus) |signed| {
            if (signed == 0) return error.Refused;
            m = positive(signed);
            if (m == 1) return 0;
            try self.charge(divisionCost(a, m) + 100);
            base = @mod(a, m);
            if (a < 0 and base != 0) try self.charge(220);
        }
        const count = bits(exponent);
        var remaining = exponent;
        for (0..count) |bit| {
            try self.charge(64);
            if (@mod(remaining, 2) != 0) result = try self.powerProduct(result, base, m);
            remaining = @divTrunc(remaining, 2);
            if (bit + 1 < count) base = try self.powerProduct(base, base, m);
        }
        if (modulus) |signed| {
            if (signed < 0 and result != 0) {
                try self.charge(220);
                result -= m;
            }
        }
        return result;
    }

    fn powerProduct(self: *Counter, a: i512, b: i512, modulus: i512) Refused!i512 {
        if (modulus == 0) {
            try self.charge(multiplyCost(a, b));
            return product(a, b);
        }
        const wide = a * b;
        try self.charge(productCost(a, b) + 120);
        if (words(modulus) == 1) {
            try self.charge(smallDivisionCost(@intCast(wide), 16, @intCast(modulus)));
        } else {
            try self.charge(160 + bits(wide) * 224);
        }
        return @mod(wide, modulus);
    }
};

fn bounded(value: i512) Refused!void {
    if (value <= -ast.limit or value >= ast.limit) return error.Refused;
}

fn product(a: i512, b: i512) Refused!i512 {
    const value = a * b;
    try bounded(value);
    return value;
}

fn positive(value: i512) i512 {
    return if (value < 0) -value else value;
}

fn bits(value: i512) usize {
    const magnitude: u512 = @intCast(positive(value));
    return @intCast(512 - @clz(magnitude));
}

fn words(value: i512) usize {
    return (bits(value) + 31) / 32;
}

fn decimalDigits(value: i512) usize {
    var magnitude = positive(value);
    var count: usize = 1;
    while (magnitude >= 10) : (magnitude = @divTrunc(magnitude, 10)) count += 1;
    return count;
}

// Charge the supplied limb span; small divisors use two native halfword divisions per limb.
fn smallDivisionCost(value: u512, limbs: usize, divisor: u32) usize {
    if (divisor <= 65535) return 24 + limbs * 20;
    var cost: usize = 24;
    var remainder: u64 = 0;
    var index = limbs;
    while (index != 0) {
        index -= 1;
        const word: u32 = @truncate(value >> @intCast(index * 32));
        cost += if (remainder == 0) @as(usize, 14) else 32 * 16 + 20;
        remainder = ((remainder << 32) | word) % divisor;
    }
    return cost;
}

fn divisionCost(a: i512, b: i512) usize {
    const numerator = positive(a);
    const divisor = positive(b);
    if (numerator < divisor) return 160;
    if (divisor <= 65535) return 120 + smallDivisionCost(@intCast(numerator), words(a), @intCast(divisor));
    return longDivisionCost(numerator, divisor);
}

// Replay only quotient estimation for cost admission; the emitter still retains all original arithmetic nodes.
fn longDivisionCost(a: i512, b: i512) usize {
    const na = (bits(a) + 15) / 16;
    const nb = (bits(b) + 15) / 16;
    const shift: u9 = @intCast(16 * nb - bits(b));
    const divisor = b << shift;
    var remaining = a << shift;
    const top_shift: u9 = @intCast(16 * (nb - 1));
    const second_shift: u9 = @intCast(16 * (nb - 2));
    const top = divisor >> top_shift;
    const second = (divisor >> second_shift) & 65535;
    var total: usize = 240 + (na + nb) * 20;
    var position = na - nb + 1;
    while (position != 0) {
        position -= 1;
        const digit_shift: u9 = @intCast(16 * position);
        const window = remaining >> digit_shift;
        const prefix = window >> top_shift;
        var digit = @min(@divTrunc(prefix, top), 65535);
        var rem = prefix - digit * top;
        total += 112 + nb * 20;
        while (rem < 65536 and digit * second > rem * 65536 + ((window >> second_shift) & 65535)) {
            digit -= 1;
            rem += top;
            total += 24;
        }
        const exact = @divTrunc(window, divisor);
        if (digit > exact) total += 24 + nb * 16;
        remaining -= (exact * divisor) << digit_shift;
    }
    return total;
}

fn productCost(a: i512, b: i512) usize {
    return 200 + words(a) * words(b) * 24 + words(a) * 10;
}

fn multiplyCost(a: i512, b: i512) usize {
    return productCost(a, b) + 140;
}

fn formatCost(value: i512, print: bool) usize {
    const digits = decimalDigits(value);
    const sign: usize = @intFromBool(value < 0);
    var magnitude = positive(value);
    var format: usize = 64 + digits * 12 + sign * 12;
    while (true) {
        format += smallDivisionCost(@intCast(magnitude), words(magnitude), 10000) + 24;
        magnitude = @divTrunc(magnitude, 10000);
        if (magnitude == 0) break;
    }
    return format + (if (print) 48 + (digits + sign) * 16 else @as(usize, 0));
}
