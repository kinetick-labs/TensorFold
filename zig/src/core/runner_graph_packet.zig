//! Bounded literal binding packets and pure frontend serialization; operations contain no executable graph IR.
const std = @import("std");
const Allocator = std.mem.Allocator;
pub const max_packet = 1 << 20;
pub const max_ops = 8192;
pub const max_values = 8192;
pub const max_arity = 16;
pub const max_prints = 128;
pub const max_output = 4096;
pub const max_digits: usize = 256;
pub const max_groups = (max_digits + 30) / 31;
pub const Value = [max_groups][128]f32;
pub const bit_scale: f32 = 128;
pub const reference_scale: f32 = 16777216;

pub const Copy = struct {
    ref: u32,
    from: u16 = 0,
    to: u16 = 0,
    digits: u16 = 62,
};

pub const Op = struct {
    op: []const u8,
    width: u16 = 62,
    a: u32 = 0,
    b: ?u32 = null,
    args: ?[]const u32 = null,
    outputs: u8 = 1,
    copies: []const Copy = &.{},
    flags: ?u32 = null,
    precision: u8 = 0,
    checked: bool = false,
};

pub const Style = enum { integer, fixed, comma, integral_float, boolean, fraction, tuple, byte, prepared };
pub const Ending = enum { newline, space, empty, comma_space, close, slash, close_space, close_empty, close_comma_space };
pub const Print = struct {
    ref: u32,
    second: ?u32 = null,
    style: Style = .integer,
    precision: u8 = 0,
    ending: Ending = .newline,
    prefix: bool = false,
};

pub const Packet = struct {
    version: u32 = 1,
    values: []const []const u8,
    plan: ?[]const u8 = null,
    ops: []const Op = &.{},
    result: u32 = 0,
    prints: []const Print = &.{},
};

pub const Plan = struct {
    name: []const u8,
    inputs: u32,
    literals: []const []const u8 = &.{},
    ops: []const Op,
    result: u32,
    algorithm: ?[]const u8 = null,
    width: u16 = 62,
    precision: u8 = 0,
};

pub const Family = enum { add, sub, mul, div, mod, pow };
pub const Algorithm = enum { powmod, gcd, isqrt, range_product, factorial, digit_sum, divmod, square_offset, fraction, fixed_div, divexact, smallest_factor, factorize, f64_ratio, f64_mul, f64_sub, f64_isclose, f64_repr, f64_format };
pub const Shape = enum { sum, squares, factorial, fib, list_sum, sorted_distance };

pub const Builder = struct {
    pub fn encode(a: Allocator, packet: Packet) ![]u8 {
        if (packet.version != 1 or packet.values.len > max_values or packet.ops.len > max_ops or packet.prints.len > max_prints)
            return error.GraphPacketSize;
        const bytes = try std.json.Stringify.valueAlloc(a, packet, .{});
        errdefer a.free(bytes);
        if (bytes.len > max_packet) return error.GraphPacketSize;
        return bytes;
    }

    /// Binds a complete SSA program to preloaded kernels and compiled output formatters.
    pub fn program(a: Allocator, values: []const []const u8, ops: []const Op, prints: []const Print) ![]u8 {
        if (prints.len == 0) return error.GraphPrint;
        for (values) |value| _ = try literal(value);
        return encode(a, .{ .values = values, .ops = ops, .prints = prints });
    }

    pub fn named(a: Allocator, name: []const u8, values: []const []const u8) ![]u8 {
        return encode(a, .{ .values = values, .plan = name });
    }

    /// Selects a preloaded plan from operand metadata without evaluating its answer.
    pub fn binary(a: Allocator, family: Family, lhs: []const u8, rhs: []const u8) ![]u8 {
        return binaryWidth(a, family, lhs, rhs, 128);
    }

    /// Optional basic capacity is explicit; callers select 256 only with an extended loaded bank.
    pub fn binaryWidth(a: Allocator, family: Family, lhs: []const u8, rhs: []const u8, max_width: u16) ![]u8 {
        if (max_width != 128 and max_width != 256) return error.GraphTemplate;
        _ = try literal(lhs);
        _ = try literal(rhs);
        const wa = lhs.len - @intFromBool(lhs[0] == '-' or lhs[0] == '+');
        const wb = rhs.len - @intFromBool(rhs[0] == '-' or rhs[0] == '+');
        if (wa > max_width or wb > max_width) return error.GraphLiteral;
        if (family == .pow) {
            const exponent = std.fmt.parseInt(i16, rhs, 10) catch return error.GraphExponent;
            if (exponent < 0 or exponent > 999) return error.GraphExponent;
            const legacy_needed = wa * @max(@as(usize, 1), @as(usize, @intCast(exponent)));
            const needed = if (max_width == 256 and legacy_needed > 128) @max(@as(usize, 128), @max(wa, powerDigits(lhs, @intCast(exponent)))) else legacy_needed;
            const width: u16 = if (max_width == 256 and needed > 128) 256 else if (needed > 62) 128 else 62;
            const name = try std.fmt.allocPrint(a, "pow-{d}-{d}", .{ exponent, width });
            defer a.free(name);
            return named(a, name, &.{lhs});
        }
        const needed = switch (family) {
            .add, .sub => @max(wa, wb) + 1,
            .mul => wa + wb,
            .div, .mod => @max(wa, wb),
            .pow => unreachable,
        };
        const width = for ([_]u16{ 1, 2, 4, 8, 16, 32, 62, 128, 256 }) |n| {
            if (n >= @min(@as(usize, max_width), needed)) break n;
        } else unreachable;
        const name = try std.fmt.allocPrint(a, "{s}-{d}", .{ @tagName(family), width });
        defer a.free(name);
        return named(a, name, &.{ lhs, rhs });
    }

    /// Selects an offline integer algorithm using operand widths without evaluating the result.
    pub fn integer(a: Allocator, family: Algorithm, operands: []const []const u8) ![]u8 {
        if (family == .fixed_div) return error.GraphPrecision;
        const arity = algorithmArity(family);
        if (operands.len != arity) return error.GraphSlotArity;
        var needed: usize = 1;
        for (operands) |operand| {
            _ = try literal(operand);
            const digits = operand.len - @intFromBool(operand[0] == '-' or operand[0] == '+');
            if (digits > (if (family == .divmod or family == .powmod) @as(usize, 256) else 128)) return error.GraphLiteral;
            needed = @max(needed, digits);
        }
        const width: u16 = if (family == .factorial or family == .range_product or
            std.mem.startsWith(u8, @tagName(family), "f64_")) 128 else blk: {
            for ([_]u16{ 1, 2, 4, 8, 16, 32, 62, 128, 256 }) |n| if (needed <= n) break :blk n;
            unreachable;
        };
        const name = try std.fmt.allocPrint(a, "{s}-{d}", .{ @tagName(family), width });
        defer a.free(name);
        return named(a, name, operands);
    }

    /// Precision is bounded schedule metadata; all quotient and rounding work stays in prepared FFNs.
    pub fn fixedDivision(a: Allocator, lhs: []const u8, rhs: []const u8, places: u8) ![]u8 {
        if (places > 18) return error.GraphPrecision;
        _ = try literal(lhs);
        _ = try literal(rhs);
        if (lhs.len - @intFromBool(lhs[0] == '-' or lhs[0] == '+') > 128 or
            rhs.len - @intFromBool(rhs[0] == '-' or rhs[0] == '+') > 128)
            return error.GraphNarrowBinding;
        const needed = @max(lhs.len - @intFromBool(lhs[0] == '-' or lhs[0] == '+') + places + 1, rhs.len - @intFromBool(rhs[0] == '-' or rhs[0] == '+'));
        const width = for ([_]u8{ 1, 2, 4, 8, 16, 32, 62, 128 }) |n| {
            if (n >= @min(@as(usize, 128), needed)) break n;
        } else unreachable;
        const name = try std.fmt.allocPrint(a, "fixed_div-{d}-{d}", .{ places, width });
        defer a.free(name);
        return named(a, name, &.{ lhs, rhs });
    }

    /// Binds a recognized bounded-loop shape to its offline control schedule.
    pub fn loop(a: Allocator, shape: Shape, bound: u16, values: []const []const u8) ![]u8 {
        return loopWidth(a, shape, bound, values, 62);
    }

    pub fn loopWidth(a: Allocator, shape: Shape, bound: u16, values: []const []const u8, width: u16) ![]u8 {
        if (shape == .sorted_distance) {
            if (!widthValid(width) or width > 128 or bound > 16) return error.GraphLoop;
        } else if (width != 62 and width != 128) return error.GraphLoop;
        if (bound > 128 or values.len != (if (shape == .list_sum or shape == .sorted_distance) @as(usize, bound) else 0))
            return error.GraphLoop;
        for (values) |value| {
            _ = try literal(value);
            if (value.len - @intFromBool(value[0] == '-' or value[0] == '+') > 128)
                return error.GraphNarrowBinding;
        }
        const name = try std.fmt.allocPrint(a, "loop-{s}-{d}-{d}", .{ @tagName(shape), bound, width });
        defer a.free(name);
        return named(a, name, values);
    }

    /// Selects a bounded compare-exchange schedule; pair count is control metadata, never operand arithmetic.
    pub fn sortedDistance(a: Allocator, values: []const []const u8) ![]u8 {
        if (values.len > 16) return error.GraphLoop;
        var digits: usize = 1;
        for (values) |value| {
            _ = try literal(value);
            const occupied = value.len - @intFromBool(value[0] == '-' or value[0] == '+');
            if (occupied > 128) return error.GraphNarrowBinding;
            digits = @max(digits, occupied);
        }
        var text: [8]u8 = undefined;
        const pairs = values.len * (values.len -| 1);
        const overhead = (try std.fmt.bufPrint(&text, "{d}", .{pairs})).len;
        const width = for ([_]u8{ 1, 2, 4, 8, 16, 32, 62, 128 }) |n| {
            if (n >= @min(@as(usize, 128), digits + overhead)) break n;
        } else unreachable;
        return loopWidth(a, .sorted_distance, @intCast(values.len), values, width);
    }
};

pub fn algorithmArity(family: Algorithm) u8 {
    return switch (family) {
        .f64_mul, .f64_sub => 4,
        .f64_isclose => 8,
        .f64_ratio, .f64_repr, .f64_format => 2,
        .powmod => 3,
        .gcd, .range_product, .divmod, .square_offset, .fraction, .fixed_div, .divexact, .smallest_factor, .factorize => 2,
        .isqrt, .factorial, .digit_sum => 1,
    };
}

pub fn algorithmOutputs(family: Algorithm) u8 {
    return if (family == .divmod or family == .fraction or family == .f64_ratio or family == .f64_mul or family == .f64_sub or family == .f64_format) 2 else 1;
}

pub fn algorithmWidthValid(family: Algorithm, width: u16) bool {
    return widthValid(width) and (width <= 128 or (width == 256 and (family == .divmod or family == .powmod)));
}

// Width 255 is an internal alias for unchanged lazy binding head coverage; public banks reject it.
pub fn widthValid(n: u16) bool {
    return switch (n) {
        1, 2, 4, 8, 16, 32, 62, 128, 255, 256 => true,
        else => false,
    };
}

pub fn literal(text: []const u8) !Value {
    if (text.len == 0) return error.GraphLiteral;
    const signed = text[0] == '-' or text[0] == '+';
    const digits = text[@intFromBool(signed)..];
    if (digits.len == 0 or digits.len > max_digits) return error.GraphLiteral;
    var value: Value = @splat(@splat(0));
    for (&value) |*row| {
        row[0] = reference_scale;
        row[1] = reference_scale;
    }
    value[0][126] = if (text[0] == '-') bit_scale else 0;
    for (digits, 0..) |ch, i| {
        if (ch < '0' or ch > '9') return error.GraphLiteral;
        const digit = digits.len - i - 1;
        for (0..4) |bit| value[digit / 31][2 + digit % 31 * 4 + bit] =
            if ((ch - '0') >> @as(u3, @intCast(bit)) & 1 != 0) bit_scale else 0;
    }
    return value;
}

/// A literal magnitude bound chooses storage; no power result is evaluated on the host.
fn powerDigits(text: []const u8, exponent: u16) usize {
    if (exponent == 0) return 1;
    const digits = text[@intFromBool(text[0] == '-' or text[0] == '+')..];
    const magnitude = std.fmt.parseInt(u1024, digits, 10) catch return digits.len * exponent;
    const bits: usize = if (magnitude <= 1) 0 else 1024 - @as(usize, @clz(magnitude - 1));
    // 30103/100000 is a strict upper bound on log10(2).
    return bits * exponent * 30103 / 100000 + 1;
}

/// Chooses an output template from occupied tensor bits without decoding the integer.
pub fn outputWidth(value: *const Value) u16 {
    var width: u16 = 62;
    for (62..max_digits) |digit| {
        const bits = value[digit / 31][2 + digit % 31 * 4 ..][0..4];
        for (bits) |bit| if (bit != 0) {
            if (digit >= 128) return 256;
            width = 128;
        };
    }
    return width;
}

pub fn copyBits(out: *Value, source: *const Value, c: Copy) void {
    for (0..c.digits) |i| {
        const src = @as(usize, c.from) + i;
        const dst = @as(usize, c.to) + i;
        @memcpy(out[dst / 31][2 + dst % 31 * 4 ..][0..4], source[src / 31][2 + src % 31 * 4 ..][0..4]);
    }
}

test "frontend binding selects offline plans and rejects invalid controls" {
    const t = std.testing;
    const sum = try Builder.binary(t.allocator, .add, "-999", "1");
    defer t.allocator.free(sum);
    try t.expect(std.mem.indexOf(u8, sum, "add-4") != null);
    const power = try Builder.binary(t.allocator, .pow, "2", "100");
    defer t.allocator.free(power);
    try t.expect(std.mem.indexOf(u8, power, "pow-100-128") != null);
    try t.expectError(error.GraphExponent, Builder.binary(t.allocator, .pow, "2", "-1"));
    try t.expectError(error.GraphLiteral, Builder.binary(t.allocator, .mul, "12x", "3"));
    const squares = try Builder.loop(t.allocator, .squares, 10, &.{});
    defer t.allocator.free(squares);
    try t.expect(std.mem.indexOf(u8, squares, "loop-squares-10-62") != null);
    try t.expectError(error.GraphLoop, Builder.loop(t.allocator, .squares, 129, &.{}));
    try t.expectError(error.GraphLoop, Builder.loop(t.allocator, .list_sum, 2, &.{"1"}));
}

test "128 digit bindings span five groups and retain explicit wide plans" {
    const t = std.testing;
    const text: [128]u8 = @splat('9');
    const too_wide: [max_digits + 1]u8 = @splat('9');
    const value = try literal(&text);
    try t.expectEqual(bit_scale, value[4][14]);
    try t.expectEqual(@as(f32, 0), value[4][15]);
    try t.expectError(error.GraphLiteral, literal(&too_wide));
    var copied: Value = @splat(@splat(0));
    copyBits(&copied, &value, .{ .ref = 0, .from = 124, .to = 30, .digits = 4 });
    try t.expectEqual(bit_scale, copied[0][122]);
    try t.expectEqual(bit_scale, copied[1][10]);
    const sum = try Builder.binary(t.allocator, .add, &text, "1");
    defer t.allocator.free(sum);
    try t.expect(std.mem.indexOf(u8, sum, "add-128") != null);
    const power = try Builder.binary(t.allocator, .pow, &text, "1");
    defer t.allocator.free(power);
    try t.expect(std.mem.indexOf(u8, power, "pow-1-128") != null);
    const loop = try Builder.loopWidth(t.allocator, .factorial, 80, &.{}, 128);
    defer t.allocator.free(loop);
    try t.expect(std.mem.indexOf(u8, loop, "loop-factorial-80-128") != null);
    try t.expectError(error.GraphLoop, Builder.loopWidth(t.allocator, .fib, 10, &.{}, 32));
}

test "256 digit basic bindings preserve the final group and refuse unsupported algorithms" {
    const t = std.testing;
    const text: [256]u8 = @splat('9');
    const value = try literal(&text);
    try t.expectEqual(@as(usize, 9), max_groups);
    try t.expect(widthValid(256));
    try t.expect(algorithmWidthValid(.divmod, 256));
    try t.expect(algorithmWidthValid(.powmod, 256));
    const modular = try Builder.integer(t.allocator, .powmod, &.{ &text, "3", &text });
    defer t.allocator.free(modular);
    try t.expect(std.mem.indexOf(u8, modular, "powmod-256") != null);
    try t.expectError(error.GraphLiteral, Builder.integer(t.allocator, .gcd, &.{ &text, "1" }));
    try t.expect(!algorithmWidthValid(.gcd, 256));
    try t.expect(!algorithmWidthValid(.divmod, 255));
    try t.expect(!widthValid(257));
    try t.expectError(error.GraphTemplate, Builder.binaryWidth(t.allocator, .add, "1", "1", 255));
    try t.expectEqual(@as(u16, 256), outputWidth(&value));
    try t.expectEqual(bit_scale, value[8][30]);
    try t.expectEqual(@as(f32, 0), value[8][31]);
    var copied: Value = @splat(@splat(0));
    copyBits(&copied, &value, .{ .ref = 0, .from = 248, .to = 248, .digits = 8 });
    try t.expectEqual(bit_scale, copied[8][30]);
    try t.expectEqual(@as(u16, 256), outputWidth(&copied));
    var complete: Value = @splat(@splat(0));
    copyBits(&complete, &value, .{ .ref = 0, .digits = 256 });
    try t.expectEqual(bit_scale, complete[8][30]);
    const excessive: [257]u8 = @splat('9');
    try t.expectError(error.GraphLiteral, literal(&excessive));
    try t.expectError(error.GraphLiteral, Builder.binaryWidth(t.allocator, .mul, &excessive, "1", 256));
    for ([_]usize{ 129, 191, 256 }) |digits| {
        const boundary = try literal(text[0..digits]);
        try t.expectEqual(@as(u16, 256), outputWidth(&boundary));
        const product = try Builder.binaryWidth(t.allocator, .mul, text[0..digits], "1", 256);
        defer t.allocator.free(product);
        try t.expect(std.mem.indexOf(u8, product, "mul-256") != null);
    }
    const narrow = try literal("1");
    try t.expectEqual(@as(u16, 62), outputWidth(&narrow));
    const old: [128]u8 = @splat('9');
    const old_value = try literal(&old);
    try t.expectEqual(@as(u16, 128), outputWidth(&old_value));
    const sum = try Builder.binaryWidth(t.allocator, .add, &text, "1", 256);
    defer t.allocator.free(sum);
    try t.expect(std.mem.indexOf(u8, sum, "add-256") != null);
    const power = try Builder.binaryWidth(t.allocator, .pow, "3", "400", 256);
    defer t.allocator.free(power);
    try t.expect(std.mem.indexOf(u8, power, "pow-400-256") != null);
    const zero_power = try Builder.binaryWidth(t.allocator, .pow, &text, "0", 256);
    defer t.allocator.free(zero_power);
    try t.expect(std.mem.indexOf(u8, zero_power, "pow-0-256") != null);
    try t.expectError(error.GraphLiteral, Builder.binary(t.allocator, .add, &text, "0"));
    const narrow_power = try Builder.binaryWidth(t.allocator, .pow, "2", "400", 256);
    defer t.allocator.free(narrow_power);
    try t.expect(std.mem.indexOf(u8, narrow_power, "pow-400-128") != null);
    const division = try Builder.binaryWidth(t.allocator, .div, &text, "1", 256);
    defer t.allocator.free(division);
    try t.expect(std.mem.indexOf(u8, division, "div-256") != null);
    const remainder = try Builder.binaryWidth(t.allocator, .mod, &text, "1", 256);
    defer t.allocator.free(remainder);
    try t.expect(std.mem.indexOf(u8, remainder, "mod-256") != null);
    try t.expectError(error.GraphNarrowBinding, Builder.fixedDivision(t.allocator, &text, "1", 0));
    try t.expectError(error.GraphLiteral, Builder.integer(t.allocator, .gcd, &.{ &text, "1" }));
    try t.expectError(error.GraphLoop, Builder.loopWidth(t.allocator, .sorted_distance, 0, &.{}, 256));
    try t.expectError(error.GraphNarrowBinding, Builder.sortedDistance(t.allocator, &.{&text}));
}

test "integer algorithms bind arity and maximum operand width" {
    const t = std.testing;
    const narrow: [40]u8 = @splat('9');
    const wide_text: [63]u8 = @splat('9');
    const too_wide: [129]u8 = @splat('9');
    const modular = try Builder.integer(t.allocator, .powmod, &.{ &narrow, "3", &narrow });
    defer t.allocator.free(modular);
    try t.expect(std.mem.indexOf(u8, modular, "powmod-62") != null);
    const wide = try Builder.integer(t.allocator, .gcd, &.{ &wide_text, "3" });
    defer t.allocator.free(wide);
    try t.expect(std.mem.indexOf(u8, wide, "gcd-128") != null);
    const power = try Builder.binary(t.allocator, .pow, "2", "10");
    defer t.allocator.free(power);
    try t.expect(std.mem.indexOf(u8, power, "pow-10-62") != null);
    try t.expectError(error.GraphSlotArity, Builder.integer(t.allocator, .powmod, &.{ "2", "3" }));
    try t.expectError(error.GraphLiteral, Builder.integer(t.allocator, .isqrt, &.{"x"}));
    try t.expectError(error.GraphLiteral, Builder.integer(t.allocator, .factorial, &.{&too_wide}));
}

test "whole-cell bindings retain operands and bound formatting and schedule controls" {
    const t = std.testing;
    const whole = try Builder.program(t.allocator, &.{ "-7", "3" }, &.{.{ .op = "divmod", .width = 1, .a = 0, .b = 1, .outputs = 2, .checked = true }}, &.{.{ .ref = 2, .second = 3, .style = .tuple }});
    defer t.allocator.free(whole);
    var parsed = try std.json.parseFromSlice(Packet, t.allocator, whole, .{});
    defer parsed.deinit();
    try t.expectEqualStrings("-7", parsed.value.values[0]);
    try t.expectEqualStrings("3", parsed.value.values[1]);
    try t.expect(parsed.value.ops[0].checked);
    try t.expectEqual(Style.tuple, parsed.value.prints[0].style);
    try t.expectEqual(@as(?u32, 3), parsed.value.prints[0].second);
    const fixed = try Builder.fixedDivision(t.allocator, "1", "8", 3);
    defer t.allocator.free(fixed);
    try t.expect(std.mem.indexOf(u8, fixed, "fixed_div-3-") != null);
    try t.expectError(error.GraphPrecision, Builder.fixedDivision(t.allocator, "1", "8", 19));
    const sorted = try Builder.sortedDistance(t.allocator, &.{ "-2", "1", "5" });
    defer t.allocator.free(sorted);
    try t.expect(std.mem.indexOf(u8, sorted, "loop-sorted_distance-3-") != null);
    const too_many: [17][]const u8 = @splat("0");
    try t.expectError(error.GraphLoop, Builder.sortedDistance(t.allocator, &too_many));
    const too_many_prints: [max_prints + 1]Print = @splat(.{ .ref = 0 });
    try t.expectError(error.GraphPacketSize, Builder.program(t.allocator, &.{"1"}, &.{}, &too_many_prints));
}
