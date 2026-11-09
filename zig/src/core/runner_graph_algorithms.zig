//! Integer algorithms orchestrate prepared FFNs and transfer raw rows without decoding operands.
const std = @import("std");
const api = @import("runner_api.zig");
const packet = @import("runner_graph_packet.zig");
const factor = @import("runner_graph_factor.zig");
const End = @FieldType(api.RunEnd, "end");
const Cancel = ?*const std.atomic.Value(bool);

/// Executes an algorithm in request-local scratch and copies only its typed outputs back to the caller.
pub fn compute(m: anytype, kind: []const u8, width: u16, precision: u8, refs: []const u32, dest: usize, budget: u64, out: *std.ArrayList(u8), cancel: Cancel) !End {
    if (std.mem.startsWith(u8, kind, "f64_")) return @import("runner_graph_f64.zig").compute(m, kind, width, refs, dest, budget, out, cancel);
    if (std.mem.eql(u8, kind, "factorize")) return error.GraphAlgorithm;
    const family = std.meta.stringToEnum(packet.Algorithm, kind) orelse return error.GraphAlgorithm;
    const outputs = packet.algorithmOutputs(family);
    if (!packet.algorithmWidthValid(family, width) or refs.len != packet.algorithmArity(family) or dest + outputs > m.values.len)
        return error.GraphAlgorithmScratch;
    if (precision > 18 or (family != .fixed_div and precision != 0)) return error.GraphPrecision;
    const original = m.values;
    const zero = try packet.literal("0");
    var scratch: [32]packet.Value = @splat(zero);
    scratch[1] = try packet.literal("1");
    for (refs, 0..) |ref, i| {
        if (ref >= original.len) return error.GraphReference;
        for (width..packet.max_digits) |digit| {
            const bits = original[ref][digit / 31][2 + digit % 31 * 4 ..][0..4];
            for (bits) |bit| if (bit != 0) return error.GraphNarrowBinding;
        }
        scratch[2 + i] = original[ref];
    }
    m.values = &scratch;
    defer m.values = original;
    const work_width = if (std.mem.eql(u8, kind, "smallest_factor")) @max(width, 8) else width;
    var c: Context(@TypeOf(m)) = .{ .m = m, .width = work_width, .precision = precision, .budget = budget, .out = out, .cancel = cancel, .input_digits = occupiedDigits(&scratch[2]) };
    const result = c.calculate(kind) catch |err| switch (err) {
        error.GraphAlgorithmStopped => return c.end,
        error.GraphAlgorithmLimit => return .budget,
        error.GraphAlgorithmArithmetic => blk: {
            scratch[0][0][127] = packet.bit_scale;
            break :blk [2]u32{ 0, 0 };
        },
        else => return err,
    };
    const values = [2]packet.Value{ scratch[result[0]], scratch[result[1]] };
    @memcpy(original[dest .. dest + outputs], values[0..outputs]);
    return .halted;
}

/// Named legacy algorithms retain their token schedules; new fractional formats use the compiled formatter.
pub fn run(m: anytype, kind: []const u8, width: u16, budget: u64, out: *std.ArrayList(u8), cancel: Cancel) !api.RunEnd {
    if (std.mem.startsWith(u8, kind, "f64_")) return @import("runner_graph_f64.zig").run(m, kind, budget, out, cancel);
    if (std.mem.eql(u8, kind, "factorize")) return runFactorize(m, width, budget, out, cancel);
    const family = std.meta.stringToEnum(packet.Algorithm, kind) orelse return error.GraphAlgorithm;
    const refs = [_]u32{ 0, 1, 2 };
    var end = try compute(m, kind, width, m.precision, refs[0..packet.algorithmArity(family)], 0, budget, out, cancel);
    if (end != .halted) return .{ .end = end, .steps = m.steps };
    if (std.mem.eql(u8, kind, "smallest_factor")) {
        end = try @import("runner_graph_print.zig").emit(m, .{ .ref = 0 }, budget, out, cancel);
        return .{ .end = end, .steps = m.steps };
    }
    if (family == .fraction or family == .fixed_div) {
        end = try @import("runner_graph_print.zig").emit(m, .{ .ref = 0, .second = if (family == .fraction) 1 else null, .style = if (family == .fraction) .fraction else .fixed, .precision = m.precision }, budget, out, cancel);
        return .{ .end = end, .steps = m.steps };
    }
    var c: Context(@TypeOf(m)) = .{ .m = m, .width = width, .budget = budget, .out = out, .cancel = cancel, .input_digits = 1 };
    const failed = m.values[0][0][127];
    if (failed != 0 and failed != packet.bit_scale) return error.GraphPredicate;
    if (family == .divmod and failed == 0) {
        c.emitNamed("emit_first", 0) catch |err| switch (err) {
            error.GraphAlgorithmStopped => return .{ .end = c.end, .steps = m.steps },
            else => return err,
        };
        c.emitNamed("emit_last", 1) catch |err| switch (err) {
            error.GraphAlgorithmStopped => return .{ .end = c.end, .steps = m.steps },
            else => return err,
        };
    } else c.emit(0) catch |err| switch (err) {
        error.GraphAlgorithmStopped => return .{ .end = c.end, .steps = m.steps },
        else => return err,
    };
    return .{ .end = .halted, .steps = m.steps };
}

fn runFactorize(m: anytype, width: u16, budget: u64, out: *std.ArrayList(u8), cancel: Cancel) !api.RunEnd {
    if (!packet.widthValid(width) or width > 128 or m.values.len < 2) return error.GraphAlgorithmScratch;
    if (m.precision != 0) return error.GraphPrecision;
    const original = m.values;
    const zero = try packet.literal("0");
    var scratch: [32]packet.Value = @splat(zero);
    scratch[1] = try packet.literal("1");
    for (original[0..2], 0..) |value, i| {
        for (width..packet.max_digits) |digit| {
            for (value[digit / 31][2 + digit % 31 * 4 ..][0..4]) |bit|
                if (bit != 0) return error.GraphNarrowBinding;
        }
        scratch[2 + i] = value;
    }
    m.values = &scratch;
    defer m.values = original;
    var c: Context(@TypeOf(m)) = .{ .m = m, .width = @max(width, 8), .budget = budget, .out = out, .cancel = cancel, .input_digits = occupiedDigits(&scratch[2]) };
    factor.factorize(&c) catch |err| switch (err) {
        error.GraphAlgorithmStopped => return .{ .end = c.end, .steps = m.steps },
        error.GraphAlgorithmLimit => return .{ .end = .budget, .steps = m.steps },
        error.GraphAlgorithmArithmetic => {
            scratch[0][0][127] = packet.bit_scale;
            const end = try @import("runner_graph_print.zig").emit(m, .{ .ref = 0 }, budget, out, cancel);
            return .{ .end = end, .steps = m.steps };
        },
        else => return err,
    };
    return .{ .end = .halted, .steps = m.steps };
}

fn Context(comptime M: type) type {
    return struct {
        m: M,
        width: u16,
        budget: u64,
        out: *std.ArrayList(u8),
        cancel: Cancel,
        input_digits: usize,
        precision: u8 = 0,
        end: End = .halted,

        const Self = @This();

        pub fn call(c: *Self, name: []const u8, refs: []const u32, dest: usize) !void {
            const width = if (std.mem.eql(u8, name, "meta")) @max(c.width, 8) else c.width;
            var needed: usize = 1;
            for (refs) |ref| needed = @max(needed, occupiedDigits(&c.m.values[ref]));
            if (std.mem.eql(u8, name, "mul")) {
                needed = occupiedDigits(&c.m.values[refs[0]]) + occupiedDigits(&c.m.values[refs[1]]);
            } else if (std.mem.eql(u8, name, "add") or std.mem.eql(u8, name, "sub") or std.mem.eql(u8, name, "scale10")) {
                needed += 1;
            } else if (std.mem.eql(u8, name, "meta")) needed = @max(needed, 8);
            const selected = bucket(@min(needed, width));
            const t = (if (c.m.adaptive_widths) c.m.bank.find(name, selected) else null) orelse
                c.m.bank.find(name, width) orelse return error.GraphTemplate;
            c.end = try c.m.invoke(t, refs, dest, c.budget, c.out, c.cancel, false);
            if (c.end != .halted) return error.GraphAlgorithmStopped;
            for (dest..dest + t.outputs) |i| try clean(&c.m.values[i]);
        }

        fn emit(c: *Self, ref: u32) !void {
            return c.emitNamed("emit", ref);
        }

        fn emitNamed(c: *Self, name: []const u8, ref: u32) !void {
            const needed = if (c.m.adaptive_widths) occupiedDigits(&c.m.values[ref]) else c.width;
            const width: u16 = if (needed > 128) 256 else if (needed > 62) 128 else 62;
            const t = c.m.bank.find(name, width) orelse return error.GraphEmitMissing;
            c.end = try c.m.invoke(t, &.{ref}, 0, c.budget, c.out, c.cancel, true);
            if (c.end != .halted) return error.GraphAlgorithmStopped;
        }

        pub fn move(c: *Self, dest: usize, source: usize) void {
            c.m.values[dest] = c.m.values[source];
        }

        pub fn meta(c: *Self, a: u32, b: u32) !void {
            try c.call("meta", &.{ a, b }, 20);
        }

        pub fn flag(c: *Self, digit: usize) !bool {
            return predicate(&c.m.values[22], digit);
        }

        pub fn nonzero(c: *Self, ref: u32) !bool {
            try c.meta(ref, 0);
            return c.flag(0);
        }

        /// Slots 24..30 are private to long division; both result slots must be outside that interval.
        fn unsigned(c: *Self, a: u32, b: u32, q: usize, r: usize) !void {
            c.move(24, a);
            c.move(25, b);
            c.move(26, 0);
            c.move(27, 0);
            var digit: usize = c.width;
            while (digit > 0) {
                digit -= 1;
                c.move(28, 0);
                packet.copyBits(&c.m.values[28], &c.m.values[24], .{
                    .ref = 24,
                    .from = @intCast(digit),
                    .digits = 1,
                });
                try c.call("divstep", &.{ 27, 25, 28 }, 29);
                c.move(27, 29);
                packet.copyBits(&c.m.values[26], &c.m.values[30], .{
                    .ref = 30,
                    .to = @intCast(digit),
                    .digits = 1,
                });
            }
            c.move(q, 26);
            c.move(r, 27);
        }

        fn calculate(c: *Self, kind: []const u8) ![2]u32 {
            if (std.mem.eql(u8, kind, "smallest_factor")) return .{ try factor.smallest(c), 0 };
            if (std.mem.eql(u8, kind, "divmod") or std.mem.eql(u8, kind, "divexact")) {
                try c.meta(2, 3);
                c.move(5, 20);
                c.move(6, 21);
                try c.unsigned(5, 6, 7, 8);
                if (std.mem.eql(u8, kind, "divexact")) {
                    if (try c.nonzero(8)) return error.GraphAlgorithmArithmetic;
                    try c.call("round_div", &.{ 7, 8, 2, 3 }, 10);
                    return .{ 10, 0 };
                }
                try c.call("divfinal", &.{ 7, 8, 2, 3 }, 10);
                return .{ 10, 11 };
            }
            if (std.mem.eql(u8, kind, "fraction")) return c.fraction();
            if (std.mem.eql(u8, kind, "fixed_div")) return .{ try c.fixedDivision(), 0 };
            const result: u32 = if (std.mem.eql(u8, kind, "gcd")) try c.gcd() else if (std.mem.eql(u8, kind, "isqrt")) try c.isqrt() else if (std.mem.eql(u8, kind, "digit_sum")) try c.digitSum() else if (std.mem.eql(u8, kind, "powmod")) try c.powmod() else if (std.mem.eql(u8, kind, "square_offset")) try c.squareOffset() else try c.product(std.mem.eql(u8, kind, "factorial"));
            return .{ result, 0 };
        }

        fn fraction(c: *Self) ![2]u32 {
            const divisor = try c.gcd();
            c.move(15, divisor);
            try c.meta(2, 3);
            if (!try c.flag(1)) return error.GraphAlgorithmArithmetic;
            c.move(5, 20);
            c.move(6, 21);
            try c.unsigned(5, 15, 7, 8);
            try c.unsigned(6, 15, 9, 10);
            try c.call("divfinal", &.{ 7, 0, 2, 3 }, 10);
            return .{ 10, 9 };
        }

        fn fixedDivision(c: *Self) !u32 {
            try c.meta(2, 3);
            if (!try c.flag(1)) return error.GraphAlgorithmArithmetic;
            c.move(5, 20);
            c.move(6, 21);
            try c.unsigned(5, 6, 7, 8);
            for (0..c.precision) |_| {
                try c.call("scale10", &.{7}, 7);
                try c.call("divstep", &.{ 8, 6, 0 }, 10);
                c.move(8, 10);
                try c.call("add", &.{ 7, 11 }, 7);
            }
            try c.call("round_div", &.{ 7, 8, 2, 3 }, 10);
            return 10;
        }

        fn gcd(c: *Self) !u32 {
            try c.meta(2, 3);
            c.move(5, 20);
            c.move(6, 21);
            var iterations: usize = 0;
            while (try c.nonzero(6)) {
                if (iterations >= @as(usize, c.width) * 8) return error.GraphAlgorithmLimit;
                iterations += 1;
                try c.unsigned(5, 6, 7, 8);
                c.move(5, 6);
                c.move(6, 8);
            }
            return 5;
        }

        fn isqrt(c: *Self) !u32 {
            try c.meta(2, 0);
            if (try c.flag(2)) return error.GraphAlgorithmArithmetic;
            if (!try c.flag(0)) return 0;
            c.move(5, 20);
            const nines: [packet.max_digits]u8 = @splat('9');
            c.m.values[6] = try packet.literal(nines[0 .. (c.input_digits + 1) / 2]);
            for (0..@as(usize, c.width) * 8) |_| {
                try c.unsigned(5, 6, 7, 8);
                try c.call("half", &.{6}, 12);
                c.move(9, 12);
                c.move(10, 13);
                try c.call("half", &.{7}, 12);
                try c.call("add", &.{ 9, 12 }, 11);
                if (try predicate(&c.m.values[10], 0)) {
                    if (try predicate(&c.m.values[13], 0)) {
                        try c.call("add", &.{ 11, 1 }, 11);
                    }
                }
                try c.meta(11, 6);
                if (!try c.flag(4)) return 6;
                c.move(6, 11);
            }
            return error.GraphAlgorithmLimit;
        }

        fn digitSum(c: *Self) !u32 {
            try c.call("digit_sum", &.{2}, 6);
            return 6;
        }

        fn squareOffset(c: *Self) !u32 {
            try c.meta(2, 3);
            if (!try c.flag(1)) return 0;
            try c.call("add", &.{ 2, 3 }, 5);
            try c.call("add", &.{ 2, 5 }, 6);
            try c.call("mul", &.{ 6, 3 }, 7);
            return 7;
        }

        /// Range products use [start, stop); factorial includes n and never constructs n+1.
        fn product(c: *Self, factorial: bool) !u32 {
            try c.meta(2, if (factorial) 0 else 3);
            if (try c.flag(2)) return error.GraphAlgorithmArithmetic;
            if (try c.flag(3)) return error.GraphAlgorithmArithmetic;
            c.move(5, if (factorial) 1 else 20);
            c.move(6, if (factorial) 20 else 21);
            c.move(7, 1);
            var iterations: usize = 0;
            while (true) {
                try c.meta(5, 6);
                const less = try c.flag(4);
                const equal = try c.flag(5);
                if (!less and !(factorial and equal)) return 7;
                if (iterations >= 10000) return error.GraphAlgorithmLimit;
                iterations += 1;
                try c.call("mul", &.{ 7, 5 }, 7);
                if (factorial and equal) return 7;
                try c.call("add", &.{ 5, 1 }, 5);
            }
        }

        /// One prepared Horner step consumes a decimal multiplier digit; modulus stays in slot 7.
        fn modmul(c: *Self, a: u32, b: u32, dest: usize) !void {
            c.move(17, a);
            c.move(18, b);
            c.move(16, 0);
            var digit = occupiedDigits(&c.m.values[18]);
            while (digit > 0) {
                digit -= 1;
                c.move(19, 0);
                packet.copyBits(&c.m.values[19], &c.m.values[18], .{
                    .ref = 18,
                    .from = @intCast(digit),
                    .digits = 1,
                });
                try c.call("mulmodstep", &.{ 16, 17, 19, 7 }, 16);
            }
            c.move(dest, 16);
        }

        fn powmod(c: *Self) !u32 {
            try c.meta(2, 4);
            c.move(5, 20);
            c.move(7, 21);
            c.move(9, 22);
            if (!try c.flag(1)) return error.GraphAlgorithmArithmetic;
            try c.meta(3, 0);
            if (try c.flag(2)) return error.GraphAlgorithmArithmetic;
            c.move(6, 20);
            try c.unsigned(5, 7, 10, 11);
            try c.call("divfinal", &.{ 10, 11, 2, 7 }, 10);
            c.move(5, 11);
            try c.unsigned(1, 7, 10, 11);
            c.move(8, 11);
            var iterations: usize = 0;
            while (try c.nonzero(6)) {
                if (iterations >= @as(usize, c.width) * 4) return error.GraphAlgorithmLimit;
                iterations += 1;
                try c.call("half", &.{6}, 12);
                c.move(6, 12);
                if (try predicate(&c.m.values[13], 0)) try c.modmul(8, 5, 8);
                if (try c.nonzero(6)) try c.modmul(5, 5, 5);
            }
            if (try predicate(&c.m.values[9], 3)) {
                if (try c.nonzero(8)) try c.call("sub", &.{ 8, 7 }, 8);
            }
            return 8;
        }
    };
}

fn occupiedDigits(value: *const packet.Value) usize {
    var digits: usize = packet.max_digits;
    while (digits > 1) : (digits -= 1) {
        const at = digits - 1;
        for (value[at / 31][2 + at % 31 * 4 ..][0..4]) |bit| if (bit != 0) return digits;
    }
    return digits;
}

fn bucket(needed: usize) u16 {
    for ([_]u16{ 1, 2, 4, 8, 16, 32, 62, 128, 256 }) |width| if (needed <= width) return width;
    unreachable;
}

fn clean(value: *const packet.Value) !void {
    const bit = value[0][127];
    if (bit == packet.bit_scale) return error.GraphAlgorithmArithmetic;
    if (bit != 0) return error.GraphPredicate;
}

fn predicate(value: *const packet.Value, digit: usize) !bool {
    try clean(value);
    const bits = value[digit / 31][2 + digit % 31 * 4 ..][0..4];
    for (bits[1..]) |bit| if (bit != 0) return error.GraphPredicate;
    if (bits[0] == 0) return false;
    if (bits[0] == packet.bit_scale) return true;
    return error.GraphPredicate;
}
