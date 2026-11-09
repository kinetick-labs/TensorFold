//! Bounded binary64 operations use exact FFN integers and a single nearest-even rounding boundary.
const std = @import("std");
const packet = @import("runner_graph_packet.zig");
const context = @import("runner_graph_f64_context.zig");

pub fn calculate(c: anytype, kind: []const u8) ![2]u32 {
    if (std.mem.eql(u8, kind, "f64_ratio")) {
        if (context.occupiedDigits(&c.m.values[2]) > 62 or context.occupiedDigits(&c.m.values[3]) > 62)
            return error.GraphF64Arithmetic;
        try ratio(c, 2, 3, 32, 33);
    } else if (std.mem.eql(u8, kind, "f64_mul")) {
        try multiply(c, 2, 3, 4, 5, 32, 33);
    } else if (std.mem.eql(u8, kind, "f64_sub")) {
        try subtract(c, 2, 3, 4, 5, 32, 33);
    } else if (std.mem.eql(u8, kind, "f64_isclose")) {
        try isclose(c);
    } else return error.GraphAlgorithm;
    return .{ 32, 33 };
}

/// The canonical finite domain includes raw signed zero with exponent zero.
pub fn ensureFloat(c: anytype, m: u32, e: u32) !void {
    const inputs = [2]packet.Value{ c.m.values[m], c.m.values[e] };
    c.m.values[110] = inputs[0];
    c.m.values[111] = inputs[1];
    try constants(c);
    if (try c.less(111, 116) or try c.less(117, 111)) return error.GraphF64Arithmetic;
    try c.meta(110, 113);
    const nonzero = try c.flag(0);
    const below = try c.flag(4);
    if (!nonzero) {
        if (try c.nonzero(111)) return error.GraphF64Arithmetic;
        return;
    }
    if (below) return error.GraphF64Arithmetic;
    try c.meta(110, 114);
    if (!try c.flag(4)) return error.GraphF64Arithmetic;
}

/// Normalize an exact ratio before the final 53-bit quotient and remainder are rounded together.
pub fn ratio(c: anytype, n: u32, d: u32, out_m: usize, out_e: usize) !void {
    const inputs = [2]packet.Value{ c.m.values[n], c.m.values[d] };
    c.m.values[64] = inputs[0];
    c.m.values[65] = inputs[1];
    try constants(c);
    try c.meta(64, 65);
    if (!try c.flag(1)) return error.GraphF64Arithmetic;
    const negative = try c.flag(2) != try c.flag(3);
    c.move(66, 253);
    c.move(67, 254);
    c.move(68, 0);
    try c.literal(72, if (negative) "-1" else "1");
    if (!try c.nonzero(66)) {
        try c.call("decimal_mul", &.{ 0, 72 }, 69);
        return output(c, 69, 68, out_m, out_e);
    }
    var steps: usize = 0;
    while (try c.less(66, 67)) {
        try limit(&steps);
        try c.call("decimal_mul", &.{ 66, 112 }, 66);
        try c.call("sub", &.{ 68, 1 }, 68);
    }
    while (true) {
        try c.call("decimal_mul", &.{ 67, 112 }, 70);
        if (try c.less(66, 70)) break;
        try limit(&steps);
        c.move(67, 70);
        try c.call("add", &.{ 68, 1 }, 68);
    }
    try c.call("decimal_mul", &.{ 66, 113 }, 66);
    try c.literal(71, "52");
    try c.call("sub", &.{ 68, 71 }, 68);
    try c.divide(66, 67, 70, 71);
    try c.call("round_div", &.{ 70, 71, 72, 67 }, 69);
    try c.meta(69, 114);
    if (try c.flag(5)) {
        c.move(70, 253);
        try c.call("half", &.{70}, 73);
        try c.call("decimal_mul", &.{ 73, 72 }, 69);
        try c.call("add", &.{ 68, 1 }, 68);
    }
    try ensureFloat(c, 69, 68);
    try output(c, 69, 68, out_m, out_e);
}

/// Exact mantissa multiplication is truncated once, retaining guard and sticky predicates.
pub fn multiply(c: anytype, m1: u32, e1: u32, m2: u32, e2: u32, out_m: usize, out_e: usize) !void {
    try inputs4(c, m1, e1, m2, e2);
    try ensureFloat(c, 64, 65);
    try ensureFloat(c, 66, 67);
    try c.call("decimal_mul", &.{ 64, 66 }, 68);
    try c.call("add", &.{ 65, 67 }, 69);
    try normalize(c, 68, 69, out_m, out_e);
}

/// Alignment is exact in decimal lanes, followed by one rounding of the signed difference.
pub fn subtract(c: anytype, m1: u32, e1: u32, m2: u32, e2: u32, out_m: usize, out_e: usize) !void {
    try inputs4(c, m1, e1, m2, e2);
    try ensureFloat(c, 64, 65);
    try ensureFloat(c, 66, 67);
    try c.meta(64, 66);
    const both_zero = !try c.flag(0) and !try c.flag(1);
    const negative_zero = try c.flag(2) and !try c.flag(3);
    if (both_zero) {
        try c.literal(68, if (negative_zero) "-0" else "0");
        return output(c, 68, 0, out_m, out_e);
    }
    var steps: usize = 0;
    while (try c.less(65, 67)) {
        try limit(&steps);
        try c.call("decimal_mul", &.{ 66, 112 }, 66);
        try c.call("sub", &.{ 67, 1 }, 67);
    }
    while (try c.less(67, 65)) {
        try limit(&steps);
        try c.call("decimal_mul", &.{ 64, 112 }, 64);
        try c.call("sub", &.{ 65, 1 }, 65);
    }
    try c.call("sub", &.{ 64, 66 }, 68);
    try normalize(c, 68, 65, out_m, out_e);
}

fn normalize(c: anytype, m: u32, e: u32, out_m: usize, out_e: usize) !void {
    const inputs = [2]packet.Value{ c.m.values[m], c.m.values[e] };
    c.m.values[80] = inputs[0];
    c.m.values[81] = inputs[1];
    try constants(c);
    try c.meta(80, 0);
    const negative = try c.flag(2);
    c.move(84, 253);
    if (!try c.flag(0)) return output(c, 80, 0, out_m, out_e);
    var steps: usize = 0;
    while (try c.less(84, 113)) {
        try limit(&steps);
        try c.call("decimal_mul", &.{ 84, 112 }, 84);
        try c.call("sub", &.{ 81, 1 }, 81);
    }
    var guard = false;
    var sticky = false;
    while (!try c.less(84, 114)) {
        try limit(&steps);
        try c.call("half", &.{84}, 86);
        sticky = sticky or guard;
        guard = try c.nonzero(87);
        c.move(84, 86);
        try c.call("add", &.{ 81, 1 }, 81);
    }
    if (guard) {
        try c.call("half", &.{84}, 86);
        if (sticky or try c.nonzero(87)) try c.call("add", &.{ 84, 1 }, 84);
    }
    if (try c.equal(84, 114)) {
        try c.call("half", &.{84}, 86);
        c.move(84, 86);
        try c.call("add", &.{ 81, 1 }, 81);
    }
    try c.literal(88, if (negative) "-1" else "1");
    try c.call("decimal_mul", &.{ 84, 88 }, 89);
    try ensureFloat(c, 89, 81);
    try output(c, 89, 81, out_m, out_e);
}

fn isclose(c: anytype) !void {
    const inputs = [8]packet.Value{
        c.m.values[2], c.m.values[3], c.m.values[4], c.m.values[5],
        c.m.values[6], c.m.values[7], c.m.values[8], c.m.values[9],
    };
    for (inputs, 128..) |value, i| c.m.values[i] = value;
    for (0..4) |i| try ensureFloat(c, @intCast(128 + i * 2), @intCast(129 + i * 2));
    for ([_]u32{ 132, 134 }) |m| {
        try c.meta(m, 0);
        if (try c.flag(0) and try c.flag(2)) return error.GraphF64Arithmetic;
    }
    const previous = c.extended;
    c.extended = true;
    defer c.extended = previous;
    var close = try c.equal(128, 130) and try c.equal(129, 131);
    if (!close) {
        try subtract(c, 128, 129, 130, 131, 136, 137);
        try c.meta(136, 0);
        c.move(136, 253);
        try c.meta(128, 0);
        c.move(138, 253);
        try multiply(c, 132, 133, 138, 129, 138, 139);
        close = try positiveLE(c, 136, 137, 138, 139);
        if (!close) {
            try c.meta(130, 0);
            c.move(138, 253);
            try multiply(c, 132, 133, 138, 131, 138, 139);
            close = try positiveLE(c, 136, 137, 138, 139);
        }
        if (!close) close = try positiveLE(c, 136, 137, 134, 135);
    }
    c.move(32, if (close) 1 else 0);
    c.move(33, 0);
}

fn positiveLE(c: anytype, am: u32, ae: u32, bm: u32, be: u32) !bool {
    if (!try c.nonzero(am)) return true;
    if (!try c.nonzero(bm)) return false;
    if (try c.less(ae, be)) return true;
    if (!try c.equal(ae, be)) return false;
    return try c.less(am, bm) or try c.equal(am, bm);
}

fn inputs4(c: anytype, m1: u32, e1: u32, m2: u32, e2: u32) !void {
    const inputs = [4]packet.Value{ c.m.values[m1], c.m.values[e1], c.m.values[m2], c.m.values[e2] };
    for (inputs, 64..) |value, i| c.m.values[i] = value;
}

fn output(c: anytype, m: u32, e: u32, out_m: usize, out_e: usize) !void {
    if (out_m == out_e or out_m >= c.m.values.len or out_e >= c.m.values.len)
        return error.GraphAlgorithmScratch;
    const values = [2]packet.Value{ c.m.values[m], c.m.values[e] };
    c.m.values[out_m] = values[0];
    c.m.values[out_e] = values[1];
}

fn constants(c: anytype) !void {
    try c.literal(112, "2");
    try c.literal(113, "4503599627370496");
    try c.literal(114, "9007199254740992");
    try c.literal(115, "-1");
    try c.literal(116, if (c.extended) "-256" else "-100");
    try c.literal(117, if (c.extended) "256" else "100");
}

fn limit(steps: *usize) !void {
    if (steps.* >= 1000) return error.GraphF64Arithmetic;
    steps.* += 1;
}
