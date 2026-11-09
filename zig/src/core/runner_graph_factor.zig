//! Bounded trial division follows prepared FFN predicates and keeps operand arithmetic inside the graph bank.
const std = @import("std");
const packet = @import("runner_graph_packet.zig");
const printing = @import("runner_graph_print.zig");

const max_factors = packet.max_digits * 4;
const max_iterations = 10000 + max_factors;

/// Inputs are n>=1 and inclusive bound 1..10000; negative signs, including signed zero, are domain errors.
pub fn smallest(c: anytype) !u32 {
    return trial(c, null);
}

/// Emits (remaining, count, f1, ..., fk) with a newline; n=1 emits (1, 0), and zero is rejected.
pub fn factorize(c: anytype) !void {
    var factors: std.ArrayList(packet.Value) = .empty;
    defer factors.deinit(c.m.gpa);
    _ = try trial(c, &factors);
    var bytes: std.ArrayList(u8) = .empty;
    defer bytes.deinit(c.m.gpa);
    try emit(c, 5, .comma_space, true, &bytes);
    try emit(c, 9, if (factors.items.len == 0) .close else .comma_space, false, &bytes);
    for (factors.items, 0..) |value, i| {
        c.m.values[14] = value;
        try emit(c, 14, if (i + 1 == factors.items.len) .close else .comma_space, false, &bytes);
    }
    if (c.out.items.len > packet.max_output - bytes.items.len) return error.GraphOutputSize;
    try c.out.appendSlice(c.m.gpa, bytes.items);
}

fn emit(c: anytype, ref: u32, ending: packet.Ending, prefix: bool, bytes: *std.ArrayList(u8)) !void {
    c.end = try printing.emit(c.m, .{ .ref = ref, .ending = ending, .prefix = prefix }, c.budget, bytes, c.cancel);
    if (c.end != .halted) return error.GraphAlgorithmStopped;
}

/// Slots 5..11 hold remaining, candidate, quotient, remainder, count, two and the bound ceiling.
fn trial(c: anytype, factors: ?*std.ArrayList(packet.Value)) !u32 {
    try c.meta(2, 3);
    if (try c.flag(2) or try c.flag(3)) return error.GraphAlgorithmArithmetic;
    if (!try c.flag(0) or !try c.flag(1)) return error.GraphAlgorithmArithmetic;
    c.move(5, 20);
    c.m.values[11] = try packet.literal("10000");
    try c.meta(11, 3);
    if (try c.flag(4)) return error.GraphAlgorithmArithmetic;
    c.move(9, 0);
    try c.meta(5, 1);
    if (try c.flag(5)) return 1;
    c.m.values[10] = try packet.literal("2");
    c.move(6, 10);
    var even = true;
    for (0..max_iterations) |_| {
        if (even) {
            try c.call("half", &.{5}, 12);
            c.move(7, 12);
            c.move(8, 13);
        } else try divide(c);
        try c.meta(7, 6);
        if (try c.flag(4)) {
            if (factors) |list| {
                try retain(c, list, 5);
                c.move(5, 1);
            }
            return 5;
        }
        try c.meta(3, 6);
        if (try c.flag(4)) return 0;
        if (!try c.nonzero(8)) {
            if (factors) |list| {
                try retain(c, list, 6);
                c.move(5, 7);
                try c.meta(5, 1);
                if (try c.flag(5)) return 5;
                continue;
            }
            return 6;
        }
        try c.call("add", &.{ 6, if (even) @as(u32, 1) else 10 }, 6);
        even = false;
    }
    return error.GraphAlgorithmLimit;
}

fn retain(c: anytype, factors: *std.ArrayList(packet.Value), ref: u32) !void {
    if (factors.items.len >= max_factors) return error.GraphAlgorithmLimit;
    try factors.append(c.m.gpa, c.m.values[ref]);
    try c.call("add", &.{ 9, 1 }, 9);
}

/// Only occupied digit lanes are scheduled; quotient assembly copies FFN output bits without decoding them.
fn divide(c: anytype) !void {
    c.move(24, 5);
    c.move(25, 6);
    c.move(26, 0);
    c.move(27, 0);
    var digit: usize = c.width;
    while (digit > 1) : (digit -= 1) {
        const at = digit - 1;
        var occupied = false;
        for (c.m.values[24][at / 31][2 + at % 31 * 4 ..][0..4]) |bit|
            occupied = occupied or bit != 0;
        if (occupied) break;
    }
    while (digit > 0) {
        digit -= 1;
        c.move(28, 0);
        packet.copyBits(&c.m.values[28], &c.m.values[24], .{ .ref = 24, .from = @intCast(digit), .digits = 1 });
        try c.call("divstep", &.{ 27, 25, 28 }, 29);
        c.move(27, 29);
        packet.copyBits(&c.m.values[26], &c.m.values[30], .{ .ref = 30, .to = @intCast(digit), .digits = 1 });
    }
    c.move(7, 26);
    c.move(8, 27);
}
