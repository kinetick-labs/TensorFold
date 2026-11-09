//! Request-local FFN dispatch and predicates for the bounded binary64 family.
const std = @import("std");
const api = @import("runner_api.zig");
const packet = @import("runner_graph_packet.zig");
const End = @FieldType(api.RunEnd, "end");

pub fn Context(comptime M: type) type {
    return struct {
        m: M,
        width: u8 = 128,
        budget: u64,
        out: *std.ArrayList(u8),
        cancel: ?*const std.atomic.Value(bool),
        end: End = .halted,
        extended: bool = false,

        const Self = @This();

        pub fn call(c: *Self, name: []const u8, refs: []const u32, dest: usize) !void {
            var needed: usize = 1;
            for (refs) |ref| {
                if (ref >= c.m.values.len) return error.GraphReference;
                try clean(&c.m.values[ref]);
                needed = @max(needed, occupiedDigits(&c.m.values[ref]));
            }
            if (std.mem.eql(u8, name, "mul") or std.mem.eql(u8, name, "decimal_mul")) {
                needed = occupiedDigits(&c.m.values[refs[0]]) + occupiedDigits(&c.m.values[refs[1]]);
            } else if (std.mem.eql(u8, name, "add") or std.mem.eql(u8, name, "sub") or
                std.mem.eql(u8, name, "scale10") or std.mem.eql(u8, name, "round_div"))
            {
                needed += 1;
            }
            const raw_meta = std.mem.eql(u8, name, "raw_meta");
            const selected = if (raw_meta) @max(@as(u8, 8), bucket(@min(needed, c.width))) else bucket(@min(needed, c.width));
            const t = (if (c.m.adaptive_widths) c.m.bank.find(name, selected) else null) orelse
                c.m.bank.find(name, c.width) orelse return error.GraphTemplate;
            if (dest + t.outputs > c.m.values.len) return error.GraphAlgorithmScratch;
            c.end = try c.m.invoke(t, refs, dest, c.budget, c.out, c.cancel, false);
            if (c.end != .halted) return error.GraphAlgorithmStopped;
            for (dest..dest + t.outputs) |i| try clean(&c.m.values[i]);
        }

        pub fn move(c: *Self, dest: usize, source: usize) void {
            c.m.values[dest] = c.m.values[source];
        }

        pub fn literal(c: *Self, dest: usize, text: []const u8) !void {
            c.m.values[dest] = try packet.literal(text);
        }

        pub fn meta(c: *Self, a: u32, b: u32) !void {
            try c.call("raw_meta", &.{ a, b }, 253);
        }

        pub fn flag(c: *Self, index: usize) !bool {
            if (index >= 6) return error.GraphPredicate;
            return predicate(&c.m.values[255], index);
        }

        pub fn nonzero(c: *Self, ref: u32) !bool {
            try c.meta(ref, 0);
            return c.flag(0);
        }

        pub fn sign(c: *Self, ref: u32) !bool {
            try c.meta(ref, 0);
            return c.flag(2);
        }

        pub fn less(c: *Self, a: u32, b: u32) !bool {
            try c.call("compare", &.{ a, b }, 252);
            try c.meta(252, 0);
            return try c.flag(0) and try c.flag(2);
        }

        pub fn equal(c: *Self, a: u32, b: u32) !bool {
            try c.call("compare", &.{ a, b }, 252);
            return !try c.nonzero(252);
        }

        /// Unsigned long division consumes occupied decimal lanes; slots 240..246 are private.
        pub fn divide(c: *Self, a: u32, b: u32, q: usize, r: usize) !void {
            if ((q >= 240 and q <= 246) or (r >= 240 and r <= 246) or q == r)
                return error.GraphAlgorithmScratch;
            const inputs = [2]packet.Value{ c.m.values[a], c.m.values[b] };
            c.m.values[240] = inputs[0];
            c.m.values[241] = inputs[1];
            try c.meta(240, 241);
            if (try c.flag(2) or try c.flag(3) or !try c.flag(1)) return error.GraphF64Arithmetic;
            c.move(242, 0);
            c.move(243, 0);
            var digit = occupiedDigits(&c.m.values[240]);
            while (digit > 0) {
                digit -= 1;
                c.move(244, 0);
                packet.copyBits(&c.m.values[244], &c.m.values[240], .{
                    .ref = 240,
                    .from = @intCast(digit),
                    .digits = 1,
                });
                try c.call("divstep", &.{ 243, 241, 244 }, 245);
                c.move(243, 245);
                packet.copyBits(&c.m.values[242], &c.m.values[246], .{
                    .ref = 246,
                    .to = @intCast(digit),
                    .digits = 1,
                });
            }
            const result = [2]packet.Value{ c.m.values[242], c.m.values[243] };
            c.m.values[q] = result[0];
            c.m.values[r] = result[1];
        }
    };
}

pub fn occupiedDigits(value: *const packet.Value) usize {
    var digits: usize = packet.max_digits;
    while (digits > 1) : (digits -= 1) {
        const at = digits - 1;
        for (value[at / 31][2 + at % 31 * 4 ..][0..4]) |bit| if (bit != 0) return digits;
    }
    return digits;
}

fn bucket(needed: usize) u8 {
    for ([_]u8{ 1, 2, 4, 8, 16, 32, 62, 128 }) |width| if (needed <= width) return width;
    unreachable;
}

fn clean(value: *const packet.Value) !void {
    const bit = value[0][127];
    if (bit == packet.bit_scale) return error.GraphF64Arithmetic;
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
