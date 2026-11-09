//! Whole-program output invokes prepared ASCII weights; native code handles only format controls and FFN predicates.
const std = @import("std");
const packet = @import("runner_graph_packet.zig");
const api = @import("runner_api.zig");
const End = @FieldType(api.RunEnd, "end");

/// Validate every byte before any output so invalid later bytes cannot leave a partial successful prefix.
pub fn validateBytes(m: anytype, prints: []const packet.Print, budget: u64, out: *std.ArrayList(u8), cancel: ?*const std.atomic.Value(bool)) !End {
    const original = m.values;
    var scratch: [2]packet.Value = undefined;
    m.values = &scratch;
    defer m.values = original;
    for (prints) |p| {
        if (p.style != .byte) continue;
        scratch[0] = original[p.ref];
        const t = m.bank.find("byte_check", 128) orelse return error.GraphTemplate;
        const end = try m.invoke(t, &.{0}, 1, budget, out, cancel, false);
        if (end != .halted) return end;
        if (scratch[1][0][127] == packet.bit_scale) return error.GraphArithmetic;
        if (scratch[1][0][127] != 0) return error.GraphPredicate;
    }
    return .halted;
}

pub fn emit(m: anytype, p: packet.Print, budget: u64, out: *std.ArrayList(u8), cancel: ?*const std.atomic.Value(bool)) !End {
    const original = m.values;
    if (p.ref >= original.len) return error.GraphReference;
    const zero = try packet.literal("0");
    var scratch: [32]packet.Value = @splat(zero);
    scratch[1] = try packet.literal("1");
    scratch[2] = original[p.ref];
    if (p.second) |ref| {
        if (ref >= original.len) return error.GraphReference;
        scratch[3] = original[ref];
    }
    m.values = &scratch;
    defer m.values = original;
    for (2..(if (p.second == null) @as(usize, 3) else 4)) |i| {
        if (scratch[i][0][127] == 0) continue;
        if (scratch[i][0][127] != packet.bit_scale) return error.GraphPredicate;
        scratch[2][0][127] = packet.bit_scale;
        return part(m, 2, .integer, 0, .newline, false, budget, out, cancel);
    }
    if (p.style == .prepared) {
        const width = packet.outputWidth(&scratch[2]);
        const t = m.bank.find(if (width <= 62) "format_f64" else "format", width) orelse return error.GraphTemplate;
        return m.invoke(t, &.{ 2, 3 }, 0, budget, out, cancel, true);
    }
    if (p.style == .byte) {
        const t = m.bank.find("emit_byte", 4) orelse return error.GraphTemplate;
        return m.invoke(t, &.{2}, 0, budget, out, cancel, true);
    }
    if (p.style == .fraction) {
        const t = m.bank.find("meta", 128) orelse return error.GraphTemplate;
        const end = try m.invoke(t, &.{ 3, 1 }, 20, budget, out, cancel, false);
        if (end != .halted) return end;
        const flags = scratch[22];
        if (flags[0][127] != 0) return error.GraphPredicate;
        if (flags[0][2] != packet.bit_scale or flags[0][10] != 0) return error.GraphPrint;
        const bits = flags[0][22..26];
        for (bits[1..]) |bit| if (bit != 0) return error.GraphPredicate;
        if (bits[0] == packet.bit_scale) return part(m, 2, .integer, 0, p.ending, p.prefix, budget, out, cancel);
        if (bits[0] != 0) return error.GraphPredicate;
        const first = try part(m, 2, .integer, 0, .slash, p.prefix, budget, out, cancel);
        if (first != .halted) return first;
        return part(m, 3, .integer, 0, p.ending, false, budget, out, cancel);
    }
    if (p.style == .tuple) {
        const first = try part(m, 2, .integer, 0, .comma_space, true, budget, out, cancel);
        if (first != .halted) return first;
        const ending: packet.Ending = switch (p.ending) {
            .newline => .close,
            .space => .close_space,
            .empty => .close_empty,
            .comma_space => .close_comma_space,
            else => return error.GraphPrint,
        };
        return part(m, 3, .integer, 0, ending, false, budget, out, cancel);
    }
    return part(m, 2, p.style, p.precision, p.ending, p.prefix, budget, out, cancel);
}

fn part(m: anytype, ref: u32, style: packet.Style, precision: u8, ending: packet.Ending, prefix: bool, budget: u64, out: *std.ArrayList(u8), cancel: ?*const std.atomic.Value(bool)) !End {
    if (style == .fraction or style == .tuple or style == .prepared or style == .byte or precision > 18 or (style != .fixed and precision != 0))
        return error.GraphPrint;
    const mode: u32 = switch (style) {
        .integer => 0,
        .fixed => 1,
        .comma => 2,
        .integral_float => 3,
        .boolean => 4,
        else => unreachable,
    };
    const control = @as(u32, precision) + mode * 100 + @as(u32, @backingInt(ending)) * 1000 + @as(u32, @intFromBool(prefix)) * 10000;
    var text: [16]u8 = undefined;
    m.values[0] = try packet.literal(try std.fmt.bufPrint(&text, "{d}", .{control}));
    const t = m.bank.find("format", packet.outputWidth(&m.values[ref])) orelse return error.GraphEmitMissing;
    return m.invoke(t, &.{ ref, 0 }, 0, budget, out, cancel, true);
}
