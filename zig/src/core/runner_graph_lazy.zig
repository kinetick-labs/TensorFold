//! Logical bindings validated offline and again before publishing a lazily decoded circuit.
const std = @import("std");
const model = @import("steps_v1/runner_circuit_model.zig");
const packet = @import("runner_graph_packet.zig");
const missing = std.math.maxInt(u32);
pub const Slots = [packet.max_arity][packet.max_groups]u32;

pub const Binding = struct {
    inputs: u8 = 0,
    outputs: u8 = 0,
    in_slots: Slots = @splat(@splat(missing)),
    out_slots: Slots = @splat(@splat(missing)),

    pub fn inspect(c: *const model.Circuit, width: u8) !Binding {
        if (!packet.widthValid(width)) return error.GraphTemplate;
        if (c.program.a_bit != packet.bit_scale or c.program.a_ref != packet.reference_scale)
            return error.GraphValueScale;
        if (c.tables.len != c.table_names.len) return error.GraphSlots;
        var out: Binding = .{};
        for (c.table_names, 0..) |name, table| {
            if (std.mem.startsWith(u8, name, "output.cursor")) continue;
            const input = std.mem.startsWith(u8, name, "in");
            const output = std.mem.startsWith(u8, name, "out");
            if (!input and !output) continue;
            const prefix: usize = if (input) 2 else 3;
            const dot = std.mem.indexOfScalarPos(u8, name, prefix, '.') orelse return error.GraphSlots;
            const logical = std.fmt.parseInt(u8, name[prefix..dot], 10) catch return error.GraphSlots;
            const offset = std.fmt.parseInt(u8, name[dot + 1 ..], 10) catch return error.GraphSlots;
            if (logical >= packet.max_arity or offset % 31 != 0 or offset >= width) return error.GraphSlots;
            const target = if (input) &out.in_slots else &out.out_slots;
            const group = offset / 31;
            if (target[logical][group] != missing or c.tables[table].rows.len != 128 * 128)
                return error.GraphSlots;
            target[logical][group] = @intCast(table);
            const arity = if (input) &out.inputs else &out.outputs;
            arity.* = @max(arity.*, logical + 1);
        }
        if (out.inputs == 0 or out.outputs == 0) return error.GraphSlots;
        const groups = (@as(usize, width) + 30) / 31;
        for (out.in_slots[0..out.inputs]) |s| for (s[0..groups]) |idx| if (idx == missing)
            return error.GraphSlots;
        for (out.out_slots[0..out.outputs]) |s| for (s[0..groups]) |idx| if (idx == missing)
            return error.GraphSlots;
        return out;
    }
};
