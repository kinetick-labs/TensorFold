//! Checked bounded binary64 orchestration over prepared integer FFNs and circuit-owned decimal formatting.
const std = @import("std");
const api = @import("runner_api.zig");
const packet = @import("runner_graph_packet.zig");
const context = @import("runner_graph_f64_context.zig");
const math = @import("runner_graph_f64_math.zig");
const printing = @import("runner_graph_f64_print.zig");
const End = @FieldType(api.RunEnd, "end");
const Cancel = ?*const std.atomic.Value(bool);

pub fn compute(m: anytype, kind: []const u8, width: u16, refs: []const u32, dest: usize, budget: u64, out: *std.ArrayList(u8), cancel: Cancel) !End {
    if (width != 128 or std.mem.eql(u8, kind, "f64_repr")) return error.GraphAlgorithm;
    const family = std.meta.stringToEnum(packet.Algorithm, kind) orelse return error.GraphAlgorithm;
    const outputs = packet.algorithmOutputs(family);
    if (refs.len != packet.algorithmArity(family) or dest + outputs > m.values.len) return error.GraphSlotArity;
    const original = m.values;
    const zero = try packet.literal("0");
    var scratch: [256]packet.Value = @splat(zero);
    scratch[1] = try packet.literal("1");
    for (refs, 0..) |ref, i| {
        if (ref >= original.len) return error.GraphReference;
        scratch[2 + i] = original[ref];
    }
    m.values = &scratch;
    defer m.values = original;
    var c: context.Context(@TypeOf(m)) = .{ .m = m, .budget = budget, .out = out, .cancel = cancel };
    const result = (if (std.mem.eql(u8, kind, "f64_format")) printing.prepare(&c, 2, 3) else math.calculate(&c, kind)) catch |err| switch (err) {
        error.GraphAlgorithmStopped => return c.end,
        else => return err,
    };
    const values = [2]packet.Value{ scratch[result[0]], scratch[result[1]] };
    @memcpy(original[dest .. dest + outputs], values[0..outputs]);
    return .halted;
}

pub fn run(m: anytype, kind: []const u8, budget: u64, out: *std.ArrayList(u8), cancel: Cancel) !api.RunEnd {
    if (std.mem.eql(u8, kind, "f64_repr")) {
        const original = m.values;
        const zero = try packet.literal("0");
        var scratch: [256]packet.Value = @splat(zero);
        scratch[1] = try packet.literal("1");
        scratch[2] = original[0];
        scratch[3] = original[1];
        m.values = &scratch;
        defer m.values = original;
        var c: context.Context(@TypeOf(m)) = .{ .m = m, .budget = budget, .out = out, .cancel = cancel };
        printing.render(&c, 2, 3) catch |err| switch (err) {
            error.GraphAlgorithmStopped => return .{ .end = c.end, .steps = m.steps },
            else => return err,
        };
        return .{ .end = .halted, .steps = m.steps };
    }
    const family = std.meta.stringToEnum(packet.Algorithm, kind) orelse return error.GraphAlgorithm;
    const refs = [_]u32{ 0, 1, 2, 3, 4, 5, 6, 7 };
    const end = try compute(m, kind, m.width, refs[0..packet.algorithmArity(family)], 0, budget, out, cancel);
    if (end != .halted) return .{ .end = end, .steps = m.steps };
    const printed = try @import("runner_graph_print.zig").emit(m, .{
        .ref = 0,
        .second = if (packet.algorithmOutputs(family) == 2) 1 else null,
        .style = if (packet.algorithmOutputs(family) == 2) .tuple else .boolean,
    }, budget, out, cancel);
    return .{ .end = printed, .steps = m.steps };
}
