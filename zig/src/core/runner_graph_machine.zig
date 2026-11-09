//! A fresh binding machine replays immutable template calls and transfers raw typed tensor rows.
const std = @import("std");
const api = @import("runner_api.zig");
const State = @import("steps_v1/runner_circuit_state.zig").State;
const state_cache = @import("runner_graph_state_cache.zig");
const eval = @import("steps_v1/runner_circuit_eval.zig");
const bank_mod = @import("runner_graph_bank.zig");
const packet = @import("runner_graph_packet.zig");
const algorithms = @import("runner_graph_algorithms.zig");
const printing = @import("runner_graph_print.zig");
const Allocator = std.mem.Allocator;

pub const Machine = struct {
    gpa: Allocator,
    bank: *const bank_mod.Bank,
    parsed: std.json.Parsed(packet.Packet),
    values: []packet.Value,
    ops: []const packet.Op,
    result: u32,
    initial: usize,
    steps: u64 = 0,
    used: bool = false,
    trace: ?*std.ArrayList(u32) = null,
    states: state_cache.Cache = .{},
    reuse_states: bool = true,
    adaptive_widths: bool = true,
    algorithm: ?[]const u8 = null,
    width: u16 = 62,
    precision: u8 = 0,
    prints: []const packet.Print = &.{},
    decode: bool = true, // false: a template not yet prepared is error.GraphUnprepared, never decoded on this thread

    pub fn boot(a: Allocator, bank: *const bank_mod.Bank, bytes: []const u8) !Machine {
        if (bytes.len > packet.max_packet) return error.GraphPacketSize;
        const parsed = try std.json.parseFromSlice(packet.Packet, a, bytes, .{});
        errdefer parsed.deinit();
        const p = parsed.value;
        if (p.version != 1 or p.values.len > packet.max_values) return error.GraphPacket;
        var ops = p.ops;
        var result = p.result;
        var constants: []const packet.Value = &.{};
        var prepared_count: ?usize = null;
        var algorithm: ?[]const u8 = null;
        var width: u16 = 62;
        var precision: u8 = 0;
        if (p.plan) |name| {
            if (p.ops.len != 0 or p.result != 0 or p.prints.len != 0) return error.GraphNamedPlanOverride;
            const plan = bank.plan(name) orelse return error.GraphPlan;
            if (p.values.len != plan.inputs) return error.GraphSlotArity;
            ops = plan.ops;
            result = plan.result;
            constants = plan.constants;
            prepared_count = plan.total_values;
            algorithm = plan.algorithm;
            width = plan.width;
            precision = plan.precision;
        }
        if (constants.len > packet.max_values - p.values.len) return error.GraphPlanSize;
        const initial = p.values.len + constants.len;
        const count = prepared_count orelse try bank.validate(initial, ops, result);
        try bank.validatePrints(p.prints, count);
        const values = try a.alloc(packet.Value, count);
        errdefer a.free(values);
        for (p.values, values[0..p.values.len]) |text, *v| v.* = try packet.literal(text);
        if (algorithm != null) for (values[0..p.values.len]) |*v| try fits(v, width);
        @memcpy(values[p.values.len..initial], constants);
        return .{ .gpa = a, .bank = bank, .parsed = parsed, .values = values, .ops = ops, .result = result, .initial = initial, .algorithm = algorithm, .width = width, .precision = precision, .prints = p.prints };
    }

    pub fn deinit(m: *Machine) void {
        m.states.deinit(m.gpa);
        m.gpa.free(m.values);
        m.parsed.deinit();
        m.* = undefined;
    }

    pub fn run(m: *Machine, budget: u64, out: *std.ArrayList(u8), cancel: ?*const std.atomic.Value(bool)) !api.RunEnd {
        if (m.used) return error.GraphMachineAlreadyRun;
        m.used = true;
        if (m.algorithm) |kind| return algorithms.run(m, kind, m.width, budget, out, cancel);
        var count = m.initial;
        for (m.ops) |op| {
            if (cancelled(cancel)) return .{ .end = .cancelled, .steps = m.steps };
            if (std.mem.eql(u8, op.op, "bind")) {
                var v: packet.Value = @splat(@splat(0));
                for (op.copies) |c| packet.copyBits(&v, &m.values[c.ref], c);
                if (op.flags) |ref| @memcpy(v[0][126..128], m.values[ref][0][126..128]);
                m.values[count] = v;
            } else {
                const family = std.meta.stringToEnum(packet.Algorithm, op.op);
                const t = if (family == null) m.bank.find(op.op, op.width).? else null;
                const arity = if (family) |f| packet.algorithmArity(f) else t.?.inputs;
                var refs: [packet.max_arity]u32 = undefined;
                const args = if (op.args) |args| args else blk: {
                    refs[0] = op.a;
                    if (op.b) |b| refs[1] = b;
                    break :blk refs[0..arity];
                };
                const end = if (family != null) try algorithms.compute(m, op.op, op.width, op.precision, args, count, budget, out, cancel) else try m.invoke(t.?, args, count, budget, out, cancel, false);
                if (end != .halted) return .{ .end = end, .steps = m.steps };
            }
            if (op.checked) for (m.values[count .. count + op.outputs]) |value| {
                const flag = value[0][127];
                if (flag == packet.bit_scale) return error.GraphArithmetic;
                if (flag != 0) return error.GraphPredicate;
            };
            count += op.outputs;
        }
        if (m.prints.len != 0) {
            const validated = try printing.validateBytes(m, m.prints, budget, out, cancel);
            if (validated != .halted) return .{ .end = validated, .steps = m.steps };
            for (m.prints) |p| {
                const end = try printing.emit(m, p, budget, out, cancel);
                if (end != .halted) return .{ .end = end, .steps = m.steps };
            }
            return .{ .end = .halted, .steps = m.steps };
        }
        const width = packet.outputWidth(&m.values[m.result]);
        const emitter = m.bank.find("emit", width) orelse return error.GraphEmitMissing;
        const end = try m.invoke(emitter, &.{m.result}, 0, budget, out, cancel, true);
        return .{ .end = end, .steps = m.steps };
    }

    pub fn invoke(m: *Machine, t: *const bank_mod.Template, refs: []const u32, dest: usize, budget: u64, out: *std.ArrayList(u8), cancel: ?*const std.atomic.Value(bool), emit: bool) !End {
        if (refs.len != t.inputs) return error.GraphSlotArity;
        if (m.steps >= budget) return .budget;
        if (cancelled(cancel)) return .cancelled;
        for (refs) |ref| try fits(&m.values[ref], t.width);
        if (!m.decode and !t.isPrepared()) return error.GraphUnprepared;
        try t.prepare();
        if (cancelled(cancel)) return .cancelled;
        var fresh: State = undefined;
        const state = if (m.reuse_states) try m.states.get(m.gpa, t) else blk: {
            fresh = try State.init(m.gpa, &t.circuit);
            break :blk &fresh;
        };
        defer if (m.reuse_states) state_cache.reset(state) else state.deinit();
        const groups = (@as(usize, t.width) + 30) / 31;
        for (refs, 0..) |ref, logical| for (0..groups) |group| {
            const table = t.in_slots[logical][group];
            try state.own(table);
            var row: [128]f32 = undefined;
            @memcpy(&row, state.row(table, 0));
            @memcpy(row[2..128], m.values[ref][group][2..128]);
            try state.writeRow(table, 0, &row);
        };
        var token = t.circuit.go;
        while (token != t.circuit.halt) {
            if (cancelled(cancel)) return .cancelled;
            if (m.steps >= budget) return .budget;
            token = try eval.step(state, token);
            m.steps += 1;
            if (m.trace) |trace| try trace.append(m.gpa, token);
            const fam, const payload = t.circuit.split(token) orelse return error.GraphOutputToken;
            if (t.circuit.fault_family) |fault| if (fam == fault) return .fault;
            if (fam == t.circuit.byte_family) {
                if (!emit or payload > 255) return error.GraphUnexpectedByte;
                if (out.items.len >= packet.max_output) return error.GraphOutputSize;
                try out.append(m.gpa, @intCast(payload));
            }
        }
        if (!emit) {
            for (0..t.outputs) |logical| {
                var v: packet.Value = @splat(@splat(0));
                for (0..groups) |group| {
                    const table = t.out_slots[logical][group];
                    if (!state.rowPresent(table, 0)) return error.GraphOutputAbsent;
                    @memcpy(&v[group], state.row(table, 0));
                }
                m.values[dest + logical] = v;
            }
        }
        return .halted;
    }
};

const End = @FieldType(api.RunEnd, "end");

fn cancelled(flag: ?*const std.atomic.Value(bool)) bool {
    return if (flag) |f| f.load(.acquire) else false;
}

fn fits(v: *const packet.Value, width: u16) !void {
    for (width..packet.max_digits) |digit| {
        const bits = v[digit / 31][2 + digit % 31 * 4 ..][0..4];
        for (bits) |bit| if (bit != 0) return error.GraphNarrowBinding;
    }
}
