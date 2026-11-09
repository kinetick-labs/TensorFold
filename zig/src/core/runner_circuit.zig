//! The R1 machine: a compiled circuit booted from its snapshot plus a program, stepped one trace token at a time.
const std = @import("std");
const api = @import("runner_api.zig");
const model = @import("runner_circuit_model.zig");
const eval = @import("runner_circuit_eval.zig");
const state_mod = @import("runner_circuit_state.zig");
const elf = @import("runner_circuit_elf.zig");

pub const Circuit = model.Circuit;
pub const State = state_mod.State;
const Allocator = std.mem.Allocator;

/// Monotonic nanoseconds, for exec_ns and rates only (never for results).
pub fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

/// Steps between cancel checks.
pub const cancel_every: u64 = 4096;

pub const Machine = struct {
    state: State,
    next: u32,
    halted: bool = false,
    faulted: bool = false,

    /// The circuit's boot snapshot with `program` loaded as data; the computing is the weights'.
    pub fn boot(gpa: Allocator, c: *const Circuit, program: api.Program) !Machine {
        var s = try State.init(gpa, c);
        errdefer s.deinit();
        switch (program.kind) {
            .calc => {
                if (c.program.kind != .calc) return error.CircuitNotCalc;
                try patchInput(&s, program.bytes);
            },
            .elf => try elf.load(&s, program.bytes),
            .graph => return error.CircuitProgramKind, // graph packets run on the graph backend (T1)
        }
        return .{ .state = s, .next = c.go };
    }

    pub fn deinit(m: *Machine, gpa: Allocator) void {
        _ = gpa;
        m.state.deinit();
        m.* = undefined;
    }

    /// Up to `budget` more steps, until HALT or `cancel`; BYTE payloads below 256 go to `out`, each token to `trace`.
    pub fn run(m: *Machine, budget: u64, out: *std.ArrayList(u8), trace: ?*std.ArrayList(u32), cancel: *const std.atomic.Value(bool)) !api.RunEnd {
        const c = m.state.circuit;
        const gpa = m.state.gpa;
        var steps: u64 = 0;
        while (!m.halted) : (steps += 1) {
            if (steps == budget) return .{ .end = .budget, .steps = steps };
            if (steps % cancel_every == 0 and cancel.load(.acquire)) return .{ .end = .cancelled, .steps = steps };
            const t = try eval.step(&m.state, m.next);
            if (trace) |tr| try tr.append(gpa, t);
            const fam, const pay = c.split(t).?;
            if (fam == c.byte_family and pay < 256) try out.append(gpa, @intCast(pay));
            if (c.fault_family) |f| m.faulted = m.faulted or fam == f;
            m.halted = t == c.halt;
            m.next = t;
        }
        return .{ .end = if (m.faulted) .fault else .halted, .steps = steps };
    }

    /// Canonical bytes of the full state: the tables, channel histories and step count, then the next input token.
    pub fn snapshot(m: *const Machine, gpa: Allocator) ![]u8 {
        const body = try m.state.canonical(gpa);
        defer gpa.free(body);
        const out = try gpa.alloc(u8, body.len + 4);
        @memcpy(out[0..body.len], body);
        std.mem.writeInt(u32, out[body.len..][0..4], m.next, .little);
        return out;
    }
};

/// Writes the calc expression and its NUL at the firmware's input address, as calc.machine patches RAM.
pub fn patchInput(s: *State, text: []const u8) !void {
    const p = s.circuit.program;
    if (std.mem.indexOfScalar(u8, text, 0) != null) return error.CircuitInputNul;
    if (text.len + 1 > p.input_capacity) return error.CircuitInputTooLong;
    for (0..text.len + 1) |i| {
        const addr = p.input_address + @as(u32, @intCast(i));
        const byte: u32 = if (i < text.len) text[i] else 0;
        const shift: u5 = @intCast(8 * (addr & 3));
        const word = try readWord(s, addr >> 2);
        try writeWord(s, addr >> 2, word & ~(@as(u32, 0xff) << shift) | byte << shift);
    }
}

const RamRow = struct { table: u32, row: u32 };

/// RAM word wi lives in head ram<h>, h = (wi & 15) | (wi >> 12) << 4, value head (wi >> 4) & 1, row (wi >> 5) & 127.
fn ramRow(s: *const State, wi: u32) !RamRow {
    if (wi >= 1 << 13) return error.CircuitRamAddress;
    const base = s.circuit.program.ram_tables[(wi & 15) | (wi >> 12) << 4];
    if (base == std.math.maxInt(u32)) return error.CircuitRamTables;
    return .{ .table = base + (wi >> 4 & 1), .row = wi >> 5 & 127 };
}

pub fn readWord(s: *State, wi: u32) !u32 {
    const at = try ramRow(s, wi);
    if (!s.rowPresent(at.table, at.row)) return 0;
    const r = s.row(at.table, at.row);
    var word: u32 = 0;
    for (0..32) |i| {
        const v = r[2 + i];
        if (v == s.circuit.program.a_bit) word |= @as(u32, 1) << @intCast(i) else if (v != 0) return error.CircuitRamRow;
    }
    return word;
}

/// A nonzero word carries the ballast in columns 0 and 1 and A_BIT per set bit in columns 2..33; zero is absent.
pub fn writeWord(s: *State, wi: u32, word: u32) !void {
    const at = try ramRow(s, wi);
    s.force = true;
    if (word == 0) return s.clearRow(at.table, at.row);
    var r: [model.hd]model.Scalar = @splat(0);
    r[0] = s.circuit.program.a_ref;
    r[1] = s.circuit.program.a_ref;
    for (0..32) |i| if (word >> @intCast(i) & 1 == 1) {
        r[2 + i] = s.circuit.program.a_bit;
    };
    try s.writeRow(at.table, at.row, &r);
}
