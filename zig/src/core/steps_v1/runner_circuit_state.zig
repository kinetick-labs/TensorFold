//! Per-machine circuit state: the DeltaNet tables (the RAM op's rows), per-channel conv history and canonical bytes.
const std = @import("std");
const model = @import("runner_circuit_model.zig");

const Scalar = model.Scalar;
const hd = model.hd;
const table_floats = hd * hd;
const history = model.lags - 1;

/// Read-only tables stay shared with the circuit; tables a key or decay can change are copied per machine.
pub const State = struct {
    gpa: std.mem.Allocator,
    circuit: *const model.Circuit,
    tables: [][]Scalar,
    owned: []bool,
    present: []u128,
    history: []Scalar,
    steps: u64 = 0,
    x: []Scalar,
    active: std.ArrayList(u32),
    seen: []bool,
    work: []Scalar,

    pub fn init(gpa: std.mem.Allocator, c: *const model.Circuit) !State {
        const tables = try gpa.alloc([]Scalar, c.tables.len);
        errdefer gpa.free(tables);
        var copied: usize = 0;
        errdefer for (c.tables[0..copied], tables[0..copied]) |t, s| if (t.writable) gpa.free(s);
        for (c.tables, tables) |t, *s| {
            s.* = if (t.writable) try gpa.dupe(Scalar, t.rows) else @constCast(t.rows);
            copied += 1;
        }
        const owned = try gpa.alloc(bool, c.tables.len);
        errdefer gpa.free(owned);
        for (c.tables, owned) |t, *o| o.* = t.writable;
        const present = try gpa.alloc(u128, c.tables.len);
        errdefer gpa.free(present);
        for (c.tables, present) |t, *p| p.* = t.present;
        const hist = try gpa.alloc(Scalar, @as(usize, c.channels) * history);
        errdefer gpa.free(hist);
        @memset(hist, 0);
        const x = try gpa.alloc(Scalar, c.dims);
        errdefer gpa.free(x);
        @memset(x, 0);
        const seen = try gpa.alloc(bool, c.dims);
        errdefer gpa.free(seen);
        @memset(seen, false);
        var active: std.ArrayList(u32) = .empty;
        try active.ensureTotalCapacity(gpa, c.dims);
        errdefer active.deinit(gpa);
        const work = try gpa.alloc(Scalar, workSize(c));
        return .{ .gpa = gpa, .circuit = c, .tables = tables, .owned = owned, .present = present, .history = hist, .x = x, .active = active, .seen = seen, .work = work };
    }

    pub fn deinit(s: *State) void {
        for (s.owned, s.tables) |o, rows| if (o) s.gpa.free(rows);
        s.gpa.free(s.tables);
        s.gpa.free(s.owned);
        s.gpa.free(s.present);
        s.gpa.free(s.history);
        s.gpa.free(s.x);
        s.gpa.free(s.seen);
        s.active.deinit(s.gpa);
        s.gpa.free(s.work);
        s.* = undefined;
    }

    /// Zeroes the residual for a new step; only listed dims can be nonzero.
    pub fn clearResidual(s: *State) void {
        for (s.active.items) |d| {
            s.x[d] = 0;
            s.seen[d] = false;
        }
        s.active.clearRetainingCapacity();
    }

    /// Lists dim d as possibly nonzero (capacity is the dim count, so this never allocates).
    pub fn touch(s: *State, d: u32) void {
        if (s.seen[d]) return;
        s.seen[d] = true;
        s.active.appendAssumeCapacity(d);
    }

    pub fn writable(s: *const State, table: u32) bool {
        return s.circuit.tables[table].writable;
    }

    pub fn row(s: *State, table: u32, r: u32) []Scalar {
        return s.tables[table][r * hd ..][0..hd];
    }

    pub fn rowPresent(s: *const State, table: u32, r: u32) bool {
        return s.present[table] >> @intCast(r) & 1 == 1;
    }

    /// Copies a shared table into this machine before the host rewrites it (an ELF boot's ROM and DROM).
    pub fn own(s: *State, table: u32) !void {
        if (s.owned[table]) return;
        s.tables[table] = try s.gpa.dupe(Scalar, s.tables[table]);
        s.owned[table] = true;
    }

    /// Replaces one row (the RAM op's write) of a table this machine owns.
    pub fn writeRow(s: *State, table: u32, r: u32, values: []const Scalar) !void {
        if (!s.owned[table]) return error.CircuitTableReadOnly;
        if (r >= hd or values.len != hd) return error.CircuitIndex;
        @memcpy(s.row(table, r), values);
        s.present[table] |= @as(u128, 1) << @intCast(r);
    }

    pub fn clearRow(s: *State, table: u32, r: u32) !void {
        if (!s.owned[table]) return error.CircuitTableReadOnly;
        if (r >= hd) return error.CircuitIndex;
        @memset(s.row(table, r), 0);
        s.present[table] &= ~(@as(u128, 1) << @intCast(r));
    }

    /// The last three pre-convolution projections of global channel i, newest first.
    pub fn channelHistory(s: *State, i: u32) *[history]Scalar {
        return s.history[i * history ..][0..history];
    }

    /// Bytes this machine owns beyond the shared circuit (copied tables, history, scratch), for admission.
    pub fn ownedBytes(s: *const State) usize {
        var n: usize = (s.history.len + s.x.len + s.work.len) * @sizeOf(Scalar);
        for (s.owned) |o| n += if (o) table_floats * @sizeOf(Scalar) else 0;
        return n;
    }

    /// Canonical bytes: steps, then every table and every channel history as binary64 LE, absent rows as zeros.
    pub fn canonical(s: *const State, gpa: std.mem.Allocator) ![]u8 {
        const n = 8 + (s.tables.len * table_floats + s.history.len) * 8;
        const out = try gpa.alloc(u8, n);
        std.mem.writeInt(u64, out[0..8], s.steps, .little);
        var at: usize = 8;
        for (s.tables, s.present) |t, p| for (0..hd) |r| {
            const live = p >> @intCast(r) & 1 == 1;
            for (t[r * hd ..][0..hd]) |v| {
                const w: f64 = if (live) v else 0;
                std.mem.writeInt(u64, out[at..][0..8], @bitCast(w + 0), .little);
                at += 8;
            }
        };
        for (s.history) |v| {
            const w: f64 = v;
            std.mem.writeInt(u64, out[at..][0..8], @bitCast(w + 0), .little);
            at += 8;
        }
        return out;
    }
};

fn workSize(c: *const model.Circuit) usize {
    var n: usize = 0;
    for (c.batches) |*b| n = @max(n, switch (b.*) {
        .ffn => |*f| f.outs.len + f.up.forms.outputs + f.gate.forms.outputs,
        .heads => |*h| h.proj.columns() + h.outs.len + 5 * hd + h.proj.forms.outputs,
    });
    return n;
}
