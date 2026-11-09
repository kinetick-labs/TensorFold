//! Per-machine circuit state: the DeltaNet tables (the RAM op's rows), slot values, head frames and canonical bytes.
const std = @import("std");
const model = @import("runner_circuit_model.zig");
const head = @import("runner_circuit_head.zig");
const Memo = @import("runner_circuit_memo.zig").Memo;

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
    steps: u64 = 0,
    /// Each slot's value as of the last step that computed it.
    val: []Scalar,
    /// Per unit: the step stamp (step + 1) it was marked dirty, its inputs last changed, an output base changed.
    dirty: []u64,
    changed: []u64,
    rebase: []u64,
    /// Per output slot: the unit's value before adding its base, so a base change alone just adds again.
    pre: []Scalar,
    /// Per unit: the outputs whose base changed this step (stored at the unit's first output slot).
    rb: []u32,
    rb_len: []u32,
    rb_mark: []u64,
    /// Per unit input (at the unit's in_off): its slot's value and a bit set while the value is nonzero.
    in_vals: []Scalar,
    in_bits: []u64,
    /// Per unit: its last run decayed a table, so the next step must run it again.
    unstable: []bool,
    /// Per chain end, one bit per link that holds something other than +0 (every pre starts +0).
    chain_bits: []u64,
    /// Units to visit this step and those already due next step (lag window open or unstable), one bit each.
    active: []u64,
    keep: []u64,
    /// Per head (by plan) its last runs' inputs; per unit its results by exact input.
    frames: [][head.frames_per_unit]head.Frame,
    picks: [][head.frames_per_unit]head.Picks,
    /// Per unit: its last outputs were all +0.
    rest: []bool,
    memos: []Memo,
    groups: []head.Group,
    /// Dense input scratch, zero outside a running unit; the same for steps t-1..t-3 while a head reads them.
    x: []Scalar,
    past: [history][]Scalar,
    /// A running unit's nonzero dims, its memo key, selecting dims and the input index of each nonzero dim.
    nz: []u32,
    key: []u64,
    nz2: []u32,
    local: []u32,
    words: []u32,
    work: []Scalar,
    /// FFN parts' form values and node outputs of their last compute.
    ffn_up: []Scalar,
    ffn_gate: []Scalar,
    ffn_y: []Scalar,
    /// Run every unit on the next step (a fresh machine or a host edit of the tables).
    force: bool = true,
    /// Run every unit on every step without its memo (the reference the skipping is checked against).
    always: bool = false,

    pub fn init(gpa: std.mem.Allocator, c: *const model.Circuit) !State {
        var s: State = .{ .gpa = gpa, .circuit = c, .tables = &.{}, .owned = &.{}, .present = &.{}, .val = &.{}, .dirty = &.{}, .changed = &.{}, .rebase = &.{}, .pre = &.{}, .rb = &.{}, .rb_len = &.{}, .rb_mark = &.{}, .in_vals = &.{}, .in_bits = &.{}, .unstable = &.{}, .chain_bits = &.{}, .active = &.{}, .keep = &.{}, .frames = &.{}, .picks = &.{}, .rest = &.{}, .memos = &.{}, .groups = &.{}, .x = &.{}, .past = @splat(&.{}), .nz = &.{}, .key = &.{}, .nz2 = &.{}, .local = &.{}, .words = &.{}, .work = &.{}, .ffn_up = &.{}, .ffn_gate = &.{}, .ffn_y = &.{} };
        errdefer s.deinit();
        const f = &c.flow;
        const n = f.units.len;
        s.owned = try zeroed(gpa, bool, c.tables.len);
        s.tables = try gpa.alloc([]Scalar, c.tables.len);
        for (c.tables, s.tables) |t, *rows| rows.* = @constCast(t.rows);
        s.present = try gpa.alloc(u128, c.tables.len);
        for (c.tables, s.present) |t, *p| p.* = t.present;
        for (c.tables, 0..) |t, i| if (t.writable) try s.own(@intCast(i));
        s.val = try zeroed(gpa, Scalar, f.slots);
        s.dirty = try zeroed(gpa, u64, n);
        s.changed = try zeroed(gpa, u64, n);
        s.rebase = try zeroed(gpa, u64, n);
        s.pre = try zeroed(gpa, Scalar, f.slots);
        s.rb = try zeroed(gpa, u32, f.slots);
        s.rb_len = try zeroed(gpa, u32, n);
        s.rb_mark = try zeroed(gpa, u64, f.slots);
        s.in_vals = try zeroed(gpa, Scalar, f.total_ins);
        s.in_bits = try zeroed(gpa, u64, f.total_ins / 64 + 1);
        s.unstable = try zeroed(gpa, bool, n);
        s.chain_bits = try zeroed(gpa, u64, f.chain_words);
        s.active = try zeroed(gpa, u64, n / 64 + 1);
        s.keep = try zeroed(gpa, u64, n / 64 + 1);
        s.frames = try gpa.alloc([head.frames_per_unit]head.Frame, f.plans.len);
        for (s.frames) |*fs| fs.* = @splat(.{});
        s.picks = try gpa.alloc([head.frames_per_unit]head.Picks, f.plans.len);
        for (s.picks) |*fs| fs.* = @splat(.{});
        s.rest = try gpa.alloc(bool, n);
        @memset(s.rest, true);
        s.memos = try gpa.alloc(Memo, n);
        for (s.memos) |*m| m.* = .{};
        s.groups = try gpa.alloc(head.Group, f.groups);
        @memset(s.groups, .{});
        s.x = try zeroed(gpa, Scalar, c.dims);
        for (&s.past) |*p| p.* = try zeroed(gpa, Scalar, c.dims);
        s.nz = try gpa.alloc(u32, f.max_dims);
        s.key = try gpa.alloc(u64, f.max_dims);
        s.nz2 = try gpa.alloc(u32, f.max_dims);
        s.local = try gpa.alloc(u32, f.max_dims);
        s.words = try gpa.alloc(u32, wordsSize(c));
        s.work = try gpa.alloc(Scalar, workSize(c));
        s.ffn_up = try zeroed(gpa, Scalar, f.ffn_totals[0]);
        s.ffn_gate = try zeroed(gpa, Scalar, f.ffn_totals[1]);
        s.ffn_y = try zeroed(gpa, Scalar, f.ffn_totals[2]);
        return s;
    }

    pub fn deinit(s: *State) void {
        if (s.owned.len == s.tables.len) for (s.owned, s.tables) |o, rows| if (o) s.gpa.free(rows);
        s.gpa.free(s.tables);
        s.gpa.free(s.owned);
        s.gpa.free(s.present);
        s.gpa.free(s.val);
        s.gpa.free(s.dirty);
        s.gpa.free(s.changed);
        s.gpa.free(s.rebase);
        s.gpa.free(s.pre);
        s.gpa.free(s.rb);
        s.gpa.free(s.rb_len);
        s.gpa.free(s.rb_mark);
        s.gpa.free(s.in_vals);
        s.gpa.free(s.in_bits);
        s.gpa.free(s.unstable);
        s.gpa.free(s.chain_bits);
        s.gpa.free(s.active);
        s.gpa.free(s.keep);
        for (s.frames) |*fs| for (fs) |*fr| fr.deinit(s.gpa);
        s.gpa.free(s.frames);
        for (s.picks) |*fs| for (fs) |*fr| fr.deinit(s.gpa);
        s.gpa.free(s.picks);
        s.gpa.free(s.rest);
        for (s.memos) |*m| m.deinit(s.gpa);
        s.gpa.free(s.memos);
        for (s.groups) |*g| g.memo.deinit(s.gpa);
        s.gpa.free(s.groups);
        s.gpa.free(s.x);
        for (s.past) |p| s.gpa.free(p);
        s.gpa.free(s.nz);
        s.gpa.free(s.key);
        s.gpa.free(s.nz2);
        s.gpa.free(s.local);
        s.gpa.free(s.words);
        s.gpa.free(s.work);
        s.gpa.free(s.ffn_up);
        s.gpa.free(s.ffn_gate);
        s.gpa.free(s.ffn_y);
        s.* = undefined;
    }

    /// Dim d's value at the end of the last step (the residual a dense evaluator would hold).
    pub fn value(s: *const State, d: u32) Scalar {
        return s.val[s.circuit.flow.final[d]];
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

    /// Bytes this machine owns beyond the shared circuit (copied tables, scratch, frames and memos), for admission.
    pub fn ownedBytes(s: *const State) usize {
        var n: usize = (s.x.len * (1 + history) + s.work.len + s.val.len) * @sizeOf(Scalar) + s.dirty.len * 17;
        for (s.owned) |o| n += if (o) table_floats * @sizeOf(Scalar) else 0;
        for (s.frames) |fs| for (fs) |fr| {
            n += fr.dims.capacity * 4 + fr.vals.capacity * 4;
        };
        for (s.picks) |fs| for (fs) |fr| {
            n += fr.words.capacity * 4;
        };
        for (s.memos) |m| n += m.bytes();
        for (s.groups) |g| n += g.memo.bytes();
        return n;
    }

    /// Canonical bytes: steps, then every table and every channel's last three projections as binary64 LE.
    pub fn canonical(s: *const State, gpa: std.mem.Allocator) ![]u8 {
        const c = s.circuit;
        const n = 8 + (s.tables.len * table_floats + @as(usize, c.channels) * history) * 8;
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
        const scratch: *State = @constCast(s);
        for (c.flow.units) |u| if (u.kind == .heads) head.history(scratch, u.plan, out, &at);
        std.debug.assert(at == n);
        return out;
    }
};

fn zeroed(gpa: std.mem.Allocator, comptime T: type, n: usize) ![]T {
    const out = try gpa.alloc(T, n);
    @memset(out, std.mem.zeroes(T));
    return out;
}

fn wordsSize(c: *const model.Circuit) usize {
    var n: usize = 0;
    for (c.flow.units) |u| n = @max(n, switch (u.kind) {
        .ffn => |f| f.outs.len,
        .heads => |h| 2 + 2 * (h.heads[0].q.len + h.heads[0].k.len + h.heads[0].value_heads.len),
    });
    return n;
}

fn workSize(c: *const model.Circuit) usize {
    var n: usize = 0;
    for (c.flow.units) |u| n = @max(n, switch (u.kind) {
        .ffn => |f| f.outs.len,
        .heads => |h| @max(h.outs.len + hd + c.flow.plans[u.plan].zrep.len, c.flow.plans[u.plan].sel.columns() + c.flow.plans[u.plan].sel.forms.outputs),
    });
    return n;
}
