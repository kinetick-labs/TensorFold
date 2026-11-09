//! A step as single-assignment dataflow: every write is a slot that keeps its value across steps, units read slots.
const std = @import("std");
const model = @import("runner_circuit_model.zig");
const head = @import("runner_circuit_head.zig");
const ffn = @import("runner_circuit_ffn.zig");

/// The slot of a dim nothing has written in the step yet; it never changes.
pub const zero_slot: u32 = 0;
pub const no_slot = std.math.maxInt(u32);

/// One recomputable piece of a step: an independent part of an FFN batch or a single head.
pub const Unit = struct {
    kind: union(enum) { ffn: *const model.Ffn, heads: *const model.Heads },
    /// Dims the unit's forms read (ascending) and the slot each one reads.
    dims: []const u32,
    ins: []const u32,
    /// The slot each output writes and, for an `into` output, the slot it adds to.
    outs: []const u32,
    base: []const u32,
    /// Largest conv lag among the unit's channels: a head reruns that many steps after its inputs change.
    lag: u32,
    /// The unit's CPU plan (index into Flow.plans for a head, Flow.ffn_plans for an FFN batch).
    plan: u32 = no_slot,
    /// Where the unit's inputs start in the per-machine input arrays.
    in_off: u32 = 0,
    /// An FFN part whose outputs are all +0 at all-zero inputs: a fresh machine need not run it until an input moves.
    quiet: bool = false,
};

pub const Flow = struct {
    slots: u32,
    units: []const Unit,
    plans: []const head.Plan,
    ffn_plans: []const ffn.Plan,
    /// Sizes of the per-machine FFN arrays: up forms, gate forms and nodes.
    ffn_totals: [3]u32,
    /// Distinct feature dims and their slots; a feature dim's value is its last nonzero feature, else zero.
    feature_dims: []const u32,
    feature_slots: []const u32,
    /// Units reading each slot as an input, and units adding their outputs onto it (CSR by slot).
    reader_starts: []const u32,
    readers: []const u32,
    /// The reading unit's input index of each reader edge, and where that unit's inputs start.
    reader_local: []const u32,
    reader_off: []const u32,
    /// Per base edge: the unit, the output that adds onto the slot, and the unit's first output slot.
    base_starts: []const u32,
    base_readers: []const u32,
    base_out: []const u32,
    base_first: []const u32,
    /// The slot holding each dim at the end of a step, and the output code's slots.
    final: []const u32,
    code: []const u32,
    max_dims: u32,
    total_ins: u32,
    /// Per slot: the unit writing it; a chain link no unit reads (elided, never stored) and the chain end it feeds.
    slot_unit: []const u32,
    elided: []const bool,
    chain_end: []const u32,
    /// Per chain end: its links (slots, first to itself) in chain_links and the slot the first link adds onto.
    chain_start: []const u32,
    chain_len: []const u32,
    chain_base: []const u32,
    chain_links: []const u32,
    /// Per slot: its link index in its chain, or no_slot; per chain end, where its link bits start (one bit per link).
    chain_pos: []const u32,
    chain_bits_at: []const u32,
    chain_words: u32,
    /// Per head plan: its group of heads with the same input slots and selecting dims (one key per step), or no_slot.
    plan_group: []const u32,
    groups: u32,
};

/// Dims a set of forms reads: the union of their sources, ascending.
fn reads(a: std.mem.Allocator, sets: []const *const model.Sparse) ![]u32 {
    var out: std.ArrayList(u32) = .empty;
    for (sets) |m| try out.appendSlice(a, m.used);
    std.mem.sort(u32, out.items, {}, std.sort.asc(u32));
    var n: usize = 0;
    for (out.items) |d| {
        if (n > 0 and out.items[n - 1] == d) continue;
        out.items[n] = d;
        n += 1;
    }
    return out.items[0..n];
}

/// Heads reading the same slots through the same selecting dims share one key per step and one input-frame ring.
fn selectionGroups(a: std.mem.Allocator, units: []const Unit, plans: []head.Plan) !struct { []u32, u32 } {
    const group = try a.alloc(u32, plans.len);
    @memset(group, no_slot);
    var n: u32 = 0;
    for (units, 0..) |u, i| if (u.kind == .heads and !plans[u.plan].general) {
        for (units[0..i]) |v| if (v.kind == .heads and !plans[v.plan].general and std.mem.eql(u32, u.ins, v.ins) and
            std.mem.eql(bool, plans[u.plan].sel_dim, plans[v.plan].sel_dim))
        {
            if (group[v.plan] == no_slot) {
                group[v.plan] = n;
                n += 1;
            }
            group[u.plan] = group[v.plan];
            plans[u.plan].owner = plans[v.plan].owner;
            break;
        };
    };
    const size = try a.alloc(u32, n);
    defer a.free(size);
    @memset(size, 0);
    for (plans, group) |*p, g| if (g != no_slot) {
        p.member = size[g];
        size[g] += 1;
    };
    return .{ group, n };
}

fn maxLag(h: *const model.Heads) u32 {
    var lag: u32 = 0;
    for (h.taps) |t| for (1..model.lags) |k| {
        if (t[k] != 0) lag = @max(lag, @as(u32, @intCast(k)));
    };
    return lag;
}

pub fn build(a: std.mem.Allocator, c: *const model.Circuit) !Flow {
    const visible = try a.alloc(u32, c.dims);
    @memset(visible, zero_slot);
    var slots: u32 = 1;
    var feature_dims: std.ArrayList(u32) = .empty;
    var feature_slots: std.ArrayList(u32) = .empty;
    for (c.features) |f| if (visible[f.dim] == zero_slot) {
        visible[f.dim] = slots;
        try feature_dims.append(a, f.dim);
        try feature_slots.append(a, slots);
        slots += 1;
    };
    var units: std.ArrayList(Unit) = .empty;
    var plans: std.ArrayList(head.Plan) = .empty;
    var ffn_plans: std.ArrayList(ffn.Plan) = .empty;
    var ffn_totals: [3]u32 = @splat(0);
    var max_dims: u32 = 0;
    for (c.batches) |*b| switch (b.*) {
        .ffn => |*whole| for (whole.split) |*f| {
            const dims = if (f.dims.len > 0) f.dims else try reads(a, &.{ &f.up.forms, &f.gate.forms });
            var u = try unit(a, .{ .ffn = f }, dims, f.outs, 0, visible, &slots);
            u.plan = @intCast(ffn_plans.items.len);
            u.quiet = try ffn.quiet(a, f);
            try ffn_plans.append(a, try ffn.build(a, f, &ffn_totals));
            try units.append(a, u);
        },
        .heads => |*hs| for (hs.split) |*h| {
            var u = try unit(a, .{ .heads = h }, try reads(a, &.{&h.proj.forms}), h.outs, maxLag(h), visible, &slots);
            u.plan = @intCast(plans.items.len);
            var plan = try head.build(a, h, u.dims);
            plan.owner = u.plan;
            try plans.append(a, plan);
            try units.append(a, u);
        },
    };
    var total_ins: u32 = 0;
    for (units.items) |*u| {
        max_dims = @max(max_dims, @as(u32, @intCast(u.dims.len)));
        u.in_off = total_ins;
        total_ins += @intCast(u.dims.len);
    }
    const mark = try a.alloc(u32, slots);
    defer a.free(mark);
    const code = try a.alloc(u32, c.code_dims.len);
    for (c.code_dims, code) |d, *s| s.* = visible[d];
    const ch = try chains(a, units.items, slots, visible);
    const plan_group, const groups = try selectionGroups(a, units.items, plans.items);
    const ins_csr = try readersOf(a, units.items, slots, mark, .ins, ch);
    const base_csr = try readersOf(a, units.items, slots, mark, .base, ch);
    return .{
        .slots = slots,
        .units = units.items,
        .plans = plans.items,
        .ffn_plans = ffn_plans.items,
        .ffn_totals = ffn_totals,
        .feature_dims = feature_dims.items,
        .feature_slots = feature_slots.items,
        .reader_starts = ins_csr[0],
        .readers = ins_csr[1],
        .reader_local = ins_csr[2],
        .reader_off = ins_csr[3],
        .base_starts = base_csr[0],
        .base_readers = base_csr[1],
        .base_out = base_csr[2],
        .base_first = base_csr[3],
        .final = visible,
        .code = code,
        .max_dims = max_dims,
        .total_ins = total_ins,
        .plan_group = plan_group,
        .groups = groups,
        .slot_unit = ch.slot_unit,
        .elided = ch.elided,
        .chain_end = ch.end,
        .chain_start = ch.start,
        .chain_len = ch.len,
        .chain_base = ch.base,
        .chain_links = ch.links,
        .chain_pos = ch.pos,
        .chain_bits_at = ch.bits_at,
        .chain_words = ch.words,
    };
}

/// Reader edges by slot: units reading it as an input (deduplicated), or each output adding onto it (in-unit too).
fn readersOf(a: std.mem.Allocator, units: []const Unit, slots: u32, mark: []u32, comptime field: enum { ins, base }, ch: Chains) !struct { []u32, []u32, []u32, []u32 } {
    @memset(mark, no_slot);
    var edge_slot: std.ArrayList(u32) = .empty;
    var edge: [3]std.ArrayList(u32) = @splat(.empty);
    for (units, 0..) |u, i| {
        const first_out = if (u.outs.len > 0) u.outs[0] else slots;
        for (if (field == .ins) u.ins else u.base, 0..) |s, j| {
            if (s == zero_slot or s == no_slot) continue;
            if (field == .ins and (s >= first_out or mark[s] == i)) continue;
            mark[s] = @intCast(i);
            if (field == .base and ch.elided[s]) continue;
            var reader: u32 = @intCast(i);
            var k: u32 = @intCast(j);
            if (field == .base and ch.elided[u.outs[j]]) {
                const end = ch.end[u.outs[j]];
                reader = ch.slot_unit[end];
                k = end - units[reader].outs[0];
            }
            try edge_slot.append(a, s);
            try edge[0].append(a, reader);
            try edge[1].append(a, k);
            try edge[2].append(a, if (field == .ins) u.in_off else units[reader].outs[0]);
        }
    }
    const starts = try a.alloc(u32, slots + 1);
    @memset(starts, 0);
    for (edge_slot.items) |s| starts[s + 1] += 1;
    for (1..starts.len) |s| starts[s] += starts[s - 1];
    var out: [3][]u32 = undefined;
    for (&out) |*o| o.* = try a.alloc(u32, edge_slot.items.len);
    const fill = try a.dupe(u32, starts[0..slots]);
    defer a.free(fill);
    for (edge_slot.items, 0..) |s, e| {
        for (0..3) |k| out[k][fill[s]] = edge[k].items[e];
        fill[s] += 1;
    }
    return .{ starts, out[0], out[1], out[2] };
}

const Chains = struct { slot_unit: []u32, elided: []bool, end: []u32, start: []u32, len: []u32, base: []u32, links: []u32, pos: []u32, bits_at: []u32, words: u32 };

/// Into-chains: an output slot no unit reads and the step does not end on is elided; its chain end folds the links.
fn chains(a: std.mem.Allocator, units: []const Unit, slots: u32, visible: []const u32) !Chains {
    var ch: Chains = undefined;
    ch.slot_unit = try a.alloc(u32, slots);
    @memset(ch.slot_unit, no_slot);
    const base = try a.alloc(u32, slots);
    const next = try a.alloc(u32, slots);
    const read = try a.alloc(bool, slots);
    defer a.free(base);
    defer a.free(next);
    defer a.free(read);
    @memset(base, no_slot);
    @memset(next, no_slot);
    @memset(read, false);
    for (visible) |v| read[v] = true;
    for (units, 0..) |u, i| {
        for (u.ins) |sl| read[sl] = true;
        for (u.outs, u.base) |o, b| {
            ch.slot_unit[o] = @intCast(i);
            base[o] = b;
            if (b != no_slot) next[b] = o;
        }
    }
    ch.elided = try a.alloc(bool, slots);
    for (ch.elided, 0..) |*e, sl| e.* = ch.slot_unit[sl] != no_slot and !read[sl] and next[sl] != no_slot;
    ch.end = try a.alloc(u32, slots);
    ch.start = try a.alloc(u32, slots);
    ch.len = try a.alloc(u32, slots);
    @memset(ch.end, no_slot);
    @memset(ch.len, 0);
    @memset(ch.start, 0);
    ch.base = try a.alloc(u32, slots);
    @memset(ch.base, no_slot);
    var links: std.ArrayList(u32) = .empty;
    ch.pos = try a.alloc(u32, slots);
    @memset(ch.pos, no_slot);
    ch.bits_at = try a.alloc(u32, slots);
    @memset(ch.bits_at, no_slot);
    ch.words = 0;
    for (0..slots) |sl| {
        if (ch.slot_unit[sl] == no_slot or ch.elided[sl] or base[sl] == no_slot or !ch.elided[base[sl]]) continue;
        var first: u32 = @intCast(sl);
        while (base[first] != no_slot and ch.elided[base[first]]) first = base[first];
        ch.start[sl] = @intCast(links.items.len);
        var cur = first;
        while (true) {
            try links.append(a, cur);
            if (cur == sl) break;
            ch.end[cur] = @intCast(sl);
            cur = next[cur];
        }
        ch.len[sl] = @intCast(links.items.len - ch.start[sl]);
        ch.base[sl] = base[first];
        for (links.items[ch.start[sl]..], 0..) |l, k| ch.pos[l] = @intCast(k);
        ch.bits_at[sl] = ch.words;
        ch.words += (ch.len[sl] + 63) / 64;
    }
    ch.links = links.items;
    return ch;
}

fn unit(a: std.mem.Allocator, kind: @FieldType(Unit, "kind"), dims: []const u32, outs: []const model.Out, lag: u32, visible: []u32, slots: *u32) !Unit {
    const ins = try a.alloc(u32, dims.len);
    for (dims, ins) |d, *s| s.* = visible[d];
    const out_slots = try a.alloc(u32, outs.len);
    const base = try a.alloc(u32, outs.len);
    for (outs, out_slots, base) |o, *s, *b| {
        b.* = if (o.into) visible[o.dim] else no_slot;
        s.* = slots.*;
        visible[o.dim] = slots.*;
        slots.* += 1;
    }
    return .{ .kind = kind, .dims = dims, .ins = ins, .outs = out_slots, .base = base, .lag = lag };
}
