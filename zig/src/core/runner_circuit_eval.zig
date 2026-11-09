//! Exact CPU evaluation of one circuit step: token features, SwiGLU units, one-hot DeltaNet heads, output code.
const std = @import("std");
const model = @import("runner_circuit_model.zig");
const state = @import("runner_circuit_state.zig");
const flow = @import("runner_circuit_flow.zig");
const head = @import("runner_circuit_head.zig");
const ffn = @import("runner_circuit_ffn.zig");
const Memo = @import("runner_circuit_memo.zig").Memo;
const State = state.State;

const Scalar = model.Scalar;
const hd = model.hd;

/// The token's feature values: a feature dim holds its last nonzero feature, else zero.
fn embed(s: *State, token: u32, stamp: u64) !void {
    const c = s.circuit;
    const fam, const pay = c.split(token) orelse return error.CircuitUnknownToken;
    for (c.features) |f| {
        const v = f.value(fam, pay, c.family_bits);
        if (v != 0) s.x[f.dim] = v;
    }
    for (c.flow.feature_dims, c.flow.feature_slots) |d, slot| {
        set(s, slot, s.x[d], stamp);
        s.x[d] = 0;
    }
}

/// One token through the circuit in node order; only units marked active run or finish, the rest keep their outputs.
pub fn step(s: *State, token: u32) !u32 {
    const c = s.circuit;
    const stamp = s.steps + 1;
    // Only heads read tables; an FFN part matches its inputs once it ran, and from the start if it is quiet.
    if (s.force or s.always) {
        @memset(s.dirty, stamp);
        for (c.flow.units, 0..) |u, i| if (s.always or u.kind == .heads or (s.steps == 0 and !u.quiet)) mark(s.active, i);
    }
    s.force = false;
    try embed(s, token, stamp);
    for (s.active, 0..) |*word, w| while (word.* != 0) {
        const bit = @ctz(word.*);
        const i = w * 64 + bit;
        const u = &c.flow.units[i];
        if (s.dirty[i] == stamp) s.changed[i] = stamp;
        const due = (s.changed[i] != 0 and stamp - s.changed[i] <= u.lag) or s.unstable[i];
        if (due and !(u.kind == .heads and try head.skips(s, u.plan, i))) {
            try run(s, u, i, stamp);
        } else finish(s, u, i, stamp);
        // A unit's own outputs may re-mark it while it runs; clearing after the run drops that.
        word.* &= ~(@as(u64, 1) << @intCast(bit));
        if ((s.changed[i] != 0 and stamp + 1 - s.changed[i] <= u.lag) or s.unstable[i]) mark(s.keep, i);
    };
    std.mem.swap([]u64, &s.active, &s.keep);
    s.steps += 1;
    return decode(c, s);
}

fn mark(bits: []u64, i: usize) void {
    bits[i / 64] |= @as(u64, 1) << @intCast(i % 64);
}

/// A slot takes a new value; units reading it rerun this step when its bits change and keep their nonzero inputs.
fn set(s: *State, slot: u32, v: Scalar, stamp: u64) void {
    if (@as(u32, @bitCast(v)) == @as(u32, @bitCast(s.val[slot]))) return;
    s.val[slot] = v;
    const f = &s.circuit.flow;
    const lo = f.reader_starts[slot];
    const hi = f.reader_starts[slot + 1];
    for (f.readers[lo..hi], f.reader_local[lo..hi], f.reader_off[lo..hi]) |r, j, off| {
        s.dirty[r] = stamp;
        mark(s.active, r);
        const was = s.in_vals[off + j] != 0;
        s.in_vals[off + j] = v;
        if (was != (v != 0)) s.in_bits[(off + j) / 64] ^= @as(u64, 1) << @intCast((off + j) % 64);
    }
    const blo = f.base_starts[slot];
    const bhi = f.base_starts[slot + 1];
    for (f.base_readers[blo..bhi], f.base_out[blo..bhi], f.base_first[blo..bhi]) |r, k, first| queue(s, r, first, k, stamp);
}

/// Queues output k of unit r (first output slot `first`) to be finished this step: its value or base changed.
fn queue(s: *State, r: u32, first: u32, k: u32, stamp: u64) void {
    if (s.rb_mark[first + k] == stamp) return;
    s.rb_mark[first + k] = stamp;
    mark(s.active, r);
    if (s.rebase[r] != stamp) {
        s.rebase[r] = stamp;
        s.rb_len[r] = 0;
    }
    s.rb[first + s.rb_len[r]] = k;
    s.rb_len[r] += 1;
}

/// Sets every queued output of unit i: its value plus its base, or for a chain end the fold of its links in order.
fn finish(s: *State, u: *const flow.Unit, i: usize, stamp: u64) void {
    if (s.rebase[i] != stamp) return;
    const f = &s.circuit.flow;
    const first = u.outs[0];
    var at: usize = 0;
    while (at < s.rb_len[i]) : (at += 1) {
        const k = s.rb[first + at];
        const sl = first + k;
        var v: Scalar = undefined;
        if (f.chain_len[sl] > 0) {
            v = fold(s, sl);
        } else {
            const b = u.base[k];
            v = if (b == flow.no_slot) s.pre[sl] else s.val[b] + s.pre[sl];
        }
        set(s, sl, v, stamp);
    }
}

/// base + links in order over links not +0: adding +0 only turns a -0 sum into +0, and -0 is the empty-sum identity.
fn fold(s: *State, end: u32) Scalar {
    const f = &s.circuit.flow;
    const n = f.chain_len[end];
    const links = f.chain_links[f.chain_start[end]..][0..n];
    const b = f.chain_base[end];
    var v: Scalar = if (b == flow.no_slot) -0.0 else s.val[b];
    var added: usize = 0;
    for (s.chain_bits[f.chain_bits_at[end]..][0 .. (n + 63) / 64], 0..) |word, w| {
        var bits = word;
        while (bits != 0) : (bits &= bits - 1) {
            v = v + s.pre[links[w * 64 + @ctz(bits)]];
            added += 1;
        }
    }
    if (added < n and @as(u32, @bitCast(v)) == @as(u32, @bitCast(@as(Scalar, -0.0)))) v = 0;
    return v;
}

/// Records whether chain link `sl` now holds something other than +0.
fn linkBit(s: *State, sl: u32, v: Scalar) void {
    const f = &s.circuit.flow;
    const pos = f.chain_pos[sl];
    if (pos == flow.no_slot) return;
    const end = if (f.elided[sl]) f.chain_end[sl] else sl;
    const word = &s.chain_bits[f.chain_bits_at[end] + pos / 64];
    const bit = @as(u64, 1) << @intCast(pos % 64);
    if (@as(u32, @bitCast(v)) != 0) word.* |= bit else word.* &= ~bit;
}

/// Gathers a unit's nonzero inputs, ascending, into s.x, s.nz (dims) and s.local (input indices); returns their count.
fn nonzeros(s: *State, u: *const flow.Unit) usize {
    const lo = u.in_off;
    const hi = lo + u.dims.len;
    var n: usize = 0;
    var w = lo / 64;
    while (w * 64 < hi) : (w += 1) {
        var bits = s.in_bits[w];
        if (w * 64 < lo) bits &= ~@as(u64, 0) << @intCast(lo % 64);
        if ((w + 1) * 64 > hi) bits &= (@as(u64, 1) << @intCast(hi % 64)) - 1;
        while (bits != 0) : (bits &= bits - 1) {
            const j: u32 = @intCast(w * 64 + @ctz(bits) - lo);
            const d = u.dims[j];
            s.x[d] = s.in_vals[lo + j];
            s.nz[n] = d;
            s.local[n] = j;
            n += 1;
        }
    }
    return n;
}

fn run(s: *State, u: *const flow.Unit, i: usize, stamp: u64) !void {
    const n = nonzeros(s, u);
    const list = s.local[0..n];
    const nz = s.nz[0..n];
    defer for (nz) |d| {
        s.x[d] = 0;
    };
    const values = switch (u.kind) {
        .ffn => try ffnCached(s, u, &s.circuit.flow.ffn_plans[u.plan], i, nz, list),
        .heads => try head.run(s, u.plan, i, nz, list),
    };
    if (u.kind == .heads) s.rest[i] = for (values) |v| {
        if (@as(u32, @bitCast(v)) != 0) break false;
    } else true;
    if (u.outs.len == 0) return;
    const f = &s.circuit.flow;
    const first = u.outs[0];
    const pre = s.pre[first..][0..u.outs.len];
    if (!std.mem.eql(u32, @ptrCast(pre), @ptrCast(values))) for (values, pre, 0..) |v, *p, k| {
        if (@as(u32, @bitCast(v)) == @as(u32, @bitCast(p.*))) continue;
        p.* = v;
        const sl = first + @as(u32, @intCast(k));
        linkBit(s, sl, v);
        if (!f.elided[sl]) {
            queue(s, @intCast(i), first, @intCast(k), stamp);
            continue;
        }
        const end = f.chain_end[sl];
        const r = f.slot_unit[end];
        const end_first = f.units[r].outs[0];
        queue(s, r, end_first, end - end_first, stamp);
    };
    finish(s, u, i, stamp);
}

/// An FFN part is a pure function of its nonzero inputs: a memo hit returns the outputs it computed for them.
fn ffnCached(s: *State, u: *const flow.Unit, p: *const ffn.Plan, i: usize, nz: []const u32, local: []const u32) ![]const Scalar {
    const x, const ids = if (p.f.dims.len > 0) .{ s.in_vals[u.in_off..][0..u.dims.len], local } else .{ s.x, nz };
    if (s.always or s.memos[i].off() or s.memos[i].cold()) return ffn.run(s, p, x, ids);
    const key = s.key[0..nz.len];
    for (key, nz, local) |*k, d, j| k.* = @as(u64, j) << 32 | @as(u32, @bitCast(s.x[d]));
    const h = Memo.hash(key);
    if (s.memos[i].find(key, h)) |words| return @ptrCast(words);
    const y = ffn.run(s, p, x, ids);
    try s.memos[i].put(s.gpa, key, h, @ptrCast(y));
    return y;
}

/// The output code: family bits then 16 payload bits, each the sign of a code dim (+kappa is 1, -kappa is 0).
pub fn decode(c: *const model.Circuit, s: *const State) !u32 {
    var code: u64 = 0;
    for (c.flow.code, 0..) |slot, i| {
        const v = s.val[slot];
        if (v == 0 or !std.math.isFinite(v)) return error.CircuitOutputCode;
        code |= @as(u64, @intFromBool(v > 0)) << @intCast(i);
    }
    const mask = (@as(u64, 1) << @intCast(c.family_bits)) - 1;
    const fam = code & mask;
    const pay: u32 = @intCast(code >> @intCast(c.family_bits));
    if (fam >= c.families.len) return error.CircuitOutputFamily;
    const f = c.families[@intCast(fam)];
    if (pay >= f.size) return error.CircuitOutputPayload;
    return f.base + pay;
}
