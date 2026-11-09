//! Exact CPU evaluation of one circuit step: token features, SwiGLU units, one-hot DeltaNet heads, output code.
const std = @import("std");
const model = @import("runner_circuit_model.zig");
const State = @import("runner_circuit_state.zig").State;

const Scalar = model.Scalar;
const hd = model.hd;

/// The residual a token starts from: dim 0 is one, each feature dim a function of the token's family and payload.
pub fn embed(s: *State, token: u32) !void {
    const c = s.circuit;
    const fam, const pay = c.split(token) orelse return error.CircuitUnknownToken;
    s.clearResidual();
    s.x[0] = 1;
    s.touch(0);
    for (c.features) |f| {
        const v = f.value(fam, pay, c.family_bits);
        if (v == 0) continue;
        s.x[f.dim] = v;
        s.touch(f.dim);
    }
}

/// Feeds one token through every batch in stored node order and returns the token the output code names.
pub fn step(s: *State, token: u32) !u32 {
    const c = s.circuit;
    try embed(s, token);
    for (c.batches) |*b| switch (b.*) {
        .ffn => |*f| ffn(s, f),
        .heads => |*h| try heads(s, h),
    };
    s.steps += 1;
    return decode(c, s.x);
}

fn commit(s: *State, outs: []const model.Out, values: []const Scalar) void {
    for (outs, values) |o, v| {
        s.x[o.dim] = if (o.into) s.x[o.dim] + v else v;
        s.touch(o.dim);
    }
}

/// Silent units (gate <= 0) add exactly zero; each form's units are sorted by constant, so the walk stops early.
fn ffn(s: *State, f: *const model.Ffn) void {
    const y = s.work[0..f.outs.len];
    const up = s.work[f.outs.len..][0..f.up.forms.outputs];
    const gate = s.work[f.outs.len + up.len ..][0..f.gate.forms.outputs];
    @memset(y, 0);
    f.up.forms.apply(s.x, s.active.items, up);
    f.gate.forms.apply(s.x, s.active.items, gate);
    for (gate, f.starts[0 .. f.starts.len - 1], f.starts[1..]) |v, lo, hi| for (f.order[lo..hi]) |t| {
        const g = f.gate.bias[t] + v;
        if (!(g > 0)) break;
        unit(f, t, g, up, y);
    };
    for (f.always) |t| unit(f, t, f.gate.bias[t], up, y);
    commit(s, f.outs, y);
}

fn unit(f: *const model.Ffn, t: u32, g: Scalar, up: []const Scalar, y: []Scalar) void {
    const uf = f.up.form[t];
    const u = if (uf == model.no_form) f.up.bias[t] else f.up.bias[t] + up[uf];
    y[f.slot[t]] += f.coef[t] * u * g;
}

fn heads(s: *State, h: *const model.Heads) !void {
    const width = h.proj.columns();
    const p = s.work[0..width];
    const values = s.work[width..][0..h.outs.len];
    const rest = s.work[width + h.outs.len ..];
    const v = rest[2 * hd .. 3 * hd];
    const z = rest[3 * hd .. 4 * hd];
    const o = rest[4 * hd .. 5 * hd];
    h.proj.eval(s.x, s.active.items, rest[5 * hd ..][0..h.proj.forms.outputs], p);
    for (p[0..h.channels], h.taps, 0..) |*e, t, i| {
        const past = s.channelHistory(h.first_channel + @as(u32, @intCast(i)));
        const now = e.*;
        e.* = @max(t[0] * now + t[1] * past[0] + t[2] * past[1] + t[3] * past[2], 0);
        past[2] = past[1];
        past[1] = past[0];
        past[0] = now;
    }
    for (h.heads) |*head| {
        var key: ?u32 = null;
        for (head.k) |k| if (p[k.column] > 0) {
            if (key != null) return error.CircuitKeyNotOneHot;
            key = k.at;
        };
        var rows: [hd]u32 = undefined;
        var n: usize = 0;
        for (head.q) |q| if (p[q.column] > 0) {
            rows[n] = q.at;
            n += 1;
        };
        for (head.value_heads) |*vh| {
            if (vh.a) |col| try decay(s, vh.table, p[col]);
            if (key) |r| if (try writeEnabled(vh.beta, p)) {
                scatter(vh.v, p, v, false);
                try s.writeRow(vh.table, r, v);
            };
            const out = values[vh.out_first..][0..vh.out_cols.len];
            if (!try read(s, vh.table, rows[0..n], o)) {
                @memset(out, 0);
                continue;
            }
            scatter(vh.z, p, z, true);
            for (vh.out_cols, vh.out_w, out) |col, w, *y| y.* = o[col] * z[col] * w;
        }
    }
    commit(s, h.outs, values);
}

fn scatter(picks: []const model.Pick, p: []const Scalar, into: []Scalar, rectify: bool) void {
    @memset(into, 0);
    for (picks) |k| into[k.at] = if (rectify) @max(p[k.column], 0) else p[k.column];
}

/// beta: absent is 1; 1 writes, an integer <= 0 does not, anything else breaks the circuit's exact range.
fn writeEnabled(beta: ?u32, p: []const Scalar) !bool {
    const b = p[beta orelse return true];
    if (b == 1) return true;
    if (b <= 0 and b == @round(b)) return false;
    return error.CircuitWriteGate;
}

fn decay(s: *State, table: u32, selector: Scalar) !void {
    if (selector == model.keep_selector) return;
    if (!s.writable(table)) return error.CircuitTableReadOnly;
    if (selector == model.halve_selector) {
        var live = s.present[table];
        while (live != 0) : (live &= live - 1) {
            for (s.row(table, @ctz(live))) |*e| e.* *= 0.5;
        }
        return;
    }
    if (selector != model.clear_selector) return error.CircuitDecaySelector;
    @memset(s.tables[table], 0);
    s.present[table] = 0;
}

/// Sum of the rows a multi-hot query selects over the ballast norm o / sqrt((o0^2 + o1^2) / hd); false if none.
fn read(s: *State, table: u32, rows: []const u32, o: []Scalar) !bool {
    var any = false;
    for (rows) |r| {
        if (!s.rowPresent(table, r)) continue;
        if (!any) @memset(o, 0);
        for (o, s.row(table, r)) |*a, b| a.* += b;
        any = true;
    }
    if (!any) return false;
    const ref = o[0] * o[0] + o[1] * o[1];
    if (ref == 0) {
        for (o) |a| if (a != 0) return error.CircuitReadBallast;
        return false;
    }
    const norm = @sqrt(ref / @as(Scalar, hd));
    for (o) |*a| a.* /= norm;
    return true;
}

/// The output code: family bits then 16 payload bits, each the sign of a code dim (+kappa is 1, -kappa is 0).
pub fn decode(c: *const model.Circuit, x: []const Scalar) !u32 {
    var code: u64 = 0;
    for (c.code_dims, 0..) |d, i| {
        if (x[d] == 0 or !std.math.isFinite(x[d])) return error.CircuitOutputCode;
        code |= @as(u64, @intFromBool(x[d] > 0)) << @intCast(i);
    }
    const mask = (@as(u64, 1) << @intCast(c.family_bits)) - 1;
    const fam = code & mask;
    const pay: u32 = @intCast(code >> @intCast(c.family_bits));
    if (fam >= c.families.len) return error.CircuitOutputFamily;
    const f = c.families[@intCast(fam)];
    if (pay >= f.size) return error.CircuitOutputPayload;
    return f.base + pay;
}
