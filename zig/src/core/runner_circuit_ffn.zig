//! An FFN batch part on the CPU: every form by scatter, then every firing unit by the sorted walk.
const std = @import("std");
const model = @import("runner_circuit_model.zig");
const State = @import("runner_circuit_state.zig").State;

const Scalar = model.Scalar;
const none = model.no_form;

pub const Plan = struct {
    f: *const model.Ffn,
    /// The part's up and gate forms with dense rows for inputs that reach many forms.
    up: model.Sparse,
    gate: model.Sparse,
    /// Where this part's forms and nodes live in the per-machine arrays.
    up_off: u32,
    gate_off: u32,
    node_off: u32,
};

pub fn build(a: std.mem.Allocator, f: *const model.Ffn, offs: *[3]u32) !Plan {
    const p: Plan = .{
        .f = f,
        .up = try f.up.forms.densify(a),
        .gate = try f.gate.forms.densify(a),
        .up_off = offs[0],
        .gate_off = offs[1],
        .node_off = offs[2],
    };
    offs[0] += f.up.forms.outputs;
    offs[1] += f.gate.forms.outputs;
    offs[2] += @intCast(f.outs.len);
    return p;
}

/// The part's outputs for the running inputs (nz sources ascending, values in x); silent units (gate <= 0) add zero.
pub fn run(s: *State, p: *const Plan, x: []const Scalar, nz: []const u32) []const Scalar {
    const f = p.f;
    const up = s.ffn_up[p.up_off..][0..f.up.forms.outputs];
    const gate = s.ffn_gate[p.gate_off..][0..f.gate.forms.outputs];
    const y = s.ffn_y[p.node_off..][0..f.outs.len];
    @memset(y, 0);
    p.up.apply(x, nz, up);
    p.gate.apply(x, nz, gate);
    fire(f, up, gate, y);
    return y;
}

/// Whether every output is +0 when every input is zero: a fresh machine's zeroed outputs already match the part.
pub fn quiet(a: std.mem.Allocator, f: *const model.Ffn) !bool {
    const up = try a.alloc(Scalar, f.up.forms.outputs + f.gate.forms.outputs + f.outs.len);
    defer a.free(up);
    @memset(up, 0);
    const y = up[up.len - f.outs.len ..];
    fire(f, up[0..f.up.forms.outputs], up[f.up.forms.outputs..][0..f.gate.forms.outputs], y);
    for (y) |v| if (@as(u32, @bitCast(v)) != 0) return false;
    return true;
}

/// Adds every firing unit's output to y from the forms' values.
fn fire(f: *const model.Ffn, up: []const Scalar, gate: []const Scalar, y: []Scalar) void {
    for (gate, f.starts[0 .. f.starts.len - 1], f.starts[1..]) |v, lo, hi| for (f.order[lo..hi]) |t| {
        const g = f.gate.bias[t] + v;
        if (!(g > 0)) break;
        y[f.slot[t]] += f.coef[t] * upValue(f, t, up) * g;
    };
    for (f.always) |t| y[f.slot[t]] += f.coef[t] * upValue(f, t, up) * f.gate.bias[t];
}

fn upValue(f: *const model.Ffn, t: u32, up: []const Scalar) Scalar {
    const uf = f.up.form[t];
    return if (uf == none) f.up.bias[t] else f.up.bias[t] + up[uf];
}
