//! A plain f32 CPU reference of one Hy4-flash decoder layer (transformers HYV4FlashDecoderLayer, PR #50903) with
//! the dense MLP: hyper-connection mixing (sigmoid, no Sinkhorn), gated MLA with learned sinks, interleaved 3D M-RoPE,
//! a sliding window or the compressed indexer's group recall, and the sandwich norms. It is the oracle the CUDA kernels
//! are checked against, not a fast path: every token is recomputed and nothing is cached.

const std = @import("std");
const Allocator = std.mem.Allocator;

pub const Index = struct {
    heads: u32,
    dim: u32,
    kpool: u32,
    /// `index_topk`: the context length past which the indexer recalls groups instead of attending densely
    topk: u32,
};

pub const Dims = struct {
    hidden: u32,
    hc: u32,
    heads: u32,
    nope: u32,
    rope: u32,
    v: u32,
    q_lora: u32,
    kv_lora: u32,
    inter: u32,
    eps: f32,
    hc_eps: f32,
    hc_magnitude: f32,
    theta: f64,
    sections: [3]u32,
    /// sliding layers: keys `i - window < j <= i`
    window: ?u32 = null,
    /// indexed layers
    index: ?Index = null,
};

pub const Hc = struct { fn_: []const f32, base: []const f32, scale: []const f32 };

pub const IndexWeights = struct {
    query: []const f32, // [heads * dim, q_lora]
    head_weight: []const f32, // [heads, hidden]
    token_key: []const f32, // [dim, hidden]
    token_key_norm: []const f32,
    key_value: []const f32, // [dim, dim]
    key_gate: []const f32, // [dim, dim]
    key_norm: []const f32,
    ape: []const f32, // [kpool, dim]
};

pub const Weights = struct {
    attn_hc: Hc,
    ffn_hc: Hc,
    input_norm: []const f32,
    post_attention_norm: []const f32,
    post_mlp_norm: []const f32,
    q_a: []const f32,
    q_a_norm: []const f32,
    q_b: []const f32,
    kv_a: []const f32,
    kv_a_norm: []const f32,
    kv_b: []const f32,
    gate: []const f32,
    o: []const f32,
    post_proj_norm: []const f32,
    sinks: []const f32,
    mlp_gate: []const f32,
    mlp_up: []const f32,
    mlp_down: []const f32,
    index: ?IndexWeights = null,
};

/// `out[o] = W[o, :] . x` for an nn.Linear weight `[out.len, x.len]`.
fn linear(out: []f32, w: []const f32, x: []const f32) void {
    std.debug.assert(w.len == out.len * x.len);
    for (out, 0..) |*o, r| {
        var acc: f32 = 0;
        for (w[r * x.len ..][0..x.len], x) |a, b| acc += a * b;
        o.* = acc;
    }
}

fn invRms(x: []const f32, eps: f32) f32 {
    var s: f32 = 0;
    for (x) |e| s += e * e;
    return 1.0 / @sqrt(s / @as(f32, @floatFromInt(x.len)) + eps);
}

fn rmsNorm(out: []f32, x: []const f32, w: []const f32, eps: f32) void {
    const r = invRms(x, eps);
    for (out, x, w) |*o, e, g| o.* = g * (e * r);
}

fn sigmoid(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
}

fn silu(x: f32) f32 {
    return x * sigmoid(x);
}

/// The rotary angle of pair `k` (of `rope / 2`) at a 3D position: pairs `1, 4, 7 ...` below `3 * sections[1]` follow
/// the height, `2, 5, 8 ...` below `3 * sections[2]` the width, every other one the time.
pub fn angle(d: *const Dims, pos: [3]i32, k: u32) f64 {
    const plane: usize = if (k % 3 == 1 and k < 3 * d.sections[1]) 1 else if (k % 3 == 2 and k < 3 * d.sections[2]) 2 else 0;
    const inv = 1.0 / std.math.pow(f64, d.theta, @as(f64, @floatFromInt(2 * k)) / @as(f64, @floatFromInt(d.rope)));
    return @as(f64, @floatFromInt(pos[plane])) * @as(f32, @floatCast(inv));
}

/// GLM-style rotation of adjacent pairs, in place.
pub fn rotate(d: *const Dims, x: []f32, pos: [3]i32) void {
    std.debug.assert(x.len == d.rope);
    for (0..d.rope / 2) |k| {
        const t: f32 = @floatCast(angle(d, pos, @intCast(k)));
        const c = @cos(t);
        const s = @sin(t);
        const a = x[2 * k];
        const b = x[2 * k + 1];
        x[2 * k] = a * c - b * s;
        x[2 * k + 1] = b * c + a * s;
    }
}

/// One hyper-connection: the `post` weights of each stream and the streams mixed into one input.
fn hyper(d: *const Dims, h: Hc, streams: []const f32, post: []f32, out: []f32) void {
    const n = d.hc;
    var mix: [16]f32 = undefined;
    std.debug.assert(2 * n <= mix.len);
    linear(mix[0 .. 2 * n], h.fn_, streams);
    const r = invRms(streams, d.eps);
    for (0..n) |s| {
        const pre = sigmoid(mix[s] * r * h.scale[0] + h.base[s]) + d.hc_eps;
        post[s] = d.hc_magnitude * sigmoid(mix[n + s] * r * h.scale[1] + h.base[n + s]) + d.hc_eps;
        if (s == 0) @memset(out, 0);
        for (out, streams[s * d.hidden ..][0..d.hidden]) |*o, e| o.* += pre * e;
    }
}

fn pos3(positions: []const i32, t: usize, i: usize) [3]i32 {
    return .{ positions[i], positions[t + i], positions[2 * t + i] };
}

/// Which keys each query may see: causal, then the window or the indexer's recalled groups. `[t][t]`.
fn allowed(a: Allocator, d: *const Dims, w: *const Weights, x: []const f32, q_resid: []const f32, positions: []const i32, t: usize) ![]bool {
    const ok = try a.alloc(bool, t * t);
    for (0..t) |i| for (0..t) |j| {
        ok[i * t + j] = j <= i and (if (d.window) |win| i - j < win else true);
    };
    const ix = d.index orelse return ok;
    if (t <= ix.topk) return ok;
    const iw = w.index.?;
    const dim = ix.dim;
    const groups = t / ix.kpool;
    // token keys, then gated pooling over each group's kpool slots, channel by channel
    const keys = try a.alloc(f32, t * dim);
    defer a.free(keys);
    const tmp = try a.alloc(f32, dim);
    defer a.free(tmp);
    for (0..t) |i| {
        linear(tmp, iw.token_key, x[i * d.hidden ..][0..d.hidden]);
        rmsNorm(keys[i * dim ..][0..dim], tmp, iw.token_key_norm, d.eps);
    }
    const gk = try a.alloc(f32, groups * dim);
    defer a.free(gk);
    const vals = try a.alloc(f32, ix.kpool * dim);
    defer a.free(vals);
    const gates = try a.alloc(f32, ix.kpool * dim);
    defer a.free(gates);
    const rd = d.rope;
    for (0..groups) |g| {
        for (0..ix.kpool) |s| {
            const tok = keys[(g * ix.kpool + s) * dim ..][0..dim];
            linear(vals[s * dim ..][0..dim], iw.key_value, tok);
            linear(gates[s * dim ..][0..dim], iw.key_gate, tok);
            for (gates[s * dim ..][0..dim], iw.ape[s * dim ..][0..dim]) |*e, p| e.* += p;
        }
        for (0..dim) |c| {
            var most: f32 = -std.math.inf(f32);
            for (0..ix.kpool) |s| most = @max(most, gates[s * dim + c]);
            var sum: f32 = 0;
            var acc: f32 = 0;
            for (0..ix.kpool) |s| {
                const e = @exp(gates[s * dim + c] - most);
                sum += e;
                acc += vals[s * dim + c] * e;
            }
            tmp[c] = acc / sum;
        }
        const k = gk[g * dim ..][0..dim];
        rmsNorm(k, tmp, iw.key_norm, d.eps);
        rotate(d, k[dim - rd ..], pos3(positions, t, g * ix.kpool));
    }
    const q = try a.alloc(f32, ix.heads * dim);
    defer a.free(q);
    const hw = try a.alloc(f32, ix.heads);
    defer a.free(hw);
    const logits = try a.alloc(f32, groups);
    defer a.free(logits);
    const recall = ix.topk / ix.kpool - 1;
    const hscale = 1.0 / @sqrt(@as(f32, @floatFromInt(ix.heads * dim)));
    for (0..t) |i| {
        linear(q, iw.query, q_resid[i * d.q_lora ..][0..d.q_lora]);
        for (0..ix.heads) |h| rotate(d, q[h * dim + dim - rd ..][0..rd], pos3(positions, t, i));
        linear(hw, iw.head_weight, x[i * d.hidden ..][0..d.hidden]);
        const own = i / ix.kpool;
        for (0..groups) |g| {
            var l: f32 = 0;
            for (0..ix.heads) |h| {
                var s: f32 = 0;
                for (q[h * dim ..][0..dim], gk[g * dim ..][0..dim]) |p, e| s += p * e;
                l += hw[h] * hscale * @max(s, 0);
            }
            logits[g] = if (g >= own) -std.math.inf(f32) else l;
        }
        var keep: [256]bool = @splat(false);
        std.debug.assert(groups <= keep.len);
        // the `recall` best closed groups (fewer exist only before the query's own group closes any)
        for (0..@min(recall, own)) |_| {
            var best: usize = 0;
            var top = -std.math.inf(f32);
            for (0..own) |g| if (!keep[g] and logits[g] > top) {
                top = logits[g];
                best = g;
            };
            keep[best] = true;
        }
        for (0..t) |j| {
            const g = j / ix.kpool;
            if (!(keep[g] or g == own)) ok[i * t + j] = false;
        }
    }
    return ok;
}

/// Gated MLA with sinks over `x` `[t][hidden]` (already normed); `out` `[t][hidden]`.
fn attention(a: Allocator, d: *const Dims, w: *const Weights, x: []const f32, positions: []const i32, t: usize, out: []f32) !void {
    const H = d.heads;
    const qk = d.nope + d.rope;
    const q_resid = try a.alloc(f32, t * d.q_lora);
    defer a.free(q_resid);
    const q = try a.alloc(f32, t * H * qk);
    defer a.free(q);
    const kpass = try a.alloc(f32, t * d.kv_lora);
    defer a.free(kpass);
    const krot = try a.alloc(f32, t * d.rope);
    defer a.free(krot);
    const kv = try a.alloc(f32, t * H * (d.nope + d.v));
    defer a.free(kv);
    const gate = try a.alloc(f32, t * H * d.v);
    defer a.free(gate);
    const tmp = try a.alloc(f32, @max(d.q_lora, d.kv_lora + d.rope));
    defer a.free(tmp);
    for (0..t) |i| {
        const xi = x[i * d.hidden ..][0..d.hidden];
        linear(tmp[0..d.q_lora], w.q_a, xi);
        rmsNorm(q_resid[i * d.q_lora ..][0..d.q_lora], tmp[0..d.q_lora], w.q_a_norm, d.eps);
        linear(q[i * H * qk ..][0 .. H * qk], w.q_b, q_resid[i * d.q_lora ..][0..d.q_lora]);
        for (0..H) |h| rotate(d, q[(i * H + h) * qk + d.nope ..][0..d.rope], pos3(positions, t, i));
        linear(tmp[0 .. d.kv_lora + d.rope], w.kv_a, xi);
        rmsNorm(kpass[i * d.kv_lora ..][0..d.kv_lora], tmp[0..d.kv_lora], w.kv_a_norm, d.eps);
        @memcpy(krot[i * d.rope ..][0..d.rope], tmp[d.kv_lora..][0..d.rope]);
        rotate(d, krot[i * d.rope ..][0..d.rope], pos3(positions, t, i));
        linear(kv[i * H * (d.nope + d.v) ..][0 .. H * (d.nope + d.v)], w.kv_b, kpass[i * d.kv_lora ..][0..d.kv_lora]);
        linear(gate[i * H * d.v ..][0 .. H * d.v], w.gate, xi);
    }
    const ok = try allowed(a, d, w, x, q_resid, positions, t);
    defer a.free(ok);
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(qk)));
    const scores = try a.alloc(f32, t);
    defer a.free(scores);
    const heads_out = try a.alloc(f32, H * d.v);
    defer a.free(heads_out);
    for (0..t) |i| {
        for (0..H) |h| {
            const qi = q[(i * H + h) * qk ..][0..qk];
            var most = w.sinks[h];
            for (0..t) |j| {
                if (!ok[i * t + j]) continue;
                const kj = kv[(j * H + h) * (d.nope + d.v) ..][0..d.nope];
                var s: f32 = 0;
                for (qi[0..d.nope], kj) |p, e| s += p * e;
                for (qi[d.nope..], krot[j * d.rope ..][0..d.rope]) |p, e| s += p * e;
                scores[j] = s * scale;
                most = @max(most, scores[j]);
            }
            var sum = @exp(w.sinks[h] - most);
            const o = heads_out[h * d.v ..][0..d.v];
            @memset(o, 0);
            for (0..t) |j| {
                if (!ok[i * t + j]) continue;
                const p = @exp(scores[j] - most);
                sum += p;
                for (o, kv[(j * H + h) * (d.nope + d.v) + d.nope ..][0..d.v]) |*e, vv| e.* += p * vv;
            }
            for (o, gate[(i * H + h) * d.v ..][0..d.v]) |*e, g| e.* = e.* / sum * sigmoid(g);
        }
        const yi = out[i * d.hidden ..][0..d.hidden];
        const proj = try a.alloc(f32, d.hidden);
        defer a.free(proj);
        linear(proj, w.o, heads_out);
        rmsNorm(yi, proj, w.post_proj_norm, d.eps);
    }
}

/// One decoder layer over `streams` `[t][hc][hidden]`, in place; `positions` is `[3][t]` (time, height, width).
pub fn forward(a: Allocator, d: *const Dims, w: *const Weights, positions: []const i32, t: usize, streams: []f32) !void {
    const n = d.hc;
    const H = d.hidden;
    std.debug.assert(streams.len == t * n * H and positions.len == 3 * t);
    const post = try a.alloc(f32, t * n);
    defer a.free(post);
    const x = try a.alloc(f32, t * H);
    defer a.free(x);
    const y = try a.alloc(f32, t * H);
    defer a.free(y);
    const mixed = try a.alloc(f32, H);
    defer a.free(mixed);
    for (0..t) |i| {
        hyper(d, w.attn_hc, streams[i * n * H ..][0 .. n * H], post[i * n ..][0..n], mixed);
        rmsNorm(x[i * H ..][0..H], mixed, w.input_norm, d.eps);
    }
    try attention(a, d, w, x, positions, t, y);
    residual(d, streams, post, y, t);
    const up = try a.alloc(f32, d.inter);
    defer a.free(up);
    const gt = try a.alloc(f32, d.inter);
    defer a.free(gt);
    for (0..t) |i| {
        hyper(d, w.ffn_hc, streams[i * n * H ..][0 .. n * H], post[i * n ..][0..n], mixed);
        const xi = x[i * H ..][0..H];
        rmsNorm(xi, mixed, w.post_attention_norm, d.eps);
        linear(gt, w.mlp_gate, xi);
        linear(up, w.mlp_up, xi);
        for (gt, up) |*g, u| g.* = silu(g.*) * u;
        linear(mixed, w.mlp_down, gt);
        rmsNorm(y[i * H ..][0..H], mixed, w.post_mlp_norm, d.eps);
    }
    residual(d, streams, post, y, t);
}

/// `streams[s] += post[s] * y` per token.
fn residual(d: *const Dims, streams: []f32, post: []const f32, y: []const f32, t: usize) void {
    for (0..t) |i| for (0..d.hc) |s| {
        const p = post[i * d.hc + s];
        for (streams[(i * d.hc + s) * d.hidden ..][0..d.hidden], y[i * d.hidden ..][0..d.hidden]) |*e, v| e.* += p * v;
    };
}
