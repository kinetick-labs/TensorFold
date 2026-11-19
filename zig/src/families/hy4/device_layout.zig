//! Where every planned GPU tensor lands in the one device arena the Hy4 loader allocates: 256-byte aligned parts
//! (an EXL3 group's trellis, suh and svh in the stored [K/16, N/16, 16 * bits] layout the trellis kernels read with
//! cuda.exl3.strides(k, n, halves, false); a plain tensor as stored), and each decoder layer's slots by role, so the
//! forward never looks a name up. Built from the load plan alone: no GPU, no tensor bytes.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Config = @import("config.zig");
const load_plan = @import("load_plan.zig");

pub const alignment: u64 = 256;

pub const Part = struct { offset: u64, bytes: u64 };

pub const Slot = struct {
    entry: u32,
    /// plain: [0]; EXL3: trellis, suh, svh
    parts: [3]Part,
    exl3: ?load_plan.Exl3,
    dtype: ?load_plan.Dtype,

    pub fn offset(s: Slot) u64 {
        return s.parts[0].offset;
    }
};

/// Words between k tiles and 128-column blocks of the stored layout: cuda.exl3.strides(k, n, halves, false).
pub fn storedStrides(e: load_plan.Exl3) struct { k: u64, nb: u64 } {
    const words: u64 = @as(u64, e.halves) * 4;
    return .{ .k = (@as(u64, e.n) / 16) * words, .nb = 8 * words };
}

pub const none = std.math.maxInt(u32);

pub const Hc = struct { fn_: u32 = none, base: u32 = none, scale: u32 = none };

pub const Layer = struct {
    input_norm: u32 = none,
    post_attention_norm: u32 = none,
    attn_hc: Hc = .{},
    ffn_hc: Hc = .{},
    q_a: u32 = none,
    q_a_norm: u32 = none,
    q_b: u32 = none,
    kv_a: u32 = none,
    kv_a_norm: u32 = none,
    kv_b: u32 = none,
    gate: u32 = none,
    o: u32 = none,
    post_proj_norm: u32 = none,
    sinks: u32 = none,
    /// compressed indexer (indexed layers): query, token key, head weights, key/value, key gate, norms, ape
    indexer: [8]u32 = @splat(none),
    // dense MLP
    mlp_gate: u32 = none,
    mlp_up: u32 = none,
    mlp_down: u32 = none,
    post_mlp_norm: u32 = none,
    // latent MoE
    router: u32 = none,
    router_bias: u32 = none,
    to_latent: u32 = none,
    from_latent: u32 = none,
    combine_norm: u32 = none,
    shared: [3]u32 = @splat(none),
    /// slot of expert 0's gate_proj; expert e's gate, up, down are `experts + 3 * e + {0, 1, 2}`
    experts: u32 = none,
    /// Engram layer's GPU-side tensors (key projections, norms, value projection, scales); its tables stay on NVMe
    engram: bool = false,
};

pub const Layout = struct {
    slots: []const Slot,
    names: std.StringHashMapUnmanaged(u32),
    layers: []const Layer,
    embed: u32,
    norm: u32,
    lm_head: u32,
    head_hc: Hc,
    total: u64,

    pub fn find(l: *const Layout, name: []const u8) ?u32 {
        return l.names.get(name);
    }
};

fn round(x: u64) u64 {
    return std.mem.alignForward(u64, x, alignment);
}

/// The arena layout of `plan`'s GPU entries (NVMe ones are skipped), and every layer's slots.
pub fn build(a: Allocator, c: *const Config.Config, plan: *const load_plan.Plan) !Layout {
    var slots: std.ArrayList(Slot) = .empty;
    var names: std.StringHashMapUnmanaged(u32) = .empty;
    var at: u64 = 0;
    for (plan.entries, 0..) |e, i| {
        if (e.place != .gpu) continue;
        var s: Slot = .{ .entry = @intCast(i), .parts = @splat(.{ .offset = 0, .bytes = 0 }), .exl3 = e.exl3, .dtype = e.dtype };
        if (e.exl3) |x| {
            const sizes = [3]u64{ @as(u64, x.k) * x.n * x.halves / 16, 2 * @as(u64, x.k), 2 * @as(u64, x.n) };
            for (&s.parts, sizes) |*p, n| {
                p.* = .{ .offset = at, .bytes = n };
                at = round(at + n);
            }
        } else {
            s.parts[0] = .{ .offset = at, .bytes = e.bytes };
            at = round(at + e.bytes);
        }
        try names.put(a, e.name, @intCast(slots.items.len));
        try slots.append(a, s);
    }
    var l: Layout = .{ .slots = slots.items, .names = names, .layers = &.{}, .embed = none, .norm = none, .lm_head = none, .head_hc = .{}, .total = at };
    const need = struct {
        fn f(lay: *const Layout, al: Allocator, comptime fmt: []const u8, args: anytype) !u32 {
            const n = try std.fmt.allocPrint(al, fmt, args);
            return lay.find(n) orelse {
                std.log.err("hy4 layout: no slot for {s}", .{n});
                return error.MissingSlot;
            };
        }
    }.f;
    // a projection is either `name` (EXL3) or `name.weight` (plain)
    const proj = struct {
        fn f(lay: *const Layout, al: Allocator, comptime fmt: []const u8, args: anytype) !u32 {
            const n = try std.fmt.allocPrint(al, fmt, args);
            if (lay.find(n)) |s| return s;
            const w = try std.fmt.allocPrint(al, "{s}.weight", .{n});
            return lay.find(w) orelse {
                std.log.err("hy4 layout: no slot for {s}", .{n});
                return error.MissingSlot;
            };
        }
    }.f;
    l.embed = try need(&l, a, "model.embed_tokens.weight", .{});
    l.norm = try need(&l, a, "model.norm.weight", .{});
    l.lm_head = try proj(&l, a, "lm_head", .{});
    l.head_hc = .{ .fn_ = try need(&l, a, "model.hc_head.hc_head_fn", .{}), .base = try need(&l, a, "model.hc_head.hc_head_base", .{}), .scale = try need(&l, a, "model.hc_head.hc_head_scale", .{}) };
    const t = &c.text;
    const layers = try a.alloc(Layer, t.layers);
    for (layers, 0..) |*y, li| {
        const p = try std.fmt.allocPrint(a, "model.layers.{d}.", .{li});
        y.* = .{};
        y.input_norm = try need(&l, a, "{s}input_layernorm.weight", .{p});
        y.post_attention_norm = try need(&l, a, "{s}post_attention_layernorm.weight", .{p});
        y.attn_hc = .{ .fn_ = try need(&l, a, "{s}hc_attn_layer.hc_pre.hc_fn", .{p}), .base = try need(&l, a, "{s}hc_attn_layer.hc_pre.hc_base", .{p}), .scale = try need(&l, a, "{s}hc_attn_layer.hc_pre.hc_scale", .{p}) };
        y.ffn_hc = .{ .fn_ = try need(&l, a, "{s}hc_mlp_layer.hc_pre.hc_fn", .{p}), .base = try need(&l, a, "{s}hc_mlp_layer.hc_pre.hc_base", .{p}), .scale = try need(&l, a, "{s}hc_mlp_layer.hc_pre.hc_scale", .{p}) };
        y.q_a = try need(&l, a, "{s}self_attn.q_a_proj.weight", .{p});
        y.q_a_norm = try need(&l, a, "{s}self_attn.q_a_layernorm.weight", .{p});
        y.q_b = try proj(&l, a, "{s}self_attn.q_b_proj", .{p});
        y.kv_a = try need(&l, a, "{s}self_attn.kv_a_proj_with_mqa.weight", .{p});
        y.kv_a_norm = try need(&l, a, "{s}self_attn.kv_a_layernorm.weight", .{p});
        y.kv_b = try need(&l, a, "{s}self_attn.kv_b_proj.weight", .{p});
        y.gate = try proj(&l, a, "{s}self_attn.linear_gate", .{p});
        y.o = try proj(&l, a, "{s}self_attn.o_proj", .{p});
        y.post_proj_norm = try need(&l, a, "{s}self_attn.post_proj_norm.weight", .{p});
        y.sinks = try need(&l, a, "{s}self_attn.learnable_sink_param", .{p});
        if (t.attn[li] == .indexed) for (&y.indexer, [_][]const u8{ "query_projection.weight", "token_key_projection.weight", "head_weight_projection.weight", "key_value_projection.weight", "key_gate_projection.weight", "token_key_norm.weight", "key_norm.weight", "intra_group_ape" }) |*s, n| {
            s.* = try need(&l, a, "{s}self_attn.compressed_indexer.{s}", .{ p, n });
        };
        switch (t.mlp[li]) {
            .dense => {
                y.mlp_gate = try proj(&l, a, "{s}mlp.gate_proj", .{p});
                y.mlp_up = try proj(&l, a, "{s}mlp.up_proj", .{p});
                y.mlp_down = try proj(&l, a, "{s}mlp.down_proj", .{p});
                y.post_mlp_norm = try need(&l, a, "{s}mlp.post_fc2_norm.weight", .{p});
            },
            .sparse => {
                y.router = try need(&l, a, "{s}mlp.gate.weight", .{p});
                y.router_bias = try need(&l, a, "{s}mlp.gate.e_score_correction_bias", .{p});
                y.to_latent = try need(&l, a, "{s}mlp.routed_expert_down_proj.weight", .{p});
                y.from_latent = try need(&l, a, "{s}mlp.routed_expert_up_proj.weight", .{p});
                y.combine_norm = try need(&l, a, "{s}mlp.combine_output_norm.weight", .{p});
                for (&y.shared, [_][]const u8{ "gate_proj", "up_proj", "down_proj" }) |*s, n| s.* = try proj(&l, a, "{s}mlp.shared_experts.{s}", .{ p, n });
                y.experts = try proj(&l, a, "{s}mlp.experts.0.gate_proj", .{p});
                // the stride the forward relies on: expert e's three groups follow expert 0's in plan order
                for (0..t.experts) |e| for ([_][]const u8{ "gate_proj", "up_proj", "down_proj" }, 0..) |n, j| {
                    if (try proj(&l, a, "{s}mlp.experts.{d}.{s}", .{ p, e, n }) != y.experts + 3 * e + j) return error.ExpertOrder;
                };
            },
        }
        y.engram = t.engramAt(@intCast(li)) != null;
    }
    l.layers = layers;
    return l;
}

/// Whether the arena fits beside `reserve` bytes (KV, activations, the vision tower's scratch) in `free` bytes.
pub fn fits(l: *const Layout, free: u64, reserve: u64) bool {
    return l.total + reserve <= free;
}
