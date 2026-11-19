//! The loaded arena (cuda.zig's Device) as the forward's structures: each decoder layer's gm.Layer, the head, the
//! vision tower, and the Engram host side (hashers and the sidecar's FP8 rows with their scales). Small Engram
//! vectors that the kernels read as one `[n, width]` block (per-head norms, per-stream norms and scales) are copied
//! side by side into one extra buffer.

const std = @import("std");
const Allocator = std.mem.Allocator;
const cuda = @import("cuda");
const hy4 = @import("cuda.zig");
const config = hy4.config;
const load_plan = hy4.load_plan;
const device_layout = hy4.device_layout;
const pack_read = hy4.pack_read;
const engram_hash = @import("engram_hash.zig");
const gm = @import("gpu_model.zig");
const gv = @import("gpu_vision.zig");
const fwd = @import("fwd.zig");
const layer_ref = @import("layer_ref.zig");

pub fn dtOf(d: ?load_plan.Dtype) !fwd.Dt {
    return switch (d orelse return error.Hy4Dtype) {
        .F32 => .f32,
        .BF16 => .bf16,
        .F16 => .f16,
        else => error.Hy4Dtype,
    };
}

pub const Binder = struct {
    a: Allocator,
    d: *const cuda.Driver,
    dev: *const hy4.Device,
    plan: *const load_plan.Plan,
    extra: ?cuda.DeviceBuffer = null,
    extra_at: u64 = 0,

    fn slotOf(b: *const Binder, name: []const u8) !u32 {
        return b.dev.layout.find(name) orelse {
            std.log.err("hy4 bind: no slot {s}", .{name});
            return error.Hy4MissingSlot;
        };
    }

    fn named(b: *const Binder, comptime fmt: []const u8, args: anytype) !u32 {
        var buf: [320]u8 = undefined;
        return b.slotOf(try std.fmt.bufPrint(&buf, fmt, args));
    }

    pub fn w(b: *const Binder, slot: u32) !fwd.W {
        if (slot == device_layout.none) return error.Hy4MissingSlot;
        return .{ .ptr = b.dev.ptr(slot), .dt = try dtOf(b.dev.layout.slots[slot].dtype) };
    }

    fn shape(b: *const Binder, slot: u32) []const u64 {
        return b.plan.entries[b.dev.layout.slots[slot].entry].shape;
    }

    /// A projection slot: an EXL3 group, or a plain `[n, k...]` weight.
    pub fn lin(b: *const Binder, slot: u32) !gm.Lin {
        if (slot == device_layout.none) return error.Hy4MissingSlot;
        if (b.dev.layout.slots[slot].exl3 != null) return .{ .exl3 = try b.dev.weight(slot) };
        const sh = b.shape(slot);
        if (sh.len < 2) return error.Hy4Shape;
        var k: u64 = 1;
        for (sh[1..]) |x| k *= x;
        return .{ .plain = .{ .w = try b.w(slot), .k = @intCast(k), .n = @intCast(sh[0]) } };
    }

    fn linBias(b: *const Binder, ws: u32, bs: u32) !gm.Lin {
        var l = try b.lin(ws);
        switch (l) {
            .plain => |*p| p.b = try b.w(bs),
            .exl3 => return error.Hy4Shape,
        }
        return l;
    }

    fn proj(b: *const Binder, ws: u32, bs: u32) !gv.Proj {
        const l = try b.linBias(ws, bs);
        return .{ .w = l.plain.w, .b = l.plain.b, .k = l.plain.k, .n = l.plain.n };
    }

    /// Slots of the same dtype and size copied back to back into the extra buffer.
    fn joined(b: *Binder, slots: []const u32) !fwd.W {
        const first = b.dev.layout.slots[slots[0]];
        const each = first.parts[0].bytes;
        const at = b.extra.?.ptr + b.extra_at;
        for (slots, 0..) |s, i| {
            const sl = b.dev.layout.slots[s];
            if (sl.parts[0].bytes != each or sl.dtype != first.dtype) return error.Hy4Shape;
            try b.d.check(b.d.api.cuMemcpyDtoD_v2(at + i * each, b.dev.ptr(s), each), "cuMemcpyDtoD");
        }
        b.extra_at = std.mem.alignForward(u64, b.extra_at + each * slots.len, 256);
        if (b.extra_at > b.extra.?.len) return error.Hy4Shape;
        return .{ .ptr = at, .dt = try dtOf(first.dtype) };
    }

    fn hc(b: *const Binder, h: device_layout.Hc) fwd.Kernels.Hc {
        return .{ .fn_ = b.dev.ptr(h.fn_), .base = b.dev.ptr(h.base), .scale = b.dev.ptr(h.scale) };
    }
};

pub fn dims(t: *const config.Text, li: u32) layer_ref.Dims {
    const h = t.layerHeads(li);
    var d: layer_ref.Dims = .{ .hidden = t.hidden, .hc = t.hc_mult, .heads = h.heads, .nope = h.qk_nope, .rope = h.qk_rope, .v = h.v, .q_lora = t.q_lora, .kv_lora = t.kv_lora, .inter = t.dense_width, .eps = t.eps, .hc_eps = t.hc_eps, .hc_magnitude = t.hc_magnitude, .theta = h.theta, .sections = h.mrope };
    switch (t.attn[li]) {
        .sliding => d.window = t.window,
        .indexed => d.index = .{ .heads = t.index_heads, .dim = t.index_dim, .kpool = t.index_pool, .topk = t.index_topk },
    }
    return d;
}

pub const Model = struct {
    layers: []gm.Layer,
    head: gm.Head,
    tower: ?gv.Tower,
    extra: cuda.DeviceBuffer,
};

/// Every layer, the head and (`vision`) the tower over the loaded arena; `main` reads the tower's position grid.
pub fn model(a: Allocator, d: *const cuda.Driver, c: *const config.Config, dev: *const hy4.Device, plan: *const load_plan.Plan, main: *pack_read.Pack, vision: bool) !Model {
    const t = &c.text;
    const H = t.hidden;
    const n = t.hc_mult;
    const heads_e = t.hashHeads();
    const extra_bytes: u64 = @as(u64, t.engram_len) * (4096 + heads_e * 96 * 4 + 3 * n * H * 4 + 3 * 256);
    var b: Binder = .{ .a = a, .d = d, .dev = dev, .plan = plan, .extra = try cuda.DeviceBuffer.alloc(d, @max(extra_bytes, 256)) };
    errdefer b.extra.?.free();
    const L = dev.layout;
    const layers = try a.alloc(gm.Layer, t.layers);
    for (layers, 0..) |*y, i| {
        const li: u32 = @intCast(i);
        const s = L.layers[i];
        y.dims = dims(t, li);
        y.attn_hc = b.hc(s.attn_hc);
        y.ffn_hc = b.hc(s.ffn_hc);
        y.input_norm = try b.w(s.input_norm);
        y.post_attention_norm = try b.w(s.post_attention_norm);
        y.attn = .{ .q_a = try b.lin(s.q_a), .q_a_norm = try b.w(s.q_a_norm), .q_b = try b.lin(s.q_b), .kv_a = try b.lin(s.kv_a), .kv_a_norm = try b.w(s.kv_a_norm), .kv_b = try b.w(s.kv_b), .gate = try b.lin(s.gate), .o = try b.lin(s.o), .post_norm = try b.w(s.post_proj_norm), .sinks = dev.ptr(s.sinks) };
        if (t.attn[i] == .indexed) {
            const ix = s.indexer; // query, token key, head weights, key/value, key gate, token key norm, key norm, ape
            y.attn.index = .{ .query = try b.lin(ix[0]), .token_key = try b.lin(ix[1]), .head_weight = try b.lin(ix[2]), .key_value = dev.ptr(ix[3]), .key_gate = dev.ptr(ix[4]), .token_key_norm = try b.w(ix[5]), .key_norm = dev.ptr(ix[6]), .ape = dev.ptr(ix[7]) };
            for ([_]u32{ ix[3], ix[4], ix[6], ix[7] }) |f| if (try dtOf(L.slots[f].dtype) != .f32) return error.Hy4Dtype;
        }
        switch (t.mlp[i]) {
            .dense => y.mlp = .{ .dense = .{ .gate = try b.lin(s.mlp_gate), .up = try b.lin(s.mlp_up), .down = try b.lin(s.mlp_down), .norm = try b.w(s.post_mlp_norm) } },
            .sparse => {
                const ex = try a.alloc([3]gm.Lin, t.experts);
                for (ex, 0..) |*e, k| e.* = .{ try b.lin(s.experts + 3 * @as(u32, @intCast(k))), try b.lin(s.experts + 3 * @as(u32, @intCast(k)) + 1), try b.lin(s.experts + 3 * @as(u32, @intCast(k)) + 2) };
                if (try dtOf(L.slots[s.router_bias].dtype) != .f32) return error.Hy4Dtype;
                y.mlp = .{ .moe = .{ .experts = t.experts, .top = t.topk, .latent = t.latent, .width = t.expert_width, .scale = t.routed_scale, .clamp = t.swiglu_limit, .router = try b.lin(s.router), .bias = dev.ptr(s.router_bias), .pre = try b.lin(s.to_latent), .post = try b.lin(s.from_latent), .expert = ex, .shared = .{ try b.lin(s.shared[0]), try b.lin(s.shared[1]), try b.lin(s.shared[2]) }, .norm = try b.w(s.combine_norm) } };
            },
        }
        y.engram = null;
        if (s.engram) {
            const p = li;
            var hn: [config.max_hash_heads]u32 = undefined;
            for (0..heads_e) |h| hn[h] = try b.named("model.layers.{d}.engram_layer.lookup_per_head_norms.{d}.weight", .{ p, h });
            var nk: [8]u32 = undefined;
            var nq: [8]u32 = undefined;
            var sc: [8]u32 = undefined;
            const keys = try a.alloc(gm.Lin, n);
            for (0..n) |si| {
                nk[si] = try b.named("model.layers.{d}.engram_layer.norm_key.{d}.weight", .{ p, si });
                nq[si] = try b.named("model.layers.{d}.engram_layer.norm_query.{d}.weight", .{ p, si });
                sc[si] = try b.named("model.layers.{d}.engram_layer.unified_per_hc_scales.{d}", .{ p, si });
                keys[si] = try b.linBias(try b.named("model.layers.{d}.engram_layer.key_projs.{d}.weight", .{ p, si }), try b.named("model.layers.{d}.engram_layer.key_projs.{d}.bias", .{ p, si }));
            }
            const e: gm.Engram = .{
                .heads = heads_e,
                .head_dim = t.engramHeadDim(),
                .head_norms = try b.joined(hn[0..heads_e]),
                .value = try b.linBias(try b.named("model.layers.{d}.engram_layer.value_proj.weight", .{p}), try b.named("model.layers.{d}.engram_layer.value_proj.bias", .{p})),
                .keys = keys,
                .norm_key = try b.joined(nk[0..n]),
                .norm_query = try b.joined(nq[0..n]),
                .pre_gamma = try b.w(try b.named("model.layers.{d}.engram_layer.unified_pre_gamma", .{p})),
                .scales = try b.joined(sc[0..n]),
            };
            if (e.norm_query.dt != e.scales.dt) return error.Hy4Dtype;
            y.engram = e;
        }
    }
    if (try dtOf(L.slots[L.embed].dtype) != .bf16) return error.Hy4Dtype;
    const head: gm.Head = .{ .embed = dev.ptr(L.embed), .hc = b.hc(L.head_hc), .norm = try b.w(L.norm), .lm_head = try b.lin(L.lm_head) };
    var tower: ?gv.Tower = null;
    if (vision) {
        const v = if (c.vision) |*vv| vv else return error.Hy4NoVision;
        const blocks = try a.alloc(gv.Block, v.layers);
        for (blocks, 0..) |*bk, i| {
            const pre = "vit.layers.{d}.";
            bk.* = .{
                .norm1_w = try b.w(try b.named(pre ++ "input_layernorm.weight", .{i})),
                .norm1_b = try b.w(try b.named(pre ++ "input_layernorm.bias", .{i})),
                .q = try b.proj(try b.named(pre ++ "self_attn.q_proj.weight", .{i}), try b.named(pre ++ "self_attn.q_proj.bias", .{i})),
                .k = try b.proj(try b.named(pre ++ "self_attn.k_proj.weight", .{i}), try b.named(pre ++ "self_attn.k_proj.bias", .{i})),
                .v = try b.proj(try b.named(pre ++ "self_attn.v_proj.weight", .{i}), try b.named(pre ++ "self_attn.v_proj.bias", .{i})),
                .o = try b.proj(try b.named(pre ++ "self_attn.o_proj.weight", .{i}), try b.named(pre ++ "self_attn.o_proj.bias", .{i})),
                .norm2_w = try b.w(try b.named(pre ++ "post_attention_layernorm.weight", .{i})),
                .norm2_b = try b.w(try b.named(pre ++ "post_attention_layernorm.bias", .{i})),
                .fc1 = try b.proj(try b.named(pre ++ "mlp.dense_h_to_4h.weight", .{i}), try b.named(pre ++ "mlp.dense_h_to_4h.bias", .{i})),
                .fc2 = try b.proj(try b.named(pre ++ "mlp.dense_4h_to_h.weight", .{i}), try b.named(pre ++ "mlp.dense_4h_to_h.bias", .{i})),
            };
        }
        const grid = try main.f32s(a, "vit.embeddings.position_embedding.weight");
        const edge = v.max_image / v.patch;
        if (grid.len != (@as(usize, edge) * edge + 1) * v.hidden) return error.Hy4Shape;
        tower = .{
            .hidden = v.hidden,
            .inter = v.width,
            .heads = v.heads,
            .patch = v.patch,
            .merge = v.merge,
            .out = v.out_hidden,
            .edge = edge,
            .eps = v.eps,
            .patch_proj = try b.proj(try b.slotOf("vit.embeddings.patch_embedding.weight"), try b.slotOf("vit.embeddings.patch_embedding.bias")),
            .positions = grid[v.hidden..],
            .blocks = blocks,
            .merge_norm = try b.w(try b.slotOf("vit.perceive.norm.weight")),
            .conv = try b.proj(try b.slotOf("vit.perceive.proj.0.weight"), try b.slotOf("vit.perceive.proj.0.bias")),
            .out_proj = try b.proj(try b.slotOf("vit.perceive.proj.2.weight"), try b.slotOf("vit.perceive.proj.2.bias")),
            .newline = try b.w(try b.slotOf("vit.perceive.image_newline")),
        };
    }
    return .{ .layers = layers, .head = head, .tower = tower, .extra = b.extra.? };
}

/// One Engram layer's host side: the hash and the sidecar table's rows (FP8 E4M3 values, one bf16 scale a row).
pub const EngramHost = struct {
    layer: u32,
    hasher: engram_hash.Hasher,
    offsets: []u64,
    file: std.Io.File,
    values_at: u64,
    scales_at: u64,
    dim: u32,
    rows: u64,
};

/// float8_e4m3fn's 256 values.
pub fn e4m3Table() [256]f32 {
    var out: [256]f32 = undefined;
    for (&out, 0..) |*o, i| {
        const b: u8 = @intCast(i);
        const sign: f32 = if (b & 0x80 != 0) -1 else 1;
        const e: i32 = (b >> 3) & 0xf;
        const m: f32 = @floatFromInt(b & 7);
        o.* = if (e == 15 and (b & 7) == 7) std.math.nan(f32) else if (e == 0) sign * m / 8.0 * std.math.ldexp(@as(f32, 1), -6) else sign * (1 + m / 8.0) * std.math.ldexp(@as(f32, 1), e - 7);
    }
    return out;
}

test "Hy4 bind: the E4M3 table decodes the format's corners" {
    const t = e4m3Table();
    try std.testing.expectEqual(@as(f32, 0), t[0]);
    try std.testing.expectEqual(@as(f32, 448), t[0x7e]);
    try std.testing.expectEqual(@as(f32, -448), t[0xfe]);
    try std.testing.expectEqual(@as(f32, 1), t[0x38]);
    try std.testing.expectEqual(std.math.ldexp(@as(f32, 1), -9), t[1]);
    try std.testing.expect(std.math.isNan(t[0x7f]));
}
