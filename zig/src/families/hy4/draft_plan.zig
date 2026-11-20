//! The DFly2 drafter's tensors (`model.dfly_layers.*`, group `draft` of the load plan) and the shapes the engine will
//! bind them with, checked against a pack's safetensors headers before anything is uploaded. The drafter follows
//! Tencent AngelSpec's published DFly2 / DSpark / DFlash design (Apache-2.0; models/draft/dfly2.py, dspark.py,
//! dflash.py, hy4_mla.py), whose module names the pack keeps:
//!
//! - five decoder layers, each a gated MLA (LoRA q, kv_a + kv_a_layernorm, kv_b, elementwise sigmoid gate over the
//!   heads' values, per-head learnable sink, post-projection norm) and a dense SwiGLU MLP (EXL3 projections);
//! - draft layer i attends over the target's latent cache of layer `dfly_target_kv_layers[i]` (its normed latents and
//!   roped keys as cached), kept to the target indexer's top-k there, plus the block's own latents;
//! - `mtp_fusion` (slot0): eh_proj(enorm(E(anchor)) :: hnorm(target final hidden)) through `gate`, added to the anchor
//!   slot of the block input; `hidden_correction`: h + SwiGLU(hidden_norm(h) :: embed_norm(E(previous token))) before
//!   the target's LM head; `confidence_head`: one logit a slot (accept probability, for an adaptive block length);
//!   `final_norm` after the last layer. The block is the anchor then `dfly_mask_token_id` slots.

const std = @import("std");
const config = @import("config.zig");

pub const Kind = enum { plain, exl3, vector };

/// One tensor: `plain` [n, k] weights, `exl3` projections of k inputs and n outputs (suh [k], svh [n]), `vector` [n].
pub const Want = struct { name: []const u8, kind: Kind, k: u64 = 0, n: u64 };

pub const Mismatch = struct { name: []const u8, why: []const u8 };

/// Every tensor the engine binds, in a stable order; `a` owns the names.
pub fn wanted(a: std.mem.Allocator, t: *const config.Text, d: *const config.Draft) ![]Want {
    var list: std.ArrayList(Want) = .empty;
    const H: u64 = t.hidden;
    if (d.targets_len < d.layers) return error.Hy4DraftConfig;
    for (0..d.layers) |i| {
        const tl = d.targets[i];
        if (tl >= t.layers or t.attn[tl] != .indexed) return error.Hy4DraftConfig;
        const hd = t.indexed;
        const heads: u64 = hd.heads;
        const p = struct {
            fn name(al: std.mem.Allocator, li: usize, comptime rest: []const u8) ![]const u8 {
                return std.fmt.allocPrint(al, "model.dfly_layers.{d}." ++ rest, .{li});
            }
        };
        try list.appendSlice(a, &.{
            .{ .name = try p.name(a, i, "input_layernorm.weight"), .kind = .vector, .n = H },
            .{ .name = try p.name(a, i, "post_attention_layernorm.weight"), .kind = .vector, .n = H },
            .{ .name = try p.name(a, i, "self_attn.q_a_proj.weight"), .kind = .plain, .k = H, .n = t.q_lora },
            .{ .name = try p.name(a, i, "self_attn.q_a_layernorm.weight"), .kind = .vector, .n = t.q_lora },
            .{ .name = try p.name(a, i, "self_attn.q_b_proj"), .kind = .exl3, .k = t.q_lora, .n = heads * (hd.qk_nope + hd.qk_rope) },
            .{ .name = try p.name(a, i, "self_attn.kv_a_proj_with_mqa.weight"), .kind = .plain, .k = H, .n = t.kv_lora + hd.qk_rope },
            .{ .name = try p.name(a, i, "self_attn.kv_a_layernorm.weight"), .kind = .vector, .n = t.kv_lora },
            .{ .name = try p.name(a, i, "self_attn.kv_b_proj.weight"), .kind = .plain, .k = t.kv_lora, .n = heads * (hd.qk_nope + hd.v) },
            .{ .name = try p.name(a, i, "self_attn.linear_gate"), .kind = .exl3, .k = H, .n = heads * hd.v },
            .{ .name = try p.name(a, i, "self_attn.o_proj"), .kind = .exl3, .k = heads * hd.v, .n = H },
            .{ .name = try p.name(a, i, "self_attn.post_proj_norm.weight"), .kind = .vector, .n = H },
            .{ .name = try p.name(a, i, "self_attn.learnable_sink_param"), .kind = .vector, .n = heads },
            .{ .name = try p.name(a, i, "mlp.gate_proj"), .kind = .exl3, .k = H, .n = 0 },
            .{ .name = try p.name(a, i, "mlp.up_proj"), .kind = .exl3, .k = H, .n = 0 },
            .{ .name = try p.name(a, i, "mlp.down_proj"), .kind = .exl3, .k = 0, .n = H },
        });
    }
    try list.appendSlice(a, &.{
        .{ .name = "model.dfly_layers.final_norm.weight", .kind = .vector, .n = H },
        .{ .name = "model.dfly_layers.mtp_fusion.eh_proj.weight", .kind = .plain, .k = 2 * H, .n = H },
        .{ .name = "model.dfly_layers.mtp_fusion.enorm.weight", .kind = .vector, .n = H },
        .{ .name = "model.dfly_layers.mtp_fusion.hnorm.weight", .kind = .vector, .n = H },
        .{ .name = "model.dfly_layers.mtp_fusion.gate", .kind = .vector, .n = H },
        .{ .name = "model.dfly_layers.confidence_head.proj.weight", .kind = .plain, .k = H, .n = 1 },
        .{ .name = "model.dfly_layers.confidence_head.proj.bias", .kind = .vector, .n = 1 },
    });
    if (d.hidden_correction) try list.appendSlice(a, &.{
        .{ .name = "model.dfly_layers.hidden_correction.hidden_norm.weight", .kind = .vector, .n = H },
        .{ .name = "model.dfly_layers.hidden_correction.embed_norm.weight", .kind = .vector, .n = H },
        .{ .name = "model.dfly_layers.hidden_correction.gate_proj.weight", .kind = .plain, .k = 2 * H, .n = H },
        .{ .name = "model.dfly_layers.hidden_correction.up_proj.weight", .kind = .plain, .k = 2 * H, .n = H },
        .{ .name = "model.dfly_layers.hidden_correction.down_proj.weight", .kind = .plain, .k = H, .n = H },
    });
    return list.toOwnedSlice(a);
}

/// The headers' shape of `name` (safetensors `[dtype, shape]` pairs as `[name] -> {shape}` lookups).
pub const Headers = struct {
    ctx: *const anyopaque,
    shape: *const fn (*const anyopaque, []const u8) ?[]const u64,
};

/// The first wanted tensor the headers miss or shape otherwise; 0 widths (the MLP's intermediate) are taken from the
/// pack and must agree between gate, up and down. The correction's intermediate is the hidden width (AngelSpec).
pub fn check(ws: []const Want, h: Headers) ?Mismatch {
    var mlp_width: u64 = 0;
    for (ws) |w| {
        switch (w.kind) {
            .vector => {
                const sh = h.shape(h.ctx, w.name) orelse return .{ .name = w.name, .why = "missing" };
                var n: u64 = 1;
                for (sh) |x| n *= x;
                if (n != w.n) return .{ .name = w.name, .why = "length" };
            },
            .plain => {
                const sh = h.shape(h.ctx, w.name) orelse return .{ .name = w.name, .why = "missing" };
                if (sh.len != 2) return .{ .name = w.name, .why = "rank" };
                if (!widthOk(&mlp_width, w.n, sh[0]) or !widthOk(&mlp_width, w.k, sh[1])) return .{ .name = w.name, .why = "shape" };
            },
            .exl3 => {
                var buf: [256]u8 = undefined;
                const suh = h.shape(h.ctx, std.fmt.bufPrint(&buf, "{s}.suh", .{w.name}) catch return .{ .name = w.name, .why = "name" }) orelse return .{ .name = w.name, .why = "missing suh" };
                const svh = h.shape(h.ctx, std.fmt.bufPrint(&buf, "{s}.svh", .{w.name}) catch return .{ .name = w.name, .why = "name" }) orelse return .{ .name = w.name, .why = "missing svh" };
                _ = h.shape(h.ctx, std.fmt.bufPrint(&buf, "{s}.trellis", .{w.name}) catch return .{ .name = w.name, .why = "name" }) orelse return .{ .name = w.name, .why = "missing trellis" };
                if (suh.len != 1 or svh.len != 1) return .{ .name = w.name, .why = "rank" };
                if (!widthOk(&mlp_width, w.k, suh[0]) or !widthOk(&mlp_width, w.n, svh[0])) return .{ .name = w.name, .why = "shape" };
            },
        }
    }
    return null;
}

/// `want` 0: the one free width (the MLP's), learned from the first and held to after.
fn widthOk(free: *u64, want: u64, got: u64) bool {
    if (want != 0) return want == got;
    if (free.* == 0) free.* = got;
    return free.* == got or free.* == 0;
}

const JsonHeaders = struct {
    obj: std.json.ObjectMap,
    a: std.mem.Allocator,
    fn shape(ctx: *const anyopaque, name: []const u8) ?[]const u64 {
        const self: *const JsonHeaders = @ptrCast(@alignCast(ctx));
        const v = self.obj.get(name) orelse return null;
        if (v != .array or v.array.items.len != 2 or v.array.items[1] != .array) return null;
        const dims = v.array.items[1].array.items;
        const out = self.a.alloc(u64, dims.len) catch return null;
        for (dims, out) |dv, *o| o.* = switch (dv) {
            .integer => |x| @intCast(x),
            else => return null,
        };
        return out;
    }
};

test "draft plan: the pack's DFly2 tensors (fixtures/dfly_headers.json) have the shapes the engine binds" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try config.Config.parse(a, @embedFile("fixtures/config.json"));
    const d = c.draft orelse return error.TestUnexpectedResult;
    const ws = try wanted(a, &c.text, &d);
    try std.testing.expectEqual(@as(usize, 5 * 15 + 7 + 5), ws.len);
    const parsed = try std.json.parseFromSlice(std.json.Value, a, @embedFile("fixtures/dfly_headers.json"), .{});
    var jh: JsonHeaders = .{ .obj = parsed.value.object, .a = a };
    const h: Headers = .{ .ctx = &jh, .shape = JsonHeaders.shape };
    if (check(ws, h)) |m| {
        std.debug.print("draft plan mismatch: {s} ({s})\n", .{ m.name, m.why });
        return error.TestUnexpectedResult;
    }
    // every pack tensor is bound (EXL3 groups count their trellis / suh / svh / mul1)
    var bound: usize = 0;
    for (ws) |w| bound += if (w.kind == .exl3) 4 else 1;
    try std.testing.expectEqual(parsed.value.object.count(), bound);
    // a wrong width is caught
    var bad = try a.dupe(Want, ws);
    bad[4].n += 1;
    try std.testing.expect(check(bad, h) != null);
}
