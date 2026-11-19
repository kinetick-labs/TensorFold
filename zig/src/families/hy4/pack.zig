//! The SAGE REXL3 pack's tensor names (Tencent's checkpoint names, not transformers' renames) checked against what
//! the config needs, from `model.safetensors.index.json` and the Engram sidecar's `engram.safetensors.index.json`.
//! A projection is either EXL3 (`.trellis`, `.suh`, `.svh`, `.mul1`) or a plain `.weight`; every other tensor is plain.
//! The check refuses a pack with a missing tensor, half an EXL3 projection or a name it does not know, before any
//! weight is read.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Config = @import("config.zig");

pub const exl3_parts = [_][]const u8{ "trellis", "suh", "svh", "mul1" };

pub const Need = enum { tensor, projection };

/// One name the config needs: a tensor, or a projection's base name.
pub const Expect = struct { name: []const u8, need: Need };

/// transformers' names (PR #50903 conversion mapping) for the pack's, segment by segment; others are unchanged.
pub const renames = [_][2][]const u8{
    .{ "learnable_sink_param", "sinks" },
    .{ "linear_gate", "gate_proj" },
    .{ "post_fc2_norm", "post_mlp_layernorm" },
    .{ "routed_expert_down_proj", "pre_expert_proj" },
    .{ "routed_expert_up_proj", "post_expert_proj" },
    .{ "hc_attn_layer.hc_pre.hc_", "attn_hc." },
    .{ "hc_mlp_layer.hc_pre.hc_", "ffn_hc." },
};

/// The transformers name of a pack tensor (caller frees).
pub fn canonical(a: Allocator, name: []const u8) Allocator.Error![]u8 {
    var out = try a.dupe(u8, name);
    for (renames) |r| {
        if (std.mem.indexOf(u8, out, r[0]) == null) continue;
        const n = std.mem.replacementSize(u8, out, r[0], r[1]);
        const next = try a.alloc(u8, n);
        _ = std.mem.replace(u8, out, r[0], r[1], next);
        a.free(out);
        out = next;
    }
    return out;
}

const Names = struct {
    a: Allocator,
    list: std.ArrayList(Expect) = .empty,

    fn add(n: *Names, need: Need, comptime fmt: []const u8, args: anytype) !void {
        try n.list.append(n.a, .{ .name = try std.fmt.allocPrint(n.a, fmt, args), .need = need });
    }
};

/// Every name of the GPU pack the config needs (arena-allocated).
pub fn expected(a: Allocator, c: *const Config.Config) ![]Expect {
    var n: Names = .{ .a = a };
    const t = &c.text;
    try n.add(.tensor, "model.embed_tokens.weight", .{});
    try n.add(.tensor, "model.norm.weight", .{});
    for ([_][]const u8{ "base", "fn", "scale" }) |p| try n.add(.tensor, "model.hc_head.hc_head_{s}", .{p});
    try n.add(.projection, "lm_head", .{});
    for (0..t.layers) |l| {
        const p = try std.fmt.allocPrint(a, "model.layers.{d}.", .{l});
        for ([_][]const u8{ "input_layernorm.weight", "post_attention_layernorm.weight", "self_attn.kv_a_layernorm.weight", "self_attn.kv_a_proj_with_mqa.weight", "self_attn.kv_b_proj.weight", "self_attn.learnable_sink_param", "self_attn.post_proj_norm.weight", "self_attn.q_a_layernorm.weight", "self_attn.q_a_proj.weight" }) |s| try n.add(.tensor, "{s}{s}", .{ p, s });
        for ([_][]const u8{ "attn", "mlp" }) |h| for ([_][]const u8{ "base", "fn", "scale" }) |s| try n.add(.tensor, "{s}hc_{s}_layer.hc_pre.hc_{s}", .{ p, h, s });
        for ([_][]const u8{ "q_b_proj", "o_proj", "linear_gate" }) |s| try n.add(.projection, "{s}self_attn.{s}", .{ p, s });
        if (t.attn[l] == .indexed) for ([_][]const u8{ "head_weight_projection.weight", "intra_group_ape", "key_gate_projection.weight", "key_norm.weight", "key_value_projection.weight", "query_projection.weight", "token_key_norm.weight", "token_key_projection.weight" }) |s| try n.add(.tensor, "{s}self_attn.compressed_indexer.{s}", .{ p, s });
        switch (t.mlp[l]) {
            .dense => {
                for ([_][]const u8{ "gate_proj", "up_proj", "down_proj" }) |s| try n.add(.projection, "{s}mlp.{s}", .{ p, s });
                try n.add(.tensor, "{s}mlp.post_fc2_norm.weight", .{p});
            },
            .sparse => {
                for ([_][]const u8{ "combine_output_norm.weight", "gate.weight", "gate.e_score_correction_bias", "routed_expert_down_proj.weight", "routed_expert_up_proj.weight" }) |s| try n.add(.tensor, "{s}mlp.{s}", .{ p, s });
                for ([_][]const u8{ "gate_proj", "up_proj", "down_proj" }) |s| try n.add(.projection, "{s}mlp.shared_experts.{s}", .{ p, s });
                for (0..t.experts) |e| for ([_][]const u8{ "gate_proj", "up_proj", "down_proj" }) |s| try n.add(.projection, "{s}mlp.experts.{d}.{s}", .{ p, e, s });
            },
        }
        if (t.engramAt(@intCast(l)) != null) {
            const e = try std.fmt.allocPrint(a, "{s}engram_layer.", .{p});
            for (0..t.hc_mult) |h| {
                try n.add(.tensor, "{s}key_projs.{d}.weight", .{ e, h });
                try n.add(.tensor, "{s}key_projs.{d}.bias", .{ e, h });
                try n.add(.tensor, "{s}norm_key.{d}.weight", .{ e, h });
                try n.add(.tensor, "{s}norm_query.{d}.weight", .{ e, h });
                try n.add(.tensor, "{s}unified_per_hc_scales.{d}", .{ e, h });
            }
            for (0..t.hashHeads()) |h| try n.add(.tensor, "{s}lookup_per_head_norms.{d}.weight", .{ e, h });
            for ([_][]const u8{ "lookup_table", "unified_pre_gamma", "value_proj.weight", "value_proj.bias" }) |s| try n.add(.tensor, "{s}{s}", .{ e, s });
        }
    }
    if (c.draft) |d| {
        for (0..d.layers) |l| {
            const p = try std.fmt.allocPrint(a, "model.dfly_layers.{d}.", .{l});
            for ([_][]const u8{ "input_layernorm.weight", "post_attention_layernorm.weight", "self_attn.kv_a_layernorm.weight", "self_attn.kv_a_proj_with_mqa.weight", "self_attn.kv_b_proj.weight", "self_attn.learnable_sink_param", "self_attn.post_proj_norm.weight", "self_attn.q_a_layernorm.weight", "self_attn.q_a_proj.weight" }) |s| try n.add(.tensor, "{s}{s}", .{ p, s });
            for ([_][]const u8{ "self_attn.q_b_proj", "self_attn.o_proj", "self_attn.linear_gate", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj" }) |s| try n.add(.projection, "{s}{s}", .{ p, s });
        }
        for ([_][]const u8{ "confidence_head.proj.bias", "confidence_head.proj.weight", "final_norm.weight", "mtp_fusion.eh_proj.weight", "mtp_fusion.enorm.weight", "mtp_fusion.gate", "mtp_fusion.hnorm.weight" }) |s| try n.add(.tensor, "model.dfly_layers.{s}", .{s});
        if (d.hidden_correction) for ([_][]const u8{ "down_proj.weight", "embed_norm.weight", "gate_proj.weight", "hidden_norm.weight", "up_proj.weight" }) |s| try n.add(.tensor, "model.dfly_layers.hidden_correction.{s}", .{s});
    }
    if (c.vision) |v| {
        for ([_][]const u8{ "embeddings.patch_embedding.bias", "embeddings.patch_embedding.weight", "embeddings.position_embedding.weight", "perceive.image_newline", "perceive.norm.weight", "perceive.proj.0.bias", "perceive.proj.0.weight", "perceive.proj.2.bias", "perceive.proj.2.weight" }) |s| try n.add(.tensor, "vit.{s}", .{s});
        for (0..v.layers) |l| for ([_][]const u8{ "input_layernorm", "post_attention_layernorm", "mlp.dense_4h_to_h", "mlp.dense_h_to_4h", "self_attn.k_proj", "self_attn.o_proj", "self_attn.q_proj", "self_attn.v_proj" }) |s| {
            try n.add(.tensor, "vit.layers.{d}.{s}.weight", .{ l, s });
            try n.add(.tensor, "vit.layers.{d}.{s}.bias", .{ l, s });
        };
    }
    return n.list.items;
}

/// The Engram sidecar's names: each Engram layer's FP8 table and its per-row scales.
pub fn expectedSidecar(a: Allocator, c: *const Config.Config) ![][]const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    for (c.text.engramLayers()) |e| {
        const layer = e.id - 1;
        try out.append(a, try std.fmt.allocPrint(a, "model.layers.{d}.engram_layer.multi_head_embedding.embedding.weight", .{layer}));
        try out.append(a, try std.fmt.allocPrint(a, "model.layers.{d}.engram_layer.multi_head_embedding.embedding.weight.scale_bf16", .{layer}));
    }
    return out.items;
}

pub const Report = struct {
    tensors: usize = 0,
    exl3: usize = 0,
    plain_projections: usize = 0,
    shards: usize = 0,
    total_size: u64 = 0,
    missing: []const []const u8 = &.{},
    unknown: []const []const u8 = &.{},
    partial: []const []const u8 = &.{},

    pub fn ok(r: *const Report) bool {
        return r.missing.len == 0 and r.unknown.len == 0 and r.partial.len == 0;
    }
};

/// An index's `weight_map` against `want`; `report.missing`, `.unknown` and `.partial` list what is wrong (arena).
pub fn check(a: Allocator, index_json: []const u8, want: []const Expect, sidecar: []const []const u8) !Report {
    const parsed = try std.json.parseFromSliceLeaky(std.json.Value, a, index_json, .{});
    const map = (parsed.object.get("weight_map") orelse return error.BadIndex).object;
    var r: Report = .{};
    if (parsed.object.get("metadata")) |m| if (m.object.get("total_size")) |s| {
        r.total_size = @intCast(s.integer);
    };
    var seen: std.StringHashMapUnmanaged(void) = .empty;
    var shards: std.StringHashMapUnmanaged(void) = .empty;
    var missing: std.ArrayList([]const u8) = .empty;
    var partial: std.ArrayList([]const u8) = .empty;
    var unknown: std.ArrayList([]const u8) = .empty;
    var it = map.iterator();
    while (it.next()) |e| try shards.put(a, e.value_ptr.string, {});
    r.shards = shards.count();
    for (want) |w| switch (w.need) {
        .tensor => {
            if (map.contains(w.name)) {
                try seen.put(a, w.name, {});
                r.tensors += 1;
            } else try missing.append(a, w.name);
        },
        .projection => {
            const plain = try std.fmt.allocPrint(a, "{s}.weight", .{w.name});
            if (map.contains(plain)) {
                try seen.put(a, plain, {});
                r.plain_projections += 1;
                continue;
            }
            var have: usize = 0;
            for (exl3_parts) |part| {
                const name = try std.fmt.allocPrint(a, "{s}.{s}", .{ w.name, part });
                if (map.contains(name)) {
                    have += 1;
                    try seen.put(a, name, {});
                }
            }
            if (have == exl3_parts.len) r.exl3 += 1 else if (have == 0) try missing.append(a, w.name) else try partial.append(a, w.name);
        },
    };
    for (sidecar) |s| if (map.contains(s)) try seen.put(a, s, {});
    it = map.iterator();
    while (it.next()) |e| if (!seen.contains(e.key_ptr.*)) try unknown.append(a, e.key_ptr.*);
    r.missing = missing.items;
    r.partial = partial.items;
    r.unknown = unknown.items;
    return r;
}
