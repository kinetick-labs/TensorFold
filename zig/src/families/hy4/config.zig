//! Hy4-flash (`hy_v4_flash`, tencent/Hy4-flash) config.json: the text backbone (text_config), the vision tower
//! (vision_config), DFly2's drafter fields and the media token ids, read as transformers' HYV4FlashConfig reads them
//! (PR huggingface/transformers#50903): the published `swa_*` and `compressed_indexer_*` aliases, the legacy
//! `deepseek_sparse_attention` layer name and the per-layer-type RoPE. Absent keys keep the release's values.

const std = @import("std");

pub const max_layers = 64;
pub const max_engram_layers = 4;
/// Hash heads a layer may hold: (max n-gram order - 1) x heads per order.
pub const max_hash_heads = 32;

/// `layer_types`: an indexed (compressed-indexer sparse) layer or a sliding-window one.
pub const Attn = enum(u8) { indexed, sliding };
/// `mlp_layer_types`: a dense SwiGLU or a LatentMoE layer.
pub const Mlp = enum(u8) { dense, sparse };

/// One attention layer type's head geometry and RoPE (MLA: one latent, expanded per head).
pub const Heads = struct {
    heads: u32,
    qk_nope: u32,
    qk_rope: u32,
    v: u32,
    theta: f64,
    /// Interleaved M-RoPE sections (time, height, width), in rotary pairs; they sum to qk_rope / 2.
    mrope: [3]u32 = .{ 12, 10, 10 },

    pub fn qkDim(h: Heads) u32 {
        return h.qk_nope + h.qk_rope;
    }
};

pub const EngramLayer = struct {
    /// The 1-based id the config and the hash seed use (`layer_idx + 1`); the checkpoint stores the layer's tensors
    /// under `model.layers.{id - 1}.engram_layer`.
    id: u32,
    /// Hash moduli, n-gram-major: order 2's heads, then order 3's.
    sizes: [max_hash_heads]u64 = @splat(0),
};

pub const Text = struct {
    vocab: u32 = 120832,
    hidden: u32 = 3072,
    layers: u32 = 42,
    dense_width: u32 = 13824,
    q_lora: u32 = 1024,
    kv_lora: u32 = 512,
    indexed: Heads = .{ .heads = 32, .qk_nope = 192, .qk_rope = 64, .v = 256, .theta = 10000000.0 },
    sliding: Heads = .{ .heads = 64, .qk_nope = 192, .qk_rope = 64, .v = 256, .theta = 10000.0 },
    window: u32 = 512,
    experts: u32 = 512,
    shared_experts: u32 = 1,
    topk: u32 = 16,
    expert_width: u32 = 1536,
    /// LatentMoE: routed experts read and write this width, through `routed_expert_up_proj` / `_down_proj`.
    latent: u32 = 1536,
    routed_scale: f32 = 3.998,
    norm_topk: bool = true,
    swiglu_limit: f32 = 10.0,
    eps: f32 = 1e-5,
    index_heads: u32 = 16,
    index_dim: u32 = 128,
    index_topk: u32 = 2048,
    /// Tokens pooled into one compressed indexer key (`index_kpool`, published `compressed_indexer_compress_ratio`).
    index_pool: u32 = 4,
    hc_mult: u32 = 4,
    hc_eps: f32 = 1e-6,
    hc_magnitude: f32 = 2.0,
    engram_max_ngram: u32 = 3,
    engram_heads: u32 = 8,
    engram_width: u32 = 768,
    engram_pad: u32 = 120002,
    engram_seed: u64 = 0,
    engram_fp8: bool = true,
    engram_map_size: u32 = 120818,
    engram: [max_engram_layers]EngramLayer = undefined,
    engram_len: u32 = 0,
    attn: [max_layers]Attn = undefined,
    mlp: [max_layers]Mlp = undefined,
    bos: u32 = 120000,
    eos: u32 = 120025,
    pad: u32 = 120002,

    pub fn layerHeads(t: *const Text, layer: u32) Heads {
        return switch (t.attn[layer]) {
            .indexed => t.indexed,
            .sliding => t.sliding,
        };
    }

    pub fn engramLayers(t: *const Text) []const EngramLayer {
        return t.engram[0..t.engram_len];
    }

    /// The Engram table of decoder layer `layer` (0-based), if it has one.
    pub fn engramAt(t: *const Text, layer: u32) ?*const EngramLayer {
        for (t.engram[0..t.engram_len]) |*e| if (e.id == layer + 1) return e;
        return null;
    }

    pub fn hashHeads(t: *const Text) u32 {
        return (t.engram_max_ngram - 1) * t.engram_heads;
    }

    pub fn engramHeadDim(t: *const Text) u32 {
        return t.engram_width / t.engram_heads;
    }
};

pub const Vision = struct {
    layers: u32 = 27,
    hidden: u32 = 1152,
    width: u32 = 4304,
    heads: u32 = 16,
    patch: u32 = 16,
    merge: u32 = 2,
    channels: u32 = 3,
    out_hidden: u32 = 3072,
    /// The learned position grid's edge: max_image_size / patch (2048 / 16 = 128).
    max_image: u32 = 2048,
    min_image: u32 = 512,
    eps: f32 = 1e-5,
    gelu_tanh: bool = true,
};

pub const Draft = struct {
    layers: u32 = 5,
    tokens: u32 = 5,
    mask_token: u32 = 120009,
    index_topk: u32 = 2048,
    hidden_correction: bool = true,
    targets: [8]u32 = @splat(0),
    targets_len: u32 = 0,
};

pub const Config = struct {
    text: Text = .{},
    vision: ?Vision = .{},
    draft: ?Draft = .{},
    image_token: u32 = 120120,
    image_start: u32 = 120118,
    image_end: u32 = 120119,
    image_newline: u32 = 120121,
    video_token: u32 = 120124,
    video_start: u32 = 120122,
    video_end: u32 = 120123,

    /// config.json in `dir`.
    pub fn read(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Config {
        const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
        defer gpa.free(path);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 24));
        defer gpa.free(bytes);
        return parse(gpa, bytes);
    }

    pub fn parse(gpa: std.mem.Allocator, bytes: []const u8) !Config {
        const parsed = std.json.parseFromSlice(std.json.Value, gpa, bytes, .{ .parse_numbers = false }) catch return error.BadConfig;
        defer parsed.deinit();
        if (parsed.value != .object) return error.BadConfig;
        const top = parsed.value.object;
        if (top.get("model_type")) |v| if (v != .string or !std.mem.eql(u8, v.string, "hy_v4_flash")) return error.UnsupportedArchitecture;
        const text = if (top.get("text_config")) |v| (if (v == .object) v.object else return error.BadConfig) else top;
        var c: Config = .{};
        try readText(&c.text, text);
        c.vision = if (top.get("vision_config")) |v| switch (v) {
            .object => |o| try readVision(o),
            .null => null,
            else => return error.BadConfig,
        } else null;
        c.draft = try readDraft(top);
        const ids = .{
            .{ "image_token", "image_token_id" },     .{ "image_start", "image_start_token_id" },
            .{ "image_end", "image_end_token_id" },   .{ "image_newline", "image_newline_token_id" },
            .{ "video_token", "video_token_id" },     .{ "video_start", "video_start_token_id" },
            .{ "video_end", "video_end_token_id" },
        };
        inline for (ids) |p| if (top.get(p[1])) |v| {
            @field(c, p[0]) = try intAs(u32, v);
        };
        try c.validate();
        return c;
    }

    /// Shapes the loader and the kernels assume; a config that breaks one is refused before any allocation.
    pub fn validate(c: *const Config) !void {
        const t = &c.text;
        if (t.layers == 0 or t.layers > max_layers or t.hidden == 0 or t.hidden % 128 != 0 or t.vocab == 0) return error.BadConfig;
        if (t.topk == 0 or t.topk > t.experts or t.expert_width % 128 != 0 or t.latent % 128 != 0) return error.BadConfig;
        if (t.index_pool == 0 or t.index_topk % t.index_pool != 0 or t.index_topk / t.index_pool < 2) return error.BadConfig;
        if (t.hc_mult == 0 or t.hc_mult > 8 or t.window == 0) return error.BadConfig;
        inline for (.{ t.eps, t.hc_eps, t.routed_scale, t.swiglu_limit, t.hc_magnitude }) |v|
            if (!std.math.isFinite(v) or v <= 0) return error.BadConfig;
        for ([_]Heads{ t.indexed, t.sliding }) |h| {
            if (h.heads == 0 or h.qk_rope == 0 or h.qk_rope % 2 != 0 or h.v == 0) return error.BadConfig;
            if (h.mrope[0] + h.mrope[1] + h.mrope[2] != h.qk_rope / 2) return error.BadConfig;
            if (!std.math.isFinite(h.theta) or h.theta <= 1) return error.BadConfig;
        }
        // one latent and one rope key per token, shared by both layer types (the KV cache's row)
        if (t.indexed.qk_rope != t.sliding.qk_rope) return error.UnsupportedCacheLayout;
        if (t.index_dim <= t.indexed.qk_rope) return error.BadConfig;
        if (t.engram_max_ngram < 2 or t.engram_heads == 0 or t.hashHeads() > max_hash_heads) return error.BadConfig;
        if (t.engram_width % t.engram_heads != 0) return error.BadConfig;
        if (t.engram_len > 0 and t.engram_pad >= t.engram_map_size) return error.BadConfig;
        for (t.engramLayers()) |e| {
            if (e.id == 0 or e.id > t.layers) return error.BadConfig;
            for (e.sizes[0..t.hashHeads()]) |s| if (s < 2) return error.BadConfig;
        }
        if (t.bos >= t.vocab or t.eos >= t.vocab or t.pad >= t.vocab or t.engram_pad >= t.vocab) return error.BadConfig;
        if (c.vision) |v| {
            if (v.hidden == 0 or v.heads == 0 or v.hidden % v.heads != 0 or v.patch == 0 or v.merge == 0) return error.BadConfig;
            if (v.max_image % v.patch != 0 or v.out_hidden != t.hidden) return error.BadConfig;
        }
        if (c.draft) |d| for (d.targets[0..d.targets_len]) |l| if (l >= t.layers or t.attn[l] != .indexed) return error.BadConfig;
        inline for (.{ c.image_token, c.image_start, c.image_end, c.image_newline, c.video_token, c.video_start, c.video_end }) |id|
            if (id >= t.vocab) return error.BadConfig;
    }
};

fn readText(t: *Text, o: std.json.ObjectMap) !void {
    if (o.get("model_type")) |v| if (v != .string or !std.mem.eql(u8, v.string, "hy_v4_flash")) return error.UnsupportedArchitecture;
    const ints = .{
        .{ "vocab", "vocab_size" },                   .{ "hidden", "hidden_size" },
        .{ "layers", "num_hidden_layers" },           .{ "dense_width", "intermediate_size" },
        .{ "q_lora", "q_lora_rank" },                 .{ "kv_lora", "kv_lora_rank" },
        .{ "window", "sliding_window" },              .{ "experts", "n_routed_experts" },
        .{ "shared_experts", "n_shared_experts" },    .{ "topk", "num_experts_per_tok" },
        .{ "expert_width", "moe_intermediate_size" }, .{ "latent", "routed_expert_hidden_size" },
        .{ "index_heads", "index_n_heads" },          .{ "index_dim", "index_head_dim" },
        .{ "index_topk", "index_topk" },              .{ "index_pool", "index_kpool" },
        .{ "index_pool", "compressed_indexer_compress_ratio" },
        .{ "hc_mult", "hc_mult" },                    .{ "engram_max_ngram", "engram_max_ngram_size" },
        .{ "engram_heads", "engram_n_head_per_ngram" }, .{ "engram_width", "engram_n_embed_per_ngram" },
        .{ "engram_pad", "engram_pad_id" },           .{ "engram_seed", "engram_seed" },
        .{ "engram_map_size", "engram_lookup_table_size" },
        .{ "bos", "bos_token_id" },                   .{ "eos", "eos_token_id" },
        .{ "pad", "pad_token_id" },
    };
    inline for (ints) |p| if (o.get(p[1])) |v| {
        @field(t, p[0]) = try intAs(@TypeOf(@field(t, p[0])), v);
    };
    const floats = .{
        .{ "routed_scale", "routed_scaling_factor" }, .{ "swiglu_limit", "swiglu_limit" }, .{ "eps", "rms_norm_eps" },
        .{ "hc_eps", "hc_eps" },                      .{ "hc_magnitude", "hc_magnitude" },
    };
    inline for (floats) |p| if (o.get(p[1])) |v| {
        @field(t, p[0]) = @floatCast(try float(v));
    };
    inline for (.{ .{ "norm_topk", "norm_topk_prob" }, .{ "engram_fp8", "engram_fp8_embedding" } }) |p| if (o.get(p[1])) |v| {
        @field(t, p[0]) = if (v == .bool) v.bool else return error.BadConfig;
    };
    // The kernels implement sigmoid scores with a bias-corrected top-k in one group, MLA, iHC and sinks.
    if (o.get("moe_router_use_sigmoid")) |v| if (v != .bool or !v.bool) return error.UnsupportedScoring;
    inline for (.{ "n_group", "topk_group" }) |k| if (o.get(k)) |v| if (try intAs(u32, v) != 1) return error.UnsupportedScoring;
    if (o.get("use_mla")) |v| if (v != .bool or !v.bool) return error.UnsupportedArchitecture;
    if (o.get("enable_ihc")) |v| if (v != .bool or !v.bool) return error.UnsupportedArchitecture;
    if (o.get("hidden_act")) |v| if (v != .string or !std.mem.eql(u8, v.string, "silu")) return error.UnsupportedArchitecture;

    t.indexed.heads = if (o.get("num_attention_heads")) |v| try intAs(u32, v) else t.indexed.heads;
    if (o.get("qk_nope_head_dim")) |v| t.indexed.qk_nope = try intAs(u32, v);
    if (o.get("qk_rope_head_dim")) |v| t.indexed.qk_rope = try intAs(u32, v);
    if (o.get("v_head_dim")) |v| t.indexed.v = try intAs(u32, v);
    // the sliding layers default to the indexed shapes, then their `swa_*` aliases (64 heads unless named)
    t.sliding.qk_nope = t.indexed.qk_nope;
    t.sliding.qk_rope = t.indexed.qk_rope;
    t.sliding.v = t.indexed.v;
    if (o.get("swa_num_attention_heads")) |v| t.sliding.heads = try intAs(u32, v);
    if (o.get("swa_qk_nope_head_dim")) |v| t.sliding.qk_nope = try intAs(u32, v);
    if (o.get("swa_qk_rope_head_dim")) |v| t.sliding.qk_rope = try intAs(u32, v);
    if (o.get("swa_v_head_dim")) |v| t.sliding.v = try intAs(u32, v);
    try readRope(t, o);

    if (t.layers == 0 or t.layers > max_layers) return error.BadConfig;
    const n = t.layers;
    if (o.get("layer_types")) |v| {
        if (v != .array or v.array.items.len != n) return error.BadConfig;
        for (v.array.items, 0..) |x, i| t.attn[i] = try attnOf(x);
    } else for (0..n) |i| {
        // three sliding layers for every indexed one, starting with two indexed layers
        t.attn[i] = if (i < 2 or (i - 1) % 4 == 0) .indexed else .sliding;
    }
    if (o.get("mlp_layer_types")) |v| {
        if (v != .array or v.array.items.len != n) return error.BadConfig;
        for (v.array.items, 0..) |x, i| {
            if (x != .string) return error.BadConfig;
            t.mlp[i] = if (std.mem.eql(u8, x.string, "dense")) .dense else if (std.mem.eql(u8, x.string, "sparse")) .sparse else return error.BadConfig;
        }
    } else for (0..n) |i| {
        t.mlp[i] = if (i == 0) .dense else .sparse;
    }
    try readEngram(t, o);
}

fn attnOf(v: std.json.Value) !Attn {
    if (v != .string) return error.BadConfig;
    const names = .{ .{ "indexed_attention", Attn.indexed }, .{ "deepseek_sparse_attention", Attn.indexed }, .{ "sliding_attention", Attn.sliding } };
    inline for (names) |p| if (std.mem.eql(u8, v.string, p[0])) return p[1];
    return error.UnsupportedArchitecture;
}

/// `rope_parameters`, keyed by layer type (the legacy `deepseek_sparse_attention` key is the indexed type).
fn readRope(t: *Text, o: std.json.ObjectMap) !void {
    const rp = o.get("rope_parameters") orelse return;
    if (rp == .null) return;
    if (rp != .object) return error.BadConfig;
    var it = rp.object.iterator();
    while (it.next()) |e| {
        const h: *Heads = switch (try attnOf(.{ .string = e.key_ptr.* })) {
            .indexed => &t.indexed,
            .sliding => &t.sliding,
        };
        if (e.value_ptr.* != .object) return error.BadConfig;
        const p = e.value_ptr.object;
        if (p.get("rope_type")) |v| if (v != .string or !std.mem.eql(u8, v.string, "default")) return error.UnsupportedRope;
        if (p.get("mrope_interleaved")) |v| if (v != .bool or !v.bool) return error.UnsupportedRope;
        if (p.get("rope_theta")) |v| h.theta = try float(v);
        if (p.get("mrope_section")) |v| {
            if (v != .array or v.array.items.len != 3) return error.BadConfig;
            for (v.array.items, 0..) |x, i| h.mrope[i] = try intAs(u32, x);
        }
    }
}

fn readEngram(t: *Text, o: std.json.ObjectMap) !void {
    const ids = o.get("engram_layer_ids") orelse return;
    if (ids == .null) return;
    if (ids != .array or ids.array.items.len > max_engram_layers) return error.BadConfig;
    const sizes = o.get("engram_layer_head_vocab_sizes") orelse return error.BadConfig;
    if (sizes != .object) return error.BadConfig;
    const orders = t.engram_max_ngram - 1;
    if (t.engram_max_ngram < 2 or orders * t.engram_heads > max_hash_heads) return error.BadConfig;
    for (ids.array.items) |x| {
        const id = try intAs(u32, x);
        var key_buf: [16]u8 = undefined;
        const key = std.fmt.bufPrint(&key_buf, "{d}", .{id}) catch unreachable;
        const per = sizes.object.get(key) orelse return error.BadConfig;
        if (per != .array or per.array.items.len != orders) return error.BadConfig;
        var e: EngramLayer = .{ .id = id };
        for (per.array.items, 0..) |order, oi| {
            if (order != .array or order.array.items.len != t.engram_heads) return error.BadConfig;
            for (order.array.items, 0..) |s, h| e.sizes[oi * t.engram_heads + h] = try intAs(u64, s);
        }
        t.engram[t.engram_len] = e;
        t.engram_len += 1;
    }
}

fn readVision(o: std.json.ObjectMap) !Vision {
    var v: Vision = .{};
    const ints = .{
        .{ "layers", "num_hidden_layers" }, .{ "hidden", "hidden_size" },         .{ "width", "intermediate_size" },
        .{ "heads", "num_attention_heads" }, .{ "patch", "patch_size" },          .{ "merge", "spatial_merge_size" },
        .{ "channels", "num_channels" },    .{ "out_hidden", "out_hidden_size" }, .{ "max_image", "max_image_size" },
        .{ "min_image", "min_image_size" },
    };
    inline for (ints) |p| if (o.get(p[1])) |x| {
        @field(v, p[0]) = try intAs(u32, x);
    };
    if (o.get("rms_norm_eps")) |x| v.eps = @floatCast(try float(x));
    if (o.get("hidden_act")) |x| {
        if (x != .string) return error.BadConfig;
        // transformers' "gelu" is the erf form; "gelu_pytorch_tanh" the tanh one
        v.gelu_tanh = if (std.mem.eql(u8, x.string, "gelu")) false else if (std.mem.eql(u8, x.string, "gelu_pytorch_tanh")) true else return error.UnsupportedArchitecture;
    }
    if (o.get("projector_type")) |x| if (x != .string or !std.mem.eql(u8, x.string, "rmsnorm_mlp_adapter")) return error.UnsupportedArchitecture;
    if (o.get("temporal_patch_size")) |x| if (try intAs(u32, x) != 1) return error.UnsupportedArchitecture;
    return v;
}

fn readDraft(top: std.json.ObjectMap) !?Draft {
    const n = top.get("dfly_num_hidden_layers") orelse return null;
    var d: Draft = .{ .layers = try intAs(u32, n) };
    if (top.get("dfly_mask_token_id")) |v| d.mask_token = try intAs(u32, v);
    if (top.get("dfly_target_index_topk")) |v| d.index_topk = try intAs(u32, v);
    if (top.get("dfly_enable_hidden_correction")) |v| d.hidden_correction = if (v == .bool) v.bool else return error.BadConfig;
    if (top.get("speculative_config")) |v| if (v == .object) if (v.object.get("num_speculative_tokens")) |k| {
        d.tokens = try intAs(u32, k);
    };
    if (top.get("dfly_target_index_layer")) |v| {
        if (v != .array or v.array.items.len > d.targets.len) return error.BadConfig;
        for (v.array.items, 0..) |x, i| d.targets[i] = try intAs(u32, x);
        d.targets_len = @intCast(v.array.items.len);
    }
    // DFly2 reads the target's KV and its indexer picks at the same layers
    if (top.get("dfly_target_kv_layers")) |v| {
        if (v != .array or v.array.items.len != d.targets_len) return error.UnsupportedArchitecture;
        for (v.array.items, 0..) |x, i| if (try intAs(u32, x) != d.targets[i]) return error.UnsupportedArchitecture;
    }
    if (d.layers != d.targets_len) return error.UnsupportedArchitecture;
    return d;
}

fn numberText(v: std.json.Value) ![]const u8 {
    return switch (v) {
        .number_string => |s| s,
        .string => |s| s,
        .integer => error.BadConfig, // parse_numbers = false keeps every number as text
        else => error.BadConfig,
    };
}

fn intAs(comptime T: type, v: std.json.Value) !T {
    const text = try numberText(v);
    const n = std.fmt.parseInt(u64, text, 10) catch {
        // 4.0-style integers
        const f = std.fmt.parseFloat(f64, text) catch return error.BadConfig;
        if (f < 0 or f != @floor(f) or f > 1.8e19) return error.BadConfig;
        return std.math.cast(T, @as(u64, @intFromFloat(f))) orelse error.BadConfig;
    };
    return std.math.cast(T, n) orelse error.BadConfig;
}

fn float(v: std.json.Value) !f64 {
    const n = std.fmt.parseFloat(f64, try numberText(v)) catch return error.BadConfig;
    if (!std.math.isFinite(n)) return error.BadConfig;
    return n;
}
