//! The Hy4 loader's plan: every tensor the config needs, found in the pack's shard headers, its form checked (EXL3
//! groups as the existing trellis kernels read them, plain tensors at the shapes the config implies), and placed:
//! GPU-resident (text, vision, the DFly2 drafter) or read from the Engram sidecar on NVMe. Nothing is read but
//! headers, so a pack that would fail the load fails here first, and the plan's byte totals say whether it fits.
//!
//! EXL3 rules are those of families/glm/cuda_rexl3.zig `shapeOf` (vcruz305's REXL3/SAGE packs, the same packer):
//! trellis int16 [K/16, N/16, 16 * bits], suh fp16 [K], svh fp16 [N], a `mul1` int32 marker, K and N multiples of
//! 128, 8 * halves words with 2..16 halves.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Config = @import("config.zig");
const pack = @import("pack.zig");
const pack_read = @import("pack_read.zig");

pub const Dtype = pack_read.Dtype;

pub const Meta = struct { shard: []const u8, dtype: Dtype, shape: []const u64 };

/// Where a plan looks names up: the pack's shard headers, or a JSON map of them (tests).
pub const Source = struct {
    ctx: *anyopaque,
    lookup: *const fn (ctx: *anyopaque, name: []const u8) anyerror!?Meta,

    pub fn get(s: Source, name: []const u8) !?Meta {
        return s.lookup(s.ctx, name);
    }

    pub fn ofPack(p: *pack_read.Pack) Source {
        return .{ .ctx = p, .lookup = struct {
            fn f(ctx: *anyopaque, name: []const u8) anyerror!?Meta {
                const pk: *pack_read.Pack = @ptrCast(@alignCast(ctx));
                if (!pk.has(name)) return null;
                const i = try pk.info(name);
                return .{ .shard = i.shard, .dtype = i.dtype, .shape = i.shape };
            }
        }.f };
    }
};

/// `{"name": ["shard", "DTYPE", [shape...]], ...}`, as tools/hy4/pack_headers.py writes it.
pub const JsonHeaders = struct {
    a: Allocator,
    map: std.json.ObjectMap,

    pub fn parse(a: Allocator, text: []const u8) !JsonHeaders {
        const v = try std.json.parseFromSliceLeaky(std.json.Value, a, text, .{});
        return .{ .a = a, .map = v.object };
    }

    pub fn source(h: *JsonHeaders) Source {
        return .{ .ctx = h, .lookup = struct {
            fn f(ctx: *anyopaque, name: []const u8) anyerror!?Meta {
                const s: *JsonHeaders = @ptrCast(@alignCast(ctx));
                const e = (s.map.get(name) orelse return null).array.items;
                const dims = e[2].array.items;
                const shape = try s.a.alloc(u64, dims.len);
                for (shape, dims) |*o, d| o.* = @intCast(d.integer);
                return .{ .shard = e[0].string, .dtype = std.meta.stringToEnum(Dtype, e[1].string) orelse return error.UnknownDtype, .shape = shape };
            }
        }.f };
    }
};

pub const Group = enum { embed, head, attention, indexer, hc, dense_mlp, router, shared_expert, routed_experts, engram, engram_table, vision, draft };
pub const Place = enum { gpu, nvme };

pub const Exl3 = struct {
    k: u32,
    n: u32,
    /// 2 * bits; the kernels take 2..16, odd ones (half-bit steps) only with the mul1 codebook every group here has
    halves: u8,

    pub fn bytes(e: Exl3) u64 {
        return @as(u64, e.k) * e.n * e.halves / 16 + 2 * (@as(u64, e.k) + e.n) + 4;
    }
};

pub const Entry = struct {
    /// a tensor's name, or an EXL3 group's base name
    name: []const u8,
    group: Group,
    place: Place,
    bytes: u64,
    exl3: ?Exl3 = null,
    dtype: ?Dtype = null,
    shape: []const u64 = &.{},
    layer: ?u32 = null,
};

pub const Options = struct { vision: bool = true, draft: bool = false };

pub const Plan = struct {
    entries: []const Entry,
    bytes: [std.enums.values(Group).len]u64,
    gpu_bytes: u64,
    nvme_bytes: u64,
    exl3_groups: usize,
    /// EXL3 groups by width in halves (index = halves)
    widths: [17]usize,

    pub fn groupBytes(p: *const Plan, g: Group) u64 {
        return p.bytes[@intFromEnum(g)];
    }
};

pub const Problem = struct { name: []const u8 = "", what: []const u8 = "" };

fn dtypeBytes(d: Dtype) u64 {
    return switch (d) {
        .F32, .I32 => 4,
        .BF16, .F16, .I16 => 2,
        .F8_E4M3, .U8 => 1,
        .I64 => 8,
    };
}

fn count(shape: []const u64) u64 {
    var n: u64 = 1;
    for (shape) |s| n *= s;
    return n;
}

/// The (K, N) a projection must have: K inputs, N outputs. Null where the config does not say (the drafter's).
pub fn projectionDims(c: *const Config.Config, name: []const u8) ?[2]u32 {
    const t = &c.text;
    if (std.mem.eql(u8, name, "lm_head")) return .{ t.hidden, t.vocab };
    const pre = "model.layers.";
    if (!std.mem.startsWith(u8, name, pre)) return null;
    const rest = name[pre.len..];
    const dot = std.mem.indexOfScalar(u8, rest, '.') orelse return null;
    const layer = std.fmt.parseInt(u32, rest[0..dot], 10) catch return null;
    if (layer >= t.layers) return null;
    const h = t.layerHeads(layer);
    const tail = rest[dot + 1 ..];
    if (std.mem.eql(u8, tail, "self_attn.q_b_proj")) return .{ t.q_lora, h.heads * h.qkDim() };
    if (std.mem.eql(u8, tail, "self_attn.o_proj")) return .{ h.heads * h.v, t.hidden };
    if (std.mem.eql(u8, tail, "self_attn.linear_gate")) return .{ t.hidden, h.heads * h.v };
    if (std.mem.eql(u8, tail, "mlp.gate_proj") or std.mem.eql(u8, tail, "mlp.up_proj")) return .{ t.hidden, t.dense_width };
    if (std.mem.eql(u8, tail, "mlp.down_proj")) return .{ t.dense_width, t.hidden };
    if (std.mem.eql(u8, tail, "mlp.shared_experts.gate_proj") or std.mem.eql(u8, tail, "mlp.shared_experts.up_proj")) return .{ t.hidden, t.expert_width * t.shared_experts };
    if (std.mem.eql(u8, tail, "mlp.shared_experts.down_proj")) return .{ t.expert_width * t.shared_experts, t.hidden };
    if (std.mem.startsWith(u8, tail, "mlp.experts.")) {
        if (std.mem.endsWith(u8, tail, ".down_proj")) return .{ t.expert_width, t.latent };
        return .{ t.latent, t.expert_width };
    }
    return null;
}

/// The shape a plain tensor must have, where the config fixes it (null: not checked).
pub fn tensorShape(a: Allocator, c: *const Config.Config, name: []const u8) !?[]const u64 {
    const t = &c.text;
    const S = struct {
        fn of(al: Allocator, dims: []const u64) ![]const u64 {
            return al.dupe(u64, dims);
        }
    };
    if (std.mem.eql(u8, name, "model.embed_tokens.weight")) return try S.of(a, &.{ t.vocab, t.hidden });
    if (std.mem.eql(u8, name, "model.norm.weight")) return try S.of(a, &.{t.hidden});
    const pre = "model.layers.";
    if (!std.mem.startsWith(u8, name, pre)) return null;
    const rest = name[pre.len..];
    const dot = std.mem.indexOfScalar(u8, rest, '.') orelse return null;
    const layer = std.fmt.parseInt(u32, rest[0..dot], 10) catch return null;
    if (layer >= t.layers) return null;
    const h = t.layerHeads(layer);
    const tail = rest[dot + 1 ..];
    const table = [_]struct { []const u8, []const u64 }{
        .{ "input_layernorm.weight", &.{t.hidden} },
        .{ "post_attention_layernorm.weight", &.{t.hidden} },
        .{ "self_attn.q_a_proj.weight", &.{ t.q_lora, t.hidden } },
        .{ "self_attn.q_a_layernorm.weight", &.{t.q_lora} },
        .{ "self_attn.kv_a_proj_with_mqa.weight", &.{ t.kv_lora + h.qk_rope, t.hidden } },
        .{ "self_attn.kv_a_layernorm.weight", &.{t.kv_lora} },
        .{ "self_attn.kv_b_proj.weight", &.{ h.heads * (h.qk_nope + h.v), t.kv_lora } },
        .{ "self_attn.learnable_sink_param", &.{h.heads} },
        .{ "self_attn.post_proj_norm.weight", &.{t.hidden} },
        .{ "hc_attn_layer.hc_pre.hc_fn", &.{ 2 * t.hc_mult, t.hc_mult * t.hidden } },
        .{ "hc_mlp_layer.hc_pre.hc_fn", &.{ 2 * t.hc_mult, t.hc_mult * t.hidden } },
        .{ "mlp.gate.weight", &.{ t.experts, t.hidden } },
        .{ "mlp.gate.e_score_correction_bias", &.{t.experts} },
        .{ "mlp.routed_expert_down_proj.weight", &.{ t.latent, t.hidden } },
        .{ "mlp.routed_expert_up_proj.weight", &.{ t.hidden, t.latent } },
        .{ "mlp.combine_output_norm.weight", &.{t.hidden} },
        .{ "self_attn.compressed_indexer.query_projection.weight", &.{ t.index_heads * t.index_dim, t.q_lora } },
        .{ "self_attn.compressed_indexer.token_key_projection.weight", &.{ t.index_dim, t.hidden } },
        .{ "self_attn.compressed_indexer.head_weight_projection.weight", &.{ t.index_heads, t.hidden } },
        .{ "self_attn.compressed_indexer.intra_group_ape", &.{ t.index_pool, t.index_dim } },
        .{ "engram_layer.value_proj.weight", &.{ t.hidden, t.engram_width * (t.engram_max_ngram - 1) } },
        .{ "engram_layer.lookup_table", &.{t.engram_map_size} },
    };
    for (table) |e| if (std.mem.eql(u8, tail, e[0])) return try S.of(a, e[1]);
    return null;
}

fn groupOf(name: []const u8) Group {
    if (std.mem.startsWith(u8, name, "vit.")) return .vision;
    if (std.mem.startsWith(u8, name, "model.dfly_layers.")) return .draft;
    if (std.mem.eql(u8, name, "model.embed_tokens.weight")) return .embed;
    if (std.mem.eql(u8, name, "lm_head") or std.mem.startsWith(u8, name, "model.hc_head.") or std.mem.eql(u8, name, "model.norm.weight")) return .head;
    if (std.mem.indexOf(u8, name, ".multi_head_embedding.") != null) return .engram_table;
    if (std.mem.indexOf(u8, name, ".engram_layer.") != null) return .engram;
    if (std.mem.indexOf(u8, name, ".compressed_indexer.") != null) return .indexer;
    if (std.mem.indexOf(u8, name, ".hc_attn_layer.") != null or std.mem.indexOf(u8, name, ".hc_mlp_layer.") != null) return .hc;
    if (std.mem.indexOf(u8, name, ".mlp.experts.") != null) return .routed_experts;
    if (std.mem.indexOf(u8, name, ".mlp.shared_experts.") != null) return .shared_expert;
    if (std.mem.indexOf(u8, name, ".mlp.gate.") != null or std.mem.indexOf(u8, name, ".mlp.routed_expert_") != null or std.mem.indexOf(u8, name, ".mlp.combine_output_norm") != null) return .router;
    if (std.mem.indexOf(u8, name, ".mlp.") != null) return .dense_mlp;
    return .attention;
}

fn layerOf(name: []const u8) ?u32 {
    const pre = "model.layers.";
    if (!std.mem.startsWith(u8, name, pre)) return null;
    const rest = name[pre.len..];
    const dot = std.mem.indexOfScalar(u8, rest, '.') orelse return null;
    return std.fmt.parseInt(u32, rest[0..dot], 10) catch null;
}

fn exl3Of(src: Source, a: Allocator, base: []const u8, problem: *Problem) !?Exl3 {
    const tr = try src.get(try std.fmt.allocPrint(a, "{s}.trellis", .{base})) orelse return null;
    const fail = struct {
        fn f(p: *Problem, n: []const u8, what: []const u8) error{BadExl3} {
            p.* = .{ .name = n, .what = what };
            return error.BadExl3;
        }
    }.f;
    const suh = try src.get(try std.fmt.allocPrint(a, "{s}.suh", .{base})) orelse return fail(problem, base, "no suh");
    const svh = try src.get(try std.fmt.allocPrint(a, "{s}.svh", .{base})) orelse return fail(problem, base, "no svh");
    const mul1 = try src.get(try std.fmt.allocPrint(a, "{s}.mul1", .{base})) orelse return fail(problem, base, "no mul1 marker (only the mul1 codebook is planned)");
    if (tr.dtype != .I16 or tr.shape.len != 3) return fail(problem, base, "trellis is not int16 [K/16, N/16, words]");
    if (tr.shape[2] % 8 != 0) return fail(problem, base, "trellis words are not 8 * halves");
    const halves = tr.shape[2] / 8;
    if (halves < 2 or halves > 16) return fail(problem, base, "width outside 1..8 bits");
    const k = tr.shape[0] * 16;
    const n = tr.shape[1] * 16;
    if (k % 128 != 0 or n % 128 != 0) return fail(problem, base, "K or N not a multiple of 128");
    if (suh.dtype != .F16 or count(suh.shape) != k) return fail(problem, base, "suh is not fp16 [K]");
    if (svh.dtype != .F16 or count(svh.shape) != n) return fail(problem, base, "svh is not fp16 [N]");
    if (mul1.dtype != .I32 or count(mul1.shape) != 1) return fail(problem, base, "mul1 is not an int32 scalar");
    return .{ .k = @intCast(k), .n = @intCast(n), .halves = @intCast(halves) };
}

/// The plan for `c` over `src` (the GPU pack) and `sidecar` (the Engram sidecar's headers; null: not planned).
/// On error.BadPack / error.BadExl3, `problem` names the tensor and what is wrong. Arena-allocated.
pub fn build(a: Allocator, c: *const Config.Config, src: Source, sidecar: ?Source, opt: Options, problem: *Problem) !Plan {
    var entries: std.ArrayList(Entry) = .empty;
    var plan: Plan = .{ .entries = &.{}, .bytes = @splat(0), .gpu_bytes = 0, .nvme_bytes = 0, .exl3_groups = 0, .widths = @splat(0) };
    const want = try pack.expected(a, c);
    for (want) |w| {
        const group = groupOf(w.name);
        if (group == .vision and !opt.vision) continue;
        if (group == .draft and !opt.draft) continue;
        var e: Entry = .{ .name = w.name, .group = group, .place = .gpu, .bytes = 0, .layer = layerOf(w.name) };
        switch (w.need) {
            .projection => {
                const dims = projectionDims(c, w.name);
                if (try exl3Of(src, a, w.name, problem)) |x| {
                    if (dims) |d| if (d[0] != x.k or d[1] != x.n) {
                        problem.* = .{ .name = w.name, .what = try std.fmt.allocPrint(a, "EXL3 group is {d} x {d}, the config wants {d} x {d}", .{ x.k, x.n, d[0], d[1] }) };
                        return error.BadPack;
                    };
                    e.exl3 = x;
                    e.bytes = x.bytes();
                    plan.exl3_groups += 1;
                    plan.widths[x.halves] += 1;
                } else {
                    const pn = try std.fmt.allocPrint(a, "{s}.weight", .{w.name});
                    const m = try src.get(pn) orelse {
                        problem.* = .{ .name = w.name, .what = "neither an EXL3 group nor a plain weight" };
                        return error.BadPack;
                    };
                    if (dims) |d| if (m.shape.len != 2 or m.shape[0] != d[1] or m.shape[1] != d[0]) {
                        problem.* = .{ .name = pn, .what = "plain projection's shape is not [N, K] as the config has it" };
                        return error.BadPack;
                    };
                    e.name = pn;
                    e.dtype = m.dtype;
                    e.shape = m.shape;
                    e.bytes = count(m.shape) * dtypeBytes(m.dtype);
                }
            },
            .tensor => {
                const m = try src.get(w.name) orelse {
                    problem.* = .{ .name = w.name, .what = "missing" };
                    return error.BadPack;
                };
                if (try tensorShape(a, c, w.name)) |s| if (!std.mem.eql(u64, s, m.shape)) {
                    problem.* = .{ .name = w.name, .what = "shape differs from the config's" };
                    return error.BadPack;
                };
                e.dtype = m.dtype;
                e.shape = m.shape;
                e.bytes = count(m.shape) * dtypeBytes(m.dtype);
            },
        }
        try entries.append(a, e);
    }
    if (sidecar) |side| {
        const t = &c.text;
        for (try pack.expectedSidecar(a, c)) |name| {
            const m = try side.get(name) orelse {
                problem.* = .{ .name = name, .what = "missing from the Engram sidecar" };
                return error.BadPack;
            };
            const scale = std.mem.endsWith(u8, name, ".scale_bf16");
            const ok = if (scale) m.dtype == .BF16 and m.shape.len == 2 and m.shape[1] == 1 else m.dtype == .F8_E4M3 and m.shape.len == 2 and m.shape[1] == t.engramHeadDim();
            if (!ok) {
                problem.* = .{ .name = name, .what = if (scale) "per-row scales are not bf16 [rows, 1]" else "table is not e4m3 [rows, engram head dim]" };
                return error.BadPack;
            }
            try entries.append(a, .{ .name = name, .group = .engram_table, .place = .nvme, .bytes = count(m.shape) * dtypeBytes(m.dtype), .dtype = m.dtype, .shape = m.shape, .layer = layerOf(name) });
        }
    }
    for (entries.items) |e| {
        plan.bytes[@intFromEnum(e.group)] += e.bytes;
        switch (e.place) {
            .gpu => plan.gpu_bytes += e.bytes,
            .nvme => plan.nvme_bytes += e.bytes,
        }
    }
    plan.entries = entries.items;
    return plan;
}

/// One Engram table's rows as the sidecar stores them: `rows` e4m3 vectors of `dim` bytes from `values_at`, and one
/// bf16 scale each from `scales_at` (file offsets in `shard`). A row is two reads, `dim` and 2 bytes; the loader's
/// NVMe path (families/deepseek_v41/engram_rows.zig, 264-byte DSV4.1 rows) needs its row size made a parameter or
/// these tables repacked as `dim + 2`-byte rows before it can serve them.
pub const EngramTable = struct { layer: u32, shard: []const u8, rows: u64, dim: u32, values_at: u64, scales_at: u64 };

pub fn engramTables(a: Allocator, c: *const Config.Config, side: *pack_read.Pack) ![]EngramTable {
    var out: std.ArrayList(EngramTable) = .empty;
    for (c.text.engramLayers()) |e| {
        const layer = e.id - 1;
        const v = try side.info(try std.fmt.allocPrint(a, "model.layers.{d}.engram_layer.multi_head_embedding.embedding.weight", .{layer}));
        const s = try side.info(try std.fmt.allocPrint(a, "model.layers.{d}.engram_layer.multi_head_embedding.embedding.weight.scale_bf16", .{layer}));
        if (!std.mem.eql(u8, v.shard, s.shard)) return error.SplitEngramTable;
        try out.append(a, .{ .layer = layer, .shard = v.shard, .rows = v.shape[0], .dim = @intCast(v.shape[1]), .values_at = v.begin, .scales_at = s.begin });
    }
    return out.items;
}
