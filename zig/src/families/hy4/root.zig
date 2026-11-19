//! Hy4-flash (tencent/Hy4-flash) family: the host-side pieces the CUDA engine is built on, tested without a GPU.
//! Reference: transformers PR #50903 (`hy_v4_flash`) and `hy_v4_vl`'s vision tower; numbers below come from that
//! code run on the published config and checkpoint tensors (tools/hy4/).

const std = @import("std");
pub const config = @import("config.zig");
pub const engram_hash = @import("engram_hash.zig");
pub const image_grid = @import("image_grid.zig");

const fixture = @embedFile("fixtures/config.json");

test "Hy4 config: the release's text, vision and DFly2 shapes" {
    const c = try config.Config.parse(std.testing.allocator, fixture);
    const t = &c.text;
    try std.testing.expectEqual(@as(u32, 42), t.layers);
    try std.testing.expectEqual(@as(u32, 3072), t.hidden);
    try std.testing.expectEqual(@as(u32, 120832), t.vocab);
    try std.testing.expectEqual(@as(u32, 32), t.indexed.heads);
    try std.testing.expectEqual(@as(u32, 64), t.sliding.heads);
    try std.testing.expectEqual(@as(u32, 256), t.indexed.qkDim());
    try std.testing.expectEqual(@as(u32, 256), t.sliding.v);
    try std.testing.expectEqual(@as(f64, 10000000.0), t.indexed.theta);
    try std.testing.expectEqual(@as(f64, 10000.0), t.sliding.theta);
    try std.testing.expectEqual([3]u32{ 12, 10, 10 }, t.sliding.mrope);
    try std.testing.expectEqual(@as(u32, 4), t.index_pool);
    try std.testing.expectEqual(@as(u32, 1536), t.latent);
    try std.testing.expectEqual(@as(u32, 16), t.topk);
    try std.testing.expectApproxEqAbs(@as(f32, 3.998), t.routed_scale, 1e-6);
    // 2 x indexed, then 10 x (3 x sliding, indexed); one dense layer then 41 MoE layers
    var indexed: u32 = 0;
    for (0..t.layers) |l| {
        const want: config.Attn = if (l < 2 or (l - 1) % 4 == 0) .indexed else .sliding;
        try std.testing.expectEqual(want, t.attn[l]);
        if (t.attn[l] == .indexed) indexed += 1;
        try std.testing.expectEqual(@as(config.Mlp, if (l == 0) .dense else .sparse), t.mlp[l]);
    }
    try std.testing.expectEqual(@as(u32, 12), indexed);
    // Engram: config ids 2, 12, 22, 32 are decoder layers 1, 11, 21, 31
    try std.testing.expectEqual(@as(u32, 4), t.engram_len);
    try std.testing.expect(t.engramAt(1) != null and t.engramAt(1).?.id == 2);
    try std.testing.expect(t.engramAt(2) == null);
    try std.testing.expectEqual(@as(u64, 10312007), t.engramAt(1).?.sizes[0]);
    try std.testing.expectEqual(@as(u64, 10313201), t.engramAt(31).?.sizes[15]);
    try std.testing.expectEqual(@as(u32, 16), t.hashHeads());
    try std.testing.expectEqual(@as(u32, 96), t.engramHeadDim());
    const v = c.vision.?;
    try std.testing.expectEqual(@as(u32, 27), v.layers);
    try std.testing.expectEqual(@as(u32, 1152), v.hidden);
    try std.testing.expectEqual(@as(u32, 4304), v.width);
    try std.testing.expect(!v.gelu_tanh);
    try std.testing.expectEqual(@as(u32, 2048 / 16), v.max_image / v.patch);
    const d = c.draft.?;
    try std.testing.expectEqual(@as(u32, 5), d.layers);
    try std.testing.expectEqualSlices(u32, &.{ 25, 29, 33, 37, 41 }, d.targets[0..d.targets_len]);
    try std.testing.expectEqual(@as(u32, 120120), c.image_token);
    try std.testing.expectEqual(@as(u32, 120121), c.image_newline);
}

test "Hy4 config: unknown routing, layer names and broken Engram tables are refused" {
    const a = std.testing.allocator;
    const swap = struct {
        fn of(gpa: std.mem.Allocator, from: []const u8, to: []const u8) ![]u8 {
            const n = std.mem.replacementSize(u8, fixture, from, to);
            const out = try gpa.alloc(u8, n);
            _ = std.mem.replace(u8, fixture, from, to, out);
            return out;
        }
    };
    const cases = .{
        .{ "\"moe_router_use_sigmoid\":true", "\"moe_router_use_sigmoid\":false", error.UnsupportedScoring },
        .{ "\"sliding_attention\",\"deepseek_sparse_attention\"", "\"sliding_attention\",\"full_attention\"", error.UnsupportedArchitecture },
        .{ "[10312007,", "[10312007,1,", error.BadConfig },
        .{ "\"mrope_section\":[12,10,10]", "\"mrope_section\":[12,10,11]", error.BadConfig },
        .{ "\"dfly_target_kv_layers\":[25", "\"dfly_target_kv_layers\":[24", error.UnsupportedArchitecture },
        .{ "\"model_type\":\"hy_v4_flash\",\"image_token_id\"", "\"model_type\":\"hy_v4\",\"image_token_id\"", error.UnsupportedArchitecture },
    };
    inline for (cases) |k| {
        const bad = try swap.of(a, k[0], k[1]);
        defer a.free(bad);
        try std.testing.expect(!std.mem.eql(u8, bad, fixture));
        try std.testing.expectError(k[2], config.Config.parse(a, bad));
    }
}

/// The release's normalized ids of the tokens the hash vectors use (lookup_table rows, checkpoint layer 1).
fn sampleMap(a: std.mem.Allocator) ![]i32 {
    const map = try a.alloc(i32, 120818);
    @memset(map, 0);
    const pairs = [_][2]u32{ .{ 120000, 91802 }, .{ 9707, 7863 }, .{ 15, 15 }, .{ 77777, 58569 }, .{ 120120, 91922 }, .{ 334, 280 }, .{ 120002, 91804 }, .{ 5, 5 }, .{ 120817, 92619 } };
    for (pairs) |p| map[p[0]] = @intCast(p[1]);
    return map;
}

test "Hy4 Engram: hash ids match the reference at layers 2, 12 and 22" {
    const a = std.testing.allocator;
    const c = try config.Config.parse(a, fixture);
    const map = try sampleMap(a);
    defer a.free(map);
    try std.testing.expectEqual(engram_hash.release_vocab, engram_hash.compressedVocab(map));
    // begin, text, a picture's two slots, text, the pad, an id past the map, text
    const tokens = [_]u32{ 120000, 9707, 15, 77777, 120120, 120120, 334, 120002, 120831, 5 };
    const want = .{
        .{ 2, 0, [16]u32{ 4258903, 5898252, 6636782, 270168, 4842181, 4927947, 8186064, 3200044, 9219594, 10279359, 9120768, 4620943, 1972793, 6953, 6071913, 1326055 } },
        .{ 2, 3, [16]u32{ 6158650, 498910, 2988815, 3319770, 146277, 7096659, 2195977, 2459146, 8212819, 9781388, 8076325, 943455, 4105885, 5622112, 3279121, 5600540 } },
        .{ 2, 9, [16]u32{ 8343620, 7346317, 8122312, 7195212, 7901973, 444072, 6264579, 4681764, 2518221, 2312179, 9601928, 5129061, 7206980, 5962909, 3062389, 4840879 } },
        .{ 12, 0, [16]u32{ 8690649, 8580402, 7366840, 1103745, 9893583, 6026396, 8365880, 8850998, 6467954, 5056909, 3050448, 9265312, 8966820, 3294345, 5625414, 6640886 } },
        .{ 12, 3, [16]u32{ 1656312, 9049754, 4924937, 1693858, 6339124, 7450436, 2325299, 9950290, 612441, 5622947, 10040218, 9368165, 9276206, 505857, 7438621, 2217901 } },
        .{ 12, 9, [16]u32{ 3799449, 8241736, 7313302, 9644829, 6325654, 2152289, 1278223, 3843207, 1781311, 8443670, 9931559, 2623396, 5579011, 7657755, 10012472, 8261471 } },
        .{ 22, 0, [16]u32{ 2399484, 2005409, 916610, 4980744, 3044152, 7321627, 39109, 7380195, 3003707, 10168088, 9821496, 7029273, 8368637, 2142522, 9783591, 2504876 } },
        .{ 22, 6, [16]u32{ 6972329, 4262142, 6545510, 779720, 5970337, 5773884, 6934387, 8738244, 5929874, 10229331, 7245543, 9573652, 3779696, 8627800, 9195738, 7601153 } },
        .{ 22, 8, [16]u32{ 1957272, 5633780, 3730795, 3454011, 7490814, 3358564, 2052351, 5058675, 4124271, 4157990, 7451025, 1859578, 4684359, 6972060, 9529554, 8946087 } },
    };
    inline for (want) |w| {
        const e = c.text.engramAt(w[0] - 1).?;
        const h = try engram_hash.Hasher.of(&c.text, e, map, try engram_hash.releaseMultipliers(e.id, c.text.engram_seed, map));
        var got: [16]u32 = undefined;
        h.row(&tokens, 0, w[1], &got);
        const expect: [16]u32 = w[2];
        try std.testing.expectEqualSlices(u32, &expect, &got);
    }
    const mask = [_]bool{ true, true, true, true, false, false, true, false, true, true };
    for (tokens, mask) |tok, m| try std.testing.expectEqual(m, engram_hash.owns(tok, c.image_token, c.video_token, c.text.engram_pad));
}

test "Hy4 Engram: a document's first tokens hash its earlier slots as the pad" {
    const a = std.testing.allocator;
    const c = try config.Config.parse(a, fixture);
    const map = try sampleMap(a);
    defer a.free(map);
    const e = c.text.engramAt(1).?;
    const h = try engram_hash.Hasher.of(&c.text, e, map, try engram_hash.releaseMultipliers(e.id, 0, map));
    // [x, x, 120000, 9707, 15, ...] read from start 2 is [120000, 9707, ...] read from 0
    const packed_ = [_]u32{ 5, 334, 120000, 9707, 15, 77777 };
    const alone = [_]u32{ 120000, 9707, 15, 77777 };
    for (0..4) |i| {
        var x: [16]u32 = undefined;
        var y: [16]u32 = undefined;
        h.row(&packed_, 2, i + 2, &x);
        h.row(&alone, 0, i, &y);
        try std.testing.expectEqualSlices(u32, &y, &x);
    }
    try std.testing.expectError(error.UnsupportedEngramMap, engram_hash.releaseMultipliers(2, 1, map));
    var offsets: [16]u64 = undefined;
    engram_hash.headOffsets(e.sizes[0..16], &offsets);
    try std.testing.expectEqual(@as(u64, 0), offsets[0]);
    try std.testing.expectEqual(@as(u64, 10312007), offsets[1]);
}

test "Hy4 pictures: smart resize, token counts and M-RoPE positions" {
    const l: image_grid.Limits = .{};
    // 1000 x 750 rounds to multiples of 32
    const s = try image_grid.smartResize(750, 1000, l);
    try std.testing.expectEqual(image_grid.Size{ .height = 736, .width = 992 }, s);
    const g = image_grid.Grid.of(s, l);
    try std.testing.expectEqual(image_grid.Grid{ .h = 46, .w = 62 }, g);
    try std.testing.expectEqual(@as(u32, 23 * 32), g.tokens(2));
    // a 4000 x 3000 photo is capped at 2048 x 2048 pixels
    const big = try image_grid.smartResize(3000, 4000, l);
    try std.testing.expect(@as(u64, big.height) * big.width <= l.max_pixels);
    try std.testing.expectEqual(image_grid.Size{ .height = 1760, .width = 2336 }, big);
    // a tiny icon grows to the floor
    const tiny = try image_grid.smartResize(10, 20, l);
    try std.testing.expect(@as(u64, tiny.height) * tiny.width >= l.min_pixels);
    try std.testing.expectEqual(image_grid.Size{ .height = 64, .width = 96 }, tiny);
    try std.testing.expectError(error.AspectRatio, image_grid.smartResize(1, 300, l));
    // python's round: 48 / 32 = 1.5 -> 2, 80 / 32 = 2.5 -> 2
    try std.testing.expectEqual(image_grid.Size{ .height = 64, .width = 64 }, try image_grid.smartResize(48, 80, l));

    // the docstring's 2 x 2 merged frame: rows of 3 (2 columns + newline), next text at max + 1
    const f: image_grid.Grid = .{ .h = 4, .w = 4 };
    var t: [6]u32 = undefined;
    var h: [6]u32 = undefined;
    var w: [6]u32 = undefined;
    const next = f.positions(2, 10, &t, &h, &w);
    try std.testing.expectEqualSlices(u32, &.{ 10, 10, 10, 10, 10, 10 }, &t);
    try std.testing.expectEqualSlices(u32, &.{ 10, 10, 10, 11, 11, 11 }, &h);
    try std.testing.expectEqualSlices(u32, &.{ 10, 11, 12, 10, 11, 12 }, &w);
    try std.testing.expectEqual(@as(u32, 13), next);
}

test "Hy4 pictures: the position grid's resampling taps and patch order" {
    // n = edge: scale (128.1 / 128) samples just left of each cell
    const tap = image_grid.positionTap(128, 128, 0);
    try std.testing.expectEqual(@as(u32, 0), tap.lo);
    const mid = image_grid.positionTap(128, 64, 10);
    // (10.5) / (64.1 / 128) - 0.5 = 20.4672...
    try std.testing.expectEqual(@as(u32, 20), mid.lo);
    try std.testing.expectEqual(@as(u32, 21), mid.hi);
    try std.testing.expectApproxEqAbs(@as(f32, 0.46724), mid.frac, 1e-4);
    const last = image_grid.positionTap(128, 2, 1);
    try std.testing.expectEqual(@as(u32, 90), last.lo);
    try std.testing.expectApproxEqAbs(@as(f32, 0.928574), last.frac, 1e-4);
    const same = image_grid.positionTap(128, 128, 1);
    try std.testing.expect(same.lo == 1 and same.frac == 0);

    // torch's F.interpolate of a y * 1000 + x grid (edge 128) to 3 x 5 at scale (n + 0.1) / 128
    const edge = 128;
    const grid = try std.testing.allocator.alloc(f32, edge * edge);
    defer std.testing.allocator.free(grid);
    for (0..edge) |y| for (0..edge) |x| {
        grid[y * edge + x] = @floatFromInt(y * 1000 + x);
    };
    var out: [15]f32 = undefined;
    image_grid.resamplePositions(grid, edge, 1, .{ .h = 3, .w = 5 }, &out);
    const torch_ref = [15]f32{ 20157.208984375, 20182.306640625, 20207.40625, 20232.50390625, 20257.603515625, 61447.53125, 61472.62890625, 61497.7265625, 61522.828125, 61547.921875, 102737.8515625, 102762.9453125, 102788.046875, 102813.140625, 102838.2421875 };
    for (torch_ref, out) |a, b| try std.testing.expectApproxEqAbs(a, b, 0.02);

    // patch (y, x) is channel-major 2 x 2 pixels
    const sz: image_grid.Size = .{ .height = 2, .width = 4 };
    var px: [24]f32 = undefined;
    for (&px, 0..) |*p, i| p.* = @floatFromInt(i);
    var pt: [24]f32 = undefined;
    image_grid.patchify(&px, sz, 2, &pt);
    try std.testing.expectEqualSlices(f32, &.{ 0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21 }, pt[0..12]);
    try std.testing.expectEqualSlices(f32, &.{ 2, 3, 6, 7 }, pt[12..16]);
}

pub const reply = @import("reply.zig");

test "Hy4 reply: thinking and tool calls become the GLM forms, whole and streamed" {
    const a = std.testing.allocator;
    const hy = "Need the tool.</think:opensource><tool_calls:opensource><tool_call:opensource>weather<arg_key:opensource>city</arg_key:opensource><arg_value:opensource>Paris</arg_value:opensource></tool_call:opensource><tool_call:opensource>time<arg_key:opensource>tz</arg_key:opensource><arg_value:opensource>\"CET\"</arg_value:opensource></tool_call:opensource></tool_calls:opensource>";
    const glm = "Need the tool.</think><tool_call>weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call><tool_call>time<arg_key>tz</arg_key><arg_value>\"CET\"</arg_value></tool_call>";
    const whole = try reply.toGlm(a, hy);
    defer a.free(whole);
    try std.testing.expectEqualStrings(glm, whole);
    // every split point, including inside tags, gives the same text
    for (1..hy.len) |cut| {
        var s: reply.Stream = .{};
        defer s.deinit(a);
        var out: std.ArrayList(u8) = .empty;
        defer out.deinit(a);
        try s.feed(a, hy[0..cut], &out);
        try s.feed(a, hy[cut..], &out);
        try s.finish(a, &out);
        try std.testing.expectEqualStrings(glm, out.items);
    }
    // text that only looks like a tag's start stays text; GLM's own tags pass through
    const plain = try reply.toGlm(a, "a < b <think> <tool_call:x>");
    defer a.free(plain);
    try std.testing.expectEqualStrings("a < b <think> <tool_call:x>", plain);
    var s: reply.Stream = .{};
    defer s.deinit(a);
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(a);
    try s.feed(a, "x <arg_ke", &out);
    try std.testing.expectEqualStrings("x ", out.items);
    try s.finish(a, &out);
    try std.testing.expectEqualStrings("x <arg_ke", out.items);
    try std.testing.expectEqualStrings("high", reply.reasoningEffort(null));
    try std.testing.expectEqualStrings("high", reply.reasoningEffort(true));
    try std.testing.expectEqualStrings("no_think", reply.reasoningEffort(false));
}

pub const layer_ref = @import("layer_ref.zig");

/// The f32 tensors of a small safetensors file, by name (the header's offsets into the bytes after it).
const Tensors = struct {
    bytes: []const u8,
    header: std.json.Parsed(std.json.Value),

    fn open(a: std.mem.Allocator, bytes: []const u8) !Tensors {
        const n = std.mem.readInt(u64, bytes[0..8], .little);
        return .{ .bytes = bytes[8 + n ..], .header = try std.json.parseFromSlice(std.json.Value, a, bytes[8..][0..n], .{}) };
    }

    fn f32s(t: *const Tensors, a: std.mem.Allocator, name: []const u8) ![]f32 {
        const m = t.header.value.object.get(name) orelse return error.MissingTensor;
        const off = m.object.get("data_offsets").?.array.items;
        const lo: usize = @intCast(off[0].integer);
        const hi: usize = @intCast(off[1].integer);
        const out = try a.alloc(f32, (hi - lo) / 4);
        for (out, 0..) |*o, i| o.* = @bitCast(std.mem.readInt(u32, t.bytes[lo + 4 * i ..][0..4], .little));
        return out;
    }

    fn i64s(t: *const Tensors, a: std.mem.Allocator, name: []const u8) ![]i64 {
        const m = t.header.value.object.get(name) orelse return error.MissingTensor;
        const off = m.object.get("data_offsets").?.array.items;
        const lo: usize = @intCast(off[0].integer);
        const hi: usize = @intCast(off[1].integer);
        const out = try a.alloc(i64, (hi - lo) / 8);
        for (out, 0..) |*o, i| o.* = std.mem.readInt(i64, t.bytes[lo + 8 * i ..][0..8], .little);
        return out;
    }

    fn i32s(t: *const Tensors, a: std.mem.Allocator, name: []const u8) ![]i32 {
        const f = try t.f32s(a, name);
        return @ptrCast(f);
    }
};

test "Hy4 layer reference: indexed, sliding and MoE decoder layers match transformers" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const st = try Tensors.open(a, @embedFile("fixtures/layers.safetensors"));
    const positions = try st.i32s(a, "positions");
    const t = positions.len / 3;
    const shared: layer_ref.Dims = .{ .hidden = 32, .hc = 4, .heads = 2, .nope = 8, .rope = 12, .v = 8, .q_lora = 16, .kv_lora = 8, .inter = 48, .eps = 1e-5, .hc_eps = 1e-6, .hc_magnitude = 2.0, .theta = 1e7, .sections = .{ 2, 2, 2 } };
    var indexed = shared;
    indexed.index = .{ .heads = 2, .dim = 16, .kpool = 4, .topk = 8 };
    var sliding = shared;
    sliding.heads = 3;
    sliding.theta = 1e4;
    sliding.window = 4;
    for ([_]layer_ref.Dims{ indexed, sliding, sliding }, [_][]const u8{ "l0.", "l1.", "l2." }) |d, pre| {
        const S = struct {
            st: *const Tensors,
            a: std.mem.Allocator,
            pre: []const u8,
            fn get(s: @This(), name: []const u8) ![]f32 {
                return s.st.f32s(s.a, try std.mem.concat(s.a, u8, &.{ s.pre, name }));
            }
        };
        const s: S = .{ .st = &st, .a = a, .pre = pre };
        var w: layer_ref.Weights = .{
            .attn_hc = .{ .fn_ = try s.get("attn_hc.fn"), .base = try s.get("attn_hc.base"), .scale = try s.get("attn_hc.scale") },
            .ffn_hc = .{ .fn_ = try s.get("ffn_hc.fn"), .base = try s.get("ffn_hc.base"), .scale = try s.get("ffn_hc.scale") },
            .input_norm = try s.get("input_layernorm.weight"),
            .post_attention_norm = try s.get("post_attention_layernorm.weight"),
            .post_mlp_norm = s.get("post_mlp_layernorm.weight") catch &.{},
            .q_a = try s.get("self_attn.q_a_proj.weight"),
            .q_a_norm = try s.get("self_attn.q_a_layernorm.weight"),
            .q_b = try s.get("self_attn.q_b_proj.weight"),
            .kv_a = try s.get("self_attn.kv_a_proj_with_mqa.weight"),
            .kv_a_norm = try s.get("self_attn.kv_a_layernorm.weight"),
            .kv_b = try s.get("self_attn.kv_b_proj.weight"),
            .gate = try s.get("self_attn.gate_proj.weight"),
            .o = try s.get("self_attn.o_proj.weight"),
            .post_proj_norm = try s.get("self_attn.post_proj_norm.weight"),
            .sinks = try s.get("self_attn.sinks"),
        };
        if (s.get("mlp.gate.weight")) |router| {
            w.moe = .{
                .experts = 8,
                .top = 3,
                .latent = 16,
                .width = 12,
                .scale = 3.998,
                .clamp = 10,
                .router = router,
                .bias = try s.get("mlp.gate.e_score_correction_bias"),
                .pre = try s.get("mlp.pre_expert_proj.weight"),
                .post = try s.get("mlp.post_expert_proj.weight"),
                .gate_up = try s.get("mlp.experts.gate_up_proj"),
                .down = try s.get("mlp.experts.down_proj"),
                .shared_gate = try s.get("mlp.shared_experts.gate_proj.weight"),
                .shared_up = try s.get("mlp.shared_experts.up_proj.weight"),
                .shared_down = try s.get("mlp.shared_experts.down_proj.weight"),
                .norm = try s.get("mlp.combine_output_norm.weight"),
            };
        } else |_| {
            w.mlp_gate = try s.get("mlp.gate_proj.weight");
            w.mlp_up = try s.get("mlp.up_proj.weight");
            w.mlp_down = try s.get("mlp.down_proj.weight");
        }
        if (d.index != null) w.index = .{
            .query = try s.get("self_attn.compressed_indexer.query_projection.weight"),
            .head_weight = try s.get("self_attn.compressed_indexer.head_weight_projection.weight"),
            .token_key = try s.get("self_attn.compressed_indexer.token_key_projection.weight"),
            .token_key_norm = try s.get("self_attn.compressed_indexer.token_key_norm.weight"),
            .key_value = try s.get("self_attn.compressed_indexer.key_value_projection.weight"),
            .key_gate = try s.get("self_attn.compressed_indexer.key_gate_projection.weight"),
            .key_norm = try s.get("self_attn.compressed_indexer.key_norm.weight"),
            .ape = try s.get("self_attn.compressed_indexer.intra_group_ape"),
        };
        const streams = try s.get("in");
        const want = try s.get("out");
        try layer_ref.forward(a, &d, &w, positions, t, streams);
        var worst: f32 = 0;
        for (streams, want) |g, e| worst = @max(worst, @abs(g - e) / @max(1.0, @abs(e)));
        if (worst > 2e-4) std.debug.print("{s} worst relative error {e}\n", .{ pre, worst });
        try std.testing.expect(worst <= 2e-4);
    }
}

pub const preprocess = @import("preprocess.zig");

test "Hy4 pictures: preprocessing matches HYV4VLImageProcessor (upscaled, downscaled, tall)" {
    const a = std.testing.allocator;
    const ref = try std.json.parseFromSlice(std.json.Value, a, @embedFile("fixtures/pictures.json"), .{});
    defer ref.deinit();
    for (ref.value.array.items) |case| {
        const o = case.object;
        const w: u32 = @intCast(o.get("width").?.integer);
        const h: u32 = @intCast(o.get("height").?.integer);
        // tools/hy4/picture_fixture.py's drawing
        const rgb = try a.alloc(u8, @as(usize, w) * h * 3);
        defer a.free(rgb);
        for (0..h) |y| for (0..w) |x| for (0..3) |c| {
            rgb[(y * w + x) * 3 + c] = @intCast((x * 7 + y * 13 + c * 50 + (x * y) % 31) % 256);
        };
        var p = try preprocess.picture(a, rgb, w, h, .{});
        defer p.deinit(a);
        const grid = o.get("grid").?.array.items;
        try std.testing.expectEqual(@as(u32, @intCast(grid[1].integer)), p.grid.h);
        try std.testing.expectEqual(@as(u32, @intCast(grid[2].integer)), p.grid.w);
        var sum: f64 = 0;
        for (p.patches) |v| sum += v;
        const every = o.get("every97").?.array.items;
        var off: usize = 0;
        for (every, 0..) |e, k| {
            const want: f32 = @floatCast(switch (e) {
                .float => |f| f,
                .integer => |n| @as(f64, @floatFromInt(n)),
                else => unreachable,
            });
            // one uint8 step is 2 / 255
            if (@abs(p.patches[k * 97] - want) > 1e-6) off += 1;
        }
        if (off > 0) std.debug.print("{d}x{d}: {d} of {d} samples differ, sum {d} vs {d}\n", .{ w, h, off, every.len, sum, o.get("sum").?.float });
        try std.testing.expectEqual(@as(usize, 0), off);
        try std.testing.expectApproxEqAbs(o.get("sum").?.float, sum, 1e-2);
    }
}

pub const pack = @import("pack.zig");

/// The release config cut to the fixture slice of the pack index (tools/hy4/pack_index_fixture.py).
fn sliceConfig(a: std.mem.Allocator) !config.Config {
    var c = try config.Config.parse(a, fixture);
    c.text.layers = 3;
    c.text.experts = 2;
    c.text.engram_len = 1; // config id 2: decoder layer 1
    c.vision.?.layers = 1;
    c.draft.?.layers = 1;
    return c;
}

test "Hy4 pack: the real index's slice has every tensor the config needs, EXL3 or plain, and nothing else" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try sliceConfig(a);
    const want = try pack.expected(a, &c);
    const index = @embedFile("fixtures/pack_index.json");
    const r = try pack.check(a, index, want, &.{});
    for (r.missing) |m| std.debug.print("missing {s}\n", .{m});
    for (r.unknown) |m| std.debug.print("unknown {s}\n", .{m});
    for (r.partial) |m| std.debug.print("partial {s}\n", .{m});
    try std.testing.expect(r.ok());
    // layer 0's attention and dense MLP are plain; layers 1-2: 3 attention + 3 shared + 2 x 3 expert projections each,
    // plus the lm head and the draft layer's 6
    try std.testing.expectEqual(@as(usize, 6), r.plain_projections);
    try std.testing.expectEqual(@as(usize, 1 + 2 * (3 + 3 + 6) + 6), r.exl3);
    try std.testing.expectEqual(@as(u64, 99754676490), r.total_size);

    // half an EXL3 projection, a dropped tensor and a stray name are each refused
    const cases = [_][3][]const u8{
        .{ "\"model.layers.1.mlp.experts.0.up_proj.suh\":", "\"model.layers.1.mlp.experts.0.up_proj.xxx\":", "partial" },
        .{ "\"model.layers.2.self_attn.learnable_sink_param\":", "\"model.layers.2.self_attn.learnable_sink_parameter\":", "missing" },
        .{ "\"model.norm.weight\":", "\"model.norm.weight\":\"x\",\"model.extra.weight\":", "unknown" },
    };
    for (cases) |k| {
        const n = std.mem.replacementSize(u8, index, k[0], k[1]);
        const bad = try a.alloc(u8, n);
        try std.testing.expectEqual(@as(usize, 1), std.mem.replace(u8, index, k[0], k[1], bad));
        const b = try pack.check(a, bad, want, &.{});
        try std.testing.expect(!b.ok());
        const list = if (std.mem.eql(u8, k[2], "partial")) b.partial else if (std.mem.eql(u8, k[2], "missing")) b.missing else b.unknown;
        try std.testing.expect(list.len >= 1);
    }
    const renamed = try pack.canonical(a, "model.layers.3.hc_attn_layer.hc_pre.hc_fn");
    try std.testing.expectEqualStrings("model.layers.3.attn_hc.fn", renamed);
    try std.testing.expectEqualStrings("model.layers.0.mlp.post_mlp_layernorm.weight", try pack.canonical(a, "model.layers.0.mlp.post_fc2_norm.weight"));
}

test "Hy4 pack: the whole downloaded pack (HY4_PACK_DIR, on demand)" {
    const dir = std.testing.environ.getPosix("HY4_PACK_DIR") orelse return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    const read = struct {
        fn f(al: std.mem.Allocator, d: []const u8, name: []const u8) ![]u8 {
            return std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(al, &.{ d, name }), al, .limited(1 << 28));
        }
    }.f;
    const c = try config.Config.parse(a, try read(a, dir, "config.json"));
    const r = try pack.check(a, try read(a, dir, "model.safetensors.index.json"), try pack.expected(a, &c), &.{});
    std.debug.print("pack: {d} tensors, {d} EXL3 and {d} plain projections, {d} shards, {d} bytes; missing {d}, unknown {d}, partial {d}\n", .{ r.tensors, r.exl3, r.plain_projections, r.shards, r.total_size, r.missing.len, r.unknown.len, r.partial.len });
    for (r.missing[0..@min(r.missing.len, 10)]) |m| std.debug.print("missing {s}\n", .{m});
    for (r.unknown[0..@min(r.unknown.len, 10)]) |m| std.debug.print("unknown {s}\n", .{m});
    try std.testing.expect(r.ok());
    const side = try pack.check(a, try read(a, dir, "engram.safetensors.index.json"), &.{}, try pack.expectedSidecar(a, &c));
    try std.testing.expectEqual(@as(usize, 0), side.unknown.len);
    try std.testing.expectEqual(@as(usize, 2), side.shards);
}

pub const engram_ref = @import("engram_ref.zig");

test "Hy4 Engram reference: windows, hash rows, FP8 reads and the gated residual match transformers" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const st = try Tensors.open(a, @embedFile("fixtures/engram.safetensors"));
    const positions = try st.i32s(a, "positions");
    const t = positions.len / 3;
    const ids: []const u32 = @ptrCast(try st.i32s(a, "ids"));
    const want_windows: []const u32 = @ptrCast(try st.i32s(a, "ngram"));
    const wins = try a.alloc(u32, t * 3);
    engram_ref.windows(ids, positions[0..t], 3, 49, wins);
    try std.testing.expectEqualSlices(u32, want_windows, wins);
    const mask = try a.alloc(bool, t);
    for (mask, ids) |*m, id| m.* = id != 120120 and id != 120124 and id != 49;

    const cat = struct {
        fn of(al: std.mem.Allocator, tt: *const Tensors, comptime fmt: []const u8, n: usize) ![]f32 {
            var out: std.ArrayList(f32) = .empty;
            for (0..n) |i| try out.appendSlice(al, try tt.f32s(al, try std.fmt.allocPrint(al, fmt, .{i})));
            return out.items;
        }
    };
    const e: engram_ref.Engram = .{
        .hidden = 32,
        .hc = 4,
        .order = 3,
        .heads_per_order = 2,
        .head_dim = 8,
        .pad = 49,
        .map = try st.i32s(a, "engram_layer.lookup_table"),
        .mult = try st.i64s(a, "multipliers"),
        .sizes = &.{ 31, 37, 41, 43 },
        .table = try st.f32s(a, "engram_layer.multi_head_embedding.embedding.weight"),
        .head_norms = try cat.of(a, &st, "engram_layer.lookup_per_head_norms.{d}.weight", 4),
        .key_w = try cat.of(a, &st, "engram_layer.key_projs.{d}.weight", 4),
        .key_b = try cat.of(a, &st, "engram_layer.key_projs.{d}.bias", 4),
        .norm_key = try cat.of(a, &st, "engram_layer.norm_key.{d}.weight", 4),
        .norm_query = try cat.of(a, &st, "engram_layer.norm_query.{d}.weight", 4),
        .value_w = try st.f32s(a, "engram_layer.value_proj.weight"),
        .value_b = try st.f32s(a, "engram_layer.value_proj.bias"),
        .pre_gamma = try st.f32s(a, "engram_layer.unified_pre_gamma"),
        .scales = try cat.of(a, &st, "engram_layer.unified_per_hc_scales.{d}", 4),
    };
    const streams = try st.f32s(a, "in");
    const res = try a.alloc(f32, streams.len);
    try engram_ref.residual(a, &e, wins, mask, streams, res);
    const want_res = try st.f32s(a, "engram_out");
    var worst: f32 = 0;
    for (res, want_res) |g, w| worst = @max(worst, @abs(g - w) / @max(1.0, @abs(w)));
    if (worst > 2e-4) std.debug.print("engram residual worst relative error {e}\n", .{worst});
    try std.testing.expect(worst <= 2e-4);
    // E4M3: the grid, ties to even, subnormals
    try std.testing.expectEqual(@as(f32, 448), engram_ref.e4m3(448));
    try std.testing.expectEqual(@as(f32, 1.125), engram_ref.e4m3(1.0625 + 0.001));
    try std.testing.expectEqual(@as(f32, 1.0), engram_ref.e4m3(1.0625));
    try std.testing.expectEqual(@as(f32, 0.001953125), engram_ref.e4m3(0.0025));

    // the whole decoder layer: the Engram residual, then attention and the dense MLP
    for (streams, res) |*x, r| x.* += r;
    const d: layer_ref.Dims = .{ .hidden = 32, .hc = 4, .heads = 3, .nope = 8, .rope = 12, .v = 8, .q_lora = 16, .kv_lora = 8, .inter = 48, .eps = 1e-5, .hc_eps = 1e-6, .hc_magnitude = 2.0, .theta = 1e4, .sections = .{ 2, 2, 2 }, .window = 4 };
    const w: layer_ref.Weights = .{
        .attn_hc = .{ .fn_ = try st.f32s(a, "attn_hc.fn"), .base = try st.f32s(a, "attn_hc.base"), .scale = try st.f32s(a, "attn_hc.scale") },
        .ffn_hc = .{ .fn_ = try st.f32s(a, "ffn_hc.fn"), .base = try st.f32s(a, "ffn_hc.base"), .scale = try st.f32s(a, "ffn_hc.scale") },
        .input_norm = try st.f32s(a, "input_layernorm.weight"),
        .post_attention_norm = try st.f32s(a, "post_attention_layernorm.weight"),
        .post_mlp_norm = try st.f32s(a, "post_mlp_layernorm.weight"),
        .q_a = try st.f32s(a, "self_attn.q_a_proj.weight"),
        .q_a_norm = try st.f32s(a, "self_attn.q_a_layernorm.weight"),
        .q_b = try st.f32s(a, "self_attn.q_b_proj.weight"),
        .kv_a = try st.f32s(a, "self_attn.kv_a_proj_with_mqa.weight"),
        .kv_a_norm = try st.f32s(a, "self_attn.kv_a_layernorm.weight"),
        .kv_b = try st.f32s(a, "self_attn.kv_b_proj.weight"),
        .gate = try st.f32s(a, "self_attn.gate_proj.weight"),
        .o = try st.f32s(a, "self_attn.o_proj.weight"),
        .post_proj_norm = try st.f32s(a, "self_attn.post_proj_norm.weight"),
        .sinks = try st.f32s(a, "self_attn.sinks"),
        .mlp_gate = try st.f32s(a, "mlp.gate_proj.weight"),
        .mlp_up = try st.f32s(a, "mlp.up_proj.weight"),
        .mlp_down = try st.f32s(a, "mlp.down_proj.weight"),
    };
    try layer_ref.forward(a, &d, &w, positions, t, streams);
    const want = try st.f32s(a, "out");
    worst = 0;
    for (streams, want) |g, x| worst = @max(worst, @abs(g - x) / @max(1.0, @abs(x)));
    if (worst > 2e-4) std.debug.print("engram layer worst relative error {e}\n", .{worst});
    try std.testing.expect(worst <= 2e-4);
}

pub const vision_ref = @import("vision_ref.zig");
pub const pack_read = @import("pack_read.zig");

fn visionWeights(a: std.mem.Allocator, ctx: anytype, get: fn (@TypeOf(ctx), std.mem.Allocator, []const u8) anyerror![]f32, layers: usize) !vision_ref.Weights {
    const blocks = try a.alloc(vision_ref.Block, layers);
    for (blocks, 0..) |*b, i| {
        const n = try vision_ref.blockNames(a, i);
        var s: [16][]f32 = undefined;
        for (&s, n) |*o, name| o.* = try get(ctx, a, name);
        b.* = .{ .norm1_w = s[0], .norm1_b = s[1], .q_w = s[2], .q_b = s[3], .k_w = s[4], .k_b = s[5], .v_w = s[6], .v_b = s[7], .o_w = s[8], .o_b = s[9], .norm2_w = s[10], .norm2_b = s[11], .fc1_w = s[12], .fc1_b = s[13], .fc2_w = s[14], .fc2_b = s[15] };
    }
    return .{
        .patch_w = try get(ctx, a, "vit.embeddings.patch_embedding.weight"),
        .patch_b = try get(ctx, a, "vit.embeddings.patch_embedding.bias"),
        .positions = try get(ctx, a, "vit.embeddings.position_embedding.weight"),
        .blocks = blocks,
        .merge_norm = try get(ctx, a, "vit.perceive.norm.weight"),
        .conv_w = try get(ctx, a, "vit.perceive.proj.0.weight"),
        .conv_b = try get(ctx, a, "vit.perceive.proj.0.bias"),
        .out_w = try get(ctx, a, "vit.perceive.proj.2.weight"),
        .out_b = try get(ctx, a, "vit.perceive.proj.2.bias"),
        .newline = try get(ctx, a, "vit.perceive.image_newline"),
    };
}

fn visionCheck(a: std.mem.Allocator, d: *const vision_ref.Dims, w: *const vision_ref.Weights, st: *const Tensors) !f32 {
    const grid_v = try st.i32s(a, "grid");
    const grids = try a.alloc(image_grid.Grid, grid_v.len / 3);
    for (grids, 0..) |*g, i| g.* = .{ .h = @intCast(grid_v[3 * i + 1]), .w = @intCast(grid_v[3 * i + 2]) };
    const got = try vision_ref.forward(a, d, w, try st.f32s(a, "pixels"), grids);
    const want = try st.f32s(a, "features");
    try std.testing.expectEqual(want.len, got.len);
    var worst: f32 = 0;
    for (got, want) |g, x| worst = @max(worst, @abs(g - x) / @max(1.0, @abs(x)));
    return worst;
}

test "Hy4 vision reference: a shrunken tower on two pictures matches HYV4VLVisionModel" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const st = try Tensors.open(a, @embedFile("fixtures/vision.safetensors"));
    const get = struct {
        fn f(s: *const Tensors, al: std.mem.Allocator, name: []const u8) anyerror![]f32 {
            return s.f32s(al, name);
        }
    }.f;
    const w = try visionWeights(a, &st, get, 2);
    const d: vision_ref.Dims = .{ .hidden = 32, .inter = 48, .heads = 4, .patch = 4, .out = 24, .edge = 8 };
    const worst = try visionCheck(a, &d, &w, &st);
    if (worst > 2e-4) std.debug.print("vision worst relative error {e}\n", .{worst});
    try std.testing.expect(worst <= 2e-4);
}

test "Hy4 vision reference: the release's 27-layer tower read from the pack (HY4_PACK_DIR, on demand)" {
    const dir = std.testing.environ.getPosix("HY4_PACK_DIR") orelse return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var p = try pack_read.Pack.open(a, std.testing.io, dir, "model.safetensors.index.json");
    defer p.close();
    const get = struct {
        fn f(pk: *pack_read.Pack, al: std.mem.Allocator, name: []const u8) anyerror![]f32 {
            return pk.f32s(al, name);
        }
    }.f;
    const c = try config.Config.parse(a, fixture);
    const v = c.vision.?;
    const w = try visionWeights(a, &p, get, v.layers);
    const d: vision_ref.Dims = .{ .hidden = v.hidden, .inter = v.width, .heads = v.heads, .patch = v.patch, .merge = v.merge, .out = v.out_hidden, .edge = v.max_image / v.patch, .eps = v.eps };
    const st = try Tensors.open(a, @embedFile("fixtures/vision_real.safetensors"));
    const worst = try visionCheck(a, &d, &w, &st);
    std.debug.print("vision (release weights) worst relative error {e}\n", .{worst});
    try std.testing.expect(worst <= 2e-3);
}

pub const load_plan = @import("load_plan.zig");

fn smallConfig(a: std.mem.Allocator) !config.Config {
    var c = try config.Config.parse(a, fixture);
    c.text.layers = 3;
    c.text.experts = 8;
    c.text.engram_len = 1;
    return c;
}

test "Hy4 load plan: three layers of the real headers, EXL3 groups as the trellis kernels take them, placed and summed" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try smallConfig(a);
    var h = try load_plan.JsonHeaders.parse(a, @embedFile("fixtures/pack_headers.json"));
    var problem: load_plan.Problem = .{};
    const p = load_plan.build(a, &c, h.source(), h.source(), .{}, &problem) catch |e| {
        std.debug.print("{s}: {s}\n", .{ problem.name, problem.what });
        return e;
    };
    // layer 0's projections are plain bf16, layers 1-2 EXL3: lm_head, 3 attention groups and the shared expert's
    // 3 each, and 8 experts x 3
    try std.testing.expectEqual(@as(usize, 1 + 2 * (3 + 3 + 24)), p.exl3_groups);
    try std.testing.expectEqual(@as(usize, 3), p.widths[16]); // 8 bits: lm_head and both shared experts' down_proj
    try std.testing.expect(p.groupBytes(.routed_experts) > 0 and p.groupBytes(.vision) == 933_258_720);
    try std.testing.expectEqual(@as(u64, 164_994_380 * 96 + 164_994_380 * 2), p.nvme_bytes);
    for (p.entries) |e| if (std.mem.eql(u8, e.name, "model.layers.1.mlp.experts.0.gate_proj")) {
        try std.testing.expectEqual(load_plan.Exl3{ .k = 1536, .n = 1536, .halves = 10 }, e.exl3.?);
    };
    for (p.entries) |e| if (std.mem.eql(u8, e.name, "model.layers.0.self_attn.q_b_proj.weight")) {
        try std.testing.expectEqual(@as(?load_plan.Dtype, .BF16), e.dtype);
    };
    const text_only = try load_plan.build(a, &c, h.source(), null, .{ .vision = false }, &problem);
    for (text_only.entries) |e| try std.testing.expect(e.group != .vision and e.place == .gpu);
}

test "Hy4 load plan: a group without its marker, a wrong width and a mis-shaped router are refused by name" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try smallConfig(a);
    var problem: load_plan.Problem = .{};
    {
        var h = try load_plan.JsonHeaders.parse(a, @embedFile("fixtures/pack_headers.json"));
        _ = h.map.orderedRemove("model.layers.2.self_attn.o_proj.mul1");
        try std.testing.expectError(error.BadExl3, load_plan.build(a, &c, h.source(), null, .{}, &problem));
        try std.testing.expectEqualStrings("model.layers.2.self_attn.o_proj", problem.name);
    }
    {
        var h = try load_plan.JsonHeaders.parse(a, @embedFile("fixtures/pack_headers.json"));
        var cc = c;
        cc.text.latent = 1024;
        try std.testing.expectError(error.BadPack, load_plan.build(a, &cc, h.source(), null, .{}, &problem));
    }
    {
        var h = try load_plan.JsonHeaders.parse(a, @embedFile("fixtures/pack_headers.json"));
        var cc = c;
        cc.text.experts = 7;
        try std.testing.expectError(error.BadPack, load_plan.build(a, &cc, h.source(), null, .{}, &problem));
        try std.testing.expectEqualStrings("model.layers.1.mlp.gate.weight", problem.name);
    }
}

test "Hy4 load plan: the whole pack (HY4_PACK_DIR or HY4_HEADERS, on demand)" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    const c = try config.Config.parse(a, fixture);
    var problem: load_plan.Problem = .{};
    var side_pack: pack_read.Pack = undefined;
    var main_pack: pack_read.Pack = undefined;
    var json: load_plan.JsonHeaders = undefined;
    var src: load_plan.Source = undefined;
    var side: load_plan.Source = undefined;
    if (std.testing.environ.getPosix("HY4_PACK_DIR")) |dir| {
        main_pack = try pack_read.Pack.open(a, io, dir, "model.safetensors.index.json");
        side_pack = try pack_read.Pack.open(a, io, dir, "engram.safetensors.index.json");
        src = load_plan.Source.ofPack(&main_pack);
        side = load_plan.Source.ofPack(&side_pack);
    } else if (std.testing.environ.getPosix("HY4_HEADERS")) |path| {
        json = try load_plan.JsonHeaders.parse(a, try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 28)));
        src = json.source();
        side = json.source();
    } else return error.SkipZigTest;
    const p = load_plan.build(a, &c, src, side, .{ .draft = true }, &problem) catch |e| {
        std.debug.print("{s}: {s}\n", .{ problem.name, problem.what });
        return e;
    };
    std.debug.print("plan: {d} entries, {d} EXL3 groups, GPU {d} bytes, NVMe {d} bytes\n", .{ p.entries.len, p.exl3_groups, p.gpu_bytes, p.nvme_bytes });
    for (std.enums.values(load_plan.Group)) |g| std.debug.print("  {s}: {d}\n", .{ @tagName(g), p.groupBytes(g) });
    for (p.widths, 0..) |n, w| if (n > 0) std.debug.print("  {d} groups at {d}/2 bits\n", .{ n, w });
    try std.testing.expectEqual(@as(usize, 63253), p.exl3_groups);
    // every byte both indexes' total_size counts is planned (99 754 676 490 + 64 680 628 768), EXL3 sizes from shapes
    try std.testing.expectEqual(@as(u64, 164_435_305_258), p.gpu_bytes + p.nvme_bytes);
    const l = try device_layout.build(a, &c, &p);
    std.debug.print("device arena: {d} bytes over {d} slots, {d} layers\n", .{ l.total, l.slots.len, l.layers.len });
    try std.testing.expectEqual(@as(usize, 42), l.layers.len);
}

pub const device_layout = @import("device_layout.zig");

test "Hy4 device layout: aligned parts, every layer's slots, experts in stride, sized from the plan" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try smallConfig(a);
    var h = try load_plan.JsonHeaders.parse(a, @embedFile("fixtures/pack_headers.json"));
    var problem: load_plan.Problem = .{};
    const p = try load_plan.build(a, &c, h.source(), h.source(), .{}, &problem);
    const l = try device_layout.build(a, &c, &p);
    var sum: u64 = 0;
    var last_end: u64 = 0;
    for (l.slots) |s| for (s.parts) |part| {
        if (part.bytes == 0) continue;
        try std.testing.expect(part.offset % device_layout.alignment == 0 and part.offset >= last_end);
        last_end = part.offset + part.bytes;
        sum += part.bytes;
    };
    try std.testing.expectEqual(p.gpu_bytes, sum + 4 * p.exl3_groups); // mul1 markers stay on the host
    try std.testing.expect(l.total >= last_end and l.total - last_end < device_layout.alignment);
    // layer 0: dense, plain bf16 projections; layer 1: sparse with an Engram; layer 2: sparse, sliding
    try std.testing.expect(l.layers[0].mlp_gate != device_layout.none and l.slots[l.layers[0].q_b].exl3 == null);
    try std.testing.expect(l.layers[1].engram and l.layers[1].indexer[0] != device_layout.none);
    try std.testing.expect(!l.layers[2].engram and l.layers[2].indexer[0] == device_layout.none);
    const e1 = l.slots[l.layers[1].experts + 3 * 7 + 2].exl3.?; // expert 7's down_proj
    try std.testing.expectEqual(@as(u32, 1536), e1.k);
    const st = device_layout.storedStrides(l.slots[l.layers[2].o].exl3.?);
    try std.testing.expectEqual(@as(u64, (3072 / 16) * 12 * 4), st.k);
    try std.testing.expect(device_layout.fits(&l, l.total + 1, 1) and !device_layout.fits(&l, l.total, 1));
}
