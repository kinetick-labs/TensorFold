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
