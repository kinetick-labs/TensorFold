//! DFlash2 configuration and validated kernel widths (qwen3_5's cuda_draft_config.zig against Flash Next's target).
const std = @import("std");

/// What the drafter must agree with in its target: the residual width, the vocabulary and the layers it taps.
pub const Target = struct { hidden: usize, vocab: usize, layers: usize };
const lp_top = 20;
pub const Config = struct {
    hidden: usize,
    intermediate: usize,
    layers: usize,
    vocab: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    block: usize,
    mask: u32,
    window: usize,
    group: usize,
    rank: usize,
    topk: usize,
    eps: f32,
    theta: f32,
    taps: [8]u32,
    n_taps: usize,

    pub fn qWidth(c: Config) usize {
        return c.heads * c.head_dim;
    }
    pub fn kvWidth(c: Config) usize {
        return c.kv_heads * c.head_dim;
    }
    pub fn dynamic(c: Config) usize {
        return 4 * (c.hidden / c.group);
    }
    /// The ring's rows: the window and a block more.
    pub fn ringRows(c: Config) usize {
        return c.window + 16;
    }
    pub fn ringBytes(c: Config) usize {
        return c.layers * 2 * c.ringRows() * c.kvWidth() * 2;
    }

    pub fn read(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, target: Target) !Config {
        const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
        defer gpa.free(path);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 20));
        defer gpa.free(bytes);
        return parse(gpa, bytes, target);
    }

    pub fn parse(gpa: std.mem.Allocator, bytes: []const u8, target: Target) !Config {
        const doc = try std.json.parseFromSlice(std.json.Value, gpa, bytes, .{});
        defer doc.deinit();
        if (doc.value != .object) return error.BadDraftConfig;
        const o = doc.value.object;
        const d = (o.get("dflash_config") orelse return error.BadDraftConfig);
        if (d != .object) return error.BadDraftConfig;
        var c: Config = .{
            .hidden = try int(o, "hidden_size"),
            .intermediate = try int(o, "intermediate_size"),
            .layers = try int(o, "num_hidden_layers"),
            .vocab = try int(o, "vocab_size"),
            .heads = try int(o, "num_attention_heads"),
            .kv_heads = try int(o, "num_key_value_heads"),
            .head_dim = try int(o, "head_dim"),
            .block = try int(d.object, "block_size"),
            .mask = @intCast(try whole(d.object, "mask_token_id")),
            .window = try int(o, "sliding_window"),
            .group = try int(d.object, "conv_group_size"),
            .rank = try int(d.object, "selector_rank"),
            .topk = try int(d.object, "selector_top_k"),
            .eps = @floatCast(try num(o, "rms_norm_eps")),
            .theta = @floatCast(try num((o.get("rope_parameters") orelse return error.BadDraftConfig).object, "rope_theta")),
            .taps = @splat(0),
            .n_taps = 0,
        };
        const taps = d.object.get("target_layer_ids") orelse return error.BadDraftConfig;
        if (taps != .array or taps.array.items.len == 0 or taps.array.items.len > c.taps.len) return error.BadDraftConfig;
        for (taps.array.items, 0..) |t, i| {
            if (t != .integer or t.integer < 0 or t.integer >= target.layers) return error.BadDraftConfig;
            c.taps[i] = @intCast(t.integer);
        }
        c.n_taps = taps.array.items.len;
        // what the kernels fix: the head width, two-tap convolutions, a block of 8, the selector's top-k within a row
        if (c.head_dim != 128 or try int(d.object, "conv_kernel_size") != 2 or c.block != 8 or c.topk > lp_top or c.mask >= c.vocab) return error.UnsupportedDraftConfig;
        if (c.hidden != target.hidden or c.vocab != target.vocab or c.hidden % 256 != 0 or c.hidden % c.group != 0 or c.heads % c.kv_heads != 0 or c.layers > 8) return error.UnsupportedDraftConfig;
        _ = std.math.mul(usize, c.heads, c.head_dim) catch return error.UnsupportedDraftConfig;
        _ = std.math.mul(usize, c.kv_heads, c.head_dim) catch return error.UnsupportedDraftConfig;
        if (c.window < 16 or c.window > 1 << 16) return error.UnsupportedDraftConfig;
        // a tap is a completed layer's rows, read as the next layer starts: the last layer has none
        for (c.taps[0..c.n_taps]) |t| if (t + 1 >= target.layers) return error.UnsupportedDraftConfig;
        // the checkpoint's 8-row block (7 drafts a chain); TF_DFLASH_BLOCK=16 runs the reference's 16-row blocks
        if (std.c.getenv("TF_DFLASH_BLOCK")) |v| if (std.mem.eql(u8, std.mem.span(v), "16")) {
            c.block = 16;
        };
        return c;
    }
};

/// A size, count or divisor: positive, so a zero can't reach a remainder or an empty allocation.
fn int(o: std.json.ObjectMap, key: []const u8) !usize {
    const v = try whole(o, key);
    if (v == 0) return error.BadDraftConfig;
    return v;
}

/// A token id: any whole number.
fn whole(o: std.json.ObjectMap, key: []const u8) !usize {
    const v = o.get(key) orelse return error.BadDraftConfig;
    if (v != .integer or v.integer < 0) return error.BadDraftConfig;
    return @intCast(v.integer);
}

fn num(o: std.json.ObjectMap, key: []const u8) !f64 {
    return switch (o.get(key) orelse return error.BadDraftConfig) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => error.BadDraftConfig,
    };
}

test "the Flash Next DFlash2 config: taps 5..45 of 48 layers, a block of 8, 2048-row window" {
    const gpa = std.testing.allocator;
    const text =
        \\{"dflash_config":{"block_size":8,"conv_group_size":16,"conv_kernel_size":2,"mask_token_id":248077,"selector_rank":256,
        \\ "selector_top_k":16,"target_layer_ids":[5,15,25,35,45]},"head_dim":128,"hidden_size":2560,"intermediate_size":7680,
        \\ "num_attention_heads":32,"num_hidden_layers":5,"num_key_value_heads":8,"rms_norm_eps":1e-06,
        \\ "rope_parameters":{"rope_theta":10000},"sliding_window":2048,"vocab_size":248320}
    ;
    const c = try Config.parse(gpa, text, .{ .hidden = 2560, .vocab = 248320, .layers = 48 });
    try std.testing.expectEqual(@as(usize, 5), c.n_taps);
    try std.testing.expectEqual(@as(u32, 45), c.taps[4]);
    try std.testing.expectEqual(@as(usize, 8), c.block);
    try std.testing.expectEqual(@as(usize, 640), c.dynamic());
    try std.testing.expectError(error.UnsupportedDraftConfig, Config.parse(gpa, text, .{ .hidden = 2560, .vocab = 248320, .layers = 46 }));
    try std.testing.expectError(error.UnsupportedDraftConfig, Config.parse(gpa, text, .{ .hidden = 4096, .vocab = 248320, .layers = 48 }));
}
