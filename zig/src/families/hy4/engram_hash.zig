//! Hy4-flash's Engram hashes (transformers HYV4FlashEngramLayer.hash_ngrams, PR #50903): each token, with the
//! `max_ngram - 1` tokens before it in its document, is mapped through the checkpoint's normalized vocabulary
//! (`lookup_table`), multiplied by the layer's odd multipliers, xor-mixed per n-gram order and reduced modulo each
//! hash head's prime vocabulary. The ids are row offsets into the layer's flat table, before the head offsets.

const std = @import("std");
const Config = @import("config.zig");

pub const max_order = 3;

/// The multipliers NumPy's `default_rng(seed + 10007 * layer_id).integers(0, int64_max // vocab // 2, 3) * 2 + 1`
/// draws for the release (seed 0, a normalized vocabulary of 92,620 ids). They define the hash; a checkpoint whose
/// map or seed differs needs its own (tools/hy4/engram_reference.py prints them).
pub const release_vocab: u32 = 92620;
pub const release_seed: u64 = 0;
pub const release = [_]struct { id: u32, mult: [max_order]u64 }{
    .{ .id = 2, .mult = .{ 81853575799317, 39690085502883, 24229915018165 } },
    .{ .id = 12, .mult = .{ 19457948045719, 44729556510419, 84779352922315 } },
    .{ .id = 22, .mult = .{ 72974230057307, 21530602121221, 55998057243097 } },
    .{ .id = 32, .mult = .{ 29128509562787, 16745155128523, 22474588302733 } },
};

/// The release multipliers of layer `id`, after checking the map and seed they were drawn for.
pub fn releaseMultipliers(id: u32, seed: u64, map: []const i32) ![max_order]u64 {
    if (seed != release_seed or compressedVocab(map) != release_vocab) return error.UnsupportedEngramMap;
    for (release) |r| if (r.id == id) return r.mult;
    return error.UnsupportedEngramMap;
}

/// `int(lookup_table.max()) + 1`: the bound the multipliers were drawn under.
pub fn compressedVocab(map: []const i32) u64 {
    var most: i32 = -1;
    for (map) |v| most = @max(most, v);
    return @intCast(most + 1);
}

/// One layer's hash: the normalized vocabulary, the multipliers and the moduli (order 2's heads, then order 3's).
pub const Hasher = struct {
    map: []const i32,
    pad: u32,
    order: u32,
    heads: u32,
    mult: [max_order]u64,
    sizes: []const u64,

    pub fn of(t: *const Config.Text, e: *const Config.EngramLayer, map: []const i32, mult: [max_order]u64) !Hasher {
        if (t.engram_max_ngram > max_order or map.len == 0 or t.engram_pad >= map.len) return error.BadConfig;
        for (map) |v| if (v < 0) return error.BadEngramMap;
        return .{ .map = map, .pad = t.engram_pad, .order = t.engram_max_ngram, .heads = t.engram_heads, .mult = mult, .sizes = e.sizes[0..t.hashHeads()] };
    }

    /// Ids past the map (resized embeddings) read as the pad's normalized id.
    fn compressed(h: *const Hasher, token: u32) u64 {
        const t = if (token < h.map.len) token else h.pad;
        return @intCast(h.map[t]);
    }

    /// The hash ids of the token at `tokens[at]`, whose document starts at `start` (earlier slots read as the pad):
    /// `out` gets `(order - 1) * heads` ids.
    pub fn row(h: *const Hasher, tokens: []const u32, start: usize, at: usize, out: []u32) void {
        std.debug.assert(out.len == h.sizes.len and at < tokens.len and start <= at);
        var mixed: [max_order]u64 = undefined;
        for (0..h.order) |o| {
            const tok = if (at - start >= o) tokens[at - o] else h.pad;
            // bounded draws: a product with any normalized id stays inside int64, as in the reference
            mixed[o] = h.compressed(tok) * h.mult[o];
        }
        var mix = mixed[0];
        for (2..h.order + 1) |n| {
            mix ^= mixed[n - 1];
            const first = (n - 2) * h.heads;
            for (0..h.heads) |k| out[first + k] = @intCast(mix % h.sizes[first + k]);
        }
    }
};

/// `engram_mask`: whether the token at a position owns an n-gram (not a picture's, a video's or the pad's slot).
pub fn owns(token: u32, image_token: u32, video_token: u32, pad: u32) bool {
    return token != image_token and token != video_token and token != pad;
}

/// Flat-table row offsets of each hash head (`[0] + sizes[:-1]` cumulated).
pub fn headOffsets(sizes: []const u64, out: []u64) void {
    var at: u64 = 0;
    for (sizes, out) |s, *o| {
        o.* = at;
        at += s;
    }
}
