//! Flash Next's MTP drafts on CUDA (Python qwen4_exp/cuda/mtp.py): the head, its draft chains and verify windows.

const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const wt = @import("cuda_prompt_wt.zig");
const load = @import("cuda_prompt_load.zig");
const batch = @import("cuda_batch.zig");
const prompt = @import("cuda_prompt.zig");
const gdn = @import("cuda_gdn.zig");

const dims = wt.dims;
const streams = wt.streams;
const wide = wt.wide;
const vocab = wt.vocab;
const eps = wt.eps;
const layers_n = wt.layers_n;
const linear_n = wt.linear_n;
const state_n = wt.state_n;
const out_sk = wt.out_sk;
const kv_heads = wt.kv_heads;
const head_dim = wt.head_dim;
const index_dim = wt.index_dim;
const ple_tail_n = wt.ple_tail_n;
const Q4 = wt.Q4;
const HcW = wt.HcW;

pub const Sampling = lanes.sampling.Sampling;

/// Drafts a verify window can hold beside its pending token.
pub const max_chain: usize = max_rows - 1;
/// Preserve the existing recipe unless an explicit depth is selected.
pub const default_depth: usize = 6;
/// The live recipe's --mtp-confidence (Python's CONFIDENCE default is 0.7).
pub const default_confidence: f64 = 0.6;
/// The served hybrid's per-draft floor above `product_streams` live streams (the old recipe engine's wide_confidence).
pub const wide_confidence: f64 = 0.5;

/// A chain's stop rule from a `confidence` value: c > 0, each later draft only at or above c; c < 0, a later draft j only while
/// p1 * ... * pj >= -c (the running product: it keeps going on confident heads and stops early on lukewarm ones, whatever the
/// depth); 0, every draft to the depth. The first draft is always proposed. Drafts only: the kept tokens are the target's either way.
pub const Stop = struct {
    c: f64,
    product: f64 = 1.0,

    /// Whether draft j (0-based) with probability `p` is proposed, and whether the chain may go on.
    pub fn take(s: *Stop, j: usize, p: f64) struct { propose: bool, more: bool } {
        if (s.c > 0) {
            const low = p < s.c;
            if (low and j > 0) return .{ .propose = false, .more = false };
            return .{ .propose = true, .more = !low };
        }
        if (s.c < 0) {
            s.product *= p;
            // the first draft always goes; below the line it ends the chain (no later product can reach it)
            if (s.product < -s.c) return .{ .propose = j == 0, .more = false };
        }
        return .{ .propose = true, .more = true };
    }
};

/// The stop rule for a step drafting `drafting` streams together: a running product (c < 0) only up to `product_streams` of them,
/// the per-draft `wide_confidence` above.
pub fn confidenceFor(c: f64, drafting: usize, product_streams: usize) f64 {
    if (c >= 0) return c;
    return if (drafting <= product_streams) c else wide_confidence;
}

test "the product rule serves up to product_streams streams, the per-draft floor above" {
    try std.testing.expectEqual(@as(f64, 0.6), confidenceFor(0.6, 9, 5));
    try std.testing.expectEqual(@as(f64, -0.4), confidenceFor(-0.4, 1, 5));
    try std.testing.expectEqual(@as(f64, -0.4), confidenceFor(-0.4, 5, 5));
    try std.testing.expectEqual(wide_confidence, confidenceFor(-0.4, 6, 5));
    try std.testing.expectEqual(wide_confidence, confidenceFor(-0.4, 3, 2));
    try std.testing.expectEqual(@as(f64, 0), confidenceFor(0, 9, 1));
}

test "the stop rules: per draft, running product, none" {
    var a: Stop = .{ .c = 0.7 };
    try std.testing.expect(a.take(0, 0.5).propose and !a.take(0, 0.5).more);
    try std.testing.expect(!a.take(1, 0.5).propose);
    var low: Stop = .{ .c = -0.4 };
    const first = low.take(0, 0.3); // a weak first draft: proposed, and the chain ends there
    try std.testing.expect(first.propose and !first.more);
    var b: Stop = .{ .c = -0.4 };
    try std.testing.expect(b.take(0, 0.9).more); // 0.9
    try std.testing.expect(b.take(1, 0.8).propose); // 0.72
    try std.testing.expect(b.take(2, 0.6).propose); // 0.432
    try std.testing.expect(!b.take(3, 0.9).propose); // 0.389
    var c: Stop = .{ .c = 0 };
    try std.testing.expect(c.take(5, 0.01).propose);
}

/// A verify window's rows at most: the pending token and a chain, or a copied continuation.
pub const max_rows: usize = 16;
/// qmm.split_for(2560, 2560): fc_embedding and fc_hidden sum K in four slices, in order.
pub const fc_sk: usize = 4;
/// Head rows one launch takes at most (a prompt's rows go through in pieces).
pub const piece_rows: usize = 256;
/// Normed-stream partial sums a row (cuda_batch.zig's pss_n).
pub const pss_n = (dims / 256) * streams;
pub const conv_bytes = (gdn.taps - 1) * gdn.conv_dim * 2;
pub const proj_bytes = gdn.proj_width * 2;

const core = @import("core");
const proposal = core.proposal_vocab;

fn cint(v: usize) c_int {
    return @intCast(v);
}

/// The draft vocabulary's ids, sorted (Python draft_token_ids("default")): column j of the draft head is ids[j].
pub fn draftIds(gpa: std.mem.Allocator) ![]u32 {
    return proposal.legacy.ids(gpa, @import("proposal_table.zig").text, .{ .vocab = vocab, .skip_past_vocab = true, .delimiters = " \t\r\n," });
}

// ---- sampling ------------------------------------------------------------------------------------------------------

pub fn promote(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

/// A row's first maximum (argmax's choice): its column.
pub fn argmaxRow(row: []const u16) usize {
    var best = promote(row[0]);
    var at: usize = 0;
    for (row[1..], 1..) |b, i| {
        const v = promote(b);
        if (v > best) {
            best = v;
            at = i;
        }
    }
    return at;
}

/// Structured output: every token whose bit is clear in `bits` (token i in word i / 32) scores bf16 -inf.
pub fn maskRow(row: []u16, bits: []const u32) void {
    for (row, 0..) |*v, i| {
        if (i / 32 >= bits.len or (bits[i / 32] >> @intCast(i % 32)) & 1 == 0) v.* = 0xFF80;
    }
}

/// The keyed draw over one bf16 score row, column j being token ids[j] (or j), as lanes.sampling.choose draws.
pub fn drawRow(gpa: std.mem.Allocator, row: []const u16, ids: ?[]const u32, position: u64, sm: ?Sampling) !u32 {
    return drawRowWith(gpa, row, ids, position, sm, false);
}

/// Inserts column `col` (score `v`) into the `held` best so far, best first, an equal score ranking below the earlier column.
fn insertTop(top_v: []f32, top_c: []usize, held: *usize, v: f32, col: usize) void {
    const k = top_v.len;
    if (held.* == k and !(v > top_v[k - 1])) return;
    var at: usize = if (held.* < k) held.* else k - 1;
    if (held.* < k) held.* += 1;
    while (at > 0 and v > top_v[at - 1]) : (at -= 1) {
        top_v[at] = top_v[at - 1];
        top_c[at] = top_c[at - 1];
    }
    top_v[at] = v;
    top_c[at] = col;
}

/// The k best columns of `row`, best first (`top_v`, `top_c` of length k), by one pass over the row.
fn scanTop(row: []const u16, top_v: []f32, top_c: []usize) void {
    var held: usize = 0;
    for (row, 0..) |b, col| insertTop(top_v, top_c, &held, promote(b), col);
}

/// scanTop with sixteen columns tested at once against the k-th best so far: a column that does not beat it is skipped by
/// insertTop too (a NaN beats nothing in either), so the held columns, their order and their scores are scanTop's bit for bit.
fn scanTopWide(row: []const u16, top_v: []f32, top_c: []usize) void {
    const lanes_n = 16;
    const k = top_v.len;
    var held: usize = 0;
    var col: usize = 0;
    while (col < row.len and held < k) : (col += 1) insertTop(top_v, top_c, &held, promote(row[col]), col);
    while (col + lanes_n <= row.len) : (col += lanes_n) {
        const w: @Vector(lanes_n, u16) = row[col..][0..lanes_n].*;
        const f: @Vector(lanes_n, f32) = @bitCast(@as(@Vector(lanes_n, u32), w) << @splat(16));
        const thr: @Vector(lanes_n, f32) = @splat(top_v[k - 1]);
        if (!@reduce(.Or, f > thr)) continue;
        for (row[col..][0..lanes_n], col..) |b, c| insertTop(top_v, top_c, &held, promote(b), c);
    }
    while (col < row.len) : (col += 1) insertTop(top_v, top_c, &held, promote(row[col]), col);
}

/// drawRow, the k best columns found by scanTopWide when `wide` (the same draw, found faster).
pub fn drawRowWith(gpa: std.mem.Allocator, row: []const u16, ids: ?[]const u32, position: u64, sm: ?Sampling, wide_scan: bool) !u32 {
    const id = struct {
        fn of(map: ?[]const u32, col: usize) u64 {
            return if (map) |m| m[col] else col;
        }
    }.of;
    const s = sm orelse return @intCast(id(ids, argmaxRow(row)));
    if (s.temperature <= 0) return @intCast(id(ids, argmaxRow(row)));
    const n = row.len;
    const k = if (s.top_k == 0 or s.top_k >= n) n else @as(usize, s.top_k);
    const vals = try gpa.alloc(f64, k);
    defer gpa.free(vals);
    const keys = try gpa.alloc(u64, k);
    defer gpa.free(keys);
    if (k == n) {
        for (vals, keys, row, 0..) |*v, *key, b, col| {
            v.* = promote(b);
            key.* = id(ids, col);
        }
    } else {
        // the k best columns, best first; a later column with an equal value ranks below (the id order)
        const top_v = try gpa.alloc(f32, k);
        defer gpa.free(top_v);
        const top_c = try gpa.alloc(usize, k);
        defer gpa.free(top_c);
        if (wide_scan) scanTopWide(row, top_v, top_c) else scanTop(row, top_v, top_c);
        for (vals, keys, top_v, top_c) |*v, *key, tv, c| {
            v.* = tv;
            key.* = id(ids, c);
        }
    }
    return @intCast(try lanes.sampling.choose(gpa, vals, keys, position, s));
}

/// The draft head's keyed draw at `position` over its bf16 row (column j being token ids[j]) and the draw's temperature-1
/// probability under the head, `lse` the row's log-sum-exp (Python Engine.sample_draft). A greedy `sm` takes the first best
/// column. The lone head step (Ops.pick) and the batched one (Ops.draftMany) both draw through here, so a stream's drafts
/// do not depend on whether it drafted alone.
pub fn drawPick(gpa: std.mem.Allocator, row: []const u16, ids: []const u32, lse: f32, position: u64, sm: ?Sampling) !Pick {
    return drawPickWith(gpa, row, ids, lse, position, sm, false);
}

/// drawPick over drawRowWith.
pub fn drawPickWith(gpa: std.mem.Allocator, row: []const u16, ids: []const u32, lse: f32, position: u64, sm: ?Sampling, wide_scan: bool) !Pick {
    const tok = try drawRowWith(gpa, row, ids, position, sm, wide_scan);
    const col = std.sort.binarySearch(u32, ids, tok, struct {
        fn order(t: u32, x: u32) std.math.Order {
            return std.math.order(t, x);
        }
    }.order) orelse return .{ .token = tok, .prob = 0 };
    return .{ .token = tok, .prob = @exp(@as(f64, promote(row[col])) - @as(f64, lse)) };
}

/// Whether `sm` draws keyed samples (a temperature above zero) rather than taking the best column.
pub fn isSampled(sm: ?Sampling) bool {
    return if (sm) |s| s.temperature > 0 else false;
}

// ---- a row's first candidates from the GPU -------------------------------------------------------------------------------

/// Candidates a row's draw reads at most (a request's top_k; a larger one takes the whole row).
pub const cand_max: usize = 64;

/// TF_FLASHNEXT_SAMPLE_CAND=1: a sampled draw (a verify window's rows, the draft head's row) reads its row's top_k best
/// scores and columns, selected on the device (fn_row_cand), instead of the whole row copied to the host and scanned. The
/// candidates are the set `drawRow` keeps, so the draws are the same bits. Unset or any other value: the whole row.
pub fn candidatesOn() bool {
    const v = std.c.getenv("TF_FLASHNEXT_SAMPLE_CAND") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}

/// How many candidates a draw of `sm` over a row of `n` scores needs, null when it must read the row whole (greedy; no top_k;
/// a top_k as wide as the row or past cand_max).
pub fn candidateCount(sm: ?Sampling, n: usize) ?usize {
    const s = sm orelse return null;
    if (s.temperature <= 0 or s.top_k == 0 or s.top_k >= n or s.top_k > cand_max) return null;
    return s.top_k;
}

/// Rows of scratch for fn_row_cand's outputs: a shared pass's windows side by side (up to 32 of 16 rows would not fit; a pass is
/// 64 rows at most), or the lanes of a batched head step.
pub const cand_slots: usize = 64;
/// Device scratch for fn_row_cand's outputs: vals [cand_slots, cand_max] f32, cols [cand_slots, cand_max] i32, flags [cand_slots] i32.
pub const cand_bytes: usize = 2 * cand_slots * cand_max * 4 + cand_slots * 4;

/// What `Ops.candidates` copies back: slot s holds vals / cols [s * cand_max ..] and flags[s] (non-zero: read that row whole).
pub const Cands = struct {
    vals: [cand_slots * cand_max]f32 = undefined,
    cols: [cand_slots * cand_max]i32 = undefined,
    flags: [cand_slots]i32 = undefined,
};

comptime {
    // the device scratch and its host copy are one layout: a single download fills a Cands
    std.debug.assert(@sizeOf(Cands) == cand_bytes);
}

/// A bf16 score's 16-bit sort key (fn_row_cand's): larger scores larger keys, +0 and -0 one key (NaN has none).
pub fn candidateKey(bits: u16) u16 {
    const u: u32 = if (bits & 0x7fff == 0) 0 else bits;
    return @intCast(if (u & 0x8000 != 0) ~u & 0xffff else u | 0x8000);
}

/// Whether a bf16 score is a NaN (a row with one has no total order: the host reads it whole).
pub fn candidateNan(bits: u16) bool {
    return bits & 0x7fff > 0x7f80;
}

/// The row's first `count` scores and their columns in the draw's order (the larger score first, of equal scores the lower
/// column; +0 and -0 equal), as fn_row_cand selects them: the key of the count-th largest by two 8-bit histogram passes, the
/// keys above it, then the lowest columns among those equal to it. `vals` and `cols` take them in any order (the kernel's
/// atomics fix none; callers sort). False for a row with a NaN (nothing written), else the row's candidates are in place.
pub fn topCandidates(row: []const u16, count: usize, vals: []f32, cols: []i32) bool {
    const k = @min(count, row.len);
    var hist: [256]u32 = @splat(0);
    for (row) |b| {
        if (candidateNan(b)) return false;
        hist[candidateKey(b) >> 8] += 1;
    }
    var left: u32 = @intCast(k);
    var hi: usize = 255;
    var above: u32 = 0;
    while (hi > 0 and above + hist[hi] < left) : (hi -= 1) above += hist[hi];
    left -= above;
    hist = @splat(0);
    for (row) |b| if (candidateKey(b) >> 8 == hi) {
        hist[candidateKey(b) & 255] += 1;
    };
    var lo: usize = 255;
    above = 0;
    while (lo > 0 and above + hist[lo] < left) : (lo -= 1) above += hist[lo];
    left -= above; // the keys equal to the cut that are taken: the lowest columns
    const cut: u16 = @intCast(hi << 8 | lo);
    const taken_above = k - left;
    var n_above: usize = 0;
    var n_tie: usize = 0;
    for (row, 0..) |b, col| {
        const key = candidateKey(b);
        if (key > cut) {
            vals[n_above] = promote(b);
            cols[n_above] = @intCast(col);
            n_above += 1;
        } else if (key == cut and n_tie < left) {
            vals[taken_above + n_tie] = promote(b);
            cols[taken_above + n_tie] = @intCast(col);
            n_tie += 1;
        }
    }
    return true;
}

/// drawRow over a row's candidates (`vals` and `cols` from topCandidates / fn_row_cand, `count` = the request's top_k of them,
/// any order): the keyed draw among the top_k best columns, token ids[col] (or col). The set and the key of each are drawRow's,
/// and choose orders them by score and id, so the draw is drawRow's over the whole row.
pub fn drawCandidates(gpa: std.mem.Allocator, vals: []const f32, cols: []const i32, ids: ?[]const u32, position: u64, sm: ?Sampling) !u32 {
    const s = sm orelse return error.NotSampled;
    if (vals.len != cols.len or vals.len == 0 or vals.len > cand_max) return error.BadCandidates;
    var v: [cand_max]f64 = undefined;
    var key: [cand_max]u64 = undefined;
    for (vals, cols, 0..) |x, c, i| {
        v[i] = x;
        key[i] = if (ids) |m| m[@intCast(c)] else @intCast(c);
    }
    return @intCast(try lanes.sampling.choose(gpa, v[0..vals.len], key[0..vals.len], position, s));
}

/// drawPick over a row's candidates: the drawn token's probability from its own candidate score.
pub fn drawPickCandidates(gpa: std.mem.Allocator, vals: []const f32, cols: []const i32, ids: []const u32, lse: f32, position: u64, sm: ?Sampling) !Pick {
    const tok = try drawCandidates(gpa, vals, cols, ids, position, sm);
    for (vals, cols) |x, c| if (ids[@intCast(c)] == tok) return .{ .token = tok, .prob = @exp(@as(f64, x) - @as(f64, lse)) };
    return .{ .token = tok, .prob = 0 };
}

// ---- the head's weights --------------------------------------------------------------------------------------------

pub const Head = struct {
    gpa: std.mem.Allocator,
    norm_e: cuda.DeviceBuffer, // [D] fp32
    norm_h: cuda.DeviceBuffer, // [S D] fp32
    fc_e: Q4,
    fc_h: Q4,
    layer: load.Layer,
    mixer: HcW,
    /// lm_head's rows at `ids`, in their order.
    draft: Q4,
    ids: []const u32,
    artifact: proposal.Artifact,
    service: lanes.proposal_head.Service,
    module: cuda.Module,
    best_fn: cuda.Function,
    /// fn_row_cand and the device scratch of its outputs, with TF_FLASHNEXT_SAMPLE_CAND=1 (else null).
    cand_fn: ?cuda.Function,
    cand_buf: ?cuda.DeviceBuffer,

    pub fn init(gpa: std.mem.Allocator, io: std.Io, driver: *cuda.Driver, stream: *cuda.Stream, dir: []const u8) !*Head {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        const h = try gpa.create(Head);
        errdefer gpa.destroy(h);
        h.gpa = gpa;
        var store = try wt.Store.open(gpa, io, dir);
        defer store.deinit();
        try store.detectNorms();
        var buf: [120]u8 = undefined;
        const ne = try store.normScale(gpa, try store.tensor("language_model.mtp.pre_fc_norm_embedding.weight"));
        defer gpa.free(ne);
        const nh = try store.normScale(gpa, try store.tensor("language_model.mtp.pre_fc_norm_hidden.weight"));
        defer gpa.free(nh);
        if (ne.len != dims or nh.len != wide) return error.UnexpectedTensor;
        h.norm_e = try cuda.DeviceBuffer.fromHost(driver, std.mem.sliceAsBytes(ne));
        errdefer h.norm_e.free();
        h.norm_h = try cuda.DeviceBuffer.fromHost(driver, std.mem.sliceAsBytes(nh));
        errdefer h.norm_h.free();
        h.fc_e = try fc(gpa, driver, &store, &buf, "language_model.mtp.fc_embedding");
        errdefer h.fc_e.free();
        h.fc_h = try fc(gpa, driver, &store, &buf, "language_model.mtp.fc_hidden");
        errdefer h.fc_h.free();
        h.mixer = try wt.loadHc(gpa, driver, &store, "language_model.mtp.hyper_connection_mixer", false);
        errdefer h.mixer.free();
        {
            var pack_mod = try cuda.Module.load(driver, cuda.kernels.experts_pack);
            defer pack_mod.unload();
            const pack_fn = try pack_mod.function(wt.pack_symbol);
            h.layer = try load.loadLayerAt(gpa, driver, stream, &store, pack_fn, "language_model.mtp.layers.0", false, false);
        }
        errdefer h.layer.free(gpa);
        const ids = try draftIds(gpa);
        defer gpa.free(ids);
        const head_load = @import("cuda_mtp_load.zig");
        const role = try head_load.mode();
        const dtype = (try store.tensor("language_model.lm_head.weight")).dtype;
        h.artifact = try proposal.legacy.artifact(gpa, ids, .{ .vocab = vocab, .hidden = dims, .format = if (dtype == .bf16) "bf16" else "affine4", .group = if (dtype == .bf16) 0 else 32, .scale_recipe = if (dtype == .bf16) @tagName(role) else "stored affine4 group32" });
        errdefer h.artifact.deinit();
        h.ids = h.artifact.ids();
        h.service = try lanes.proposal_head.Service.legacy(gpa, &h.artifact);
        h.draft = try head_load.draft(gpa, driver, &store, &h.artifact, role);
        errdefer h.draft.free();
        h.module = try cuda.Module.load(driver, cuda.kernels.flashnext_mtp);
        errdefer h.module.unload();
        h.best_fn = try h.module.function("fn_row_best");
        h.cand_fn = null;
        h.cand_buf = null;
        if (candidatesOn()) {
            h.cand_fn = try h.module.function("fn_row_cand");
            h.cand_buf = try cuda.DeviceBuffer.alloc(driver, cand_bytes);
        }
        return h;
    }

    fn fc(gpa: std.mem.Allocator, driver: *cuda.Driver, store: *wt.Store, buf: []u8, name: []const u8) !Q4 {
        _ = buf;
        return @import("cuda_projection_load.zig").load(gpa, driver, store, name, dims, dims);
    }

    pub fn deinit(h: *Head) void {
        if (h.cand_buf) |*b| b.free();
        h.module.unload();
        h.draft.free();
        h.artifact.deinit();
        h.layer.free(h.gpa);
        h.mixer.free();
        h.fc_h.free();
        h.fc_e.free();
        h.norm_h.free();
        h.norm_e.free();
        h.gpa.destroy(h);
    }

    pub fn view(h: *const Head) core.proposal_rows.Scores {
        const map = h.service.compact.map.?;
        return .{ .vocab = vocab, .logical = @intCast(map.ids.len), .stride = @intCast(map.ids.len), .ids = map.ids };
    }

    /// The BF16 legacy lane is the same qlinear owner; group-32 affine keeps its existing unqualified callback.
    pub fn project(h: *const Head, pass: anytype, input: u64, sums: u64, out: u64, rows: usize) !void {
        if (h.draft.native) |weight| {
            const view_: cuda.qlinear.proposal.Head = .{ .weight = .{ .bf16 = weight }, .role = .decode, .vocab = vocab, .logical = @intCast(h.ids.len), .stride = @intCast(h.ids.len), .ids = h.ids, .generation = h.artifact.fingerprint };
            return view_.project(.{ .lane = &pass.k.native_lane }, pass.stream, .{ .x = input, .sums = sums, .part = pass.s.native_part }, out, rows);
        }
        _ = try pass.lane(input, sums, h.draft, out, pass.s.dummy, rows, 1);
    }

    /// fn_row_cand over `rows` rows of `v` scores `ld` apart: each row's first `count` candidates into scratch slots
    /// [slot, slot + rows) (cand_max entries each), which `candidates` reads back.
    pub fn cand(h: *const Head, stream: cuda.Stream, scores: u64, v: usize, ld: usize, rows: usize, count: usize, slot: usize) !void {
        const f = h.cand_fn orelse return error.NoCandidates;
        const buf = h.cand_buf orelse return error.NoCandidates;
        if (rows == 0 or slot + rows > cand_slots or count == 0 or count > cand_max) return error.BadCandidates;
        var a: cuda.Args = .{};
        a.add(scores);
        a.add(cint(v));
        a.add(@as(c_longlong, @intCast(ld)));
        a.add(cint(count));
        a.add(cint(cand_max));
        a.add(buf.ptr + slot * cand_max * 4);
        a.add(buf.ptr + (cand_slots + slot) * cand_max * 4);
        a.add(buf.ptr + 2 * cand_slots * cand_max * 4 + slot * 4);
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 1024 } }, stream, &a);
    }

    /// fn_row_best over `rows` rows of `v` scores `ld` apart: [rows, 4] fp32 at `out` (best, log-sum-exp, id bits).
    pub fn best(h: *const Head, stream: cuda.Stream, scores: u64, v: usize, ld: usize, rows: usize, out: u64) !void {
        var a: cuda.Args = .{};
        a.add(scores);
        a.add(cint(v));
        a.add(@as(c_longlong, @intCast(ld)));
        a.add(out);
        try cuda.launch.launch(h.best_fn, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 1024 } }, stream, &a);
    }
};

/// Device bytes the head's weights take beyond the main model's (for the server's admission estimate).
pub fn headBytes() usize {
    const experts = (wt.experts_n + 1) * 3 * wt.moe_w * dims / 2 * 5 / 4;
    const attn = (wt.proj_n + dims) * dims / 2 * 5 / 4;
    const draft = 79_591 * dims / 2 * 5 / 4;
    return experts + attn + draft + 2 * dims * dims / 2 * 5 / 4;
}

pub const headFormat = @import("cuda_mtp_draft.zig").headFormat;
pub const Pick = @import("cuda_mtp_draft.zig").Pick;
pub const Chain = @import("cuda_mtp_draft.zig").Chain;
pub const Draft = @import("cuda_mtp_draft.zig").Draft;
pub const Ops = @import("cuda_mtp_ops.zig").Ops;
pub const check = @import("cuda_mtp_check.zig").check;

test {
    _ = @import("cuda_host_draw.zig");
}

test "the draft vocabulary is 79,591 sorted ids under the vocabulary" {
    const ids = try draftIds(std.testing.allocator);
    defer std.testing.allocator.free(ids);
    try std.testing.expectEqual(@as(usize, 79_591), ids.len);
    for (ids[1..], ids[0 .. ids.len - 1]) |b, a| try std.testing.expect(b > a);
}

test "a keyed draw over the top-k columns is the draw over every column" {
    const gpa = std.testing.allocator;
    var row: [300]u16 = undefined;
    var prng = std.Random.DefaultPrng.init(3);
    for (&row) |*v| v.* = @as(u16, 0x3f00) + prng.random().int(u8); // values near 0.5..1, many ties
    const all = struct {
        fn draw(a: std.mem.Allocator, r: []const u16, position: u64, s: Sampling) !u64 {
            const vals = try a.alloc(f64, r.len);
            defer a.free(vals);
            const ids = try a.alloc(u64, r.len);
            defer a.free(ids);
            for (vals, ids, r, 0..) |*v, *id, b, i| {
                v.* = promote(b);
                id.* = i;
            }
            return lanes.sampling.choose(a, vals, ids, position, s);
        }
    }.draw;
    for (0..64) |position| {
        const s: Sampling = .{ .seed = 11, .temperature = 1.0, .top_k = 20, .top_p = 0.95 };
        try std.testing.expectEqual(try all(gpa, &row, position, s), try drawRow(gpa, &row, null, position, s));
    }
    try std.testing.expectEqual(@as(u32, @intCast(argmaxRow(&row))), try drawRow(gpa, &row, null, 0, null));
}

test "a draft pick is the keyed draw over the head's row and that token's probability under it" {
    const gpa = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(9);
    var row: [400]u16 = undefined;
    var ids: [400]u32 = undefined;
    for (&row, &ids, 0..) |*v, *id, col| {
        v.* = @as(u16, 0x3f00) + prng.random().int(u8);
        id.* = @intCast(1000 + 3 * col); // the draft vocabulary: ascending ids, as draftIds gives them
    }
    var top: f32 = -std.math.inf(f32);
    for (row) |b| top = @max(top, promote(b));
    var sum: f64 = 0;
    for (row) |b| sum += @exp(@as(f64, promote(b)) - top);
    const lse: f32 = @floatCast(top + @log(sum));
    const s: Sampling = .{ .seed = 3, .temperature = 1.0, .top_k = 20, .top_p = 0.95 };
    for (0..64) |position| {
        const got = try drawPick(gpa, &row, &ids, lse, position, s);
        try std.testing.expectEqual(try drawRow(gpa, &row, &ids, position, s), got.token);
        const col = std.mem.indexOfScalar(u32, &ids, got.token).?;
        try std.testing.expectApproxEqAbs(@exp(@as(f64, promote(row[col])) - @as(f64, lse)), got.prob, 1e-12);
        try std.testing.expect(got.prob > 0 and got.prob < 1);
    }
    // greedy (no sampling, or temperature 0): the first best column, its probability the head's confidence
    const best = argmaxRow(&row);
    for ([_]?Sampling{ null, .{ .seed = 3, .temperature = 0 } }) |greedy| {
        const got = try drawPick(gpa, &row, &ids, lse, 5, greedy);
        try std.testing.expectEqual(ids[best], got.token);
        try std.testing.expectApproxEqAbs(@exp(@as(f64, promote(row[best])) - @as(f64, lse)), got.prob, 1e-12);
    }
    try std.testing.expect(isSampled(s) and !isSampled(null) and !isSampled(.{ .seed = 1, .temperature = 0 }));
}

/// A bf16 row of `n` scores drawn from a few values so that ties at any cut are common, with both zeros and both infinities.
fn tiedRow(prng: *std.Random.DefaultPrng, row: []u16, levels: u8) void {
    const specials = [_]u16{ 0x0000, 0x8000, 0x7f80, 0xff80, 0x3f80, 0xbf80 };
    for (row) |*v| {
        const r = prng.random().int(u16);
        v.* = if (r % 17 == 0) specials[r / 17 % specials.len] else @as(u16, 0x3f00) + prng.random().uintLessThan(u8, levels);
    }
}

test "a row's candidates are its top_k best columns, ties to the lower column, whatever the order they come in" {
    var prng = std.Random.DefaultPrng.init(21);
    var row: [1500]u16 = undefined;
    var vals: [cand_max]f32 = undefined;
    var cols: [cand_max]i32 = undefined;
    for ([_]u8{ 1, 2, 5, 40, 200 }) |levels| for ([_]usize{ 1, 2, 3, 20, 31, 64 }) |count| {
        for ([_]usize{ 70, 300, 1500 }) |n| {
            tiedRow(&prng, row[0..n], levels);
            try std.testing.expect(topCandidates(row[0..n], count, &vals, &cols));
            // the brute force: every column ranked by score (equal scores by column), the first `count`
            var rank: [1500]u16 = undefined;
            for (rank[0..n], 0..) |*r, i| r.* = @intCast(i);
            std.mem.sort(u16, rank[0..n], row[0..n], struct {
                fn before(r: []const u16, a: u16, b: u16) bool {
                    const x = promote(r[a]);
                    const y = promote(r[b]);
                    return x > y or (x == y and a < b);
                }
            }.before);
            const k = @min(count, n);
            var want: [cand_max]i32 = undefined;
            for (want[0..k], rank[0..k]) |*w, r| w.* = r;
            std.mem.sort(i32, want[0..k], {}, std.sort.asc(i32));
            std.mem.sort(i32, cols[0..k], {}, std.sort.asc(i32));
            try std.testing.expectEqualSlices(i32, want[0..k], cols[0..k]);
            // and each candidate's score is its column's
            _ = topCandidates(row[0..n], count, &vals, &cols);
            for (vals[0..k], cols[0..k]) |x, c| try std.testing.expectEqual(promote(row[@intCast(c)]), x);
        }
    };
    // a NaN anywhere sends the row to the host whole
    row[5] = 0x7fc0;
    try std.testing.expect(!topCandidates(row[0..300], 20, &vals, &cols));
    // keys order the bf16 values: -inf < negatives < -0 = +0 < positives < +inf
    try std.testing.expectEqual(candidateKey(0x0000), candidateKey(0x8000));
    try std.testing.expect(candidateKey(0xff80) < candidateKey(0xbf80) and candidateKey(0xbf80) < candidateKey(0x8000));
    try std.testing.expect(candidateKey(0x8000) < candidateKey(0x0001) and candidateKey(0x3f80) < candidateKey(0x7f80));
    try std.testing.expect(candidateKey(0x8001) < candidateKey(0x0000));
}

test "a keyed draw over the candidates is the draw over the whole row, and a pick's probability the same" {
    const gpa = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(77);
    var row: [900]u16 = undefined;
    var ids: [900]u32 = undefined;
    for (&ids, 0..) |*id, col| id.* = @intCast(5 + 2 * col);
    var vals: [cand_max]f32 = undefined;
    var cols: [cand_max]i32 = undefined;
    for ([_]u8{ 3, 30, 200 }) |levels| {
        tiedRow(&prng, &row, levels);
        const lse: f32 = 7.0; // any finite log-sum-exp: the probabilities are exp(score - lse), infinite scores included
        for ([_]u32{ 1, 2, 20, 40, 64 }) |top_k| for ([_]f64{ 1.0, 0.95, 0.5 }) |top_p| for ([_]f64{ 1.0, 0.7 }) |temperature| {
            const s: Sampling = .{ .seed = 41 + top_k, .temperature = temperature, .top_k = top_k, .top_p = top_p, .min_p = if (top_p == 0.5) 0.05 else 0 };
            const count = candidateCount(s, row.len).?;
            try std.testing.expectEqual(@as(usize, top_k), count);
            try std.testing.expect(topCandidates(&row, count, &vals, &cols));
            for (0..24) |position| {
                try std.testing.expectEqual(try drawRow(gpa, &row, null, position, s), try drawCandidates(gpa, vals[0..count], cols[0..count], null, position, s));
                try std.testing.expectEqual(try drawRow(gpa, &row, &ids, position, s), try drawCandidates(gpa, vals[0..count], cols[0..count], &ids, position, s));
                const whole = try drawPick(gpa, &row, &ids, lse, position, s);
                const cand = try drawPickCandidates(gpa, vals[0..count], cols[0..count], &ids, lse, position, s);
                try std.testing.expectEqual(whole.token, cand.token);
                try std.testing.expectEqual(whole.prob, cand.prob);
            }
        };
    }
    // the draws that read a row whole: greedy, no top_k, a top_k as wide as the row or past the candidates' room
    try std.testing.expectEqual(@as(?usize, null), candidateCount(null, 900));
    try std.testing.expectEqual(@as(?usize, null), candidateCount(.{ .seed = 1, .temperature = 0 }, 900));
    try std.testing.expectEqual(@as(?usize, null), candidateCount(.{ .seed = 1, .top_k = 0 }, 900));
    try std.testing.expectEqual(@as(?usize, null), candidateCount(.{ .seed = 1, .top_k = 65 }, 900));
    try std.testing.expectEqual(@as(?usize, null), candidateCount(.{ .seed = 1, .top_k = 20 }, 20));
    try std.testing.expectEqual(@as(?usize, 20), candidateCount(.{ .seed = 1, .top_k = 20 }, 21));
}

test "the wide scan holds the columns, order and scores of the plain scan, ties and NaNs included" {
    var prng = std.Random.DefaultPrng.init(404);
    var row: [1100]u16 = undefined;
    for ([_]u8{ 1, 2, 5, 40, 200 }) |levels| for ([_]usize{ 1, 2, 3, 20, 31, 64, 300 }) |k| for ([_]usize{ 1, 15, 16, 17, 64, 333, 1100 }) |n| {
        tiedRow(&prng, row[0..n], levels);
        for ([_]bool{ false, true }) |with_nan| {
            if (with_nan and n > 40) row[n / 2] = 0x7fc0 + prng.random().uintLessThan(u16, 8);
            var a_v: [300]f32 = undefined;
            var a_c: [300]usize = undefined;
            var b_v: [300]f32 = undefined;
            var b_c: [300]usize = undefined;
            const kk = @min(k, n);
            @memset(a_v[0..kk], 0);
            @memset(b_v[0..kk], 0);
            scanTop(row[0..n], a_v[0..kk], a_c[0..kk]);
            scanTopWide(row[0..n], b_v[0..kk], b_c[0..kk]);
            const held = @min(kk, n);
            try std.testing.expectEqualSlices(usize, a_c[0..held], b_c[0..held]);
            for (a_v[0..held], b_v[0..held]) |x, y| try std.testing.expectEqual(@as(u32, @bitCast(x)), @as(u32, @bitCast(y)));
        }
    };
}

test "a masked row draws only allowed tokens, greedy the best of them" {
    const gpa = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(5);
    for (0..64) |position| {
        var row: [300]u16 = undefined;
        for (&row) |*v| v.* = @as(u16, 0x3f00) + prng.random().int(u8);
        var bits: [10]u32 = @splat(0); // tokens 0..319: three allowed, the vocabulary's best usually not among them
        const allowed = [_]u32{ 7, @intCast(64 + position), 299 };
        for (allowed) |t| bits[t / 32] |= @as(u32, 1) << @intCast(t % 32);
        var best: u32 = allowed[0];
        for (allowed) |t| if (promote(row[t]) > promote(row[best]) or (promote(row[t]) == promote(row[best]) and t < best)) {
            best = t;
        };
        maskRow(&row, &bits);
        try std.testing.expectEqual(best, try drawRow(gpa, &row, null, position, null));
        const s: Sampling = .{ .seed = 11, .temperature = 1.0, .top_k = 20, .top_p = 0.95 };
        const t = try drawRow(gpa, &row, null, position, s);
        try std.testing.expect(std.mem.indexOfScalar(u32, &allowed, t) != null);
    }
}

test "draft chains fit one verify window while the default stays six" {
    try std.testing.expectEqual(@as(usize, 16), max_chain + 1);
    try std.testing.expectEqual(@as(usize, 6), default_depth);
    try std.testing.expect(default_depth <= max_chain);
}
