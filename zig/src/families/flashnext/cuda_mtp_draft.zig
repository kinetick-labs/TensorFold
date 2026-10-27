//! A drafting stream's state: the head's cache, a window's buffers and what a partly kept window rolls back to.

const std = @import("std");
const cuda = @import("cuda");
const wt = @import("cuda_prompt_wt.zig");
const batch = @import("cuda_batch.zig");
const prompt = @import("cuda_prompt.zig");
const gdn = @import("cuda_gdn.zig");
const mtp = @import("cuda_mtp.zig");
const kv_format = @import("cuda_kv.zig");

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
const max_chain = mtp.max_chain;
const max_rows = mtp.max_rows;
const conv_bytes = mtp.conv_bytes;
const proj_bytes = mtp.proj_bytes;

/// TF_FLASHNEXT_KEEP_LATE=1: a window kept in part does not replay its kept rows on the DeltaNet states. The rows' k, v and gates (the
/// front's, as the verify pass computed them) wait in the lane's log, and the next window's chain takes them first, in the same
/// pass over the states (cuda_gdn's fn_gdn_rows_late); a window kept whole settles on its end states and empties the log.
pub fn lateOn() bool {
    if (Draft.late_force) |on| return on;
    const v = std.c.getenv("TF_FLASHNEXT_KEEP_LATE") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}

/// Rows a lane's log holds (a window's rows go after the waiting ones) and the most that may wait after a rollback (a window's widest).
pub const late_cap: usize = 2 * max_rows;
pub const late_max_rows: usize = max_rows;
const late_layer_bytes: usize = late_cap * (gdn.log_row_k + gdn.log_row_v + gdn.log_row_gb);

/// The format of the MTP head's own attention cache for a stream of `cap` positions when the engine's main caches are `main`
/// and a stream is charged its `window`'s bytes with the head's cache in `main` too (cuda_lanes_load.zig streamBytes). The head
/// is one layer and its drafts are only proposals (the main model verifies every token), but fp8 rows there cost it about a
/// fifth of its accepted drafts (fewer, less confident drafts). A bf16 head cache takes (2,048 - 1,056) bytes more a position,
/// so it is used whenever the stream still fits what it is charged: all but a stream within a twentieth of the full window.
/// TF_FLASHNEXT_HEAD_KV=fp8 keeps the head's cache in the main format.
pub fn headFormat(main: kv_format.Kv, cap: usize, window: usize, draft_n: usize, rows: usize) kv_format.Kv {
    if (main == .bf16) return .bf16;
    if (std.c.getenv("TF_FLASHNEXT_HEAD_KV")) |v| if (std.mem.eql(u8, std.mem.span(v), "fp8")) return main;
    const used = prompt.seqBytesFor(main, cap) + Draft.bytesFor(.bf16, cap, draft_n, rows);
    const charged = prompt.seqBytesFor(main, window) + Draft.bytesFor(main, window, draft_n, rows);
    return if (used <= charged) .bf16 else main;
}

/// One draft and its temperature-1 probability under the head.
pub const Pick = struct { token: u32, prob: f64 };

/// A chain of drafts the head holds for the next window.
pub const Chain = struct {
    tokens: [max_chain]u32 = undefined,
    probs: [max_chain]f64 = undefined,
    n: usize = 0,

    pub fn slice(c: *const Chain) []const u32 {
        return c.tokens[0..c.n];
    }
};

/// A sequence's head cache and the buffers a window needs: its rows' scores and streams, and what it rolls back to.
pub const Draft = struct {
    stream: ?cuda.Stream = null,
    media: ?@import("cuda_media.zig").View = null,
    gpa: std.mem.Allocator,
    kc: cuda.DeviceBuffer,
    vc: cuda.DeviceBuffer,
    ikc: cuda.DeviceBuffer,
    /// The format of kc and vc (the engine's --kv-cache).
    kv: kv_format.Kv = .bf16,
    /// Its pooled indexer keys [cap / 4, 128] bf16, for sparse attention past 2,048 keys (Python State.mtp_pooled).
    pooled: cuda.DeviceBuffer,
    /// Entries the head's cache holds (Python mtp_len), the last `drafted` of them a chain's.
    len: usize = 0,
    drafted: usize = 0,
    cap: usize,
    /// The last pass's rows of main-model streams [rows_cap, wide]: a prompt chunk's, then each window's.
    streams: cuda.DeviceBuffer,
    rows_cap: usize,
    /// The prompt's last row of streams in `streams`, which the first draft reads with the first token.
    prompt_last: u64 = 0,
    /// The head's last output row (its streams after the MoE write-back): the next chained draft's input.
    out: cuda.DeviceBuffer,
    /// A window's scores [max_rows, vocab], the draft head's [ids], best values [max_rows, 4].
    logits: cuda.DeviceBuffer,
    dlogits: cuda.DeviceBuffer,
    best: cuda.DeviceBuffer,
    /// Before a window: each DeltaNet layer's conv window and (from the pass) projection rows; the n-gram tail.
    saved: cuda.DeviceBuffer,
    /// A window's DeltaNet end states: swapped in when every row is kept, else the kept rows replay on the sequence's.
    spare: [linear_n]cuda.DeviceBuffer = undefined,
    /// A window ran with the states written to `spare` and is not yet kept or rolled back.
    open: bool = false,
    keep: batch.Keep = .{},
    /// With TF_FLASHNEXT_KEEP_LATE: the DeltaNet layers' logs, and the rows waiting in them for the states (also in `keep.pend`).
    log: ?cuda.DeviceBuffer = null,
    pend: usize = 0,
    /// The sequence's conv windows' addresses on the device (fn_conv_roll), and the first's they were made for.
    conv_table: ?cuda.DeviceBuffer = null,
    conv_first: u64 = 0,
    history: [2]i64 = .{ 0, 0 },
    unused_history: [2]i64 = .{ 0, 0 },
    fed0: usize = 0,
    rows: usize = 0,
    ids: [max_rows]u32 = undefined,
    /// The scores a sampled draw reads on the host (a window's rows, or the draft head's row).
    host: []u16,
    best_host: [max_rows * 4]f32 = undefined,

    pub const late_max = late_max_rows;
    /// A test's choice for the Drafts it makes next (null: the environment's).
    pub var late_force: ?bool = null;
    pub const hist_at: usize = 0;
    pub const proj_at: usize = hist_at + linear_n * conv_bytes;
    pub const tail_at: usize = proj_at + linear_n * max_rows * proj_bytes;
    /// The window's n-gram rows (the pass's ple_nrow), saved for a rollback to shift the kept ones into the tail.
    pub const nrow_at: usize = tail_at + ple_tail_n * wide * 2;
    const saved_bytes: usize = nrow_at + max_rows * wide * 2;

    /// Device bytes `init(cap, draft_n, rows)` allocates.
    pub fn bytes(cap: usize, draft_n: usize, rows: usize) usize {
        return bytesFor(.bf16, cap, draft_n, rows);
    }

    /// bytes with the head's cache in format `format`.
    pub fn bytesFor(format: kv_format.Kv, cap: usize, draft_n: usize, rows: usize) usize {
        return cap * format.position() + prompt.pooledBytes(cap) + @max(rows, max_rows) * wide * 2 + wide * 2 + max_rows * vocab * 2 + draft_n * 2 + max_rows * 16 + saved_bytes + linear_n * state_n * 4 + (if (lateOn()) linear_n * late_layer_bytes else 0);
    }

    /// A head cache of `cap` entries; `rows` the widest prompt chunk whose streams it keeps.
    pub fn init(gpa: std.mem.Allocator, driver: *cuda.Driver, stream: cuda.Stream, cap: usize, draft_n: usize, rows: usize) !Draft {
        return initFor(.bf16, gpa, driver, stream, cap, draft_n, rows);
    }

    /// init with the head's cache in format `format`.
    pub fn initFor(format: kv_format.Kv, gpa: std.mem.Allocator, driver: *cuda.Driver, stream: cuda.Stream, cap: usize, draft_n: usize, rows: usize) !Draft {
        var d: Draft = .{ .stream = stream, .gpa = gpa, .cap = cap, .kv = format, .rows_cap = @max(rows, max_rows), .kc = undefined, .vc = undefined, .ikc = undefined, .pooled = undefined, .streams = undefined, .out = undefined, .logits = undefined, .dlogits = undefined, .best = undefined, .saved = undefined, .host = undefined };
        d.kc = try cuda.DeviceBuffer.alloc(driver, cap * format.keyRow());
        errdefer d.kc.free();
        d.vc = try cuda.DeviceBuffer.alloc(driver, cap * format.valueRow());
        errdefer d.vc.free();
        d.ikc = try cuda.DeviceBuffer.alloc(driver, cap * index_dim * 2);
        errdefer d.ikc.free();
        d.pooled = try cuda.DeviceBuffer.alloc(driver, prompt.pooledBytes(cap));
        errdefer d.pooled.free();
        d.streams = try cuda.DeviceBuffer.alloc(driver, d.rows_cap * wide * 2);
        errdefer d.streams.free();
        d.out = try cuda.DeviceBuffer.alloc(driver, wide * 2);
        errdefer d.out.free();
        d.logits = try cuda.DeviceBuffer.alloc(driver, max_rows * vocab * 2);
        errdefer d.logits.free();
        d.dlogits = try cuda.DeviceBuffer.alloc(driver, draft_n * 2);
        errdefer d.dlogits.free();
        d.best = try cuda.DeviceBuffer.alloc(driver, max_rows * 16);
        errdefer d.best.free();
        d.saved = try cuda.DeviceBuffer.alloc(driver, saved_bytes);
        errdefer d.saved.free();
        var made: usize = 0;
        errdefer for (d.spare[0..made]) |*b| b.free();
        while (made < linear_n) : (made += 1) d.spare[made] = try cuda.DeviceBuffer.alloc(driver, state_n * 4);
        if (lateOn()) d.log = try cuda.DeviceBuffer.alloc(driver, linear_n * late_layer_bytes);
        errdefer if (d.log) |*b| b.free();
        d.host = try gpa.alloc(u16, @max(max_rows * vocab, draft_n));
        errdefer gpa.free(d.host);
        errdefer cuda.completion.drain(driver, stream, "Flash Next draft fills did not drain before rollback");
        // zeroed on the compute stream, ahead of the passes that read them
        for ([_]cuda.DeviceBuffer{ d.kc, d.vc, d.ikc, d.pooled }) |b| try b.fill8(0, stream.handle);
        var li: usize = 0;
        for (&d.keep.layers, 0..) |*lk, layer| {
            if (!wt.linearLayer(layer)) continue;
            lk.hist = d.saved.ptr + hist_at + li * conv_bytes;
            lk.proj = d.saved.ptr + proj_at + li * max_rows * proj_bytes;
            if (d.log) |b| {
                lk.log_k = b.ptr + li * late_layer_bytes;
                lk.log_v = lk.log_k + late_cap * gdn.log_row_k;
                lk.log_gb = lk.log_v + late_cap * gdn.log_row_v;
            }
            li += 1;
        }
        d.aim();
        return d;
    }

    /// Each DeltaNet layer's window end state into its spare buffer.
    fn aim(d: *Draft) void {
        var li: usize = 0;
        for (&d.keep.layers, 0..) |*lk, layer| {
            if (!wt.linearLayer(layer)) continue;
            lk.state_out = d.spare[li].ptr;
            li += 1;
        }
    }

    /// An open window kept whole: its end states become the sequence's (buffers swapped, nothing copied).
    pub fn settle(d: *Draft, seq: *prompt.Seq) void {
        if (!d.open) return;
        for (seq.state, &d.spare) |*a, *b| std.mem.swap(cuda.DeviceBuffer, a, b);
        d.aim();
        d.open = false;
        // the end states took the waiting rows in
        d.pend = 0;
    }

    pub fn deinit(d: *Draft) void {
        if (d.stream) |stream| cuda.completion.drain(stream.d, stream, "Flash Next draft work did not drain before release");
        d.gpa.free(d.host);
        for (&d.spare) |*b| b.free();
        if (d.log) |*b| b.free();
        if (d.conv_table) |*b| b.free();
        d.saved.free();
        d.best.free();
        d.dlogits.free();
        d.logits.free();
        d.out.free();
        d.streams.free();
        d.pooled.free();
        d.ikc.free();
        d.vc.free();
        d.kc.free();
    }

    /// Drop the head's cache (a new prompt).
    pub fn reset(d: *Draft) void {
        d.len = 0;
        d.drafted = 0;
        d.rows = 0;
        d.pend = 0;
    }
};

test "a head cache in fp8 takes 1,056 + 256 bytes a position, its pooled blocks and window buffers as in bf16" {
    for ([_]usize{ 1, 4, 5, 8192 }) |cap| {
        try std.testing.expectEqual(Draft.bytes(cap, 100, 64), Draft.bytesFor(.bf16, cap, 100, 64));
        try std.testing.expectEqual(cap * (2048 - 1056), Draft.bytesFor(.bf16, cap, 100, 64) - Draft.bytesFor(.fp8, cap, 100, 64));
    }
    try std.testing.expectEqual(@as(usize, 1056 + 256), Draft.bytesFor(.fp8, 6, 1, 1) - Draft.bytesFor(.fp8, 5, 1, 1));
}
