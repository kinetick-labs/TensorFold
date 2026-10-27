//! Flash Next's --kv-cache fp8 at admission and in the memory plan, without a checkpoint, a driver or a GPU.
const std = @import("std");
const flashnext = @import("flashnext");
const refusals = @import("cuda_refusal.zig");
const options = @import("cuda_options.zig");
const memory = @import("cuda_memory.zig");

const F = flashnext.native;
const GiB: u64 = 1 << 30;
/// The MTP head's draft vocabulary (draft_vocab_default.txt) and a pass's rows: only the head's small fixed buffers see them.
const draft_ids = 79591;
const chunk = 1536;
const window = 262144;

fn room(gib: f64) u64 {
    return @intFromFloat(gib * @as(f64, @floatFromInt(GiB)));
}

test "Flash Next takes --kv-cache fp8 at family admission and its options carry the flag" {
    const asked: @import("engine_api").Open = .{ .dir = "d", .model_type = F.model_type, .kv8 = true };
    try std.testing.expect(refusals.family(F, asked) == null);
    var opts: F.Options = .{ .context = 4096, .drafts = true };
    var why: []const u8 = "";
    try std.testing.expect(options.set(F, asked, &opts, &why));
    try std.testing.expect(opts.kv8);
    opts = .{ .context = 4096, .drafts = true };
    try std.testing.expect(options.set(F, .{ .dir = "d", .model_type = F.model_type }, &opts, &why));
    try std.testing.expect(!opts.kv8);
}

test "a stream's bytes at the window: 7.8 GiB in bf16, 4.6 GiB in fp8, the rest the same" {
    const bf = flashnext.lanes_load.streamBytes(.bf16, window, draft_ids, chunk, false);
    const f8 = flashnext.lanes_load.streamBytes(.fp8, window, draft_ids, chunk, false);
    // 13 attention caches (12 layers and the head's) at 262,144 tokens save (2,048 - 1,056) bytes each
    try std.testing.expectEqual(@as(usize, 13 * window * (2048 - 1056)), bf - f8);
    try std.testing.expect(bf > 7 * GiB and bf < 8 * GiB);
    try std.testing.expect(f8 > 4 * GiB and f8 < 5 * GiB);
    // without a head, only the sequence
    try std.testing.expectEqual(flashnext.prompt.seqBytesFor(.fp8, window), flashnext.lanes_load.streamBytes(.fp8, window, null, chunk, false));
    try std.testing.expectEqual(flashnext.prompt.seqBytes(window), flashnext.lanes_load.streamBytes(.bf16, window, null, chunk, false));
}

test "the Spark's 24.36 GiB of sequence memory admits five full-window streams in fp8 where bf16 admits three" {
    const bf = flashnext.lanes_load.streamBytes(.bf16, window, draft_ids, chunk, false);
    const f8 = flashnext.lanes_load.streamBytes(.fp8, window, draft_ids, chunk, false);
    try std.testing.expectEqual(@as(u32, 3), try memory.admit(room(24.36), bf, 0, 16, false));
    try std.testing.expectEqual(@as(u32, 5), try memory.admit(room(24.36), f8, 0, 16, false));
    // five at once is refused in bf16 as a fixed count, and admitted in fp8
    try std.testing.expectError(error.TooMany, memory.admit(room(24.36), bf, 0, 5, true));
    try std.testing.expectEqual(@as(u32, 5), try memory.admit(room(24.36), f8, 0, 5, true));
    // streams a GiB: fp8 never admits fewer, and from 10 GiB of room it admits more
    var gib: f64 = 5;
    while (gib <= 100) : (gib += 0.5) {
        const a = memory.admit(room(gib), bf, 0, 64, false) catch 0;
        const b = memory.admit(room(gib), f8, 0, 64, false) catch 0;
        try std.testing.expect(b >= a);
        try std.testing.expect(@as(u64, b) * f8 <= room(gib) or b == 0);
        if (gib >= 10) try std.testing.expect(b > a);
    }
}

test "the prompt cache keeps about 1.6 times the states in fp8" {
    // 16 GiB of kept states of 32,768 tokens, each with the head's rows
    const at = 32768;
    const bf = flashnext.snap.bytesFor(.bf16, at, true);
    const f8 = flashnext.snap.bytesFor(.fp8, at, true);
    try std.testing.expectEqual(flashnext.snap.bytes(at, true), bf);
    try std.testing.expectEqual(@as(usize, 13 * at * (2048 - 1056)), bf - f8);
    const kept_bf = 16 * GiB / bf;
    const kept_f8 = 16 * GiB / f8;
    try std.testing.expect(kept_f8 * 10 >= kept_bf * 15);
}
