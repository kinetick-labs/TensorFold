//! TF_FLASHNEXT_CHUNK's rows a pass: one row to the largest measured pass, and the default when unset.
const std = @import("std");
const prompt = @import("cuda_prompt.zig");
const batch = @import("cuda_batch.zig");

test "TF_FLASHNEXT_CHUNK takes 1 to the largest measured pass and refuses the rest" {
    try std.testing.expectEqual(batch.default_chunk, try prompt.parseChunk(null));
    try std.testing.expectEqual(@as(usize, 4096), try prompt.parseChunk("4096"));
    try std.testing.expectEqual(batch.max_chunk, try prompt.parseChunk("16384"));
    for ([_][]const u8{ "0", "16385", "-1", "4k", "" }) |bad| try std.testing.expectError(error.BadChunk, prompt.parseChunk(bad));
}
