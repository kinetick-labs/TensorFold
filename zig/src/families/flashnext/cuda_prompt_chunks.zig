//! Where a prompt's chunks end.
const std = @import("std");

/// The end of the prompt chunk that starts at row `at`: `chunk` rows, cut at the next cache mark. With `tail` set
/// (TF_FLASHNEXT_PREFILL_TAIL), a last chunk that would be under `tail` rows takes half of the one before it instead: two
/// chunks of about equal rows (each past the 1,024 the prompt expert tiles need) in the scratch a full chunk already sizes,
/// where the old engine made one chunk of both (a larger scratch).
pub fn planStop(chunk: usize, total: usize, at: usize, marks: []const u32, tail: usize) usize {
    var stop = @min(total, at + chunk);
    if (tail > 0 and stop < total and total - stop < tail) {
        const half = (total - at + 1) / 2;
        const even = (half + 15) / 16 * 16;
        stop = at + @min(even, chunk);
    }
    for (marks) |m| if (m > at and m < stop) {
        stop = m;
        break;
    };
    return stop;
}

fn plan(chunk: usize, total: usize, marks: []const u32, tail: usize, out: []usize) []usize {
    var at: usize = 0;
    var n: usize = 0;
    while (at < total) : (n += 1) {
        const stop = planStop(chunk, total, at, marks, tail);
        out[n] = stop - at;
        at = stop;
    }
    return out[0..n];
}

test "the tail rule evens out a short last chunk and leaves the rest alone" {
    var buf: [16]usize = undefined;
    try std.testing.expectEqualSlices(usize, &.{ 3072, 3072, 3072, 3072, 193 }, plan(3072, 12481, &.{}, 0, &buf));
    try std.testing.expectEqualSlices(usize, &.{ 3072, 3072, 3072, 1648, 1617 }, plan(3072, 12481, &.{}, 512, &buf));
    try std.testing.expectEqualSlices(usize, &.{ 3072, 3072, 3072, 3072, 600 }, plan(3072, 12888, &.{}, 512, &buf));
    try std.testing.expectEqualSlices(usize, &.{ 3072, 3072 }, plan(3072, 6144, &.{}, 512, &buf));
    try std.testing.expectEqualSlices(usize, &.{100}, plan(3072, 100, &.{}, 512, &buf));
    try std.testing.expectEqualSlices(usize, &.{ 3072, 1584, 1584 }, plan(3072, 6240, &.{}, 512, &buf));
    // a cache mark still cuts a chunk, and the chunks after it are planned afresh
    try std.testing.expectEqualSlices(usize, &.{ 1000, 3072, 1616, 1612 }, plan(3072, 7300, &.{1000}, 512, &buf));
}
