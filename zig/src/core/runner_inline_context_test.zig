//! Default inline cells whose reply context means something other than the span's value (2026-11-02 default audit).
const std = @import("std");
const scan = @import("runner_scan.zig");
const t = std.testing;

/// The cell the last text closes after `before` streamed through one scanner, with whether the reply keeps its value.
fn last(a: std.mem.Allocator, before: []const []const u8, text: []const u8) !?bool {
    var s: scan.Scanner = .{ .cfg = .{ .inline_math = true } };
    defer s.deinit(a);
    for (before) |b| _ = try s.feed(a, b, true);
    const e = try s.feed(a, text, true);
    const c = e.closed orelse return null;
    if (c.form != .inline_expr or e.trigger == null) return null;
    return c.own_context;
}

test "runner scan: after a join cue a bare `a + b` keeps the model's value; other spans and later lines do not" {
    const a = t.allocator;
    // live 0220: the model joined digit groups and wrote 1119392
    const joining = [_][]const u8{ "Concatenate: 1119 392 629 836 636\n", "= 1119392629836636? Let's join step:\n" };
    try t.expectEqual(@as(?bool, true), try last(a, &joining, "1119 + 392 ="));
    // live 0216: the cue two lines up, commas grouped
    const two_up = [_][]const u8{ "Result groups: 1143 | 594255903. Concatenate: 1143594255903?\n", "Let's do it:\n" };
    try t.expectEqual(@as(?bool, true), try last(a, &two_up, "1,143 + 594,255,903 ="));
    try t.expectEqual(@as(?bool, false), try last(a, &joining, "1119 + 392*2 ="));
    // a label before the sum (smoke r0-27d7abd723f6 group addition, a digit count) is addition
    try t.expectEqual(@as(?bool, false), try last(a, &joining, "g0: 868,587,520 + 216,486,895 ="));
    try t.expectEqual(@as(?bool, false), try last(a, &joining, "Count digits: 20+19="));
    try t.expectEqual(@as(?bool, false), try last(a, &joining, "1119 - 392 ="));
    const later = [_][]const u8{ "Concatenate them later.\n", "a\n", "b\n", "c\n" };
    try t.expectEqual(@as(?bool, false), try last(a, &later, "1119 + 392 ="));
    try t.expectEqual(@as(?bool, false), try last(a, &.{}, "1119 + 392 ="));
}

test "runner scan: fractions over a common denominator keep the model's numerators; integers still open" {
    const a = t.allocator;
    // live 0022: `-1/2=-420` over the lcm 840, where the engine would write -0.5
    const lcd = [_][]const u8{ "Let's compute exact. Use denominator lcm 840? Terms:\n", "1=840/840\n" };
    try t.expectEqual(@as(?bool, true), try last(a, &lcd, "-1/2="));
    try t.expectEqual(@as(?bool, true), try last(a, &.{"Common LCD is 72.\n"}, "7/12 ="));
    try t.expectEqual(@as(?bool, false), try last(a, &lcd, "6 * 7 ="));
    try t.expectEqual(@as(?bool, false), try last(a, &.{}, "-1/2="));
}

test "runner scan: running sums of place values still open (smoke r0-27d7abd723f6 wrote them right)" {
    const a = t.allocator;
    for ([_][]const u8{ "So quotient 20,282,422? Sum: 20,000,000+200,000=", "Sum 400,000+50,000=", "Sum 100+20+9 =", "Sum: 592+16,905=" }) |line|
        try t.expectEqual(@as(?bool, false), try last(a, &.{}, line));
}
