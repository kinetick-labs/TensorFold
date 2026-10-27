//! The lane core's prices for a shared Flash Next round on CUDA, from the environment. Nothing is timed at load on this backend, so
//! without these a shared round is unpriced: every draft the head's confidence rule lets through is verified. With them the core
//! trims a round's drafts to the rows that pay (windows.allocate), and measures each round's time past the forward as its overhead.
const std = @import("std");
const lanes = @import("lanes");

const Cost = lanes.config.Cost;
/// Rows a shared forward holds at most (facts' batch_rows).
pub const max_rows = 64;

var table: [max_rows]Cost = undefined;
var table_len: usize = 0;
var read = false;

/// TF_FLASHNEXT_ROUND_MS: a shared verify's forward ms by its rows. "a+b" is a + b x rows (rows 1 to 64); a comma list gives
/// rows 1, 2, 3 ... (the lane core interpolates between and extends past the last). Unset, or any other text: no prices.
pub fn parse(text: []const u8, out: []Cost) usize {
    const t = std.mem.trim(u8, text, " \t");
    if (t.len == 0) return 0;
    if (std.mem.indexOfScalar(u8, t, ',') == null) {
        const plus = std.mem.indexOfScalar(u8, t, '+') orelse return 0;
        const a = std.fmt.parseFloat(f64, std.mem.trim(u8, t[0..plus], " ")) catch return 0;
        const b = std.fmt.parseFloat(f64, std.mem.trim(u8, t[plus + 1 ..], " ")) catch return 0;
        if (!(a > 0) or !(b >= 0) or !std.math.isFinite(a + b)) return 0;
        for (out, 1..) |*c, rows| c.* = .{ .width = @intCast(rows), .ms = a + b * @as(f64, @floatFromInt(rows)) };
        return out.len;
    }
    var n: usize = 0;
    var it = std.mem.splitScalar(u8, t, ',');
    while (it.next()) |part| {
        if (n == out.len) return 0;
        const ms = std.fmt.parseFloat(f64, std.mem.trim(u8, part, " ")) catch return 0;
        if (!(ms > 0) or !std.math.isFinite(ms)) return 0;
        out[n] = .{ .width = @intCast(n + 1), .ms = ms };
        n += 1;
    }
    return if (n >= 2) n else 0;
}

/// The prices TF_FLASHNEXT_ROUND_MS gives (read once; empty without them). The slice lives for the process.
pub fn sharedCosts() []const Cost {
    if (!read) {
        read = true;
        if (std.c.getenv("TF_FLASHNEXT_ROUND_MS")) |v| {
            table_len = parse(std.mem.span(v), &table);
            if (table_len == 0) std.log.warn("flash next: TF_FLASHNEXT_ROUND_MS={s} is neither \"a+b\" nor a list of at least two positive ms: shared rounds stay unpriced", .{std.mem.span(v)});
        }
    }
    return table[0..table_len];
}

/// TF_FLASHNEXT_DRAFT_ROW_MS: a shared round's ms for each row past one a stream (a draft), apart from the table read at the
/// round's streams (the lane core's A3 split). Needs TF_FLASHNEXT_ROUND_MS; unset or not a positive number: 0, the table by rows.
pub fn draftRowMs() f64 {
    const v = std.c.getenv("TF_FLASHNEXT_DRAFT_ROW_MS") orelse return 0;
    const ms = std.fmt.parseFloat(f64, std.mem.span(v)) catch return 0;
    return if (ms > 0 and std.math.isFinite(ms) and sharedCosts().len > 0) ms else 0;
}

test "round prices parse as a line or a list and refuse anything else" {
    var out: [max_rows]Cost = undefined;
    try std.testing.expectEqual(@as(usize, 64), parse("56+1.5", &out));
    try std.testing.expectEqual(@as(u32, 1), out[0].width);
    try std.testing.expectApproxEqAbs(@as(f64, 57.5), out[0].ms, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 56 + 1.5 * 20), out[19].ms, 1e-12);
    try std.testing.expectEqual(@as(u32, 64), out[63].width);
    try std.testing.expectEqual(@as(usize, 64), parse(" 60 + 0 ", &out));
    try std.testing.expectEqual(@as(usize, 3), parse("60, 61.5 ,63", &out));
    try std.testing.expectEqual(@as(u32, 3), out[2].width);
    try std.testing.expectApproxEqAbs(@as(f64, 61.5), out[1].ms, 1e-12);
    for ([_][]const u8{ "", "  ", "60", "60,", ",60", "60,x", "60,-1", "0,5", "a+b", "56+", "+1", "-4+1", "56+-1", "nan+1", "56+inf", "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65" }) |bad| {
        try std.testing.expectEqual(@as(usize, 0), parse(bad, &out));
    }
}

test "the round prices and the draft row price come from the environment" {
    const env = @import("sys").env;
    defer _ = env.unsetenv("TF_FLASHNEXT_DRAFT_ROW_MS");
    _ = env.unsetenv("TF_FLASHNEXT_DRAFT_ROW_MS");
    // the process reads TF_FLASHNEXT_ROUND_MS once; here it is unset, so there are no prices and no split
    try std.testing.expectEqual(@as(usize, 0), sharedCosts().len);
    try std.testing.expectEqual(@as(f64, 0), draftRowMs());
    _ = env.setenv("TF_FLASHNEXT_DRAFT_ROW_MS", "1.5", 1);
    try std.testing.expectEqual(@as(f64, 0), draftRowMs()); // a split of a table that is not there
}
