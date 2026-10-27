//! A shared round's greedy streams drafted in one head pass a step (cuda_lanes.zig's draftFn), and the depth log.
const std = @import("std");
const lanes = @import("lanes");
const mtp = @import("cuda_mtp.zig");
const mtp_ops = @import("cuda_mtp_ops.zig");
const ple = @import("cuda_ple.zig");
const be = lanes.backend;

/// TF_FLASHNEXT_BATCH_HEADS=0 drafts a shared round's streams one at a time (the batched head gives the same drafts).
pub fn batchHeads() bool {
    const v = std.c.getenv("TF_FLASHNEXT_BATCH_HEADS") orelse return true;
    return !std.mem.eql(u8, std.mem.span(v), "0");
}

/// TF_FLASHNEXT_SAMPLED_HEADS=1: a shared round's sampled streams (temperature above zero) join the batched head pass too,
/// each drawing its drafts with its own key on the host from its row of the pass (the draws a lone head step makes), instead of
/// drafting one stream at a time. Off by default: the drafts are only proposals, so replies do not depend on it.
pub fn sampledHeads() bool {
    const v = std.c.getenv("TF_FLASHNEXT_SAMPLED_HEADS") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}

/// TF_FLASHNEXT_DEPTH_LOG=1: every 200 draft calls, the drafts a stream got by round width (streams drafting together).
var depth_hist: [5][max_hist]u64 = @splat(@splat(0));
var depth_calls: u64 = 0;
const max_hist = 17;

/// Greedy requests (and sampled ones with TF_FLASHNEXT_SAMPLED_HEADS=1; no trees, no media) drafted together: draftFn's rollbacks, one head pass a step; `done` marks them.
pub fn batched(self: anytype, requests: []const be.DraftRequest, done: []bool) !void {
    var ds: [mtp_ops.max_lanes]*mtp.Draft = undefined;
    var nexts: [mtp_ops.max_lanes][]const u32 = undefined;
    var firsts: [mtp_ops.max_lanes][1]u32 = undefined;
    var rows: [mtp_ops.max_lanes]u64 = undefined;
    var positions: [mtp_ops.max_lanes]u64 = undefined;
    var sms: [mtp_ops.max_lanes]?mtp.Sampling = undefined;
    var counts: [mtp_ops.max_lanes]usize = undefined;
    var outs: [mtp_ops.max_lanes]*mtp.Chain = undefined;
    var which: [mtp_ops.max_lanes]usize = undefined;
    var n: usize = 0;
    // TF_FLASHNEXT_DRAFT_AHEAD=1: each token's n-gram rows are asked for as the head picks it
    var ahead_buf = ple.Ahead.init(&self.engine.ngram);
    const ahead: ?*ple.Ahead = if (ahead_buf) |*a| a else null;
    for (requests, 0..) |r, ri| {
        if (r.lanes != null) continue;
        if (mtp.isSampled(r.stream.sampling) and !sampledHeads()) continue;
        const l = try self.lane(r.stream);
        const d = if (l.draft) |*x| x else continue;
        if (d.media != null) continue;
        if (r.rows) |path| {
            for (path, 0..) |row, i| if (row != i) return error.TreesNotBuilt;
            if (r.follow.len < path.len) return error.PositionMismatch;
        }
        if (r.rows) |path| try self.ops().rollback(&l.seq, d, path.len) else {
            if (d.open) try self.ops().rollback(&l.seq, d, d.rows);
        }
        if (r.position != l.seq.fed + 1) return error.PositionMismatch;
        const room = l.seq.cap - l.seq.fed - 1;
        counts[n] = @min(@min(r.depth, self.depth), room);
        if (r.rows) |path| {
            nexts[n] = r.follow[0..path.len];
            rows[n] = d.streams.ptr;
        } else {
            firsts[n] = .{self.value(r.first orelse return error.NoFirstToken)};
            nexts[n] = &firsts[n];
            rows[n] = d.prompt_last;
        }
        if (ahead) |a| {
            a.lane(n, l.seq.history[0..]);
            a.push(n, nexts[n][nexts[n].len - 1]);
        }
        ds[n] = d;
        positions[n] = r.position;
        sms[n] = r.stream.sampling;
        outs[n] = &l.chain;
        which[n] = ri;
        n += 1;
    }
    if (n < 2) {
        // one greedy stream: it drafts alone below (its rollback is idempotent)
        return;
    }
    try self.ops().draftMany(ds[0..n], nexts[0..n], rows[0..n], counts[0..n], self.confidenceNow(requests.len), outs[0..n], positions[0..n], sms[0..n], ahead);
    for (which[0..n]) |ri| done[ri] = true;
}

/// TF_FLASHNEXT_DEPTH_LOG=1: each stream's chain length by how many streams drafted together.
pub fn depthLog(self: anytype, requests: []const be.DraftRequest) void {
    const v = std.c.getenv("TF_FLASHNEXT_DEPTH_LOG") orelse return;
    if (!std.mem.eql(u8, std.mem.span(v), "1")) return;
    const w = @min(requests.len, 5) - 1;
    for (requests) |r| {
        const l = self.lane(r.stream) catch continue;
        depth_hist[w][@min(l.chain.n, max_hist - 1)] += 1;
    }
    depth_calls += 1;
    if (depth_calls % 200 != 0) return;
    for (depth_hist, 1..) |row, width| {
        var any: u64 = 0;
        for (row) |c| any += c;
        if (any == 0) continue;
        std.debug.print("DEPTH width={d}{s}", .{ width, if (width == 5) "+" else "" });
        for (row, 0..) |c, d| if (c > 0) std.debug.print(" {d}:{d}", .{ d, c });
        std.debug.print("\n", .{});
    }
}

test "sampled streams join the batched head only with TF_FLASHNEXT_SAMPLED_HEADS=1" {
    const env = @import("sys").env;
    defer _ = env.unsetenv("TF_FLASHNEXT_SAMPLED_HEADS");
    _ = env.unsetenv("TF_FLASHNEXT_SAMPLED_HEADS");
    try std.testing.expect(!sampledHeads());
    for ([_][*:0]const u8{ "0", "", "true" }) |v| {
        _ = env.setenv("TF_FLASHNEXT_SAMPLED_HEADS", v, 1);
        try std.testing.expect(!sampledHeads());
    }
    _ = env.setenv("TF_FLASHNEXT_SAMPLED_HEADS", "1", 1);
    try std.testing.expect(sampledHeads());
}
