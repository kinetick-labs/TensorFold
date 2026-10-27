//! TF_FLASHNEXT_KEEP_LATE: a window kept in part leaves its rows in the draft's log for the next window's chain instead of replaying
//! them, and everything the engine sees afterwards (scores, hidden rows, conv windows, n-gram tails, the DeltaNet states once
//! they are taken) is byte for byte what the replaying rollback leaves.
const std = @import("std");
const cuda = @import("cuda");
const flash = @import("flashnext");
const check = @import("check.zig");
const prompt = flash.prompt;
const mtp = flash.mtp;

const lanes_n = 3;
/// Rounds a script runs: lane 0's 25 pairs of consecutive keeps twice over, then 20 rounds keeping one row (the log overflows).
const pairs_rounds = 50;
const rounds = pairs_rounds + 20;

const Side = struct {
    seqs: [lanes_n]prompt.Seq = undefined,
    drafts: [lanes_n]mtp.Draft = undefined,
    made: usize = 0,

    fn deinit(s: *Side, gpa: std.mem.Allocator) void {
        for (s.seqs[0..s.made], s.drafts[0..s.made]) |*q, *d| {
            d.deinit();
            q.deinit(gpa);
        }
    }
};

pub fn run(gpu: check.Gpu, dir: []const u8, ids_path: []const u8) !void {
    const source = try std.Io.Dir.cwd().readFileAlloc(gpu.io, ids_path, gpu.gpa, .limited(1 << 20));
    defer gpu.gpa.free(source);
    var base: std.ArrayList(u32) = .empty;
    defer base.deinit(gpu.gpa);
    var words = std.mem.tokenizeAny(u8, source, " \r\n\t");
    while (words.next()) |word| try base.append(gpu.gpa, try std.fmt.parseInt(u32, word, 10));
    if (base.items.len < 16) return error.BadPrompt;
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    var glue = try flash.Glue.load(gpu.d);
    defer glue.deinit();
    const tri: flash.Tri = .{ .s = stream, .own = &glue };
    const e = try prompt.Engine.init(gpu.gpa, gpu.io, @constCast(gpu.d), &stream, dir);
    defer e.deinit();
    const ops: mtp.Ops(flash.Tri) = .{ .e = e, .tri = tri, .h = undefined };
    var shared = try cuda.DeviceBuffer.alloc(gpu.d, 64 * prompt.vocabulary * 2);
    defer shared.free();
    for ([_]usize{ @min(base.items.len, 100), 2600 }) |length| {
        const ids = try gpu.gpa.alloc(u32, length);
        defer gpu.gpa.free(ids);
        for (ids, 0..) |*t, i| t.* = base.items[i % base.items.len];
        for ([_]bool{ true, false }) |together| try script(gpu, e, ops, &stream, shared.ptr, ids, together);
    }
}

/// Lane `i`'s window in round `r`: its width and kept rows. Lane 0 is 5 rows with every pair of keeps in turn, lane 1 varies in
/// width (1-row windows take the waiting rows in place), lane 2 is 16 rows wide (the log's whole capacity with 16 waiting).
fn plan(i: usize, r: usize) struct { width: usize, keep: usize } {
    switch (i) {
        0 => {
            const keep = if (r < pairs_rounds) 1 + (if (r % 2 == 0) (r / 2 / 5) % 5 else (r / 2) % 5) else 1;
            return .{ .width = 5, .keep = keep };
        },
        1 => {
            const width = 1 + (r * 5 + 3) % 8;
            return .{ .width = width, .keep = 1 + (r * 7 + 2) % width };
        },
        else => return .{ .width = 16, .keep = 1 + (r * 3 + 1) % 16 },
    }
}

fn script(gpu: check.Gpu, e: *prompt.Engine, ops: mtp.Ops(flash.Tri), stream: *cuda.Stream, shared: u64, ids: []const u32, together: bool) !void {
    const gpa = gpu.gpa;
    const row_bytes = prompt.vocabulary * 2;
    const cap = ids.len + rounds * 16 + 64;
    var replay: Side = .{};
    defer replay.deinit(gpa);
    var late: Side = .{};
    defer late.deinit(gpa);
    inline for (.{ .{ &replay, false }, .{ &late, true } }) |pair| {
        const side = pair[0];
        mtp.Draft.late_force = pair[1];
        defer mtp.Draft.late_force = null;
        for (0..lanes_n) |i| {
            const prompt_len = ids.len - 3 * i;
            side.seqs[i] = try e.newSeq(cap);
            side.drafts[i] = mtp.Draft.init(gpa, e.driver, stream.*, cap, 1, e.chunk) catch |err| {
                side.seqs[i].deinit(gpa);
                return err;
            };
            side.made += 1;
            try e.forward(flash.Tri, ops.tri, &side.seqs[i], ids[0..prompt_len], .{});
        }
    }
    try check.expect(late.drafts[0].log != null and replay.drafts[0].log == null, "the late side has its log, the replaying one none", .{});
    const hidden = replay.drafts[0].streams.len / replay.drafts[0].rows_cap;
    const want = try gpa.alloc(u8, 16 * @max(row_bytes, hidden));
    defer gpa.free(want);
    const got = try gpa.alloc(u8, 16 * @max(row_bytes, hidden));
    defer gpa.free(got);
    var window: [lanes_n][16]u32 = undefined;
    var compared: usize = 0;
    var waited: usize = 0;
    for (0..rounds) |r| {
        var seqs: [lanes_n]*prompt.Seq = undefined;
        var ds: [lanes_n]*mtp.Draft = undefined;
        var idss: [lanes_n][]const u32 = undefined;
        for (0..lanes_n) |i| {
            const w = plan(i, r).width;
            for (window[i][0..w], 0..) |*t, k| t.* = @intCast((37 + 101 * i + 17 * k + 29 * r) % 50_000);
            try ops.verify(&replay.seqs[i], &replay.drafts[i], window[i][0..w]);
            seqs[i] = &late.seqs[i];
            ds[i] = &late.drafts[i];
            idss[i] = window[i][0..w];
            if (!together) try ops.verify(seqs[i], ds[i], idss[i]);
        }
        if (together) try ops.verifyMany(&seqs, &ds, &idss, shared);
        try stream.synchronize();
        for (0..lanes_n) |i| {
            const w = plan(i, r).width;
            inline for (.{ .{ "scores", row_bytes, "logits" }, .{ "hidden rows", hidden, "streams" } }) |f| {
                try e.driver.check(e.driver.api.cuMemcpyDtoH_v2(want.ptr, @field(replay.drafts[i], f[2]).ptr, w * f[1]), "cuMemcpyDtoH");
                try e.driver.check(e.driver.api.cuMemcpyDtoH_v2(got.ptr, @field(late.drafts[i], f[2]).ptr, w * f[1]), "cuMemcpyDtoH");
                try check.sameBytes("a window after waiting rows has the replaying side's " ++ f[0], got[0 .. w * f[1]], want[0 .. w * f[1]]);
            }
            compared += 1;
            const keep = plan(i, r).keep;
            try ops.rollback(&replay.seqs[i], &replay.drafts[i], keep);
            try ops.rollback(&late.seqs[i], &late.drafts[i], keep);
            try stream.synchronize();
            try check.expect(replay.seqs[i].fed == late.seqs[i].fed, "kept positions match", .{});
            try check.expect(std.mem.eql(i64, &replay.seqs[i].history, &late.seqs[i].history), "kept n-gram histories match", .{});
            try sameConv(gpu, replay.seqs[i], late.seqs[i]);
            try sameBuf(gpu, "the n-gram tail after a rollback", replay.seqs[i].tail, late.seqs[i].tail);
            waited += late.drafts[i].pend;
            // the states: equal whenever none wait; a lane's waiting rows are taken now every seventh round and its states compared
            if (late.drafts[i].pend > 0 and r % 7 == 6 and r < pairs_rounds) {
                try ops.flushLate(&late.seqs[i], &late.drafts[i]);
                try stream.synchronize();
            }
            if (late.drafts[i].pend == 0) try sameStates(gpu, replay.seqs[i], late.seqs[i]);
        }
    }
    // every lane's waiting rows taken, then one plain row on each
    for (0..lanes_n) |i| {
        try ops.flushLate(&late.seqs[i], &late.drafts[i]);
        try stream.synchronize();
        try sameStates(gpu, replay.seqs[i], late.seqs[i]);
        try e.forward(flash.Tri, ops.tri, &replay.seqs[i], &.{13}, .{});
        try e.forward(flash.Tri, ops.tri, &late.seqs[i], &.{13}, .{});
        try stream.synchronize();
        try check.sameBytes("the row after the last rollback", late.seqs[i].logits, replay.seqs[i].logits);
    }
    check.pass("keep late, {s}, {d} lanes x {d} rounds at {d} prompt tokens: {d} windows' scores and hidden rows, conv windows, tails and states byte-exact ({d} waiting rows seen)", .{ if (together) "shared passes" else "lone passes", lanes_n, rounds, ids.len, compared, waited });
}

fn sameBuf(gpu: check.Gpu, what: []const u8, want: cuda.DeviceBuffer, got: cuda.DeviceBuffer) !void {
    const xs = try check.download(gpu, want);
    defer gpu.gpa.free(xs);
    const ys = try check.download(gpu, got);
    defer gpu.gpa.free(ys);
    try check.sameBytes(what, ys, xs);
}

fn sameConv(gpu: check.Gpu, a: prompt.Seq, b: prompt.Seq) !void {
    for (a.conv, b.conv) |x, y| try sameBuf(gpu, "a conv window after a rollback", x, y);
}

fn sameStates(gpu: check.Gpu, a: prompt.Seq, b: prompt.Seq) !void {
    for (a.state, b.state) |x, y| try sameBuf(gpu, "a DeltaNet state after a rollback", x, y);
}
