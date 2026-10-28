//! The MoE shared expert's second stream: forked from the compute stream where its input rows are ready, joined where the
//! experts' output is first read. One per scratch (so one per engine and rank), made on first use.
//!
//! Only the schedule moves: the side stream runs the same kernels on the same arguments as the compute stream did, so the
//! output bits are the same. The shared expert owns shared_gate/up/act/out and the split-K partials (native_part), which
//! nothing else touches between the fork and the join; its slot of y is a column range the routed kernels do not write.
//! Every use of those buffers outside the window is ordered behind the join by the compute stream, and a free waits on the
//! compute stream, which has the join in front of everything queued after it.
const std = @import("std");
const cuda = @import("cuda");

/// Where a layer's shared expert runs.
pub const Mode = enum {
    /// On the compute stream, after the routed experts (the v1.1.4 order).
    off,
    /// On the side stream in decode rounds, drafted windows and other passes that are not prompt chunks.
    decode,
    /// Also in prompt chunks.
    all,
};

/// TF_FLASHNEXT_SHARED_SIDE: 0 off, 1 decode and windows, 2 every pass (read once per process).
pub fn mode() Mode {
    const v = std.c.getenv("TF_FLASHNEXT_SHARED_SIDE") orelse return .off;
    const t = std.mem.span(v);
    if (std.mem.eql(u8, t, "1")) return .decode;
    if (std.mem.eql(u8, t, "2")) return .all;
    return .off;
}

/// TF_FLASHNEXT_SHARED_SIDE_PRIO=high: the side stream gets the context's greatest priority, so its few CTAs are placed
/// ahead of the routed experts' grid instead of behind it.
fn highPriority() bool {
    const v = std.c.getenv("TF_FLASHNEXT_SHARED_SIDE_PRIO") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "high");
}

/// TF_FLASHNEXT_SHARED_SIDE_POISON=1: debugging. The side stream fills the shared expert's scratch with NaN bytes before
/// each use; any byte it reads that it did not write itself then shows up in the replies.
pub fn poison() bool {
    const v = std.c.getenv("TF_FLASHNEXT_SHARED_SIDE_POISON") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}

pub const Side = struct {
    stream: cuda.Stream,
    /// Recorded on the compute stream (fork), recorded on the side stream (join); ordering only.
    fork: cuda.Event,
    join: cuda.Event,
    mode: Mode,

    pub fn init(d: *const cuda.Driver, m: Mode) !Side {
        var stream = if (highPriority()) try cuda.Stream.initPriority(d, .highest) else try cuda.Stream.init(d, true);
        errdefer stream.deinit();
        var fork = try cuda.Event.init(d, false);
        errdefer fork.deinit();
        const join = try cuda.Event.init(d, false);
        return .{ .stream = stream, .fork = fork, .join = join, .mode = m };
    }

    /// Waits for the side stream, then releases it; the caller frees nothing the stream could still be using.
    pub fn deinit(sd: *Side) void {
        sd.stream.deinit();
        sd.fork.deinit();
        sd.join.deinit();
    }

    /// Whether a pass (`prompt`: a prompt chunk) puts its shared experts here.
    pub fn applies(sd: *const Side, prompt: bool) bool {
        return switch (sd.mode) {
            .off => false,
            .decode => !prompt,
            .all => true,
        };
    }

    /// The side stream starts behind everything `main` has queued so far (the rows the shared expert reads).
    pub fn begin(sd: *const Side, main: cuda.Stream) !void {
        try sd.fork.record(main);
        try sd.stream.wait(sd.fork);
    }

    /// `main` continues behind everything the side stream has queued: call before anything reads its outputs.
    pub fn end(sd: *const Side, main: cuda.Stream) !void {
        try sd.join.record(sd.stream);
        try main.wait(sd.join);
    }
};

/// The scratch's side stream, made on the first call; null when the knob is off.
pub fn get(slot: *?Side, d: *const cuda.Driver) !?*Side {
    if (slot.*) |*sd| return sd;
    const m = mode();
    if (m == .off) return null;
    slot.* = try Side.init(d, m);
    return &slot.*.?;
}

const tt = std.testing;

const Fake = struct {
    var log: [32]u32 = undefined;
    var n: usize = 0;
    var made: usize = 0;
    var destroyed: usize = 0;
    fn put(code: u32) void {
        log[n] = code;
        n += 1;
    }
    fn stream(out: *cuda.abi.Stream, _: c_uint) callconv(.c) cuda.abi.Result {
        made += 1;
        out.* = @ptrFromInt(0x1000);
        return 0;
    }
    fn streamDestroy(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        destroyed += 1;
        return 0;
    }
    fn sync(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return 0;
    }
    fn event(out: *cuda.abi.Event, _: c_uint) callconv(.c) cuda.abi.Result {
        made += 1;
        out.* = @ptrFromInt(0x2000 + made);
        return 0;
    }
    fn eventDestroy(_: cuda.abi.Event) callconv(.c) cuda.abi.Result {
        destroyed += 1;
        return 0;
    }
    // codes: 1 record(main), 2 wait(side), 3 record(side), 4 wait(main)
    fn record(_: cuda.abi.Event, s: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        put(if (@intFromPtr(s) == 0x1000) 3 else 1);
        return 0;
    }
    fn wait(s: cuda.abi.Stream, _: cuda.abi.Event, _: c_uint) callconv(.c) cuda.abi.Result {
        put(if (@intFromPtr(s) == 0x1000) 2 else 4);
        return 0;
    }
};

test "the side stream forks behind the compute stream's queue and the compute stream joins behind the side stream's" {
    var d: cuda.Driver = undefined;
    d.api.cuStreamCreate = Fake.stream;
    d.api.cuStreamDestroy_v2 = Fake.streamDestroy;
    d.api.cuStreamSynchronize = Fake.sync;
    d.api.cuEventCreate = Fake.event;
    d.api.cuEventDestroy_v2 = Fake.eventDestroy;
    d.api.cuEventRecord = Fake.record;
    d.api.cuStreamWaitEvent = Fake.wait;
    Fake.n = 0;
    Fake.made = 0;
    Fake.destroyed = 0;
    const main: cuda.Stream = .{ .d = &d, .handle = @ptrFromInt(0x99) };
    var sd = try Side.init(&d, .decode);
    try sd.begin(main);
    try sd.end(main);
    try sd.begin(main); // the next layer reuses both events
    try sd.end(main);
    try tt.expectEqualSlices(u32, &.{ 1, 2, 3, 4, 1, 2, 3, 4 }, Fake.log[0..Fake.n]);
    sd.deinit();
    try tt.expectEqual(Fake.made, Fake.destroyed);
}

test "the modes pick the passes that fork" {
    var sd: Side = undefined;
    sd.mode = .off;
    try tt.expect(!sd.applies(false) and !sd.applies(true));
    sd.mode = .decode;
    try tt.expect(sd.applies(false) and !sd.applies(true));
    sd.mode = .all;
    try tt.expect(sd.applies(false) and sd.applies(true));
}
