//! Events the engine thread hands one request, read by the request's own thread.
const std = @import("std");
const api = @import("engine_api");
const Allocator = std.mem.Allocator;

pub const Mailbox = struct {
    io: std.Io,
    gpa: Allocator,
    mutex: std.Io.Mutex = .init,
    cond: std.Io.Condition = .init,
    audio: usize = 0,
    tokens: std.ArrayList(u32) = .empty,
    chunks: std.ArrayList(usize) = .empty, // where each round's tokens end
    rows: std.ArrayList(api.LogprobRow) = .empty, // one a token of `tokens`, for requests with logprobs
    prompt_rows: ?[]api.LogprobRow = null, // one a prompt token after the first, for requests with prompt_logprobs
    cached: ?u32 = null,
    prefilled_ns: ?i96 = null,
    finished: bool = false,
    reason: api.Reason = .stop,
    stats: api.Stats = .{},
    widths: []u32 = &.{},
    raised: []bool = &.{},
    telemetry: []u8 = "",
    runner: []u8 = "",
    message: []u8 = "",
    cells: std.ArrayList(std.ArrayList(Cell)) = .empty, // the runner's records by index, versions oldest first (Event.cells)

    /// A runner record's version: its field, the field's bytes through its last landed byte, its JSON (gpa).
    pub const Cell = struct { content: bool, end: u64, json: []u8 };

    pub fn onEvent(ctx: *anyopaque, _: api.Id, event: *const api.Event) void {
        const m: *Mailbox = @ptrCast(@alignCast(ctx));
        m.mutex.lockUncancelable(m.io);
        defer m.mutex.unlock(m.io);
        switch (event.*) {
            .prefilled => |cached| {
                m.cached = cached;
                m.prefilled_ns = std.Io.Clock.awake.now(m.io).toNanoseconds();
            },
            .logprobs => |r| m.rows.appendSlice(m.gpa, r) catch {},
            .prompt_logprobs => |r| m.prompt_rows = m.gpa.dupe(api.LogprobRow, r) catch null,
            .tokens => |t| {
                m.tokens.appendSlice(m.gpa, t) catch {};
                m.chunks.append(m.gpa, m.tokens.items.len) catch {};
            },
            .audio => |n| m.audio = n,
            .cells => |notes| for (notes) |n| m.note(n) catch {},
            .finished => |f| {
                m.finished = true;
                m.reason = f.reason;
                m.stats = f.stats;
                m.widths = m.gpa.dupe(u32, f.stats.prefill_widths) catch &.{};
                m.raised = m.gpa.dupe(bool, f.stats.prefill_raised) catch &.{};
                m.telemetry = m.gpa.dupe(u8, f.stats.telemetry_json) catch "";
                m.runner = m.gpa.dupe(u8, f.runner) catch "";
                m.message = m.gpa.dupe(u8, f.message) catch "";
            },
        }
        m.cond.signal(m.io);
    }

    fn note(m: *Mailbox, n: anytype) Allocator.Error!void {
        const json = try m.gpa.dupe(u8, n.json);
        errdefer m.gpa.free(json);
        while (m.cells.items.len <= n.index) try m.cells.append(m.gpa, .empty);
        try m.cells.items[n.index].append(m.gpa, .{ .content = n.content, .end = n.end, .json = json });
    }

    /// Drops versions older than each record's newest fully `shown` one; n: records up to one with none (lock held).
    pub fn covered(m: *Mailbox, shown: [2]usize) struct { n: usize, sig: u64 } {
        var sig: u64 = 0;
        for (m.cells.items, 0..) |*versions, n| {
            var k = versions.items.len;
            while (k > 0 and versions.items[k - 1].end > shown[@intFromBool(versions.items[k - 1].content)]) k -= 1;
            if (k == 0) return .{ .n = n, .sig = sig };
            for (versions.items[0 .. k - 1]) |old| m.gpa.free(old.json);
            std.mem.copyForwards(Cell, versions.items, versions.items[k - 1 ..]);
            versions.items.len -= k - 1;
            sig += versions.items[0].end + 1;
        }
        return .{ .n = m.cells.items.len, .sig = sig };
    }

    pub fn deinit(m: *Mailbox) void {
        for (m.cells.items) |*versions| {
            for (versions.items) |c| m.gpa.free(c.json);
            versions.deinit(m.gpa);
        }
        m.cells.deinit(m.gpa);
        m.tokens.deinit(m.gpa);
        m.chunks.deinit(m.gpa);
        m.rows.deinit(m.gpa);
        if (m.prompt_rows) |r| m.gpa.free(r);
        m.gpa.free(m.widths);
        m.gpa.free(m.raised);
        m.gpa.free(m.telemetry);
        m.gpa.free(m.runner);
        m.gpa.free(m.message);
    }
};
