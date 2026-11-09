//! A reply's runner receipt (runtime.runner): cell counts, holds and a record per owned cell's span, route and write.
const std = @import("std");
const abstain = @import("runner_abstain.zig");
const answer = @import("runner_commit.zig");
const Allocator = std.mem.Allocator;

/// A span longer than this is recorded as its sha256 and length: the reply holds it in full.
const span_max = 512;

/// One owned cell's write as it landed in the reply.
const Record = struct {
    cell: u32,
    content: bool, // its field: content (after the think end), else reasoning
    at: u64, // the field's bytes before the write
    token: u32, // the reply's tokens before the write
    round: u32, // the session's ticks when the cell was owned
    want: []u32, // the result's tokens (gpa)
    landed: u32 = 0, // of them, committed in order (fewer: the reply's budget cut them, or another token came)
    drafted: u32 = 0, // of those, committed in the trigger's own commit: the model's verify agreed (sync, never held)
    sync: bool = false, // settled at its trigger (sync_graph), else held and forced after the hold
    value: std.ArrayList(u8) = .empty, // the landed tokens' bytes
    told: ?u32 = null, // landed when `notes` last gave it to the stream (null: never)
};

/// A record as it stands, for the stream (`notes`): its field, the field's bytes through its last landed byte, JSON.
pub const Note = struct { index: u32, content: bool, end: u64, json: []const u8 };

/// The records and the counters they need; every committed token passes through `token` (Session.feed).
pub const Log = struct {
    on: bool = true,
    tokens: u32 = 0,
    bytes: [2]u64 = .{ 0, 0 }, // per field: reasoning, content
    content: bool = false, // the last token's field
    rounds: u32 = 0, // Session.tick calls: one per engine round
    writing: bool = false, // the last record's tokens are still landing
    records: std.ArrayList(Record) = .empty,
    out: std.ArrayList(u8) = .empty, // the receipt JSON (write)
    live: std.ArrayList(u8) = .empty, // the JSON `notes` gave last
    notes: std.ArrayList(Note) = .empty,
    abstains_on: bool = false, // PREFIX_ABSTAINS (records on): abstained cells too, with their sources (runner_abstain.zig)
    text: [2]std.ArrayList(u8) = .{ .empty, .empty }, // abstain records: per field, the reply's bytes (to find sources)
    closes: std.ArrayList(abstain.Close) = .empty, // abstain records: where each cell's source closed
    abstains: std.ArrayList(abstain.Abstain) = .empty,
    dropped: u32 = 0, // abstain records out of memory
    anchors_lost: bool = false, // a source anchor out of memory: the records stand, the census is incomplete
    commit: ?answer.Record = null, // PREFIX_COMMIT fired: its close and answer, landing after its cell's write

    /// A committed token: counted in its field, and matched against a write still landing.
    pub fn token(l: *Log, gpa: Allocator, id: u32, bytes: []const u8, think_open: bool) Allocator.Error!void {
        if (!l.on) return;
        l.content = !think_open;
        if (l.abstains_on) l.text[@intFromBool(l.content)].appendSlice(gpa, bytes) catch {
            l.anchors_lost = true;
        };
        defer {
            l.tokens += 1;
            l.bytes[@intFromBool(l.content)] += bytes.len;
        }
        if (!l.writing) return if (l.commit) |*c| c.land(gpa, id, bytes, l.bytes[1], l.tokens);
        const r = &l.records.items[l.records.items.len - 1];
        l.writing = r.want[r.landed] == id;
        if (!l.writing) return;
        try r.value.appendSlice(gpa, bytes);
        r.landed += 1;
        if (r.sync and l.rounds == r.round) r.drafted += 1;
        l.writing = r.landed < r.want.len;
    }

    /// Cell `cell` owned with `ids` to force after the token just fed (copied).
    pub fn owned(l: *Log, gpa: Allocator, cell: u32, ids: []const u32) Allocator.Error!void {
        if (!l.on) return;
        l.writing = false;
        const want = try gpa.dupe(u32, ids);
        errdefer gpa.free(want);
        try l.records.append(gpa, .{ .cell = cell, .content = l.content, .at = l.bytes[@intFromBool(l.content)], .token = l.tokens, .round = l.rounds, .want = want });
        l.writing = ids.len > 0;
    }

    /// The record just owned settled at its trigger (sync_graph): its tokens in this commit are the model's own drafts.
    pub fn synced(l: *Log) void {
        if (l.records.items.len > 0) l.records.items[l.records.items.len - 1].sync = true;
    }

    /// Cell `cell`'s source closed with the token fed last (abstain records: where to find it).
    pub fn closed(l: *Log, gpa: Allocator, cell: u32) void {
        if (!l.on or !l.abstains_on) return;
        l.closes.append(gpa, .{ .cell = cell, .content = l.content, .end = l.bytes[@intFromBool(l.content)] }) catch {
            l.anchors_lost = true;
        };
    }

    /// Cell `a.cell` abstained here: the reply's field, bytes, tokens and rounds at its decision (abstain records).
    pub fn abstained(l: *Log, gpa: Allocator, a: abstain.Abstain) void {
        if (!l.on or !l.abstains_on) return;
        var x = a;
        x.content = l.content;
        x.at = l.bytes[@intFromBool(l.content)];
        x.token = l.tokens;
        x.round = l.rounds;
        l.abstains.append(gpa, x) catch {
            l.dropped += 1;
        };
    }

    pub fn deinit(l: *Log, gpa: Allocator) void {
        for (l.records.items) |*r| {
            gpa.free(r.want);
            r.value.deinit(gpa);
        }
        l.records.deinit(gpa);
        if (l.commit) |*c| c.deinit(gpa);
        l.out.deinit(gpa);
        l.live.deinit(gpa);
        l.notes.deinit(gpa);
        for (&l.text) |*t| t.deinit(gpa);
        l.closes.deinit(gpa);
        l.abstains.deinit(gpa);
    }
};

/// Session `s`'s receipt JSON (runner.Session), in its log's buffer until the session is destroyed ("": out of memory).
pub fn write(s: anytype) []const u8 {
    const l = &s.rec;
    l.out.clearRetainingCapacity();
    const cfg = if (comptime @typeInfo(@TypeOf(s.r)) == .pointer) &s.r.cfg else null;
    put(s.cells.items, l, s.r.gpa, s.held_ms, s.resume_ms, cfg) catch return "";
    return l.out.items;
}

/// Session `s`'s records new or grown since the last call (null: none, or records off), for a stream cut early.
pub fn notes(s: anytype) ?[]const Note {
    const l = &s.rec;
    if (!l.on) return null;
    fresh(s.cells.items, l, s.r.gpa) catch return null; // out of memory: the next call gives them again
    return if (l.notes.items.len > 0) l.notes.items else null;
}

fn fresh(cells: anytype, l: *Log, gpa: Allocator) Allocator.Error!void {
    l.notes.clearRetainingCapacity();
    l.live.clearRetainingCapacity();
    for (l.records.items, 0..) |r, i| if (r.told != r.landed) {
        const from = l.live.items.len;
        try record(&l.live, gpa, cells, r);
        try l.notes.append(gpa, .{ .index = @intCast(i), .content = r.content, .end = r.at + r.value.items.len, .json = l.live.items[from..] });
    };
    var from: usize = 0;
    for (l.notes.items) |*n| {
        n.json = l.live.items[from..][0..n.json.len]; // the buffer may have moved as it grew
        from += n.json.len;
        l.records.items[n.index].told = l.records.items[n.index].landed;
    }
}

pub const form_names = [_][]const u8{ "fence", "repl", "inline", "trace" };

fn put(cells: anytype, l: *Log, gpa: Allocator, held_ms: u64, resume_ms: u64, cfg: anytype) Allocator.Error!void {
    var forms = std.mem.zeroes([4][4]u32);
    var kinds = std.mem.zeroes([3]u32);
    for (cells) |c| {
        const f = &forms[if (c.stated != null) 3 else @backingInt(c.form)];
        f[0] += 1;
        switch (c.told) {
            .none => {},
            .owned => {
                f[1] += 1;
                f[3] += c.owned_tokens;
                if (c.translation == .program) kinds[@backingInt(c.translation.program.kind)] += 1;
            },
            .abstained => f[2] += 1,
        }
    }
    const o = &l.out;
    try o.append(gpa, '{');
    for (forms, form_names) |f, name|
        try o.print(gpa, "\"{s}\":{{\"closed\":{d},\"owned\":{d},\"abstained\":{d},\"owned_tokens\":{d}}},", .{ name, f[0], f[1], f[2], f[3] });
    try o.print(gpa, "\"owned_by\":{{\"calc\":{d},\"elf\":{d},\"graph\":{d}}},\"held_ms\":{d},\"resume_ms\":{d}", .{ kinds[0], kinds[1], kinds[2], held_ms, resume_ms });
    if (l.on) {
        try o.appendSlice(gpa, ",\"cells\":[");
        for (l.records.items, 0..) |r, i| {
            if (i > 0) try o.append(gpa, ',');
            try record(o, gpa, cells, r);
        }
        try o.append(gpa, ']');
        if (l.abstains_on) try abstain.write(o, gpa, cells, l, cfg);
        if (l.commit) |c| try answer.write(o, gpa, cells, c);
    }
    try o.append(gpa, '}');
}

/// One record's JSON object, as the receipt's `cells` and the stream's notes give it.
fn record(o: *std.ArrayList(u8), gpa: Allocator, cells: anytype, r: Record) Allocator.Error!void {
    const c = cells[r.cell];
    const route = if (c.translation == .program) @tagName(c.translation.program.kind) else "none";
    const status = if (c.result) |res| @tagName(res.status) else "none";
    try o.print(gpa, "{{\"cell\":{d},\"form\":\"{s}\",\"route\":\"{s}\",\"status\":\"{s}\",", .{ r.cell, form_names[if (c.stated != null) 3 else @backingInt(c.form)], route, status });
    try text(o, gpa, "lang", c.lang);
    if (c.code.len <= span_max) try text(o, gpa, "span", c.code) else {
        var digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(c.code, &digest, .{});
        try o.print(gpa, "\"span_sha256\":\"{s}\",\"span_len\":{d},", .{ std.fmt.bytesToHex(digest, .lower), c.code.len });
    }
    if (c.stated) |st| try text(o, gpa, "display", st.display);
    try text(o, gpa, "value", r.value.items);
    try o.print(gpa, "\"field\":\"{s}\",\"at\":{d},\"token\":{d},\"ids\":{d},\"landed\":{d},\"drafted\":{d},\"forced\":{d},\"path\":\"{s}\"}}", .{ if (r.content) "content" else "reasoning", r.at, r.token, r.want.len, r.landed, r.drafted, r.landed - r.drafted, if (r.sync) "sync" else "held" });
}

/// `"name":"bytes",` as a JSON string; bytes that are not UTF-8 (a cut mid-character) go as `"name_hex"` instead.
pub fn text(o: *std.ArrayList(u8), gpa: Allocator, name: []const u8, bytes: []const u8) Allocator.Error!void {
    if (!std.unicode.utf8ValidateSlice(bytes)) {
        try o.print(gpa, "\"{s}_hex\":\"", .{name});
        for (bytes) |b| try o.print(gpa, "{x:0>2}", .{b});
        return o.appendSlice(gpa, "\",");
    }
    try o.print(gpa, "\"{s}\":", .{name});
    try quoted(o, gpa, bytes);
    try o.append(gpa, ',');
}

/// `bytes` as one JSON value: a string, or `{"hex":"…"}` when they are not UTF-8.
pub fn quoted(o: *std.ArrayList(u8), gpa: Allocator, bytes: []const u8) Allocator.Error!void {
    if (!std.unicode.utf8ValidateSlice(bytes)) {
        try o.appendSlice(gpa, "{\"hex\":\"");
        for (bytes) |b| try o.print(gpa, "{x:0>2}", .{b});
        return o.appendSlice(gpa, "\"}");
    }
    try o.append(gpa, '"');
    for (bytes) |b| switch (b) {
        '"' => try o.appendSlice(gpa, "\\\""),
        '\\' => try o.appendSlice(gpa, "\\\\"),
        0...0x1f, 0x7f => try o.print(gpa, "\\u{x:0>4}", .{b}),
        else => try o.append(gpa, b),
    };
    try o.append(gpa, '"');
}
