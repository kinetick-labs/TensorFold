//! Graph programs to the graph backend (a sequence's pieces side by side), calc and ELF to the circuit, by job id bits.
const std = @import("std");
const api = @import("runner_api.zig");
const prog = @import("runner_graph_program.zig");
const graph = @import("runner_graph.zig");
const Allocator = std.mem.Allocator;
const log = std.log.scoped(.runner_graph);

pub const Dispatch = struct {
    gpa: Allocator,
    circuit: api.Backend,
    graph: api.Backend,
    /// The bank TFGS2 templates lower on; without one a template piece faults.
    bank: ?*const graph.Bank = null,
    seqs: std.ArrayList(*Seq) = .empty,
    next_seq: api.JobId = 1,

    const tag: api.JobId = 1 << 63;
    const seq_tag: api.JobId = 1 << 62;

    /// One line of a sequence: a bank packet, a TFGS2 template, or text ('L') printed as written; kind 0 is malformed.
    const Piece = struct {
        kind: u8,
        sep: u8,
        body: []const u8,
        template: bool = false,
        value: [160]u8 = undefined,
        value_len: ?usize = null, // the run's integer output, for later templates
        job: ?api.JobId = null,
        done: ?api.Result = null, // its run's status and counts; the printed value is in `text`
        text: [160]u8 = undefined,
        len: ?usize = null, // null: the value is not one `kind` can print
    };

    const Seq = struct {
        id: api.JobId,
        stream: u64,
        budget: u64,
        pieces: []Piece,
        next: usize = 0, // the first piece not yet printed
        out: std.ArrayList(u8) = .empty,
        steps: u64 = 0,
        exec_ns: u64 = 0,
        end: ?api.Result = null,
        delivered: bool = false,
        bytes: []u8,

        /// Appends a piece's text and its separator (S a space, N a newline, J none); false when the run ended here.
        fn say(q: *Seq, gpa: Allocator, text: []const u8, sep: u8) bool {
            const end: []const u8 = switch (sep) {
                'S' => " ",
                'N' => "\n",
                'J' => "",
                else => {
                    q.end = .{ .status = .fault, .steps = q.steps };
                    return false;
                },
            };
            q.out.appendSlice(gpa, text) catch {
                q.end = .{ .status = .abstain, .abstain = .too_big };
                return false;
            };
            q.out.appendSlice(gpa, end) catch {};
            return true;
        }
    };

    pub fn backend(d: *Dispatch) api.Backend {
        return .{ .ctx = d, .vtable = &.{ .start = start, .poll = poll, .cancel = cancel, .release = release, .now = now } };
    }
    pub fn deinit(d: *Dispatch) void {
        for (d.seqs.items) |q| d.free(q);
        d.seqs.deinit(d.gpa);
    }
    fn stop(d: *Dispatch, q: *Seq) void {
        for (q.pieces) |*p| if (p.job) |j| {
            d.graph.cancel(j);
            p.job = null;
        };
    }
    fn free(d: *Dispatch, q: *Seq) void {
        d.stop(q);
        q.out.deinit(d.gpa);
        d.gpa.free(q.pieces);
        d.gpa.free(q.bytes);
        d.gpa.destroy(q);
    }
    /// A stream's delivered sequences: their `out` lives until its next call.
    fn forget(d: *Dispatch, stream: u64, all: bool) void {
        var i: usize = 0;
        while (i < d.seqs.items.len) {
            const q = d.seqs.items[i];
            if (q.stream == stream and (all or q.delivered)) d.free(d.seqs.swapRemove(i)) else i += 1;
        }
    }
    fn start(ctx: *anyopaque, stream: u64, p: api.Program, budget: u64) error{ OutOfMemory, Busy }!api.JobId {
        const d: *Dispatch = @ptrCast(@alignCast(ctx));
        if (p.kind != .graph) return d.circuit.start(stream, p, budget);
        if (!std.mem.startsWith(u8, p.bytes, "TFGS")) return tag | try d.graph.start(stream, p, budget);
        const v2 = std.mem.startsWith(u8, p.bytes, prog.seq_magic_v2);
        const known = v2 or std.mem.startsWith(u8, p.bytes, prog.seq_magic);
        // a sequence version this runner does not know faults where it would otherwise be read wrong
        if (!known) log.warn("graph sequence {s}: unknown version, refused", .{p.bytes[0..@min(p.bytes.len, 6)]});
        d.forget(stream, false);
        const q = try d.gpa.create(Seq);
        errdefer d.gpa.destroy(q);
        const bytes = try d.gpa.dupe(u8, p.bytes);
        errdefer d.gpa.free(bytes);
        const pieces = try split(d.gpa, if (known) bytes[prog.seq_magic.len..] else "", v2);
        errdefer d.gpa.free(pieces);
        q.* = .{ .id = d.next_seq, .stream = stream, .budget = budget, .pieces = pieces, .bytes = bytes };
        if (!known) q.end = .{ .status = .fault };
        try d.seqs.append(d.gpa, q);
        d.next_seq += 1;
        d.step(q);
        return seq_tag | q.id;
    }
    /// A sequence's lines (`<kind><sep> <body>\n`); text after the last newline is not a piece.
    fn split(gpa: Allocator, lines: []const u8, v2: bool) Allocator.Error![]Piece {
        const n = std.mem.count(u8, lines, "\n");
        const pieces = try gpa.alloc(Piece, n);
        var it = std.mem.splitScalar(u8, lines, '\n');
        for (pieces) |*p| {
            const line = it.next().?;
            p.* = if (line.len < 4 or line[2] != ' ') .{ .kind = 0, .sep = 0, .body = "" } else .{ .kind = line[0], .sep = line[1], .body = line[3..] };
            p.template = v2 and p.kind != 'L' and p.body[0] == '@';
        }
        return pieces;
    }
    /// Collects, starts what workers allow, prints in order; ends at the last piece or the first (in order) not ok.
    fn step(d: *Dispatch, q: *Seq) void {
        d.collect(q);
        d.launch(q);
        while (q.end == null) {
            if (q.next == q.pieces.len) {
                q.end = .{ .status = .ok, .out = q.out.items, .steps = q.steps, .exec_ns = q.exec_ns };
                break;
            }
            const p = &q.pieces[q.next];
            if (p.kind == 0) {
                q.end = .{ .status = .fault, .steps = q.steps };
                break;
            }
            if (p.kind == 'L') {
                q.next += 1;
                if (!q.say(d.gpa, p.body, p.sep)) break;
                continue;
            }
            const r = p.done orelse return;
            q.next += 1;
            q.steps += r.steps;
            q.exec_ns += r.exec_ns;
            // a piece run beside earlier ones had the whole budget; in turn it would have had what they left
            if (r.status == .ok and q.steps > q.budget) {
                q.end = .{ .status = .budget, .steps = q.budget, .exec_ns = q.exec_ns };
                break;
            }
            if (r.status != .ok) {
                q.end = .{ .status = r.status, .abstain = r.abstain, .steps = @min(q.steps, q.budget), .exec_ns = q.exec_ns };
                break;
            }
            const len = p.len orelse {
                q.end = .{ .status = .fault, .steps = q.steps };
                break;
            };
            if (!q.say(d.gpa, p.text[0..len], p.sep)) break;
        }
        d.stop(q);
    }
    /// Polls every running piece; a finished piece's value is copied out at once (the backend frees it next call).
    fn collect(d: *Dispatch, q: *Seq) void {
        for (q.pieces[q.next..]) |*p| if (p.job) |j| switch (d.graph.poll(j)) {
            .running => {},
            .done => |r| {
                p.job = null;
                p.done = r;
                p.done.?.out = "";
                const v = std.mem.trimEnd(u8, r.out, "\n");
                if (r.status == .ok) if (format(p.kind, v, &p.text)) |text| {
                    p.len = text.len;
                };
                if (r.status == .ok and p.kind == 'I' and p.len != null and v.len <= p.value.len) {
                    @memcpy(p.value[0..v.len], v);
                    p.value_len = v.len;
                }
            },
        };
    }
    /// Starts waiting pieces, largest packet first, until the graph backend is busy: the longest runs set the time.
    fn launch(d: *Dispatch, q: *Seq) void {
        if (q.end != null) return;
        while (true) {
            var best: ?*Piece = null;
            for (q.pieces[q.next..], q.next..) |*p, i| {
                if (p.kind == 0 or p.kind == 'L' or p.job != null or p.done != null) continue;
                if (p.template and !d.ready(q, i)) continue;
                if (best == null or p.body.len > best.?.body.len) best = p;
            }
            const p = best orelse return;
            if (p.template) switch (d.bind(q, p)) {
                .started, .ended => {},
                .busy => return,
            } else p.job = d.graph.start(q.stream, .{ .kind = .graph, .bytes = p.body }, q.budget) catch return;
        }
    }
    /// Whether every piece template `i` reads has its value (a missing one ends the run before `i` prints).
    fn ready(_: *Dispatch, q: *Seq, i: usize) bool {
        var it = std.mem.splitScalar(u8, q.pieces[i].body, '$');
        _ = it.next();
        while (it.next()) |rest| {
            var n: usize = 0;
            while (n < rest.len and std.ascii.isDigit(rest[n])) n += 1;
            const k = std.fmt.parseInt(usize, rest[0..n], 10) catch return true;
            // a reference to itself or a later piece is malformed: bind reports it
            if (k >= i) return true;
            if (q.pieces[k].value_len == null) return false;
        }
        return true;
    }
    /// Lowers a ready template on the bank and starts it; ended: it is done without a job (no bank, or no packet).
    fn bind(d: *Dispatch, q: *Seq, p: *Piece) enum { started, ended, busy } {
        const bank = d.bank orelse {
            log.warn("graph sequence template without a bank", .{});
            p.done = .{ .status = .fault };
            return .ended;
        };
        var arena_state = std.heap.ArenaAllocator.init(d.gpa);
        defer arena_state.deinit();
        const arena = arena_state.allocator();
        const values = arena.alloc(?[]const u8, q.pieces.len) catch return .busy;
        for (q.pieces, values) |*o, *v| v.* = if (o != p and o.value_len != null) o.value[0..o.value_len.?] else null;
        const bytes = (prog.bind(arena, bank, p.body, values) catch return .busy) orelse {
            p.done = .{ .status = .abstain, .abstain = .unsupported };
            return .ended;
        };
        p.job = d.graph.start(q.stream, .{ .kind = .graph, .bytes = bytes }, q.budget) catch return .busy;
        return .started;
    }
    /// A piece's printed value: I as is, B as Python's bool, F with `.0`, `1`..`6` at that fixed Decimal scale.
    fn format(kind: u8, v: []const u8, buf: []u8) ?[]const u8 {
        const digits = v[@intFromBool(v.len > 0 and v[0] == '-')..];
        if (digits.len == 0 or digits.len + 3 > buf.len) return null;
        for (digits) |c| if (!std.ascii.isDigit(c)) return null;
        return switch (kind) {
            'I' => blk: {
                @memcpy(buf[0..v.len], v);
                break :blk buf[0..v.len];
            },
            // a comparison's graph prints 1 or 0
            'B' => if (std.mem.eql(u8, v, "1")) copy(buf, "True") else if (std.mem.eql(u8, v, "0")) copy(buf, "False") else null,
            'F' => std.fmt.bufPrint(buf, "{s}.0", .{v}) catch null,
            '1'...'6' => fixed(v, kind - '0', buf),
            else => null,
        };
    }
    fn copy(buf: []u8, s: []const u8) []const u8 {
        @memcpy(buf[0..s.len], s);
        return buf[0..s.len];
    }
    /// Decimal's str of coefficient v at `scale` places (the runtime's tf_wfixed): 0.0042, -12.50.
    fn fixed(v: []const u8, scale: usize, buf: []u8) ?[]const u8 {
        const neg = v[0] == '-';
        const digits = v[@intFromBool(neg)..];
        var w: std.Io.Writer = .fixed(buf);
        if (neg) w.writeByte('-') catch return null;
        if (digits.len <= scale) {
            w.writeAll("0.") catch return null;
            w.splatByteAll('0', scale - digits.len) catch return null;
            w.writeAll(digits) catch return null;
        } else {
            w.writeAll(digits[0 .. digits.len - scale]) catch return null;
            w.writeByte('.') catch return null;
            w.writeAll(digits[digits.len - scale ..]) catch return null;
        }
        return w.buffered();
    }
    fn find(d: *Dispatch, id: api.JobId) ?*Seq {
        for (d.seqs.items) |q| if (q.id == id) return q;
        return null;
    }
    fn poll(ctx: *anyopaque, job: api.JobId) api.Poll {
        const d: *Dispatch = @ptrCast(@alignCast(ctx));
        if (job & tag != 0) return d.graph.poll(job & ~tag);
        if (job & seq_tag == 0) return d.circuit.poll(job);
        const q = d.find(job & ~seq_tag) orelse return .{ .done = .{ .status = .abstain, .abstain = .unsupported } };
        d.step(q);
        var r = q.end orelse return .running;
        r.out = q.out.items;
        q.delivered = true;
        return .{ .done = r };
    }
    fn cancel(ctx: *anyopaque, job: api.JobId) void {
        const d: *Dispatch = @ptrCast(@alignCast(ctx));
        if (job & tag != 0) return d.graph.cancel(job & ~tag);
        if (job & seq_tag == 0) return d.circuit.cancel(job);
        for (d.seqs.items, 0..) |q, i| if (q.id == job & ~seq_tag) {
            d.free(d.seqs.swapRemove(i));
            return;
        };
    }
    fn release(ctx: *anyopaque, stream: u64) void {
        const d: *Dispatch = @ptrCast(@alignCast(ctx));
        d.forget(stream, true);
        d.graph.release(stream);
        d.circuit.release(stream);
    }
    fn now(ctx: *anyopaque, p: api.Program, budget: u64, quick: u64) ?api.Result {
        const d: *Dispatch = @ptrCast(@alignCast(ctx));
        // one packet only: a sequence's pieces run as jobs
        return if (p.kind == .graph and !std.mem.startsWith(u8, p.bytes, "TFGS")) d.graph.now(p, budget, quick) else null;
    }
};

test "runner graph route: graph programs reach the graph backend, the rest the circuit, by job id" {
    const Fake = struct {
        name: u8,
        started: u32 = 0,
        released: u32 = 0,
        fn backend(f: *@This()) api.Backend {
            return .{ .ctx = f, .vtable = &.{ .start = st, .poll = pl, .cancel = cn, .release = rl, .now = nw } };
        }
        fn st(ctx: *anyopaque, _: u64, _: api.Program, _: u64) error{ OutOfMemory, Busy }!api.JobId {
            const f: *@This() = @ptrCast(@alignCast(ctx));
            f.started += 1;
            return 7;
        }
        fn pl(ctx: *anyopaque, job: api.JobId) api.Poll {
            const f: *@This() = @ptrCast(@alignCast(ctx));
            return .{ .done = .{ .status = .ok, .out = if (f.name == 'g') "graph" else "circuit", .steps = job } };
        }
        fn cn(_: *anyopaque, _: api.JobId) void {}
        fn nw(ctx: *anyopaque, _: api.Program, _: u64, _: u64) ?api.Result {
            const f: *@This() = @ptrCast(@alignCast(ctx));
            return .{ .status = .ok, .out = if (f.name == 'g') "graph" else "circuit" };
        }
        fn rl(ctx: *anyopaque, _: u64) void {
            const f: *@This() = @ptrCast(@alignCast(ctx));
            f.released += 1;
        }
    };
    var c: Fake = .{ .name = 'c' };
    var g: Fake = .{ .name = 'g' };
    var d: Dispatch = .{ .gpa = std.testing.allocator, .circuit = c.backend(), .graph = g.backend() };
    const b = d.backend();
    const gj = try b.start(1, .{ .kind = .graph, .bytes = "{}" }, 10);
    const cj = try b.start(1, .{ .kind = .calc, .bytes = "2*3" }, 10);
    try std.testing.expectEqualStrings("graph", b.poll(gj).done.out);
    try std.testing.expectEqual(@as(u64, 7), b.poll(gj).done.steps);
    try std.testing.expectEqualStrings("circuit", b.poll(cj).done.out);
    // a run here (settling holds) is the graph backend's only
    try std.testing.expectEqualStrings("graph", b.now(.{ .kind = .graph, .bytes = "{}" }, 10, 10).?.out);
    try std.testing.expectEqual(null, b.now(.{ .kind = .calc, .bytes = "2*3" }, 10, 10));
    b.release(1);
    try std.testing.expect(c.started == 1 and g.started == 1 and c.released == 1 and g.released == 1);
}

/// A graph backend of `workers` slots: packet `v:n` prints v after n polls (`!` a budget end); start order is kept.
const Slots = struct {
    workers: u32,
    jobs: [16]struct { body: []const u8 = "", left: u32 = 0, live: bool = false } = @splat(.{}),
    started: [16][]const u8 = undefined,
    n: u32 = 0,
    most: u32 = 0,
    cancelled: u32 = 0,

    fn backend(f: *Slots) api.Backend {
        return .{ .ctx = f, .vtable = &.{ .start = st, .poll = pl, .cancel = cn, .release = rl, .now = nw } };
    }
    fn live(f: *Slots) u32 {
        var k: u32 = 0;
        for (f.jobs) |j| k += @intFromBool(j.live);
        return k;
    }
    fn st(ctx: *anyopaque, _: u64, p: api.Program, _: u64) error{ OutOfMemory, Busy }!api.JobId {
        const f: *Slots = @ptrCast(@alignCast(ctx));
        if (f.live() >= f.workers) return error.Busy;
        const colon = std.mem.indexOfScalar(u8, p.bytes, ':').?;
        f.jobs[f.n] = .{ .body = p.bytes[0..colon], .left = std.fmt.parseInt(u32, p.bytes[colon + 1 ..], 10) catch unreachable, .live = true };
        f.started[f.n] = p.bytes;
        f.n += 1;
        f.most = @max(f.most, f.live());
        return f.n - 1;
    }
    fn pl(ctx: *anyopaque, job: api.JobId) api.Poll {
        const f: *Slots = @ptrCast(@alignCast(ctx));
        const j = &f.jobs[@intCast(job)];
        if (j.left > 0) {
            j.left -= 1;
            return .running;
        }
        j.live = false;
        if (std.mem.eql(u8, j.body, "!")) return .{ .done = .{ .status = .budget, .steps = 10 } };
        return .{ .done = .{ .status = .ok, .out = j.body, .steps = 10 } };
    }
    fn cn(ctx: *anyopaque, job: api.JobId) void {
        const f: *Slots = @ptrCast(@alignCast(ctx));
        f.jobs[@intCast(job)].live = false;
        f.cancelled += 1;
    }
    fn rl(_: *anyopaque, _: u64) void {}
    fn nw(_: *anyopaque, _: api.Program, _: u64, _: u64) ?api.Result {
        return null;
    }
};

fn finish(b: api.Backend, id: api.JobId) api.Result {
    while (true) switch (b.poll(id)) {
        .running => {},
        .done => |r| return r,
    };
}

test "runner graph dispatch: a sequence's pieces run side by side, largest first, and print in order as in turn" {
    var c: Slots = .{ .workers = 1 };
    var g: Slots = .{ .workers = 2 };
    var d: Dispatch = .{ .gpa = std.testing.allocator, .circuit = c.backend(), .graph = g.backend() };
    defer d.deinit();
    const b = d.backend();
    const r = finish(b, try b.start(1, .{ .kind = .graph, .bytes = "TFGS1\nIS 7:1\nLJ (\nIN 1155:6\nBN 1:33\n" }, 100));
    try std.testing.expectEqual(api.Status.ok, r.status);
    try std.testing.expectEqualStrings("7 (1155\nTrue\n", r.out);
    try std.testing.expectEqual(@as(u64, 30), r.steps);
    try std.testing.expectEqual(@as(u32, 2), g.most);
    // the two largest packets went first; the smallest waited for a free worker
    try std.testing.expectEqualStrings("1155:6", g.started[0]);
    try std.testing.expectEqualStrings("1:33", g.started[1]);
    try std.testing.expectEqualStrings("7:1", g.started[2]);
    b.release(1);
}

test "runner graph dispatch: the first piece in print order that is not ok ends the run; later ones are cancelled" {
    var c: Slots = .{ .workers = 1 };
    var g: Slots = .{ .workers = 3 };
    var d: Dispatch = .{ .gpa = std.testing.allocator, .circuit = c.backend(), .graph = g.backend() };
    defer d.deinit();
    const b = d.backend();
    const r = finish(b, try b.start(1, .{ .kind = .graph, .bytes = "TFGS1\nIS 7:0\nIN !:0\nIN 123456:9\n" }, 100));
    try std.testing.expectEqual(api.Status.budget, r.status);
    try std.testing.expectEqualStrings("7 ", r.out);
    try std.testing.expectEqual(@as(u32, 1), g.cancelled);
    try std.testing.expectEqual(@as(u32, 0), g.live());
    // pieces that each fit but together pass the budget end as a budget, as they would have in turn
    const over = finish(b, try b.start(2, .{ .kind = .graph, .bytes = "TFGS1\nIS 1:0\nIS 2:0\nIN 3:0\n" }, 25));
    try std.testing.expectEqual(api.Status.budget, over.status);
    try std.testing.expectEqualStrings("1 2 ", over.out);
    // a malformed line is a fault where it is reached; the value a kind cannot print is one too
    try std.testing.expectEqual(api.Status.fault, finish(b, try b.start(3, .{ .kind = .graph, .bytes = "TFGS1\nIS 1:0\nbad\n" }, 100)).status);
    try std.testing.expectEqual(api.Status.fault, finish(b, try b.start(4, .{ .kind = .graph, .bytes = "TFGS1\nBN 7:0\n" }, 100)).status);
    b.release(1);
    b.release(2);
}

test "runner graph dispatch: a TFGS2 template waits for the piece it reads; no bank or an unknown version faults" {
    var c: Slots = .{ .workers = 1 };
    var g: Slots = .{ .workers = 2 };
    var d: Dispatch = .{ .gpa = std.testing.allocator, .circuit = c.backend(), .graph = g.backend() };
    defer d.deinit();
    const b = d.backend();
    const id = try b.start(1, .{ .kind = .graph, .bytes = "TFGS2\nIS 7:2\nBN @=$0;7\n" }, 100);
    // two workers, yet only the read piece runs: the template has no value to bind until it ends
    try std.testing.expect(b.poll(id) == .running);
    try std.testing.expectEqual(@as(u32, 1), g.n);
    const r = finish(b, id);
    try std.testing.expectEqual(api.Status.fault, r.status);
    try std.testing.expectEqualStrings("7 ", r.out);
    try std.testing.expectEqual(@as(u32, 1), g.n);
    // a version this runner does not know is refused, never read as TFGS1 or as one packet
    const later = finish(b, try b.start(2, .{ .kind = .graph, .bytes = "TFGS3\nIS 7:0\n" }, 100));
    try std.testing.expectEqual(api.Status.fault, later.status);
    try std.testing.expectEqualStrings("", later.out);
    try std.testing.expectEqual(@as(u32, 1), g.n);
    // in TFGS1 a body starting `@` is a packet like any other: it runs
    _ = finish(b, try b.start(3, .{ .kind = .graph, .bytes = "TFGS1\nIN @:0\n" }, 100));
    try std.testing.expectEqualStrings("@:0", g.started[1]);
    b.release(1);
    b.release(2);
    b.release(3);
}
