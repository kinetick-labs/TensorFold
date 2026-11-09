//! The CPU Backend: each job boots a fresh machine on its own worker thread; the engine thread never waits on a run.
const std = @import("std");
const api = @import("runner_api.zig");
const rc = @import("runner_circuit.zig");
const place = @import("runner_place.zig");

const Allocator = std.mem.Allocator;
const log = std.log.scoped(.runner_circuit);
const too_long: [300]u8 = @splat('9');

const Job = struct {
    id: api.JobId,
    stream: u64,
    program: []u8,
    kind: @FieldType(api.Program, "kind"),
    budget: u64,
    thread: ?std.Thread = null,
    done: std.atomic.Value(bool) = .init(false),
    cancel: std.atomic.Value(bool) = .init(false),
    result: api.Result = .{ .status = .abstain, .abstain = .unsupported },
    out: std.ArrayList(u8) = .empty,
};

const Cpu = struct {
    gpa: Allocator,
    circuit: *const rc.Circuit,
    workers: u8,
    placer: ?*place.Placer = null,
    next_id: api.JobId = 1,
    jobs: std.ArrayList(*Job) = .empty,
    dropped: std.ArrayList(*Job) = .empty,
    delivered: std.ArrayList(*Job) = .empty,

    fn live(c: *Cpu) usize {
        return c.jobs.items.len + c.dropped.items.len;
    }

    fn free(c: *Cpu, j: *Job) void {
        if (j.thread) |t| t.join();
        j.out.deinit(c.gpa);
        c.gpa.free(j.program);
        c.gpa.destroy(j);
    }

    /// Joins and frees cancelled jobs whose worker has stopped.
    fn reap(c: *Cpu) void {
        var i: usize = 0;
        while (i < c.dropped.items.len) {
            const j = c.dropped.items[i];
            if (!j.done.load(.acquire)) {
                i += 1;
                continue;
            }
            _ = c.dropped.swapRemove(i);
            c.free(j);
        }
    }

    /// `out` of a stream's previous result stays valid only until the next call for that stream.
    fn forget(c: *Cpu, stream: u64) void {
        var i: usize = 0;
        while (i < c.delivered.items.len) {
            if (c.delivered.items[i].stream != stream) {
                i += 1;
                continue;
            }
            c.free(c.delivered.swapRemove(i));
        }
    }

    fn find(c: *Cpu, id: api.JobId) ?usize {
        for (c.jobs.items, 0..) |j, i| if (j.id == id) return i;
        return null;
    }

    fn drop(c: *Cpu, i: usize) void {
        const j = c.jobs.swapRemove(i);
        j.cancel.store(true, .release);
        c.dropped.append(c.gpa, j) catch {
            if (j.thread) |t| t.join();
            c.free(j);
        };
    }
};

fn work(c: *Cpu, j: *Job) void {
    defer j.done.store(true, .release);
    if (c.placer) |p| p.enter();
    const t0 = rc.nowNs();
    var m = rc.Machine.boot(c.gpa, c.circuit, .{ .kind = j.kind, .bytes = j.program }) catch |e| {
        j.result = .{ .status = .abstain, .abstain = switch (e) {
            error.CircuitInputTooLong => .too_long,
            error.CircuitInputNul => .syntax,
            else => .unsupported,
        } };
        if (e != error.CircuitInputTooLong) log.warn("circuit boot refused: {s}", .{@errorName(e)});
        return;
    };
    defer m.deinit(c.gpa);
    const end = m.run(j.budget, &j.out, null, &j.cancel) catch |e| {
        log.warn("circuit run failed after {d} steps: {s}", .{ m.state.steps, @errorName(e) });
        j.result = .{ .status = .abstain, .abstain = .unsupported, .steps = m.state.steps };
        return;
    };
    j.result = .{
        .status = switch (end.end) {
            .halted => .ok,
            .budget => .budget,
            .fault => .fault,
            .cancelled => .abstain,
        },
        .steps = end.steps,
        .exec_ns = rc.nowNs() - t0,
        .abstain = if (end.end == .cancelled) .unsupported else null,
    };
}

fn start(ctx: *anyopaque, stream: u64, program: api.Program, budget: u64) error{ OutOfMemory, Busy }!api.JobId {
    const c: *Cpu = @ptrCast(@alignCast(ctx));
    c.reap();
    c.forget(stream);
    if (c.live() >= c.workers) return error.Busy;
    const j = try c.gpa.create(Job);
    errdefer c.gpa.destroy(j);
    const bytes = try c.gpa.dupe(u8, program.bytes);
    errdefer c.gpa.free(bytes);
    j.* = .{ .id = c.next_id, .stream = stream, .program = bytes, .kind = program.kind, .budget = budget };
    try c.jobs.ensureUnusedCapacity(c.gpa, 1);
    j.thread = std.Thread.spawn(.{}, work, .{ c, j }) catch return error.Busy;
    c.jobs.appendAssumeCapacity(j);
    c.next_id += 1;
    return j.id;
}

fn poll(ctx: *anyopaque, id: api.JobId) api.Poll {
    const c: *Cpu = @ptrCast(@alignCast(ctx));
    c.reap();
    const i = c.find(id) orelse return .{ .done = .{ .status = .abstain, .abstain = .unsupported } };
    const j = c.jobs.items[i];
    if (!j.done.load(.acquire)) return .running;
    _ = c.jobs.swapRemove(i);
    if (j.thread) |t| t.join();
    j.thread = null;
    c.forget(j.stream);
    c.delivered.append(c.gpa, j) catch {
        c.free(j);
        return .{ .done = .{ .status = .abstain, .abstain = .unsupported } };
    };
    var r = j.result;
    r.out = j.out.items;
    return .{ .done = r };
}

fn cancel(ctx: *anyopaque, id: api.JobId) void {
    const c: *Cpu = @ptrCast(@alignCast(ctx));
    if (c.find(id)) |i| c.drop(i);
    c.reap();
}

fn release(ctx: *anyopaque, stream: u64) void {
    const c: *Cpu = @ptrCast(@alignCast(ctx));
    var i: usize = 0;
    while (i < c.jobs.items.len) {
        if (c.jobs.items[i].stream == stream) c.drop(i) else i += 1;
    }
    c.forget(stream);
    c.reap();
}

const vtable: api.Backend.VTable = .{ .start = start, .poll = poll, .cancel = cancel, .release = release };

/// A CPU backend running at most `workers` machines at once over a circuit the caller keeps alive.
pub fn backend(gpa: Allocator, circuit: *const rc.Circuit, workers: u8) !api.Backend {
    return placed(gpa, circuit, workers, null);
}

/// As `backend`, each job's thread placed by `placer` (which the caller keeps alive) before it boots a machine.
pub fn placed(gpa: Allocator, circuit: *const rc.Circuit, workers: u8, placer: ?*place.Placer) !api.Backend {
    if (workers == 0) return error.NoWorkers;
    const c = try gpa.create(Cpu);
    c.* = .{ .gpa = gpa, .circuit = circuit, .workers = workers, .placer = placer };
    return .{ .ctx = c, .vtable = &vtable };
}

/// Cancels every job, waits for the workers and frees the backend.
pub fn destroy(b: api.Backend) void {
    const c: *Cpu = @ptrCast(@alignCast(b.ctx));
    for (c.jobs.items) |j| j.cancel.store(true, .release);
    for ([_]*std.ArrayList(*Job){ &c.jobs, &c.dropped, &c.delivered }) |list| {
        for (list.items) |j| c.free(j);
        list.deinit(c.gpa);
    }
    c.gpa.destroy(c);
}

test "the backend runs jobs off the caller's thread, reports Busy at its worker limit and frees on release" {
    const tiny = @import("runner_circuit_tiny_test.zig");
    const gpa = std.testing.allocator;
    const bytes = try tiny.build(gpa, .good);
    defer gpa.free(bytes);
    var circuit = try @import("runner_circuit_load.zig").fromBytes(gpa, bytes);
    defer circuit.deinit(gpa);
    const b = try backend(gpa, &circuit, 1);
    defer destroy(b);
    const id = try b.start(7, .{ .kind = .calc, .bytes = "1+1" }, 100);
    try std.testing.expectError(error.Busy, b.start(8, .{ .kind = .calc, .bytes = "2" }, 100));
    var result: api.Result = undefined;
    while (true) switch (b.poll(id)) {
        .running => std.Thread.yield() catch {},
        .done => |r| {
            result = r;
            break;
        },
    };
    try std.testing.expectEqual(api.Status.ok, result.status);
    try std.testing.expectEqualStrings("hi", result.out);
    try std.testing.expectEqual(@as(u64, 3), result.steps);
    const capped = try b.start(8, .{ .kind = .calc, .bytes = "" }, 1);
    while (true) switch (b.poll(capped)) {
        .running => std.Thread.yield() catch {},
        .done => |r| {
            try std.testing.expectEqual(api.Status.budget, r.status);
            try std.testing.expectEqualStrings("h", r.out);
            break;
        },
    };
    const long = try b.start(9, .{ .kind = .calc, .bytes = &too_long }, 100);
    while (true) switch (b.poll(long)) {
        .running => std.Thread.yield() catch {},
        .done => |r| {
            try std.testing.expectEqual(api.Abstain.too_long, r.abstain.?);
            break;
        },
    };
    b.release(7);
    b.release(8);
    b.release(9);
}
