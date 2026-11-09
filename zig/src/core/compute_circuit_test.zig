//! S3 (core/compute_circuit.zig) with a scripted frontend and backend (the arm-d proof: compute_circuit_proof_test).
const std = @import("std");
const api = @import("runner_api.zig");
const s3 = @import("compute_circuit.zig");
const lane = @import("compute_lane.zig");

const gpa = std.testing.allocator;

/// Code "abstain:<reason>" abstains; other code is a calc program whose bytes are the code (the backend's script).
const Front = struct {
    cells: u32 = 0,
    last: api.Cell = .{ .form = .repl, .lang = "", .code = "" },

    fn frontend(f: *Front) api.Frontend {
        return .{ .ctx = f, .translate = translate };
    }

    fn translate(ctx: ?*anyopaque, a: std.mem.Allocator, cell: api.Cell) std.mem.Allocator.Error!api.Translation {
        _ = a;
        const f: *Front = @ptrCast(@alignCast(ctx.?));
        f.cells += 1;
        f.last = cell;
        if (std.mem.startsWith(u8, cell.code, "abstain:")) return .{ .abstain = std.meta.stringToEnum(api.Abstain, cell.code["abstain:".len..]).? };
        if (std.mem.startsWith(u8, cell.code, "elf:")) return .{ .program = .{ .kind = .elf, .bytes = cell.code["elf:".len..] } };
        return .{ .program = .{ .kind = .calc, .bytes = cell.code } };
    }
};

/// Program bytes script the run: ok:<out>, budget:<out>, fault:<out>, abstain, forever, else ok; `busy` refuses starts.
const Back = struct {
    busy: u32 = 0,
    started: u32 = 0,
    cancelled: u32 = 0,
    released: u32 = 0,
    script: []const u8 = "",

    fn backend(b: *Back) api.Backend {
        return .{ .ctx = b, .vtable = &.{ .start = start, .poll = poll, .cancel = cancel, .release = release } };
    }

    fn start(ctx: *anyopaque, stream: u64, p: api.Program, budget: u64) error{ OutOfMemory, Busy }!api.JobId {
        _ = .{ stream, budget };
        const b: *Back = @ptrCast(@alignCast(ctx));
        if (b.busy > 0) {
            b.busy -= 1;
            return error.Busy;
        }
        b.started += 1;
        b.script = p.bytes;
        return b.started;
    }

    fn poll(ctx: *anyopaque, job: api.JobId) api.Poll {
        _ = job;
        const b: *Back = @ptrCast(@alignCast(ctx));
        const colon = std.mem.findScalar(u8, b.script, ':') orelse b.script.len;
        const out = if (colon < b.script.len) b.script[colon + 1 ..] else "";
        const head = b.script[0..colon];
        if (std.mem.eql(u8, head, "forever")) return .running;
        if (std.mem.eql(u8, head, "abstain")) return .{ .done = .{ .status = .abstain, .abstain = .too_big } };
        const status = std.meta.stringToEnum(api.Status, head) orelse .ok; // a real cell: runs ok
        return .{ .done = .{ .status = status, .out = out, .steps = 7, .exec_ns = 1000 } };
    }

    fn cancel(ctx: *anyopaque, job: api.JobId) void {
        _ = job;
        const b: *Back = @ptrCast(@alignCast(ctx));
        b.cancelled += 1;
    }

    fn release(ctx: *anyopaque, stream: u64) void {
        _ = stream;
        const b: *Back = @ptrCast(@alignCast(ctx));
        b.released += 1;
    }
};

const Fixture = struct {
    front: Front = .{},
    back: Back = .{},
    arena: std.heap.ArenaAllocator = .init(gpa),

    fn exec(f: *Fixture, cfg: s3.Config) s3.Exec {
        return .{ .frontend = f.front.frontend(), .backend = f.back.backend(), .max_steps = 1000, .cfg = cfg };
    }

    fn run(f: *Fixture, cfg: s3.Config, code: []const u8, hook: ?lane.Cancel) !s3.Outcome {
        var e = f.exec(cfg);
        return e.run(f.arena.allocator(), code, hook);
    }
};

fn expectOwned(want: []const u8, got: s3.Outcome) !void {
    try std.testing.expect(got == .owned);
    try std.testing.expectEqualStrings(want, got.owned.body);
}

fn expectDeclined(want: s3.Decline, got: s3.Outcome) !void {
    try std.testing.expect(got == .declined);
    try std.testing.expectEqual(want, got.declined);
}

test "compute runner: a run that halts ok is the lane's body: one trailing newline cut, nothing printed is [no output]" {
    var f: Fixture = .{};
    defer f.arena.deinit();
    try expectOwned("42", try f.run(.{}, "ok:42\n", null));
    try expectOwned("1\n2", try f.run(.{}, "ok:1\n2\n", null));
    try expectOwned("[no output]", try f.run(.{}, "ok:", null));
    // the sandbox's own rule on the same stdout gives the same bytes
    const sandbox = try lane.runner.body(f.arena.allocator(), .{ .status = .ok, .stdout = "830876824090243401591366835513\n" }, 0, 0);
    try expectOwned(sandbox, try f.run(.{}, "ok:830876824090243401591366835513\n", null));
    const o = try f.run(.{}, "ok:x\n", null);
    try std.testing.expectEqual(s3.Kind.calc, o.owned.kind);
    try std.testing.expectEqual(@as(u64, 7), o.owned.steps);
}

test "compute runner: an ELF program runs only with _ELF on; off, it declines as elf and starts nothing" {
    var f: Fixture = .{};
    defer f.arena.deinit();
    try expectDeclined(.elf, try f.run(.{}, "elf:ok:42\n", null));
    try std.testing.expectEqual(@as(u32, 0), f.back.started);
    const o = try f.run(.{ .elf = true }, "elf:ok:42\n", null);
    try expectOwned("42", o);
    try std.testing.expectEqual(s3.Kind.elf, o.owned.kind);
}

test "compute runner: a lane call is a fresh python fence cell: no earlier call's names" {
    var f: Fixture = .{};
    defer f.arena.deinit();
    _ = try f.run(.{}, "ok:1\n", null);
    _ = try f.run(.{}, "ok:2\n", null);
    try std.testing.expectEqual(api.Form.fence, f.front.last.form);
    try std.testing.expectEqualStrings("python", f.front.last.lang);
    try std.testing.expectEqual(@as(usize, 0), f.front.last.prior.len);
}

test "compute runner: every frontend abstain declines with its reason and starts nothing" {
    var f: Fixture = .{};
    defer f.arena.deinit();
    inline for (@typeInfo(api.Abstain).@"enum".field_names) |name| {
        try expectDeclined(@field(s3.Decline, name), try f.run(.{}, "abstain:" ++ name, null));
    }
    try std.testing.expectEqual(@as(u32, 0), f.back.started);
}

test "compute runner: backend abstain, budget and fault decline; their partial output is never a body" {
    var f: Fixture = .{};
    defer f.arena.deinit();
    try expectDeclined(.too_big, try f.run(.{}, "abstain", null));
    try expectDeclined(.budget, try f.run(.{}, "budget:12", null));
    try expectDeclined(.fault, try f.run(.{}, "fault:12", null));
    try std.testing.expectEqual(@as(u32, 3), f.back.released);
}

test "compute runner: a busy backend is retried until a slot frees; past busy_ms the call declines busy" {
    var f: Fixture = .{};
    defer f.arena.deinit();
    f.back.busy = 3;
    try expectOwned("5", try f.run(.{}, "ok:5\n", null));
    f.back.busy = 1 << 30;
    try expectDeclined(.busy, try f.run(.{ .busy_ms = 5 }, "ok:5\n", null));
    try std.testing.expectEqual(@as(u32, 1), f.back.started);
}

test "compute runner: a run past wall_ms is cancelled and declines wall; a gone client declines cancelled" {
    var f: Fixture = .{};
    defer f.arena.deinit();
    try expectDeclined(.wall, try f.run(.{ .wall_ms = 5 }, "forever", null));
    try std.testing.expectEqual(@as(u32, 1), f.back.cancelled);
    const Gone = struct {
        fn yes(ctx: *anyopaque) bool {
            _ = ctx;
            return true;
        }
    };
    var dummy: u8 = 0;
    try expectDeclined(.cancelled, try f.run(.{}, "forever", .{ .ctx = &dummy, .gone = Gone.yes }));
    try std.testing.expectEqual(@as(u32, 2), f.back.cancelled);
    try std.testing.expectEqual(@as(u32, 2), f.back.released);
}

test "compute runner: S3 knobs: defaults, values and bad values" {
    const Env = struct {
        var pairs: []const [2][]const u8 = &.{};
        fn get(name: []const u8) ?[]const u8 {
            for (pairs) |p| if (std.mem.eql(u8, p[0], name)) return p[1];
            return null;
        }
    };
    Env.pairs = &.{};
    try std.testing.expectEqual(s3.Config{}, try s3.Config.parse(Env.get, "X"));
    Env.pairs = &.{ .{ "X_WALL_MS", "50" }, .{ "X_BUSY_MS", "0" }, .{ "X_VERIFY", "1" }, .{ "X_ELF", "1" } };
    try std.testing.expectEqual(s3.Config{ .wall_ms = 50, .busy_ms = 0, .verify = true, .elf = true }, try s3.Config.parse(Env.get, "X"));
    Env.pairs = &.{.{ "X_VERIFY", "2" }};
    try std.testing.expectError(error.BadKnob, s3.Config.parse(Env.get, "X"));
    Env.pairs = &.{.{ "X_WALL_MS", "0" }};
    try std.testing.expectError(error.BadKnob, s3.Config.parse(Env.get, "X"));
}

test "compute runner: S3's switch sits under the lane's names, off by default and off without the lane" {
    const Env = struct {
        var pairs: []const [2][]const u8 = &.{};
        fn get(name: []const u8) ?[]const u8 {
            for (pairs) |kv| if (std.mem.eql(u8, kv[0], name)) return kv[1];
            return null;
        }
    };
    const lanes = @import("compute_binding.zig");
    try std.testing.expectEqualStrings("TF_COMPUTE_LANE_RUNNER", s3.binding(.{}).prefix);
    try std.testing.expectEqualStrings("TF_DSV41_COMPUTE_LANE_RUNNER", s3.binding(lanes.dsv41).alias.?);
    Env.pairs = &.{ .{ "TF_COMPUTE_LANE", "1" }, .{ "TF_RUNNER", "1" }, .{ "TF_RUNNER_CIRCUIT", "/c.tfpc" } };
    try std.testing.expectEqual(null, try s3.open(.{}, gpa, &Env.get, true, null, null)); // the lane and runner on: S3 off
    Env.pairs = &.{ .{ "TF_COMPUTE_LANE_RUNNER", "1" }, .{ "TF_COMPUTE_LANE_RUNNER_CIRCUIT", "/c.tfpc" } };
    try std.testing.expectEqual(null, try s3.open(.{}, gpa, &Env.get, false, null, null)); // no lane: nothing to run
    Env.pairs = &.{ .{ "TF_COMPUTE_LANE_RUNNER", "1" }, .{ "TF_DSV41_COMPUTE_LANE_RUNNER", "1" } };
    try std.testing.expectError(error.BadKnob, s3.open(lanes.dsv41, gpa, &Env.get, true, null, null));
    // DeepSeek-V4.1's switch reads S3's knobs under its own names (a bad one stops the boot before a runner opens)
    Env.pairs = &.{ .{ "TF_DSV41_COMPUTE_LANE_RUNNER", "1" }, .{ "TF_DSV41_COMPUTE_LANE_RUNNER_CIRCUIT", "/c.tfpc" }, .{ "TF_DSV41_COMPUTE_LANE_RUNNER_WALL_MS", "0" } };
    try std.testing.expectError(error.BadKnob, s3.open(lanes.dsv41, gpa, &Env.get, true, null, null));
}
