//! The runner's thread placement: knobs, the auto choice, and real runner threads on their CPUs at their niceness.
const std = @import("std");
const builtin = @import("builtin");
const place = @import("runner_place.zig");
const runner = @import("runner.zig");
const rt = @import("runner_test.zig");
const backend = @import("runner_circuit_backend.zig");
const linux = std.os.linux;

const gpa = std.testing.allocator;

fn env(pairs: []const [2][]const u8) *const fn ([]const u8) ?[]const u8 {
    const S = struct {
        var vars: []const [2][]const u8 = &.{};
        fn get(name: []const u8) ?[]const u8 {
            for (vars) |p| if (std.mem.eql(u8, p[0], name)) return p[1];
            return null;
        }
    };
    S.vars = pairs;
    return S.get;
}

fn list(text: []const u8) place.Set {
    var set: place.Set = undefined;
    std.debug.assert(@import("compute_sandbox.zig").cpuList(text, &set));
    return set;
}

fn show(set: *const place.Set) []const u8 {
    const S = struct {
        var buf: [256]u8 = undefined;
    };
    return place.format(set, &S.buf);
}

test "runner placement: knobs parse auto, inherit, a CPU list and a niceness; bad values refuse" {
    const auto = try runner.Config.parse(env(&.{ .{ "R", "1" }, .{ "R_CIRCUIT", "c" } }), "R");
    try std.testing.expect(auto.?.place.cpus == .auto);
    try std.testing.expectEqual(@as(u8, 10), auto.?.place.nice);
    const inherit = try place.Config.parse(env(&.{.{ "R_CPUS", "inherit" }}), "R");
    try std.testing.expect(inherit.cpus == .inherit);
    const pinned = try place.Config.parse(env(&.{ .{ "R_CPUS", "0-4,10-14" }, .{ "R_NICE", "15" } }), "R");
    try std.testing.expectEqualStrings("0-4,10-14", show(&pinned.cpus.list));
    try std.testing.expectEqual(@as(u8, 15), pinned.nice);
    for ([_][2][]const u8{ .{ "R_CPUS", "4-2" }, .{ "R_CPUS", "big" }, .{ "R_NICE", "20" }, .{ "R_NICE", "-1" } }) |kv|
        try std.testing.expectError(error.BadKnob, place.Config.parse(env(&.{kv}), "R"));
}

/// GB10's Grace: cores 0-4 and 10-14 small (718, 731), 5-9 and 15-19 big (997-1024), 19 the plan link's auto pin.
fn gb10(cpu: u32) ?u32 {
    return switch (cpu) {
        0...4 => 718,
        10...14 => 731,
        5...9 => 997,
        15...18 => 1017,
        19 => 1024,
        else => null,
    };
}

fn uniform(_: u32) ?u32 {
    return 1024;
}

test "runner placement: auto keeps off the engine's CPUs and, where core sizes differ, off the big cores" {
    const all = list("0-19");
    var engine: place.Set = @splat(0);
    place.add(&engine, 19);
    try std.testing.expectEqualStrings("0-4,10-14", show(&place.choose(&all, &engine, gb10)));
    // a container given big cores only: they are all one size, so everything but the engine's
    try std.testing.expectEqualStrings("5-9,15-18", show(&place.choose(&list("5-9,15-19"), &engine, gb10)));
    try std.testing.expectEqualStrings("0-18", show(&place.choose(&all, &engine, uniform)));
    try std.testing.expect(place.empty(&place.choose(&list("19"), &engine, uniform)));
    // the real machine: never a CPU outside the container's, never the engine's
    const allowed = place.allowedCpus();
    var avoid: place.Set = @splat(0);
    place.add(&avoid, 0);
    const got = place.choose(&allowed, &avoid, place.capacity);
    for (0..1024) |c| if (place.has(&got, @intCast(c))) {
        try std.testing.expect(place.has(&allowed, @intCast(c)) and c != 0);
    };
}

/// The calling thread's CPU mask, niceness and name.
const Seen = struct {
    mask: place.Set = @splat(0),
    nice: isize = 0,
    name: [16]u8 = @splat(0),

    fn now() Seen {
        var s: Seen = .{};
        _ = linux.syscall3(.sched_getaffinity, 0, @sizeOf(place.Set), @intFromPtr(&s.mask));
        s.nice = 20 - @as(isize, @bitCast(linux.syscall2(.getpriority, 0, 0)));
        _ = linux.prctl(@backingInt(linux.PR.GET_NAME), @intFromPtr(&s.name), 0, 0, 0);
        return s;
    }

    fn named(s: *const Seen) []const u8 {
        return std.mem.sliceTo(&s.name, 0);
    }
};

/// Two CPUs this test may use: the engine's stand-in and the runner's; null on a one-CPU machine.
fn twoCpus() ?[2]u32 {
    var mask: place.Set = @splat(0);
    _ = linux.syscall3(.sched_getaffinity, 0, @sizeOf(place.Set), @intFromPtr(&mask));
    var first: ?u32 = null;
    var last: u32 = 0;
    for (0..1024) |c| if (place.has(&mask, @intCast(c))) {
        if (first == null) first = @intCast(c);
        last = @intCast(c);
    };
    const f = first orelse return null;
    return if (f == last) null else .{ f, last };
}

fn pinSelf(cpu: u32) void {
    var set: place.Set = @splat(0);
    place.add(&set, cpu);
    _ = linux.syscall3(.sched_setaffinity, 0, @sizeOf(place.Set), @intFromPtr(&set));
}

test "runner placement: a thread the pinned engine starts runs on the runner's CPUs at nice 10 or more, not the engine's" {
    if (builtin.os.tag != .linux) return error.SkipZigTest;
    const cpus = twoCpus() orelse return error.SkipZigTest;
    var p: place.Placer = .{ .cfg = .{ .cpus = .{ .list = @splat(0) } } };
    place.add(&p.cfg.cpus.list, cpus[1]);
    const Engine = struct {
        fn child(pl: ?*place.Placer, out: *Seen) void {
            if (pl) |x| x.enter();
            out.* = Seen.now();
        }
        // the engine thread: pinned (as the plan link pins it), then it starts runner threads
        fn run(cpu: u32, pl: *place.Placer, placed: *Seen, unplaced: *Seen, own: *Seen) void {
            pinSelf(cpu);
            const a = std.Thread.spawn(.{}, child, .{ pl, placed }) catch return;
            a.join();
            const b = std.Thread.spawn(.{}, child, .{ null, unplaced }) catch return;
            b.join();
            own.* = Seen.now();
        }
    };
    var placed: Seen = .{};
    var unplaced: Seen = .{};
    var own: Seen = .{};
    const t = try std.Thread.spawn(.{}, Engine.run, .{ cpus[0], &p, &placed, &unplaced, &own });
    t.join();
    var want: place.Set = @splat(0);
    place.add(&want, cpus[1]);
    var engine: place.Set = @splat(0);
    place.add(&engine, cpus[0]);
    try std.testing.expectEqualSlices(u64, &want, &placed.mask);
    try std.testing.expect(placed.nice >= 10);
    try std.testing.expectEqualStrings("tf-runner", placed.named());
    // the control: without the placer a runner thread inherits the engine's CPU and niceness
    try std.testing.expectEqualSlices(u64, &engine, &unplaced.mask);
    // and the engine thread itself is untouched
    try std.testing.expectEqualSlices(u64, &engine, &own.mask);
    try std.testing.expectEqual(own.nice, unplaced.nice);
    try std.testing.expectEqual(@as(u32, 1), p.entered.load(.acquire));
}

/// Every placed thread's mask must equal the placer's set: counts the threads that matched and that did not.
const Watch = struct {
    var good: std.atomic.Value(u32) = .init(0);
    var bad: std.atomic.Value(u32) = .init(0);

    fn check(p: *place.Placer) void {
        const s = Seen.now();
        const ok = std.mem.eql(u64, &s.mask, &p.cfg.cpus.list) and s.nice >= p.cfg.nice;
        _ = (if (ok) &good else &bad).fetchAdd(1, .monotonic);
    }

    fn reset() void {
        good.store(0, .monotonic);
        bad.store(0, .monotonic);
    }
};

fn placerOn(cpu: u32) place.Placer {
    var p: place.Placer = .{ .cfg = .{ .cpus = .{ .list = @splat(0) } }, .on_enter = Watch.check };
    place.add(&p.cfg.cpus.list, cpu);
    return p;
}

test "runner placement: the CPU backend's job threads run placed" {
    if (builtin.os.tag != .linux) return error.SkipZigTest;
    const cpus = twoCpus() orelse return error.SkipZigTest;
    Watch.reset();
    var p = placerOn(cpus[1]);
    const bytes = try @import("runner_circuit_tiny_test.zig").build(gpa, .good);
    defer gpa.free(bytes);
    var circuit = try @import("runner_circuit_load.zig").fromBytes(gpa, bytes);
    defer circuit.deinit(gpa);
    const b = try backend.placed(gpa, &circuit, 2, &p);
    defer backend.destroy(b);
    for (0..3) |i| {
        const id = try b.start(i, .{ .kind = .calc, .bytes = "" }, 100);
        while (b.poll(id) == .running) std.Thread.yield() catch {};
    }
    try std.testing.expectEqual(@as(u32, 3), Watch.good.load(.acquire));
    try std.testing.expectEqual(@as(u32, 0), Watch.bad.load(.acquire));
}

test "runner placement: a reply's translation runs placed and the reply is unchanged" {
    if (builtin.os.tag != .linux) return error.SkipZigTest;
    const cpus = twoCpus() orelse return error.SkipZigTest;
    Watch.reset();
    var p = placerOn(cpus[1]);
    const inherited = try rt.one(rt.mul_cell, .{});
    defer gpa.free(inherited);
    const placed = try rt.one(rt.mul_cell, .{ .placer = &p });
    defer gpa.free(placed);
    try std.testing.expectEqualStrings(inherited, placed);
    try std.testing.expect(std.mem.indexOf(u8, placed, rt.product) != null);
    try std.testing.expect(Watch.good.load(.acquire) >= 1);
    try std.testing.expectEqual(@as(u32, 0), Watch.bad.load(.acquire));
}
