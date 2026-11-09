//! Where the runner's CPU threads run (not the engine thread's inherited pin): a CPU set and niceness off the engine's.
const std = @import("std");
const builtin = @import("builtin");
const sandbox = @import("compute_sandbox.zig");
const linux = std.os.linux;
const log = std.log.scoped(.runner);

/// A CPU mask (1024 CPUs), the layout sched_setaffinity and compute_sandbox.cpuList use.
pub const Set = [16]u64;

pub const Config = struct {
    /// <prefix>_CPUS: auto (unset), inherit (the spawning thread's mask, as before), or a list like 0-4,10-14
    cpus: union(enum) { auto, inherit, list: Set } = .auto,
    /// <prefix>_NICE: the least niceness a runner thread runs at (never below the engine's own)
    nice: u8 = 10,
    /// The CPUs the engine's own threads are pinned to (plan link, RoCE proxy); auto keeps off them. Engine-set.
    engine: ?*const fn (set: *Set) void = null,

    pub fn parse(get: *const fn ([]const u8) ?[]const u8, comptime prefix: []const u8) error{BadKnob}!Config {
        var c: Config = .{};
        const cpus = std.mem.trim(u8, get(prefix ++ "_CPUS") orelse "", " ");
        if (std.mem.eql(u8, cpus, "inherit")) c.cpus = .inherit else if (cpus.len > 0 and !std.mem.eql(u8, cpus, "auto")) {
            var set: Set = undefined;
            if (!sandbox.cpuList(cpus, &set)) return bad(prefix ++ "_CPUS", cpus);
            c.cpus = .{ .list = set };
        }
        const nice = std.mem.trim(u8, get(prefix ++ "_NICE") orelse "", " ");
        if (nice.len > 0) {
            c.nice = std.fmt.parseInt(u8, nice, 10) catch return bad(prefix ++ "_NICE", nice);
            if (c.nice > 19) return bad(prefix ++ "_NICE", nice);
        }
        return c;
    }

    fn bad(name: []const u8, value: []const u8) error{BadKnob} {
        log.warn("{s}={s} is not a valid value (auto, inherit or a CPU list like 0-4,10-14; nice 0-19)", .{ name, value });
        return error.BadKnob;
    }
};

/// Runner threads call `enter` first; auto resolves at the first one, after the plan link pinned (its first plan).
pub const Placer = struct {
    cfg: Config,
    state: std.atomic.Value(u8) = .init(0), // 0 unresolved, 1 resolving, 2 resolved
    set: Set = @splat(0), // resolved; empty leaves the inherited mask
    entered: std.atomic.Value(u32) = .init(0),
    /// Tests observe the placed thread here.
    on_enter: ?*const fn (p: *Placer) void = null,

    /// Moves the calling runner thread onto the runner's CPUs at the runner's niceness and names it tf-runner.
    pub fn enter(p: *Placer) void {
        if (comptime builtin.os.tag != .linux) return;
        const set = p.resolved();
        if (!empty(&set)) _ = linux.syscall3(.sched_setaffinity, 0, @sizeOf(Set), @intFromPtr(&set));
        // the kernel's getpriority is 20 - nice; raising nice needs no privilege, lowering it does (kept as is)
        const inherited = 20 - @as(isize, @bitCast(linux.syscall2(.getpriority, 0, 0)));
        if (inherited < p.cfg.nice) _ = linux.syscall3(.setpriority, 0, 0, p.cfg.nice);
        _ = linux.prctl(@backingInt(linux.PR.SET_NAME), @intFromPtr("tf-runner"), 0, 0, 0);
        _ = p.entered.fetchAdd(1, .monotonic);
        if (p.on_enter) |f| f(p);
    }

    fn resolved(p: *Placer) Set {
        while (true) {
            switch (p.state.cmpxchgStrong(0, 1, .acquire, .acquire) orelse 0) {
                0 => break,
                2 => return p.set,
                else => std.Thread.yield() catch {},
            }
        }
        p.set = p.resolve();
        p.state.store(2, .release);
        return p.set;
    }

    fn resolve(p: *Placer) Set {
        var text: [256]u8 = undefined;
        switch (p.cfg.cpus) {
            .inherit => {
                log.info("in-model runner threads: inherited CPUs (_CPUS=inherit), nice {d}", .{p.cfg.nice});
                return @splat(0);
            },
            .list => |set| {
                log.info("in-model runner threads: CPUs {s}, nice {d}", .{ format(&set, &text), p.cfg.nice });
                return set;
            },
            .auto => {
                var avoid: Set = @splat(0);
                if (p.cfg.engine) |f| f(&avoid);
                const allowed = allowedCpus();
                const set = choose(&allowed, &avoid, capacity);
                var a: [256]u8 = undefined;
                var e: [256]u8 = undefined;
                if (empty(&set)) {
                    log.warn("in-model runner threads: no CPU left beside the engine's ({s} minus {s}); they inherit the engine thread's", .{ format(&allowed, &a), format(&avoid, &e) });
                } else log.info("in-model runner threads: CPUs {s} (auto: {s} minus the engine's {s}), nice {d}", .{ format(&set, &text), format(&allowed, &a), format(&avoid, &e), p.cfg.nice });
                return set;
            },
        }
    }
};

/// Auto: `allowed` minus `avoid`; with two core sizes (smallest under 80% of largest), only the small cluster.
pub fn choose(allowed: *const Set, avoid: *const Set, cap: *const fn (cpu: u32) ?u32) Set {
    var out: Set = @splat(0);
    for (&out, allowed, avoid) |*o, a, v| o.* = a & ~v;
    var low: u32 = std.math.maxInt(u32);
    var high: u32 = 0;
    var it = cpusOf(&out);
    while (it.next()) |cpu| if (cap(cpu)) |c| {
        low = @min(low, c);
        high = @max(high, c);
    };
    // one core type at slightly different clocks (GB10's big cores: 997 and 1017) is one cluster
    if (high == 0 or @as(u64, low) * 5 >= @as(u64, high) * 4) return out;
    const mid = low + (high - low) / 2;
    it = cpusOf(&out);
    while (it.next()) |cpu| {
        const c = cap(cpu) orelse std.math.maxInt(u32);
        if (c >= mid) out[cpu / 64] &= ~(@as(u64, 1) << @intCast(cpu % 64));
    }
    return out;
}

pub fn empty(set: *const Set) bool {
    for (set) |w| if (w != 0) return false;
    return true;
}

pub fn has(set: *const Set, cpu: u32) bool {
    return cpu < 1024 and set[cpu / 64] >> @intCast(cpu % 64) & 1 == 1;
}

pub fn add(set: *Set, cpu: u32) void {
    if (cpu < 1024) set[cpu / 64] |= @as(u64, 1) << @intCast(cpu % 64);
}

const Cpus = struct {
    set: *const Set,
    at: u32 = 0,

    fn next(it: *Cpus) ?u32 {
        while (it.at < 1024) {
            const cpu = it.at;
            it.at += 1;
            if (has(it.set, cpu)) return cpu;
        }
        return null;
    }
};

fn cpusOf(set: *const Set) Cpus {
    return .{ .set = set };
}

/// `set` as a CPU list (0-4,10-14), cut short to fit `buf`.
pub fn format(set: *const Set, buf: []u8) []const u8 {
    var w: std.Io.Writer = .fixed(buf);
    var cpu: u32 = 0;
    var first = true;
    while (cpu < 1024) : (cpu += 1) {
        if (!has(set, cpu)) continue;
        var end = cpu;
        while (end + 1 < 1024 and has(set, end + 1)) end += 1;
        if (end == cpu) w.print("{s}{d}", .{ if (first) "" else ",", cpu }) catch break else w.print("{s}{d}-{d}", .{ if (first) "" else ",", cpu, end }) catch break;
        first = false;
        cpu = end;
    }
    if (first) w.writeAll("none") catch {};
    return w.buffered();
}

/// The container's CPUs (cgroup cpuset, else online), not the calling thread's mask (a TP rank's single plan CPU).
pub fn allowedCpus() Set {
    var set: Set = undefined;
    var buf: [512]u8 = undefined;
    var path: [320]u8 = undefined;
    if (readSmall("/proc/self/cgroup", &buf)) |text| {
        var lines = std.mem.splitScalar(u8, text, '\n');
        while (lines.next()) |line| if (std.mem.startsWith(u8, line, "0::")) {
            // the nearest cgroup that sets a cpuset: a session scope's own file is empty, its slice's is not
            var dir = std.mem.trimEnd(u8, line[3..], "/");
            while (true) {
                const p = std.fmt.bufPrintSentinel(&path, "/sys/fs/cgroup{s}/cpuset.cpus.effective", .{dir}, 0) catch break;
                var cb: [512]u8 = undefined;
                if (readSmall(p, &cb)) |cpus| if (sandbox.cpuList(std.mem.trim(u8, cpus, " \n"), &set)) return set;
                if (dir.len == 0) break;
                dir = dir[0 .. std.mem.findScalarLast(u8, dir, '/') orelse 0];
            }
        };
    }
    if (readSmall("/sys/devices/system/cpu/online", &buf)) |cpus| if (sandbox.cpuList(std.mem.trim(u8, cpus, " \n"), &set)) return set;
    return @splat(0);
}

/// A core's cpu_capacity (sysfs; null where the kernel does not report one).
pub fn capacity(cpu: u32) ?u32 {
    var pb: [64]u8 = undefined;
    const path = std.fmt.bufPrintSentinel(&pb, "/sys/devices/system/cpu/cpu{d}/cpu_capacity", .{cpu}, 0) catch return null;
    var b: [16]u8 = undefined;
    const text = readSmall(path, &b) orelse return null;
    return std.fmt.parseInt(u32, std.mem.trim(u8, text, " \n"), 10) catch null;
}

fn readSmall(path: [*:0]const u8, buf: []u8) ?[]const u8 {
    if (comptime builtin.os.tag != .linux) return null;
    const rc = linux.open(path, .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, 0);
    if (linux.errno(rc) != .SUCCESS) return null;
    const fd: i32 = @intCast(rc);
    defer _ = linux.close(fd);
    const n = linux.read(fd, buf.ptr, buf.len);
    if (linux.errno(n) != .SUCCESS) return null;
    return buf[0..n];
}
