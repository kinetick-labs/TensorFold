//! runner-place-bench [ENGINE_CPU] [SECONDS] [WORKERS] [ROUNDS]: host-loop stand-ins idle, beside the runner, inherit.
const std = @import("std");
const linux = std.os.linux;
const place = @import("runner_place.zig");
const backend = @import("runner_circuit_backend.zig");
const rc = @import("runner_circuit.zig");
const api = @import("runner_api.zig");

const multiply = "123456789012345678901234567890*987654321098765432109876543210";
const burst_iters = 150_000; // ~0.2 ms of dependent integer work on a Grace core
const gap_ns = 500 * std.time.ns_per_us; // the host loop waits on the GPU between bursts

fn nowNs() u64 {
    var ts: linux.timespec = undefined;
    _ = linux.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

fn nap(ns: u64) void {
    const ts: linux.timespec = .{ .sec = @intCast(ns / std.time.ns_per_s), .nsec = @intCast(ns % std.time.ns_per_s) };
    _ = linux.nanosleep(&ts, null);
}

/// This thread's schedstat: ns on a CPU and ns runnable but waiting for one.
fn sched() [2]u64 {
    var buf: [128]u8 = undefined;
    const opened = linux.open("/proc/thread-self/schedstat", .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, 0);
    if (linux.errno(opened) != .SUCCESS) return .{ 0, 0 };
    const fd: i32 = @intCast(opened);
    defer _ = linux.close(fd);
    const n = linux.read(fd, &buf, buf.len);
    if (linux.errno(n) != .SUCCESS) return .{ 0, 0 };
    var it = std.mem.tokenizeAny(u8, buf[0..n], " \n");
    const run = std.fmt.parseInt(u64, it.next() orelse "0", 10) catch 0;
    const wait = std.fmt.parseInt(u64, it.next() orelse "0", 10) catch 0;
    return .{ run, wait };
}

fn pin(set: *const place.Set) bool {
    return linux.errno(linux.syscall3(.sched_setaffinity, 0, @sizeOf(place.Set), @intFromPtr(set))) == .SUCCESS;
}

fn one(cpu: u32) place.Set {
    var s: place.Set = @splat(0);
    place.add(&s, cpu);
    return s;
}

/// A host-loop stand-in: bursts of fixed work with GPU-wait gaps; the measure is each period beyond its gap.
const Proxy = struct {
    mask: place.Set,
    stop: *std.atomic.Value(bool),
    times: std.ArrayList(u64) = .empty,
    gpa: std.mem.Allocator,
    sink: u64 = 0,
    used: [2]u64 = .{ 0, 0 }, // run and wait ns over the phase

    fn run(p: *Proxy) void {
        _ = pin(&p.mask);
        const s0 = sched();
        defer {
            const s1 = sched();
            p.used = .{ s1[0] - s0[0], s1[1] - s0[1] };
        }
        var x: u64 = 0x9E3779B97F4A7C15;
        var last = nowNs();
        while (!p.stop.load(.acquire)) {
            for (0..burst_iters) |_| x = (x ^ (x >> 13)) *% 0x2545F4914F6CDD1D +% 7;
            nap(gap_ns);
            // burst + gap + wake-up delay: a CPU taken from the loop shows here, before and after a burst
            const now = nowNs();
            p.times.append(p.gpa, now - last - gap_ns) catch return;
            last = now;
        }
        p.sink = x;
    }
};

/// The engine thread's part: on the engine CPU it keeps every runner worker busy, polling as the engine's tick does.
const Driver = struct {
    engine: place.Set,
    b: api.Backend,
    workers: u8,
    program: []const u8,
    stop: *std.atomic.Value(bool),
    jobs: u64 = 0,
    steps: u64 = 0,
    cpu_ns: u64 = 0,

    fn run(d: *Driver) void {
        _ = pin(&d.engine);
        const s0 = sched();
        defer d.cpu_ns = sched()[0] - s0[0];
        var live: [64]?api.JobId = @splat(null);
        var stream: u64 = 0;
        while (!d.stop.load(.acquire)) {
            for (live[0..d.workers]) |*slot| {
                if (slot.*) |id| switch (d.b.poll(id)) {
                    .running => continue,
                    .done => |r| {
                        d.jobs += 1;
                        d.steps += r.steps;
                        slot.* = null;
                    },
                };
                stream += 1;
                slot.* = d.b.start(stream, .{ .kind = .calc, .bytes = d.program }, 1 << 30) catch null;
            }
            nap(std.time.ns_per_ms);
        }
        for (live[0..d.workers]) |slot| if (slot) |id| d.b.cancel(id);
    }
};

/// idle: no runner; placed: TF_RUNNER_CPUS=auto; inherit: the control (runner threads keep the engine's CPU).
const Phase = enum { idle, placed, inherit };

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const own = place.allowedCpus();
    var mine: place.Set = @splat(0);
    _ = linux.syscall3(.sched_getaffinity, 0, @sizeOf(place.Set), @intFromPtr(&mine));
    // default engine CPU: the biggest core this process may use, as TF_TP_PLAN_PIN=auto picks
    var engine_cpu: u32 = 0;
    var best: u32 = 0;
    for (0..1024) |c| if (place.has(&mine, @intCast(c))) if (place.capacity(@intCast(c))) |cap| if (cap > best) {
        best = cap;
        engine_cpu = @intCast(c);
    };
    if (args.len > 1) engine_cpu = try std.fmt.parseInt(u32, args[1], 10);
    const seconds: u64 = if (args.len > 2) try std.fmt.parseInt(u64, args[2], 10) else 10;
    const workers: u8 = if (args.len > 3) try std.fmt.parseInt(u8, args[3], 10) else 2;
    const rounds: u64 = if (args.len > 4) try std.fmt.parseInt(u64, args[4], 10) else 3;
    if (!place.has(&mine, engine_cpu)) {
        var b: [256]u8 = undefined;
        std.debug.print("FAIL engine CPU {d} is not in this process's CPUs {s}\n", .{ engine_cpu, place.format(&mine, &b) });
        return 2;
    }
    var circuit: rc.Circuit = undefined;
    var program: []const u8 = multiply;
    if (init.environ_map.get("TF_RUNNER_CIRCUIT")) |path| {
        circuit = try rc.Circuit.load(gpa, path);
    } else {
        const bytes = try @import("runner_circuit_tiny_test.zig").build(gpa, .good);
        defer gpa.free(bytes);
        circuit = try @import("runner_circuit_load.zig").fromBytes(gpa, bytes);
        program = "";
    }
    defer circuit.deinit(gpa);
    engine_set = one(engine_cpu);
    // the engine's unpinned threads: the cores left beside the engine CPU and the runner's auto set
    const runner_auto = place.choose(&own, &engine_set, place.capacity);
    var rest: place.Set = undefined;
    for (&rest, mine, runner_auto, engine_set) |*r, m, a, e| r.* = m & ~a & ~e;
    const floating = !place.empty(&rest);
    var b1: [256]u8 = undefined;
    var b2: [256]u8 = undefined;
    var b3: [256]u8 = undefined;
    var b4: [256]u8 = undefined;
    std.debug.print("engine CPU {d}; process CPUs {s}; container CPUs {s}; runner auto {s}; floating loop {s}; {d} workers; {d} rounds of {d} s a phase\n", .{ engine_cpu, place.format(&mine, &b1), place.format(&own, &b2), place.format(&runner_auto, &b3), if (floating) place.format(&rest, &b4) else "none (no core left)", workers, rounds, seconds });
    var fixed: [3]std.ArrayList(u64) = @splat(.empty);
    var float: [3]std.ArrayList(u64) = @splat(.empty);
    defer for (&fixed, &float) |*a, *c| {
        a.deinit(gpa);
        c.deinit(gpa);
    };
    var jobs: [3]u64 = @splat(0);
    var proxy_wait: [3]u64 = @splat(0);
    var driver_cpu: [3]u64 = @splat(0);
    var steps: [3]u64 = @splat(0);
    // phases interleave round by round, so the machine's background load drifts across all three alike
    for (0..rounds) |_| for ([_]Phase{ .idle, .placed, .inherit }) |phase| {
        const k = @backingInt(phase);
        var stop: std.atomic.Value(bool) = .init(false);
        var fp: Proxy = .{ .mask = engine_set, .stop = &stop, .gpa = gpa, .times = fixed[k] };
        var lp: Proxy = .{ .mask = rest, .stop = &stop, .gpa = gpa, .times = float[k] };
        var placer: place.Placer = .{ .cfg = .{ .cpus = if (phase == .inherit) .inherit else .auto, .engine = engineSet } };
        var b: ?api.Backend = null;
        var driver: Driver = undefined;
        var dt: ?std.Thread = null;
        if (phase != .idle) {
            b = try backend.placed(gpa, &circuit, workers, &placer);
            driver = .{ .engine = engine_set, .b = b.?, .workers = workers, .program = program, .stop = &stop };
            dt = try std.Thread.spawn(.{}, Driver.run, .{&driver});
        }
        const ft = try std.Thread.spawn(.{}, Proxy.run, .{&fp});
        const lt: ?std.Thread = if (floating) try std.Thread.spawn(.{}, Proxy.run, .{&lp}) else null;
        nap(seconds * std.time.ns_per_s);
        stop.store(true, .release);
        ft.join();
        if (lt) |t| t.join();
        if (dt) |t| t.join();
        if (b) |x| backend.destroy(x);
        fixed[k] = fp.times;
        proxy_wait[k] += fp.used[1];
        float[k] = lp.times;
        if (phase != .idle) {
            jobs[k] += driver.jobs;
            driver_cpu[k] += driver.cpu_ns;
            steps[k] += driver.steps;
        }
    };
    var base: [2]u64 = .{ 1, 1 };
    var base_n: usize = 1;
    var verdict_ok = true;
    for ([_]Phase{ .idle, .placed, .inherit }) |phase| {
        const k = @backingInt(phase);
        const f = stats(fixed[k].items);
        const l = stats(float[k].items);
        if (phase == .idle) base = .{ @max(f.p50, 1), @max(l.p50, 1) };
        const rf = @as(f64, @floatFromInt(f.p50)) / @as(f64, @floatFromInt(base[0]));
        const rl = @as(f64, @floatFromInt(l.p50)) / @as(f64, @floatFromInt(base[1]));
        if (phase == .idle) base_n = @max(f.n, 1);
        const rate = @as(f64, @floatFromInt(f.n)) / @as(f64, @floatFromInt(base_n));
        std.debug.print("RESULT {t}: engine-CPU loop period beyond its gap p50 {d} us p99 {d} us (x{d:.3} of idle), {d} loops (x{d:.3})", .{ phase, f.p50 / 1000, f.p99 / 1000, rf, f.n, rate });
        if (floating) std.debug.print("; floating loop p50 {d} us p99 {d} us (x{d:.3})", .{ l.p50 / 1000, l.p99 / 1000, rl });
        const secs = rounds * seconds;
        std.debug.print("; proxy runnable-waiting {d} ms/s", .{proxy_wait[k] / 1_000_000 / secs});
        if (phase != .idle) std.debug.print("; driver (engine thread) {d} ms/s CPU; runner {d} jobs, {d} steps/s", .{ driver_cpu[k] / 1_000_000 / secs, jobs[k], steps[k] / secs });
        std.debug.print("\n", .{});
        if (phase == .placed and (rf > 1.05 or rate < 0.95 or (floating and rl > 1.05))) verdict_ok = false;
    }
    std.debug.print("{s}: the placed runner at full load {s} the engine's loops within 5% of idle (p50 period, loops)\n", .{ if (verdict_ok) "PASS" else "FAIL", if (verdict_ok) "keeps" else "does not keep" });
    return if (verdict_ok) 0 else 1;
}

const Stats = struct { n: usize, p50: u64, p99: u64 };

fn stats(t: []u64) Stats {
    if (t.len == 0) return .{ .n = 0, .p50 = 0, .p99 = 0 };
    std.mem.sort(u64, t, {}, std.sort.asc(u64));
    return .{ .n = t.len, .p50 = t[t.len / 2], .p99 = t[t.len * 99 / 100] };
}

var engine_set: place.Set = @splat(0);

fn engineSet(set: *place.Set) void {
    for (set, engine_set) |*s, e| s.* |= e;
}
