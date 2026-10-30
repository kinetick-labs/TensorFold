//! Compute lane jobs through the real trampoline: results, limits, cancellation, slots and every confinement layer.
const std = @import("std");
const builtin = @import("builtin");
const lane = @import("compute_lane.zig");
const linux = std.os.linux;
const expect = std.testing.expect;
const expectEqual = std.testing.expectEqual;
const expectEqualStrings = std.testing.expectEqualStrings;

test {
    _ = @import("compute_lane_unit_test.zig");
}

const launcher = [_][]const u8{@import("compute_lane_options").sandbox_exe};
const python = "/usr/bin/python3";

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;

/// Skips off Linux or without the interpreter; a missing Landlock or seccomp is a failure, not a skip.
fn ready() !void {
    if (builtin.os.tag != .linux) return error.SkipZigTest;
    const fd = linux.open(python, .{ .PATH = true, .CLOEXEC = true }, 0);
    if (linux.errno(fd) != .SUCCESS) return error.SkipZigTest;
    _ = linux.close(@intCast(fd));
}

fn now() u64 {
    if (builtin.os.tag != .linux) return 0;
    var ts: linux.timespec = undefined;
    _ = linux.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

const Job = struct {
    arena: std.heap.ArenaAllocator = .init(std.heap.page_allocator),
    fn deinit(j: *Job) void {
        j.arena.deinit();
    }
    fn run(j: *Job, cfg: lane.Config, code: []const u8, cancel: ?lane.Cancel) !lane.Outcome {
        var pool: lane.Pool = .init(cfg, &launcher);
        return pool.run(j.arena.allocator(), code, cancel);
    }
    /// Runs `code` and requires `status`, showing the job's output when it differs.
    fn expectStatus(j: *Job, cfg: lane.Config, code: []const u8, status: lane.Status) !lane.Outcome {
        const o = try j.run(cfg, code, null);
        if (o.status != status) {
            std.debug.print("status {s}, exit {?d}, signal {?d}\nstdout: {s}\nstderr: {s}\n", .{ @tagName(o.status), o.exit_code, o.signal, o.stdout, o.stderr });
            return error.TestUnexpectedResult;
        }
        return o;
    }
};

test "compute sandbox: self test passes (Landlock and seccomp present)" {
    try ready();
    var j: Job = .{};
    defer j.deinit();
    var pool: lane.Pool = .init(.{}, &launcher);
    try pool.selfTest(j.arena.allocator());
}

test "compute sandbox: a job from a thread at nice 19 runs and keeps that niceness" {
    try ready();
    const Run = struct {
        status: lane.Status = .unavailable,
        out: []const u8 = "",
        fn go(r: *@This(), j: *Job) void {
            _ = linux.syscall3(.setpriority, 0, 0, 19); // this thread only: the job inherits it
            const o = j.expectStatus(.{}, "import os\nos.nice(0)", .ok) catch return;
            r.status = o.status;
            r.out = o.stdout;
        }
    };
    var j: Job = .{};
    defer j.deinit();
    var r: Run = .{};
    const t = try std.Thread.spawn(.{}, Run.go, .{ &r, &j });
    t.join();
    try expectEqual(lane.Status.ok, r.status);
    try expectEqualStrings("19\n", r.out);
}

test "compute sandbox: final expression prints its repr" {
    try ready();
    var j: Job = .{};
    defer j.deinit();
    const o = try j.expectStatus(.{}, "x=20\nx+22", .ok);
    try expectEqualStrings("42\n", o.stdout);
    try expectEqualStrings("", o.stderr);
    const s = try j.expectStatus(.{}, "s='a'\ns\nNone", .ok);
    try expectEqualStrings("", s.stdout);
}

test "compute sandbox: an exception exits 1 with the user's traceback only" {
    try ready();
    var j: Job = .{};
    defer j.deinit();
    const o = try j.expectStatus(.{}, "1/0", .exit_error);
    errdefer std.debug.print("stderr: {s}\n", .{o.stderr});
    try expectEqual(@as(?u8, 1), o.exit_code);
    try expect(std.mem.find(u8, o.stderr, "ZeroDivisionError") != null);
    try expect(std.mem.find(u8, o.stderr, "<code>") != null);
    try expect(std.mem.find(u8, o.stderr, "<string>") == null); // the runner's own frame
    try expect(std.mem.find(u8, o.stderr, "ast.py") == null);
    try expect(std.mem.find(u8, o.stderr, "exec(compile") == null);
    try expect(std.mem.startsWith(u8, o.stderr, "Traceback (most recent call last):\n  File \"<code>\", line 1, in <module>\n"));
    const e = try j.expectStatus(.{}, "def f(:\n", .exit_error);
    try expect(std.mem.find(u8, e.stderr, "SyntaxError") != null);
    try expect(std.mem.find(u8, e.stderr, "<string>") == null);
    const x = try j.expectStatus(.{}, "import sys\nprint('bye')\nsys.exit(3)", .exit_error);
    try expectEqual(@as(?u8, 3), x.exit_code);
    try expectEqualStrings("bye\n", x.stdout);
}

test "compute sandbox: no network" {
    try ready();
    var j: Job = .{};
    defer j.deinit();
    const code = "import socket\ntry:\n socket.socket()\n print('open')\nexcept OSError as e:\n print('denied', e.errno)";
    try expectEqualStrings("denied 1\n", (try j.expectStatus(.{}, code, .ok)).stdout);
}

test "compute sandbox: no files outside the interpreter" {
    try ready();
    var j: Job = .{};
    defer j.deinit();
    const code =
        \\import os
        \\def check(name, f):
        \\    try:
        \\        f()
        \\        print(name, 'open')
        \\    except OSError:
        \\        print(name, 'denied')
        \\check('passwd', lambda: open('/etc/passwd').read())
        \\check('root', lambda: os.listdir('/'))
        \\check('proc', lambda: os.listdir('/proc'))
        \\check('home', lambda: os.listdir('/home'))
        \\check('roothome', lambda: os.listdir('/root'))
        \\check('tmp', lambda: open('/tmp/x', 'w').write('x'))
        \\check('cwd', lambda: open('x', 'w').write('x'))
        \\check('usr', lambda: open('/usr/x', 'w').write('x'))
    ;
    const want = "passwd denied\nroot denied\nproc denied\nhome denied\nroothome denied\ntmp denied\ncwd denied\nusr denied\n";
    try expectEqualStrings(want, (try j.expectStatus(.{}, code, .ok)).stdout);
}

test "compute sandbox: the engine's environment never reaches the job" {
    try ready();
    var j: Job = .{};
    defer j.deinit();
    try expectEqual(@as(c_int, 0), setenv("TF_COMPUTE_CANARY", "secret", 1));
    const o = try j.expectStatus(.{}, "import os\nsorted(os.environ)", .ok);
    const keys = "'LANG', 'PATH', 'PYTHONDONTWRITEBYTECODE', 'PYTHONHASHSEED', 'PYTHONIOENCODING'";
    const with_ctype = "['LANG', 'LC_CTYPE', 'PATH', 'PYTHONDONTWRITEBYTECODE', 'PYTHONHASHSEED', 'PYTHONIOENCODING']\n";
    if (!std.mem.eql(u8, o.stdout, with_ctype)) try expectEqualStrings("[" ++ keys ++ "]\n", o.stdout);
}

test "compute sandbox: threads run, fork does not" {
    try ready();
    var j: Job = .{};
    defer j.deinit();
    const code =
        \\import os, threading
        \\try:
        \\    os.fork()
        \\    print('forked')
        \\except OSError as e:
        \\    print('fork denied', e.errno)
        \\out = []
        \\t = threading.Thread(target=lambda: out.append(1))
        \\t.start()
        \\t.join()
        \\print('thread', out)
    ;
    try expectEqualStrings("fork denied 1\nthread [1]\n", (try j.expectStatus(.{}, code, .ok)).stdout);
}

test "compute sandbox: the job cannot signal outside its sandbox" {
    try ready();
    var j: Job = .{};
    defer j.deinit();
    const code = "import os\ntry:\n os.kill(os.getppid(), 0)\n print('reached')\nexcept OSError as e:\n print('denied', e.errno)";
    try expectEqualStrings("denied 1\n", (try j.expectStatus(.{}, code, .ok)).stdout);
}

test "compute sandbox: never runs as root" {
    try ready();
    var j: Job = .{};
    defer j.deinit();
    const o = try j.expectStatus(.{}, "import os\nprint(os.getuid(), os.geteuid())", .ok);
    const uid = linux.getuid();
    const child: u32 = if (uid == 0) 65534 else uid;
    var buf: [32]u8 = undefined;
    try expectEqualStrings(try std.fmt.bufPrint(&buf, "{d} {d}\n", .{ child, child }), o.stdout);
}

test "compute sandbox: the wall limit kills a busy job" {
    try ready();
    var j: Job = .{};
    defer j.deinit();
    const t = now();
    const o = try j.expectStatus(.{ .max_ms = 500 }, "print('start')\nwhile True: pass", .timeout);
    try expect(now() - t < 3 * std.time.ns_per_s);
    try expect(o.exec_ns >= 500 * std.time.ns_per_ms and o.exec_ns < 3 * std.time.ns_per_s);
    try expectEqualStrings("start\n", o.stdout);
}

test "compute sandbox: the output cap keeps the first bytes and kills" {
    try ready();
    var j: Job = .{};
    defer j.deinit();
    const o = try j.expectStatus(.{ .max_output_bytes = 1000 }, "print('x'*100000)\nimport time\ntime.sleep(5)", .output_limit);
    try expectEqual(@as(usize, 1000), o.stdout.len);
    try expect(o.exec_ns < 3 * std.time.ns_per_s);
}

test "compute sandbox: the memory cap" {
    try ready();
    var j: Job = .{};
    defer j.deinit();
    const o = try j.expectStatus(.{ .mem_mb = 256 }, "b=bytearray(2*1024**3)", .exit_error);
    try expect(std.mem.find(u8, o.stderr, "MemoryError") != null);
}

const Gone = struct {
    start: u64,
    fn gone(ctx: *anyopaque) bool {
        const g: *Gone = @ptrCast(@alignCast(ctx));
        return now() - g.start > 200 * std.time.ns_per_ms;
    }
};

test "compute sandbox: a cancelled job is killed promptly" {
    try ready();
    var j: Job = .{};
    defer j.deinit();
    var g: Gone = .{ .start = now() };
    const o = try j.run(.{}, "import time\ntime.sleep(5)", .{ .ctx = &g, .gone = Gone.gone });
    try expectEqual(lane.Status.cancelled, o.status);
    try expect(now() - g.start < 1500 * std.time.ns_per_ms);
}

fn slot(pool: *lane.Pool, out: *lane.Status) void {
    var arena: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
    defer arena.deinit();
    const o = pool.run(arena.allocator(), "import time;time.sleep(0.3);print(1)", null) catch {
        out.* = .unavailable;
        return;
    };
    out.* = if (std.mem.eql(u8, o.stdout, "1\n")) o.status else .exit_error;
}

test "compute sandbox: one worker runs jobs one at a time" {
    try ready();
    var pool: lane.Pool = .init(.{ .workers = 1 }, &launcher);
    var got = [_]lane.Status{ .unavailable, .unavailable };
    const t = now();
    var threads: [2]std.Thread = undefined;
    for (&threads, &got) |*th, *g| th.* = try std.Thread.spawn(.{}, slot, .{ &pool, g });
    for (threads) |th| th.join();
    try expectEqual([_]lane.Status{ .ok, .ok }, got);
    try expect(now() - t >= 600 * std.time.ns_per_ms);
    try expectEqual(@as(u32, 0), pool.busy.load(.acquire));
}

test "compute sandbox: hash seed and random are fixed" {
    try ready();
    var j: Job = .{};
    defer j.deinit();
    const code = "print(hash('abc'));import random;print(random.random())";
    const first = try j.expectStatus(.{}, code, .ok);
    const second = try j.expectStatus(.{}, code, .ok);
    try expectEqualStrings(first.stdout, second.stdout);
}

test "compute sandbox: a missing interpreter is unavailable and fails the self test" {
    try ready();
    var j: Job = .{};
    defer j.deinit();
    const cfg: lane.Config = .{ .python = "/usr/bin/no-such-python" };
    const o = try j.expectStatus(cfg, "print(1)", .unavailable);
    try expect(std.mem.startsWith(u8, o.stderr, "compute sandbox: "));
    var pool: lane.Pool = .init(cfg, &launcher);
    try std.testing.expectError(error.SandboxUnavailable, pool.selfTest(j.arena.allocator()));
}
