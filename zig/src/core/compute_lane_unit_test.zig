//! Compute lane checks that spawn nothing: knobs, the result body, the trampoline's arguments and the seccomp filter.
const std = @import("std");
const builtin = @import("builtin");
const lane = @import("compute_lane.zig");
const runner = @import("compute_runner.zig");
const sandbox = @import("compute_sandbox.zig");
const seccomp = @import("compute_seccomp.zig");
const landlock = @import("compute_landlock.zig");
const expect = std.testing.expect;
const expectEqual = std.testing.expectEqual;
const expectEqualStrings = std.testing.expectEqualStrings;

var knobs: []const [2][]const u8 = &.{};

fn lookup(name: []const u8) ?[]const u8 {
    for (knobs) |k| if (std.mem.eql(u8, k[0], name)) return k[1];
    return null;
}

fn parse(set: []const [2][]const u8) error{BadKnob}!?lane.Config {
    knobs = set;
    return lane.Config.parse(&lookup, "TF_X");
}

test "compute lane config: off unless the flag is 1" {
    try expectEqual(null, try parse(&.{}));
    try expectEqual(null, try parse(&.{.{ "TF_X", "" }}));
    try expectEqual(null, try parse(&.{ .{ "TF_X", "0" }, .{ "TF_X_WORKERS", "3" } }));
    try std.testing.expectError(error.BadKnob, parse(&.{.{ "TF_X", "yes" }}));
    const c = (try parse(&.{.{ "TF_X", "1" }})).?;
    try expectEqual(@as(u8, 2), c.workers);
    try expectEqual(@as(u32, 10_000), c.max_ms);
    try expectEqual(@as(u32, 64 * 1024), c.max_output_bytes);
    try expectEqual(@as(u32, 512), c.max_result_tokens);
    try expectEqual(@as(u32, 8), c.max_calls);
    try expectEqual(@as(u32, 1024), c.mem_mb);
    try expectEqualStrings("/usr/bin/python3", c.python);
    try expectEqualStrings("", c.cpus);
}

test "compute lane config: every knob parses" {
    const c = (try parse(&.{
        .{ "TF_X", "1" },                          .{ "TF_X_WORKERS", "3" },    .{ "TF_X_MAX_MS", "2500" },
        .{ "TF_X_MAX_RESULT_TOKENS", "64" },       .{ "TF_X_MAX_CALLS", "2" },  .{ "TF_X_MEM_MB", "256" },
        .{ "TF_X_PYTHON", "/opt/py/bin/python3" }, .{ "TF_X_CPUS", "1,3,5-7" },
    })).?;
    try expectEqual(@as(u8, 3), c.workers);
    try expectEqual(@as(u32, 2500), c.max_ms);
    try expectEqual(@as(u32, 64), c.max_result_tokens);
    try expectEqual(@as(u32, 2), c.max_calls);
    try expectEqual(@as(u32, 256), c.mem_mb);
    try expectEqualStrings("/opt/py/bin/python3", c.python);
    try expectEqualStrings("1,3,5-7", c.cpus);
}

test "compute lane config: malformed or out-of-range knobs refuse" {
    const bad = [_][2][]const u8{
        .{ "TF_X_WORKERS", "0" },     .{ "TF_X_WORKERS", "300" },    .{ "TF_X_WORKERS", "two" },
        .{ "TF_X_MAX_MS", "0" },      .{ "TF_X_MAX_MS", "-5" },      .{ "TF_X_MEM_MB", "8" },
        .{ "TF_X_MAX_CALLS", "1e3" }, .{ "TF_X_PYTHON", "python3" }, .{ "TF_X_CPUS", "3-1" },
        .{ "TF_X_CPUS", "a" },        .{ "TF_X_CPUS", "2048" },      .{ "TF_X_MAX_RESULT_TOKENS", "0" },
    };
    for (bad) |k| try std.testing.expectError(error.BadKnob, parse(&.{ .{ "TF_X", "1" }, k }));
}

test "compute lane: cpu lists" {
    var set: [16]u64 = undefined;
    try expect(sandbox.cpuList("16-19", &set));
    try expectEqual(@as(u64, 0xF0000), set[0]);
    try expect(sandbox.cpuList("1,3,5-7,64", &set));
    try expectEqual(@as(u64, 0b11101010), set[0]);
    try expectEqual(@as(u64, 1), set[1]);
    for ([_][]const u8{ "", ",", "1,", "-3", "4-", "1-1024", "x" }) |t| try expect(!sandbox.cpuList(t, &set));
}

test "compute lane: trampoline arguments round-trip and refuse other versions" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const job: sandbox.Job = .{ .max_ms = 500, .mem_mb = 256, .cpus = "", .python = "/usr/bin/python3" };
    const argv = try sandbox.encode(arena.allocator(), &.{ "/proc/self/exe", "__compute_sandbox" }, job);
    try expectEqual(@as(usize, 7), argv.len);
    var args: [5][:0]const u8 = undefined;
    for (&args, argv[2..]) |*d, s| d.* = std.mem.span(s.?);
    const back = sandbox.decode(&args).?;
    try expectEqual(job.max_ms, back.max_ms);
    try expectEqual(job.mem_mb, back.mem_mb);
    try expectEqualStrings(job.python, back.python);
    args[0] = "v0";
    try expectEqual(null, sandbox.decode(&args));
    try expectEqual(null, sandbox.decode(args[1..]));
}

test "compute lane: interpreter install root" {
    try expectEqual(null, try landlock.installRoot("/usr/bin/python3.12"));
    try expectEqualStrings("/opt/py", (try landlock.installRoot("/opt/py/bin/python3")).?);
    try std.testing.expectError(error.Root, landlock.installRoot("/bin/python3"));
}

fn body(o: lane.Outcome) ![]u8 {
    return runner.body(std.testing.allocator, o, 500, 1000);
}

test "compute lane: result body for every status" {
    const cases = [_]struct { o: lane.Outcome, want: []const u8 }{
        .{ .o = .{ .status = .ok, .stdout = "42\n" }, .want = "42" },
        .{ .o = .{ .status = .ok, .stdout = "a\n\n" }, .want = "a\n" },
        .{ .o = .{ .status = .ok }, .want = "[no output]" },
        .{ .o = .{ .status = .ok, .stdout = "\n" }, .want = "[no output]" },
        .{ .o = .{ .status = .ok, .stdout = "x\n", .stderr = "warn\n" }, .want = "x\n\n[stderr]\nwarn" },
        .{ .o = .{ .status = .ok, .stderr = "warn" }, .want = "[stderr]\nwarn" },
        .{ .o = .{ .status = .exit_error, .exit_code = 1, .stderr = "E\n" }, .want = "[stderr]\nE\n[exit status 1]" },
        .{ .o = .{ .status = .exit_error, .signal = 24, .stdout = "p" }, .want = "p\n[killed by signal 24]" },
        .{ .o = .{ .status = .timeout }, .want = "[timed out after 500 ms]" },
        .{ .o = .{ .status = .output_limit, .stdout = "xx" }, .want = "xx\n[output truncated at 1000 bytes]" },
        .{ .o = .{ .status = .cancelled, .stdout = "a\n" }, .want = "a\n[cancelled]" },
        .{ .o = .{ .status = .unavailable, .stderr = "why" }, .want = "[stderr]\nwhy\n[sandbox unavailable]" },
        .{ .o = .{ .status = .ok, .stdout = "t", .exec_ns = 123456789 }, .want = "t" },
    };
    for (cases) |k| {
        const got = try body(k.o);
        defer std.testing.allocator.free(got);
        try expectEqualStrings(k.want, got);
    }
}

/// The filter's verdict for one call, by a small classic-BPF interpreter over the instructions seccomp loads.
fn verdict(nr: u32, arch: u32, arg0: u32) u32 {
    var data: [16]u32 = @splat(0);
    data[seccomp.nr_at / 4] = nr;
    data[seccomp.arch_at / 4] = arch;
    data[seccomp.arg0_at / 4] = arg0;
    const BPF = std.os.linux.BPF;
    var acc: u32 = 0;
    var pc: usize = 0;
    while (pc < seccomp.program.len) : (pc += 1) {
        const i = seccomp.program[pc];
        switch (i.code) {
            BPF.LD | BPF.W | BPF.ABS => acc = data[i.k / 4],
            BPF.RET | BPF.K => return i.k,
            BPF.JMP | BPF.JEQ | BPF.K => pc += if (acc == i.k) i.jt else i.jf,
            BPF.JMP | BPF.JGE | BPF.K => pc += if (acc >= i.k) i.jt else i.jf,
            BPF.JMP | BPF.JSET | BPF.K => pc += if (acc & i.k != 0) i.jt else i.jf,
            else => return 0xdead,
        }
    }
    return 0xdead; // fell off the end: the kernel refuses such a filter
}

fn sys(comptime name: []const u8) u32 {
    return @intCast(@backingInt(@field(std.os.linux.SYS, name)));
}

test "compute lane: seccomp filter verdicts" {
    if (!seccomp.supported) return error.SkipZigTest;
    const RET = std.os.linux.SECCOMP.RET;
    const arch = seccomp.audit_arch;
    try expectEqual(RET.KILL_PROCESS, verdict(sys("read"), 0x40000003, 0)); // i386 audit arch
    try expectEqual(RET.ALLOW, verdict(sys("read"), arch, 0));
    try expectEqual(RET.ALLOW, verdict(sys("execve"), arch, 0));
    try expectEqual(RET.ALLOW, verdict(sys("landlock_create_ruleset"), arch, 0));
    for ([_]u32{ sys("socket"), sys("ptrace"), sys("bpf"), sys("io_uring_setup"), sys("unshare"), sys("mount") }) |n|
        try expectEqual(RET.ERRNO | seccomp.eperm, verdict(n, arch, 0));
    inline for (seccomp.denied) |name| if (@hasField(std.os.linux.SYS, name))
        try expectEqual(RET.ERRNO | seccomp.eperm, verdict(sys(name), arch, 0));
    try expectEqual(RET.ERRNO | seccomp.enosys, verdict(sys("clone3"), arch, 0));
    try expectEqual(RET.ERRNO | seccomp.eperm, verdict(sys("clone"), arch, 17)); // SIGCHLD: a fork
    try expectEqual(RET.ALLOW, verdict(sys("clone"), arch, 0x3d0f00)); // glibc's pthread_create flags
    if (builtin.cpu.arch == .x86_64) {
        try expectEqual(RET.KILL_PROCESS, verdict(seccomp.x32_bit | sys("read"), arch, 0));
        try expectEqual(RET.ERRNO | seccomp.eperm, verdict(sys("fork"), arch, 0));
        try expectEqual(RET.ERRNO | seccomp.eperm, verdict(sys("vfork"), arch, 0));
    }
}
