//! Boot probe of the circuit route: known cells through the frontend, the compiler and the loaded circuit.
const std = @import("std");
const api = @import("runner_api.zig");

/// Loops compiled to RV32 and CPython's (or C's) output: sums, // and %, a product, a string read from ROM bytes.
pub const cells = [_]struct { lang: []const u8 = "python", code: []const u8, want: []const u8 }{
    .{ .code = "total = 0\nfor i in range(10):\n    total += i * i\nprint(total)\n", .want = "285\n" },
    .{ .code = "print(sum(i*i for i in range(10)))\n", .want = "285\n" },
    .{ .code = "print(sum(1000003//d for d in range(3, 40)))\n", .want = "2753534\n" },
    .{ .code = "print(sum(1000003%d for d in range(3, 40)))\n", .want = "357\n" },
    .{ .code = "f = 1\nfor i in range(1, 13):\n    f *= i\nprint(f)\n", .want = "479001600\n" },
    .{ .lang = "c", .code = "int main(void) { puts(\"circuit route\"); return 0; }\n", .want = "circuit route\n" },
};

/// Steps a probe run may take (each cell needs far fewer): a circuit that loops is wrong, not slow.
pub const steps = 100_000;

pub const Outcome = struct {
    ran: u32 = 0, // cells run and right
    steps: u64 = 0, // their steps, summed
    wrong: ?[]const u8 = null, // the first cell that abstained, did not start or printed anything but Python's (in `why`)
};

/// Every cell when the compiler works (else none), up to the first that is wrong.
pub fn run(a: std.mem.Allocator, fe: api.Frontend, back: api.Backend, compiler: bool, why: []u8) !Outcome {
    var o: Outcome = .{};
    if (!compiler) return o;
    const stream = std.math.maxInt(u64); // no reply's (sessions count up from 1)
    defer back.release(stream);
    for (cells, 0..) |c, i| {
        var arena = std.heap.ArenaAllocator.init(a);
        defer arena.deinit();
        const t = try fe.translate(fe.ctx, arena.allocator(), .{ .form = .fence, .lang = c.lang, .code = c.code });
        const p = switch (t) {
            .program => |p| p,
            .abstain => |no| {
                o.wrong = std.fmt.bufPrint(why, "probe cell {d} did not translate ({t}) though the compiler works", .{ i, no }) catch why;
                return o;
            },
        };
        const job = back.start(stream, p, steps) catch |e| {
            o.wrong = std.fmt.bufPrint(why, "probe cell {d} ({t}) did not start: {t}", .{ i, p.kind, e }) catch why;
            return o;
        };
        const r = wait(back, job);
        if (r.status != .ok or !std.mem.eql(u8, r.out, c.want)) {
            o.wrong = std.fmt.bufPrint(why, "probe cell {d} ({t}, {d} bytes) gave {t} '{f}' after {d} steps, not '{f}'", .{ i, p.kind, p.bytes.len, r.status, std.ascii.hexEscape(r.out[0..@min(r.out.len, 64)], .lower), r.steps, std.ascii.hexEscape(c.want, .lower) }) catch why;
            return o;
        }
        o.ran += 1;
        o.steps += r.steps;
    }
    return o;
}

fn wait(back: api.Backend, job: api.JobId) api.Result {
    while (true) switch (back.poll(job)) {
        .running => {
            var ts: std.c.timespec = .{ .sec = 0, .nsec = std.time.ns_per_ms };
            _ = std.c.nanosleep(&ts, null);
        },
        .done => |r| return r,
    };
}

const gpa = std.testing.allocator;

/// Every cell becomes an ELF of its code; the backend prints the cell's own want, or `out` for every cell.
const Fake = struct {
    out: ?[]const u8 = null,
    which: usize = 0,
    abstain: bool = false,
    started: u32 = 0,
    released: u32 = 0,

    fn frontend(f: *Fake) api.Frontend {
        return .{ .ctx = f, .translate = translate };
    }
    fn translate(ctx: ?*anyopaque, a: std.mem.Allocator, cell: api.Cell) std.mem.Allocator.Error!api.Translation {
        const f: *Fake = @ptrCast(@alignCast(ctx.?));
        if (f.abstain) return .{ .abstain = .unsupported };
        return .{ .program = .{ .kind = .elf, .bytes = try a.dupe(u8, cell.code) } };
    }
    fn backend(f: *Fake) api.Backend {
        return .{ .ctx = f, .vtable = &.{ .start = start, .poll = poll, .cancel = cancel, .release = release } };
    }
    fn start(ctx: *anyopaque, _: u64, p: api.Program, _: u64) error{ OutOfMemory, Busy }!api.JobId {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        for (cells, 0..) |c, i| if (std.mem.eql(u8, c.code, p.bytes)) {
            f.which = i;
        };
        f.started += 1;
        return f.started;
    }
    fn poll(ctx: *anyopaque, _: api.JobId) api.Poll {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        return .{ .done = .{ .status = .ok, .out = f.out orelse cells[f.which].want, .steps = 599 } };
    }
    fn cancel(_: *anyopaque, _: api.JobId) void {}
    fn release(ctx: *anyopaque, _: u64) void {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        f.released += 1;
    }
};

test "runner probe: Python's output passes; a wrong value or a translation that abstains refuses; no compiler skips" {
    var why: [256]u8 = undefined;
    var ok: Fake = .{};
    const got = try run(gpa, ok.frontend(), ok.backend(), true, &why);
    try std.testing.expect(got.wrong == null and got.ran == cells.len and got.steps == 599 * cells.len);
    try std.testing.expectEqual(@as(u32, 1), ok.released);
    // the failure the probe exists for: a circuit (or compiler) that runs the loop to a different value
    var wrong: Fake = .{ .out = "284\n" };
    const bad = try run(gpa, wrong.frontend(), wrong.backend(), true, &why);
    try std.testing.expectEqualStrings("probe cell 0 (elf, 62 bytes) gave ok '284\\x0a' after 599 steps, not '285\\x0a'", bad.wrong.?);
    var unlinked: Fake = .{ .abstain = true };
    const no = try run(gpa, unlinked.frontend(), unlinked.backend(), true, &why);
    try std.testing.expectEqualStrings("probe cell 0 did not translate (unsupported) though the compiler works", no.wrong.?);
    var none: Fake = .{ .out = "284\n" };
    const skipped = try run(gpa, none.frontend(), none.backend(), false, &why);
    try std.testing.expect(skipped.wrong == null and skipped.ran == 0 and none.started == 0);
}

test "runner probe: the cells on the real frontend and the circuit at TF_RUNNER_CIRCUIT (with TF_RUNNER_LLD)" {
    const path = std.c.getenv("TF_RUNNER_CIRCUIT") orelse return error.SkipZigTest;
    const circuit = @import("runner_circuit.zig");
    const backend = @import("runner_circuit_backend.zig");
    const frontend = @import("runner_frontend.zig");
    var c = try circuit.Circuit.load(gpa, std.mem.span(path));
    defer c.deinit(gpa);
    const b = try backend.backend(gpa, &c, 1);
    defer backend.destroy(b);
    var ctx: frontend.Context = .{ .io = std.testing.io };
    if (std.c.getenv("TF_RUNNER_LLD")) |l| ctx.compiler.linker = std.mem.span(l);
    // with TF_RUNNER_CIRCUIT set, a host without the RV32 toolchain fails here by name, not as a missing 285
    const compiler = try @import("runner_frontend_boot.zig").selfTest(gpa, std.testing.io, .{ .compiler = ctx.compiler });
    if (!compiler) std.debug.print("runner probe: clang {s} with linker {s} cannot build RV32 ELF\n", .{ ctx.compiler.clang, ctx.compiler.linker orelse "-fuse-ld=lld" });
    try std.testing.expect(compiler);
    var why: [256]u8 = undefined;
    const got = try run(gpa, ctx.frontend(), b, true, &why);
    if (got.wrong) |w| std.debug.print("runner probe: {s}\n", .{w});
    std.debug.print("runner probe: {d} probe cells = Python's output in {d} steps on {s}\n", .{ got.ran, got.steps, std.mem.span(path) });
    try std.testing.expectEqual(@as(u32, cells.len), got.ran);
}
