//! The build's time limit: a slow compiler keeps what is left of the translate deadline, not a shorter cap of its own.
const std = @import("std");
const compiler = @import("runner_frontend_program_compile.zig");
const elf_cache = @import("runner_frontend_elf_cache.zig");
const t = std.testing;

const body = "int main(void) {\n return 0;\n}\n";

/// Milliseconds compile() took to decline `body` with `clang`, the deadline `deadline_ms` away (or none).
fn timed(clang: []const u8, deadline_ms: ?i64) !i64 {
    const start = std.Io.Clock.awake.now(t.io);
    var config: compiler.Config = .{ .clang = clang, .opaque_inputs = true, .estimated_steps = 40 };
    if (deadline_ms) |ms| config.deadline = start.addDuration(.fromMilliseconds(ms));
    try t.expectEqual(@as(?[]u8, null), try compiler.compile(t.allocator, t.io, body, config));
    return @intCast(@divTrunc(start.durationTo(std.Io.Clock.awake.now(t.io)).toNanoseconds(), std.time.ns_per_ms));
}

test "a compiler slower than 1 s still ends on its own inside the translate deadline; a nearer deadline stops it" {
    elf_cache.reset(t.io);
    defer elf_cache.reset(t.io);
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const file = try tmp.dir.createFile(t.io, "slow-clang", .{ .permissions = .fromMode(0o700) });
    try file.writeStreamingAll(t.io, "#!/bin/sh\nsleep 1.3\nexit 1\n");
    file.close(t.io);
    const clang = try tmp.dir.realPathFileAlloc(t.io, "slow-clang", t.allocator);
    defer t.allocator.free(clang);
    // the first ELF compile of a process on a cold page cache took 1.0 s on a DGX Spark and was cut at the old 1000 ms cap
    try t.expect(try timed(clang, 1900) >= 1300);
    try t.expect(try timed(clang, null) >= 1300);
    try t.expect(try timed(clang, 300) < 1000);
}
