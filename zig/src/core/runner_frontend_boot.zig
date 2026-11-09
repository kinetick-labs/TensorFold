//! Model-free boot capability check; only the compiler runs on the host, never the compiled RV32 program.
const std = @import("std");
const program = @import("runner_frontend_program.zig");

pub const Config = struct {
    compiler: program.Config = .{},
    required: bool = false,
};

const source =
    \\int main(void) {
    \\ volatile int value = 7;
    \\ value = value * 6;
    \\ return value == 42 ? 0 : 1;
    \\}
;

/// Returns compiler capability after compile() validates the ELF; required boot fails closed on unavailability.
pub fn selfTest(a: std.mem.Allocator, io: std.Io, config: Config) !bool {
    const elf = try program.compile(a, io, source, config.compiler);
    defer if (elf) |bytes| a.free(bytes);
    return capability(elf != null, config.required);
}

fn capability(available: bool, required: bool) error{RunnerUnavailable}!bool {
    if (required and !available) return error.RunnerUnavailable;
    return available;
}

test "optional and required boot capability policy" {
    try std.testing.expect(!try capability(false, false));
    try std.testing.expect(try capability(true, false));
    try std.testing.expect(try capability(true, true));
    try std.testing.expectError(error.RunnerUnavailable, capability(false, true));
}

test "missing compiler permits ordinary boot and refuses required boot" {
    const a = std.testing.allocator;
    var config: Config = .{ .compiler = .{ .clang = "/nonexistent-tensorfold-boot-clang" } };
    try std.testing.expect(!try selfTest(a, std.testing.io, config));
    config.required = true;
    try std.testing.expectError(error.RunnerUnavailable, selfTest(a, std.testing.io, config));
}
