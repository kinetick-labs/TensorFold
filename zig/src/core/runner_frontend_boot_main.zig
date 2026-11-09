//! Standalone compiler boot probe; enabling the runner requires a working RV32 toolchain.
const std = @import("std");
const boot = @import("runner_frontend_boot.zig");
const usage = "Usage: tf-runner-frontend-check [--require] [--clang PATH] [--linker PATH] [--circuit PATH] [--help]\n";

const Options = struct {
    config: boot.Config = .{},
    help: bool = false,
    circuit: ?[]const u8 = null, // also run the loop probe (core/runner_probe.zig) on this circuit: wrong is exit 1
};

pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    var buffer: [512]u8 = undefined;
    var output = std.Io.File.stderr().writerStreaming(init.io, &buffer);
    defer output.interface.flush() catch {};
    const options = parseWith(args[1..], environment(init.environ_map)) catch {
        try output.interface.writeAll(usage);
        return 2;
    };
    if (options.help) {
        try output.interface.writeAll(usage);
        return 0;
    }
    const available = boot.selfTest(init.gpa, init.io, options.config) catch |err| switch (err) {
        error.RunnerUnavailable => {
            try output.interface.writeAll("runner compiler: unavailable (required; clang/lld RV32 ELF check failed)\n");
            return 1;
        },
        else => return err,
    };
    try output.interface.writeAll(if (available)
        "runner compiler: available (rv32im_zba ELF validated)\n"
    else
        "runner compiler: unavailable (optional; clang/lld RV32 ELF check failed)\n");
    if (options.circuit) |path| return probe(init.gpa, init.io, &output.interface, path, options.config.compiler, available);
    return 0;
}

/// The image's circuit route: the probe cells compiled here and run on `path` must print what Python or C prints.
fn probe(gpa: std.mem.Allocator, io: std.Io, out: *std.Io.Writer, path: []const u8, compiler: @import("runner_frontend_program.zig").Config, available: bool) !u8 {
    if (!available) {
        try out.print("runner circuit {s}: not probed (no RV32 compiler)\n", .{path});
        return 1;
    }
    var c = @import("runner_circuit.zig").Circuit.load(gpa, path) catch |e| {
        try out.print("runner circuit {s}: does not load ({t})\n", .{ path, e });
        return 1;
    };
    defer c.deinit(gpa);
    const backend = @import("runner_circuit_backend.zig");
    const b = try backend.backend(gpa, &c, 1);
    defer backend.destroy(b);
    var ctx: @import("runner_frontend.zig").Context = .{ .io = io, .compiler = compiler };
    var why: [256]u8 = undefined;
    const got = try @import("runner_probe.zig").run(gpa, ctx.frontend(), b, true, &why);
    if (got.wrong) |w| {
        try out.print("runner circuit {s}: WRONG: {s}\n", .{ path, w });
        return 1;
    }
    try out.print("runner circuit {s}: {d} probe cells print the right output in {d} steps\n", .{ path, got.ran, got.steps });
    return 0;
}

fn parse(args: []const [:0]const u8) error{InvalidArguments}!Options {
    return parseWith(args, .{});
}

fn environment(env: *const std.process.Environ.Map) Options {
    var options: Options = .{};
    if (env.get("TF_RUNNER_CLANG")) |path| options.config.compiler.clang = path;
    options.config.compiler.linker = env.get("TF_RUNNER_LLD");
    // fail closed when the image's runner is switched on, under the generic name or a family's (runner_binding.zig)
    options.config.required = on(env.get("TF_RUNNER")) or on(env.get("TF_DSV41_RUNNER"));
    return options;
}

fn on(v: ?[]const u8) bool {
    const value = v orelse return false;
    return value.len != 0 and !std.mem.eql(u8, value, "0") and !std.ascii.eqlIgnoreCase(value, "false") and
        !std.ascii.eqlIgnoreCase(value, "off") and !std.ascii.eqlIgnoreCase(value, "no");
}

fn parseWith(args: []const [:0]const u8, initial: Options) error{InvalidArguments}!Options {
    var options = initial;
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--require")) {
            options.config.required = true;
        } else if (std.mem.eql(u8, arg, "--help")) {
            options.help = true;
        } else if (std.mem.eql(u8, arg, "--clang") or std.mem.eql(u8, arg, "--linker") or std.mem.eql(u8, arg, "--circuit")) {
            i += 1;
            if (i == args.len or args[i].len == 0 or std.mem.startsWith(u8, args[i], "--"))
                return error.InvalidArguments;
            if (std.mem.eql(u8, arg, "--clang")) {
                options.config.compiler.clang = args[i];
            } else if (std.mem.eql(u8, arg, "--circuit")) {
                options.circuit = args[i];
            } else {
                options.config.compiler.linker = args[i];
            }
        } else return error.InvalidArguments;
    }
    return options;
}

test "CLI defaults optional and passes explicit toolchain paths to the boot probe" {
    try std.testing.expect(!(try parse(&.{})).config.required);
    const options = try parse(&.{ "--require", "--clang", "/missing/clang", "--linker", "/missing/ld.lld" });
    try std.testing.expect(options.config.required);
    try std.testing.expectEqualStrings("/missing/clang", options.config.compiler.clang);
    try std.testing.expectEqualStrings("/missing/ld.lld", options.config.compiler.linker.?);
    try std.testing.expectEqualStrings("/c.tfpc", (try parse(&.{ "--circuit", "/c.tfpc" })).circuit.?);
    try std.testing.expectEqual(null, options.circuit);
}

test "CLI refuses missing toolchain paths and unknown arguments" {
    try std.testing.expectError(error.InvalidArguments, parse(&.{"--clang"}));
    try std.testing.expectError(error.InvalidArguments, parse(&.{ "--linker", "--require" }));
    try std.testing.expectError(error.InvalidArguments, parse(&.{ "--clang", "" }));
    try std.testing.expectError(error.InvalidArguments, parse(&.{"--unknown"}));
}

test "environment enables fail closed only for a requested runner" {
    var env = std.process.Environ.Map.init(std.testing.allocator);
    defer env.deinit();
    try std.testing.expect(!environment(&env).config.required);
    for ([_][]const u8{ "", "0", "false", "off", "no", "FALSE" }) |value| {
        try env.put("TF_DSV41_RUNNER", value);
        try std.testing.expect(!environment(&env).config.required);
    }
    for ([_][]const u8{ "1", "true", "on", "yes", "invalid" }) |value| {
        try env.put("TF_DSV41_RUNNER", value);
        try std.testing.expect(environment(&env).config.required);
    }
    try env.put("TF_DSV41_RUNNER", "0");
    try env.put("TF_RUNNER", "1");
    try std.testing.expect(environment(&env).config.required);
    try env.put("TF_RUNNER_CLANG", "/missing/compiler");
    try env.put("TF_RUNNER_LLD", "/missing/linker");
    const options = environment(&env);
    try std.testing.expectEqualStrings("/missing/compiler", options.config.compiler.clang);
    try std.testing.expectEqualStrings("/missing/linker", options.config.compiler.linker.?);
    const overridden = try parseWith(&.{ "--clang", "/explicit/compiler" }, options);
    try std.testing.expectEqualStrings("/explicit/compiler", overridden.config.compiler.clang);
}
