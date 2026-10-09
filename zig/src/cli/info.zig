//! The `info` command: one checkpoint's family, format, context, weights size and memory floor.
const std = @import("std");
const Allocator = std.mem.Allocator;
const hub = @import("hub.zig");

pub const GiB: u64 = 1 << 30;

/// Kept back for kernels, activations and buffers that grow with long prompts; the same margin the
/// server's prompt-cache budget reserves (native/cache_fit.zig's MARGIN).
pub const runtime_margin: u64 = 2 * GiB;

/// Prints what a Zig family would serve from `model` (a directory or a cached repo id).
/// Returns the process exit code.
pub fn run(a: Allocator, io: std.Io, out: *std.Io.Writer, env: ?*const std.process.Environ.Map, override: ?[]const u8, model: []const u8) !u8 {
    const dir: []const u8 = blk: {
        if (hub.isDir(io, model)) break :blk model;
        if (hub.isRepoIdLike(model)) {
            const root = try hub.cacheDir(a, env, override);
            if (try hub.cachedSnapshot(a, io, root, model)) |snapshot| break :blk snapshot;
            try out.print("{s} is not in the Hugging Face cache; run: tensorfold pull {s}\n", .{ model, model });
            return 1;
        }
        try out.print("{s} is neither a directory nor a Hugging Face repo id (owner/name)\n", .{model});
        return 1;
    };
    const model_type = hub.modelType(a, io, dir);
    const family = hub.family(model_type) orelse {
        try out.print("{s}: no registered Zig family serves model_type {s}; the 0.6 line may: tensorfold@0.6\n", .{ model, model_type });
        return 1;
    };
    const weights = try hub.sizeOf(a, io, dir);
    try out.print("family: {s} ({s})\n", .{ family.title, family.model_type });
    if (hub.format(a, io, dir)) |f| switch (f) {
        .affine => |q| try out.print("format: {d}-bit, group {d}\n", .{ q.bits, q.group }),
        .exl3 => |q| {
            try out.print("format: exl3, codebook {s}", .{q.codebook});
            if (q.bits) |b| try out.print(", mean {d:.2} bits/weight", .{b});
            if (q.head_bits) |h| try out.print(", head {d}", .{h});
            if (q.mtp_bits) |m| try out.print(", mtp {d}", .{m});
            try out.writeAll("\n");
        },
    } else {
        try out.print("format: full precision\n", .{});
    }
    if (hub.configInt(a, io, dir, "max_position_embeddings")) |context| {
        try out.print("context: {d}\n", .{context});
    }
    try out.print("weights: {d:.2} GiB ({d} bytes)\n", .{ @as(f64, @floatFromInt(weights)) / GiB, weights });
    try out.print("memory floor: {d:.2} GiB (weights plus a {d:.2} GiB runtime margin)\n", .{
        @as(f64, @floatFromInt(weights + runtime_margin)) / GiB,
        @as(f64, @floatFromInt(runtime_margin)) / GiB,
    });
    return 0;
}

test "info prints family, format, context, weights and floor for a cached checkpoint" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const io = std.testing.io;
    const root = try std.fmt.allocPrint(a, ".tf-info-test-{d}", .{std.Io.Clock.awake.now(io).toNanoseconds()});
    defer std.Io.Dir.cwd().deleteTree(io, root) catch {};
    const w = std.Io.Dir.cwd();
    try w.createDirPath(io, try std.fs.path.join(a, &.{ root, "models--Org--Flash/snapshots/rev1" }));
    try w.createDirPath(io, try std.fs.path.join(a, &.{ root, "models--Org--Flash/refs" }));
    try w.writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ root, "models--Org--Flash/refs/main" }), .data = "rev1" });
    try w.writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ root, "models--Org--Flash/snapshots/rev1/config.json" }), .data = "{\"model_type\": \"qwen4_exp\", \"quantization\": {\"bits\": 6, \"group_size\": 32}, \"max_position_embeddings\": 65536}" });
    var weights: [8192]u8 = undefined;
    @memset(&weights, 'x');
    try w.writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ root, "models--Org--Flash/snapshots/rev1/weights.safetensors" }), .data = &weights });

    var out: std.Io.Writer.Allocating = .init(a);
    try std.testing.expectEqual(@as(u8, 0), try run(a, io, &out.writer, null, root, "Org/Flash"));
    try std.testing.expect(has(out.written(), "family: Qwen3.8 Flash Next (qwen4_exp)"));
    try std.testing.expect(has(out.written(), "format: 6-bit, group 32"));
    try std.testing.expect(has(out.written(), "context: 65536"));
    try std.testing.expect(has(out.written(), "weights: 0.00 GiB (8300 bytes)"));
    try std.testing.expect(has(out.written(), "memory floor: 2.00 GiB"));

    var refused: std.Io.Writer.Allocating = .init(a);
    try w.createDirPath(io, try std.fs.path.join(a, &.{ root, "models--Org--Draft/snapshots/rev2" }));
    try w.writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ root, "models--Org--Draft/snapshots/rev2/config.json" }), .data = "{\"model_type\": \"gemma4\"}" });
    try std.testing.expectEqual(@as(u8, 1), try run(a, io, &refused.writer, null, root, "Org/Draft"));
    try std.testing.expect(has(refused.written(), "no registered Zig family serves model_type gemma4"));

    // An ExLlamaV3 pack: the format line names the codebook and the two fixed heads off the config alone,
    // while the per-tensor width stays a load-time read (format.py:bits_of).
    try w.createDirPath(io, try std.fs.path.join(a, &.{ root, "models--Org--Exl3/snapshots/rev3" }));
    try w.writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ root, "models--Org--Exl3/snapshots/rev3/config.json" }), .data = "{\"model_type\": \"qwen4_exp\", \"quantization_config\": {\"quant_method\": \"exl3\", \"version\": \"1.4.4\", \"bits\": 4.05, \"head_bits\": 6, \"mtp_bits\": 4, \"codebook\": \"mul1\", \"out_scales\": \"always\"}}" });
    try w.writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ root, "models--Org--Exl3/snapshots/rev3/weights.safetensors" }), .data = &weights });

    var exl3: std.Io.Writer.Allocating = .init(a);
    try std.testing.expectEqual(@as(u8, 0), try run(a, io, &exl3.writer, null, root, "Org/Exl3"));
    try std.testing.expect(has(exl3.written(), "family: Qwen3.8 Flash Next (qwen4_exp)"));
    try std.testing.expect(has(exl3.written(), "format: exl3, codebook mul1, mean 4.05 bits/weight, head 6, mtp 4"));
}

fn has(text: []const u8, needle: []const u8) bool {
    if (std.mem.indexOf(u8, text, needle) != null) return true;
    std.debug.print("missing \"{s}\" in:\n{s}\n", .{ needle, text });
    return false;
}
