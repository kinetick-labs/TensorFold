//! A store that invalidates its previous view on every read, like advancing mapped shards.
const std = @import("std");
const core = @import("core");
const source = @import("cuda_projection_load.zig");
const Tensor = core.safetensors.Tensor;
const fp8 = core.quant.fp8;

const Fake = struct {
    scratch: [1024]u8 = undefined,
    bad: enum { none, weight, scale, extent, width, missing } = .none,

    pub fn tensor(self: *Fake, name: []const u8) !Tensor {
        const part: usize = if (std.mem.startsWith(u8, name, "root.a.")) 0 else if (std.mem.startsWith(u8, name, "root.b.")) 1 else return error.MissingTensor;
        const n: usize = if (part == 0) 3 else 5;
        const k: usize = if (self.bad == .width and part == 1) 128 else 64;
        const scale = std.mem.endsWith(u8, name, ".weight_scale");
        if (scale and self.bad == .missing) return error.MissingTensor;
        if (!scale and !std.mem.endsWith(u8, name, ".weight")) return error.MissingTensor;
        const cols = if (scale) k / 32 else k;
        const len = n * cols;
        @memset(&self.scratch, 0xee);
        for (self.scratch[0..len], 0..) |*v, i| v.* = @truncate(i * 17 + part * 53 + @as(usize, if (scale) 129 else 7));
        return .{
            .dtype = if (scale) (if (self.bad == .scale) .f8_e4m3 else .u8) else (if (self.bad == .weight) .bf16 else .f8_e4m3),
            .rank = 2,
            .shape = .{ n, cols, 1, 1, 1 },
            .bytes = self.scratch[0 .. len - @as(usize, if (self.bad == .extent) 1 else 0)],
        };
    }
};

fn trial(a: std.mem.Allocator) !void {
    var fake: Fake = .{};
    var value = try source.readStitched(a, &fake, "root", &.{ ".a", ".b" });
    defer value.deinit(a);
    try std.testing.expectEqual(@as(usize, 8), value.n);
    try std.testing.expectEqual(@as(usize, 64), value.k);
    for (0..2) |part| {
        const start: usize = if (part == 0) 0 else 3;
        const n: usize = if (part == 0) 3 else 5;
        for (0..n * 64) |i| try std.testing.expectEqual(@as(u8, @truncate(i * 17 + part * 53 + 7)), value.codes[start * 64 + i]);
        for (0..n * 2) |i| try std.testing.expectEqual(@as(u8, @truncate(i * 17 + part * 53 + 129)), value.scales[start * 2 + i]);
    }
}

test "MXFP8 fusion owns each shard plane and preserves exact output row order" {
    try trial(std.testing.allocator);
}

test "MXFP8 fusion cleans every partial allocation on failure" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, trial, .{});
}

fn fromBf16(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

test "the widened plane holds each code times its group's power of two, exactly" {
    const a = std.testing.allocator;
    const n: usize = 4;
    const k: usize = 64; // two groups a row
    const codes = try a.alloc(u8, n * k);
    defer a.free(codes);
    const scales = try a.alloc(u8, n * (k / 32));
    defer a.free(scales);
    for (codes, 0..) |*c, i| c.* = @truncate(i); // every e4m3 code once
    for (scales, 0..) |*s, i| s.* = @truncate(96 + i * 8); // 2^-31 ... 2^25, a shipped checkpoint's range
    const value = source.Micro{ .n = n, .k = k, .codes = codes, .scales = scales };
    const wide = try source.widen(a, value);
    defer a.free(wide);
    for (0..n) |row| {
        for (0..k) |col| {
            const code = codes[row * k + col];
            const e = scales[row * (k / 32) + col / 32];
            const want: f64 = fp8.e4m3(code) * std.math.pow(f64, 2, @as(f64, @floatFromInt(e)) - 127);
            const got = fromBf16(std.mem.readInt(u16, wide[(row * k + col) * 2 ..][0..2], .little));
            if (std.math.isNan(want)) {
                try std.testing.expect(std.math.isNan(got)); // e4m3's 0x7f/0xff stay NaN
                continue;
            }
            try std.testing.expectEqual(@as(f32, @floatCast(want)), got);
        }
    }
}

test "MXFP8 source refuses incompatible shapes, dtypes, extents and missing scales" {
    const a = std.testing.allocator;
    for ([_]@FieldType(Fake, "bad"){ .weight, .scale, .extent, .width }) |bad| {
        var fake: Fake = .{ .bad = bad };
        try std.testing.expectError(error.UnexpectedTensor, source.readStitched(a, &fake, "root", &.{ ".a", ".b" }));
    }
    var fake: Fake = .{ .bad = .missing };
    try std.testing.expectError(error.MissingTensor, source.readMicro(a, &fake, "root.a"));
    try std.testing.expectError(error.UnexpectedTensor, source.readStitched(a, &fake, "root", &.{}));
    try std.testing.expectError(error.UnexpectedTensor, source.readStitched(a, &fake, "root", &.{ ".a", ".a", ".a", ".a", ".a", ".a", ".a" }));
}
