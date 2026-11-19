//! A picture into the vision tower's input, as transformers' HYV4VLImageProcessor (torchvision backend) makes it:
//! smart resize to multiples of 32, torch's antialiased uint8 bicubic, (v - 127.5) / 127.5 (its fused rescale and normalize), then row-major 16 x 16
//! patches, channel-major inside each.

const std = @import("std");
const resample = @import("picture_resample");
const image_grid = @import("image_grid.zig");

pub const Picture = struct {
    size: image_grid.Size,
    grid: image_grid.Grid,
    /// `[patches][3 * patch * patch]`
    patches: []f32,

    pub fn deinit(p: *Picture, a: std.mem.Allocator) void {
        a.free(p.patches);
    }
};

/// Packed RGB `width` x `height` into patches; refuses sizes outside the limits as smartResize does.
pub fn picture(a: std.mem.Allocator, rgb: []const u8, width: u32, height: u32, limits: image_grid.Limits) !Picture {
    const size = try image_grid.smartResize(height, width, limits);
    const resized = try resample.resize(a, .torch, .of(rgb, width, height), size.width, size.height);
    defer a.free(resized);
    const patches = try a.alloc(f32, 3 * @as(usize, size.width) * size.height);
    errdefer a.free(patches);
    normalizedPatches(resized, size, limits.patch, patches);
    return .{ .size = size, .grid = .of(size, limits), .patches = patches };
}

/// Resized packed RGB to normalized row-major patches in one pass.
pub fn normalizedPatches(rgb: []const u8, s: image_grid.Size, patch: u32, out: []f32) void {
    std.debug.assert(rgb.len == out.len and out.len == 3 * @as(usize, s.width) * s.height);
    var i: usize = 0;
    for (0..s.height / patch) |py| for (0..s.width / patch) |px| for (0..3) |ch| for (0..patch) |dy| for (0..patch) |dx| {
        const v: f32 = @floatFromInt(rgb[((py * patch + dy) * s.width + px * patch + dx) * 3 + ch]);
        // the processor fuses the rescale into the normalization: mean and std 0.5 / (1 / 255) = 127.5
        out[i] = (v - 127.5) / 127.5;
        i += 1;
    };
}
