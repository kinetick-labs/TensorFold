//! --vision for our Flash Next engine: upstream's Qwen-VL tower (vision/qwen_cuda.zig) from its own folder, as Media.
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const lanes = @import("lanes");
const shared = @import("qwen_vision");
const Media = lanes.stream.Media;

/// The tower's folder (vision_tower.* bf16 with config.json and the processor files); else the checkpoint's own.
pub const env_dir = "TENSORFOLD_VISION_TOWER_DIR";
/// A folder each image prompt's Media is written to (features, rows, positions) for the bridge-vs-helper check.
pub const env_dump = "TENSORFOLD_VISION_DUMP";
pub const Limits = shared.media.Limits;

pub fn towerDir(checkpoint: []const u8) []const u8 {
    const p = std.c.getenv(env_dir) orelse return checkpoint;
    const v = std.mem.span(p);
    return if (v.len > 0) v else checkpoint;
}

pub const Vision = struct {
    pub const template_wraps = true; // the template writes vision_start and vision_end around the image token
    gpa: std.mem.Allocator,
    io: std.Io,
    d: *const cuda.Driver,
    stream: cuda.Stream,
    tower: *shared.Tower,
    settings: shared.Settings,
    limits: Limits,
    marks: struct { open: u32, image: u32, close: u32 },
    mark_ids: [3]u32,
    dumped: u32 = 0,

    /// Loads the tower on this GPU before the engine reads its memory budget (rank 0, one rank).
    pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8, limits: Limits) !*Vision {
        try shared.media.checkLimits(limits);
        const settings = (try shared.Settings.read(gpa, io, dir)) orelse return error.NoVisionTower;
        const c = settings.tower;
        const v = try gpa.create(Vision);
        errdefer gpa.destroy(v);
        var stream = try cuda.Stream.init(d, true);
        errdefer stream.deinit();
        const tower = try shared.Tower.load(gpa, io, d, stream, dir, c);
        v.* = .{
            .gpa = gpa,
            .io = io,
            .d = d,
            .stream = stream,
            .tower = tower,
            .settings = settings,
            .limits = limits,
            .marks = .{ .open = c.vision_start, .image = c.image_token, .close = c.vision_end },
            .mark_ids = .{ c.vision_start, c.image_token, c.vision_end },
        };
        return v;
    }

    pub fn deinit(v: *Vision) void {
        v.tower.deinit();
        v.stream.deinit();
        v.gpa.destroy(v);
    }

    /// An image's bytes to the tower's patches (the server's prepare, on its own thread: CPU only).
    pub fn prepare(v: *Vision, a: std.mem.Allocator, bytes: []const u8, max_tokens: u32) !struct {
        pixels: []f32,
        gh: u32,
        gw: u32,
        tokens: u32,
        hash: u64,
    } {
        if (max_tokens == 0 or max_tokens > v.limits.image_tokens) return error.BadImageLimits;
        const p = try v.settings.prepare(v.io, a, bytes, max_tokens);
        return .{ .pixels = p.pixels, .gh = p.gh, .gw = p.gw, .tokens = p.tokens, .hash = p.hash };
    }

    /// A prompt's images as our engine's Media: upstream's plan (rows, M-RoPE, delta) and the tower's bf16 rows.
    pub fn media(v: *Vision, ids: []const u32, images: []const lanes.stream.Image) !*Media {
        var plan = try shared.media.placeholders(v.gpa, v.settings.tower, ids, images, v.limits);
        defer plan.deinit();
        const width = v.settings.tower.out_hidden;
        var dev = try cuda.DeviceBuffer.alloc(v.d, plan.rows * width * 2);
        defer dev.free();
        var row: usize = 0;
        for (images) |im| {
            try v.tower.encode(im.pixels, &.{.{ .h = im.gh, .w = im.gw }}, dev.ptr + row * width * 2);
            row += im.tokens;
        }
        try v.stream.synchronize();
        const n = plan.src.len;
        const features = try v.gpa.alloc(u8, plan.rows * width * 2);
        errdefer v.gpa.free(features);
        try dev.download(0, features);
        const rows = try v.gpa.alloc(u32, plan.rows);
        errdefer v.gpa.free(rows);
        var k: usize = 0;
        for (plan.src, 0..) |src, r| if (src >= 0) {
            if (k >= rows.len or src != @as(i32, @intCast(k))) return error.BadImagePlaceholders;
            rows[k] = @intCast(r);
            k += 1;
        };
        if (k != rows.len) return error.BadImagePlaceholders;
        const positions = try v.gpa.alloc(i32, 3 * n);
        errdefer v.gpa.free(positions);
        rowMajor(plan.pos, positions, n);
        const m = try v.gpa.create(Media);
        m.* = .{ .rows = rows, .positions = positions, .delta = plan.delta, .features = features, .width = @intCast(width) };
        try m.check(n);
        v.dump(m);
        return m;
    }

    pub fn free(v: *Vision, m: *const Media) void {
        v.gpa.free(m.features);
        v.gpa.free(m.rows);
        v.gpa.free(m.positions);
        v.gpa.destroy(@constCast(m));
    }

    /// TENSORFOLD_VISION_DUMP: `bridge-<n>.{features,rows,pos}` raw (bf16, u32, i32 [n, 3]); a failed write only logs.
    fn dump(v: *Vision, m: *const Media) void {
        const p = std.c.getenv(env_dump) orelse return;
        v.dumped += 1;
        writeMedia(v.gpa, v.io, std.mem.span(p), "bridge", v.dumped, m) catch |e|
            std.log.warn("vision dump: {s}", .{@errorName(e)});
    }

    pub fn explain(err: anyerror) ?[]const u8 {
        return switch (err) {
            error.BadImageLimits, error.BadImageCount => "image count or row budget exceeds the configured limits",
            error.BadImageGrid, error.BadImagePatches, error.BadImagePlaceholders => "image rows do not match the prompt placeholders",
            error.NoVisionTower, error.UnsupportedQwenVision => "no Qwen-VL tower: set TENSORFOLD_VISION_TOWER_DIR to a vision_tower.* folder",
            else => core.qwen_vl.geometry.problem(err),
        };
    }
};

/// Upstream's planar [3, n] positions as our engine's [n, 3] rows.
pub fn rowMajor(planar: []const i32, out: []i32, n: usize) void {
    for (0..n) |r| for (0..3) |a| {
        out[r * 3 + a] = planar[a * n + r];
    };
}

/// One Media's three raw files in `dir`: `<tag>-<n>.features`, `.rows` and `.pos`.
pub fn writeMedia(a: std.mem.Allocator, io: std.Io, dir: []const u8, tag: []const u8, n: u32, m: *const Media) !void {
    const parts = [_]struct { []const u8, []const u8 }{
        .{ "features", m.features },
        .{ "rows", std.mem.sliceAsBytes(m.rows) },
        .{ "pos", std.mem.sliceAsBytes(m.positions) },
    };
    for (parts) |part| {
        const path = try std.fmt.allocPrint(a, "{s}/{s}-{d}.{s}", .{ dir, tag, n, part[0] });
        defer a.free(path);
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = part[1] });
    }
}

test "planar M-RoPE positions become [n, 3] rows" {
    const planar = [_]i32{ 0, 1, 1, 3, 0, 1, 2, 3, 0, 1, 1, 3 };
    var out: [12]i32 = undefined;
    rowMajor(&planar, &out, 4);
    try std.testing.expectEqualSlices(i32, &.{ 0, 0, 0, 1, 1, 1, 1, 2, 1, 3, 3, 3 }, &out);
}
