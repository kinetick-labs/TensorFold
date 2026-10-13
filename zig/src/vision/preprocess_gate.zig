//! preprocess-gate DIR: each DIR/<case>/ (case.json, rgb.u8, pixels.f32 from tools/zig/flashnext_vision_oracle.py)
//! through qwen_preprocess; the grid and every patch value must equal Python's bytes.
const std = @import("std");
const pre = @import("qwen_preprocess.zig");

pub fn main(init: std.process.Init) !u8 {
    const a = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const dir_path = args[1];
    var dir = try std.Io.Dir.cwd().openDir(io, dir_path, .{ .iterate = true });
    defer dir.close(io);
    var it = dir.iterate();
    var bad: usize = 0;
    var total: usize = 0;
    while (try it.next(io)) |e| {
        if (e.kind != .directory) continue;
        var sub = try dir.openDir(io, e.name, .{});
        defer sub.close(io);
        const cj = try sub.readFileAlloc(io, "case.json", a, .limited(1 << 20));
        defer a.free(cj);
        const C = struct { h: u32, w: u32, max_pixels: u64, min_pixels: u64, grid: [3]u32 };
        const parsed = try std.json.parseFromSlice(C, a, cj, .{});
        defer parsed.deinit();
        const c = parsed.value;
        const rgb = try sub.readFileAlloc(io, "rgb.u8", a, .limited(1 << 30));
        defer a.free(rgb);
        const want_bytes = try sub.readFileAlloc(io, "pixels.f32", a, .limited(1 << 31));
        defer a.free(want_bytes);
        const g: pre.Geometry = .{};
        const s = try pre.smartResize(c.h, c.w, g.factor(), c.min_pixels, c.max_pixels);
        const img = try pre.resizeBicubicAA(a, rgb, c.h, c.w, 3, s.h, s.w);
        defer a.free(img);
        const px = try pre.patchify(a, img, s.h, s.w, g);
        defer a.free(px);
        total += 1;
        const grid_ok = s.h / 16 == c.grid[1] and s.w / 16 == c.grid[2];
        const got = std.mem.sliceAsBytes(px);
        var diff: usize = 0;
        if (got.len == want_bytes.len) {
            const wf = std.mem.bytesAsSlice(f32, want_bytes);
            for (px, wf) |x, y| diff += @intFromBool(@as(u32, @bitCast(x)) != @as(u32, @bitCast(y)));
        }
        const ok = grid_ok and got.len == want_bytes.len and diff == 0;
        if (!ok) bad += 1;
        std.debug.print("{s} {s}: {d}x{d} -> {d}x{d} (python grid {d}x{d}) {d} values differ\n", .{ if (ok) "EQUAL" else "DIFFER", e.name, c.h, c.w, s.h, s.w, c.grid[1], c.grid[2], diff });
    }
    std.debug.print("{s} preprocess-gate: {d} of {d} equal\n", .{ if (bad == 0) "PASS" else "FAIL", total - bad, total });
    return if (bad == 0) 0 else 1;
}
