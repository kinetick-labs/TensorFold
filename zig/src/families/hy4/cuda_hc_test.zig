//! zig/kernels/cuda/hy4/hc.cu on a GPU against the CPU reference (layer_ref.hyper): a few tokens at the release's
//! shape (4 streams of 3072), a few MiB of device memory, so it runs beside a serving model. On demand: HY4_HC_PTX
//! names the kernel's PTX (nvcc -ptx); the driver JIT-compiles it for whatever GPU is there.

const std = @import("std");
const cuda = @import("cuda");
const layer_ref = @import("layer_ref.zig");

fn put(d: *const cuda.Driver, xs: []const f32) !cuda.DeviceBuffer {
    return cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(xs));
}

test "Hy4 CUDA hc: pre and post match the CPU reference at 4 x 3072 (HY4_HC_PTX, a GPU, on demand)" {
    const path = std.testing.environ.getPosix("HY4_HC_PTX") orelse return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const ptx_bytes = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, a, .limited(16 << 20));
    const z = try a.alloc(u8, ptx_bytes.len + 1);
    @memcpy(z[0..ptx_bytes.len], ptx_bytes);
    z[ptx_bytes.len] = 0;
    const ptx: [:0]const u8 = z[0..ptx_bytes.len :0];

    const t = 5;
    const n = 4;
    const hidden = 3072;
    const d_ref: layer_ref.Dims = .{ .hidden = hidden, .hc = n, .heads = 1, .nope = 0, .rope = 0, .v = 0, .q_lora = 0, .kv_lora = 0, .inter = 0, .eps = 1e-5, .hc_eps = 1e-6, .hc_magnitude = 2.0, .theta = 1e4, .sections = .{ 0, 0, 0 } };
    var rng = std.Random.DefaultPrng.init(4);
    const r = rng.random();
    const streams = try a.alloc(f32, t * n * hidden);
    for (streams) |*x| x.* = (r.float(f32) * 2 - 1) * 3;
    const fn_ = try a.alloc(f32, 2 * n * n * hidden);
    for (fn_) |*x| x.* = (r.float(f32) * 2 - 1) * 0.02;
    var base: [2 * n]f32 = undefined;
    for (&base) |*x| x.* = r.float(f32) - 0.5;
    const scale = [2]f32{ 0.7, 1.3 };
    const y = try a.alloc(f32, t * hidden);
    for (y) |*x| x.* = r.float(f32) * 2 - 1;

    // the reference
    const want_mixed = try a.alloc(f32, t * hidden);
    const want_post = try a.alloc(f32, t * n);
    const want_streams = try a.dupe(f32, streams);
    for (0..t) |i| layer_ref.hyper(&d_ref, .{ .fn_ = fn_, .base = &base, .scale = &scale }, streams[i * n * hidden ..][0 .. n * hidden], want_post[i * n ..][0..n], want_mixed[i * hidden ..][0..hidden]);
    for (0..t) |i| for (0..n) |s| {
        for (want_streams[(i * n + s) * hidden ..][0..hidden], y[i * hidden ..][0..hidden]) |*e, v| e.* += want_post[i * n + s] * v;
    };

    var d = try cuda.Driver.open();
    defer d.close();
    var ctx = try cuda.Context.init(&d, 0);
    defer ctx.deinit();
    var log: [8192]u8 = undefined;
    var m = cuda.Module.loadPtx(&d, ptx, &log) catch |e| {
        std.debug.print("PTX JIT: {s}\n", .{std.mem.sliceTo(&log, 0)});
        return e;
    };
    defer m.unload();
    const pre = try m.function("hy4_hc_pre");
    const post_fn = try m.function("hy4_hc_post");
    var stream = try cuda.Stream.init(&d, true);
    defer stream.deinit();

    var d_streams = try put(&d, streams);
    defer d_streams.free();
    var d_fn = try put(&d, fn_);
    defer d_fn.free();
    var d_base = try put(&d, &base);
    defer d_base.free();
    var d_scale = try put(&d, &scale);
    defer d_scale.free();
    var d_y = try put(&d, y);
    defer d_y.free();
    var d_mixed = try cuda.DeviceBuffer.alloc(&d, t * hidden * 4);
    defer d_mixed.free();
    var d_post = try cuda.DeviceBuffer.alloc(&d, t * n * 4);
    defer d_post.free();

    const cfg: cuda.Config = .{ .grid = .{ .x = t }, .block = .{ .x = 256 } };
    var args: cuda.Args = .{};
    for ([_]u64{ d_streams.ptr, d_fn.ptr, d_base.ptr, d_scale.ptr, d_mixed.ptr, d_post.ptr }) |p| args.add(p);
    args.add(@as(c_int, n));
    args.add(@as(c_int, hidden));
    args.add(@as(f32, d_ref.eps));
    args.add(@as(f32, d_ref.hc_eps));
    args.add(@as(f32, d_ref.hc_magnitude));
    try cuda.launch.launch(pre, cfg, stream, &args);
    var args2: cuda.Args = .{};
    for ([_]u64{ d_streams.ptr, d_post.ptr, d_y.ptr }) |p| args2.add(p);
    args2.add(@as(c_int, n));
    args2.add(@as(c_int, hidden));
    try cuda.launch.launch(post_fn, cfg, stream, &args2);
    try stream.synchronize();

    const got_mixed = try a.alloc(f32, t * hidden);
    const got_post = try a.alloc(f32, t * n);
    const got_streams = try a.alloc(f32, t * n * hidden);
    try d_mixed.download(0, std.mem.sliceAsBytes(got_mixed));
    try d_post.download(0, std.mem.sliceAsBytes(got_post));
    try d_streams.download(0, std.mem.sliceAsBytes(got_streams));
    const worst = struct {
        fn f(g: []const f32, w: []const f32) f32 {
            var e: f32 = 0;
            for (g, w) |x, v| e = @max(e, @abs(x - v) / @max(1.0, @abs(v)));
            return e;
        }
    }.f;
    const e_mixed = worst(got_mixed, want_mixed);
    const e_post = worst(got_post, want_post);
    const e_streams = worst(got_streams, want_streams);
    std.debug.print("hy4 hc on the GPU: worst relative error mixed {e}, post {e}, streams {e}\n", .{ e_mixed, e_post, e_streams });
    try std.testing.expect(e_mixed <= 1e-4 and e_post <= 1e-5 and e_streams <= 1e-4);
}
