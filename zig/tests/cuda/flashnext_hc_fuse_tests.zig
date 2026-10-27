//! The native read-out's fused kernels (flashnext_hcf.cu, hcn) against the five-launch chain they replace: every output byte,
//! at several row counts, then the launch times of both chains at a prompt chunk's rows.
const std = @import("std");
const cuda = @import("cuda");
const flash = @import("flashnext");
const check = @import("check.zig");
const rig = @import("flashnext_rig.zig");

const Rig = rig.Rig;
const tk = flash.tile_kernels;

const dims = 2560;
const streams = 4;
const wide = streams * dims;
const low = 320;
const n_down = low + streams;
const nc = dims / 256;
const eps: f32 = 1e-6;

const Chain = struct {
    r: *Rig,
    g: *const flash.Glue,
    lane: *const cuda.qmmf.Lane,
    t: *const tk.TileKernels,
    wdown: cuda.qmmf.Weight,
    wup: cuda.qmmf.Weight,
    h: u64,
    pss: u64,
    scale: u64,
    norm: u64,
    part: u64,
    act: u64,
    xs: u64,
    inj: u64,
    up: u64,
    mixed: u64,
    xsm: u64,
    spart: u64,
    rows: usize,

    fn part_poison(c: *Chain) !void {
        const d = c.r.gpu.d;
        const bufs = [_][2]u64{ .{ c.part, c.rows * n_down * 4 }, .{ c.mixed, c.rows * dims * 2 }, .{ c.xsm, c.rows * (dims / 32) * 4 } };
        for (bufs) |b| try (cuda.DeviceBuffer{ .d = d, .ptr = b[0], .len = b[1] }).fill8(0xff, c.r.s.handle);
    }
    fn baseNorm(c: *Chain) !void {
        try c.g.hcNormalize(c.r.s, c.h, c.pss, c.scale, c.norm, eps, c.rows, dims, streams);
    }
    fn baseDown(c: *Chain) !void {
        try c.lane.matmulF32(c.r.s, c.norm, wide, c.rows, c.wdown, c.part, c.spart);
    }
    fn reduce(c: *Chain) !void {
        try c.g.hcReduceAct(c.r.s, c.part, c.act, c.xs, c.inj, 1, c.rows, streams, low, n_down, 1);
    }
    fn baseUp(c: *Chain) !void {
        try c.lane.matmul(c.r.s, c.act, low, c.rows, c.wup, c.up, c.spart);
    }
    fn baseMix(c: *Chain) !void {
        try c.g.hcDenseMix(c.r.s, c.up, c.norm, c.mixed, c.xsm, c.rows, dims, streams);
    }
    fn newDown(c: *Chain, vi: usize) !void {
        const mt = tk.hcn_down_mt[vi];
        var a: cuda.Args = .{};
        for ([_]u64{ c.h, c.pss, c.scale, c.wdown.codes, c.part }) |v| a.add(v);
        for ([_]usize{ c.rows, n_down, wide, cuda.qmmf.splitK(n_down, wide), dims, nc, streams }) |v| a.add(@as(c_int, @intCast(v)));
        a.add(eps);
        const cdiv = (c.rows + 16 * mt - 1) / (16 * mt);
        try cuda.launch.launch(c.t.hcn_down[vi], .{ .grid = .{ .x = @intCast(cdiv * 2) }, .block = .{ .x = 224 }, .shared = tk.hcnDownSmem(mt) }, c.r.s, &a);
    }
    fn newDown2(c: *Chain, vi: usize) !void {
        const mt = tk.hcn_dv2_mt[vi];
        var a: cuda.Args = .{};
        for ([_]u64{ c.h, c.pss, c.scale, c.wdown.codes, c.part }) |v| a.add(v);
        for ([_]usize{ c.rows, n_down, wide, cuda.qmmf.splitK(n_down, wide), dims, nc, streams }) |v| a.add(@as(c_int, @intCast(v)));
        a.add(eps);
        const cdiv = (c.rows + 16 * mt - 1) / (16 * mt);
        try cuda.launch.launch(c.t.hcn_dv2[vi], .{ .grid = .{ .x = @intCast(cdiv * 2) }, .block = .{ .x = 224 }, .shared = tk.hcnDownSmem(mt) }, c.r.s, &a);
    }
    fn upDbg(c: *Chain, vi: usize) !void {
        const mt = 3;
        var a: cuda.Args = .{};
        for ([_]u64{ c.act, c.wup.codes, c.h, c.pss, c.scale, c.mixed, c.xsm }) |v| a.add(v);
        for ([_]usize{ c.rows, dims, nc }) |v| a.add(@as(c_int, @intCast(v)));
        a.add(eps);
        const cdiv = (c.rows + 16 * mt - 1) / (16 * mt);
        try cuda.launch.launch(c.t.hcn_updbg[vi], .{ .grid = .{ .x = @intCast(cdiv * (dims / 32)) }, .block = .{ .x = 256 }, .shared = tk.hcnUpSmem(mt) }, c.r.s, &a);
    }
    fn peak(c: *Chain, ctas: usize) !void {
        var a: cuda.Args = .{};
        a.add(c.part);
        a.add(@as(c_int, 4000));
        try cuda.launch.launch(c.t.hcn_peak, .{ .grid = .{ .x = @intCast(ctas) }, .block = .{ .x = 256 } }, c.r.s, &a);
    }
    fn newUp(c: *Chain, vi: usize) !void {
        const mt = tk.hcn_up_mt[vi];
        var a: cuda.Args = .{};
        for ([_]u64{ c.act, c.wup.codes, c.h, c.pss, c.scale, c.mixed, c.xsm }) |v| a.add(v);
        for ([_]usize{ c.rows, dims, nc }) |v| a.add(@as(c_int, @intCast(v)));
        a.add(eps);
        const cdiv = (c.rows + 16 * mt - 1) / (16 * mt);
        try cuda.launch.launch(c.t.hcn_up[vi], .{ .grid = .{ .x = @intCast(cdiv * (dims / 32)) }, .block = .{ .x = 256 }, .shared = tk.hcnUpSmem(mt) }, c.r.s, &a);
    }
};

fn timeOf(c: *Chain, start: cuda.Event, end: cuda.Event, comptime f: anytype, arg: anytype) !f64 {
    const reps = 20;
    try @call(.auto, f, .{ c, arg });
    try start.record(c.r.s);
    for (0..reps) |_| try @call(.auto, f, .{ c, arg });
    try end.record(c.r.s);
    try c.r.s.synchronize();
    return @as(f64, try start.elapsedMs(end)) * 1000 / reps;
}

fn timeNone(c: *Chain, start: cuda.Event, end: cuda.Event, comptime f: anytype) !f64 {
    const Wrap = struct {
        fn call(cc: *Chain, _: void) !void {
            try f(cc);
        }
    };
    return timeOf(c, start, end, Wrap.call, {});
}

fn back(r: *Rig, comptime T: type, ptr: u64, n: usize) ![]T {
    return r.back(T, ptr, n);
}

/// Mode 3 (the shared expert's output read in place) against mode 2 over a y whose last slot holds a copy of it.
fn sharedDirect(r: *Rig, g: *const flash.Glue) !void {
    const slots = 11;
    for ([_]usize{ 1, 65, 1000 }) |m| {
        const h = r.bfs(m * wide, 2);
        const inj = r.bfs(m * streams, 2);
        const y = r.f32s(m * slots * dims, -1, 1);
        const shared = r.f32s(m * dims, -1, 1);
        const wts = r.f32s(m * slots, 0, 0.3);
        const y2 = try r.a.dupe(f32, y);
        for (0..m) |row| @memcpy(y2[(row * slots + slots - 1) * dims ..][0..dims], shared[row * dims ..][0..dims]);
        const dh = try r.dev(std.mem.sliceAsBytes(h));
        const dinj = try r.dev(std.mem.sliceAsBytes(inj));
        const dy = try r.dev(std.mem.sliceAsBytes(y));
        const dy2 = try r.dev(std.mem.sliceAsBytes(y2));
        const dsh = try r.dev(std.mem.sliceAsBytes(shared));
        const dw = try r.dev(std.mem.sliceAsBytes(wts));
        const o2 = try r.out(m * wide * 2);
        const o3 = try r.out(m * wide * 2);
        const p2 = try r.out(m * nc * streams * 4);
        const p3 = try r.out(m * nc * streams * 4);
        try g.hcWriteback(r.s, dh, o2, p2, 0, dinj, dy2, dw, m * dims, m, dims, streams, 2, 10, 11, 1);
        try g.hcWriteback(r.s, dh, o3, p3, dsh, dinj, dy, dw, m * dims, m, dims, streams, 3, 10, 11, 1);
        try check.sameBytes("writeback shared in place: streams", std.mem.sliceAsBytes(try r.back(u16, o3, m * wide)), std.mem.sliceAsBytes(try r.back(u16, o2, m * wide)));
        try check.sameBytes("writeback shared in place: sums", std.mem.sliceAsBytes(try r.back(f32, p3, m * nc * streams)), std.mem.sliceAsBytes(try r.back(f32, p2, m * nc * streams)));
        check.pass("hc-fuse: write-back mode 3 (shared expert in place) = mode 2 over a scattered y, {d} rows", .{m});
    }
}

pub fn run(gpu: check.Gpu) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    var prng = std.Random.DefaultPrng.init(0x4c2);
    var r = try Rig.init(gpu, arena.allocator(), prng.random());
    defer r.deinit();
    var g = try flash.Glue.load(gpu.d);
    defer g.deinit();
    var lane = try cuda.qmmf.Lane.load(gpu.d, (try gpu.ctx.capability()) / 10);
    defer lane.unload();
    var t = try tk.TileKernels.load(gpu.d);
    defer t.unload();

    try sharedDirect(&r, &g);
    const most = 3072;
    // bf16 weights in the lane's order, rows padded to 128 with zeros
    const wd = r.bfs(n_down * wide, 0.04);
    const wu = r.bfs(wide * low, 0.4);
    const pd = try r.a.alloc(u16, cuda.nvfp4.padded(n_down) * wide);
    cuda.qmmf.packBf16(pd, wd, n_down, wide);
    const pu = try r.a.alloc(u16, cuda.nvfp4.padded(wide) * low);
    cuda.qmmf.packBf16(pu, wu, wide, low);
    const wdown = cuda.modelopt.bf16Weight(try r.dev(std.mem.sliceAsBytes(pd)), n_down, wide);
    const wup = cuda.modelopt.bf16Weight(try r.dev(std.mem.sliceAsBytes(pu)), wide, low);
    const h = r.bfs(most * wide, 3);
    const scale = r.f32s(wide, 0.5, 1.5);
    const dh = try r.dev(std.mem.sliceAsBytes(h));
    const dscale = try r.dev(std.mem.sliceAsBytes(scale));
    const dpss = try r.out(most * nc * streams * 4);
    var c: Chain = .{
        .r = &r,
        .g = &g,
        .lane = &lane,
        .t = &t,
        .wdown = wdown,
        .wup = wup,
        .h = dh,
        .pss = dpss,
        .scale = dscale,
        .norm = try r.out(most * wide * 2),
        .part = try r.out(most * n_down * 4),
        .act = try r.out(most * low * 2),
        .xs = try r.out(most * (low / 32) * 4),
        .inj = try r.out(most * streams * 2),
        .up = try r.out(most * wide * 2),
        .mixed = try r.out(most * dims * 2),
        .xsm = try r.out(most * (dims / 32) * 4),
        .spart = try r.out(8 * most * n_down * 4),
        .rows = most,
    };
    try g.hcWriteback(r.s, dh, 0, dpss, 0, 0, 0, 0, most * dims, most, dims, streams, 0, 1, 1, 1);

    if (std.c.getenv("TF_HCFUSE_PROF") != null) {
        // one launch of each fused kernel at a full chunk, for ncu
        c.rows = most;
        try c.newDown(2);
        try c.reduce();
        try c.newUp(1);
        try c.baseNorm();
        try c.baseDown();
        try c.baseUp();
        try c.baseMix();
        try r.s.synchronize();
        return;
    }
    const counts = [_]usize{ 64, 65, 193, 255, 256, 1000, 1024, 1536, 2053, 3072 };
    for (counts) |m| {
        c.rows = m;
        // the chain as the engine runs it today
        try c.baseNorm();
        try c.baseDown();
        try c.reduce();
        try c.baseUp();
        try c.baseMix();
        const part0 = try back(&r, f32, c.part, m * n_down);
        const act0 = try back(&r, u16, c.act, m * low);
        const xs0 = try back(&r, f32, c.xs, m * (low / 32));
        const inj0 = try back(&r, u16, c.inj, m * streams);
        const mixed0 = try back(&r, u16, c.mixed, m * dims);
        const xsm0 = try back(&r, f32, c.xsm, m * (dims / 32));
        for (0..tk.hcn_dv2_mt.len - 1) |vi| {
            try c.part_poison();
            try c.newDown2(vi);
            const part2 = try back(&r, f32, c.part, m * n_down);
            try check.sameBytes("down v2", std.mem.sliceAsBytes(part2), std.mem.sliceAsBytes(part0));
        }
        for (0..tk.hcn_down_mt.len) |di| {
            for (0..tk.hcn_up_mt.len) |ui| {
                try c.part_poison();
                try c.newDown(di);
                const part1 = try back(&r, f32, c.part, m * n_down);
                try check.sameBytes("down", std.mem.sliceAsBytes(part1), std.mem.sliceAsBytes(part0));
                try c.reduce();
                try c.newUp(ui);
                const act1 = try back(&r, u16, c.act, m * low);
                const xs1 = try back(&r, f32, c.xs, m * (low / 32));
                const inj1 = try back(&r, u16, c.inj, m * streams);
                const mixed1 = try back(&r, u16, c.mixed, m * dims);
                const xsm1 = try back(&r, f32, c.xsm, m * (dims / 32));
                try check.sameBytes("act", std.mem.sliceAsBytes(act1), std.mem.sliceAsBytes(act0));
                try check.sameBytes("act sums", std.mem.sliceAsBytes(xs1), std.mem.sliceAsBytes(xs0));
                try check.sameBytes("inject", std.mem.sliceAsBytes(inj1), std.mem.sliceAsBytes(inj0));
                try check.sameBytes("mixed", std.mem.sliceAsBytes(mixed1), std.mem.sliceAsBytes(mixed0));
                try check.sameBytes("mixed sums", std.mem.sliceAsBytes(xsm1), std.mem.sliceAsBytes(xsm0));
            }
        }
        check.pass("hc-fuse: {d} rows: down tiles {d}, up tiles {d}: every output byte equals the five-launch chain's", .{ m, tk.hcn_down_mt.len, tk.hcn_up_mt.len });
    }

    var start = try cuda.Event.init(gpu.d, true);
    defer start.deinit();
    var end = try cuda.Event.init(gpu.d, true);
    defer end.deinit();
    for ([_]usize{ 3072, 2560, 2048, 1536, 1024, 768, 512, 256 }) |m| {
        c.rows = m;
        const flop = 2.0 * @as(f64, @floatFromInt(m)) * wide * n_down;
        const nm = try timeNone(&c, start, end, Chain.baseNorm);
        const bd = try timeNone(&c, start, end, Chain.baseDown);
        const ra = try timeNone(&c, start, end, Chain.reduce);
        const bu = try timeNone(&c, start, end, Chain.baseUp);
        const bm = try timeNone(&c, start, end, Chain.baseMix);
        std.debug.print("RESULT hc-fuse rows {d}: base normalize {d:.1} us, down {d:.1} us ({d:.1} TF), reduce_act {d:.1} us, up {d:.1} us ({d:.1} TF), mix {d:.1} us; chain {d:.1} us\n", .{ m, nm, bd, flop / bd / 1e6, ra, bu, flop / bu / 1e6, bm, nm + bd + ra + bu + bm });
        var best_d: f64 = 1e18;
        var best_u: f64 = 1e18;
        for (0..tk.hcn_down_mt.len) |di| {
            const us = try timeOf(&c, start, end, Chain.newDown, di);
            best_d = @min(best_d, us);
            std.debug.print("RESULT hc-fuse rows {d}: new down mt={d}: {d:.1} us ({d:.1} TF)\n", .{ m, tk.hcn_down_mt[di], us, flop / us / 1e6 });
        }
        for (0..tk.hcn_dv2_mt.len) |vi| {
            const us = try timeOf(&c, start, end, Chain.newDown2, vi);
            std.debug.print("RESULT hc-fuse rows {d}: down v2 {s}: {d:.1} us ({d:.1} TF)\n", .{ m, tk.hcn_dv2_names[vi], us, flop / us / 1e6 });
        }
        for (0..tk.hcn_up_mt.len) |ui| {
            const us = try timeOf(&c, start, end, Chain.newUp, ui);
            best_u = @min(best_u, us);
            std.debug.print("RESULT hc-fuse rows {d}: new up+mix mt={d}: {d:.1} us ({d:.1} TF)\n", .{ m, tk.hcn_up_mt[ui], us, flop / us / 1e6 });
        }
        if (m == 3072) for (0..tk.hcn_updbg_names.len) |vi| {
            const us = try timeOf(&c, start, end, Chain.upDbg, vi);
            std.debug.print("RESULT hc-fuse rows {d}: up+mix mt=3 debug {s}: {d:.1} us\n", .{ m, tk.hcn_updbg_names[vi], us });
        };
        std.debug.print("RESULT hc-fuse rows {d}: chain base {d:.1} us, new best {d:.1} us\n", .{ m, nm + bd + ra + bu + bm, best_d + ra + best_u });
    }
    for ([_]usize{ 48, 96, 144 }) |ctas| {
        const us = try timeOf(&c, start, end, Chain.peak, ctas);
        const flops = @as(f64, @floatFromInt(ctas)) * 8 * 4000 * 12 * 4096;
        std.debug.print("RESULT hc-fuse mma ceiling: {d} CTAs x 8 warps: {d:.1} us = {d:.1} TF\n", .{ ctas, us, flops / us / 1e6 });
    }
    r.report("hc-fuse");
}
