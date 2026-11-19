//! The first GPU window's load: the whole pack planned (load_plan), laid out (device_layout) and uploaded into one
//! device arena (cuda.load), then spot-checked by reading arena bytes back against the pack's files. On demand:
//! HY4_LOAD_PACK names the pack directory; HY4_LOAD_RESERVE_MIB (default 6144) is kept free beside the arena. Needs
//! the GPU to itself (~93 GiB of a Spark's unified memory): run it only in a window with the serving model stopped.

const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const hy4 = @import("cuda.zig");
const load_plan = hy4.load_plan;
const device_layout = hy4.device_layout;
const pack_read = hy4.pack_read;

fn progress(done: u64, total: u64) void {
    std.debug.print("hy4 load: {d} / {d} MiB\n", .{ done >> 20, total >> 20 });
}

fn mib(x: u64) u64 {
    return x >> 20;
}

test "Hy4 CUDA load: the whole pack lands in one arena and reads back as stored (HY4_LOAD_PACK, the GPU alone, on demand)" {
    const dir = std.testing.environ.getPosix("HY4_LOAD_PACK") orelse return error.SkipZigTest;
    const reserve_mib = if (std.testing.environ.getPosix("HY4_LOAD_RESERVE_MIB")) |r| try std.fmt.parseInt(u64, r, 10) else 6144;
    const gpa = std.heap.page_allocator;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;

    const cfg_bytes = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "config.json" }), a, .limited(1 << 24));
    const c = try hy4.config.Config.parse(a, cfg_bytes);
    var main_pack = try pack_read.Pack.open(a, io, dir, "model.safetensors.index.json");
    defer main_pack.close();
    var side_pack = try pack_read.Pack.open(a, io, dir, "engram.safetensors.index.json");
    defer side_pack.close();
    var problem: load_plan.Problem = .{};
    const plan = load_plan.build(a, &c, load_plan.Source.ofPack(&main_pack), load_plan.Source.ofPack(&side_pack), .{}, &problem) catch |e| {
        std.debug.print("hy4 plan refused: {any}\n", .{problem});
        return e;
    };
    const layout = try device_layout.build(a, &c, &plan);
    std.debug.print("hy4 plan: {d} EXL3 groups, GPU {d} MiB, NVMe (Engram tables) {d} MiB; arena {d} MiB in {d} slots over {d} layers\n", .{ plan.exl3_groups, mib(plan.gpu_bytes), mib(plan.nvme_bytes), mib(layout.total), layout.slots.len, layout.layers.len });

    var d = try cuda.Driver.open();
    defer d.close();
    var ctx = try cuda.Context.init(&d, 0);
    defer ctx.deinit();
    var free0: usize = 0;
    var total: usize = 0;
    try d.check(d.api.cuMemGetInfo_v2(&free0, &total), "cuMemGetInfo");
    std.debug.print("hy4 device: {d} MiB free of {d} MiB, reserve {d} MiB\n", .{ mib(free0), mib(total), reserve_mib });

    var ck = try core.Checkpoint.openModel(gpa, io, dir);
    defer ck.close();
    const t0 = std.Io.Clock.awake.now(io);
    var dev = try hy4.load(&d, &ck, &plan, layout, reserve_mib << 20, 2 << 30, progress);
    defer dev.deinit();
    const secs = @as(f64, @floatFromInt(t0.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds())) / 1e9;
    var free1: usize = 0;
    try d.check(d.api.cuMemGetInfo_v2(&free1, &total), "cuMemGetInfo");
    std.debug.print("hy4 loaded {d} MiB in {d:.1} s ({d:.2} GB/s); {d} MiB free after\n", .{ mib(hy4.uploaded(&layout)), secs, @as(f64, @floatFromInt(hy4.uploaded(&layout))) / secs / 1e9, mib(free1) });

    // spot checks: head and tail bytes of each part of a spread of slots, from the arena against the pack's files
    var picks: std.ArrayList(u32) = .empty;
    try picks.appendSlice(a, &.{ 0, layout.embed, layout.norm, layout.lm_head, @intCast(layout.slots.len - 1) });
    for (layout.layers, 0..) |y, i| if (i % 7 == 1) {
        try picks.append(a, y.q_a);
        try picks.append(a, y.experts);
        try picks.append(a, y.experts + 3 * (c.text.experts - 1) + 2);
    };
    var it = layout.names.iterator();
    while (it.next()) |kv| if (std.mem.startsWith(u8, kv.key_ptr.*, "vit.") and std.mem.indexOf(u8, kv.key_ptr.*, "layers.13.self_attn.o_proj") != null) try picks.append(a, kv.value_ptr.*);
    var checked: usize = 0;
    var name_buf: [320]u8 = undefined;
    for (picks.items) |slot| {
        const s = layout.slots[slot];
        const e = plan.entries[s.entry];
        const parts: []const []const u8 = if (s.exl3 != null) &.{ "trellis", "suh", "svh" } else &.{""};
        for (parts, 0..) |part, k| {
            const p = s.parts[k];
            const name = if (part.len == 0) e.name else try std.fmt.bufPrint(&name_buf, "{s}.{s}", .{ e.name, part });
            const n: usize = @intCast(@min(p.bytes, 4096));
            for ([_]u64{ 0, p.bytes - n }) |off| {
                const want = try main_pack.rawRange(a, name, off, n);
                const got = try a.alloc(u8, n);
                try dev.buf.download(@intCast(p.offset + off), got);
                if (!std.mem.eql(u8, want, got)) {
                    std.debug.print("hy4 spot check: {s} bytes {d}..{d} differ\n", .{ name, off, off + n });
                    return error.TestUnexpectedResult;
                }
                checked += 1;
            }
        }
        if (s.exl3) |x| {
            const w = try dev.weight(slot);
            if (w.k != x.k or w.n != x.n or w.trellis != dev.buf.ptr + s.parts[0].offset) return error.TestUnexpectedResult;
        }
    }
    std.debug.print("hy4 spot checks: {d} ranges over {d} slots read back as stored\n", .{ checked, picks.items.len });
}
