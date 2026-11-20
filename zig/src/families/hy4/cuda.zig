//! Hy4-flash on CUDA, the loader half: the pack's planned GPU tensors (load_plan) uploaded into one device arena at
//! device_layout's offsets, EXL3 groups handed to the existing trellis kernels as cuda.exl3.Weight (stored layout,
//! mul1 codebook: what families/glm/cuda_rexl3.zig gives GLM-5.3's REXL3 packs), plain tensors as stored. The
//! Engram tables stay in the sidecar on NVMe (load_plan.engramTables). No forward yet: `load` is what a first GPU
//! window runs to prove the weights land and fit, and its host tests run without a GPU.

const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
pub const config = @import("config.zig");
pub const load_plan = @import("load_plan.zig");
pub const device_layout = @import("device_layout.zig");
pub const pack_read = @import("pack_read.zig");

/// The trellis kernels' descriptor of slot `s`'s group in an arena at `base`.
pub fn exl3Weight(base: u64, s: device_layout.Slot) !cuda.exl3.Weight {
    const x = s.exl3 orelse return error.NotExl3;
    const st = cuda.exl3.strides(x.k, x.n, x.halves, false);
    return .{ .trellis = base + s.parts[0].offset, .suh = base + s.parts[1].offset, .svh = base + s.parts[2].offset, .k = x.k, .n = x.n, .halves = x.halves, .codebook = .mul1, .stride_k = st.k, .stride_nb = st.nb };
}

pub const Device = struct {
    buf: cuda.DeviceBuffer,
    layout: device_layout.Layout,

    pub fn ptr(d: *const Device, slot: u32) u64 {
        return d.buf.ptr + d.layout.slots[slot].offset();
    }

    pub fn weight(d: *const Device, slot: u32) !cuda.exl3.Weight {
        return exl3Weight(d.buf.ptr, d.layout.slots[slot]);
    }

    pub fn deinit(d: *Device) void {
        d.buf.free();
    }
};

pub const Progress = *const fn (done: u64, total: u64) void;

/// Every byte the arena holds after the uploads (tensors copied verbatim).
pub fn uploaded(l: *const device_layout.Layout) u64 {
    var n: u64 = 0;
    for (l.slots) |s| for (s.parts) |p| {
        n += p.bytes;
    };
    return n;
}

/// The layout's tensors from `ck` (the pack opened with core.Checkpoint) into one device allocation. Pages are
/// dropped from the page cache every `release_every` bytes, since on a unified-memory Spark the cache and the arena
/// share the same RAM. Refuses when the arena and `reserve` do not fit the device's free memory.
pub fn load(d: *const cuda.Driver, ck: *core.Checkpoint, plan: *const load_plan.Plan, layout: device_layout.Layout, reserve: u64, release_every: u64, progress: ?Progress) !Device {
    var free: usize = 0;
    var total: usize = 0;
    try d.check(d.api.cuMemGetInfo_v2(&free, &total), "cuMemGetInfo");
    if (!device_layout.fits(&layout, free, reserve)) {
        std.log.err("hy4: the weights need {d} MiB and {d} MiB more are kept free, the device has {d} MiB free", .{ layout.total >> 20, reserve >> 20, free >> 20 });
        return error.OutOfDeviceMemory;
    }
    var buf = try cuda.DeviceBuffer.alloc(d, layout.total);
    errdefer buf.free();
    const want = uploaded(&layout);
    var done: u64 = 0;
    var since: u64 = 0;
    var name_buf: [320]u8 = undefined;
    for (layout.slots) |s| {
        const e = plan.entries[s.entry];
        if (s.exl3) |_| {
            for ([_][]const u8{ "trellis", "suh", "svh" }, s.parts) |part, p| {
                const t = try ck.get(try std.fmt.bufPrint(&name_buf, "{s}.{s}", .{ e.name, part }));
                if (t.bytes.len != p.bytes) return error.UnexpectedTensor;
                try buf.upload(p.offset, t.bytes);
                done += p.bytes;
                since += p.bytes;
            }
            try ck.skip(try std.fmt.bufPrint(&name_buf, "{s}.mul1", .{e.name}));
        } else {
            const t = try ck.get(e.name);
            if (t.bytes.len != s.parts[0].bytes) return error.UnexpectedTensor;
            try buf.upload(s.parts[0].offset, t.bytes);
            done += t.bytes.len;
            since += t.bytes.len;
        }
        if (since >= release_every) {
            ck.release();
            since = 0;
            if (progress) |f| f(done, want);
        }
    }
    ck.release();
    if (progress) |f| f(done, want);
    return .{ .buf = buf, .layout = layout };
}

test "Hy4 CUDA: an EXL3 slot becomes the trellis kernels' stored-layout Weight at its arena offsets" {
    const s: device_layout.Slot = .{ .entry = 0, .parts = .{ .{ .offset = 0, .bytes = 3072 * 8192 * 12 / 16 }, .{ .offset = 18874368, .bytes = 6144 }, .{ .offset = 18880512, .bytes = 16384 } }, .exl3 = .{ .k = 3072, .n = 8192, .halves = 12 }, .dtype = null };
    const w = try exl3Weight(1 << 40, s);
    try std.testing.expectEqual(@as(u64, 1 << 40), w.trellis);
    try std.testing.expectEqual(@as(u64, (1 << 40) + 18874368), w.suh);
    try std.testing.expectEqual(cuda.exl3.Codebook.mul1, w.codebook);
    const st = device_layout.storedStrides(s.exl3.?);
    try std.testing.expectEqual(st.k, w.stride_k);
    try std.testing.expectEqual(st.nb, w.stride_nb);
    // a 128-column view keeps the stored strides
    const v = try w.columns(128, 256);
    try std.testing.expectEqual(w.trellis + w.stride_nb * 4, v.trellis);
    var plain = s;
    plain.exl3 = null;
    try std.testing.expectError(error.NotExl3, exl3Weight(0, plain));
}

test "Hy4 CUDA: the small pack's layout uploads every planned GPU byte but the mul1 markers" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var c = try config.Config.parse(a, @embedFile("fixtures/config.json"));
    c.text.layers = 3;
    c.text.experts = 8;
    c.text.engram_len = 1;
    var h = try load_plan.JsonHeaders.parse(a, @embedFile("fixtures/pack_headers.json"));
    var problem: load_plan.Problem = .{};
    const p = try load_plan.build(a, &c, h.source(), null, .{}, &problem);
    const l = try device_layout.build(a, &c, &p);
    try std.testing.expectEqual(p.gpu_bytes - 4 * p.exl3_groups, uploaded(&l));
    for (l.layers[1..]) |y| _ = try exl3Weight(0, l.slots[y.experts]);
}

test "Hy4 CUDA: load compiles against the runtime and checkpoint it calls" {
    _ = &load;
    _ = &Device.weight;
    _ = &Device.ptr;
}

test {
    // the binder, the GPU model and the vision tower compile against this runtime; the e4m3 table; the grouped MoE
    // tables, the parallel Engram reads and the capacity table
    _ = @import("bind.zig");
    _ = @import("moe_grouped.zig");
    _ = @import("engram_batch.zig");
    _ = @import("capacity.zig");
    _ = @import("prof.zig");
    _ = @import("draft_plan.zig");
    std.testing.refAllDecls(@import("gpu_model.zig"));
}
