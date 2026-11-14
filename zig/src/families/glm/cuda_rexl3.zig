//! A REXL3/SAGE pack's EXL3 projections for GLM-5.3-Flash (vcruz305's mixed-width `mul1` packs): every linear but the
//! routed experts' may be a trellis group too, each at its own width. This file reads such a group as the common
//! dense kernels take it (cuda.exl3.Weight, the stored [K/16, N/16, 16 * bits] layout), and dequantizes one to bf16
//! rows on the host where a consumer keeps bf16 (the MTP head's eh_proj, the drafts' 4-bit head).
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const projection = @import("cuda_proj.zig");
const fmt = core.exl3_format;
const Tensor = core.checkpoint.Tensor;

/// Whether `module` (a full tensor-name prefix) is stored as an EXL3 group: its `.trellis` tensor exists.
pub fn has(ck: *const core.Checkpoint, module: []const u8) bool {
    var buf: [256]u8 = undefined;
    const name = std.fmt.bufPrint(&buf, "{s}.trellis", .{module}) catch return false;
    return ck.has(name);
}

/// One group's shape and width as its tensors state them.
pub const Shape = struct { k: usize, n: usize, halves: u16, codebook: cuda.exl3.Codebook };

/// The group's shape from its trellis int16 [K/16, N/16, 16 * bits] and scales fp16 suh [K], svh [N]; the codebook
/// from its marker (`mcg`, `mul1`, none: 3inst). Refuses what the kernels do not read (widths 1..8, K and N % 128).
pub fn shapeOf(trellis: Tensor, suh: Tensor, svh: Tensor, codebook: cuda.exl3.Codebook) !Shape {
    if (trellis.dtype != .i16 or trellis.rank != 3) return error.NotExl3;
    const last = trellis.dim(2);
    if (last % 8 != 0) return error.NotExl3;
    const halves = last / 8; // 16 * bits words = 8 * halves
    if (halves < 2 or halves > 16) return error.UnsupportedExl3Width;
    if (halves % 2 == 1 and codebook != .mul1) return error.UnsupportedExl3Width; // half steps are mul1's alone
    const k = trellis.dim(0) * 16;
    const n = trellis.dim(1) * 16;
    if (k % 128 != 0 or n % 128 != 0) return error.Exl3Shape;
    if (suh.dtype != .f16 or suh.numel() != k) return error.NotExl3; // packed sign words (su/sv) are not read here
    if (svh.dtype != .f16 or svh.numel() != n) return error.NotExl3;
    return .{ .k = k, .n = n, .halves = @intCast(halves), .codebook = codebook };
}

/// `module`'s codebook from its marker tensor; error.NotExl3 when a marker's value is not its magic.
pub fn codebookOf(ck: *core.Checkpoint, module: []const u8) !cuda.exl3.Codebook {
    var buf: [256]u8 = undefined;
    for (core.exl3_load.markers) |m| {
        const name = try std.fmt.bufPrint(&buf, "{s}.{s}", .{ module, m.name });
        if (!ck.has(name)) continue;
        const t = try ck.get(name);
        if (t.dtype != .i32 or t.numel() != 1 or t.bytes.len != 4) return error.NotExl3;
        if (std.mem.readInt(u32, t.bytes[0..4], .little) != m.value) return error.NotExl3;
        return switch (m.codebook) {
            .mcg => .mcg,
            .mul1 => .mul1,
            .inst3 => .inst3,
        };
    }
    return .inst3;
}

/// A group's three tensors.
pub const Group = struct { trellis: Tensor, suh: Tensor, svh: Tensor, shape: Shape };

pub fn read(ck: *core.Checkpoint, module: []const u8) !Group {
    var buf: [256]u8 = undefined;
    const cb = try codebookOf(ck, module);
    const t = try ck.get(try std.fmt.bufPrint(&buf, "{s}.trellis", .{module}));
    const su = try ck.get(try std.fmt.bufPrint(&buf, "{s}.suh", .{module}));
    const sv = try ck.get(try std.fmt.bufPrint(&buf, "{s}.svh", .{module}));
    return .{ .trellis = t, .suh = su, .svh = sv, .shape = try shapeOf(t, su, sv, cb) };
}

/// `module` uploaded through the loader as the dense kernels' Weight (named `name` with `args`, then `j`).
pub fn upload(L: anytype, ck: *core.Checkpoint, module: []const u8, comptime name: []const u8, args: anytype, j: usize) !cuda.exl3.Weight {
    const g = try read(ck, module);
    const at = args ++ .{j};
    try L.host.appendSlice(L.gpa, g.trellis.bytes);
    const trellis = try L.put(name ++ ".x3.{d}.trellis", at);
    try L.host.appendSlice(L.gpa, g.suh.bytes);
    const suh = try L.put(name ++ ".x3.{d}.suh", at);
    try L.host.appendSlice(L.gpa, g.svh.bytes);
    const svh = try L.put(name ++ ".x3.{d}.svh", at);
    return weightOf(g.shape, trellis, suh, svh);
}

pub fn weightOf(s: Shape, trellis: u64, suh: u64, svh: u64) cuda.exl3.Weight {
    const st = cuda.exl3.strides(s.k, s.n, s.halves, false);
    return .{ .trellis = trellis, .suh = suh, .svh = svh, .k = @intCast(s.k), .n = @intCast(s.n), .halves = s.halves, .codebook = s.codebook, .stride_k = st.k, .stride_nb = st.nb };
}

/// Part = cuda_formats.Part: a rank's rows (outputs) or inputs of a module; EXL3 takes whole 128-output blocks of the
/// outputs (a column view of the stored matrix) and every input (a K cut would need its own rotation blocks' scales).
pub fn proj(L: anytype, ck: *core.Checkpoint, mods: []const []const u8, parts: anytype, comptime name: []const u8, args: anytype) !projection.Proj {
    var e: projection.Exl3 = .{};
    if (mods.len == 0 or mods.len > e.ws.len) return error.UnreadableFormat;
    for (mods, parts, 0..) |m, q, j| {
        var w = try upload(L, ck, m, name, args, j);
        if (j > 0 and w.k != e.ws[0].k) {
            std.log.err("{s}: {d} inputs, stacked with a part of {d}", .{ m, w.k, e.ws[0].k });
            return error.UnreadableFormat;
        }
        if (q.cols) |c| if (c[0] != 0 or c[1] != w.k) {
            std.log.err("{s}: an input cut [{d}, {d}) of an EXL3 group (this loader serves REXL3 packs whole: TP=1)", .{ m, c[0], c[1] });
            return error.Exl3RankCut;
        };
        if (q.rows) |r| if (r[0] != 0 or r[1] != w.n) {
            w = w.columns(@intCast(r[0]), @intCast(r[1] - r[0])) catch {
                std.log.err("{s}: outputs [{d}, {d}) are not whole 128-column blocks", .{ m, r[0], r[1] });
                return error.Exl3RankCut;
            };
        };
        e.ws[j] = w;
        e.len += 1;
        e.n += w.n;
    }
    return .{ .exl3 = e };
}

/// In-place fast Walsh-Hadamard transform of a 128-value block scaled by 1/sqrt(128): exl3_format.rotate's H.
fn fwht(b: *[128]f64) void {
    var h: usize = 1;
    while (h < 128) : (h *= 2) {
        var i: usize = 0;
        while (i < 128) : (i += 2 * h) {
            for (i..i + h) |j| {
                const a = b[j];
                const c = b[j + h];
                b[j] = a + c;
                b[j + h] = a - c;
            }
        }
    }
    const inv = 1.0 / @sqrt(128.0);
    for (b) |*v| v.* *= inv;
}

/// Columns [c0, c0 + 128) of W [K, N] = diag(suh) @ H_K @ W_q @ H_N @ diag(svh), written transposed as bf16 rows
/// out[(c - c0) * K + r] (an nn.Linear's [out][in] rows); `wq` scratch holds K * 128 f64.
fn strip(g: Group, c0: usize, out: []u16, wq: []f64) void {
    const s = g.shape;
    const k = s.k;
    const tn_all = s.n / 16;
    const width: fmt.Width = .{ .halves = s.halves };
    const words = width.tileWords();
    const book: fmt.Codebook = switch (s.codebook) {
        .inst3 => .inst3,
        .mcg => .mcg,
        .mul1 => .mul1,
    };
    var placement: [256]fmt.Placement = undefined;
    fmt.tilePlacement(&placement);
    const trellis = std.mem.bytesAsSlice(i16, @as([]align(1) const u8, g.trellis.bytes));
    // W_q's tiles of the strip: rows 0..K, columns c0..c0 + 128 (eight 16-column tiles)
    for (0..k / 16) |tk| for (0..8) |t| {
        const tn = c0 / 16 + t;
        var tile: [16 * 16]i16 = undefined;
        for (0..words) |wi| tile[wi] = trellis[(tk * tn_all + tn) * words + wi];
        var states: [256]u32 = undefined;
        fmt.tileStates(width, tile[0..words], &states);
        for (0..256) |p| {
            const v: f64 = @floatCast(fmt.codebookValue(book, @intCast(states[p])));
            wq[(tk * 16 + placement[p].row) * 128 + t * 16 + placement[p].col] = v;
        }
    };
    const suh = std.mem.bytesAsSlice(f16, @as([]align(1) const u8, g.suh.bytes));
    const svh = std.mem.bytesAsSlice(f16, @as([]align(1) const u8, g.svh.bytes));
    // H_K down each column (blocks of 128 rows), then suh by rows
    var col: [128]f64 = undefined;
    for (0..128) |c| {
        var b: usize = 0;
        while (b < k) : (b += 128) {
            for (0..128) |i| col[i] = wq[(b + i) * 128 + c];
            fwht(&col);
            for (0..128) |i| wq[(b + i) * 128 + c] = col[i] * @as(f64, @floatCast(suh[b + i]));
        }
    }
    // H_N along each row (the strip is one 128-column block), then svh by columns; out transposed, bf16
    var row: [128]f64 = undefined;
    for (0..k) |r| {
        @memcpy(&row, wq[r * 128 ..][0..128]);
        fwht(&row);
        for (0..128) |c| {
            const v: f32 = @floatCast(row[c] * @as(f64, @floatCast(svh[c0 + c])));
            out[c * k + r] = core.bf16_round.bits(v);
        }
    }
}

/// The group dequantized to bf16 rows [N][K] (the nn.Linear layout bf16 consumers read), outputs [lo, hi) only,
/// on `threads` host threads; lo and hi whole 128-column blocks.
pub fn dequantRows(gpa: std.mem.Allocator, g: Group, lo: usize, hi: usize, out: []u16, threads: usize) !void {
    const k = g.shape.k;
    if (lo % 128 != 0 or hi % 128 != 0 or hi > g.shape.n or lo >= hi) return error.Exl3Shape;
    if (out.len != (hi - lo) * k) return error.Exl3Shape;
    const Job = struct {
        fn run(gg: Group, l: usize, h: usize, o: []u16, next: *std.atomic.Value(usize), kk: usize, wq: []f64) void {
            while (true) {
                const c0 = l + 128 * next.fetchAdd(1, .monotonic);
                if (c0 >= h) return;
                strip(gg, c0, o[(c0 - l) * kk ..][0 .. 128 * kk], wq);
            }
        }
    };
    var next = std.atomic.Value(usize).init(0);
    const n_threads = @max(1, @min(threads, (hi - lo) / 128));
    const scratch = try gpa.alloc(f64, n_threads * k * 128);
    defer gpa.free(scratch);
    const pool = try gpa.alloc(std.Thread, n_threads - 1);
    defer gpa.free(pool);
    var started: usize = 0;
    defer for (pool[0..started]) |t| t.join();
    for (pool, 0..) |*t, i| {
        t.* = try std.Thread.spawn(.{}, Job.run, .{ g, lo, hi, out, &next, k, scratch[(i + 1) * k * 128 ..][0 .. k * 128] });
        started += 1;
    }
    Job.run(g, lo, hi, out, &next, k, scratch[0 .. k * 128]);
}

fn testTensor(dtype: core.checkpoint.DType, shape: []const usize, bytes: []const u8) Tensor {
    var t: Tensor = .{ .dtype = dtype, .rank = @intCast(shape.len), .shape = undefined, .bytes = bytes };
    @memcpy(t.shape[0..shape.len], shape);
    return t;
}

test "GLM REXL3: a group's shape and width come from its tensors; widths and codebooks the kernels lack are refused" {
    const none: []const u8 = &.{};
    // GLM-5.3-Flash SAGE's routed down at 3 bits, a KDA qkv at 8, a lm_head at 6
    for ([_][4]usize{ .{ 128, 256, 48, 6 }, .{ 256, 1536, 128, 16 }, .{ 256, 9680, 96, 12 } }) |c| {
        const t = testTensor(.i16, &.{ c[0], c[1], c[2] }, none);
        const s = try shapeOf(t, testTensor(.f16, &.{c[0] * 16}, none), testTensor(.f16, &.{c[1] * 16}, none), .mul1);
        try std.testing.expectEqual(@as(u16, @intCast(c[3])), s.halves);
        try std.testing.expectEqual(c[0] * 16, s.k);
        try std.testing.expectEqual(c[1] * 16, s.n);
    }
    const odd = testTensor(.i16, &.{ 8, 8, 40 }, none); // 2.5 bits: mul1's half step only
    try std.testing.expectEqual(@as(u16, 5), (try shapeOf(odd, testTensor(.f16, &.{128}, none), testTensor(.f16, &.{128}, none), .mul1)).halves);
    try std.testing.expectError(error.UnsupportedExl3Width, shapeOf(odd, testTensor(.f16, &.{128}, none), testTensor(.f16, &.{128}, none), .mcg));
    try std.testing.expectError(error.NotExl3, shapeOf(odd, testTensor(.bf16, &.{128}, none), testTensor(.f16, &.{128}, none), .mul1));
    try std.testing.expectError(error.Exl3Shape, shapeOf(testTensor(.i16, &.{ 4, 8, 32 }, none), testTensor(.f16, &.{64}, none), testTensor(.f16, &.{128}, none), .mul1));
}

test "GLM REXL3: the fast Hadamard equals exl3_format.rotate's" {
    var prng = std.Random.DefaultPrng.init(7);
    var a: [128]f64 = undefined;
    for (&a) |*v| v.* = prng.random().floatNorm(f64);
    var ref: [128]f64 = a;
    fmt.rotate(&ref, 1, 128, 1);
    fwht(&a);
    for (a, ref) |x, y| try std.testing.expectApproxEqAbs(y, x, 1e-12);
}

test "GLM REXL3: strip dequantization equals exl3_format.dequantize, transposed to bf16 rows" {
    const gpa = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(11);
    const r = prng.random();
    for ([_]u16{ 6, 5, 12 }) |halves| {
        const k: usize = 256;
        const n: usize = 256;
        const words = @as(usize, halves) * 8;
        const trellis = try gpa.alloc(i16, (k / 16) * (n / 16) * words);
        defer gpa.free(trellis);
        for (trellis) |*w| w.* = r.int(i16);
        const suh = try gpa.alloc(f16, k);
        defer gpa.free(suh);
        const svh = try gpa.alloc(f16, n);
        defer gpa.free(svh);
        for (suh) |*v| v.* = @floatCast(0.5 + r.float(f32));
        for (svh) |*v| v.* = @floatCast(0.001 + 0.01 * r.float(f32));
        const ref = try gpa.alloc(f64, k * n);
        defer gpa.free(ref);
        const sc = try gpa.alloc(f64, 2 * k * n);
        defer gpa.free(sc);
        fmt.dequantize(.{ .halves = halves }, .mul1, trellis, k / 16, n / 16, suh, svh, ref, sc);
        const g: Group = .{
            .trellis = testTensor(.i16, &.{ k / 16, n / 16, words }, std.mem.sliceAsBytes(trellis)),
            .suh = testTensor(.f16, &.{k}, std.mem.sliceAsBytes(suh)),
            .svh = testTensor(.f16, &.{n}, std.mem.sliceAsBytes(svh)),
            .shape = .{ .k = k, .n = n, .halves = halves, .codebook = .mul1 },
        };
        const out = try gpa.alloc(u16, (n - 128) * k);
        defer gpa.free(out);
        try dequantRows(gpa, g, 128, n, out, 3);
        for (128..n) |c| for (0..k) |row| {
            const want: i32 = core.bf16_round.bits(@floatCast(ref[row * n + c]));
            const got: i32 = out[(c - 128) * k + row];
            try std.testing.expect(@abs(want - got) <= 1); // the f64 sums' order differs from rotate's: one bf16 step
        };
    }
}

/// Every EXL3 projection of `w`, each stream's runner grown to it now (cuda_proj.Exl3Run.reserve), so no call inside
/// a decode graph's capture allocates. Returns how many projections are EXL3 (0: an NVFP4, FP8 or Mia pack).
pub fn reserve(w: *const @import("cuda_weights.zig").Weights, runs: []const *projection.Exl3Run, s: cuda.Stream) !usize {
    var count: usize = 0;
    const Walk = struct {
        fn one(rs: []const *projection.Exl3Run, st: cuda.Stream, p: projection.Proj, n: *usize) !void {
            if (p != .exl3) return;
            n.* += 1;
            for (rs) |r| try r.reserve(st, p);
        }
        fn layer(rs: []const *projection.Exl3Run, st: cuda.Stream, l: @import("cuda_weights.zig").Layer, n: *usize) !void {
            switch (l.mixer) {
                .kda => |k| {
                    switch (k.proj) {
                        .split => |sp| try one(rs, st, sp.qkv, n),
                        .bf16 => {},
                    }
                    try one(rs, st, k.o, n);
                },
                .mla => |a| for ([_]projection.Proj{ a.proj, a.q_b, a.o, a.index.kw, a.index.qb }) |p| try one(rs, st, p, n),
            }
            switch (l.ffn) {
                .dense => |d| switch (d.w) {
                    .exl3 => |x| {
                        try one(rs, st, x.gu, n);
                        try one(rs, st, x.down, n);
                    },
                    else => {},
                },
                .moe => |m| {
                    try one(rs, st, m.shared_gu, n);
                    try one(rs, st, m.shared_down, n);
                },
            }
        }
    };
    for (w.layers) |l| try Walk.layer(runs, s, l, &count);
    if (w.mtp) |m| try Walk.layer(runs, s, m.layer, &count);
    try Walk.one(runs, s, w.head, &count);
    try s.synchronize();
    return count;
}
