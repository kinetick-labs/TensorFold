//! GLM-5.3-Flash's EXL3 routed experts on CUDA (Mia-AiLab's checkpoint: 4-bit mcg trellis experts, the rest bf16).

const std = @import("std");
const modelopt = @import("core").modelopt;
const Span = @import("cuda_shard.zig").Span;
const cuda = @import("cuda");
const experts = @import("cuda_experts.zig");

/// config.json's quantization_config (top level, else text_config's), when it has one.
fn block(root: std.json.Value) ?std.json.ObjectMap {
    if (root != .object) return null;
    if (root.object.get("quantization_config")) |q| if (q == .object) return q.object;
    if (root.object.get("text_config")) |t| if (t == .object) if (t.object.get("quantization_config")) |q| if (q == .object) return q.object;
    return null;
}

/// Whether config.json names EXL3 (quant_method "exl3"): `admit` then says whether this family reads it.
pub fn isExl3(gpa: std.mem.Allocator, config_json: []const u8) bool {
    const parsed = std.json.parseFromSlice(std.json.Value, gpa, config_json, .{}) catch return false;
    defer parsed.deinit();
    const q = block(parsed.value) orelse return false;
    const m = q.get("quant_method") orelse return false;
    return m == .string and std.mem.eql(u8, m.string, "exl3");
}

/// The EXL3 checkpoints the kernels read: Mia-AiLab's 4-bit routed experts on the "mcg" codebook (everything else
/// bf16), or a per-tensor mixed pack (vcruz305's SAGE/REXL3: no average `bits`, or the "mul1" codebook), whose every
/// group states its own width and codebook and is checked as it loads.
pub fn admit(gpa: std.mem.Allocator, config_json: []const u8, why: *modelopt.Why) !void {
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, config_json, .{});
    defer parsed.deinit();
    const q = block(parsed.value) orelse {
        why.set("no quantization_config", .{});
        return error.UnsupportedQuant;
    };
    const bits = q.get("bits") orelse std.json.Value{ .null = {} };
    const book = q.get("codebook") orelse std.json.Value{ .null = {} };
    if (mixed(bits, book)) return;
    if (bits != .integer or bits.integer != 4) {
        why.set("EXL3 at {f} bits: the CUDA family reads 4-bit EXL3 experts", .{std.json.fmt(bits, .{})});
        return error.UnsupportedQuant;
    }
    if (book != .string or !std.mem.eql(u8, book.string, "mcg")) {
        why.set("EXL3 codebook {f}: the CUDA family reads \"mcg\"", .{std.json.fmt(book, .{})});
        return error.UnsupportedQuant;
    }
}

/// A per-tensor mixed pack: neither an average width nor a codebook stated, or the "mul1" codebook (any width).
fn mixed(bits: std.json.Value, book: std.json.Value) bool {
    if (bits == .null and book == .null) return true;
    return book == .string and std.mem.eql(u8, book.string, "mul1");
}

/// One layer's EXL3 experts on a rank: trellis pointers [E] per projection, fp16 scales (suh [E][K], svh [E][N]),
/// and each projection's widths int32 [E] in half-bits (a SAGE pack mixes 2, 3 and 4 bits within one layer).
pub const Experts = struct {
    gt: u64,
    ut: u64,
    dt: u64,
    suh_g: u64,
    suh_u: u64,
    svh_g: u64,
    svh_u: u64,
    suh_d: u64,
    svh_d: u64,
    k2: u64, // gate's widths (the field Mia's uniform 4-bit layers shared across gate, up and down)
    k2u: u64 = 0, // up's and down's widths; 0: k2's
    k2d: u64 = 0,
    shx: bool, // every expert's gate and up share one input rotation: a prompt rotates a row once, not once a pick
    codebook: cuda.exl3.Codebook = .mcg,
    uniform: u32 = 8, // the half-bit width every projection shares, 0 when they mix
    prompt_k2: bool = false, // mul1, gate = up per expert, every width 4, 6 or 8: the prompt kernels take the mix
};

/// Bytes of one 16 x 16 tile at `halves` half-bits a value: 256 values, 16 * bits int16 words.
fn tileBytes(halves: usize) usize {
    return 16 * halves;
}

/// Layer `i`'s EXL3 experts through `L` (cuda_weights' loader): gate, up by `cut`'s tile columns, down by its rows.
/// Every group's width is its trellis's (last dim / 16 bits); the codebook its marker's, one for the whole layer.
pub fn load(L: anytype, i: u32, count: usize, dims: usize, inter: usize, cut: Span) !Experts {
    if (dims % 128 != 0 or cut.lo % 128 != 0 or cut.hi % 128 != 0) return error.NotExl3Mcg4;
    const width = cut.len();
    var out: Experts = undefined;
    out.shx = true;
    out.uniform = 0;
    var book: ?cuda.exl3.Codebook = null;
    var first_halves: ?usize = null;
    var uniform = true;
    const Mat = struct { k: usize, n: usize, by_cols: bool };
    const mats = [_]Mat{
        .{ .k = dims, .n = inter, .by_cols = true },
        .{ .k = dims, .n = inter, .by_cols = true },
        .{ .k = inter, .n = dims, .by_cols = false },
    };
    const k2s = try L.gpa.alloc(i32, count);
    defer L.gpa.free(k2s);
    const gk2 = try L.gpa.alloc(i32, count); // gate's widths, to check up's against
    defer L.gpa.free(gk2);
    var prompt_widths = true;
    var first_suh: ?[]const u8 = null;
    inline for (mats, .{ "gate_proj", "up_proj", "down_proj" }, .{ "gt", "ut", "dt" }, .{ "suh_g", "suh_u", "suh_d" }, .{ "svh_g", "svh_u", "svh_d" }, .{ "k2", "k2u", "k2d" }) |m, proj, tw, su, sv, kw| {
        const kt = (if (m.by_cols) m.k else width) / 16; // the rank's tiles
        const nt = (if (m.by_cols) width else m.n) / 16;
        const offsets = try L.gpa.alloc(u64, count);
        defer L.gpa.free(offsets);
        var at: u64 = 0;
        for (0..count) |e| {
            const t = try L.get("layers.{d}.mlp.experts.{d}." ++ proj ++ ".trellis", .{ i, e });
            if (t.dtype != .i16 or t.rank != 3 or t.dim(0) != m.k / 16 or t.dim(1) != m.n / 16 or t.dim(2) % 8 != 0) return error.NotExl3Mcg4;
            const halves = t.dim(2) / 8;
            if (halves < 2 or halves > 16) return error.NotExl3Mcg4;
            var cb: ?cuda.exl3.Codebook = null;
            for (@import("core").exl3_load.markers) |mk| {
                var nb: [192]u8 = undefined;
                const name = try std.fmt.bufPrint(&nb, "model.language_model.layers.{d}.mlp.experts.{d}." ++ proj ++ ".{s}", .{ i, e, mk.name });
                if (!L.ck.has(name)) continue;
                const marker = try L.ck.get(name);
                if (marker.dtype != .i32 or marker.numel() != 1 or marker.bytes.len != 4 or std.mem.readInt(u32, marker.bytes[0..4], .little) != mk.value)
                    return error.NotExl3Mcg4;
                cb = switch (mk.codebook) {
                    .mcg => .mcg,
                    .mul1 => .mul1,
                    .inst3 => .inst3,
                };
            }
            const c = cb orelse return error.NotExl3Mcg4; // 3inst experts: no checkpoint of this family has them
            if (book) |b| (if (b != c) return error.MixedExl3Codebooks) else book = c;
            if (halves % 2 == 1 and c != .mul1) return error.NotExl3Mcg4;
            if (first_halves) |f| (if (f != halves) {
                uniform = false;
            }) else first_halves = halves;
            k2s[e] = @intCast(halves);
            if (halves != 4 and halves != 6 and halves != 8) prompt_widths = false;
            if (comptime std.mem.eql(u8, proj, "gate_proj")) gk2[e] = @intCast(halves);
            if (comptime std.mem.eql(u8, proj, "up_proj")) (if (gk2[e] != halves) {
                prompt_widths = false;
            });
            const tb = tileBytes(halves);
            offsets[e] = at;
            if (m.by_cols) { // every tile row, the rank's tile columns
                const row = (m.n / 16) * tb;
                for (0..kt) |r| try L.host.appendSlice(L.gpa, t.bytes[r * row + (cut.lo / 16) * tb ..][0 .. nt * tb]);
            } else try L.host.appendSlice(L.gpa, t.bytes[(cut.lo / 16) * nt * tb ..][0 .. kt * nt * tb]); // its tile rows
            at += kt * nt * tb;
        }
        const base = try L.put("layers.{d}.moe.exl3." ++ tw, .{i});
        for (0..count) |e| {
            const ptr = base + offsets[e];
            try L.host.appendSlice(L.gpa, std.mem.asBytes(&ptr));
        }
        @field(out, tw) = try L.put("layers.{d}.moe.exl3." ++ tw ++ "_ptrs", .{i});
        for (0..count) |e| { // suh: the input's scales (gate, up: every input; down: the rank's)
            const t = try L.get("layers.{d}.mlp.experts.{d}." ++ proj ++ ".suh", .{ i, e });
            if (t.dtype != .f16 or t.numel() != m.k) return error.NotExl3Mcg4;
            const mine = if (m.by_cols) t.bytes else t.bytes[cut.lo * 2 .. cut.hi * 2];
            if (m.by_cols) { // one rotation shared by the layer's every gate and up input?
                if (first_suh) |f| (if (!std.mem.eql(u8, f, mine)) {
                    out.shx = false;
                }) else first_suh = mine;
            }
            try L.host.appendSlice(L.gpa, mine);
        }
        @field(out, su) = try L.put("layers.{d}.moe.exl3." ++ su, .{i});
        for (0..count) |e| { // svh: the output's scales (gate, up: the rank's; down: every output)
            const t = try L.get("layers.{d}.mlp.experts.{d}." ++ proj ++ ".svh", .{ i, e });
            if (t.dtype != .f16 or t.numel() != m.n) return error.NotExl3Mcg4;
            try L.host.appendSlice(L.gpa, if (m.by_cols) t.bytes[cut.lo * 2 .. cut.hi * 2] else t.bytes);
        }
        @field(out, sv) = try L.put("layers.{d}.moe.exl3." ++ sv, .{i});
        try L.host.appendSlice(L.gpa, std.mem.sliceAsBytes(k2s));
        @field(out, kw) = try L.put("layers.{d}.moe.exl3." ++ kw, .{i});
    }
    out.codebook = book orelse return error.NotExl3Mcg4;
    out.uniform = if (uniform) @intCast(first_halves.?) else 0;
    out.prompt_k2 = out.codebook == .mul1 and prompt_widths;
    return out;
}

/// The selected common EXL3 owner. This file adapts the family's checkpoint shard and slot layout only.
pub const Kernels = cuda.exl3_experts.Experts;
const common = cuda.exl3_experts;
pub const Scratch = struct { picks: u64, xg: u64, xu: u64, z: u64 };
pub const top_k = 8;
/// Rows a call to the common experts takes: each call reads every expert's weights, so a prompt pass needs few calls.
pub const chunk_rows = 1024;
/// The prompt item plan has no decode grouper's shared-memory cap; one call covers a supported whole prompt pass.
pub const prompt_rows = 4096;
fn layer(w: Experts, count: usize, dims: usize, width: usize, limit: f32) common.Layer {
    const up = if (w.k2u != 0) w.k2u else w.k2;
    const down = if (w.k2d != 0) w.k2d else w.k2;
    return .{ .gate = w.gt, .up = w.ut, .down = w.dt, .gate_k2 = w.k2, .up_k2 = up, .down_k2 = down, .suh_g = w.suh_g, .suh_u = w.suh_u, .svh_g = w.svh_g, .svh_u = w.svh_u, .suh_d = w.suh_d, .svh_d = w.svh_d, .experts = count, .dims = dims, .width = width, .codebook = w.codebook, .limit = limit, .act_mode = 0, .uniform_k2 = w.uniform, .prompt_k2 = w.prompt_k2, .shared_input = w.shx };
}
fn shape(dims: usize, width: usize) common.Layer {
    const w: Experts = .{ .gt = 0, .ut = 0, .dt = 0, .k2 = 0, .suh_g = 0, .suh_u = 0, .svh_g = 0, .svh_u = 0, .suh_d = 0, .svh_d = 0, .shx = false };
    return layer(w, cuda.grouped.max_experts, dims, width, 0);
}
/// A chunk's compacted picks ahead of the common scratch, 256-aligned; the usize keeps @min from narrowing to u7.
fn picksBytes(rows: usize) usize {
    const cap: usize = @min(rows, chunk_rows);
    return std.mem.alignForward(usize, cap * top_k * 4, 256);
}
pub fn scratchBytes(rows: usize, dims: usize, width: usize, prefill: bool, shx: bool) [4]usize {
    return scratchFor(rows, dims, width, prefill, shx, true);
}
/// scratchBytes for a pack whose prompt chunks take the prompt kernels (`uniform`: Mia-AiLab's 4 bits, or a REXL3 pack
/// whose every layer's mix they read) or the compacted path (any other mix).
pub fn scratchFor(rows: usize, dims: usize, width: usize, prefill: bool, shx: bool, uniform: bool) [4]usize {
    var l = shape(dims, width);
    l.shared_input = shx;
    const n: usize = @min(rows, chunk_rows);
    if (uniform and prefill and common.prompt.eligible(l)) return .{ common.prompt.bytes(l, @min(rows, prompt_rows), top_k + 1) catch unreachable, 4, 4, 4 };
    return .{ picksBytes(rows) + (common.scratchBytes(shape(dims, width), n, top_k) catch unreachable), 4, 4, 4 };
}
/// `height` rows of `width` bytes, `spitch` apart at `src`, to `dpitch` apart at `dst`, in one call.
fn copyRows(stream: cuda.Stream, dst: u64, dpitch: usize, src: u64, spitch: usize, width: usize, height: usize, what: []const u8) !void {
    const m: cuda.abi.Memcpy2D = .{ .src = src, .src_pitch = spitch, .dst = dst, .dst_pitch = dpitch, .width = width, .height = height };
    try stream.d.check(stream.d.api.cuMemcpy2DAsync_v2(&m, stream.handle), what);
}
/// Whether `run` takes the prompt kernels for these experts (and so can write bf16 slot rows).
pub fn promptPath(w: Experts, count: usize, dims: usize, width: usize, limit: f32) bool {
    return dims % 256 == 0 and width % 128 == 0 and common.prompt.eligible(layer(w, count, dims, width, limit));
}
/// `yb`: a prompt chunk on the prompt kernels writes bf16 slot rows (promptPath; the compacted path stays fp32).
pub fn run(k: *const Kernels, ek: *const experts.Kernels, stream: cuda.Stream, w: Experts, sc: Scratch, plan_buf: cuda.grouped.Plan, x: u64, x_stride: usize, picks: u64, slots: usize, rows: usize, count: usize, dims: usize, width: usize, limit: f32, prompt: bool, ey: u64, xd: u64, yb: bool) !void {
    _ = xd;
    if (slots != top_k + 1) return error.UnsupportedTopK;
    if (dims % 256 != 0 or width % 128 != 0) return error.NotExl3Mcg4;
    const l = layer(w, count, dims, width, limit);
    if (prompt and common.prompt.eligible(l)) {
        const scratch = try common.prompt.scratch(sc.picks, l, @min(rows, prompt_rows), slots);
        var at: usize = 0;
        while (at < rows) : (at += prompt_rows) {
            const n: usize = @min(prompt_rows, rows - at);
            try k.prompt.run(k, ek.router, stream, l, x + at * x_stride * 2, x_stride, picks + at * slots * 4, n, scratch, plan_buf, ey + at * slots * dims * @as(usize, if (yb) 2 else 4), count, yb);
        }
        return;
    }
    return compact(k, stream, w, sc, x, x_stride, picks, slots, rows, count, dims, width, limit, ey);
}

/// Compact routed picks and output for decode or prompt layers without the whole-pass kernel.
fn compact(k: *const Kernels, stream: cuda.Stream, w: Experts, sc: Scratch, x: u64, x_stride: usize, picks: u64, slots: usize, rows: usize, count: usize, dims: usize, width: usize, limit: f32, ey: u64) !void {
    const l = layer(w, count, dims, width, limit);
    const cap: usize = @min(rows, chunk_rows);
    const arena = sc.picks + picksBytes(rows);
    var at: usize = 0;
    while (at < rows) : (at += chunk_rows) {
        const n: usize = @min(chunk_rows, rows - at);
        try copyRows(stream, sc.picks, top_k * 4, picks + at * slots * 4, slots * 4, top_k * 4, n, "EXL3 compact route rows");
        const scratch = try common.scratch(arena, l, cap, top_k);
        try k.runSlots(stream, l, x + at * x_stride * 2, x_stride, .bf16, sc.picks, n, scratch);
        const output = scratch.y;
        try copyRows(stream, ey + at * slots * dims * 4, slots * dims * 4, output, top_k * dims * 4, top_k * dims * 4, n, "EXL3 routed output slots");
    }
}

test "GLM REXL3: EXL3 admission: Mia-AiLab's block and per-tensor mixed packs pass; other uniform blocks are refused by name" {
    const gpa = std.testing.allocator;
    const mia =
        \\{"quantization_config": {"quant_method": "exl3", "bits": 4, "codebook": "mcg", "head_bits": 16}}
    ;
    try std.testing.expect(isExl3(gpa, mia));
    var why: modelopt.Why = .{};
    try admit(gpa, mia, &why);
    try std.testing.expect(!isExl3(gpa, "{\"quantization_config\": {\"quant_method\": \"modelopt\"}}"));
    try std.testing.expect(!isExl3(gpa, "{}"));
    try std.testing.expectError(error.UnsupportedQuant, admit(gpa, "{\"quantization_config\": {\"quant_method\": \"exl3\", \"bits\": 3, \"codebook\": \"mcg\"}}", &why));
    try std.testing.expectEqualStrings("EXL3 at 3 bits: the CUDA family reads 4-bit EXL3 experts", why.text());
    try std.testing.expectError(error.UnsupportedQuant, admit(gpa, "{\"text_config\": {\"quantization_config\": {\"quant_method\": \"exl3\", \"bits\": 4, \"codebook\": \"3inst\"}}}", &why));
    try std.testing.expectEqualStrings("EXL3 codebook \"3inst\": the CUDA family reads \"mcg\"", why.text());
    // per-tensor mixed packs (vcruz305's SAGE/REXL3): the groups state their widths and codebooks, checked at load
    try admit(gpa, "{\"quantization_config\": {\"quant_method\": \"exl3\"}}", &why);
    try admit(gpa, "{\"text_config\": {\"quantization_config\": {\"quant_method\": \"exl3\", \"bits\": 4, \"codebook\": \"mul1\"}}}", &why);
}

test "a chunk's compacted picks end before the common scratch at every row count" {
    for (1..3 * chunk_rows) |rows| {
        const cap: usize = @min(rows, chunk_rows);
        try std.testing.expect(picksBytes(rows) >= cap * top_k * 4);
        try std.testing.expectEqual(@as(usize, 0), picksBytes(rows) % 256);
    }
}

test "GLM's routed experts run on the common EXL3 module's kernels" {
    try std.testing.expect(Kernels == cuda.exl3_experts.Experts);
}
