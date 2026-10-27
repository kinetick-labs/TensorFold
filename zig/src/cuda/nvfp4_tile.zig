//! Routed NVFP4 experts of a prompt chunk on shared-memory row tiles (nvfp4_experts_tile.cu), the decode kernel's bits.
const std = @import("std");
const driver = @import("driver.zig");
const module = @import("module.zig");
const launch_ = @import("launch.zig");
const stream_ = @import("stream.zig");
const kernels = @import("kernels.zig");
const grouped = @import("grouped.zig");
const experts = @import("experts.zig");

const Driver = driver.Driver;
const Module = module.Module;
const Function = module.Function;
const Stream = stream_.Stream;
const Plan = grouped.Plan;

/// Threads a CTA: four warps, one 32-column block each.
const threads: u32 = 128;

/// Bytes of X fragments a stage holds (four row tiles of 16 pairs, one K32 group) and of one matrix's weight block.
const x_stage: u32 = 4 * 1024;
const weight_block: u32 = 4 * 576;

/// One kernel instance: its name, matrices a block (2: gate and up, 1: down) and cp.async stages.
const Variant = struct { name: [:0]const u8, parts: u32, stages: u32 };

/// Gate/up: row tiles a pass (rt) and stages (s), by index.
const gate_up = [_]Variant{
    .{ .name = "tf_nvfp4_tile_gu_rt4_s3", .parts = 2, .stages = 3 },
    .{ .name = "tf_nvfp4_tile_gu_rt4_s4", .parts = 2, .stages = 4 },
    .{ .name = "tf_nvfp4_tile_gu_rt2_s3", .parts = 2, .stages = 3 },
    .{ .name = "tf_nvfp4_tile_gu_rt2_s4", .parts = 2, .stages = 4 },
    .{ .name = "tf_nvfp4_tile_gu_rt3_s3", .parts = 2, .stages = 3 },
};

/// Down to fp32.
const down_f32 = [_]Variant{
    .{ .name = "tf_nvfp4_tile_f32_rt4_s3", .parts = 1, .stages = 3 },
    .{ .name = "tf_nvfp4_tile_f32_rt4_s4", .parts = 1, .stages = 4 },
    .{ .name = "tf_nvfp4_tile_f32_rt2_s3", .parts = 1, .stages = 3 },
};

/// The dynamic shared bytes of an instance: S x (4 row tiles of X, then 4 warps' weight blocks).
fn sharedBytes(v: Variant) u32 {
    return v.stages * (x_stage + weight_block * v.parts);
}

/// Which of a prompt chunk's expert launches take the tile kernel.
pub const Mode = enum { off, gate_up, both };

pub const Config = struct {
    mode: Mode = .both,
    /// Pairs an item of the plan holds (the plan is routed with this tile).
    pairs: usize = 64,
    /// The instances of gate/up and of down.
    gu: usize = 1,
    down: usize = 2,
    /// Chunks of fewer rows keep the decode kernel on 16-pair items.
    min_rows: usize = 1024,
    /// Set when the mode was given explicitly: it then holds at any rank count, else only at one rank.
    any_world: bool = false,

    /// The TF_FLASHNEXT_PROMPT_EXPERT_TILE settings: mode (0 off, 1 gate/up, 2 both), _CFG "pairs,gu,down", _ROWS.
    pub fn parse(mode: ?[]const u8, cfg: ?[]const u8, rows: ?[]const u8) !Config {
        var c: Config = .{};
        if (mode) |m| {
            c.any_world = true;
            c.mode = if (std.mem.eql(u8, m, "0")) .off else if (std.mem.eql(u8, m, "1")) .gate_up else if (std.mem.eql(u8, m, "2")) .both else return error.BadExpertTile;
        }
        if (cfg) |text| {
            var it = std.mem.splitScalar(u8, text, ',');
            var v: [3]usize = undefined;
            for (&v) |*o| o.* = std.fmt.parseInt(usize, it.next() orelse return error.BadExpertTile, 10) catch return error.BadExpertTile;
            if (it.next() != null) return error.BadExpertTile;
            if (v[0] < 16 or v[0] > 1024 or v[0] % 16 != 0 or v[1] >= gate_up.len or v[2] >= down_f32.len) return error.BadExpertTile;
            c.pairs = v[0];
            c.gu = v[1];
            c.down = v[2];
        }
        if (rows) |text| {
            c.min_rows = std.fmt.parseInt(usize, text, 10) catch return error.BadExpertTile;
            if (c.min_rows == 0) return error.BadExpertTile;
        }
        return c;
    }

    pub fn fromEnv() !Config {
        const get = struct {
            fn f(name: [*:0]const u8) ?[]const u8 {
                return if (std.c.getenv(name)) |v| std.mem.span(v) else null;
            }
        }.f;
        return parse(get("TF_FLASHNEXT_PROMPT_EXPERT_TILE"), get("TF_FLASHNEXT_PROMPT_EXPERT_TILE_CFG"), get("TF_FLASHNEXT_PROMPT_EXPERT_TILE_ROWS"));
    }

    /// Gate/up of a chunk of `rows` rows on a rank group of `world` takes the tile kernel.
    pub fn applies(c: Config, rows: usize, world: usize) bool {
        return c.mode != .off and rows >= c.min_rows and (world == 1 or c.any_world);
    }

    /// Down of such a chunk does too.
    pub fn appliesDown(c: Config, rows: usize, world: usize) bool {
        return c.mode == .both and c.applies(rows, world);
    }
};

pub const Tile = struct {
    mod: Module,
    gu: [gate_up.len]Fn,
    down: [down_f32.len]Fn,
    cfg: Config,
    sms: usize,

    /// An instance and its resident CTAs an SM (at least 1) at its shared bytes.
    const Fn = struct { f: Function, per_sm: usize, shared: u32 };

    fn open(mod: Module, v: Variant) !Fn {
        const f = try mod.function(v.name);
        const shared = sharedBytes(v);
        return .{ .f = f, .per_sm = @max(1, try f.occupancy(threads, shared)), .shared = shared };
    }

    pub fn load(d: *const Driver, sms: usize, cfg: Config) !Tile {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var mod = try Module.load(d, kernels.nvfp4_experts_tile);
        errdefer mod.unload();
        var t: Tile = .{ .mod = mod, .gu = undefined, .down = undefined, .cfg = cfg, .sms = sms };
        for (&t.gu, gate_up) |*f, v| f.* = try open(mod, v);
        for (&t.down, down_f32) |*f, v| f.* = try open(mod, v);
        return t;
    }

    pub fn unload(t: *Tile) void {
        t.mod.unload();
    }

    /// CTAs a launch of `items` plan items on `nb` column blocks gets: a unit is an item and 4 column blocks.
    fn gridSize(items: usize, nb: usize, f: Fn, sms: usize) usize {
        return @min(items * ((nb + 3) / 4), f.per_sm * sms);
    }

    /// Nvfp4Experts.run on a plan routed at `cfg.pairs` pairs an item (`items`: grouped.maxItems at that tile).
    pub fn run(t: *const Tile, s: Stream, epi: experts.Epi, in: experts.Rows, l: experts.Layer, p: Plan, out: u64, items: usize, skip: i32) !void {
        if (l.format != .nvfp4) return error.FormatMismatch;
        if (epi == .down_bf16) return error.EpilogueNotBuilt;
        if (l.relu2) return error.OptionNotBuilt;
        const up = epi == .up;
        const kg = (if (up) l.dims else l.width) / 32;
        const nb = (if (up) l.width else l.dims) / experts.Nvfp4Experts.cols;
        const f = if (up) t.gu[t.cfg.gu] else t.down[t.cfg.down];
        const grid = gridSize(items, nb, f, t.sms);
        if (grid < 1) return;
        var a: launch_.Args = .{};
        a.add(in.x);
        a.add(@as(i32, @intCast(in.stride)));
        a.add(@as(i32, @intCast(if (up) in.slots else 0)));
        a.add(if (up) l.up else l.down);
        a.add(if (up) l.up_scale else l.down_scale);
        a.add(@as(i32, @intCast(kg)));
        a.add(@as(i32, @intCast(nb)));
        for ([_]u64{ p.items, p.counts, p.members, out }) |v| a.add(v);
        a.add(@as(i32, @intCast(if (up) l.width else l.dims)));
        a.add(if (up) l.limit else @as(f32, 0.0));
        a.add(skip);
        try launch_.launch(f.f, .{ .grid = .{ .x = @intCast(grid), .y = 1, .z = 1 }, .block = .{ .x = threads, .y = 1, .z = 1 }, .shared = f.shared }, s, &a);
    }
};

test "the mode and its tuning knobs parse strictly" {
    const c = try Config.parse(null, null, null);
    try std.testing.expect(c.mode == .both and !c.any_world and c.pairs == 64 and c.gu == 1 and c.down == 2 and c.min_rows == 1024);
    try std.testing.expect((try Config.parse("0", null, null)).mode == .off);
    try std.testing.expect((try Config.parse("1", null, null)).mode == .gate_up);
    const two = try Config.parse("2", null, null);
    try std.testing.expect(two.mode == .both and two.any_world);
    for ([_][]const u8{ "", "3", "on", "02", "-1" }) |bad| try std.testing.expectError(error.BadExpertTile, Config.parse(bad, null, null));
    const t = try Config.parse(null, "32,0,1", "2048");
    try std.testing.expect(t.pairs == 32 and t.gu == 0 and t.down == 1 and t.min_rows == 2048);
    for ([_][]const u8{ "", "64", "64,1", "64,1,2,3", "8,1,2", "24,1,2", "2048,1,2", "64,5,2", "64,1,3", "64,x,2", "64,1,-2" }) |bad| try std.testing.expectError(error.BadExpertTile, Config.parse(null, bad, null));
    for ([_][]const u8{ "", "0", "1k", "-1" }) |bad| try std.testing.expectError(error.BadExpertTile, Config.parse(null, null, bad));
}

test "a chunk takes the tile from min_rows, down only in mode 2, two ranks only when the mode was given" {
    const c: Config = .{};
    try std.testing.expect(!c.applies(1023, 1) and c.applies(1024, 1) and c.applies(1536, 1));
    try std.testing.expect(c.appliesDown(1536, 1) and !c.appliesDown(1023, 1));
    try std.testing.expect(!c.applies(1536, 2) and !c.appliesDown(1536, 2));
    const gu = try Config.parse("1", null, null);
    try std.testing.expect(gu.applies(1536, 2) and !gu.appliesDown(1536, 2));
    const two = try Config.parse("2", null, "2048");
    try std.testing.expect(two.appliesDown(2048, 2) and !two.applies(2047, 2));
    try std.testing.expect(!(try Config.parse("0", null, null)).applies(4096, 1));
}

test "every instance fits the 48 KiB a launch may use without an opt-in, and keeps the shared bytes the kernel indexes" {
    for (gate_up ++ down_f32) |v| try std.testing.expect(sharedBytes(v) <= 48 * 1024);
    try std.testing.expectEqual(@as(u32, 4 * (4096 + 2 * 2304)), sharedBytes(gate_up[1]));
    try std.testing.expectEqual(@as(u32, 3 * (4096 + 2304)), sharedBytes(down_f32[2]));
    // the names carry the instance's matrices and stages: gu = 2 matrices, f32 = 1; "_s<stages>" last
    for (gate_up) |v| try std.testing.expect(v.parts == 2 and std.mem.indexOf(u8, v.name, "_gu_") != null and v.name[v.name.len - 1] == '0' + @as(u8, @intCast(v.stages)));
    for (down_f32) |v| try std.testing.expect(v.parts == 1 and std.mem.indexOf(u8, v.name, "_f32_") != null and v.name[v.name.len - 1] == '0' + @as(u8, @intCast(v.stages)));
}

test "the grid covers a unit of four column blocks an item and caps at the resident CTAs" {
    const f: Tile.Fn = .{ .f = undefined, .per_sm = 2, .shared = 0 };
    try std.testing.expectEqual(@as(usize, 5 * 5), Tile.gridSize(5, 20, f, 48)); // gate/up: 640 / 32 = 20 blocks, 5 units an item
    try std.testing.expectEqual(@as(usize, 96), Tile.gridSize(1000, 80, f, 48)); // down: capped at 2 x 48
    try std.testing.expectEqual(@as(usize, 3), Tile.gridSize(1, 9, f, 48)); // a partial last group of warps
    try std.testing.expectEqual(@as(usize, 0), Tile.gridSize(0, 20, f, 48));
}

test "a plan at 64 pairs an item fits the scratch the 16-pair plan sized" {
    for ([_]usize{ 1, 11, 129 * 11, 1536 * 11, 16384 * 11 }) |pairs| {
        const cap = (pairs + 15) / 16 + 513;
        try std.testing.expect(grouped.maxItems(pairs, 513, 64) <= cap);
        try std.testing.expect(grouped.maxItems(pairs, 513, 1024) <= cap);
    }
}

test "rows the kernel does not build are refused before any launch" {
    const t: Tile = undefined; // never reached: the refusal comes first
    const s: Stream = undefined;
    const p: Plan = .{ .members = 0, .items = 0, .counts = 0, .rank = 0, .hist = 0 };
    const in: experts.Rows = .{ .x = 0, .stride = 64 };
    const l: experts.Layer = .{ .format = .nvfp4, .up = 0, .down = 0, .width = 64, .dims = 64, .experts = 2 };
    try std.testing.expectError(error.EpilogueNotBuilt, t.run(s, .down_bf16, in, l, p, 0, 1, -1));
    var relu = l;
    relu.relu2 = true;
    try std.testing.expectError(error.OptionNotBuilt, t.run(s, .up, in, relu, p, 0, 1, -1));
    var other = l;
    other.format = .fp8g;
    try std.testing.expectError(error.FormatMismatch, t.run(s, .up, in, other, p, 0, 1, -1));
}
