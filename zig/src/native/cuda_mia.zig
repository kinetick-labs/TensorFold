//! Our Flash Next CUDA engine (the MiaAI-Lab lineage: ExLlamaV3 packs, NVIDIA's FP4-Local ModelOpt checkpoint)
//! as a CUDA family beside upstream's (families/flashnext/cuda_lanes.zig, #952). Both read model_type qwen4_exp;
//! `TENSORFOLD_FLASHNEXT_ENGINE=mia` serves a checkpoint with this one, anything else (or unset) with upstream's.
//! The namespace speaks native/cuda.zig's family contract and forwards to flashnext_mia.native (cuda_native.zig).
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const api = @import("engine_api");
const mia = @import("flashnext_mia");
const N = mia.native;

/// The variable that picks this engine for qwen4_exp checkpoints.
pub const env_engine = "TENSORFOLD_FLASHNEXT_ENGINE";

/// Whether `value` (the variable's) asks for this engine: "mia"; unset, empty or "upstream" keep upstream's.
pub fn selects(value: ?[]const u8) bool {
    const v = std.mem.trim(u8, value orelse return false, " \t\r\n");
    return std.ascii.eqlIgnoreCase(v, "mia");
}

pub fn selected() bool {
    return selects(if (std.c.getenv(env_engine)) |p| std.mem.span(p) else null);
}

pub const model_type = N.model_type;
pub const formats = N.formats;
pub const default_context = N.default_context;
pub const max_segments: u32 = 1;
pub const prompt_rows: u32 = N.prompt_rows;
/// A kept state depends on its tokens alone (a resume absorbs the next prompt token into the MTP head).
pub const snap_lookahead: u32 = N.cache_rules.lookahead;

/// The serve options this engine takes: upstream's (context, drafts, segments, slide, the two-rank link, --kv-cache
/// fp8 as kv8) and ours that family_options.zig forwards by name (parallel, prompt_cache_gib, vision_helper).
pub const Options = struct {
    context: usize,
    drafts: bool,
    segments: usize = 1,
    slide: bool = false,
    tp: usize = 1,
    rank: usize = 0,
    master: ?[]const u8 = null,
    master_port: u16 = 29551,
    kv8: bool = false,
    parallel: u32 = 1,
    prompt_cache_gib: ?f64 = null,
    vision_helper: bool = false,
};

/// What native/cuda.zig drives; `target` is our prompt cache's hooks (null: this rank keeps no states).
pub const Loaded = struct {
    backend: lanes.backend.Backend,
    facts: lanes.Model,
    rows: u32,
    prompt_rows: u32 = N.prompt_rows,
    stream_bytes: usize,
    free_streams: u32 = 0,
    ctx: *anyopaque,
    deinit: *const fn (*anyopaque) void,
    target: *anyopaque,
    follow: ?*const fn (*anyopaque) anyerror!void = null,
};

/// The engine's own state behind `Loaded.ctx`: its Loaded and the cache hooks `target` points at.
const Held = struct {
    gpa: std.mem.Allocator,
    inner: N.Loaded,
    hooks: @FieldType(N.Loaded, "cache"),
};

fn held(p: *anyopaque) *Held {
    return @ptrCast(@alignCast(p));
}

fn release(p: *anyopaque) void {
    const h = held(p);
    h.inner.deinit(h.inner.ctx);
    h.gpa.destroy(h);
}

fn followFn(p: *anyopaque) anyerror!void {
    const h = held(p);
    return h.inner.follow.?(h.inner.ctx);
}

/// The window this checkpoint serves past config.json's (YaRN), read by native/cuda.zig's admission.
pub fn modelWindow(a: std.mem.Allocator, io: std.Io, dir: []const u8) i64 {
    return N.modelWindow(a, io, dir);
}

/// Resident weight bytes at this rank: the checkpoint's safetensors less the memory-mapped n-gram table.
pub fn placementWeightBytes(a: std.mem.Allocator, io: std.Io, dir: []const u8, world: usize, rank: usize, drafts: bool) !u64 {
    _ = world;
    _ = rank;
    _ = drafts;
    const own = try residentBytes(a, io, dir);
    // a checkpoint whose shards sit below the top level (FP4-Local's tensors/): upstream's count
    return if (own > 0) own else @import("cuda_memory.zig").weightBytes(io, dir);
}

pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: ?[]const u8, o: Options) !Loaded {
    // this engine replays a captured Triton set: TENSORFOLD_CUDA_KERNELS, else share/tensorfold/cuda/sm<cc>
    var dir_buf: [std.fs.max_path_bytes]u8 = undefined;
    const kdir = kernels orelse try kernelDir(io, try ctx.capability(), &dir_buf);
    const h = try gpa.create(Held);
    errdefer gpa.destroy(h);
    h.* = .{ .gpa = gpa, .hooks = null, .inner = try N.open(gpa, io, ctx, dir, kdir, .{
        .context = o.context,
        .drafts = o.drafts,
        .parallel = o.parallel,
        .prompt_cache_gib = o.prompt_cache_gib,
        .tp = @intCast(o.tp),
        .rank = @intCast(o.rank),
        .master = o.master orelse "",
        .master_port = o.master_port,
        .kv_dtype = if (o.kv8) .fp8 else .bf16,
        .vision = o.vision_helper,
    }) };
    h.hooks = h.inner.cache;
    return .{
        .backend = h.inner.backend,
        .facts = h.inner.facts,
        .rows = h.inner.rows,
        .stream_bytes = h.inner.stream_bytes,
        .ctx = h,
        .deinit = release,
        .target = h,
        .follow = if (h.inner.follow != null) followFn else null,
    };
}

// The prompt cache's copies (native/cuda_snaps.zig): our lane backend's hooks, through `target`.
pub fn snap_bytes(t: *anyopaque, at: u32) u64 {
    const k = held(t).hooks orelse return std.math.maxInt(u64);
    return k.bytes(k.ptr, at);
}

pub fn snap_save(t: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!api.prompt_cache.Saved {
    const k = held(t).hooks orelse return error.NoPromptCache;
    return k.save(k.ptr, owner, at);
}

pub fn snap_restore(t: *anyopaque, owner: ?*anyopaque, saved: api.prompt_cache.Saved) anyerror!void {
    const k = held(t).hooks orelse return error.NoPromptCache;
    return k.restore(k.ptr, owner, saved);
}

pub fn snap_drop(t: *anyopaque, saved: api.prompt_cache.Saved) void {
    const k = held(t).hooks orelse return;
    k.drop(k.ptr, saved);
}

pub fn explain(_: ?*anyopaque, err: anyerror) ?[]const u8 {
    return switch (err) {
        error.NoKernelDir => "this Flash Next engine replays a captured kernel set: set TENSORFOLD_CUDA_KERNELS to its folder (aot.json and cubins/), or install share/tensorfold/cuda/sm<cc> beside the binary",
        error.NoPromptCache => "this rank keeps no prompt states",
        error.BadDepth => "TF_FLASHNEXT_DEPTH is a draft count from 1 to the engine's maximum",
        error.UnsupportedTp => "this Flash Next engine serves --tp 1 or 2",
        else => null,
    };
}

fn kernelDir(io: std.Io, capability: u32, buf: []u8) ![]const u8 {
    if (std.c.getenv("TENSORFOLD_CUDA_KERNELS")) |p| {
        const v = std.mem.span(p);
        if (v.len > 0 and !std.mem.eql(u8, v, "native")) return checked(io, v);
    }
    var exe_buf: [std.fs.max_path_bytes]u8 = undefined;
    const n = std.process.executableDirPath(io, &exe_buf) catch return error.NoKernelDir;
    const path = std.fmt.bufPrint(buf, "{s}/../share/tensorfold/cuda/sm{d}", .{ exe_buf[0..n], capability }) catch return error.NoKernelDir;
    return checked(io, path);
}

fn checked(io: std.Io, path: []const u8) ![]const u8 {
    var d = std.Io.Dir.cwd().openDir(io, path, .{}) catch return error.NoKernelDir;
    d.close(io);
    return path;
}

/// Every top-level `*.safetensors` the loader uploads; a file holding only the n-gram table is memory-mapped and read
/// a row at a time (cuda_weights.NgramTable), so it is not counted.
fn residentBytes(a: std.mem.Allocator, io: std.Io, dir: []const u8) !u64 {
    var d = try std.Io.Dir.cwd().openDir(io, dir, .{ .iterate = true });
    defer d.close(io);
    var total: u64 = 0;
    var it = d.iterate();
    while (try it.next(io)) |e| {
        if (e.kind != .file and e.kind != .sym_link) continue;
        if (!std.mem.endsWith(u8, e.name, ".safetensors")) continue;
        const path = try std.fs.path.join(a, &.{ dir, e.name });
        defer a.free(path);
        if (mappedNotResident(a, io, path)) continue;
        const st = std.Io.Dir.cwd().statFile(io, path, .{}) catch continue;
        total += st.size;
    }
    return total;
}

fn mappedNotResident(a: std.mem.Allocator, io: std.Io, path: []const u8) bool {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const aa = arena.allocator();
    var file = std.Io.Dir.cwd().openFile(io, path, .{}) catch return false;
    defer file.close(io);
    var head: [8]u8 = undefined;
    if ((file.readPositionalAll(io, &head, 0) catch return false) != head.len) return false;
    const len: usize = @intCast(std.mem.readInt(u64, &head, .little));
    if (len == 0 or len > (16 << 20)) return false;
    const json = aa.alloc(u8, len) catch return false;
    if ((file.readPositionalAll(io, json, 8) catch return false) != len) return false;
    const doc = std.json.parseFromSliceLeaky(std.json.Value, aa, json, .{}) catch return false;
    if (doc != .object) return false;
    var any = false;
    var it = doc.object.iterator();
    while (it.next()) |kv| {
        if (std.mem.eql(u8, kv.key_ptr.*, "__metadata__")) continue;
        if (std.mem.indexOf(u8, kv.key_ptr.*, "ngram_embedding") == null) return false;
        any = true;
    }
    return any;
}

test "the engine variable picks ours only when it says mia" {
    try std.testing.expect(selects("mia"));
    try std.testing.expect(selects(" MIA\n"));
    try std.testing.expect(!selects(null));
    try std.testing.expect(!selects(""));
    try std.testing.expect(!selects("upstream"));
}
