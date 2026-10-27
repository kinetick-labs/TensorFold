//! The NVFP4 n-gram table read by positional reads instead of page faults on its 28.7 GiB mapping.
//!
//! A row is 80 code bytes and 10 scale bytes in two arrays of one shard; a fault on the mapping reads a whole page for each
//! (two 4 KiB reads with no readahead, a major fault each, serialized on the address space). Here every row is read with a
//! `pread` of its own bytes (O_DIRECT where the file system allows it, so the page cache is left to the weights) by a
//! pool of parked threads, into a direct-mapped cache of raw rows. A gather finds its rows in the cache, and the rows a
//! later pass will gather can be asked for ahead of it (`prefetch`): a drafted token's rows while the head drafts the next
//! one, a prompt chunk's rows while the chunk before it runs. The bytes are the file's, so the dequantized rows are the
//! mapped path's bits.
const std = @import("std");
const core = @import("core");
const direct_io = core.direct_io;

/// Bytes of a row's codes and scales a cache slot holds (dh / 2 + dh / 16 of a 160-wide head: 90).
pub const max_row = 96;
const words = max_row / 8;
/// Set in a slot's tag while its row is being written: the row's id + 1 is the rest.
const busy: u64 = 1 << 63;
const batch_ring = 16;
pub const max_workers = 128;
/// Most ids a worker claims at once (fewer when a batch is small, so that it spreads over the workers).
const claim_ids = 8;
/// A reader that finds its row busy waits for it this long, then reads the row itself.
const wait_spins = 20_000;

/// Where one shard's two arrays sit: file index, and byte offsets of the codes and scales.
pub const Shard = struct { file: u16, codes_at: u64, scales_at: u64 };

const Slot = extern struct { tag: u64, data: [words]u64 };

/// Reads of the last report period (TF_FLASHNEXT_GATHER_LOG=1).
pub const Stats = struct {
    hits: std.atomic.Value(u64) = .init(0),
    waited: std.atomic.Value(u64) = .init(0),
    own_reads: std.atomic.Value(u64) = .init(0),
    ahead_reads: std.atomic.Value(u64) = .init(0),
    ahead_skipped: std.atomic.Value(u64) = .init(0),
    ahead_rows: std.atomic.Value(u64) = .init(0),
};

const Batch = struct {
    ids: []i64 = &.{},
    grain: usize = 1,
    claimed: std.atomic.Value(usize) = .init(0),
    finished: std.atomic.Value(usize) = .init(0),
    done: bool = true,
};

pub const Reader = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    files: []direct_io.File,
    shards: []Shard,
    /// first row of each shard, and the row count at the end
    starts: []i64,
    row_codes: usize,
    row_scales: usize,
    slots: []Slot,
    shift: u6,
    stats: Stats = .{},
    // the read-ahead queue
    mutex: std.Io.Mutex = .init,
    batches: [batch_ring]Batch = @splat(.{}),
    head: usize = 0,
    tail: usize = 0,
    go: std.atomic.Value(u32) = .init(0),
    stopping: std.atomic.Value(bool) = .init(false),
    threads: [max_workers]std.Thread = undefined,
    spawned: usize = 0,
    want_workers: usize,

    /// `paths` are opened for reading; `slot_count` is a power of two. The reader owns nothing the arguments pass.
    pub fn create(gpa: std.mem.Allocator, io: std.Io, paths: []const [:0]const u8, shards: []const Shard, starts: []const i64, row_codes: usize, row_scales: usize, slot_count: usize, n_workers: usize) !*Reader {
        if (row_codes + row_scales > max_row or !std.math.isPowerOfTwo(slot_count) or shards.len + 1 != starts.len) return error.UnexpectedTensor;
        const r = try gpa.create(Reader);
        errdefer gpa.destroy(r);
        const files = try gpa.alloc(direct_io.File, paths.len);
        errdefer gpa.free(files);
        var opened: usize = 0;
        errdefer for (files[0..opened]) |*f| f.close();
        for (paths, files) |p, *f| {
            f.* = try direct_io.File.open(p);
            opened += 1;
        }
        const own_shards = try gpa.dupe(Shard, shards);
        errdefer gpa.free(own_shards);
        const own_starts = try gpa.dupe(i64, starts);
        errdefer gpa.free(own_starts);
        // anonymous memory, zero until a slot is written: only the slots in use are resident
        const mem = try std.posix.mmap(null, slot_count * @sizeOf(Slot), .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
        r.* = .{
            .gpa = gpa,
            .io = io,
            .files = files,
            .shards = own_shards,
            .starts = own_starts,
            .row_codes = row_codes,
            .row_scales = row_scales,
            .slots = @as([*]Slot, @ptrCast(mem.ptr))[0..slot_count],
            .shift = @intCast(@as(u32, 64) - std.math.log2_int(usize, slot_count)),
            .want_workers = @min(n_workers, max_workers),
        };
        return r;
    }

    /// Stops the workers and releases the files and the cache.
    pub fn destroy(r: *Reader) void {
        r.stopping.store(true, .release);
        _ = r.go.fetchAdd(1, .release);
        r.io.futexWake(u32, &r.go.raw, std.math.maxInt(u32));
        for (r.threads[0..r.spawned]) |th| th.join();
        for (&r.batches) |*b| if (!b.done) r.gpa.free(b.ids);
        for (r.files) |*f| f.close();
        std.posix.munmap(@as([*]align(std.heap.page_size_min) u8, @ptrCast(@alignCast(r.slots.ptr)))[0 .. r.slots.len * @sizeOf(Slot)]);
        r.gpa.free(r.files);
        r.gpa.free(r.shards);
        r.gpa.free(r.starts);
        r.gpa.destroy(r);
    }

    fn slotOf(r: *const Reader, id: u64) *Slot {
        return &r.slots[(id *% 0x9E3779B97F4A7C15) >> r.shift];
    }

    fn locate(r: *const Reader, id: i64) ?struct { shard: usize, local: u64 } {
        const n = r.shards.len;
        if (id < 0 or id >= r.starts[n]) return null;
        var lo: usize = 0;
        var hi = n;
        while (lo + 1 < hi) {
            const mid = (lo + hi) / 2;
            if (r.starts[mid] <= id) lo = mid else hi = mid;
        }
        return .{ .shard = lo, .local = @intCast(id - r.starts[lo]) };
    }

    /// The row's codes then scales from the file, ahead of nothing: two positional reads.
    fn readRaw(r: *const Reader, shard: usize, local: u64, out: *[max_row]u8) !void {
        var buf: [2 * direct_io.alignment]u8 align(direct_io.alignment) = undefined;
        const s = r.shards[shard];
        const f = r.files[s.file];
        @memcpy(out[0..r.row_codes], try f.read(&buf, s.codes_at + local * r.row_codes, r.row_codes));
        @memcpy(out[r.row_codes..][0..r.row_scales], try f.read(&buf, s.scales_at + local * r.row_scales, r.row_scales));
    }

    fn store(slot: *Slot, raw: *const [max_row]u8) void {
        for (0..words) |i| @atomicStore(u64, &slot.data[i], std.mem.readInt(u64, raw[i * 8 ..][0..8], .little), .release);
    }

    fn load(slot: *const Slot, raw: *[max_row]u8) void {
        for (0..words) |i| std.mem.writeInt(u64, raw[i * 8 ..][0..8], @atomicLoad(u64, &slot.data[i], .acquire), .little);
    }

    /// Row `id`'s raw bytes (codes, then scales): from the cache, from the read another thread has under way, or from the file.
    pub fn get(r: *Reader, id: i64, out: *[max_row]u8) !void {
        const loc = r.locate(id) orelse return error.UnexpectedTensor;
        const want: u64 = @as(u64, @intCast(id)) + 1;
        const slot = r.slotOf(want);
        var spins: u32 = 0;
        var waited = false;
        while (true) {
            const t = @atomicLoad(u64, &slot.tag, .acquire);
            if (t == want) {
                load(slot, out);
                // an overwrite during the copy shows in the tag (each data word was stored after the claim)
                if (@atomicLoad(u64, &slot.tag, .acquire) == want) {
                    _ = r.stats.hits.fetchAdd(1, .monotonic);
                    if (waited) _ = r.stats.waited.fetchAdd(1, .monotonic);
                    return;
                }
                continue;
            }
            if (t == (want | busy) and spins < wait_spins) {
                spins += 1;
                waited = true;
                if (spins % 64 == 0) std.Thread.yield() catch {} else std.atomic.spinLoopHint();
                continue;
            }
            if (t & busy == 0 and @cmpxchgStrong(u64, &slot.tag, t, want | busy, .acq_rel, .monotonic) == null) {
                r.readRaw(loc.shard, loc.local, out) catch |err| {
                    @atomicStore(u64, &slot.tag, 0, .release);
                    return err;
                };
                store(slot, out);
                @atomicStore(u64, &slot.tag, want, .release);
                _ = r.stats.own_reads.fetchAdd(1, .monotonic);
                return;
            }
            // the slot is another row's in the middle of a write: read without keeping the row
            try r.readRaw(loc.shard, loc.local, out);
            _ = r.stats.own_reads.fetchAdd(1, .monotonic);
            return;
        }
    }

    /// Reads row `id` into the cache unless it is there or under way.
    fn fetch(r: *Reader, id: i64) void {
        const loc = r.locate(id) orelse return;
        const want: u64 = @as(u64, @intCast(id)) + 1;
        const slot = r.slotOf(want);
        const t = @atomicLoad(u64, &slot.tag, .acquire);
        if (t == want or t == (want | busy) or t & busy != 0 or @cmpxchgStrong(u64, &slot.tag, t, want | busy, .acq_rel, .monotonic) != null) {
            _ = r.stats.ahead_skipped.fetchAdd(1, .monotonic);
            return;
        }
        var raw: [max_row]u8 = @splat(0);
        r.readRaw(loc.shard, loc.local, &raw) catch {
            @atomicStore(u64, &slot.tag, 0, .release);
            return;
        };
        store(slot, &raw);
        @atomicStore(u64, &slot.tag, want, .release);
        _ = r.stats.ahead_reads.fetchAdd(1, .monotonic);
    }

    /// Asks the workers to read the rows `ids` (global row ids) into the cache, in order, and returns at once. A full queue drops the call.
    pub fn prefetch(r: *Reader, ids: []const i64) void {
        if (ids.len == 0 or r.want_workers == 0) return;
        const own = r.gpa.dupe(i64, ids) catch return;
        r.mutex.lockUncancelable(r.io);
        if (r.tail - r.head == batch_ring or r.startWorkers() == 0) {
            r.mutex.unlock(r.io);
            r.gpa.free(own);
            return;
        }
        const b = &r.batches[r.tail % batch_ring];
        b.ids = own;
        b.grain = std.math.clamp((ids.len + r.want_workers - 1) / r.want_workers, 1, claim_ids);
        b.claimed.store(0, .monotonic);
        b.finished.store(0, .monotonic);
        b.done = false;
        r.tail += 1;
        r.mutex.unlock(r.io);
        _ = r.stats.ahead_rows.fetchAdd(ids.len, .monotonic);
        _ = r.go.fetchAdd(1, .release);
        r.io.futexWake(u32, &r.go.raw, @intCast(@min(r.spawned, (ids.len + b.grain - 1) / b.grain)));
    }

    /// Workers up to the wanted count (the mutex held); how many there are.
    fn startWorkers(r: *Reader) usize {
        while (r.spawned < r.want_workers) {
            r.threads[r.spawned] = std.Thread.spawn(.{}, work, .{r}) catch break;
            r.spawned += 1;
        }
        return r.spawned;
    }

    /// The next ids to read: a slice of the oldest batch that has any left (null: none), and the batch.
    fn claim(r: *Reader) ?struct { ids: []const i64, batch: *Batch } {
        r.mutex.lockUncancelable(r.io);
        defer r.mutex.unlock(r.io);
        // the newest batch first: a drafted token's few rows are wanted within milliseconds, a prompt chunk's many have a chunk's time
        var i = r.tail;
        while (i > r.head) {
            i -= 1;
            const b = &r.batches[i % batch_ring];
            const at = b.claimed.load(.monotonic);
            if (at >= b.ids.len) continue;
            const n = @min(b.grain, b.ids.len - at);
            b.claimed.store(at + n, .monotonic);
            return .{ .ids = b.ids[at..][0..n], .batch = b };
        }
        return null;
    }

    fn finish(r: *Reader, b: *Batch, n: usize) void {
        // the length before the count: the last finisher frees the ids
        const len = b.ids.len;
        if (b.finished.fetchAdd(n, .acq_rel) + n != len) return;
        r.mutex.lockUncancelable(r.io);
        defer r.mutex.unlock(r.io);
        r.gpa.free(b.ids);
        b.ids = &.{};
        b.done = true;
        while (r.head < r.tail and r.batches[r.head % batch_ring].done) r.head += 1;
    }

    fn work(r: *Reader) void {
        while (true) {
            const seen = r.go.load(.acquire);
            if (r.stopping.load(.acquire)) return;
            if (r.claim()) |c| {
                for (c.ids) |id| r.fetch(id);
                r.finish(c.batch, c.ids.len);
                continue;
            }
            r.io.futexWaitUncancelable(u32, &r.go.raw, seen);
        }
    }

    /// Returns once every asked-for row has been read (tests, and a clean stop).
    pub fn drain(r: *Reader) void {
        while (true) {
            r.mutex.lockUncancelable(r.io);
            const idle = r.head == r.tail;
            r.mutex.unlock(r.io);
            if (idle) return;
            std.Thread.yield() catch {};
        }
    }

    /// Takes the counters since the last call.
    pub fn take(r: *Reader) struct { hits: u64, waited: u64, own: u64, ahead: u64, skipped: u64, rows: u64 } {
        return .{
            .hits = r.stats.hits.swap(0, .monotonic),
            .waited = r.stats.waited.swap(0, .monotonic),
            .own = r.stats.own_reads.swap(0, .monotonic),
            .ahead = r.stats.ahead_reads.swap(0, .monotonic),
            .skipped = r.stats.ahead_skipped.swap(0, .monotonic),
            .rows = r.stats.ahead_rows.swap(0, .monotonic),
        };
    }
};

fn envOr(name: [:0]const u8, default: usize) usize {
    const v = std.c.getenv(name) orelse return default;
    return std.fmt.parseInt(usize, std.mem.span(v), 10) catch default;
}

/// TF_FLASHNEXT_GATHER_READ=pread: the NVFP4 table's rows by positional reads and a row cache (default: the mapping's page faults).
pub fn preadOn() bool {
    const v = std.c.getenv("TF_FLASHNEXT_GATHER_READ") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "pread");
}

/// TF_FLASHNEXT_GATHER_CACHE_MIB: the row cache's address space (default 256 MiB: 2^21 rows of 104 bytes; only slots in use are resident).
pub fn cacheSlots() usize {
    const mib = envOr("TF_FLASHNEXT_GATHER_CACHE_MIB", 256);
    return std.math.floorPowerOfTwo(usize, @max(1, mib << 20) / @sizeOf(Slot));
}

/// TF_FLASHNEXT_GATHER_THREADS: read-ahead threads (default 128, the most; a Spark's drive reads 100 us a row at that depth,
/// 160 us at 64 and 230 us at 32; 0: no read-ahead, every row is read by the gather).
pub fn workers() usize {
    return envOr("TF_FLASHNEXT_GATHER_THREADS", 128);
}

/// TF_FLASHNEXT_DRAFT_AHEAD=1: a drafted token's rows are read while the head drafts the next (needs GATHER_READ=pread).
pub fn draftAhead() bool {
    const v = std.c.getenv("TF_FLASHNEXT_DRAFT_AHEAD") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}

/// TF_FLASHNEXT_WARM_AHEAD=1: a prompt chunk's rows are read while the chunk before it runs (needs GATHER_READ=pread).
pub fn warmAhead() bool {
    const v = std.c.getenv("TF_FLASHNEXT_WARM_AHEAD") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}

/// TF_FLASHNEXT_PREFILL_TAIL=rows: a prompt's last chunk under this many rows is evened out with the one before it (0: off).
pub fn prefillTail() usize {
    return envOr("TF_FLASHNEXT_PREFILL_TAIL", 0);
}
