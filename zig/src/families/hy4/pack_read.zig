//! Reads tensors out of the downloaded Hy4 pack by checkpoint name: `model.safetensors.index.json` (or the Engram
//! sidecar's index) names the shard, each shard's header is parsed once, and a tensor is one positional read.
//! Plain tensors convert to f32 for the CPU references; `raw` hands EXL3 parts and FP8 tables over as bytes.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Io = std.Io;

pub const Dtype = enum { F32, BF16, F16, I32, I16, F8_E4M3, U8, I64 };

pub const Info = struct {
    shard: []const u8,
    dtype: Dtype,
    shape: []const u64,
    /// absolute byte range in the shard file
    begin: u64,
    end: u64,

    pub fn count(i: Info) u64 {
        var n: u64 = 1;
        for (i.shape) |s| n *= s;
        return n;
    }
};

pub const Pack = struct {
    a: Allocator,
    io: Io,
    dir: []const u8,
    map: std.StringHashMapUnmanaged([]const u8) = .empty,
    shards: std.StringHashMapUnmanaged(Shard) = .empty,

    const Shard = struct { file: Io.File, data_at: u64, header: std.json.Value };

    /// `index` is the index file's name in `dir` (`model.safetensors.index.json` or `engram.safetensors.index.json`).
    /// Everything allocated lives in `a` (an arena suits it).
    pub fn open(a: Allocator, io: Io, dir: []const u8, index: []const u8) !Pack {
        var p: Pack = .{ .a = a, .io = io, .dir = dir };
        const text = try Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ dir, index }), a, .limited(1 << 28));
        const parsed = try std.json.parseFromSliceLeaky(std.json.Value, a, text, .{});
        const wm = (parsed.object.get("weight_map") orelse return error.BadIndex).object;
        var it = wm.iterator();
        while (it.next()) |e| try p.map.put(a, e.key_ptr.*, e.value_ptr.string);
        return p;
    }

    pub fn close(p: *Pack) void {
        var it = p.shards.valueIterator();
        while (it.next()) |s| s.file.close(p.io);
    }

    fn shard(p: *Pack, name: []const u8) !*Shard {
        if (p.shards.getPtr(name)) |s| return s;
        const file = try Io.Dir.cwd().openFile(p.io, try std.fs.path.join(p.a, &.{ p.dir, name }), .{});
        var head: [8]u8 = undefined;
        if (try file.readPositionalAll(p.io, &head, 0) != 8) return error.BadShard;
        const n = std.mem.readInt(u64, &head, .little);
        if (n > 1 << 30) return error.BadShard;
        const buf = try p.a.alloc(u8, n);
        if (try file.readPositionalAll(p.io, buf, 8) != n) return error.BadShard;
        const header = try std.json.parseFromSliceLeaky(std.json.Value, p.a, buf, .{});
        try p.shards.put(p.a, name, .{ .file = file, .data_at = 8 + n, .header = header });
        return p.shards.getPtr(name).?;
    }

    pub fn has(p: *const Pack, name: []const u8) bool {
        return p.map.contains(name);
    }

    pub fn info(p: *Pack, name: []const u8) !Info {
        const shard_name = p.map.get(name) orelse return error.MissingTensor;
        const s = try p.shard(shard_name);
        const m = (s.header.object.get(name) orelse return error.MissingTensor).object;
        const dtype = std.meta.stringToEnum(Dtype, m.get("dtype").?.string) orelse return error.UnknownDtype;
        const shape_v = m.get("shape").?.array.items;
        const shape = try p.a.alloc(u64, shape_v.len);
        for (shape, shape_v) |*o, v| o.* = @intCast(v.integer);
        const off = m.get("data_offsets").?.array.items;
        return .{ .shard = shard_name, .dtype = dtype, .shape = shape, .begin = s.data_at + @as(u64, @intCast(off[0].integer)), .end = s.data_at + @as(u64, @intCast(off[1].integer)) };
    }

    /// The tensor's bytes as stored, in `a`.
    pub fn raw(p: *Pack, a: Allocator, name: []const u8) ![]u8 {
        const i = try p.info(name);
        const buf = try a.alloc(u8, @intCast(i.end - i.begin));
        const s = p.shards.getPtr(i.shard).?;
        if (try s.file.readPositionalAll(p.io, buf, i.begin) != buf.len) return error.ShortRead;
        return buf;
    }

    /// A float tensor (F32, BF16 or F16) widened to f32, in `a`.
    pub fn f32s(p: *Pack, a: Allocator, name: []const u8) ![]f32 {
        const i = try p.info(name);
        const bytes = try p.raw(a, name);
        defer a.free(bytes);
        const n: usize = @intCast(i.count());
        const out = try a.alloc(f32, n);
        switch (i.dtype) {
            .F32 => for (out, 0..) |*o, k| {
                o.* = @bitCast(std.mem.readInt(u32, bytes[4 * k ..][0..4], .little));
            },
            .BF16 => for (out, 0..) |*o, k| {
                o.* = @bitCast(@as(u32, std.mem.readInt(u16, bytes[2 * k ..][0..2], .little)) << 16);
            },
            .F16 => for (out, 0..) |*o, k| {
                o.* = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[2 * k ..][0..2], .little))));
            },
            else => return error.NotFloat,
        }
        return out;
    }
};
