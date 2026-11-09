//! Offline prepared circuit encoding; lazy loading restores exact runtime fields without compiling graph IR.
const std = @import("std");
const builtin = @import("builtin");
const model = @import("steps_v1/runner_circuit_model.zig");
const binding = @import("runner_graph_lazy.zig");
const Allocator = std.mem.Allocator;
const Hash = std.crypto.hash.sha2.Sha256;
const header = 160;
pub const Header = [header]u8;
const max_bytes = 1 << 30;
const Prepared = struct { name: []const u8, width: u8, circuit: model.Circuit };

fn schema(comptime T: type, h: *Hash) void {
    switch (@typeInfo(T)) {
        .@"struct" => |s| {
            h.update("struct{");
            inline for (s.field_names, s.field_types) |name, Field| {
                if (comptime T == model.Circuit and std.mem.eql(u8, name, "arena")) continue;
                h.update(name);
                schema(Field, h);
            }
            h.update("}");
        },
        .@"union" => |u| {
            h.update("union{");
            schema(u.tag_type.?, h);
            inline for (u.field_names, u.field_types) |name, Field| {
                h.update(name);
                schema(Field, h);
            }
            h.update("}");
        },
        .pointer => |p| {
            h.update(if (p.size == .slice) "slice" else "pointer");
            schema(p.child, h);
        },
        .optional => |o| {
            h.update("optional");
            schema(o.child, h);
        },
        .array => |a| {
            h.update(std.fmt.comptimePrint("array{d}", .{a.len}));
            schema(a.child, h);
        },
        .@"enum" => |e| {
            h.update("enum");
            inline for (e.field_names, e.field_values) |name, value|
                h.update(std.fmt.comptimePrint("{s}={d};", .{ name, value }));
        },
        else => h.update(@typeName(T)),
    }
}

pub fn schemaHash() [32]u8 {
    return versionedSchemaHash(false);
}

fn versionedSchemaHash(legacy: bool) [32]u8 {
    var h = Hash.init(.{});
    h.update(if (legacy) "runner-graph-prepared-v1-rle:" else "runner-graph-prepared-v2-rle-bindings:");
    h.update(builtin.zig_version_string);
    h.update(@embedFile("steps_v1/runner_circuit_model.zig"));
    h.update(@embedFile("steps_v1/runner_circuit_state.zig"));
    h.update(@embedFile("steps_v1/runner_circuit_eval.zig"));
    if (!legacy) h.update(@embedFile("runner_graph_lazy.zig"));
    schema(Prepared, &h);
    return h.finalResult();
}

fn leaf(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .int, .float => true,
        else => false,
    };
}

fn equalBits(comptime T: type, a: T, b: T) bool {
    return std.mem.eql(u8, std.mem.asBytes(&a), std.mem.asBytes(&b));
}

const Encoder = struct {
    a: Allocator,
    bytes: std.ArrayList(u8) = .empty,

    fn putLeaf(e: *Encoder, comptime T: type, values: []const T) anyerror!void {
        var runs: usize = 0;
        for (values, 0..) |v, i| {
            if (i == 0 or !equalBits(T, v, values[i - 1])) runs += 1;
        }
        const compressed = runs * (4 + @sizeOf(T)) < values.len * @sizeOf(T);
        try e.put(u8, @intFromBool(compressed));
        if (!compressed) {
            if (comptime builtin.cpu.arch.endian() == .little) {
                try e.bytes.appendSlice(e.a, std.mem.sliceAsBytes(values));
            } else for (values) |v| try e.put(T, v);
            return;
        }
        var i: usize = 0;
        while (i < values.len) {
            var end = i + 1;
            while (end < values.len and end - i < std.math.maxInt(u32) and equalBits(T, values[i], values[end]))
                end += 1;
            try e.put(u32, @intCast(end - i));
            try e.put(T, values[i]);
            i = end;
        }
    }

    fn put(e: *Encoder, comptime T: type, value: T) anyerror!void {
        switch (@typeInfo(T)) {
            .bool => try e.bytes.append(e.a, @intFromBool(value)),
            .int => {
                var b: [@sizeOf(T)]u8 = undefined;
                std.mem.writeInt(T, &b, value, .little);
                try e.bytes.appendSlice(e.a, &b);
            },
            .float => {
                const U = @Int(.unsigned, @bitSizeOf(T));
                try e.put(U, @bitCast(value));
            },
            .@"enum" => try e.put(u32, @intCast(@backingInt(value))),
            .optional => |o| {
                try e.put(bool, value != null);
                if (value) |v| try e.put(o.child, v);
            },
            .pointer => |p| {
                if (p.size == .slice) {
                    try e.put(u64, @intCast(value.len));
                    if (comptime leaf(p.child)) {
                        try e.putLeaf(p.child, value);
                    } else for (value) |v| try e.put(p.child, v);
                } else if (p.size == .one) try e.put(p.child, value.*) else @compileError("unsupported pointer");
            },
            .array => |ar| for (value) |v| try e.put(ar.child, v),
            .@"struct" => |s| inline for (s.field_names, s.field_types) |name, Field| {
                if (comptime T == model.Circuit and std.mem.eql(u8, name, "arena")) continue;
                try e.put(Field, @field(value, name));
            },
            .@"union" => |u| {
                const tag = std.meta.activeTag(value);
                try e.put(u.tag_type.?, tag);
                inline for (u.field_names, u.field_types) |name, Field| if (std.mem.eql(u8, name, @tagName(tag))) {
                    try e.put(Field, @field(value, name));
                };
            },
            else => @compileError("unsupported prepared field: " ++ @typeName(T)),
        }
    }
};

const Decoder = struct {
    a: Allocator,
    bytes: []const u8,
    at: usize = 0,
    allocated: usize = 0,

    fn take(d: *Decoder, n: usize) ![]const u8 {
        if (n > d.bytes.len - d.at) return error.GraphPackTruncated;
        defer d.at += n;
        return d.bytes[d.at..][0..n];
    }

    fn getLeaf(d: *Decoder, comptime T: type, out: []T) anyerror!void {
        const mode = try d.get(u8);
        if (mode == 0) {
            if (comptime builtin.cpu.arch.endian() == .little) {
                @memcpy(std.mem.sliceAsBytes(out), try d.take(out.len * @sizeOf(T)));
            } else for (out) |*v| v.* = try d.get(T);
        } else if (mode == 1) {
            var i: usize = 0;
            while (i < out.len) {
                const n = try d.get(u32);
                if (n == 0 or n > out.len - i) return error.GraphPackRun;
                @memset(out[i..][0..n], try d.get(T));
                i += n;
            }
        } else return error.GraphPackEncoding;
    }

    fn get(d: *Decoder, comptime T: type) anyerror!T {
        return switch (@typeInfo(T)) {
            .bool => switch ((try d.take(1))[0]) {
                0 => false,
                1 => true,
                else => error.GraphPackBoolean,
            },
            .int => std.mem.readInt(T, (try d.take(@sizeOf(T)))[0..@sizeOf(T)], .little),
            .float => @bitCast(try d.get(@Int(.unsigned, @bitSizeOf(T)))),
            .@"enum" => std.enums.fromInt(T, try d.get(u32)) orelse return error.GraphPackTag,
            .optional => |o| if (try d.get(bool)) try d.get(o.child) else null,
            .pointer => |p| blk: {
                const n: usize = if (p.size == .slice) std.math.cast(usize, try d.get(u64)) orelse
                    return error.GraphPackSize else 1;
                if (n > max_bytes / @max(1, @sizeOf(p.child))) return error.GraphPackSize;
                const size = n * @sizeOf(p.child);
                if (size > max_bytes - d.allocated) return error.GraphPackSize;
                d.allocated += size;
                if (p.size == .slice) {
                    const out = try d.a.alloc(p.child, n);
                    if (comptime leaf(p.child)) {
                        try d.getLeaf(p.child, out);
                    } else for (out) |*v| v.* = try d.get(p.child);
                    break :blk out;
                } else if (p.size == .one) {
                    const out = try d.a.create(p.child);
                    out.* = try d.get(p.child);
                    break :blk out;
                } else @compileError("unsupported pointer");
            },
            .array => |ar| blk: {
                var out: T = undefined;
                for (&out) |*v| v.* = try d.get(ar.child);
                break :blk out;
            },
            .@"struct" => |s| blk: {
                var out: T = undefined;
                inline for (s.field_names, s.field_types) |name, Field| {
                    if (comptime T == model.Circuit and std.mem.eql(u8, name, "arena")) continue;
                    @field(out, name) = try d.get(Field);
                }
                break :blk out;
            },
            .@"union" => |u| blk: {
                const tag = try d.get(u.tag_type.?);
                inline for (u.field_names, u.field_types) |name, Field| if (std.mem.eql(u8, name, @tagName(tag))) {
                    break :blk @unionInit(T, name, try d.get(Field));
                };
                return error.GraphPackTag;
            },
            else => @compileError("unsupported prepared field: " ++ @typeName(T)),
        };
    }
};

pub fn encode(a: Allocator, c: model.Circuit, name: []const u8, width: u16) ![]u8 {
    if (name.len == 0) return error.GraphPackIdentity;
    const code = try encodedWidth(width);
    const layout = try binding.Binding.inspect(&c, try bindingWidth(width));
    var e: Encoder = .{ .a = a };
    errdefer e.bytes.deinit(a);
    try e.bytes.appendNTimes(a, 0, header);
    try e.put(Prepared, .{ .name = name, .width = code, .circuit = c });
    @memcpy(e.bytes.items[0..8], "TFGRPH02");
    std.mem.writeInt(u64, e.bytes.items[8..16], e.bytes.items.len - header, .little);
    @memcpy(e.bytes.items[16..48], &schemaHash());
    Hash.hash(e.bytes.items[header..], e.bytes.items[48..80], .{});
    e.bytes.items[80] = layout.inputs;
    e.bytes.items[81] = layout.outputs;
    e.bytes.items[82] = code;
    Hash.hash(name, e.bytes.items[96..128], .{});
    Hash.hash(e.bytes.items[0..128], e.bytes.items[128..160], .{});
    return e.bytes.toOwnedSlice(a);
}

pub fn decode(a: Allocator, bytes: []const u8, name: []const u8, width: u16) !model.Circuit {
    if (bytes.len < 80) return error.GraphPackTruncated;
    var h: Header = @splat(0);
    @memcpy(h[0..80], bytes[0..80]);
    const legacy = isLegacy(&h);
    const offset: usize = if (legacy) 80 else header;
    if (bytes.len < offset) return error.GraphPackTruncated;
    @memcpy(h[0..offset], bytes[0..offset]);
    try validateHeader(&h, bytes.len, name, width, schemaHash());
    var digest: [32]u8 = undefined;
    Hash.hash(bytes[offset..], &digest, .{});
    if (!std.mem.eql(u8, bytes[48..80], &digest)) return error.GraphPackChecksum;
    var arena = std.heap.ArenaAllocator.init(a);
    errdefer arena.deinit();
    var d: Decoder = .{ .a = arena.allocator(), .bytes = bytes[offset..] };
    const prepared = try d.get(Prepared);
    if (prepared.width != (try encodedWidth(width)) or !std.mem.eql(u8, prepared.name, name)) return error.GraphPackIdentity;
    var out = prepared.circuit;
    if (d.at != d.bytes.len) return error.GraphPackTrailing;
    const layout = try binding.Binding.inspect(&out, try bindingWidth(width));
    if (!legacy and (layout.inputs != h[80] or layout.outputs != h[81])) return error.GraphSlotArity;
    out.arena = arena;
    return out;
}

fn validateHeader(h: *const Header, size: u64, name: []const u8, width: u16, expected_schema: [32]u8) !void {
    const code = try encodedWidth(width);
    const legacy = isLegacy(h);
    if (!legacy and !std.mem.eql(u8, h[0..8], "TFGRPH02")) return error.GraphPackMagic;
    const offset: u64 = if (legacy) 80 else header;
    if (size < offset or size > max_bytes or std.mem.readInt(u64, h[8..16], .little) != size - offset)
        return error.GraphPackSize;
    const expected = if (legacy) versionedSchemaHash(true) else expected_schema;
    if (!std.mem.eql(u8, h[16..48], &expected)) return error.GraphPackSchema;
    if (legacy) return;
    var digest: [32]u8 = undefined;
    Hash.hash(h[0..128], &digest, .{});
    if (!std.mem.eql(u8, h[128..160], &digest)) return error.GraphPackChecksum;
    Hash.hash(name, &digest, .{});
    if (name.len == 0 or code != h[82] or !std.mem.eql(u8, h[96..128], &digest))
        return error.GraphPackIdentity;
    if (h[80] == 0 or h[80] > 16 or h[81] == 0 or h[81] > 16) return error.GraphSlotArity;
    for (h[83..96]) |v| if (v != 0) return error.GraphPackEncoding;
}

pub fn isLegacy(h: *const Header) bool {
    return std.mem.eql(u8, h[0..8], "TFGRPH01");
}

/// Reads only the checksummed descriptor header; payload verification is mandatory before first execution.
pub fn inspect(path: []const u8, name: []const u8, width: u16, expected_schema: [32]u8) !Header {
    var threaded: std.Io.Threaded = .init_single_threaded;
    const io = threaded.io();
    const file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    const size = (try file.stat(io)).size;
    var h: Header = @splat(0);
    if (try file.readPositionalAll(io, h[0..80], 0) != 80) return error.GraphPackTruncated;
    if (!isLegacy(&h) and (try file.readPositionalAll(io, h[80..], 80)) != header - 80)
        return error.GraphPackTruncated;
    try validateHeader(&h, size, name, width, expected_schema);
    return h;
}

/// Pins deferred loading to the exact header accepted at bank boot, including its payload digest.
pub fn loadPinned(a: Allocator, path: []const u8, name: []const u8, width: u16, expected: *const Header) !model.Circuit {
    var threaded: std.Io.Threaded = .init_single_threaded;
    const bytes = try std.Io.Dir.cwd().readFileAlloc(threaded.io(), path, a, .limited(max_bytes));
    defer a.free(bytes);
    const offset: usize = if (isLegacy(expected)) 80 else header;
    if (bytes.len < offset or !std.mem.eql(u8, bytes[0..offset], expected[0..offset]))
        return error.GraphPackChanged;
    return decode(a, bytes, name, width);
}

pub fn load(a: Allocator, path: []const u8, name: []const u8, width: u16) !model.Circuit {
    var threaded: std.Io.Threaded = .init_single_threaded;
    const bytes = try std.Io.Dir.cwd().readFileAlloc(threaded.io(), path, a, .limited(max_bytes));
    defer a.free(bytes);
    return decode(a, bytes, name, width);
}

// The deployed schema keeps Prepared.width as u8; zero was never a valid old width.
fn encodedWidth(width: u16) !u8 {
    if (!@import("runner_graph_packet.zig").widthValid(width) or width == 255) return error.GraphPackIdentity;
    return if (width == 256) 0 else @intCast(width);
}

/// Slot inspection needs only the same nine group heads; numeric bounds always retain width 256.
pub fn bindingWidth(width: u16) !u8 {
    _ = try encodedWidth(width);
    return if (width == 256) 255 else @intCast(width);
}

test "prepared width extension retains byte codes and reserves zero for 256" {
    const t = std.testing;
    for ([_]u16{ 1, 2, 4, 8, 16, 32, 62, 128 }) |width| {
        try t.expectEqual(@as(u8, @intCast(width)), try encodedWidth(width));
        try t.expectEqual(@as(u8, @intCast(width)), try bindingWidth(width));
    }
    try t.expectEqual(@as(u8, 0), try encodedWidth(256));
    try t.expectEqual(@as(u8, 255), try bindingWidth(256));
    for ([_]u16{ 0, 129, 191, 255, 257 }) |width| {
        try t.expectError(error.GraphPackIdentity, encodedWidth(width));
        try t.expectError(error.GraphPackIdentity, bindingWidth(width));
    }
}
