//! Structured output on our engine: host masked draws over one bf16 logits row (copied from upstream cuda_mtp.zig's
//! promote / argmaxRow / maskRow / drawRow, which live in the upstream flashnext module; one file cannot be in both).
const std = @import("std");
const lanes = @import("lanes");
pub const Sampling = lanes.sampling.Sampling;


// ---- sampling ------------------------------------------------------------------------------------------------------

pub fn promote(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

/// A row's first maximum (argmax's choice): its column.
pub fn argmaxRow(row: []const u16) usize {
    var best = promote(row[0]);
    var at: usize = 0;
    for (row[1..], 1..) |b, i| {
        const v = promote(b);
        if (v > best) {
            best = v;
            at = i;
        }
    }
    return at;
}

/// Structured output: every token whose bit is clear in `bits` (token i in word i / 32) scores bf16 -inf.
pub fn maskRow(row: []u16, bits: []const u32) void {
    for (row, 0..) |*v, i| {
        if (i / 32 >= bits.len or (bits[i / 32] >> @intCast(i % 32)) & 1 == 0) v.* = 0xFF80;
    }
}

/// The keyed draw over one bf16 score row, column j being token ids[j] (or j), as lanes.sampling.choose draws.
pub fn drawRow(gpa: std.mem.Allocator, row: []const u16, ids: ?[]const u32, position: u64, sm: ?Sampling) !u32 {
    const id = struct {
        fn of(map: ?[]const u32, col: usize) u64 {
            return if (map) |m| m[col] else col;
        }
    }.of;
    const s = sm orelse return @intCast(id(ids, argmaxRow(row)));
    if (s.temperature <= 0) return @intCast(id(ids, argmaxRow(row)));
    const n = row.len;
    const k = if (s.top_k == 0 or s.top_k >= n) n else @as(usize, s.top_k);
    const vals = try gpa.alloc(f64, k);
    defer gpa.free(vals);
    const keys = try gpa.alloc(u64, k);
    defer gpa.free(keys);
    if (k == n) {
        for (vals, keys, row, 0..) |*v, *key, b, col| {
            v.* = promote(b);
            key.* = id(ids, col);
        }
    } else {
        // the k best columns, best first; a later column with an equal value ranks below (the id order)
        const top_v = try gpa.alloc(f32, k);
        defer gpa.free(top_v);
        const top_c = try gpa.alloc(usize, k);
        defer gpa.free(top_c);
        var held: usize = 0;
        for (row, 0..) |b, col| {
            const v = promote(b);
            if (held == k and !(v > top_v[k - 1])) continue;
            var at: usize = if (held < k) held else k - 1;
            if (held < k) held += 1;
            while (at > 0 and v > top_v[at - 1]) : (at -= 1) {
                top_v[at] = top_v[at - 1];
                top_c[at] = top_c[at - 1];
            }
            top_v[at] = v;
            top_c[at] = col;
        }
        for (vals, keys, top_v, top_c) |*v, *key, tv, c| {
            v.* = tv;
            key.* = id(ids, c);
        }
    }
    return @intCast(try lanes.sampling.choose(gpa, vals, keys, position, s));
}
