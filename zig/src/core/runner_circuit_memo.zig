//! A unit's results by exact input: a hit returns the words a pure computation produced from the same key before.
const std = @import("std");

pub const slots_per_unit = 256;
/// Misses before a unit stores anything: a short-lived machine's one-off runs never pay for slots and copies.
pub const admit = 4;

/// One entry: its hash beside one block holding the key, then the words (two per u64), so a hit reads two lines.
const Slot = struct {
    hash: u64 = 0,
    data: [*]u64 = undefined,
    cap: u32 = 0,
    key_len: u32 = 0,
    words_len: u32 = 0,
    used: bool = false,
};

/// Direct-mapped by key hash; a colliding put replaces the older entry, so a lookup only ever misses, never lies.
pub const Memo = struct {
    slots: []Slot = &.{},
    hits: u64 = 0,
    misses: u64 = 0,

    pub fn deinit(m: *Memo, gpa: std.mem.Allocator) void {
        for (m.slots) |e| if (e.cap > 0) gpa.free(e.data[0..e.cap]);
        gpa.free(m.slots);
        m.* = .{};
    }

    /// A unit that almost never repeats an input stops paying for lookups and copies; results are the same.
    pub fn off(m: *const Memo) bool {
        return m.misses >= 256 and m.hits * 32 < m.misses;
    }

    /// Counts a run that skips the lookup while the unit has missed fewer than `admit` times.
    pub fn cold(m: *Memo) bool {
        if (m.misses >= admit) return false;
        m.misses += 1;
        return true;
    }

    pub fn hash(key: []const u64) u64 {
        return std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(key));
    }

    pub fn find(m: *Memo, key: []const u64, h: u64) ?[]const u32 {
        if (m.slots.len == 0) {
            m.misses += 1;
            return null;
        }
        const e = &m.slots[h % m.slots.len];
        if (!e.used or e.hash != h or e.key_len != key.len or !std.mem.eql(u64, e.data[0..key.len], key)) {
            m.misses += 1;
            return null;
        }
        m.hits += 1;
        const words: [*]const u32 = @ptrCast(e.data + key.len);
        return words[0..e.words_len];
    }

    pub fn put(m: *Memo, gpa: std.mem.Allocator, key: []const u64, h: u64, words: []const u32) !void {
        if (m.misses < admit) return;
        if (m.slots.len == 0) {
            m.slots = try gpa.alloc(Slot, slots_per_unit);
            @memset(m.slots, .{});
        }
        const e = &m.slots[h % m.slots.len];
        e.used = false;
        const need: u32 = @intCast(key.len + (words.len + 1) / 2);
        if (need > e.cap) {
            if (e.cap > 0) gpa.free(e.data[0..e.cap]);
            e.cap = 0;
            e.data = (try gpa.alloc(u64, need)).ptr;
            e.cap = need;
        }
        @memcpy(e.data[0..key.len], key);
        const dst: [*]u32 = @ptrCast(e.data + key.len);
        @memcpy(dst[0..words.len], words);
        e.key_len = @intCast(key.len);
        e.words_len = @intCast(words.len);
        e.hash = h;
        e.used = true;
    }

    /// Bytes held by the entries, for admission.
    pub fn bytes(m: *const Memo) usize {
        var n = m.slots.len * @sizeOf(Slot);
        for (m.slots) |e| n += e.cap * 8;
        return n;
    }
};

test "a memo stores nothing before admission, then returns only what was stored under the same key" {
    const gpa = std.testing.allocator;
    var m: Memo = .{};
    defer m.deinit(gpa);
    const k1 = [_]u64{ 1, 2 };
    const k2 = [_]u64{ 1, 3 };
    for (0..admit) |_| try std.testing.expect(m.find(&k1, Memo.hash(&k1)) == null);
    try std.testing.expect(m.slots.len == 0);
    try m.put(gpa, &k1, Memo.hash(&k1), &.{ 7, 8, 9 });
    try std.testing.expectEqualSlices(u32, &.{ 7, 8, 9 }, m.find(&k1, Memo.hash(&k1)).?);
    try std.testing.expect(m.find(&k2, Memo.hash(&k1)) == null);
    try std.testing.expect(m.find(&k2, Memo.hash(&k2)) == null);
    try std.testing.expect(m.find(k1[0..1], Memo.hash(&k1)) == null);
    try m.put(gpa, &k2, Memo.hash(&k1), &.{4});
    try std.testing.expectEqualSlices(u32, &.{4}, m.find(&k2, Memo.hash(&k1)).?);
    try std.testing.expect(m.find(&k1, Memo.hash(&k1)) == null);
}
