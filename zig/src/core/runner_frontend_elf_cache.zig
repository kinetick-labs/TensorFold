//! Compiled RV32 ELF images by a hash of their normalized source and compile settings; opaque-input sources only.
const std = @import("std");

pub const Key = [32]u8;

/// Slots kept; a full cache overwrites its oldest entry.
pub const slots = 64;
/// Larger images are never kept.
pub const max_image = 256 * 1024;

const Entry = struct { key: Key, bytes: []u8 };

var mutex: std.Io.Mutex = .init;
var entries: [slots]?Entry = @splat(null);
var next: usize = 0;
var hits: u64 = 0;

/// Opaque inputs (clang cannot fold an answer) and an admission with no AST pass: the source alone decides the image.
pub fn wanted(opaque_inputs: bool, step_budget: usize, estimated_steps: ?usize) bool {
    if (!opaque_inputs) return false;
    if (step_budget == 0) return true;
    const steps = estimated_steps orelse return false;
    return steps <= step_budget;
}

/// Everything besides the fixed runtime that changes the image or its acceptance.
pub fn key(normalized: []const u8, clang: []const u8, linker: ?[]const u8, optimize: u8, stack_limit: usize, collection_stack: bool) Key {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    for ([_][]const u8{ clang, linker orelse "", normalized }) |part| {
        h.update(std.mem.asBytes(&part.len));
        h.update(part);
    }
    h.update(&.{ optimize, @intFromBool(collection_stack) });
    h.update(std.mem.asBytes(&stack_limit));
    h.update(if (linker == null) "lld" else "path");
    var out: Key = undefined;
    h.final(&out);
    return out;
}

/// A copy of the image kept for `k`, owned by the caller, or null.
pub fn get(a: std.mem.Allocator, io: std.Io, k: Key) std.mem.Allocator.Error!?[]u8 {
    mutex.lockUncancelable(io);
    defer mutex.unlock(io);
    for (&entries) |*slot| if (slot.*) |e| if (std.mem.eql(u8, &e.key, &k)) {
        hits += 1;
        return try a.dupe(u8, e.bytes);
    };
    return null;
}

/// Keeps a copy of a validated image; best effort (an allocation failure or an oversized image keeps nothing).
pub fn put(io: std.Io, k: Key, bytes: []const u8) void {
    if (bytes.len > max_image) return;
    const copy = std.heap.page_allocator.dupe(u8, bytes) catch return;
    mutex.lockUncancelable(io);
    defer mutex.unlock(io);
    for (&entries) |*slot| if (slot.*) |e| if (std.mem.eql(u8, &e.key, &k)) {
        std.heap.page_allocator.free(copy);
        return;
    };
    if (entries[next]) |old| std.heap.page_allocator.free(old.bytes);
    entries[next] = .{ .key = k, .bytes = copy };
    next = (next + 1) % slots;
}

/// Lookups answered from the cache since start (or the last reset).
pub fn hitCount(io: std.Io) u64 {
    mutex.lockUncancelable(io);
    defer mutex.unlock(io);
    return hits;
}

/// Empties the cache (tests).
pub fn reset(io: std.Io) void {
    mutex.lockUncancelable(io);
    defer mutex.unlock(io);
    for (&entries) |*slot| if (slot.*) |e| {
        std.heap.page_allocator.free(e.bytes);
        slot.* = null;
    };
    next = 0;
    hits = 0;
}

test "elf cache: only opaque inputs with a known admission, distinct settings never share an image, oldest goes first" {
    const t = std.testing;
    try t.expect(wanted(true, 0, null));
    try t.expect(wanted(true, 100, 100));
    try t.expect(!wanted(true, 100, 101));
    try t.expect(!wanted(true, 100, null));
    try t.expect(!wanted(false, 0, 5));
    reset(t.io);
    defer reset(t.io);
    const k = key("int main(void) {\n}\n", "/usr/bin/clang", null, 1, 0, false);
    for ([_]Key{
        key("int main(void) {\n }\n", "/usr/bin/clang", null, 1, 0, false),
        key("int main(void) {\n}\n", "/usr/bin/clang-18", null, 1, 0, false),
        key("int main(void) {\n}\n", "/usr/bin/clang", "", 1, 0, false),
        key("int main(void) {\n}\n", "/usr/bin/clang", null, 2, 0, false),
        key("int main(void) {\n}\n", "/usr/bin/clang", null, 1, 4096, false),
        key("int main(void) {\n}\n", "/usr/bin/clang", null, 1, 0, true),
    }) |other| try t.expect(!std.mem.eql(u8, &k, &other));
    try t.expectEqual(@as(?[]u8, null), try get(t.allocator, t.io, k));
    put(t.io, k, "image one");
    put(t.io, k, "image two");
    const got = (try get(t.allocator, t.io, k)).?;
    defer t.allocator.free(got);
    try t.expectEqualStrings("image one", got);
    try t.expectEqual(@as(u64, 1), hitCount(t.io));
    const big = try t.allocator.alloc(u8, max_image + 1);
    defer t.allocator.free(big);
    const kb = key("big", "c", null, 1, 0, false);
    put(t.io, kb, big);
    try t.expectEqual(@as(?[]u8, null), try get(t.allocator, t.io, kb));
    for (0..slots) |i| put(t.io, key(std.mem.asBytes(&i), "c", null, 1, 0, false), "filler");
    try t.expectEqual(@as(?[]u8, null), try get(t.allocator, t.io, k));
}
