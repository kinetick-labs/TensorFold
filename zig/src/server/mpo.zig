//! A multi-picture JPEG (MPO: a camera's JPEG with an APP2 "MPF" index and a second picture, often a preview, appended)
//! as its primary picture alone. The vision helper opens images with Pillow, which reads such a file as MPO with
//! n_frames 2 and refuses it as multipage; the primary picture is the photo. Animated PNG and WebP stay refused there.
const std = @import("std");
const Allocator = std.mem.Allocator;

/// The primary picture of a multi-picture JPEG, without its MPF index; null when `jpeg` is no MPO (or is malformed).
pub fn primary(a: Allocator, jpeg: []const u8) Allocator.Error!?[]u8 {
    if (jpeg.len < 4 or jpeg[0] != 0xFF or jpeg[1] != 0xD8) return null;
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    try out.appendSlice(a, jpeg[0..2]);
    var mpf = false;
    var scanning = false; // past an SOS: entropy-coded data follows it and each RSTn
    var i: usize = 2;
    while (true) {
        if (i >= jpeg.len or jpeg[i] != 0xFF) return bail(a, &out);
        while (i < jpeg.len and jpeg[i] == 0xFF) i += 1; // fill bytes before a marker
        if (i >= jpeg.len) return bail(a, &out);
        const m = jpeg[i];
        i += 1;
        if (m == 0xD9) { // the primary picture's EOI: what follows is the index's other pictures
            try out.appendSlice(a, &.{ 0xFF, 0xD9 });
            break;
        }
        if (m == 0x01 or (m >= 0xD0 and m <= 0xD7)) {
            try out.appendSlice(a, &.{ 0xFF, m });
            // a restart marker's scan goes on after it
            if (m != 0x01 and scanning) i = (try scan(a, &out, jpeg, i)) orelse return bail(a, &out);
            continue;
        }
        if (m == 0x00 or m == 0xD8 or i + 2 > jpeg.len) return bail(a, &out);
        const len = std.mem.readInt(u16, jpeg[i..][0..2], .big);
        if (len < 2 or i + len > jpeg.len) return bail(a, &out);
        const body = jpeg[i + 2 .. i + len];
        if (m == 0xE2 and std.mem.startsWith(u8, body, "MPF\x00")) {
            mpf = true; // the index goes: Pillow then reads one plain JPEG
        } else {
            try out.appendSlice(a, &.{ 0xFF, m });
            try out.appendSlice(a, jpeg[i .. i + len]);
        }
        i += len;
        if (m == 0xDA) {
            scanning = true;
            i = (try scan(a, &out, jpeg, i)) orelse return bail(a, &out);
        }
    }
    if (!mpf) return bail(a, &out);
    return try out.toOwnedSlice(a);
}

/// Entropy-coded data from `i` up to the next marker (0xFF00 stuffing and RSTn stay inside) into `out`; null: no end.
fn scan(a: Allocator, out: *std.ArrayList(u8), jpeg: []const u8, from: usize) Allocator.Error!?usize {
    var i = from;
    while (i + 1 < jpeg.len and !(jpeg[i] == 0xFF and jpeg[i + 1] != 0x00 and !(jpeg[i + 1] >= 0xD0 and jpeg[i + 1] <= 0xD7))) i += 1;
    if (i + 1 >= jpeg.len) return null;
    try out.appendSlice(a, jpeg[from..i]);
    return i;
}

fn bail(a: Allocator, out: *std.ArrayList(u8)) ?[]u8 {
    out.deinit(a);
    return null;
}

/// A base64 JPEG data URL of a multi-picture JPEG as a data URL of its primary picture; null: keep `url` as it is.
pub fn dataUrl(a: Allocator, url: []const u8) Allocator.Error!?[]const u8 {
    if (!std.mem.startsWith(u8, url, "data:")) return null;
    const comma = std.mem.indexOfScalar(u8, url, ',') orelse return null;
    const header = url[5..comma];
    if (!std.ascii.endsWithIgnoreCase(header, ";base64")) return null;
    const payload = url[comma + 1 ..];
    // only a JPEG's first bytes ("/9j/" is FF D8 FF) are worth decoding
    if (!std.mem.startsWith(u8, payload, "/9j/")) return null;
    const d = std.base64.standard.Decoder;
    const n = d.calcSizeForSlice(payload) catch return null;
    const bytes = try a.alloc(u8, n);
    defer a.free(bytes);
    d.decode(bytes, payload) catch return null;
    const one = (try primary(a, bytes)) orelse return null;
    defer a.free(one);
    const e = std.base64.standard.Encoder;
    const prefix = "data:image/jpeg;base64,";
    const res = try a.alloc(u8, prefix.len + e.calcSize(one.len));
    @memcpy(res[0..prefix.len], prefix);
    _ = e.encode(res[prefix.len..], one);
    return res;
}

const t = std.testing;

/// A tiny picture's markers: APP1, an optional APP2 MPF index, DQT, SOS with stuffed and RST bytes, EOI.
fn picture(a: Allocator, with_mpf: bool, entropy: []const u8) ![]u8 {
    var b: std.ArrayList(u8) = .empty;
    try b.appendSlice(a, &.{ 0xFF, 0xD8, 0xFF, 0xE1, 0x00, 0x08, 'E', 'x', 'i', 'f', 0, 0 });
    if (with_mpf) try b.appendSlice(a, &.{ 0xFF, 0xE2, 0x00, 0x0A, 'M', 'P', 'F', 0, 'I', 'I', 0x2A, 0 });
    try b.appendSlice(a, &.{ 0xFF, 0xDB, 0x00, 0x04, 0x00, 0x01, 0xFF, 0xDA, 0x00, 0x03, 0x01 });
    try b.appendSlice(a, entropy);
    try b.appendSlice(a, &.{ 0xFF, 0xD9 });
    return b.items;
}

test "image mpo: a multi-picture JPEG keeps its primary picture without the MPF index; the rest is dropped" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const data = [_]u8{ 0x12, 0xFF, 0x00, 0x34, 0xFF, 0xD0, 0x56, 0xFF, 0xFF, 0xD1, 0x78 }; // stuffing, RST0, fill + RST1
    const first = try picture(a, true, &data);
    const preview = try picture(a, false, &.{ 0x9A, 0xBC });
    const mpo = try std.mem.concat(a, u8, &.{ first, preview });
    const got = (try primary(a, mpo)).?;
    const want = try picture(a, false, &.{ 0x12, 0xFF, 0x00, 0x34, 0xFF, 0xD0, 0x56, 0xFF, 0xD1, 0x78 }); // one fill byte goes
    try t.expectEqualSlices(u8, want, got);
    // a plain JPEG, a non-JPEG and a truncated MPO stay as they are
    try t.expect((try primary(a, first[0 .. first.len - 2])) == null);
    try t.expect((try primary(a, preview)) == null);
    try t.expect((try primary(a, "\x89PNG\r\n\x1a\n")) == null);
    try t.expect((try primary(a, mpo[0..20])) == null);
}

test "image mpo: a base64 data URL of a multi-picture JPEG becomes its primary picture's; other URLs stay" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const e = std.base64.standard.Encoder;
    const first = try picture(a, true, &.{0x11});
    const mpo = try std.mem.concat(a, u8, &.{ first, try picture(a, false, &.{0x22}) });
    const enc = try a.alloc(u8, e.calcSize(mpo.len));
    const url = try std.mem.concat(a, u8, &.{ "data:image/jpeg;base64,", e.encode(enc, mpo) });
    const got = (try dataUrl(a, url)).?;
    const one = try picture(a, false, &.{0x11});
    const want_b = try a.alloc(u8, e.calcSize(one.len));
    try t.expectEqualStrings(try std.mem.concat(a, u8, &.{ "data:image/jpeg;base64,", e.encode(want_b, one) }), got);
    const plain = try a.alloc(u8, e.calcSize(one.len));
    try t.expect((try dataUrl(a, try std.mem.concat(a, u8, &.{ "data:image/jpeg;base64,", e.encode(plain, one) }))) == null);
    try t.expect((try dataUrl(a, "data:image/png;base64,iVBORw0KGgo=")) == null);
    try t.expect((try dataUrl(a, "https://example.com/a.jpg")) == null);
    try t.expect((try dataUrl(a, "data:image/jpeg;base64,/9j/!!")) == null);
}
