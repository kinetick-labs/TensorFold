//! Engine-owned bytes in a reply's content, which no strip may cut; strip points are the format's call openers.
const std = @import("std");
const json = @import("json");
const reply_text = @import("reply_text.zig");
const tool_parse = @import("tool_parse.zig");
const Openers = reply_text.Openers;
const Allocator = std.mem.Allocator;

/// Owned bytes of a reply's text: [lo, hi) from the first owned byte to the last, before any call block.
pub const Owned = struct {
    lo: usize,
    hi: usize,

    pub fn join(x: ?Owned, y: ?Owned) ?Owned {
        const p = x orelse return y;
        const q = y orelse return p;
        return .{ .lo = @min(p.lo, q.lo), .hi = @max(p.hi, q.hi) };
    }

    pub fn shift(x: ?Owned, by: usize) ?Owned {
        const o = x orelse return null;
        return .{ .lo = o.lo + by, .hi = o.hi + by };
    }
};

/// Where `text`'s first call block starts, as the stream's openers `o` or the final parser reads it; null: none.
pub fn callStart(a: Allocator, text: []const u8, o: Openers) Allocator.Error!?usize {
    var first = try tool_parse.firstEnvelope(a, text);
    for (o.calls, 0..) |c, which| {
        var pos: usize = 0;
        while (reply_text.findTag(text, pos, c[0])) |at| {
            if (first != null and at >= first.?) break;
            pos = at + c[0].len;
            if (reply_text.payloadOf(text[pos..], o, which, true) == .call) {
                first = at;
                break;
            }
        }
    }
    return first;
}

/// The owned span of `content` (in `raw`, `pieces` its tokens' bytes) before its first call block: `records`' writes.
pub fn of(a: Allocator, records: []const []const u8, base: usize, pieces: []const []const u8, raw: []const u8, content: []const u8, o: Openers) Allocator.Error!?Owned {
    if (records.len == 0) return null; // each a cell write: {"field":"content","token":T,"landed":N}
    const from = @intFromPtr(content.ptr) -% @intFromPtr(raw.ptr); // past raw.len: content is not raw's bytes
    // each token's first byte in `raw`, which the token bytes must spell exactly (a lossy decode is no map)
    const at = try a.alloc(usize, pieces.len + 1);
    at[0] = 0;
    for (pieces, 0..) |p, i| {
        at[i + 1] = at[i] + p.len;
        if (at[i + 1] > raw.len or !std.mem.eql(u8, raw[at[i]..at[i + 1]], p)) return null;
    }
    if (at[pieces.len] != raw.len) return null;
    const limit = (try callStart(a, content, o)) orelse content.len;
    var span: ?Owned = null;
    for (records) |text| {
        const r = switch (try json.parse(a, text)) {
            .ok => |v| v,
            .err => continue,
        };
        if (!std.mem.eql(u8, r.strField("field") orelse "", "content")) continue;
        const token = std.math.cast(usize, (r.get("token") orelse continue).int64() orelse continue) orelse continue;
        const landed = std.math.cast(usize, (r.get("landed") orelse continue).int64() orelse continue) orelse continue;
        if (token < base or token - base >= pieces.len) continue; // another segment's write
        const lo = at[token - base];
        const hi = at[@min(token - base + landed, pieces.len)];
        if (lo < from or hi - from > limit or hi == lo) continue; // outside the content, in or past a call, or no bytes
        span = Owned.join(span, .{ .lo = lo - from, .hi = hi - from });
    }
    return span;
}

/// ``pyRstrip(s).len``, never below an owned byte (the engine's, so its span stays in the text the client gets).
fn keptEnd(s: []const u8, owned: ?Owned) usize {
    const o = owned orelse return reply_text.pyRstrip(s).len;
    if (o.hi > s.len) return reply_text.pyRstrip(s).len; // a span past `s`: not its offsets
    return @max(reply_text.pyRstrip(s).len, aligned(s, o).hi);
}

/// `o` widened to whole code points of `s`, so a trim never splits one (the bytes it adds are kept, not owned).
fn aligned(s: []const u8, o: Owned) Owned {
    var lo = o.lo;
    var hi = o.hi;
    while (lo > 0 and lo < s.len and s[lo] & 0xc0 == 0x80) lo -= 1;
    while (hi < s.len and s[hi] & 0xc0 == 0x80) hi += 1;
    return .{ .lo = lo, .hi = hi };
}

/// ``pyStrip(s)`` with the engine's owned bytes kept: `owned` is in `s`'s offsets (Owned.shift moves it).
pub fn strip(s: []const u8, owned: ?Owned) []const u8 {
    const o = owned orelse return reply_text.pyStrip(s);
    if (o.hi > s.len) return reply_text.pyStrip(s);
    const lead = s.len - reply_text.pyLstrip(s).len;
    return s[@min(lead, aligned(s, o).lo)..keptEnd(s, o)];
}

/// ``pyRstrip(s)`` with the engine's owned bytes kept.
pub fn rstrip(s: []const u8, owned: ?Owned) []const u8 {
    return s[0..keptEnd(s, owned)];
}

/// ``reply_text.finishedProse`` that keeps the owned bytes, its strip points the openers `o` (none owned: as before).
pub fn finishedProse(a: Allocator, text: []const u8, o: Openers, owned: ?Owned) Allocator.Error![]const u8 {
    if (owned == null) return reply_text.finishedProse(a, text);
    // final parsing has not yet decided whether to strip the reply's content
    if (reply_text.pyLstrip(text).len != text.len) return text[0..0];
    var out: std.ArrayList(u8) = .empty;
    const prose = try reply_text.hideInto(a, &out, text, true, true, o);
    return if (prose.len != text.len) prose[0..keptEnd(prose, owned)] else prose;
}

/// ``reply_text.cutProse`` that keeps the owned bytes, its strip points the openers `o` (none owned: as before).
pub fn cutProse(a: Allocator, text: []const u8, o: Openers, owned: ?Owned) Allocator.Error![]const u8 {
    if (owned == null) return reply_text.cutProse(a, text);
    var out: std.ArrayList(u8) = .empty;
    const prose = try reply_text.hideInto(a, &out, text, false, false, o);
    return if (prose.len != text.len) prose[0..keptEnd(prose, owned)] else prose;
}
