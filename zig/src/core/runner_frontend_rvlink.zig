//! In-process link of RV32 objects (runtime, cell body) under compile's linker script; null when unsure (ld.lld then).
const std = @import("std");
const Allocator = std.mem.Allocator;

const text_base = 0x40000;
const text_end = 0x80000;
const rodata_base = 0x20000;
const rodata_end = 0x40000;
const ram_base = 0x100;
/// The linker script's ASSERT: RAM leaves 4K for the stack.
const ram_end = 0x7000;

const Kind = enum { text_init, text, rodata, data, bss };

const Input = struct { file: usize, index: usize, kind: Kind, addr: u32 = 0 };

const Object = struct {
    bytes: []const u8,
    sections: []const [40]u8,
    names: []const u8,
    symtab: ?usize = null,
    /// Output address of each section, 0 when it is not placed.
    addrs: []u32,
};

fn word(comptime T: type, b: []const u8, off: usize) ?T {
    if (off > b.len or b.len - off < @sizeOf(T)) return null;
    return std.mem.readInt(T, b[off..][0..@sizeOf(T)], .little);
}

fn field(sh: [40]u8, comptime i: usize) u32 {
    return std.mem.readInt(u32, sh[4 * i ..][0..4], .little);
}

fn cstr(b: []const u8, off: usize) ?[]const u8 {
    if (off >= b.len) return null;
    const end = std.mem.indexOfScalarPos(u8, b, off, 0) orelse return null;
    return b[off..end];
}

fn prefixed(name: []const u8, comptime base: []const u8) bool {
    return std.mem.eql(u8, name, base) or std.mem.startsWith(u8, name, base ++ ".");
}

/// The script's output section for an allocated input section; null when the script would not place it.
fn classify(name: []const u8) ?Kind {
    if (std.mem.eql(u8, name, ".text.init")) return .text_init;
    if (prefixed(name, ".text")) return .text;
    if (prefixed(name, ".rodata") or prefixed(name, ".srodata")) return .rodata;
    if (prefixed(name, ".data") or prefixed(name, ".sdata")) return .data;
    if (prefixed(name, ".bss") or prefixed(name, ".sbss")) return .bss;
    return null;
}

fn parse(a: Allocator, b: []const u8) Allocator.Error!?Object {
    if (b.len < 52 or !std.mem.eql(u8, b[0..7], "\x7fELF\x01\x01\x01")) return null;
    if (word(u16, b, 16) != 1 or word(u16, b, 18) != 243 or word(u32, b, 36) != 0) return null;
    const shoff = word(u32, b, 32) orelse return null;
    const shentsize = word(u16, b, 46) orelse return null;
    const shnum = word(u16, b, 48) orelse return null;
    const shstrndx = word(u16, b, 50) orelse return null;
    if (shentsize != 40 or shnum == 0 or shstrndx >= shnum or shoff > b.len or b.len - shoff < 40 * @as(usize, shnum)) return null;
    const sections = std.mem.bytesAsSlice([40]u8, b[shoff..][0 .. 40 * @as(usize, shnum)]);
    const strs = sections[shstrndx];
    const names_off = field(strs, 4);
    if (names_off > b.len or b.len - names_off < field(strs, 5)) return null;
    var o: Object = .{ .bytes = b, .sections = sections, .names = b[names_off..][0..field(strs, 5)], .addrs = try a.alloc(u32, shnum) };
    @memset(o.addrs, 0);
    for (sections, 0..) |sh, i| {
        const kind = field(sh, 1);
        const flags = field(sh, 2);
        if (kind == 2) {
            if (o.symtab != null) return null;
            o.symtab = i;
        }
        // SHT_REL, SHT_GROUP, SHT_SYMTAB_SHNDX and COMDAT or TLS sections are outside what the compile step emits
        if (kind == 9 or kind == 17 or kind == 18 or flags & (0x200 | 0x400) != 0) return null;
        if (kind != 1 and kind != 8 and flags & 2 != 0) return null;
        if (kind == 1 or kind == 8) if (field(sh, 4) > b.len or (kind == 1 and b.len - field(sh, 4) < field(sh, 5))) return null;
    }
    return o;
}

fn sectionName(o: Object, sh: [40]u8) ?[]const u8 {
    return cstr(o.names, field(sh, 0));
}

/// The three segments' bytes: text, rodata, data (bss is memory size only).
const Image = struct { bytes: [3]std.ArrayList(u8) = @splat(.empty) };

/// ELF32 executable bytes for `objects` (in command-line order), or null.
pub fn link(a: Allocator, objects: []const []const u8) Allocator.Error!?[]u8 {
    var arena_state = std.heap.ArenaAllocator.init(a);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const objs = try arena.alloc(Object, objects.len);
    for (objects, objs) |bytes, *o| o.* = try parse(arena, bytes) orelse return null;
    var inputs: std.ArrayList(Input) = .empty;
    for (objs, 0..) |o, f| for (o.sections, 0..) |sh, i| {
        if (field(sh, 2) & 2 == 0) continue;
        const kind = classify(sectionName(o, sh) orelse return null) orelse return null;
        if ((kind == .bss) != (field(sh, 1) == 8)) return null;
        try inputs.append(arena, .{ .file = f, .index = i, .kind = kind });
    };
    // the script's order: every .text.init first, then each output section's inputs in file and section order
    var out: Image = .{};
    var cursor = [5]u32{ text_base, 0, rodata_base, ram_base, 0 };
    for ([_]Kind{ .text_init, .text, .rodata, .data, .bss }) |kind| {
        const k = @backingInt(kind);
        if (kind == .text) cursor[k] = cursor[0];
        if (kind == .bss) cursor[k] = cursor[3];
        // as ld.lld, an output section starts at the largest alignment of its inputs (.text continues .text.init)
        var widest: u32 = 1;
        for (inputs.items) |in| if (in.kind == kind) {
            const alignment = @max(field(objs[in.file].sections[in.index], 8), 1);
            if (!std.math.isPowerOfTwo(alignment) or alignment > 4096) return null;
            widest = @max(widest, alignment);
        };
        if (kind != .text) cursor[k] = std.mem.alignForward(u32, cursor[k], widest);
        for (inputs.items) |*in| if (in.kind == kind) {
            const sh = objs[in.file].sections[in.index];
            const at = std.mem.alignForward(u32, cursor[k], @max(field(sh, 8), 1));
            const size = field(sh, 5);
            if (size > text_end) return null;
            in.addr = at;
            objs[in.file].addrs[in.index] = at;
            cursor[k] = std.math.add(u32, at, size) catch return null;
        };
    }
    if (cursor[1] > text_end or cursor[2] > rodata_end or cursor[4] > ram_end or cursor[3] > ram_end) return null;
    // section bytes into the three segments, then relocations patched in place
    const segs = [3]struct { first: u32, last: u32 }{
        .{ .first = text_base, .last = cursor[1] },
        .{ .first = rodata_base, .last = cursor[2] },
        .{ .first = ram_base, .last = cursor[3] },
    };
    for (&out.bytes, segs) |*list, seg| try list.appendNTimes(arena, 0, seg.last - seg.first);
    for (inputs.items) |in| {
        if (in.kind == .bss) continue;
        const sh = objs[in.file].sections[in.index];
        const s: usize = switch (in.kind) {
            .text_init, .text => 0,
            .rodata => 1,
            else => 2,
        };
        const off = field(sh, 4);
        @memcpy(out.bytes[s].items[in.addr - segs[s].first ..][0..field(sh, 5)], objs[in.file].bytes[off..][0..field(sh, 5)]);
    }
    var globals: std.StringHashMapUnmanaged(u32) = .empty;
    for (objs) |o| {
        const st = o.symtab orelse continue;
        var it = symbols(o, st) orelse return null;
        while (it.next()) |sym| {
            if (sym.bind == 0 or sym.shndx == 0) continue;
            const value = resolve(o, sym) orelse return null;
            const got = try globals.getOrPut(arena, sym.name);
            if (got.found_existing) return null;
            got.value_ptr.* = value;
        }
    }
    for (objs) |o| for (o.sections) |sh| {
        if (field(sh, 1) != 4) continue;
        const target = field(sh, 7);
        if (target >= o.sections.len or o.symtab == null or field(sh, 6) != o.symtab.?) return null;
        const place = o.addrs[target];
        if (place == 0) {
            if (field(o.sections[target], 2) & 2 != 0) return null;
            continue;
        }
        const s: usize = if (place >= text_base) 0 else if (place >= rodata_base) 1 else 2;
        const seg = out.bytes[s].items;
        const rel_off = field(sh, 4);
        const count = field(sh, 5) / 12;
        if (rel_off > o.bytes.len or o.bytes.len - rel_off < 12 * @as(usize, count)) return null;
        for (0..count) |r| {
            const e = rel_off + 12 * r;
            const offset = word(u32, o.bytes, e).?;
            const info = word(u32, o.bytes, e + 4).?;
            const addend: i32 = @bitCast(word(u32, o.bytes, e + 8).?);
            const sym = symbol(o, o.symtab.?, info >> 8) orelse return null;
            const value = if (sym.shndx == 0) globals.get(sym.name) orelse return null else resolve(o, sym) orelse return null;
            if (offset >= field(o.sections[target], 5)) return null;
            const p = place + offset;
            if (!patch(seg, p - segs[s].first, @truncate(info), value +% @as(u32, @bitCast(addend)), p)) return null;
        }
    };
    const entry = globals.get("_start") orelse return null;
    return try write(a, entry, .{ out.bytes[0].items, out.bytes[1].items, out.bytes[2].items }, cursor[4] - ram_base);
}

/// Whether a relocatable object leaves `symbol_name` undefined (it calls into another object for it).
pub fn references(bytes: []const u8, symbol_name: []const u8) bool {
    var buf: [4096]u8 = undefined;
    var fba = std.heap.FixedBufferAllocator.init(&buf);
    const o = (parse(fba.allocator(), bytes) catch return false) orelse return false;
    var it = symbols(o, o.symtab orelse return false) orelse return false;
    while (it.next()) |sym| if (sym.shndx == 0 and std.mem.eql(u8, sym.name, symbol_name)) return true;
    return false;
}

const Symbol = struct { name: []const u8, value: u32, shndx: u16, bind: u8, kind: u8 };

const Symbols = struct {
    o: Object,
    strings: []const u8,
    table: []const u8,
    at: usize = 1,

    fn next(it: *Symbols) ?Symbol {
        if (it.at >= it.table.len / 16) return null;
        defer it.at += 1;
        return decode(it.table, it.strings, it.at);
    }
};

fn decode(table: []const u8, strings: []const u8, i: usize) ?Symbol {
    const e = 16 * i;
    if (e + 16 > table.len) return null;
    const info = table[e + 12];
    return .{
        .name = cstr(strings, word(u32, table, e).?) orelse return null,
        .value = word(u32, table, e + 4).?,
        .shndx = word(u16, table, e + 14).?,
        .bind = info >> 4,
        .kind = info & 0xf,
    };
}

fn symbols(o: Object, st: usize) ?Symbols {
    const sh = o.sections[st];
    const link_index = field(sh, 6);
    if (link_index >= o.sections.len) return null;
    const strs = o.sections[link_index];
    const off = field(sh, 4);
    const str_off = field(strs, 4);
    if (off > o.bytes.len or o.bytes.len - off < field(sh, 5) or str_off > o.bytes.len or o.bytes.len - str_off < field(strs, 5)) return null;
    return .{ .o = o, .table = o.bytes[off..][0..field(sh, 5)], .strings = o.bytes[str_off..][0..field(strs, 5)] };
}

fn symbol(o: Object, st: usize, i: u32) ?Symbol {
    const it = symbols(o, st) orelse return null;
    return decode(it.table, it.strings, i);
}

/// A defined symbol's address: absolute, or its placed section's address plus its value (COMMON is not placed).
fn resolve(o: Object, sym: Symbol) ?u32 {
    if (sym.shndx == 0xfff1) return sym.value;
    if (sym.shndx == 0 or sym.shndx >= o.sections.len) return null;
    const base = o.addrs[sym.shndx];
    if (base == 0) return null;
    return base + sym.value;
}

/// Applies one RISC-V relocation at segment offset `at` (absolute place `p`); false when unknown or out of range.
pub fn patch(seg: []u8, at: usize, kind: u8, value: u32, p: u32) bool {
    const span: usize = if (kind == 18 or kind == 19) 8 else 4;
    if (at > seg.len or seg.len - at < span) return false;
    const insn = std.mem.readInt(u32, seg[at..][0..4], .little);
    const rel = value -% p;
    const put = struct {
        fn at32(s: []u8, i: usize, v: u32) void {
            std.mem.writeInt(u32, s[i..][0..4], v, .little);
        }
    }.at32;
    switch (kind) {
        1 => put(seg, at, value),
        26 => put(seg, at, (insn & 0xfff) | ((value +% 0x800) & 0xfffff000)),
        27 => put(seg, at, (insn & 0xfffff) | (value << 20)),
        28 => put(seg, at, (insn & 0x1fff07f) | ((value & 0xfe0) << 20) | ((value & 0x1f) << 7)),
        18, 19 => {
            const next = std.mem.readInt(u32, seg[at + 4 ..][0..4], .little);
            put(seg, at, (insn & 0xfff) | ((rel +% 0x800) & 0xfffff000));
            put(seg, at + 4, (next & 0xfffff) | (rel << 20));
        },
        16 => {
            const off: i32 = @bitCast(rel);
            if (off < -4096 or off > 4094 or off & 1 != 0) return false;
            const r = rel;
            put(seg, at, (insn & 0x1fff07f) | (((r >> 12) & 1) << 31) | (((r >> 5) & 0x3f) << 25) | (((r >> 1) & 0xf) << 8) | (((r >> 11) & 1) << 7));
        },
        17 => {
            const off: i32 = @bitCast(rel);
            if (off < -(1 << 20) or off > (1 << 20) - 2 or off & 1 != 0) return false;
            const r = rel;
            put(seg, at, (insn & 0xfff) | (((r >> 20) & 1) << 31) | (((r >> 1) & 0x3ff) << 21) | (((r >> 11) & 1) << 20) | (((r >> 12) & 0xff) << 12));
        },
        else => return false,
    }
    return true;
}

/// Header, three PT_LOAD headers (text R+X, rodata R, data+bss R+W) and the segment bytes; no section headers.
fn write(a: Allocator, entry: u32, segs: [3][]const u8, ram_size: u32) Allocator.Error![]u8 {
    var offsets: [3]u32 = undefined;
    var size: u32 = 52 + 3 * 32;
    for (segs, &offsets) |s, *off| {
        size = std.mem.alignForward(u32, size, 4);
        off.* = size;
        size += @intCast(s.len);
    }
    const b = try a.alloc(u8, size);
    @memset(b, 0);
    @memcpy(b[0..7], "\x7fELF\x01\x01\x01");
    const put16 = struct {
        fn f(x: []u8, i: usize, v: u16) void {
            std.mem.writeInt(u16, x[i..][0..2], v, .little);
        }
    }.f;
    const put32 = struct {
        fn f(x: []u8, i: usize, v: u32) void {
            std.mem.writeInt(u32, x[i..][0..4], v, .little);
        }
    }.f;
    put16(b, 16, 2);
    put16(b, 18, 243);
    put32(b, 20, 1);
    put32(b, 24, entry);
    put32(b, 28, 52);
    put16(b, 40, 52);
    put16(b, 42, 32);
    put16(b, 44, 3);
    const vaddr = [3]u32{ text_base, rodata_base, ram_base };
    const flags = [3]u32{ 5, 4, 6 };
    for (0..3) |i| {
        const ph = 52 + 32 * i;
        put32(b, ph, 1);
        put32(b, ph + 4, offsets[i]);
        put32(b, ph + 8, vaddr[i]);
        put32(b, ph + 12, vaddr[i]);
        put32(b, ph + 16, @intCast(segs[i].len));
        put32(b, ph + 20, if (i == 2) ram_size else @intCast(segs[i].len));
        put32(b, ph + 24, flags[i]);
        put32(b, ph + 28, 4);
        @memcpy(b[offsets[i]..][0..segs[i].len], segs[i]);
    }
    return b;
}
