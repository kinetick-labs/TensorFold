//! Loads an rv32im_zba ELF into a booted circuit as data (upstream elf.program and core.snapshot), never running it.
const std = @import("std");
const model = @import("runner_circuit_model.zig");
const rom = @import("runner_circuit_rom.zig");
const State = @import("runner_circuit_state.zig").State;

const hd = model.hd;
pub const drom_base: u32 = 0x20000;
pub const drom_bytes: u32 = 4 << 15;
pub const ram_bytes: u32 = 4 << 13;
const sp_init: u32 = ram_bytes;
const n_rom_heads = rom.rom_words >> 11;
const n_drom_heads = (drom_bytes / 4) >> 11;

/// The pieces of a static ELF the machine holds: ROM words, DROM bytes, RAM bytes and the entry address.
pub const Image = model.Image;

/// elf.program: PT_LOAD segments (bss zero-filled) into ROM, DROM or RAM; anything else is refused.
pub fn parse(a: std.mem.Allocator, b: []const u8) !Image {
    if (b.len < 52 or !std.mem.eql(u8, b[0..7], "\x7fELF\x01\x01\x01")) return error.ElfHeader;
    const r = std.mem.readInt;
    if (r(u16, b[16..18], .little) != 2 or r(u16, b[18..20], .little) != 243) return error.ElfMachine;
    const entry = r(u32, b[24..28], .little);
    const phoff = r(u32, b[28..32], .little);
    const phentsize = r(u16, b[42..44], .little);
    const phnum = r(u16, b[44..46], .little);
    if (phentsize != 32 or @as(u64, phoff) + @as(u64, phnum) * 32 > b.len) return error.ElfHeader;
    const words = try a.alloc(u32, rom.rom_words);
    errdefer a.free(words);
    @memset(words, 0);
    const drom = try a.alloc(u8, drom_bytes);
    errdefer a.free(drom);
    @memset(drom, 0);
    const ram = try a.alloc(u8, ram_bytes);
    errdefer a.free(ram);
    @memset(ram, 0);
    var drom_len: usize = 0;
    for (0..phnum) |i| {
        const p = b[phoff + 32 * i ..][0..32];
        const typ = r(u32, p[0..4], .little);
        if (typ == 2 or typ == 3) return error.ElfDynamic;
        const off = r(u32, p[4..8], .little);
        const vaddr = r(u32, p[8..12], .little);
        const filesz = r(u32, p[16..20], .little);
        const memsz = r(u32, p[20..24], .little);
        if (typ != 1 or memsz == 0) continue;
        if (filesz > memsz or @as(u64, off) + filesz > b.len) return error.ElfSegment;
        const end = @as(u64, vaddr) + memsz;
        const data = b[off..][0..filesz];
        if (vaddr >= rom.rom_base and end <= rom.rom_base + 4 * @as(u64, rom.rom_words)) {
            const bytes = std.mem.sliceAsBytes(words);
            @memcpy(bytes[vaddr - rom.rom_base ..][0..filesz], data);
        } else if (vaddr >= drom_base and end <= drom_base + @as(u64, drom_bytes)) {
            @memcpy(drom[vaddr - drom_base ..][0..filesz], data);
            drom_len = @max(drom_len, end - drom_base);
        } else if (end <= ram_bytes) {
            @memcpy(ram[vaddr..][0..filesz], data);
        } else return error.ElfSegment;
    }
    for (words) |*w| w.* = std.mem.littleToNative(u32, w.*);
    if (entry % 4 != 0 or entry < rom.rom_base or entry >= rom.rom_base + 4 * rom.rom_words) return error.ElfEntry;
    return .{ .rom = words, .drom = drom, .ram = ram, .entry = entry };
}

pub fn load(s: *State, image_bytes: []const u8) !void {
    const img = try parse(s.gpa, image_bytes);
    defer img.deinit(s.gpa);
    try boot(s, img);
}

fn named(s: *const State, name: []const u8) !u32 {
    for (s.circuit.table_names, 0..) |n, i| if (std.mem.eql(u8, n, name)) return @intCast(i);
    return error.CircuitTableMissing;
}

fn namedFmt(s: *const State, comptime fmt: []const u8, args: anytype) !u32 {
    var buf: [32]u8 = undefined;
    return named(s, try std.fmt.bufPrint(&buf, fmt, args));
}

const Row = [hd]model.Scalar;

fn ballast(s: *const State) Row {
    var row: Row = @splat(0);
    row[0] = s.circuit.program.a_ref;
    row[1] = s.circuit.program.a_ref;
    return row;
}

/// The snapshot replaces a table whole: rows it does not name are absent.
fn clear(s: *State, t: u32) !void {
    try s.own(t);
    @memset(s.tables[t], 0);
    s.present[t] = 0;
}

/// core.snapshot: the machine's boot state written into the circuit's tables (the heads' initial states).
pub fn boot(s: *State, img: Image) !void {
    s.force = true;
    const a_bit = s.circuit.program.a_bit;
    const anchor = (img.entry - rom.rom_base) >> 2;
    var pc = ballast(s);
    for (0..16) |i| pc[2 + i] = if (anchor >> @intCast(i) & 1 == 1) a_bit else 0;
    const pc_t = try named(s, "pc");
    try clear(s, pc_t);
    try s.writeRow(pc_t, 0, &pc);
    for ([_][]const u8{ "count", "j", "vcnt" }) |name| {
        const t = try named(s, name);
        try clear(s, t);
        var r0: Row = @splat(0);
        r0[2] = a_bit;
        try s.writeRow(t, 0, &r0);
        try s.writeRow(t, 1, &ballast(s));
    }
    const cols = try regColumns(s, try named(s, "regs.rd"));
    for ([_][]const u8{ "regs.rs1", "regs.rs2", "regs.rd" }) |name| {
        const t = try named(s, name);
        try clear(s, t);
        for (1..32) |reg| {
            var row = ballast(s);
            const value: u32 = if (reg == 2) sp_init else 0;
            for (cols, 0..) |c, i| if (value >> @intCast(i) & 1 == 1) {
                row[c] = a_bit;
            };
            try s.writeRow(t, @intCast(reg), &row);
        }
    }
    for (0..4) |h| {
        const t = try namedFmt(s, "quo{d}", .{h});
        for (0..2) |x| {
            try clear(s, t + @as(u32, @intCast(x)));
            try s.writeRow(t + @as(u32, @intCast(x)), 0, &ballast(s));
        }
    }
    const pre = try s.gpa.alloc(u32, rom.rom_words);
    defer s.gpa.free(pre);
    for (img.rom, pre, 0..) |w, *p, i| p.* = if (w == 0) 0 else rom.romWord(w, rom.rom_base + 4 * @as(u32, @intCast(i)));
    try byteRows(s, "rom", n_rom_heads, pre);
    const drom_words = try s.gpa.alloc(u32, drom_bytes / 4);
    defer s.gpa.free(drom_words);
    for (drom_words, 0..) |*w, i| w.* = std.mem.readInt(u32, img.drom[4 * i ..][0..4], .little);
    try byteRows(s, "drom", n_drom_heads, drom_words);
    for (1..3) |port| for (0..4) |h| {
        const t = try namedFmt(s, "bn{d}.{d}", .{ port, h });
        for (0..2) |x| {
            try clear(s, t + @as(u32, @intCast(x)));
            for (0..hd) |r| try s.writeRow(t + @as(u32, @intCast(x)), @intCast(r), &ballast(s));
        }
    };
    const write = @import("runner_circuit.zig").writeWord;
    for (0..ram_bytes / 4) |wi| try write(s, @intCast(wi), std.mem.readInt(u32, img.ram[4 * wi ..][0..4], .little));
}

/// core.byte_rows: per head two value heads of 64 row pairs; word wl of a block sits in row (wl >> 4) + 64 * half.
fn byteRows(s: *State, comptime prefix: []const u8, heads: u32, words: []const u32) !void {
    const per = words.len / heads;
    for (0..heads) |h| {
        const t = try namedFmt(s, prefix ++ "{d}", .{h});
        for (0..2) |x| {
            const tt = t + @as(u32, @intCast(x));
            try clear(s, tt);
            const blk = words[h * per + x * (per / 2) ..][0 .. per / 2];
            for (0..hd) |r| try s.writeRow(tt, @intCast(r), &ballast(s));
            for (blk, 0..) |w, wl| {
                const half = wl >> 3 & 1;
                const row = s.row(tt, @intCast((wl >> 4) + 64 * half));
                for (0..4) |byte| row[2 + 32 * half + 4 * (wl & 7) + byte] = @floatFromInt(w >> @intCast(8 * byte) & 0xff);
            }
        }
    }
}

/// core.reg_cols: register bit i is the first value column whose output lands in the i-th distinct x.<port> dim.
fn regColumns(s: *const State, table: u32) ![32]u32 {
    for (s.circuit.batches) |*b| switch (b.*) {
        .ffn => {},
        .heads => |*hs| for (hs.heads) |*head| for (head.value_heads) |*vh| {
            if (vh.table != table) continue;
            var cols: [32]u32 = undefined;
            var dims: [32]u32 = undefined;
            var n: usize = 0;
            for (vh.out_cols, 0..) |c, i| {
                const d = hs.outs[vh.out_first + i].dim;
                if (std.mem.indexOfScalar(u32, dims[0..n], d) != null) continue;
                if (n == 32) return error.CircuitRegisterLayout;
                dims[n] = d;
                cols[n] = c;
                n += 1;
            }
            if (n != 32) return error.CircuitRegisterLayout;
            return cols;
        },
    };
    return error.CircuitTableMissing;
}
