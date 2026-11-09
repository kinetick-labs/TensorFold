//! Qwenputer's pre-decoded ROM word (ref.rom_word): the circuit's view of an RV32 instruction, as data.
const std = @import("std");

pub const rom_base: u32 = 0x40000;
pub const rom_words: u32 = 1 << 16;

/// Uop numbers of the ROM word's first byte (ref.UOPS, then the custom digit-file ops at 64).
pub const Op = enum(u8) {
    illegal,
    lui,
    auipc,
    misa,
    addi,
    slti,
    sltiu,
    xori,
    ori,
    andi,
    slli,
    srli,
    srai,
    add,
    sub,
    sll,
    slt,
    sltu,
    xor,
    srl,
    sra,
    @"or",
    @"and",
    lb,
    lh,
    lw,
    lbu,
    lhu,
    sb,
    sh,
    sw,
    beq,
    bne,
    blt,
    bge,
    bltu,
    bgeu,
    jal,
    jalr,
    ecall,
    ebreak,
    fence,
    mul,
    mulh,
    mulhsu,
    mulhu,
    div,
    divu,
    rem,
    remu,
    sh1add,
    sh2add,
    sh3add,
    bmac = 64,
    bmsb,
    bld,
    bget,
    bprd,
    bscan,
    csrrw = 200,
    csrrs,
    csrrc,
    csrrwi,
    csrrsi,
    csrrci,
};

pub const Inst = struct { op: Op, rd: u32 = 0, rs1: u32 = 0, rs2: u32 = 0, imm: i32 = 0, dd: u32 = 0 };

fn sx(v: u32, comptime bits: u5) i32 {
    const m = v & ((@as(u32, 1) << bits) - 1);
    return if (m >> (bits - 1) == 1) @as(i32, @intCast(m)) - (@as(i32, 1) << bits) else @intCast(m);
}

/// ref.decode: one instruction word to its op and fields; anything unknown is illegal.
pub fn decode(w: u32) Inst {
    const opc = w & 0x7f;
    const rd = w >> 7 & 31;
    const f3 = w >> 12 & 7;
    const rs1 = w >> 15 & 31;
    const rs2 = w >> 20 & 31;
    const f7 = w >> 25;
    const i_imm = sx(w >> 20, 12);
    const s_imm = sx((w >> 25) << 5 | (w >> 7 & 31), 12);
    const b_imm = sx((w >> 31 & 1) << 12 | (w >> 7 & 1) << 11 | (w >> 25 & 63) << 5 | (w >> 8 & 15) << 1, 13);
    const j_imm = sx((w >> 31 & 1) << 20 | (w >> 12 & 0xff) << 12 | (w >> 20 & 1) << 11 | (w >> 21 & 0x3ff) << 1, 21);
    const upper: i32 = @bitCast(w & 0xfffff000);
    switch (opc) {
        0x37 => return .{ .op = .lui, .rd = rd, .imm = upper },
        0x17 => return .{ .op = .auipc, .rd = rd, .imm = upper },
        0x6f => return .{ .op = .jal, .rd = rd, .imm = j_imm },
        0x67 => if (f3 == 0) return .{ .op = .jalr, .rd = rd, .rs1 = rs1, .imm = i_imm },
        0x63 => if (f3 != 2 and f3 != 3) return .{ .op = ([_]Op{ .beq, .bne, .illegal, .illegal, .blt, .bge, .bltu, .bgeu })[f3], .rs1 = rs1, .rs2 = rs2, .imm = b_imm },
        0x03 => if (f3 != 3 and f3 < 6) return .{ .op = ([_]Op{ .lb, .lh, .lw, .illegal, .lbu, .lhu })[f3], .rd = rd, .rs1 = rs1, .imm = i_imm },
        0x23 => if (f3 < 3) return .{ .op = ([_]Op{ .sb, .sh, .sw })[f3], .rs1 = rs1, .rs2 = rs2, .imm = s_imm },
        0x13 => {
            if (f3 == 1 and f7 == 0) return .{ .op = .slli, .rd = rd, .rs1 = rs1, .imm = @intCast(rs2) };
            if (f3 == 5 and (f7 == 0 or f7 == 0x20)) return .{ .op = if (f7 != 0) .srai else .srli, .rd = rd, .rs1 = rs1, .imm = @intCast(rs2) };
            if (f3 != 1 and f3 != 5) return .{ .op = ([_]Op{ .addi, .illegal, .slti, .sltiu, .xori, .illegal, .ori, .andi })[f3], .rd = rd, .rs1 = rs1, .imm = i_imm };
        },
        0x33 => {
            if (f7 == 0) return .{ .op = ([_]Op{ .add, .sll, .slt, .sltu, .xor, .srl, .@"or", .@"and" })[f3], .rd = rd, .rs1 = rs1, .rs2 = rs2 };
            if (f7 == 0x20 and (f3 == 0 or f3 == 5)) return .{ .op = if (f3 == 0) .sub else .sra, .rd = rd, .rs1 = rs1, .rs2 = rs2 };
            if (f7 == 1) return .{ .op = ([_]Op{ .mul, .mulh, .mulhsu, .mulhu, .div, .divu, .rem, .remu })[f3], .rd = rd, .rs1 = rs1, .rs2 = rs2 };
            if (f7 == 0x10 and (f3 == 2 or f3 == 4 or f3 == 6)) return .{ .op = ([_]Op{ .sh1add, .sh2add, .sh3add })[f3 / 2 - 1], .rd = rd, .rs1 = rs1, .rs2 = rs2 };
        },
        0x0b => if (f3 < 5 and f7 == 0) return .{ .op = ([_]Op{ .bmac, .bmsb, .bld, .bget, .bscan })[f3], .rd = rd, .rs1 = rs1, .rs2 = rs2 },
        0x2b => if (f3 == 0) return .{ .op = .bprd, .imm = @intCast(w >> 20 & 0x7f), .dd = rd & 7 },
        else => {},
    }
    if (w == 0x00000073) return .{ .op = .ecall };
    if (opc == 0x73 and f3 != 0 and f3 != 4) return .{ .op = ([_]Op{ .illegal, .csrrw, .csrrs, .csrrc, .illegal, .csrrwi, .csrrsi, .csrrci })[f3], .rd = rd, .rs1 = rs1, .imm = @intCast(w >> 20) };
    if (w == 0x00100073) return .{ .op = .ebreak };
    if (opc == 0x0f) return .{ .op = .fence };
    return .{ .op = .illegal };
}

/// ref.csr_read: a pure read of a stubbed CSR (cycle, time, instret and high halves, mhartid read 0; misa).
fn csrRead(ins: Inst) ?u32 {
    const pure = (ins.op == .csrrs or ins.op == .csrrc or ins.op == .csrrsi or ins.op == .csrrci) and ins.rs1 == 0;
    if (!pure) return null;
    return switch (ins.imm) {
        0xc00, 0xc01, 0xc02, 0xc80, 0xc81, 0xc82, 0xf14 => 0,
        0x301 => 0x40001100,
        else => null,
    };
}

fn pack(op: Op, rd: u32, rs1: u32, rs2: u32, spare: u32) u32 {
    const b0 = @backingInt(op) | (spare & 1) << 7;
    const b1 = rd << 3 | (spare >> 1 & 7);
    const b2 = rs1 << 3 | (spare >> 4 & 7);
    const b3 = rs2 << 3 | (spare >> 7 & 7);
    return b0 | b1 << 8 | b2 << 16 | b3 << 24;
}

fn inRom(t: u32) bool {
    return t % 4 == 0 and t >= rom_base and t < rom_base + 4 * rom_words;
}

/// ref.rom_word: the pre-decoded word of instruction w at pc.
pub fn romWord(w: u32, pc: u32) u32 {
    const ins = decode(w);
    const imm_u: u32 = @bitCast(ins.imm);
    switch (ins.op) {
        .csrrw, .csrrs, .csrrc, .csrrwi, .csrrsi, .csrrci => {
            const v = csrRead(ins) orelse return pack(.illegal, 0, 0, 0, 0);
            return pack(if (v != 0) .misa else .lui, ins.rd, 0, 0, 0);
        },
        .lui, .auipc => {
            const up = (if (ins.op == .lui) imm_u else pc +% imm_u) >> 12 & 0xfffff;
            return pack(ins.op, ins.rd, up & 31, up >> 5 & 31, up >> 10);
        },
        .addi, .slti, .sltiu, .xori, .ori, .andi, .slli, .srli, .srai, .lb, .lh, .lw, .lbu, .lhu, .jalr => {
            const i = imm_u & 0xfff;
            return pack(ins.op, ins.rd, ins.rs1, i & 31, i >> 5);
        },
        .sb, .sh, .sw => {
            const i = imm_u & 0xfff;
            return pack(ins.op, ins.rs2, ins.rs1, i & 31, i >> 5);
        },
        .beq, .bne, .blt, .bge, .bltu, .bgeu => {
            const off: u32 = @as(u32, @bitCast(ins.imm >> 2)) & 0x7ff;
            const bad: u32 = @intFromBool(!inRom(pc +% imm_u));
            return pack(ins.op, off & 31, ins.rs1, ins.rs2, off >> 5 | bad << 6);
        },
        .jal => {
            const t = pc +% imm_u;
            const wi = (t -% rom_base) >> 2 & 0xffff;
            return pack(.jal, ins.rd, wi & 31, wi >> 5 & 31, wi >> 10 | @as(u32, @intFromBool(!inRom(t))) << 6);
        },
        .ecall => return pack(.ecall, 0, 17, 10, 0),
        .bprd => return pack(.bprd, 0, 0, imm_u & 31, imm_u >> 5 | ins.dd << 7),
        .bmac, .bmsb, .bld, .bget, .bscan => return pack(ins.op, ins.rd, ins.rs1, ins.rs2, 0),
        .ebreak, .fence, .illegal => return pack(ins.op, 0, 0, 0, 0),
        else => return pack(ins.op, ins.rd, ins.rs1, ins.rs2, 0),
    }
}

test "rom words match hand-packed upstream encodings" {
    // addi a0, zero, 5 at the ROM base: uop ADDI, rd 10, rs1 0, imm 5
    try std.testing.expectEqual(@as(u32, 4 | 10 << 11 | 0 << 19 | 5 << 27), romWord(0x00500513, rom_base));
    // ecall: RS1 = a7, RS2 = a0
    try std.testing.expectEqual(@as(u32, 39 | 17 << 19 | 10 << 27), romWord(0x00000073, rom_base));
    // jal ra, -4 from the ROM base targets outside the ROM: bad bit S6
    const jal = romWord(0xffdff0ef, rom_base);
    try std.testing.expectEqual(@as(u32, 37), jal & 0x7f);
    try std.testing.expectEqual(@as(u32, 1), jal >> 18 & 1);
    // csrr a0, misa reads MISA; csrr a0, mcycle (unsupported) is illegal
    try std.testing.expectEqual(@as(u32, 3 | 10 << 11), romWord(0x30102573, rom_base));
    try std.testing.expectEqual(@as(u32, 0), romWord(0xb0002573, rom_base));
    try std.testing.expectEqual(Op.sub, decode(0x40b50533).op);
    try std.testing.expectEqual(Op.sh2add, decode(0x20b54533).op);
}
