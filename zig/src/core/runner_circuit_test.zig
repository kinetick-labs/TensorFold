//! T2 host tests: the circuit container, the exact evaluator and RAM op, the machine and the CPU backend.
const std = @import("std");
const api = @import("runner_api.zig");
const model = @import("runner_circuit_model.zig");
const load = @import("runner_circuit_load.zig");
const eval = @import("runner_circuit_eval.zig");
const rc = @import("runner_circuit.zig");
const tiny = @import("runner_circuit_tiny_test.zig");
const elf = @import("runner_circuit_elf.zig");

const gpa = std.testing.allocator;
const tok = tiny.token;
const long_input: [256]u8 = @splat('9');

test {
    _ = model;
    _ = @import("runner_circuit_golden_test.zig");
    _ = @import("runner_circuit_backend.zig");
    _ = @import("runner_circuit_rom.zig");
    _ = @import("runner_circuit_memo.zig");
}

fn tinyCircuit(variant: tiny.Variant) !model.Circuit {
    const bytes = try tiny.build(gpa, variant);
    defer gpa.free(bytes);
    return load.fromBytes(gpa, bytes);
}

test "the container refuses a bad magic, version, flag, length or checksum" {
    const good = try tiny.build(gpa, .good);
    defer gpa.free(good);
    var c = try load.fromBytes(gpa, good);
    c.deinit(gpa);
    const Case = struct { at: usize, want: anyerror };
    for ([_]Case{ .{ .at = 0, .want = error.CircuitMagic }, .{ .at = 8, .want = error.CircuitVersion }, .{ .at = 24, .want = error.CircuitFlags }, .{ .at = 16, .want = error.CircuitTruncated }, .{ .at = 40, .want = error.CircuitChecksum }, .{ .at = 100, .want = error.CircuitChecksum } }) |case| {
        const bad = try gpa.dupe(u8, good);
        defer gpa.free(bad);
        bad[case.at] ^= 1;
        try std.testing.expectError(case.want, load.fromBytes(gpa, bad));
    }
    try std.testing.expectError(error.CircuitTruncated, load.fromBytes(gpa, good[0 .. good.len - 1]));
    try std.testing.expectError(error.CircuitTruncated, load.fromBytes(gpa, good[0..63]));
}

/// S + k (v - S^T k)^T for a 0/1 key, computed densely in binary64.
fn deltaRule(before: []const model.Scalar, key: []const f64, v: []const f64, after: []f64) void {
    const n = model.hd;
    for (0..n) |c| {
        var sk: f64 = 0;
        for (0..n) |j| sk += before[j * n + c] * key[j];
        for (0..n) |i| after[i * n + c] = before[i * n + c] + key[i] * (v[c] - sk);
    }
}

test "a one-hot write is the gated delta rule at beta 1, and reads return the row through the ballast norm" {
    var c = try tinyCircuit(.good);
    defer c.deinit(gpa);
    var s = try rc.State.init(gpa, &c);
    defer s.deinit();
    const ram = c.program.ram_tables[0];
    const before = try gpa.dupe(model.Scalar, s.tables[ram]);
    defer gpa.free(before);
    _ = try eval.step(&s, tok(2, 1 | 4));
    var key: [model.hd]f64 = @splat(0);
    key[1] = 1;
    var v: [model.hd]f64 = @splat(0);
    v[0] = 1 << 24;
    v[1] = 1 << 24;
    v[2] = 128;
    const want = try gpa.alloc(f64, model.hd * model.hd);
    defer gpa.free(want);
    deltaRule(before, &key, &v, want);
    for (want, s.tables[ram]) |a, b| try std.testing.expectEqual(a, @as(f64, b));
    try std.testing.expect(s.rowPresent(ram, 1));
    const Read = struct { row: u32, bit: u32 };
    for ([_]Read{ .{ .row = 1, .bit = 1 }, .{ .row = 2, .bit = 0 }, .{ .row = 3, .bit = 1 } }) |r| {
        try std.testing.expectEqual(tok(4, '0' + r.bit), try eval.step(&s, tok(3, r.row)));
        try std.testing.expectEqual(@as(model.Scalar, @floatFromInt(r.bit)), s.value(tiny.dims.rd));
    }
    _ = try eval.step(&s, tok(2, 3));
    try std.testing.expectEqual(tok(4, '0'), try eval.step(&s, tok(3, 3)));
    try std.testing.expect(s.rowPresent(ram, 3));
}

test "a write gate of 0 leaves the table untouched and a gate of 1 writes" {
    var c = try tinyCircuit(.beta_gated);
    defer c.deinit(gpa);
    var s = try rc.State.init(gpa, &c);
    defer s.deinit();
    const ram = c.program.ram_tables[0];
    const before = try gpa.dupe(model.Scalar, s.tables[ram]);
    defer gpa.free(before);
    _ = try eval.step(&s, tok(2, 1 | 4));
    try std.testing.expect(!s.rowPresent(ram, 1));
    try std.testing.expectEqualSlices(model.Scalar, before, s.tables[ram]);
    _ = try eval.step(&s, tok(2, 2));
    try std.testing.expect(s.rowPresent(ram, 2));
    try std.testing.expectEqual(tok(4, '0'), try eval.step(&s, tok(3, 2)));
    try std.testing.expectEqual(tok(4, '0'), try eval.step(&s, tok(3, 1)));
}

test "a key with two positive rows is refused, and an unmapped output code fails instead of guessing" {
    var two = try tinyCircuit(.two_hot_key);
    defer two.deinit(gpa);
    var s = try rc.State.init(gpa, &two);
    defer s.deinit();
    try std.testing.expectError(error.CircuitKeyNotOneHot, eval.step(&s, tok(2, 1)));
    var unmapped = try tinyCircuit(.unmapped_go);
    defer unmapped.deinit(gpa);
    var u = try rc.State.init(gpa, &unmapped);
    defer u.deinit();
    try std.testing.expectError(error.CircuitOutputCode, eval.step(&u, unmapped.go));
    try std.testing.expectEqual(tok(1, 0), try eval.step(&u, tok(1, 0)));
}

test "a lag-1 tap keys on the previous token, a clear decay empties the table, history is per channel" {
    var c = try tinyCircuit(.good);
    defer c.deinit(gpa);
    var s = try rc.State.init(gpa, &c);
    defer s.deinit();
    const Step = struct { t: u32, lag: model.Scalar };
    const seq = [_]Step{ .{ .t = tok(1, 0), .lag = 0 }, .{ .t = tok(2, 0), .lag = 0 }, .{ .t = tok(1, 0), .lag = 1 }, .{ .t = tok(1, 0), .lag = 1 }, .{ .t = tok(4, 'h'), .lag = 0 }, .{ .t = tok(1, 0), .lag = 0 } };
    for (seq) |st| {
        _ = try eval.step(&s, st.t);
        try std.testing.expectEqual(st.lag, s.value(tiny.dims.lagout));
    }
    try std.testing.expectEqual(@as(u128, 0), s.present[0]);
    try std.testing.expectEqual(@as(u128, 1), s.present[1]);
}

test "a lag-3 tap sees the projection three steps back, not a newer one" {
    var c = try tinyCircuit(.good);
    defer c.deinit(gpa);
    var s = try rc.State.init(gpa, &c);
    defer s.deinit();
    const seq = [_]u32{ tok(2, 0), tok(1, 0), tok(1, 0), tok(1, 0), tok(1, 0) };
    const want = [_]model.Scalar{ 0, 0, 0, 1, 1 };
    for (seq, want) |t, w| {
        _ = try eval.step(&s, t);
        try std.testing.expectEqual(w, s.value(tiny.dims.lag3out));
    }
}

test "skipping units whose inputs did not change gives the tokens and snapshots of running every unit" {
    var c = try tinyCircuit(.good);
    defer c.deinit(gpa);
    var lazy = try rc.State.init(gpa, &c);
    defer lazy.deinit();
    var full = try rc.State.init(gpa, &c);
    defer full.deinit();
    full.always = true;
    const seq = [_]u32{ c.go, tok(2, 1 | 4), tok(2, 1 | 4), tok(1, 0), tok(1, 0), tok(1, 0), tok(1, 0), tok(3, 1), tok(3, 1), tok(2, 2), tok(3, 2), tok(4, 'h'), tok(4, 'h'), tok(1, 0), tok(2, 0), tok(2, 0), tok(1, 0), tok(1, 0), tok(1, 0), tok(3, 3), tok(1, 0) };
    var skipped: usize = 0;
    for (seq) |t| {
        try std.testing.expectEqual(try eval.step(&full, t), try eval.step(&lazy, t));
        for (lazy.dirty, lazy.changed) |d, ch| skipped += @intFromBool(d != lazy.steps and ch != lazy.steps);
        const want = try full.canonical(gpa);
        defer gpa.free(want);
        const got = try lazy.canonical(gpa);
        defer gpa.free(got);
        try std.testing.expectEqualSlices(u8, want, got);
    }
    try std.testing.expect(skipped > 0);
}

test "a halving head reruns on unchanged inputs: its table halves every step, as running every unit does" {
    var c = try tinyCircuit(.halving);
    defer c.deinit(gpa);
    var lazy = try rc.State.init(gpa, &c);
    defer lazy.deinit();
    var full = try rc.State.init(gpa, &c);
    defer full.deinit();
    full.always = true;
    const seq = [_]u32{ tok(2, 1 | 4), tok(1, 0), tok(1, 0), tok(1, 0), tok(1, 0) };
    for (seq) |t| {
        try std.testing.expectEqual(try eval.step(&full, t), try eval.step(&lazy, t));
        const want = try full.canonical(gpa);
        defer gpa.free(want);
        const got = try lazy.canonical(gpa);
        defer gpa.free(got);
        try std.testing.expectEqualSlices(u8, want, got);
    }
    try std.testing.expectEqual(@as(model.Scalar, 16777216.0 / 16.0), lazy.row(3, 1)[0]);
}

test "an into chain no unit reads folds its links in order at its end as each link changes" {
    var c = try tinyCircuit(.chained);
    defer c.deinit(gpa);
    const end = c.flow.final[tiny.dims.sum];
    try std.testing.expectEqual(@as(u32, 3), c.flow.chain_len[end]);
    var s = try rc.State.init(gpa, &c);
    defer s.deinit();
    const seq = [_]u32{ tok(1, 0), tok(2, 0), tok(3, 0), tok(4, 'h'), tok(1, 0), tok(1, 0) };
    const want = [_]model.Scalar{ 1, 2, 4, 0, 1, 1 };
    for (seq, want) |t, w| {
        _ = try eval.step(&s, t);
        try std.testing.expectEqual(w, s.value(tiny.dims.sum));
    }
}

test "an FFN part that is not zero at rest runs on a fresh machine's first step and keeps its value after a forced step" {
    var c = try tinyCircuit(.resting);
    defer c.deinit(gpa);
    var quiet: usize = 0;
    for (c.flow.units) |u| quiet += @intFromBool(u.quiet);
    try std.testing.expectEqual(c.flow.ffn_plans.len - 1, quiet);
    var s = try rc.State.init(gpa, &c);
    defer s.deinit();
    for ([_]u32{ c.go, tok(1, 0), tok(2, 1 | 4), tok(1, 0) }, 0..) |t, i| {
        if (i == 2) try rc.writeWord(&s, 0, 5);
        _ = try eval.step(&s, t);
        try std.testing.expectEqual(@as(model.Scalar, 6), s.value(tiny.dims.sum));
    }
}

test "banks sharing an address share one selection key per step but keep their own rows, as running every unit does" {
    var c = try tinyCircuit(.banks);
    defer c.deinit(gpa);
    try std.testing.expectEqual(@as(u32, 1), c.flow.groups);
    var lazy = try rc.State.init(gpa, &c);
    defer lazy.deinit();
    var full = try rc.State.init(gpa, &c);
    defer full.deinit();
    full.always = true;
    const rows = [_]u32{ 1, 2, 1, 3, 0, 2, 1, 1, 2, 0, 3, 1, 2, 2, 1 };
    for (0..3) |_| for (rows, 0..) |r, i| {
        const t = if (i % 3 == 0) tok(2, r | 4) else tok(3, r);
        try std.testing.expectEqual(try eval.step(&full, t), try eval.step(&lazy, t));
        const want = try full.canonical(gpa);
        defer gpa.free(want);
        const got = try lazy.canonical(gpa);
        defer gpa.free(got);
        try std.testing.expectEqualSlices(u8, want, got);
    };
    try std.testing.expect(lazy.groups[0].memo.hits > 0);
}

test "a bank skipped on a step its group selects nothing still sees that step's inputs one step later" {
    var c = try tinyCircuit(.banks);
    defer c.deinit(gpa);
    var lazy = try rc.State.init(gpa, &c);
    defer lazy.deinit();
    var full = try rc.State.init(gpa, &c);
    defer full.deinit();
    full.always = true;
    // An A token selects no row in either bank: from the second cycle the mirror skips it, then a write reads it back.
    for (0..3) |_| for ([_]u32{ tok(2, 1 | 4), tok(1, 0), tok(2, 2), tok(3, 2), tok(3, 1), tok(1, 0), tok(2, 3), tok(3, 3) }) |t| {
        try std.testing.expectEqual(try eval.step(&full, t), try eval.step(&lazy, t));
        const want = try full.canonical(gpa);
        defer gpa.free(want);
        const got = try lazy.canonical(gpa);
        defer gpa.free(got);
        try std.testing.expectEqualSlices(u8, want, got);
    };
}

test "a node that reads a dim written earlier in stored order sees the write" {
    var c = try tinyCircuit(.good);
    defer c.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 3), c.batches.len);
    var s = try rc.State.init(gpa, &c);
    defer s.deinit();
    _ = try eval.step(&s, tok(1, 0));
    try std.testing.expectEqual(@as(model.Scalar, 1), s.value(tiny.dims.t1));
    try std.testing.expectEqual(@as(model.Scalar, 1), s.value(tiny.dims.t2));
}

test "a machine boots the calc input as RAM data and runs to HALT under a step budget and a cancel flag" {
    var c = try tinyCircuit(.good);
    defer c.deinit(gpa);
    var m = try rc.Machine.boot(gpa, &c, .{ .kind = .calc, .bytes = "A" });
    defer m.deinit(gpa);
    try std.testing.expectEqual(@as(u32, 'A'), try rc.readWord(&m.state, 64));
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);
    var trace: std.ArrayList(u32) = .empty;
    defer trace.deinit(gpa);
    var cancel: std.atomic.Value(bool) = .init(false);
    const first = try m.run(2, &out, &trace, &cancel);
    try std.testing.expectEqual(api.RunEnd{ .end = .budget, .steps = 2 }, first);
    const rest = try m.run(100, &out, &trace, &cancel);
    try std.testing.expectEqual(api.RunEnd{ .end = .halted, .steps = 1 }, rest);
    try std.testing.expectEqualStrings("hi", out.items);
    try std.testing.expectEqualSlices(u32, &.{ tok(4, 'h'), tok(4, 'i'), c.halt }, trace.items);
    try std.testing.expectEqual(api.RunEnd{ .end = .halted, .steps = 0 }, try m.run(100, &out, null, &cancel));
    var n = try rc.Machine.boot(gpa, &c, .{ .kind = .calc, .bytes = "" });
    defer n.deinit(gpa);
    cancel.store(true, .release);
    try std.testing.expectEqual(api.RunEnd{ .end = .cancelled, .steps = 0 }, try n.run(100, &out, null, &cancel));
    try std.testing.expectError(error.CircuitInputTooLong, rc.Machine.boot(gpa, &c, .{ .kind = .calc, .bytes = &long_input }));
    try std.testing.expectError(error.CircuitInputNul, rc.Machine.boot(gpa, &c, .{ .kind = .calc, .bytes = "1\x002" }));
}

test "the RAM patch round-trips words and an R read sees the booted input" {
    var c = try tinyCircuit(.good);
    defer c.deinit(gpa);
    var m = try rc.Machine.boot(gpa, &c, .{ .kind = .calc, .bytes = "AB" });
    defer m.deinit(gpa);
    try std.testing.expectEqual(@as(u32, 'A' | 'B' << 8), try rc.readWord(&m.state, 64));
    try std.testing.expectEqual(tok(4, '1'), try eval.step(&m.state, tok(3, 2)));
    var e = try rc.Machine.boot(gpa, &c, .{ .kind = .calc, .bytes = "" });
    defer e.deinit(gpa);
    try std.testing.expect(!e.state.rowPresent(c.program.ram_tables[0], 2));
    try std.testing.expectEqual(tok(4, '0'), try eval.step(&e.state, tok(3, 2)));
}

test "snapshots are canonical: equal histories give equal bytes, a write or a lag changes them" {
    var c = try tinyCircuit(.good);
    defer c.deinit(gpa);
    var a = try rc.Machine.boot(gpa, &c, .{ .kind = .calc, .bytes = "" });
    defer a.deinit(gpa);
    var b = try rc.Machine.boot(gpa, &c, .{ .kind = .calc, .bytes = "" });
    defer b.deinit(gpa);
    for ([_]u32{ tok(1, 0), tok(2, 5) }) |t| {
        _ = try eval.step(&a.state, t);
        _ = try eval.step(&b.state, t);
    }
    const sa = try a.snapshot(gpa);
    defer gpa.free(sa);
    const sb = try b.snapshot(gpa);
    defer gpa.free(sb);
    try std.testing.expectEqualSlices(u8, sa, sb);
    _ = try eval.step(&b.state, tok(1, 0));
    _ = try eval.step(&a.state, tok(4, 'h'));
    const sa2 = try a.snapshot(gpa);
    defer gpa.free(sa2);
    const sb2 = try b.snapshot(gpa);
    defer gpa.free(sb2);
    try std.testing.expect(!std.mem.eql(u8, sa2, sb2));
}

const Seg = struct { typ: u32 = 1, vaddr: u32, data: []const u8, memsz: u32 };

/// A static ELF32 RISC-V executable with the given program headers, data after the header table.
fn elfImage(entry: u32, segs: []const Seg) ![]u8 {
    var size: usize = 52 + 32 * segs.len;
    for (segs) |sg| size += sg.data.len;
    const b = try gpa.alloc(u8, size);
    @memset(b, 0);
    @memcpy(b[0..7], "\x7fELF\x01\x01\x01");
    const w = std.mem.writeInt;
    w(u16, b[16..18], 2, .little);
    w(u16, b[18..20], 243, .little);
    w(u32, b[24..28], entry, .little);
    w(u32, b[28..32], 52, .little);
    w(u16, b[42..44], 32, .little);
    w(u16, b[44..46], @intCast(segs.len), .little);
    var off: u32 = @intCast(52 + 32 * segs.len);
    for (segs, 0..) |sg, i| {
        const p = b[52 + 32 * i ..][0..32];
        w(u32, p[0..4], sg.typ, .little);
        w(u32, p[4..8], off, .little);
        w(u32, p[8..12], sg.vaddr, .little);
        w(u32, p[16..20], @intCast(sg.data.len), .little);
        w(u32, p[20..24], sg.memsz, .little);
        @memcpy(b[off..][0..sg.data.len], sg.data);
        off += @intCast(sg.data.len);
    }
    return b;
}

test "an ELF's PT_LOAD segments land in ROM words, DROM and zero-filled RAM; bad images are refused" {
    const code = [_]u8{ 0x13, 0x05, 0x50, 0x00, 0x73, 0x00, 0x00, 0x00 };
    const good = try elfImage(0x40004, &.{ .{ .vaddr = 0x40000, .data = &code, .memsz = 8 }, .{ .vaddr = 0x100, .data = "hi", .memsz = 8 }, .{ .vaddr = 0x20000, .data = "ro", .memsz = 2 } });
    defer gpa.free(good);
    const img = try elf.parse(gpa, good);
    defer img.deinit(gpa);
    try std.testing.expectEqual(@as(u32, 0x00500513), img.rom[0]);
    try std.testing.expectEqual(@as(u32, 0x73), img.rom[1]);
    try std.testing.expectEqual(@as(u32, 0), img.rom[2]);
    try std.testing.expectEqualStrings("hi\x00\x00", img.ram[0x100..0x104]);
    try std.testing.expectEqualStrings("ro", img.drom[0..2]);
    try std.testing.expectEqual(@as(u32, 0x40004), img.entry);
    const Bad = struct { entry: u32, seg: Seg, want: anyerror };
    for ([_]Bad{
        .{ .entry = 0x40000, .seg = .{ .typ = 3, .vaddr = 0x40000, .data = &code, .memsz = 8 }, .want = error.ElfDynamic },
        .{ .entry = 0x40000, .seg = .{ .vaddr = 0x7fff, .data = "ab", .memsz = 2 }, .want = error.ElfSegment },
        .{ .entry = 0x40002, .seg = .{ .vaddr = 0x40000, .data = &code, .memsz = 8 }, .want = error.ElfEntry },
        .{ .entry = 0x40000, .seg = .{ .vaddr = 0x40000, .data = &code, .memsz = 4 }, .want = error.ElfSegment },
    }) |bad| {
        const bytes = try elfImage(bad.entry, &.{bad.seg});
        defer gpa.free(bytes);
        try std.testing.expectError(bad.want, elf.parse(gpa, bytes));
    }
}
