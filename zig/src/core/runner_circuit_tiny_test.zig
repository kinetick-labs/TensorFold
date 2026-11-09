//! A hand-built circuit file (store/load RAM, a lagged key, a clear decay, a hazard pair, variants) for host tests.
const std = @import("std");

const kappa = "0.00000095367431640625";
pub const a_ref = "16777216.0";
/// Families: 0 GO, 1 A, 2 W (row bits 0-1, data bit 2), 3 R (row bits 0-1), 4 BYTE, 5 HALT.
pub const sizes = [_]u32{ 1, 1, 8, 4, 512, 1 };
pub const dims = struct {
    pub const is_w = 3;
    pub const is_r = 4;
    pub const rd = 13;
    pub const code = 14;
    pub const t1 = 33;
    pub const t2 = 34;
    pub const lagout = 35;
    pub const lag3out = 36;
    pub const sum = 37;
    pub const count = 38;
};
pub const family_bits = 3;

pub fn token(family: u32, payload: u32) u32 {
    var base: u32 = 0;
    for (sizes[0..family]) |s| base += s;
    return base + payload;
}

/// Variants that break one rule, for negative controls.
pub const Variant = enum { good, two_hot_key, unmapped_go, beta_gated, halving, chained, resting, banks };

/// The container bytes of the tiny circuit.
pub fn build(a: std.mem.Allocator, variant: Variant) ![]u8 {
    var j: std.ArrayList(u8) = .empty;
    defer j.deinit(a);
    const w = Writer{ .a = a, .j = &j };
    try w.s("{\"constants\":{\"a_bit\":128.0,\"a_ref\":" ++ a_ref ++ ",\"head_dim\":128,\"one_dim\":0,\"ref_cols\":[0,1]},\"dims\":[");
    for (0..dims.count) |i| try w.f("{s}{{\"name\":\"d{d}\"}}", .{ if (i == 0) "" else ",", i });
    try w.s("],\"features\":[");
    for (0..6) |f| try w.f("{s}{{\"dim\":{d},\"kind\":\"family\",\"family\":{d}}}", .{ if (f == 0) "" else ",", f + 1, f });
    const bits = [_][3]u32{ .{ 7, 2, 0 }, .{ 8, 2, 1 }, .{ 9, 2, 2 }, .{ 10, 3, 0 }, .{ 11, 3, 1 }, .{ 12, 4, 0 } };
    for (bits) |b| try w.f(",{{\"dim\":{d},\"kind\":\"payload_bit\",\"family\":{d},\"bit\":{d}}}", .{ b[0], b[1], b[2] });
    try w.s("],\"format\":\"tensorfold.runner.p2\",\"nodes\":[");
    try lagHead(w);
    try w.s(",");
    try lag3Head(w);
    try w.s(",");
    try ramHead(w, variant);
    if (variant == .banks) try bankHead(w);
    try w.s(",{\"kind\":\"ffn\",\"name\":\"h1\",\"out\":33,\"into\":false,\"terms\":[[1.0,[[0,1.0]],[[2,1.0]]]]}");
    try w.s(",{\"kind\":\"ffn\",\"name\":\"h2\",\"out\":34,\"into\":false,\"terms\":[[1.0,[[33,1.0]],[[0,1.0]]]]}");
    if (variant == .chained) for ([_][2]u32{ .{ 1, 2 }, .{ 2, 3 }, .{ 4, 4 } }, 0..) |t, i| {
        try w.f(",{{\"kind\":\"ffn\",\"name\":\"c{d}\",\"out\":{d},\"into\":{s},", .{ i, dims.sum, if (i == 0) "false" else "true" });
        try w.f("\"terms\":[[{d}.0,[[0,1.0]],[[{d},1.0]]]]}}", .{ t[0], t[1] });
    };
    if (variant == .resting) try w.f(",{{\"kind\":\"ffn\",\"name\":\"r\",\"out\":{d},\"into\":false,\"terms\":[[3.0,[[0,1.0]],[[0,2.0]]]]}}", .{dims.sum});
    try codeNodes(w, variant);
    try w.s("],\"output\":{\"code_dims\":[");
    for (0..family_bits + 16) |i| try w.f("{s}{d}", .{ if (i == 0) "" else ",", dims.code + i });
    try w.f("],\"family_bits\":{d},\"families\":", .{family_bits});
    try families(w);
    try w.s(",\"payload_bits\":16},\"program\":{\"input_address\":256,\"input_capacity\":256,\"kind\":\"calc\",\"entry\":262144,");
    try w.s("\"rom_le_hex\":\"");
    for (0..1 << 16) |_| try w.s("00000000");
    try w.s("\",\"drom_hex\":\"\",\"ram_hex\":\"");
    for (0..1 << 13) |_| try w.s("00000000");
    try w.s("\"},");
    try w.f("\"stats\":{{\"dims\":{d}}},\"tokens\":{{\"byte_family\":4,\"families\":", .{dims.count});
    try families(w);
    try w.f(",\"go\":0,\"halt\":{d},\"vocab\":{d}}},\"version\":1}}", .{ token(5, 0), token(5, 0) + 1 });
    return container(a, j.items);
}

pub fn container(a: std.mem.Allocator, body: []const u8) ![]u8 {
    const out = try a.alloc(u8, 64 + body.len);
    @memcpy(out[0..8], "TFIMR001");
    std.mem.writeInt(u32, out[8..12], 1, .little);
    std.mem.writeInt(u32, out[12..16], 64, .little);
    std.mem.writeInt(u64, out[16..24], body.len, .little);
    @memset(out[24..32], 0);
    std.crypto.hash.sha2.Sha256.hash(body, out[32..64], .{});
    @memcpy(out[64..], body);
    return out;
}

const Writer = struct {
    a: std.mem.Allocator,
    j: *std.ArrayList(u8),
    fn s(w: Writer, text: []const u8) !void {
        try w.j.appendSlice(w.a, text);
    }
    fn f(w: Writer, comptime fmt: []const u8, args: anytype) !void {
        const text = try std.fmt.allocPrint(w.a, fmt, args);
        defer w.a.free(text);
        try w.j.appendSlice(w.a, text);
    }
};

fn families(w: Writer) !void {
    const names = [_][]const u8{ "GO", "A", "W", "R", "BYTE", "HALT" };
    try w.s("[");
    for (names, sizes, 0..) |n, size, i| try w.f("{s}{{\"name\":\"{s}\",\"base\":{d},\"size\":{d}}}", .{ if (i == 0) "" else ",", n, token(@intCast(i), 0), size });
    try w.s("]");
}

/// One-hot row select: relu(flag + sum of literals - bits) over the payload bits of a family.
fn selectChannel(w: Writer, row: u32, flag: u32, bit_dims: [2]u32) !void {
    return selectAt(w, row, row, flag, bit_dims);
}

/// Channel `at` fires when the payload bits spell `row`.
fn selectAt(w: Writer, at: u32, row: u32, flag: u32, bit_dims: [2]u32) !void {
    var flag_coef: i32 = 1;
    for (0..2) |b| flag_coef += @intFromBool(row >> @intCast(b) & 1 == 0);
    try w.f("[{d},{{\"expr\":[[{d},{d}.0]", .{ at, flag, flag_coef });
    for (bit_dims, 0..) |d, b| try w.f(",[{d},{s}1.0]", .{ d, if (row >> @intCast(b) & 1 == 1) "" else "-" });
    try w.s(",[0,-2.0]],\"taps\":[1.0,0.0,0.0,0.0]}]");
}

/// A key from the previous token being W (a lag-1 tap), a read on A, a clear on BYTE; output to lagout.
fn lagHead(w: Writer) !void {
    try w.s("{\"kind\":\"head\",\"name\":\"lag\",\"n_cols\":128,\"levels\":\"bit\",\"tiny\":0.0,");
    try w.s("\"q\":[[0,{\"expr\":[[2,1.0]],\"taps\":[1.0,0.0,0.0,0.0]}]],\"k\":[[0,{\"expr\":[[3,1.0]],\"taps\":[0.0,1.0,0.0,0.0]}]],");
    try w.s("\"value_heads\":[{\"v\":[[0,{\"expr\":[[0," ++ a_ref ++ "]],\"taps\":[1.0,0.0,0.0,0.0]}],[1,{\"expr\":[[0," ++ a_ref ++ "]],\"taps\":[1.0,0.0,0.0,0.0]}],");
    try w.s("[2,{\"expr\":[[0,128.0]],\"taps\":[1.0,0.0,0.0,0.0]}]],\"a\":[[5,238.0]],\"beta\":null,\"z\":[[2,[[0,1.0]]]],");
    try w.s("\"out\":[[2,35,16384.0]],\"into\":false,\"init\":[]}]}");
}

/// A key from the token three back being W (taps 0,0,0,1), read on A, output to lag3out.
fn lag3Head(w: Writer) !void {
    try w.s("{\"kind\":\"head\",\"name\":\"lag3\",\"n_cols\":128,\"levels\":\"bit\",\"tiny\":0.0,");
    try w.s("\"q\":[[0,{\"expr\":[[2,1.0]],\"taps\":[1.0,0.0,0.0,0.0]}]],\"k\":[[0,{\"expr\":[[3,1.0]],\"taps\":[0.0,0.0,0.0,1.0]}]],");
    try w.s("\"value_heads\":[{\"v\":[[0,{\"expr\":[[0," ++ a_ref ++ "]],\"taps\":[1.0,0.0,0.0,0.0]}],[1,{\"expr\":[[0," ++ a_ref ++ "]],\"taps\":[1.0,0.0,0.0,0.0]}],");
    try w.s("[2,{\"expr\":[[0,128.0]],\"taps\":[1.0,0.0,0.0,0.0]}]],\"a\":[],\"beta\":null,\"z\":[[2,[[0,1.0]]]],");
    try w.s("\"out\":[[2,36,16384.0]],\"into\":false,\"init\":[]}]}");
}

/// ram0: W writes row (payload bits 0-1), data bit 2 to column 2; R reads to rd; row 3 is 1. halving: A halves table 2.
fn ramHead(w: Writer, variant: Variant) !void {
    try w.s("{\"kind\":\"head\",\"name\":\"ram0\",\"n_cols\":128,\"levels\":\"bit\",\"tiny\":0.0,\"q\":[");
    for (0..4) |r| {
        if (r > 0) try w.s(",");
        try selectChannel(w, @intCast(r), dims.is_r, .{ 10, 11 });
    }
    try w.s("],\"k\":[");
    for (0..4) |r| {
        if (r > 0) try w.s(",");
        try selectChannel(w, @intCast(r), dims.is_w, .{ 7, 8 });
    }
    if (variant == .two_hot_key) try w.s(",[9,{\"expr\":[[3,1.0]],\"taps\":[1.0,0.0,0.0,0.0]}]");
    const v = "\"v\":[[0,{\"expr\":[[3," ++ a_ref ++ "]],\"taps\":[1.0,0.0,0.0,0.0]}],[1,{\"expr\":[[3," ++ a_ref ++
        "]],\"taps\":[1.0,0.0,0.0,0.0]}],[2,{\"expr\":[[9,128.0]],\"taps\":[1.0,0.0,0.0,0.0]}]],";
    const beta = if (variant == .beta_gated) "\"beta\":[[0,1.0],[9,-1.0]]," else "\"beta\":null,";
    try w.s("],\"value_heads\":[{" ++ v ++ "\"a\":[],");
    try w.s(beta);
    try w.s("\"z\":[[2,[[4,1.0]]]],\"out\":[[2,13,16384.0]],\"into\":false,");
    try w.s("\"init\":[[3,[[0," ++ a_ref ++ "],[1," ++ a_ref ++ "],[2,128.0]]]]},{" ++ v);
    try w.s(if (variant == .halving) "\"a\":[[2,110.0]]," else "\"a\":[],");
    try w.s(beta);
    try w.s("\"z\":[],\"out\":[],\"into\":false,\"init\":[]}]}");
}

/// mirror for .banks: ram0's inputs, row r at table row r ^ 1, read to dims.sum; its data bit is the previous step's.
fn bankHead(w: Writer) !void {
    try w.s(",{\"kind\":\"head\",\"name\":\"mirror\",\"n_cols\":128,\"levels\":\"bit\",\"tiny\":0.0,\"q\":[");
    for (0..4) |r| {
        if (r > 0) try w.s(",");
        try selectAt(w, @intCast(r ^ 1), @intCast(r), dims.is_r, .{ 10, 11 });
    }
    try w.s("],\"k\":[");
    for (0..4) |r| {
        if (r > 0) try w.s(",");
        try selectAt(w, @intCast(r ^ 1), @intCast(r), dims.is_w, .{ 7, 8 });
    }
    try w.s("],\"value_heads\":[{\"v\":[[0,{\"expr\":[[3," ++ a_ref ++ "]],\"taps\":[1.0,0.0,0.0,0.0]}],[1,{\"expr\":[[3," ++ a_ref);
    try w.s("]],\"taps\":[1.0,0.0,0.0,0.0]}],[2,{\"expr\":[[9,128.0]],\"taps\":[0.0,1.0,0.0,0.0]}]],\"a\":[],\"beta\":null,");
    try w.f("\"z\":[[2,[[4,1.0]]]],\"out\":[[2,{d},16384.0]],\"into\":false,\"init\":[]}}]}}", .{dims.sum});
}

/// One FFN per code dim: GO->BYTE('h'), BYTE even->+1, BYTE odd->HALT, A->A, W->A, R->BYTE('0'+rd), HALT->HALT.
fn codeNodes(w: Writer, variant: Variant) !void {
    const Rule = struct { up: []const u8, target: u32 };
    const rules = [_]Rule{
        .{ .up = "[[1,1.0]]", .target = token(4, 'h') },
        .{ .up = "[[2,1.0]]", .target = token(1, 0) },
        .{ .up = "[[3,1.0]]", .target = token(1, 0) },
        .{ .up = "[[4,1.0]]", .target = token(4, '0') },
        .{ .up = "[[5,1.0],[12,-1.0]]", .target = token(4, 'i') },
        .{ .up = "[[12,1.0]]", .target = token(5, 0) },
        .{ .up = "[[6,1.0]]", .target = token(5, 0) },
    };
    for (0..family_bits + 16) |i| {
        try w.f(",{{\"kind\":\"ffn\",\"name\":\"code{d}\",\"out\":{d},\"into\":false,\"terms\":[", .{ i, dims.code + i });
        for (rules, 0..) |r, n| {
            if (variant == .unmapped_go and n == 0) continue;
            const code = codeOf(r.target);
            const sign: []const u8 = if (code >> @intCast(i) & 1 == 1) "" else "-";
            try w.f("{s}[{s}" ++ kappa ++ ",{s},[[0,1.0]]]", .{ if (n == @intFromBool(variant == .unmapped_go)) "" else ",", sign, r.up });
        }
        if (i == family_bits) try w.s(",[0.0000019073486328125,[[13,1.0]],[[4,1.0]]]");
        try w.s("]}");
    }
}

fn codeOf(t: u32) u64 {
    var base: u32 = 0;
    for (sizes, 0..) |s, f| {
        if (t < base + s) return @as(u64, f) | @as(u64, t - base) << family_bits;
        base += s;
    }
    unreachable;
}
