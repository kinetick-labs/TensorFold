//! Shortest bounded fixed decimal output keeps all operand arithmetic in prepared FFN kernels.
const std = @import("std");
const packet = @import("runner_graph_packet.zig");
const math = @import("runner_graph_f64_math.zig");

const s = struct {
    const original_m: u32 = 140;
    const original_e: u32 = 141;
    const magnitude: u32 = 142;
    const two: u32 = 143;
    const ten: u32 = 144;
    const minus_one: u32 = 145;
    const hundred: u32 = 146;
    const integral: u32 = 147;
    const min_k: u32 = 148;
    const max_k: u32 = 149;
    const min_probe_k: u32 = 150;
    const max_scale: u32 = 151;
    const n: u32 = 152;
    const d: u32 = 153;
    const exponent: u32 = 154;
    const k: u32 = 155;
    const scaled_n: u32 = 156;
    const temporary: u32 = 158;
    const p_minus_one: u32 = 159;
    const quotient: u32 = 160;
    const remainder: u32 = 161;
    const nearest: u32 = 162;
    const other: u32 = 163;
    const coefficient: u32 = 164;
    const scale: u32 = 165;
    const candidate_n: u32 = 166;
    const candidate_d: u32 = 167;
    const scale_work: u32 = 168;
    const back_m: u32 = 169;
    const back_e: u32 = 170;
    const trim_q: u32 = 171;
    const trim_r: u32 = 172;
    const control: u32 = 173;
    const adjusted_k: u32 = 174;
    const decade: u32 = 175;
    const sign_factor: u32 = 176;
};

/// Named repr emission publishes bytes only after preparation and the formatter both succeed.
pub fn render(c: anytype, m: u32, e: u32) !void {
    const refs = try prepare(c, m, e);
    try publish(c, refs);
}

/// Returns coefficient/control refs 164/173 without BYTE output; scratch occupies only slots 140..176.
pub fn prepare(c: anytype, m: u32, e: u32) ![2]u32 {
    c.move(s.original_m, m);
    c.move(s.original_e, e);
    try math.ensureFloat(c, s.original_m, s.original_e);
    try constants(c);
    const negative = try c.sign(s.original_m);
    c.move(s.sign_factor, if (negative) s.minus_one else 1);
    if (!try c.nonzero(s.original_m)) {
        try c.call("decimal_mul", &.{ 0, s.sign_factor }, s.coefficient);
        c.move(s.control, s.integral);
        return .{ s.coefficient, s.control };
    }
    try c.meta(s.original_m, 0);
    c.move(s.magnitude, 253);
    try exactFraction(c);
    try normalize(c);
    if (try c.less(s.k, s.min_probe_k) or try c.less(s.max_k, s.k))
        return error.GraphF64Arithmetic;
    c.move(s.scaled_n, s.n);
    c.move(s.p_minus_one, 0);
    c.move(s.decade, s.ten);
    for (1..18) |_| {
        try c.divide(s.scaled_n, s.d, s.quotient, s.remainder);
        try c.call("round_div", &.{ s.quotient, s.remainder, s.scaled_n, s.d }, s.nearest);
        try c.call("sub", &.{ s.p_minus_one, s.k }, s.scale);
        if (try c.equal(s.nearest, s.quotient)) {
            try c.call("add", &.{ s.quotient, 1 }, s.other);
        } else c.move(s.other, s.quotient);
        if (try roundtrips(c, s.nearest)) {
            c.move(s.coefficient, s.nearest);
            try finish(c);
            return .{ s.coefficient, s.control };
        }
        if (try c.nonzero(s.remainder) and try roundtrips(c, s.other)) {
            c.move(s.coefficient, s.other);
            try finish(c);
            return .{ s.coefficient, s.control };
        }
        // Only a carry to 10 can bring k=-5 into the admitted output interval, and p=1 already tried it.
        if (try c.less(s.k, s.min_k)) return error.GraphF64Arithmetic;
        try c.call("decimal_mul", &.{ s.scaled_n, s.ten }, s.scaled_n);
        try c.call("add", &.{ s.p_minus_one, 1 }, s.p_minus_one);
        try c.call("decimal_mul", &.{ s.decade, s.ten }, s.decade);
    }
    return error.GraphF64Arithmetic;
}

fn constants(c: anytype) !void {
    try c.literal(s.two, "2");
    try c.literal(s.ten, "10");
    try c.literal(s.minus_one, "-1");
    try c.literal(s.hundred, "100");
    try c.literal(s.integral, "300");
    try c.literal(s.min_k, "-4");
    try c.literal(s.max_k, "15");
    try c.literal(s.min_probe_k, "-5");
    try c.literal(s.max_scale, "20");
}

fn exactFraction(c: anytype) !void {
    c.move(s.n, s.magnitude);
    c.move(s.d, 1);
    c.move(s.exponent, s.original_e);
    for (0..100) |_| {
        if (!try c.nonzero(s.exponent)) return;
        if (try c.less(s.exponent, 0)) {
            try c.call("decimal_mul", &.{ s.d, s.two }, s.d);
            try c.call("add", &.{ s.exponent, 1 }, s.exponent);
        } else {
            try c.call("decimal_mul", &.{ s.n, s.two }, s.n);
            try c.call("sub", &.{ s.exponent, 1 }, s.exponent);
        }
    }
    if (try c.nonzero(s.exponent)) return error.GraphF64Arithmetic;
}

/// N/D is made decimal-normal without division or any host interpretation of its tensor digits.
fn normalize(c: anytype) !void {
    c.move(s.k, 0);
    for (0..64) |_| {
        if (try c.less(s.n, s.d)) {
            try c.call("decimal_mul", &.{ s.n, s.ten }, s.n);
            try c.call("sub", &.{ s.k, 1 }, s.k);
            continue;
        }
        try c.call("decimal_mul", &.{ s.d, s.ten }, s.temporary);
        if (!try c.less(s.n, s.temporary)) {
            c.move(s.d, s.temporary);
            try c.call("add", &.{ s.k, 1 }, s.k);
            continue;
        }
        return;
    }
    return error.GraphF64Arithmetic;
}

/// Candidate C has exact value C*10^-scale; canonical binary equality decides the shortest precision.
fn roundtrips(c: anytype, coefficient: u32) !bool {
    if (!try c.nonzero(coefficient)) return false;
    c.move(s.candidate_n, coefficient);
    c.move(s.candidate_d, 1);
    c.move(s.scale_work, s.scale);
    try c.meta(s.scale_work, 0);
    c.move(s.temporary, 253);
    if (try c.less(s.max_scale, s.temporary)) return error.GraphF64Arithmetic;
    for (0..20) |_| {
        if (!try c.nonzero(s.scale_work)) break;
        if (try c.less(s.scale_work, 0)) {
            try c.call("decimal_mul", &.{ s.candidate_n, s.ten }, s.candidate_n);
            try c.call("add", &.{ s.scale_work, 1 }, s.scale_work);
        } else {
            try c.call("decimal_mul", &.{ s.candidate_d, s.ten }, s.candidate_d);
            try c.call("sub", &.{ s.scale_work, 1 }, s.scale_work);
        }
    }
    if (try c.nonzero(s.scale_work)) return error.GraphF64Arithmetic;
    try c.call("decimal_mul", &.{ s.candidate_n, s.sign_factor }, s.candidate_n);
    try math.ratio(c, s.candidate_n, s.candidate_d, s.back_m, s.back_e);
    return try c.equal(s.back_m, s.original_m) and try c.equal(s.back_e, s.original_e);
}

fn finish(c: anytype) !void {
    c.move(s.adjusted_k, s.k);
    if (!try c.less(s.coefficient, s.decade))
        try c.call("add", &.{ s.adjusted_k, 1 }, s.adjusted_k);
    if (try c.less(s.adjusted_k, s.min_k) or try c.less(s.max_k, s.adjusted_k))
        return error.GraphF64Arithmetic;
    for (0..17) |_| {
        try c.divide(s.coefficient, s.ten, s.trim_q, s.trim_r);
        if (try c.nonzero(s.trim_r)) break;
        c.move(s.coefficient, s.trim_q);
        try c.call("sub", &.{ s.scale, 1 }, s.scale);
    }
    if (try c.less(0, s.scale)) {
        if (try c.less(s.max_scale, s.scale)) return error.GraphF64Arithmetic;
        try c.call("add", &.{ s.hundred, s.scale }, s.control);
    } else {
        for (0..15) |_| {
            if (!try c.nonzero(s.scale)) break;
            try c.call("decimal_mul", &.{ s.coefficient, s.ten }, s.coefficient);
            try c.call("add", &.{ s.scale, 1 }, s.scale);
        }
        if (try c.nonzero(s.scale)) return error.GraphF64Arithmetic;
        c.move(s.control, s.integral);
    }
    try c.call("decimal_mul", &.{ s.coefficient, s.sign_factor }, s.coefficient);
}

/// The emitter may stop after BYTE tokens, so no bytes are published until its successful halt.
fn publish(c: anytype, refs: [2]u32) !void {
    var bytes: std.ArrayList(u8) = .empty;
    defer bytes.deinit(c.m.gpa);
    const template = c.m.bank.find("format_f64", 62) orelse return error.GraphEmitMissing;
    c.end = try c.m.invoke(template, &refs, 0, c.budget, &bytes, c.cancel, true);
    if (c.end != .halted) return error.GraphAlgorithmStopped;
    if (bytes.items.len > packet.max_output or c.out.items.len > packet.max_output - bytes.items.len)
        return error.GraphOutputSize;
    try c.out.appendSlice(c.m.gpa, bytes.items);
}
