//! A compiled circuit in memory: batched SwiGLU units and one-hot DeltaNet heads in stored node order.
const std = @import("std");

/// Evaluation precision; the loader refuses any weight or state value this type cannot hold exactly.
pub const Scalar = f32;
/// Key/value head width and the rows a one-hot key can address.
pub const hd: u32 = 128;
/// Conv taps of a q/k/v channel: projections at t, t-1, t-2, t-3.
pub const lags: u32 = 4;
/// Decay selectors (Qwenputer's DECAY with dt_bias -110): keep, halve, clear.
pub const keep_selector: Scalar = 0;
pub const halve_selector: Scalar = 110;
pub const clear_selector: Scalar = 238;

/// y = x * W over the nonzeros, stored by source so zero inputs cost nothing.
pub const Sparse = struct {
    sources: u32,
    outputs: u32,
    starts: []const u32,
    dst: []const u32,
    w: []const Scalar,
    /// Sources with at least one entry, ascending.
    used: []const u32,

    /// y = x * W: zeroes y, then scatters the inputs listed in `active` (every other input is zero).
    pub fn apply(m: *const Sparse, x: []const Scalar, active: []const u32, y: []Scalar) void {
        @memset(y, 0);
        for (if (m.used.len < active.len) m.used else active) |i| {
            const v = x[i];
            if (v == 0) continue;
            for (m.starts[i]..m.starts[i + 1]) |e| y[m.dst[e]] += v * m.w[e];
        }
    }

    pub fn nonzeros(m: *const Sparse) usize {
        return m.dst.len;
    }
};

/// Where a node's value lands: x[dim] = value, or x[dim] += value when `into`.
pub const Out = struct { dim: u32, into: bool };

pub const no_form = std.math.maxInt(u32);

/// A batch's expressions as distinct linear forms: column c = bias[c] + forms[form[c]], the constant dim folded out.
pub const Shared = struct {
    forms: Sparse,
    form: []const u32,
    bias: []const Scalar,

    pub fn columns(s: *const Shared) usize {
        return s.form.len;
    }

    /// `scratch` holds forms.outputs values.
    pub fn eval(s: *const Shared, x: []const Scalar, active: []const u32, scratch: []Scalar, y: []Scalar) void {
        s.forms.apply(x, active, scratch);
        for (y, s.form, s.bias) |*o, f, c| o.* = if (f == no_form) c else c + scratch[f];
    }
};

/// Consecutive FFN nodes with no read of a dim written earlier in the batch: y = sum c * up * max(gate, 0).
pub const Ffn = struct {
    terms: u32,
    up: Shared,
    gate: Shared,
    coef: []const Scalar,
    slot: []const u32,
    outs: []const Out,
    /// Terms grouped by gate form (starts has forms + 1 entries), each group by gate constant, largest first.
    order: []const u32,
    starts: []const u32,
    /// Terms whose gate is a positive constant; terms with a constant gate <= 0 never fire and are dropped.
    always: []const u32,
};

/// (row or column, projection column) of a channel or expression.
pub const Pick = struct { at: u32, column: u32 };

pub const ValueHead = struct {
    table: u32,
    v: []const Pick,
    a: ?u32,
    beta: ?u32,
    z: []const Pick,
    out_first: u32,
    out_cols: []const u32,
    out_w: []const Scalar,
};

pub const Head = struct {
    q: []const Pick,
    k: []const Pick,
    value_heads: []const ValueHead,
};

/// Consecutive head nodes. Projection columns [0, channels) are conv channels with taps; the rest are plain exprs.
pub const Heads = struct {
    proj: Shared,
    channels: u32,
    first_channel: u32,
    taps: []const [lags]Scalar,
    heads: []const Head,
    outs: []const Out,
};

pub const Batch = union(enum) { ffn: Ffn, heads: Heads };

/// One input feature: its dim's value is a function of the token's family id and payload.
pub const Feature = struct {
    dim: u32,
    kind: Kind,
    family: u32 = 0,
    bit: u32 = 0,
    digit: u32 = 0,
    qd_family: u32 = 0,
    amplitude: Scalar = 0,

    pub const Kind = enum { family, payload_bit, next_digit, code_bit };

    pub fn value(f: Feature, fam: u32, pay: u32, family_bits: u32) Scalar {
        return switch (f.kind) {
            .family => @floatFromInt(@intFromBool(fam == f.family)),
            .payload_bit => if (fam == f.family) @floatFromInt(pay >> @intCast(f.bit) & 1) else 0,
            .next_digit => @floatFromInt(@intFromBool((fam == f.family and pay == f.digit) or
                (f.digit > 0 and fam == f.qd_family and pay >> 4 == f.digit - 1))),
            .code_bit => blk: {
                const code = @as(u64, fam) | @as(u64, pay) << @intCast(family_bits);
                break :blk if (code >> @intCast(f.bit) & 1 == 1) f.amplitude else -f.amplitude;
            },
        };
    }
};

pub const Family = struct { base: u32, size: u32 };

/// A head's initial table: present rows (bit mask) of hd columns each, row-major hd x hd.
pub const Table = struct {
    present: u128,
    rows: []const Scalar,
    writable: bool,
};

/// A static program's memory: ROM instruction words, DROM bytes, RAM bytes and the entry address.
pub const Image = struct {
    rom: []u32,
    drom: []u8,
    ram: []u8,
    entry: u32,

    pub fn deinit(img: Image, a: std.mem.Allocator) void {
        a.free(img.rom);
        a.free(img.drom);
        a.free(img.ram);
    }
};

/// Where the calc firmware takes its expression, the RAM tables (ram<h>) the patch edits, and the boot image.
pub const Program = struct {
    kind: enum { calc, elf },
    input_address: u32,
    input_capacity: u32,
    ram_tables: [32]u32,
    a_ref: Scalar,
    a_bit: Scalar,
    image: Image,
};

pub const Circuit = struct {
    dims: u32,
    batches: []const Batch,
    features: []const Feature,
    families: []const Family,
    family_bits: u32,
    code_dims: []const u32,
    go: u32,
    halt: u32,
    byte_family: u32,
    fault_family: ?u32,
    tables: []const Table,
    table_names: []const []const u8,
    channels: u32,
    nodes: u32,
    program: Program,
    payload_sha256: [32]u8,
    arena: std.heap.ArenaAllocator,

    /// Reads and checks a circuit file (CIRCUIT-FORMAT.md v1).
    pub fn load(gpa: std.mem.Allocator, path: []const u8) !Circuit {
        return @import("runner_circuit_load.zig").fromPath(gpa, path);
    }

    /// The arena owns everything; `gpa` is the allocator `load` was given.
    pub fn deinit(c: *Circuit, gpa: std.mem.Allocator) void {
        _ = gpa;
        c.arena.deinit();
        c.* = undefined;
    }

    /// The family id and payload of token t, or null when t is outside the vocabulary.
    pub fn split(c: *const Circuit, t: u32) ?struct { u32, u32 } {
        var lo: usize = 0;
        var hi: usize = c.families.len;
        while (lo < hi) {
            const mid = (lo + hi) / 2;
            const f = c.families[mid];
            if (t < f.base) hi = mid else if (t - f.base >= f.size) lo = mid + 1 else return .{ @intCast(mid), t - f.base };
        }
        return null;
    }

    /// Weights the evaluator reads per step, for sizing and reports.
    pub fn nonzeros(c: *const Circuit) usize {
        var n: usize = 0;
        for (c.batches) |*b| n += switch (b.*) {
            .ffn => |*f| f.up.forms.nonzeros() + f.gate.forms.nonzeros(),
            .heads => |*h| h.proj.forms.nonzeros(),
        };
        return n;
    }
};

/// Builds a Sparse from (source, output, weight) triples in any order; a source's entries keep their input order.
pub fn sparse(a: std.mem.Allocator, sources: u32, outputs: u32, src: []const u32, dst_in: []const u32, w_in: []const Scalar) !Sparse {
    if (src.len != dst_in.len or src.len != w_in.len) return error.CircuitSparseShape;
    const starts = try a.alloc(u32, sources + 1);
    @memset(starts, 0);
    for (src) |s| {
        if (s >= sources) return error.CircuitIndex;
        starts[s + 1] += 1;
    }
    for (1..starts.len) |i| starts[i] += starts[i - 1];
    const fill = try a.dupe(u32, starts[0..sources]);
    defer a.free(fill);
    const dst = try a.alloc(u32, src.len);
    const w = try a.alloc(Scalar, src.len);
    for (src, dst_in, w_in) |s, d, v| {
        if (d >= outputs) return error.CircuitIndex;
        dst[fill[s]] = d;
        w[fill[s]] = v;
        fill[s] += 1;
    }
    var used: std.ArrayList(u32) = .empty;
    for (0..sources) |i| if (starts[i + 1] > starts[i]) try used.append(a, @intCast(i));
    return .{ .sources = sources, .outputs = outputs, .starts = starts, .dst = dst, .w = w, .used = used.items };
}

test "sparse apply skips zero inputs and adds repeated entries" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const m = try sparse(arena.allocator(), 3, 2, &.{ 0, 2, 2, 1 }, &.{ 1, 0, 0, 1 }, &.{ 2, 3, 4, 5 });
    var y = [_]Scalar{ 7, 7 };
    m.apply(&.{ 1, 0, 2 }, &.{ 0, 1, 2 }, &y);
    try std.testing.expectEqualSlices(Scalar, &.{ 14, 2 }, &y);
    m.apply(&.{ 1, 0, 2 }, &.{2}, &y);
    try std.testing.expectEqualSlices(Scalar, &.{ 14, 0 }, &y);
    try std.testing.expectError(error.CircuitIndex, sparse(arena.allocator(), 3, 2, &.{3}, &.{0}, &.{1}));
}

test "features follow the format's token-bus encodings" {
    const nd: Feature = .{ .dim = 1, .kind = .next_digit, .family = 9, .qd_family = 10, .digit = 0 };
    try std.testing.expectEqual(@as(Scalar, 1), nd.value(9, 0, 5));
    try std.testing.expectEqual(@as(Scalar, 0), nd.value(10, 0xff, 5));
    const nd3: Feature = .{ .dim = 1, .kind = .next_digit, .family = 9, .qd_family = 10, .digit = 3 };
    try std.testing.expectEqual(@as(Scalar, 1), nd3.value(10, 2 << 4 | 7, 5));
    const cb: Feature = .{ .dim = 1, .kind = .code_bit, .bit = 6, .amplitude = 128 };
    try std.testing.expectEqual(@as(Scalar, 128), cb.value(3, 2, 5));
    try std.testing.expectEqual(@as(Scalar, -128), cb.value(3, 1, 5));
    const pb: Feature = .{ .dim = 1, .kind = .payload_bit, .family = 4, .bit = 2 };
    try std.testing.expectEqual(@as(Scalar, 1), pb.value(4, 4, 5));
    try std.testing.expectEqual(@as(Scalar, 0), pb.value(5, 4, 5));
}
