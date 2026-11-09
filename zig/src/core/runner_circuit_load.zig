//! Reads a TFIMR001 circuit container (CIRCUIT-FORMAT.md v1) into batched, hazard-checked runtime form.
const std = @import("std");
const model = @import("runner_circuit_model.zig");

const Scalar = model.Scalar;
const hd = model.hd;
const header_bytes = 64;

const Term = struct { u32, f64 };
const Expr = []const Term;
const Chan = struct { expr: Expr, taps: [model.lags]f64 };
const ChanAt = struct { u32, Chan };
const ExprAt = struct { u32, Expr };
const OutAt = struct { u32, u32, f64 };
const InitRow = struct { u32, []const Term };
const JValueHead = struct { v: []const ChanAt, a: Expr, beta: ?Expr, z: []const ExprAt, out: []const OutAt, into: bool, init: []const InitRow };
const JNode = struct {
    kind: []const u8,
    name: []const u8 = "",
    out: ?u32 = null,
    into: ?bool = null,
    terms: ?[]const struct { f64, Expr, Expr } = null,
    n_cols: ?u32 = null,
    levels: ?[]const u8 = null,
    q: ?[]const ChanAt = null,
    k: ?[]const ChanAt = null,
    value_heads: ?[]const JValueHead = null,
};
const JFeature = struct { dim: u32, kind: []const u8, family: ?u32 = null, bit: ?u32 = null, digit: ?u32 = null, qs_family: ?u32 = null, qd_family: ?u32 = null, family_bits: ?u32 = null, amplitude: ?f64 = null };
const JFamily = struct { name: []const u8, base: u32, size: u32 };
const Doc = struct {
    format: []const u8,
    version: u32,
    stats: struct { dims: u32 },
    constants: struct { one_dim: u32, head_dim: u32, a_ref: f64, a_bit: f64, ref_cols: [2]u32 },
    dims: []const struct { name: []const u8 },
    features: []const JFeature,
    nodes: []const JNode,
    tokens: struct { vocab: u32, families: []const JFamily, go: u32, halt: u32, byte_family: u32 },
    output: struct { family_bits: u32, payload_bits: u32, code_dims: []const u32, families: []const JFamily },
    program: struct { kind: []const u8, input_address: u32, input_capacity: u32, entry: u32, rom_le_hex: []const u8, drom_hex: []const u8, ram_hex: []const u8 },
};

/// The checked JSON payload of a container: magic, version, sizes, flags and SHA-256 all as the format requires.
pub fn payload(bytes: []const u8) !struct { []const u8, [32]u8 } {
    if (bytes.len < header_bytes) return error.CircuitTruncated;
    if (!std.mem.eql(u8, bytes[0..8], "TFIMR001")) return error.CircuitMagic;
    const r = std.mem.readInt;
    if (r(u32, bytes[8..12], .little) != 1 or r(u32, bytes[12..16], .little) != header_bytes) return error.CircuitVersion;
    if (r(u32, bytes[24..28], .little) != 0 or r(u32, bytes[28..32], .little) != 0) return error.CircuitFlags;
    if (r(u64, bytes[16..24], .little) != bytes.len - header_bytes) return error.CircuitTruncated;
    const body = bytes[header_bytes..];
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(body, &digest, .{});
    if (!std.mem.eql(u8, &digest, bytes[32..64])) return error.CircuitChecksum;
    return .{ body, digest };
}

/// Reads a container file; at most 1 GiB.
pub fn fromPath(gpa: std.mem.Allocator, path: []const u8) !model.Circuit {
    var threaded: std.Io.Threaded = .init_single_threaded;
    const io = threaded.io();
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 30));
    defer gpa.free(bytes);
    return fromBytes(gpa, bytes);
}

/// Parses and compiles a whole container; the circuit owns everything through its arena.
pub fn fromBytes(gpa: std.mem.Allocator, bytes: []const u8) !model.Circuit {
    const body, const digest = try payload(bytes);
    var parse_arena = std.heap.ArenaAllocator.init(gpa);
    defer parse_arena.deinit();
    const doc = std.json.parseFromSliceLeaky(Doc, parse_arena.allocator(), body, .{ .ignore_unknown_fields = true }) catch
        return error.CircuitJson;
    var arena = std.heap.ArenaAllocator.init(gpa);
    errdefer arena.deinit();
    var c = try compile(arena.allocator(), parse_arena.allocator(), &doc);
    c.payload_sha256 = digest;
    c.arena = arena;
    return c;
}

fn exact(v: f64) !Scalar {
    if (!std.math.isFinite(v)) return error.CircuitNumber;
    const s: Scalar = @floatCast(v);
    if (@as(f64, s) != v) return error.CircuitNotExact;
    return s;
}

fn compile(a: std.mem.Allocator, tmp: std.mem.Allocator, doc: *const Doc) !model.Circuit {
    if (!std.mem.eql(u8, doc.format, "tensorfold.runner.p2") or doc.version != 1) return error.CircuitVersion;
    const dims: u32 = @intCast(doc.dims.len);
    const k = doc.constants;
    if (dims == 0 or dims != doc.stats.dims or k.one_dim != 0 or k.head_dim != hd) return error.CircuitDims;
    if (k.ref_cols[0] != 0 or k.ref_cols[1] != 1) return error.CircuitDims;
    const families = try a.alloc(model.Family, doc.tokens.families.len);
    var end: u32 = 0;
    for (doc.tokens.families, doc.output.families, families) |f, g, *out| {
        if (f.base != end or f.size == 0 or f.size > 1 << 16 or f.base != g.base or f.size != g.size) return error.CircuitFamilies;
        out.* = .{ .base = f.base, .size = f.size };
        end += f.size;
    }
    if (doc.tokens.families.len != doc.output.families.len or end != doc.tokens.vocab) return error.CircuitFamilies;
    const o = doc.output;
    if (o.family_bits == 0 or o.family_bits > 16 or o.payload_bits != 16 or o.code_dims.len != o.family_bits + 16) return error.CircuitOutputCode;
    if (@as(usize, 1) << @intCast(o.family_bits) < families.len) return error.CircuitOutputCode;
    for (o.code_dims) |d| if (d >= dims) return error.CircuitIndex;
    if (doc.tokens.go >= end or doc.tokens.halt >= end or doc.tokens.byte_family >= families.len) return error.CircuitTokens;
    var b: Builder = .{ .a = a, .tmp = tmp, .dims = dims };
    for (doc.nodes) |*n| try b.node(n);
    try b.flush();
    var c: model.Circuit = .{
        .dims = dims,
        .batches = b.batches.items,
        .features = try features(a, doc, dims, @intCast(families.len)),
        .families = families,
        .family_bits = o.family_bits,
        .code_dims = try a.dupe(u32, o.code_dims),
        .go = doc.tokens.go,
        .halt = doc.tokens.halt,
        .byte_family = doc.tokens.byte_family,
        .fault_family = for (doc.tokens.families, 0..) |f, i| {
            if (std.mem.eql(u8, f.name, "FLT")) break @intCast(i);
        } else null,
        .tables = b.tables.items,
        .table_names = b.names.items,
        .channels = b.channels,
        .nodes = @intCast(doc.nodes.len),
        .program = try program(a, doc, b.names.items, b.tables.items),
        .flow = undefined,
        .payload_sha256 = undefined,
        .arena = undefined,
    };
    try @import("runner_circuit_parts.zig").prepare(a, &c);
    return c;
}

fn features(a: std.mem.Allocator, doc: *const Doc, dims: u32, n_families: u32) ![]model.Feature {
    const out = try a.alloc(model.Feature, doc.features.len);
    for (doc.features, out) |f, *g| {
        if (f.dim == 0 or f.dim >= dims) return error.CircuitFeatures;
        const kind = std.meta.stringToEnum(model.Feature.Kind, f.kind) orelse return error.CircuitFeatures;
        g.* = .{ .dim = f.dim, .kind = kind };
        switch (kind) {
            .family, .payload_bit => {
                g.family = f.family orelse return error.CircuitFeatures;
                if (kind == .payload_bit) g.bit = f.bit orelse return error.CircuitFeatures;
                if (g.family >= n_families or g.bit >= 16) return error.CircuitFeatures;
            },
            .next_digit => {
                g.digit = f.digit orelse return error.CircuitFeatures;
                g.family = f.qs_family orelse return error.CircuitFeatures;
                g.qd_family = f.qd_family orelse return error.CircuitFeatures;
                if (g.digit >= 8 or g.family >= n_families or g.qd_family >= n_families) return error.CircuitFeatures;
            },
            .code_bit => {
                g.bit = f.bit orelse return error.CircuitFeatures;
                g.amplitude = try exact(f.amplitude orelse return error.CircuitFeatures);
                if (f.family_bits != doc.output.family_bits or g.bit >= doc.output.family_bits + 16) return error.CircuitFeatures;
            },
        }
    }
    return out;
}

fn unhex(a: std.mem.Allocator, text: []const u8, len: usize) ![]u8 {
    if (text.len % 2 != 0 or text.len / 2 > len) return error.CircuitProgram;
    const out = try a.alloc(u8, len);
    @memset(out, 0);
    _ = std.fmt.hexToBytes(out[0 .. text.len / 2], text) catch return error.CircuitProgram;
    return out;
}

fn program(a: std.mem.Allocator, doc: *const Doc, names: []const []const u8, tables: []const model.Table) !model.Program {
    const p = doc.program;
    const rom_bytes = try unhex(a, p.rom_le_hex, 4 << 16);
    if (p.rom_le_hex.len != 2 * rom_bytes.len or p.ram_hex.len != 2 * (4 << 13)) return error.CircuitProgram;
    const rom = try a.alloc(u32, 1 << 16);
    for (rom, 0..) |*w, i| w.* = std.mem.readInt(u32, rom_bytes[4 * i ..][0..4], .little);
    a.free(rom_bytes);
    var out: model.Program = .{
        .kind = if (std.mem.eql(u8, p.kind, "calc")) .calc else if (std.mem.eql(u8, p.kind, "elf")) .elf else return error.CircuitProgram,
        .input_address = p.input_address,
        .input_capacity = p.input_capacity,
        .ram_tables = @splat(std.math.maxInt(u32)),
        .a_ref = try exact(doc.constants.a_ref),
        .a_bit = try exact(doc.constants.a_bit),
        .image = .{ .rom = rom, .drom = try unhex(a, p.drom_hex, 4 << 15), .ram = try unhex(a, p.ram_hex, 4 << 13), .entry = p.entry },
    };
    for (names, 0..) |name, t| {
        if (!std.mem.startsWith(u8, name, "ram")) continue;
        const h = std.fmt.parseInt(u5, name[3..], 10) catch continue;
        if (out.ram_tables[h] == std.math.maxInt(u32)) out.ram_tables[h] = @intCast(t);
    }
    for (out.ram_tables) |t| {
        if (t == std.math.maxInt(u32)) continue;
        if (t + 1 >= tables.len or !tables[t].writable or !tables[t + 1].writable) return error.CircuitProgram;
    }
    return out;
}

/// Accumulates nodes into batches: a batch ends at a kind change or at a read of a dim the batch already wrote.
const Builder = struct {
    a: std.mem.Allocator,
    tmp: std.mem.Allocator,
    dims: u32,
    batches: std.ArrayList(model.Batch) = .empty,
    tables: std.ArrayList(model.Table) = .empty,
    names: std.ArrayList([]const u8) = .empty,
    channels: u32 = 0,
    pending: std.ArrayList(*const JNode) = .empty,
    pending_heads: bool = false,
    written: std.DynamicBitSetUnmanaged = .{},

    fn node(b: *Builder, n: *const JNode) !void {
        if (b.written.bit_length == 0) b.written = try .initEmpty(b.tmp, b.dims);
        const is_head = std.mem.eql(u8, n.kind, "head");
        if (!is_head and !std.mem.eql(u8, n.kind, "ffn")) return error.CircuitNodeKind;
        var reads: std.ArrayList(u32) = .empty;
        var writes: std.ArrayList(u32) = .empty;
        if (is_head) try headDims(b, n, &reads, &writes) else try ffnDims(b, n, &reads, &writes);
        var hazard = b.pending.items.len > 0 and is_head != b.pending_heads;
        for (reads.items) |d| hazard = hazard or b.written.isSet(d);
        if (hazard) try b.flush();
        b.pending_heads = is_head;
        try b.pending.append(b.tmp, n);
        for (writes.items) |d| b.written.set(d);
    }

    fn check(b: *Builder, e: Expr, reads: *std.ArrayList(u32)) !void {
        for (e, 0..) |t, i| {
            if (t[0] >= b.dims) return error.CircuitIndex;
            for (e[0..i]) |u| if (u[0] == t[0]) return error.CircuitDuplicateDim;
            _ = try exact(t[1]);
            try reads.append(b.tmp, t[0]);
        }
    }

    fn ffnDims(b: *Builder, n: *const JNode, reads: *std.ArrayList(u32), writes: *std.ArrayList(u32)) !void {
        const out = n.out orelse return error.CircuitNode;
        if (out == 0 or out >= b.dims or n.into == null) return error.CircuitIndex;
        for (n.terms orelse return error.CircuitNode) |t| {
            _ = try exact(t[0]);
            try b.check(t[1], reads);
            try b.check(t[2], reads);
        }
        try writes.append(b.tmp, out);
    }

    fn headDims(b: *Builder, n: *const JNode, reads: *std.ArrayList(u32), writes: *std.ArrayList(u32)) !void {
        if (n.n_cols != hd) return error.CircuitHeadShape;
        for ([_]?[]const ChanAt{ n.q, n.k }) |chans| for (chans orelse return error.CircuitNode) |ch| try b.channel(ch, reads);
        for (n.value_heads orelse return error.CircuitNode) |*vh| {
            const own = writes.items.len;
            var mine: std.ArrayList(u32) = .empty;
            for (vh.v) |ch| try b.channel(ch, &mine);
            try b.check(vh.a, &mine);
            if (vh.beta) |e| try b.check(e, &mine);
            for (vh.z) |z| {
                if (z[0] >= hd) return error.CircuitIndex;
                try b.check(z[1], &mine);
            }
            for (mine.items) |d| for (writes.items[0..own]) |w| if (d == w) return error.CircuitHazard;
            try reads.appendSlice(b.tmp, mine.items);
            for (vh.out) |t| {
                if (t[0] >= hd or t[1] == 0 or t[1] >= b.dims) return error.CircuitIndex;
                _ = try exact(t[2]);
                try writes.append(b.tmp, t[1]);
            }
        }
    }

    fn channel(b: *Builder, ch: ChanAt, reads: *std.ArrayList(u32)) !void {
        if (ch[0] >= hd) return error.CircuitIndex;
        try b.check(ch[1].expr, reads);
        for (ch[1].taps) |t| _ = try exact(t);
    }

    fn flush(b: *Builder) !void {
        if (b.pending.items.len == 0) return;
        const batch: model.Batch = if (b.pending_heads) .{ .heads = try b.headBatch() } else .{ .ffn = try b.ffnBatch(b.pending.items) };
        try b.batches.append(b.a, batch);
        b.pending.clearRetainingCapacity();
        b.written.unsetAll();
    }

    fn ffnBatch(b: *Builder, nodes: []const *const JNode) !model.Ffn {
        var up: FormSet = .{};
        var gate: FormSet = .{};
        var coef: std.ArrayList(Scalar) = .empty;
        var slot: std.ArrayList(u32) = .empty;
        const outs = try b.a.alloc(model.Out, nodes.len);
        for (nodes, outs, 0..) |n, *out, s| {
            out.* = .{ .dim = n.out.?, .into = n.into.? };
            for (n.terms.?) |t| {
                try coef.append(b.a, try exact(t[0]));
                try slot.append(b.a, @intCast(s));
                try up.add(b, t[1]);
                try gate.add(b, t[2]);
            }
        }
        const g = try gate.finish(b);
        const starts = try b.a.alloc(u32, gate.forms + 1);
        @memset(starts, 0);
        var always: std.ArrayList(u32) = .empty;
        for (g.form, g.bias, 0..) |f, c, t| {
            if (f != model.no_form) starts[f + 1] += 1 else if (c > 0) try always.append(b.a, @intCast(t));
        }
        for (1..starts.len) |i| starts[i] += starts[i - 1];
        const order = try b.a.alloc(u32, starts[gate.forms]);
        const fill = try b.tmp.dupe(u32, starts[0..gate.forms]);
        for (g.form, 0..) |f, t| if (f != model.no_form) {
            order[fill[f]] = @intCast(t);
            fill[f] += 1;
        };
        for (0..gate.forms) |f| std.mem.sort(u32, order[starts[f]..starts[f + 1]], g.bias, struct {
            fn gt(bias: []const Scalar, x: u32, y: u32) bool {
                return bias[x] > bias[y] or (bias[x] == bias[y] and x < y);
            }
        }.gt);
        return .{ .terms = @intCast(coef.items.len), .up = try up.finish(b), .gate = g, .coef = coef.items, .slot = slot.items, .outs = outs, .order = order, .starts = starts, .always = always.items };
    }

    /// The batch as one Heads (shared forms, device layout); CPU per-head parts come from runner_circuit_parts.
    fn headBatch(b: *Builder) !model.Heads {
        const whole = try b.headsOf(b.pending.items, b.channels, @intCast(b.tables.items.len));
        b.channels += whole.channels;
        return whole;
    }

    fn headsOf(b: *Builder, nodes: []const *const JNode, first_channel: u32, first_table: u32) !model.Heads {
        var cols: FormSet = .{};
        var taps: std.ArrayList([model.lags]Scalar) = .empty;
        for (nodes) |n| {
            for ([_][]const ChanAt{ n.q.?, n.k.? }) |chans| for (chans) |ch| try b.addChannel(&cols, &taps, ch[1]);
            for (n.value_heads.?) |vh| for (vh.v) |ch| try b.addChannel(&cols, &taps, ch[1]);
        }
        const channels: u32 = @intCast(taps.items.len);
        var column = channels;
        var chan: u32 = 0;
        var table = first_table;
        var outs: std.ArrayList(model.Out) = .empty;
        const heads = try b.a.alloc(model.Head, nodes.len);
        for (nodes, heads) |n, *h| {
            const q = try b.picks(n.q.?, &chan);
            const kk = try b.picks(n.k.?, &chan);
            const vhs = try b.a.alloc(model.ValueHead, n.value_heads.?.len);
            for (n.value_heads.?, vhs) |*jv, *vh| {
                try b.tables.append(b.a, try initTable(b.a, jv, kk.len > 0 or jv.a.len > 0));
                try b.names.append(b.a, try b.a.dupe(u8, n.name));
                vh.* = .{ .table = table, .v = try b.picks(jv.v, &chan), .a = null, .beta = null, .z = &.{}, .out_first = @intCast(outs.items.len), .out_cols = &.{}, .out_w = &.{} };
                table += 1;
                if (jv.a.len > 0) vh.a = try plain(b, &cols, jv.a, &column);
                if (jv.beta) |e| vh.beta = try plain(b, &cols, e, &column);
                const z = try b.a.alloc(model.Pick, jv.z.len);
                for (jv.z, z) |e, *p| p.* = .{ .at = e[0], .column = try plain(b, &cols, e[1], &column) };
                vh.z = z;
                const out_cols = try b.a.alloc(u32, jv.out.len);
                const ws = try b.a.alloc(Scalar, jv.out.len);
                for (jv.out, out_cols, ws) |t, *c, *x| {
                    c.* = t[0];
                    x.* = try exact(t[2]);
                    try outs.append(b.a, .{ .dim = t[1], .into = jv.into });
                }
                vh.out_cols = out_cols;
                vh.out_w = ws;
            }
            h.* = .{ .q = q, .k = kk, .value_heads = vhs };
        }
        return .{
            .proj = try cols.finish(b),
            .channels = channels,
            .first_channel = first_channel,
            .taps = taps.items,
            .heads = heads,
            .outs = outs.items,
        };
    }

    fn addChannel(b: *Builder, cols: *FormSet, taps: *std.ArrayList([model.lags]Scalar), ch: Chan) !void {
        try cols.add(b, ch.expr);
        var t: [model.lags]Scalar = undefined;
        for (&t, ch.taps) |*x, v| x.* = try exact(v);
        try taps.append(b.a, t);
    }

    fn picks(b: *Builder, chans: []const ChanAt, next: *u32) ![]model.Pick {
        const out = try b.a.alloc(model.Pick, chans.len);
        for (chans, out, 0..) |ch, *p, i| {
            for (chans[0..i]) |prev| if (prev[0] == ch[0]) return error.CircuitDuplicateChannel;
            p.* = .{ .at = ch[0], .column = next.* };
            next.* += 1;
        }
        return out;
    }
};

fn plain(b: *Builder, cols: *FormSet, e: Expr, column: *u32) !u32 {
    try cols.add(b, e);
    column.* += 1;
    return column.* - 1;
}

/// Columns as distinct linear forms (keyed by their sorted non-constant terms) plus each column's constant.
const FormSet = struct {
    keys: std.StringHashMapUnmanaged(u32) = .empty,
    src: std.ArrayList(u32) = .empty,
    dst: std.ArrayList(u32) = .empty,
    w: std.ArrayList(Scalar) = .empty,
    form: std.ArrayList(u32) = .empty,
    bias: std.ArrayList(Scalar) = .empty,
    forms: u32 = 0,

    fn add(fs: *FormSet, b: *Builder, e: Expr) !void {
        const terms = try b.tmp.dupe(Term, e);
        std.mem.sort(Term, terms, {}, struct {
            fn lt(_: void, x: Term, y: Term) bool {
                return x[0] < y[0];
            }
        }.lt);
        var key: std.ArrayList(u8) = .empty;
        var bias: Scalar = 0;
        for (terms) |t| {
            const w = try exact(t[1]);
            if (t[0] == 0) {
                bias += w;
                continue;
            }
            try key.appendSlice(b.tmp, std.mem.asBytes(&t[0]));
            try key.appendSlice(b.tmp, std.mem.asBytes(&w));
        }
        try fs.bias.append(b.a, bias);
        if (key.items.len == 0) return fs.form.append(b.a, model.no_form);
        const got = try fs.keys.getOrPut(b.tmp, key.items);
        if (!got.found_existing) {
            got.value_ptr.* = fs.forms;
            for (terms) |t| if (t[0] != 0) {
                try fs.src.append(b.tmp, t[0]);
                try fs.dst.append(b.tmp, fs.forms);
                try fs.w.append(b.tmp, try exact(t[1]));
            };
            fs.forms += 1;
        }
        try fs.form.append(b.a, got.value_ptr.*);
    }

    fn finish(fs: *FormSet, b: *Builder) !model.Shared {
        return .{ .forms = try model.sparse(b.a, b.dims, fs.forms, fs.src.items, fs.dst.items, fs.w.items), .form = fs.form.items, .bias = fs.bias.items };
    }
};

fn initTable(a: std.mem.Allocator, vh: *const JValueHead, writable: bool) !model.Table {
    const rows = try a.alloc(Scalar, hd * hd);
    @memset(rows, 0);
    var present: u128 = 0;
    for (vh.init) |r| {
        if (r[0] >= hd) return error.CircuitIndex;
        const bit = @as(u128, 1) << @intCast(r[0]);
        if (present & bit != 0) return error.CircuitDuplicateRow;
        present |= bit;
        for (r[1]) |cv| {
            if (cv[0] >= hd) return error.CircuitIndex;
            rows[r[0] * hd + cv[0]] = try exact(cv[1]);
        }
    }
    return .{ .present = present, .rows = rows, .writable = writable };
}
