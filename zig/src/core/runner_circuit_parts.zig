//! CPU parts of a compiled circuit, derived from its batches alone, so a loaded and a prepared (packed) circuit match.
const std = @import("std");
const model = @import("runner_circuit_model.zig");
const flow = @import("runner_circuit_flow.zig");

const Scalar = model.Scalar;
const none = model.no_form;

/// Fills every batch's split parts and the circuit's dataflow; the batches keep their device layout.
pub fn prepare(a: std.mem.Allocator, c: *model.Circuit) !void {
    const batches = try a.dupe(model.Batch, c.batches);
    for (batches) |*b| switch (b.*) {
        .ffn => |*f| f.split = try ffnParts(a, f),
        .heads => |*h| h.split = try headParts(a, h),
    };
    c.batches = batches;
    c.flow = try flow.build(a, c);
}

/// The dims each form of a Sparse reads, as CSR by form.
fn formDims(a: std.mem.Allocator, m: *const model.Sparse) !struct { []u32, []u32 } {
    const starts = try a.alloc(u32, m.outputs + 1);
    @memset(starts, 0);
    for (m.used) |d| for (m.starts[d]..m.starts[d + 1]) |e| {
        starts[m.dst[e] + 1] += 1;
    };
    for (1..starts.len) |i| starts[i] += starts[i - 1];
    const dims = try a.alloc(u32, starts[m.outputs]);
    const fill = try a.dupe(u32, starts[0..m.outputs]);
    defer a.free(fill);
    for (m.used) |d| for (m.starts[d]..m.starts[d + 1]) |e| {
        dims[fill[m.dst[e]]] = d;
        fill[m.dst[e]] += 1;
    };
    return .{ starts, dims };
}

fn find(root: []u32, i: u32) u32 {
    var r = i;
    while (root[r] != r) r = root[r];
    return r;
}

/// Joins two union-find sets under the smaller root, so a part's root is its first node.
fn join(root: []u32, i: u32, j: u32) void {
    const x = find(root, i);
    const y = find(root, j);
    root[@max(x, y)] = @min(x, y);
}

/// An FFN batch's independent parts in node order: nodes sharing no dim read or written fall in different parts.
fn ffnParts(a: std.mem.Allocator, f: *const model.Ffn) ![]model.Ffn {
    const n = f.outs.len;
    const root = try a.alloc(u32, n);
    defer a.free(root);
    for (root, 0..) |*r, i| r.* = @intCast(i);
    var owner: std.AutoHashMapUnmanaged(u32, u32) = .empty;
    defer owner.deinit(a);
    const up_starts, const up_dims = try formDims(a, &f.up.forms);
    const gate_starts, const gate_dims = try formDims(a, &f.gate.forms);
    defer for ([_][]u32{ up_starts, up_dims, gate_starts, gate_dims }) |s| a.free(s);
    for (f.outs, 0..) |o, i| try claim(a, &owner, root, o.dim, @intCast(i));
    for (0..f.terms) |t| {
        const node = f.slot[t];
        if (f.up.form[t] != none) for (up_dims[up_starts[f.up.form[t]]..up_starts[f.up.form[t] + 1]]) |d| try claim(a, &owner, root, d, node);
        if (f.gate.form[t] != none) for (gate_dims[gate_starts[f.gate.form[t]]..gate_starts[f.gate.form[t] + 1]]) |d| try claim(a, &owner, root, d, node);
    }
    const local = try a.alloc(u32, f.up.forms.sources);
    defer a.free(local);
    const dims: Dims = .{ .up_starts = up_starts, .up_dims = up_dims, .gate_starts = gate_starts, .gate_dims = gate_dims, .local = local };
    var parts: std.ArrayList(model.Ffn) = .empty;
    for (0..n) |i| if (find(root, @intCast(i)) == i) try parts.append(a, try ffnPart(a, f, root, @intCast(i), dims));
    return parts.items;
}

/// The dims each up and gate form reads, and a dim-indexed scratch for a part's local source numbers.
const Dims = struct { up_starts: []const u32, up_dims: []const u32, gate_starts: []const u32, gate_dims: []const u32, local: []u32 };

fn claim(a: std.mem.Allocator, owner: *std.AutoHashMapUnmanaged(u32, u32), root: []u32, d: u32, node: u32) !void {
    const got = try owner.getOrPut(a, d);
    if (!got.found_existing) got.value_ptr.* = node else join(root, got.value_ptr.*, node);
}

/// The nodes rooted at `part` in batch order; forms renumbered by first use read the part's dims as ascending sources.
fn ffnPart(a: std.mem.Allocator, f: *const model.Ffn, root: []u32, part: u32, dims: Dims) !model.Ffn {
    const local = try a.alloc(u32, f.outs.len);
    defer a.free(local);
    var outs: std.ArrayList(model.Out) = .empty;
    for (f.outs, 0..) |o, i| if (find(root, @intCast(i)) == part) {
        local[i] = @intCast(outs.items.len);
        try outs.append(a, o);
    };
    var terms: std.ArrayList(u32) = .empty;
    defer terms.deinit(a);
    for (0..f.terms) |t| if (find(root, f.slot[t]) == part) try terms.append(a, @intCast(t));
    const coef = try a.alloc(Scalar, terms.items.len);
    const slot = try a.alloc(u32, terms.items.len);
    for (terms.items, coef, slot) |t, *k, *s| {
        k.* = f.coef[t];
        s.* = local[f.slot[t]];
    }
    var reads: std.ArrayList(u32) = .empty;
    for (terms.items) |t| {
        if (f.up.form[t] != none) try reads.appendSlice(a, dims.up_dims[dims.up_starts[f.up.form[t]]..dims.up_starts[f.up.form[t] + 1]]);
        if (f.gate.form[t] != none) try reads.appendSlice(a, dims.gate_dims[dims.gate_starts[f.gate.form[t]]..dims.gate_starts[f.gate.form[t] + 1]]);
    }
    std.mem.sort(u32, reads.items, {}, std.sort.asc(u32));
    var nd: usize = 0;
    for (reads.items) |d| if (nd == 0 or reads.items[nd - 1] != d) {
        reads.items[nd] = d;
        nd += 1;
    };
    const part_dims = reads.items[0..nd];
    for (part_dims, 0..) |d, i| dims.local[d] = @intCast(i);
    const up = try sharedPart(a, &f.up, terms.items, dims.local, part_dims.len);
    const gate = try sharedPart(a, &f.gate, terms.items, dims.local, part_dims.len);
    const forms = gate.forms.outputs;
    const starts = try a.alloc(u32, forms + 1);
    @memset(starts, 0);
    var always: std.ArrayList(u32) = .empty;
    for (gate.form, gate.bias, 0..) |g, b, t| {
        if (g != none) starts[g + 1] += 1 else if (b > 0) try always.append(a, @intCast(t));
    }
    for (1..starts.len) |i| starts[i] += starts[i - 1];
    const order = try a.alloc(u32, starts[forms]);
    const fill = try a.dupe(u32, starts[0..forms]);
    defer a.free(fill);
    for (gate.form, 0..) |g, t| if (g != none) {
        order[fill[g]] = @intCast(t);
        fill[g] += 1;
    };
    for (0..forms) |g| std.mem.sort(u32, order[starts[g]..starts[g + 1]], gate.bias, struct {
        fn gt(bias: []const Scalar, x: u32, y: u32) bool {
            return bias[x] > bias[y] or (bias[x] == bias[y] and x < y);
        }
    }.gt);
    return .{ .terms = @intCast(terms.items.len), .up = up, .gate = gate, .coef = coef, .slot = slot, .outs = outs.items, .order = order, .starts = starts, .always = always.items, .dims = part_dims };
}

/// Columns `cols` as their own Shared: same biases, forms renumbered by first use, sources by `local` (n) if given.
fn sharedPart(a: std.mem.Allocator, s: *const model.Shared, cols: []const u32, local: ?[]const u32, n: usize) !model.Shared {
    const remap = try a.alloc(u32, s.forms.outputs);
    defer a.free(remap);
    @memset(remap, none);
    var forms: u32 = 0;
    const form = try a.alloc(u32, cols.len);
    const bias = try a.alloc(Scalar, cols.len);
    for (cols, form, bias) |c, *fo, *b| {
        b.* = s.bias[c];
        const old = s.form[c];
        if (old == none) {
            fo.* = none;
            continue;
        }
        if (remap[old] == none) {
            remap[old] = forms;
            forms += 1;
        }
        fo.* = remap[old];
    }
    var src: std.ArrayList(u32) = .empty;
    var dst: std.ArrayList(u32) = .empty;
    var w: std.ArrayList(Scalar) = .empty;
    defer src.deinit(a);
    defer dst.deinit(a);
    defer w.deinit(a);
    const m = &s.forms;
    for (m.used) |d| for (m.starts[d]..m.starts[d + 1]) |e| if (remap[m.dst[e]] != none) {
        try src.append(a, if (local) |l| l[d] else d);
        try dst.append(a, remap[m.dst[e]]);
        try w.append(a, m.w[e]);
    };
    const sources: u32 = if (local != null) @intCast(n) else m.sources;
    return .{ .forms = try model.sparse(a, sources, forms, src.items, dst.items, w.items), .form = form, .bias = bias };
}

/// Each head of a batch alone: its channels keep their global numbers and order, its tables and outputs their own.
fn headParts(a: std.mem.Allocator, h: *const model.Heads) ![]model.Heads {
    const out = try a.alloc(model.Heads, h.heads.len);
    var c0: u32 = 0;
    var p0: u32 = h.channels;
    var o0: u32 = 0;
    for (h.heads, out) |*hd, *one| {
        var nch: u32 = @intCast(hd.q.len + hd.k.len);
        var nplain: u32 = 0;
        var nout: u32 = 0;
        for (hd.value_heads) |vh| {
            nch += @intCast(vh.v.len);
            nplain += @intFromBool(vh.a != null) + @intFromBool(vh.beta != null) + @as(u32, @intCast(vh.z.len));
            nout += @intCast(vh.out_cols.len);
        }
        const cols = try a.alloc(u32, nch + nplain);
        defer a.free(cols);
        for (cols[0..nch], c0..) |*c, i| c.* = @intCast(i);
        for (cols[nch..], p0..) |*c, i| c.* = @intCast(i);
        const map = Remap{ .c0 = c0, .p0 = p0, .channels = h.channels, .nch = nch };
        const vhs = try a.alloc(model.ValueHead, hd.value_heads.len);
        for (hd.value_heads, vhs) |vh, *v| v.* = .{
            .table = vh.table,
            .v = try map.picks(a, vh.v),
            .a = if (vh.a) |x| map.col(x) else null,
            .beta = if (vh.beta) |x| map.col(x) else null,
            .z = try map.picks(a, vh.z),
            .out_first = vh.out_first - o0,
            .out_cols = vh.out_cols,
            .out_w = vh.out_w,
        };
        const heads = try a.alloc(model.Head, 1);
        heads[0] = .{ .q = try map.picks(a, hd.q), .k = try map.picks(a, hd.k), .value_heads = vhs };
        one.* = .{
            .proj = try sharedPart(a, &h.proj, cols, null, 0),
            .channels = nch,
            .first_channel = h.first_channel + c0,
            .taps = h.taps[c0..][0..nch],
            .heads = heads,
            .outs = h.outs[o0..][0..nout],
        };
        c0 += nch;
        p0 += nplain;
        o0 += nout;
    }
    return out;
}

/// A batch column in one head's own numbering: its channels first, then its plain expressions.
const Remap = struct {
    c0: u32,
    p0: u32,
    channels: u32,
    nch: u32,

    fn col(r: Remap, c: u32) u32 {
        return if (c < r.channels) c - r.c0 else r.nch + (c - r.p0);
    }

    fn picks(r: Remap, a: std.mem.Allocator, ps: []const model.Pick) ![]model.Pick {
        const out = try a.alloc(model.Pick, ps.len);
        for (ps, out) |p, *o| o.* = .{ .at = p.at, .column = r.col(p.column) };
        return out;
    }
};
