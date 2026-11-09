//! One head on the CPU: a run keeps its inputs as a frame and caches its selecting channels; v and z are lazy.
const std = @import("std");
const model = @import("runner_circuit_model.zig");
const Memo = @import("runner_circuit_memo.zig").Memo;
const State = @import("runner_circuit_state.zig").State;

const Scalar = model.Scalar;
const hd = model.hd;
const none = std.math.maxInt(u32);

/// A query or key channel with at most one nonzero tap: at step t its value is tap * p(t - lag), kept when positive.
pub const Sel = struct { column: u32, at: u32, tap: Scalar, lag: u32 };

pub const Plan = struct {
    h: *const model.Heads,
    /// The selecting columns (queries, keys, then each value head's decay and write gate) as shared forms.
    sel: model.Shared,
    /// Per unit input: whether a selecting form reads it (the memo key holds only these).
    sel_dim: []const bool,
    q: []const Sel,
    k: []const Sel,
    lags_q: u8,
    lags_k: u8,
    /// Per value head: its decay and write-gate select columns, or none.
    a: []const u32,
    beta: []const u32,
    /// Every projection form as a row of (dim, weight), dims ascending: the order the scatter adds them in.
    rows: model.Sparse,
    /// Per output: its z value's index in zrep, or none; per value head, its distinct z columns zrep[zstart[j]..].
    zcol: []const u32,
    zrep: []const u32,
    zstart: []const u32,
    /// A query or key with several nonzero taps: selection is computed from the frames, not cached.
    general: bool,
    /// Queries and keys read only the running step: an empty selection then depends on nothing older.
    lag0: bool = false,
    /// The plan whose frames hold this head's inputs (the first head of its selection group), and its member index.
    owner: u32 = none,
    member: u32 = 0,

    fn head(p: *const Plan) *const model.Head {
        return &p.h.heads[0];
    }
};

/// A run's inputs (nonzero dims ascending), stamped with step + 1; heads reading the same slots share one ring.
pub const Frame = struct {
    stamp: u64 = 0,
    dims: std.ArrayList(u32) = .empty,
    vals: std.ArrayList(Scalar) = .empty,

    pub fn deinit(f: *Frame, gpa: std.mem.Allocator) void {
        f.dims.deinit(gpa);
        f.vals.deinit(gpa);
    }
};

/// One head's selecting results of a run, stamped with step + 1.
pub const Picks = struct {
    stamp: u64 = 0,
    words: std.ArrayList(u32) = .empty,

    pub fn deinit(f: *Picks, gpa: std.mem.Allocator) void {
        f.words.deinit(gpa);
    }
};

/// A selection group's step key: its first member to run in a step looks it up once and names its generation.
pub const Group = struct {
    memo: Memo = .{},
    stamp: u64 = 0,
    hash: u64 = 0,
    gen: u64 = 0,
    next: u64 = 0,
    /// Per memo slot, the members whose selection under that slot's generation is empty (bit = member index).
    slot_gen: [memo_slots]u64 = @splat(0),
    empty: [memo_slots]u64 = @splat(0),
};

const memo_slots = @import("runner_circuit_memo.zig").slots_per_unit;

/// The last four runs cover steps t-3..t: a unit runs every step its lag window is open, else its inputs held.
pub const frames_per_unit = 4;

pub fn build(a: std.mem.Allocator, h: *const model.Heads, dims: []const u32) !Plan {
    const hd0 = &h.heads[0];
    var cols: std.ArrayList(u32) = .empty;
    var general = false;
    const q = try sels(a, h, hd0.q, &cols, &general);
    const k = try sels(a, h, hd0.k, &cols, &general);
    const av = try a.alloc(u32, hd0.value_heads.len);
    const bv = try a.alloc(u32, hd0.value_heads.len);
    for (hd0.value_heads, av, bv) |vh, *ai, *bi| {
        ai.* = none;
        bi.* = none;
        if (vh.a) |c| {
            ai.* = @intCast(cols.items.len);
            try cols.append(a, c);
        }
        if (vh.beta) |c| {
            bi.* = @intCast(cols.items.len);
            try cols.append(a, c);
        }
    }
    const forms = &h.proj.forms;
    const remap = try a.alloc(u32, forms.outputs);
    defer a.free(remap);
    @memset(remap, none);
    var n_forms: u32 = 0;
    const sel_form = try a.alloc(u32, cols.items.len);
    const sel_bias = try a.alloc(Scalar, cols.items.len);
    for (cols.items, sel_form, sel_bias) |c, *f, *b| {
        b.* = h.proj.bias[c];
        const old = h.proj.form[c];
        if (old == model.no_form) {
            f.* = model.no_form;
            continue;
        }
        if (remap[old] == none) {
            remap[old] = n_forms;
            n_forms += 1;
        }
        f.* = remap[old];
    }
    var src: std.ArrayList(u32) = .empty;
    var dst: std.ArrayList(u32) = .empty;
    var w: std.ArrayList(Scalar) = .empty;
    var row_form: std.ArrayList(u32) = .empty;
    var row_dim: std.ArrayList(u32) = .empty;
    var row_w: std.ArrayList(Scalar) = .empty;
    for (forms.used) |d| for (forms.starts[d]..forms.starts[d + 1]) |e| {
        try row_form.append(a, forms.dst[e]);
        try row_dim.append(a, d);
        try row_w.append(a, forms.w[e]);
        if (remap[forms.dst[e]] == none) continue;
        try src.append(a, d);
        try dst.append(a, remap[forms.dst[e]]);
        try w.append(a, forms.w[e]);
    };
    const sel_forms = try model.sparse(a, forms.sources, n_forms, src.items, dst.items, w.items);
    const sel_dim = try a.alloc(bool, dims.len);
    for (dims, sel_dim) |d, *m| m.* = sel_forms.starts[d + 1] > sel_forms.starts[d];
    var zcol: std.ArrayList(u32) = .empty;
    var zrep: std.ArrayList(u32) = .empty;
    const zstart = try a.alloc(u32, hd0.value_heads.len + 1);
    for (hd0.value_heads, 0..) |vh, j| {
        zstart[j] = @intCast(zrep.items.len);
        for (vh.out_cols) |c| {
            var col: u32 = none;
            for (vh.z) |z| if (z.at == c) {
                col = z.column;
            };
            try zcol.append(a, if (col == none) none else try distinct(a, h, &zrep, zstart[j], col));
        }
    }
    zstart[hd0.value_heads.len] = @intCast(zrep.items.len);
    return .{
        .h = h,
        .sel = .{ .forms = try sel_forms.densify(a), .form = sel_form, .bias = sel_bias },
        .sel_dim = sel_dim,
        .q = q,
        .k = k,
        .lags_q = lagMask(q),
        .lags_k = lagMask(k),
        .a = av,
        .beta = bv,
        .rows = try model.sparse(a, forms.outputs, forms.sources, row_form.items, row_dim.items, row_w.items),
        .zcol = zcol.items,
        .zrep = zrep.items,
        .zstart = zstart,
        .general = general,
        .lag0 = !general and for (q) |x| {
            if (x.lag != 0) break false;
        } else for (k) |x| {
            if (x.lag != 0) break false;
        } else true,
    };
}

/// The index of a column with the same form and bias as `col` among reps[from..], appending it when new.
fn distinct(a: std.mem.Allocator, h: *const model.Heads, reps: *std.ArrayList(u32), from: u32, col: u32) !u32 {
    for (reps.items[from..], from..) |r, i| {
        if (h.proj.form[r] == h.proj.form[col] and @as(u32, @bitCast(h.proj.bias[r])) == @as(u32, @bitCast(h.proj.bias[col]))) return @intCast(i);
    }
    try reps.append(a, col);
    return @intCast(reps.items.len - 1);
}

fn sels(a: std.mem.Allocator, h: *const model.Heads, picks: []const model.Pick, cols: *std.ArrayList(u32), general: *bool) ![]Sel {
    const out = try a.alloc(Sel, picks.len);
    for (picks, out) |p, *s| {
        var taps: u32 = 0;
        s.* = .{ .column = @intCast(cols.items.len), .at = p.at, .tap = 0, .lag = 0 };
        for (h.taps[p.column], 0..) |t, lag| if (t != 0) {
            taps += 1;
            s.tap = t;
            s.lag = @intCast(lag);
        };
        if (taps > 1) general.* = true;
        try cols.append(a, p.column);
    }
    return out;
}

fn lagMask(list: []const Sel) u8 {
    var m: u8 = 0;
    for (list) |s| m |= @as(u8, 1) << @intCast(s.lag);
    return m;
}

/// The run that holds step `step`'s inputs or picks: the latest run at or before it (null before the first step).
fn frameAt(comptime T: type, frames: []const T, step: i64) ?*const T {
    if (step < 0) return null;
    var best: ?*const T = null;
    for (frames) |*f| {
        if (f.stamp == 0 or f.stamp - 1 > @as(u64, @intCast(step))) continue;
        if (best == null or f.stamp > best.?.stamp) best = f;
    }
    return best;
}

fn oldest(comptime T: type, frames: []T) *T {
    var o = &frames[0];
    for (frames[1..]) |*f| if (f.stamp < o.stamp) {
        o = f;
    };
    return o;
}

/// Dense inputs of step t-lag (lag 0 is the running step's own inputs in s.x).
const Views = struct {
    s: *State,
    frames: []const Frame,
    t: u64,
    filled: [frames_per_unit]?*const Frame = @splat(null),

    fn dense(v: *Views, lag: u32) ?[]const Scalar {
        if (lag == 0) return v.s.x;
        const f = frameAt(Frame, v.frames, @as(i64, @intCast(v.t)) - lag) orelse return null;
        const x = v.s.past[lag - 1];
        if (v.filled[lag] != f) {
            if (v.filled[lag]) |old| for (old.dims.items) |d| {
                x[d] = 0;
            };
            for (f.dims.items, f.vals.items) |d, val| x[d] = val;
            v.filled[lag] = f;
        }
        return x;
    }

    fn clear(v: *Views) void {
        for (1..frames_per_unit) |lag| if (v.filled[lag]) |f| for (f.dims.items) |d| {
            v.s.past[lag - 1][d] = 0;
        };
        v.filled = @splat(null);
    }

    /// The projection of column `col` at step t-lag; zero before the first step, as the history starts.
    fn proj(v: *Views, p: *const Plan, col: u32, lag: u32) Scalar {
        const x = v.dense(lag) orelse return 0;
        const f = p.h.proj.form[col];
        const bias = p.h.proj.bias[col];
        if (f == model.no_form) return bias;
        var sum: Scalar = 0;
        for (p.rows.starts[f]..p.rows.starts[f + 1]) |e| {
            const xv = x[p.rows.dst[e]];
            if (xv != 0) sum += xv * p.rows.w[e];
        }
        return bias + sum;
    }

    /// The conv output of channel `col` at step t, the taps over its last four projections.
    fn conv(v: *Views, p: *const Plan, col: u32) Scalar {
        const t = p.h.taps[col];
        var pr: [model.lags]Scalar = @splat(0);
        for (0..model.lags) |lag| if (t[lag] != 0) {
            pr[lag] = v.proj(p, col, @intCast(lag));
        };
        return @max(t[0] * pr[0] + t[1] * pr[1] + t[2] * pr[2] + t[3] * pr[3], 0);
    }
};

/// Selecting results of the running step: positive queries and keys at their taps, then each decay and write gate.
fn select(s: *State, p: *const Plan, nz: []const u32) []const u32 {
    const n_cols = p.sel.columns();
    const vals = s.work[0..n_cols];
    p.sel.eval(s.x, nz, s.work[n_cols..][0..p.sel.forms.outputs], vals);
    const out = s.words[0..];
    var n: usize = 0;
    for ([_][]const Sel{ p.q, p.k }) |list| {
        const count_at = n;
        n += 1;
        for (list, 0..) |q, i| {
            if (p.general) break;
            const v = q.tap * vals[q.column];
            if (!(v > 0)) continue;
            out[n] = @intCast(i);
            out[n + 1] = @bitCast(v);
            n += 2;
        }
        out[count_at] = @intCast((n - count_at - 1) / 2);
    }
    for (p.a, p.beta) |ai, bi| {
        out[n] = if (ai == none) 0 else @bitCast(vals[ai]);
        out[n + 1] = if (bi == none) @bitCast(@as(Scalar, 1)) else @bitCast(vals[bi]);
        n += 2;
    }
    return out[0..n];
}

/// Indices of the positive channels of one list at step t, ascending, from the frames at their lags.
fn positives(p: *const Plan, frames: []const Picks, t: u64, which: u1, out: []u32) usize {
    const list = if (which == 0) p.q else p.k;
    const mask = if (which == 0) p.lags_q else p.lags_k;
    var n: usize = 0;
    for (0..frames_per_unit) |lag| {
        if (mask >> @intCast(lag) & 1 == 0) continue;
        const f = frameAt(Picks, frames, @as(i64, @intCast(t)) - @as(i64, @intCast(lag))) orelse continue;
        const w = f.words.items;
        var at: usize = 0;
        if (which == 1) at = 1 + 2 * w[0];
        for (0..w[at]) |j| {
            const i = w[at + 1 + 2 * j];
            if (list[i].lag != lag) continue;
            out[n] = i;
            n += 1;
        }
    }
    std.mem.sort(u32, out[0..n], {}, std.sort.asc(u32));
    return n;
}

/// One head at step s.steps: the frame, selection, then per value head decay, write, read and its outputs.
pub fn run(s: *State, plan: u32, unit: usize, nz: []const u32, local: []const u32) ![]const Scalar {
    const p = &s.circuit.flow.plans[plan];
    const t = s.steps;
    const frames = s.frames[p.owner][0..];
    const memo = &s.memos[unit];
    const gi = s.circuit.flow.plan_group[plan];
    const words = if (gi != none and !s.always) try grouped(s, p, &s.groups[gi], memo, nz, local) else blk: {
        const n_key = selKey(s, p, nz, local);
        const key = s.key[0..n_key];
        const h = Memo.hash(key);
        const hit = if (s.always or memo.off() or memo.cold()) null else memo.find(key, h);
        if (hit) |w| break :blk w;
        const w = select(s, p, s.nz2[0..n_key]);
        if (!memo.off()) try memo.put(s.gpa, key, h, w);
        break :blk w;
    };
    var newest: u64 = 0;
    for (frames) |fr| newest = @max(newest, fr.stamp);
    if (newest != t + 1) {
        const in = oldest(Frame, frames);
        in.stamp = 0;
        in.dims.clearRetainingCapacity();
        try in.dims.appendSlice(s.gpa, nz);
        try in.vals.resize(s.gpa, nz.len);
        for (nz, in.vals.items) |d, *v| v.* = s.x[d];
        in.stamp = t + 1;
    }
    const picks = s.picks[plan][0..];
    const f = oldest(Picks, picks);
    f.stamp = 0;
    f.words.clearRetainingCapacity();
    try f.words.appendSlice(s.gpa, words);
    f.stamp = t + 1;
    var views: Views = .{ .s = s, .frames = frames, .t = t };
    defer views.clear();
    const hd0 = p.head();
    var key_row: ?u32 = null;
    var rows: [hd]u32 = undefined;
    var n_rows: usize = 0;
    if (p.general) {
        for (p.k, hd0.k) |k, pk| if (views.conv(p, pk.column) > 0) {
            if (key_row != null) return error.CircuitKeyNotOneHot;
            key_row = k.at;
        };
        for (p.q, 0..) |q, i| if (views.conv(p, hd0.q[i].column) > 0) {
            rows[n_rows] = q.at;
            n_rows += 1;
        };
    } else {
        var idx: [hd]u32 = undefined;
        const nk = positives(p, picks, t, 1, &idx);
        if (nk > 1) return error.CircuitKeyNotOneHot;
        if (nk == 1) key_row = p.k[idx[0]].at;
        n_rows = positives(p, picks, t, 0, &idx);
        for (idx[0..n_rows], rows[0..n_rows]) |i, *r| r.* = p.q[i].at;
    }
    const gates = f.words.items[f.words.items.len - 2 * hd0.value_heads.len ..];
    const values = s.work[0..p.h.outs.len];
    const v = s.work[values.len..][0..hd];
    const z_vals = s.work[values.len + hd ..];
    s.unstable[unit] = false;
    for (hd0.value_heads, 0..) |*vh, j| {
        const sel_a: Scalar = @bitCast(gates[2 * j]);
        if (p.a[j] != none and sel_a != model.keep_selector) {
            try decay(s, vh.table, sel_a);
            s.unstable[unit] = true;
        }
        if (key_row) |r| if (try writeEnabled(@bitCast(gates[2 * j + 1]))) {
            @memset(v, 0);
            for (vh.v) |pk| v[pk.at] = views.conv(p, pk.column);
            try s.writeRow(vh.table, r, v);
        };
        const out = values[vh.out_first..][0..vh.out_cols.len];
        const norm = try readNorm(s, vh.table, rows[0..n_rows]) orelse {
            @memset(out, 0);
            continue;
        };
        const zv = z_vals[0 .. p.zstart[j + 1] - p.zstart[j]];
        for (zv, p.zrep[p.zstart[j]..p.zstart[j + 1]]) |*z, col| z.* = @max(views.proj(p, col, 0), 0);
        for (vh.out_cols, vh.out_w, out, p.zcol[vh.out_first..][0..out.len]) |col, w, *y, zc| {
            var o: Scalar = 0;
            var any = false;
            for (rows[0..n_rows]) |r| if (s.rowPresent(vh.table, r)) {
                o = if (any) o + s.row(vh.table, r)[col] else 0 + s.row(vh.table, r)[col];
                any = true;
            };
            const z: Scalar = if (zc == none) 0 else zv[zc - p.zstart[j]];
            y.* = (o / norm) * z * w;
        }
    }
    return values;
}

/// The memo key (input index, value bits) and dims of a run's nonzero selecting inputs, in s.key and s.nz2.
fn selKey(s: *State, p: *const Plan, nz: []const u32, local: []const u32) usize {
    var n: usize = 0;
    for (nz, local) |d, j| if (p.sel_dim[j]) {
        s.key[n] = @as(u64, j) << 32 | @as(u32, @bitCast(s.x[d]));
        s.nz2[n] = d;
        n += 1;
    };
    return n;
}

/// A grouped head's selection: a member keeps its words per group generation, so its hit is a one-word compare.
fn grouped(s: *State, p: *const Plan, g: *Group, memo: *Memo, nz: []const u32, local: []const u32) ![]const u32 {
    if (g.stamp != s.steps + 1) {
        g.stamp = s.steps + 1;
        g.gen = 0;
        if (!(g.memo.off() or g.memo.cold())) {
            const key = s.key[0..selKey(s, p, nz, local)];
            g.hash = Memo.hash(key);
            if (g.memo.find(key, g.hash)) |w| g.gen = @as(u64, w[1]) << 32 | w[0] else {
                g.next += 1;
                try g.memo.put(s.gpa, key, g.hash, &[_]u32{ @truncate(g.next), @intCast(g.next >> 32) });
                g.gen = g.next;
            }
        }
    }
    const hit = if (g.gen != 0 and !memo.off()) memo.find(&[_]u64{g.gen}, g.hash) else null;
    const w = hit orelse select(s, p, s.nz2[0..selKey(s, p, nz, local)]);
    if (g.gen == 0) return w;
    if (hit == null and !memo.off()) try memo.put(s.gpa, &[_]u64{g.gen}, g.hash, w);
    if (p.member < 64 and emptyWords(p, w)) {
        const slot = g.hash % memo_slots;
        if (g.slot_gen[slot] != g.gen) {
            g.slot_gen[slot] = g.gen;
            g.empty[slot] = 0;
        }
        g.empty[slot] |= @as(u64, 1) << @intCast(p.member);
    }
    return w;
}

/// No query, no key and no decay: the run reads and writes nothing, and every output is +0.
fn emptyWords(p: *const Plan, w: []const u32) bool {
    if (w[0] != 0 or w[1] != 0) return false;
    for (p.a, 0..) |ai, j| {
        const sel_a: Scalar = @bitCast(w[2 + 2 * j]);
        if (ai != none and sel_a != model.keep_selector) return false;
    }
    return true;
}

/// A grouped head skips a run that changes nothing: outputs all +0, no selection now or reaching now from older runs.
pub fn skips(s: *State, plan: u32, unit: usize) !bool {
    const p = &s.circuit.flow.plans[plan];
    const gi = s.circuit.flow.plan_group[plan];
    if (gi == none or s.always or p.member >= 64 or s.unstable[unit] or !s.rest[unit]) return false;
    const g = &s.groups[gi];
    if (g.stamp != s.steps + 1 or g.gen == 0 or g.slot_gen[g.hash % memo_slots] != g.gen) return false;
    if (g.empty[g.hash % memo_slots] >> @intCast(p.member) & 1 == 0) return false;
    if (p.lag0) return true;
    // A lagged query or key reads an older run's picks: skip only if none reaches this step, and record empty picks.
    const picks = s.picks[plan][0..];
    var idx: [hd]u32 = undefined;
    if (positives(p, picks, s.steps, 0, &idx) + positives(p, picks, s.steps, 1, &idx) > 0) return false;
    const f = oldest(Picks, picks);
    f.stamp = 0;
    f.words.clearRetainingCapacity();
    try f.words.appendSlice(s.gpa, &.{ 0, 0 });
    f.stamp = s.steps + 1;
    return true;
}

/// beta: absent is 1; 1 writes, an integer <= 0 does not, anything else breaks the circuit's exact range.
fn writeEnabled(b: Scalar) !bool {
    if (b == 1) return true;
    if (b <= 0 and b == @round(b)) return false;
    return error.CircuitWriteGate;
}

fn decay(s: *State, table: u32, selector: Scalar) !void {
    if (!s.writable(table)) return error.CircuitTableReadOnly;
    if (selector == model.halve_selector) {
        var live = s.present[table];
        while (live != 0) : (live &= live - 1) {
            for (s.row(table, @ctz(live))) |*e| e.* *= 0.5;
        }
        return;
    }
    if (selector != model.clear_selector) return error.CircuitDecaySelector;
    @memset(s.tables[table], 0);
    s.present[table] = 0;
}

/// The ballast norm sqrt((o0^2 + o1^2) / hd) of the summed query rows; null when no row is present or it is zero.
fn readNorm(s: *State, table: u32, rows: []const u32) !?Scalar {
    var o: [2]Scalar = .{ 0, 0 };
    var any = false;
    for (rows) |r| if (s.rowPresent(table, r)) {
        const row = s.row(table, r);
        o = .{ o[0] + row[0], o[1] + row[1] };
        any = true;
    };
    if (!any) return null;
    const ref = o[0] * o[0] + o[1] * o[1];
    if (ref == 0) {
        for (0..hd) |c| {
            var sum: Scalar = 0;
            for (rows) |r| if (s.rowPresent(table, r)) {
                sum += s.row(table, r)[c];
            };
            if (sum != 0) return error.CircuitReadBallast;
        }
        return null;
    }
    return @sqrt(ref / @as(Scalar, hd));
}

/// Canonical history of this head's channels at step `t`: projections of steps t-1, t-2, t-3, newest first.
pub fn history(s: *State, plan: u32, out: []u8, at: *usize) void {
    const p = &s.circuit.flow.plans[plan];
    var views: Views = .{ .s = s, .frames = s.frames[p.owner][0..], .t = s.steps };
    defer views.clear();
    for (0..p.h.channels) |c| for (1..model.lags) |lag| {
        const v: Scalar = if (views.dense(@intCast(lag)) == null) 0 else views.proj(p, @intCast(c), @intCast(lag));
        const w: f64 = v;
        std.mem.writeInt(u64, out[at.*..][0..8], @bitCast(w + 0), .little);
        at.* += 8;
    };
}
