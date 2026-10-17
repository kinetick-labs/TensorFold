//! TF_FLASHNEXT_SPEC_SAMPLE=1 (sampled streams, one rank): standard speculative sampling (Leviathan et al. 2023,
//! Chen et al. 2023) in place of the keyed match. A draft x drawn from the head's truncated distribution q is kept
//! with probability min(1, p(x)/q(x)), p the target's truncated distribution at that row (the same top_k / top_p /
//! min_p / temperature cut as sampling.choose); at the first refusal the row takes a draw from norm(max(p - q, 0)).
//! Each emitted token is then distributed exactly as the plain keyed sampler's (lossless in distribution), but the
//! draws use other keys than the plain sampler's, so a sampled reply's tokens (and token_sha) differ from the
//! match rule's. Greedy rows never come here.
//! Keys: the draft draws with seed mix(seed ^ salt_draft), the keep coin with mix(seed ^ salt_coin) and the
//! residual with mix(seed ^ salt_rest), each keyed by position; rows without a head draft (a host draft, a
//! repeated held draft, the last row) take the plain keyed draw. No decision reads keys another decision read.
const std = @import("std");
const lanes = @import("lanes");
const Allocator = std.mem.Allocator;
const S = lanes.sampling;

extern "c" fn exp(x: f64) f64;
extern "c" fn log(x: f64) f64;

/// The most kept candidates a distribution holds (top_k 20 + margin read back, any top_k up to it).
pub const cap = 64;

const salt_draft: u64 = 0x5350_4543_4452_4654; // "SPECDRFT"
const salt_coin: u64 = 0x5350_4543_434f_494e; // "SPECCOIN"
const salt_rest: u64 = 0x5350_4543_5245_5354; // "SPECREST"

/// A row's truncated distribution: the kept ids and their probabilities (sum 1), at `position`, with the token
/// drawn from it (drafts) or 0.
pub const Dist = struct {
    position: u64 = 0,
    token: u64 = 0,
    n: usize = 0,
    ids: [cap]u64 = undefined,
    probs: [cap]f64 = undefined,

    pub fn prob(d: *const Dist, id: u64) f64 {
        for (d.ids[0..d.n], d.probs[0..d.n]) |i, p| if (i == id) return p;
        return 0.0;
    }
};

/// The sampling a stream's head drafts with under speculative sampling: its own keys, apart from the target's.
pub fn draftSampling(s: S.Sampling) S.Sampling {
    var x = s;
    x.seed = S.mix(s.seed ^ salt_draft);
    return x;
}

/// Whether a sampler's distribution can be formed from the candidates read back (a top_k cut within `cap`).
pub fn fits(s: ?S.Sampling) bool {
    const x = s orelse return false;
    return x.temperature > 0 and x.top_k != 0 and x.top_k <= cap;
}

const Cand = struct { value: f64, id: u64 };
fn byValueThenId(_: void, a: Cand, b: Cand) bool {
    return a.value > b.value or (a.value == b.value and a.id < b.id);
}

/// sampling.choose's kept set and its softmax over `values` (f32 logits) with `ids`: the distribution its keyed
/// Gumbel draw follows. False when it cannot be formed here (no top_k cut, or past `cap`).
pub fn dist(gpa: Allocator, values: []const f32, ids: []const u64, s: S.Sampling, out: *Dist) !bool {
    if (!fits(s)) return false;
    const width = ids.len;
    const order = try gpa.alloc(Cand, width);
    defer gpa.free(order);
    for (order, values, ids) |*o, v, id| o.* = .{ .value = v, .id = id };
    std.mem.sort(Cand, order, {}, byValueThenId);
    const k = @max(1, @min(@as(usize, s.top_k), width));
    if (k > cap) return false;
    var scaled: [cap]f64 = undefined;
    const t = @max(s.temperature, 1e-6);
    for (scaled[0..k], order[0..k]) |*x, o| x.* = o.value / t;
    var keep = k;
    var probs: [cap]f64 = undefined;
    if (s.top_p > 0.0 and s.top_p < 1.0) {
        for (probs[0..k], scaled[0..k]) |*p, x| p.* = exp(x - scaled[0]);
        const total = S.pairwiseSum(probs[0..k]);
        var cum: f64 = 0.0;
        var below: usize = 0;
        for (probs[0..k]) |*p| {
            p.* /= total;
            cum += p.*;
            if (cum < s.top_p) below += 1;
        }
        keep = @min(keep, below + 1);
    }
    if (s.min_p > 0.0) {
        const floor = scaled[0] + s.minLog();
        var n: usize = 0;
        for (scaled[0..keep]) |x| {
            if (x >= floor) n += 1;
        }
        keep = n;
    }
    // Gumbel-max over scaled[0..keep] draws softmax(scaled[0..keep])
    var sum: f64 = 0.0;
    for (0..keep) |j| {
        out.probs[j] = exp(scaled[j] - scaled[0]);
        sum += out.probs[j];
    }
    for (0..keep) |j| {
        out.probs[j] /= sum;
        out.ids[j] = order[j].id;
    }
    out.n = keep;
    return true;
}

pub const Verdict = union(enum) { keep, take: u64 };

/// The speculative decision at `position` for draft `draft` drawn from `q`, the target's distribution `p`:
/// keep it, or take the residual's draw (never the draft: its residual mass is 0).
pub fn decide(p: *const Dist, q: *const Dist, draft: u64, seed: u64, position: u64) Verdict {
    const px = p.prob(draft);
    const qx = q.prob(draft);
    const u = S.uniform(S.mix(seed ^ salt_coin), position, 0);
    if (qx > 0.0 and u * qx <= px) return .keep;
    // residual max(p - q, 0) over p's support, drawn by Gumbel-max on its own keys
    const rs = S.mix(seed ^ salt_rest);
    var best: ?u64 = null;
    var best_score = -std.math.inf(f64);
    for (p.ids[0..p.n], p.probs[0..p.n]) |id, pp| {
        const r = pp - q.prob(id);
        if (!(r > 0.0)) continue;
        const score = log(r) - log(-log(S.uniform(rs, position, id)));
        if (best == null or score > best_score) {
            best = id;
            best_score = score;
        }
    }
    if (best) |b| return .{ .take = b };
    // p <= q everywhere but the draft refused: only rounding can land here; p's own draw on the residual keys
    var arg: u64 = p.ids[0];
    best_score = -std.math.inf(f64);
    for (p.ids[0..p.n], p.probs[0..p.n]) |id, pp| {
        if (id == draft or !(pp > 0.0)) continue;
        const score = log(pp) - log(-log(S.uniform(rs, position, id)));
        if (score > best_score) {
            arg = id;
            best_score = score;
        }
    }
    return .{ .take = arg };
}

/// A tree node's siblings (multi-draft, SpecInfer / Miao et al. 2024 recursive rejection): `drafts` drawn
/// i.i.d. from `q` at one position. Sibling i is kept with probability min(1, p_i(x)/q(x)), p_0 = p; after its
/// refusal p_{i+1} = norm(max(p_i - q, 0)); past every sibling the draw is from the last p. Returns the kept
/// sibling's index, or the residual's token. The engine's windows are chains (cuda_lanes refuses parents:
/// TreesNotBuilt), so only `decide` runs today; this is the rule a tree verify must use.
pub const Multi = union(enum) { keep: usize, take: u64 };
pub fn decideMulti(p: *const Dist, q: *const Dist, drafts: []const u64, seed: u64, position: u64) Multi {
    var cur = p.*;
    const rs = S.mix(seed ^ salt_rest);
    for (drafts, 0..) |x, i| {
        const px = cur.prob(x);
        const qx = q.prob(x);
        const u = S.uniform(S.mix(seed ^ salt_coin), position, i);
        if (qx > 0.0 and u * qx <= px) return .{ .keep = i };
        var sum: f64 = 0;
        for (cur.ids[0..cur.n], cur.probs[0..cur.n]) |id, *pp| {
            pp.* = @max(pp.* - q.prob(id), 0.0);
            sum += pp.*;
        }
        if (!(sum > 0.0)) break;
        for (cur.probs[0..cur.n]) |*pp| pp.* /= sum;
    }
    return .{ .take = drawFrom(&cur, rs, position) };
}

/// The keyed Gumbel draw from `d` with `seed` at `position` (the draft's draw over its distribution).
fn drawFrom(d: *const Dist, seed: u64, position: u64) u64 {
    var arg: u64 = d.ids[0];
    var best = -std.math.inf(f64);
    for (d.ids[0..d.n], d.probs[0..d.n]) |id, p| {
        if (!(p > 0.0)) continue;
        const score = log(p) - log(-log(S.uniform(seed, position, id)));
        if (score > best) {
            arg = id;
            best = score;
        }
    }
    return arg;
}

fn mk(ids: []const u64, w: []const f64) Dist {
    var d: Dist = .{};
    var sum: f64 = 0;
    for (w) |x| sum += x;
    for (ids, w, 0..) |id, x, j| {
        d.ids[j] = id;
        d.probs[j] = x / sum;
    }
    d.n = ids.len;
    return d;
}

test "speculative sampling emits the target's distribution (chi-square) and keeps sum min(p, q) of drafts" {
    // p and q overlap partly; q has a token p lacks and misses one p has
    const p = mk(&.{ 3, 7, 11, 13, 17, 19 }, &.{ 0.40, 0.25, 0.15, 0.10, 0.06, 0.04 });
    const q = mk(&.{ 7, 3, 11, 23, 13 }, &.{ 0.45, 0.20, 0.20, 0.10, 0.05 });
    const seed: u64 = 12345;
    const n: usize = 400_000;
    var counts: [6]usize = @splat(0);
    var kept: usize = 0;
    var coupled: usize = 0;
    const ds = S.mix(seed ^ salt_draft);
    for (0..n) |pos| {
        const x = drawFrom(&q, ds, pos);
        const tok = switch (decide(&p, &q, x, seed, pos)) {
            .keep => blk: {
                kept += 1;
                break :blk x;
            },
            .take => |t| t,
        };
        // the match rule's acceptance for comparison: same keys for draft and target
        if (drawFrom(&q, seed, pos) == drawFrom(&p, seed, pos)) coupled += 1;
        var hit = false;
        for (p.ids[0..p.n], 0..) |id, j| if (id == tok) {
            counts[j] += 1;
            hit = true;
        };
        try std.testing.expect(hit);
    }
    var chi: f64 = 0;
    for (counts, p.probs[0..6]) |c, pr| {
        const e = pr * @as(f64, @floatFromInt(n));
        const d = @as(f64, @floatFromInt(c)) - e;
        chi += d * d / e;
    }
    // df 5: P(chi2 > 25) ~ 1.4e-4
    try std.testing.expect(chi < 25.0);
    var overlap: f64 = 0;
    for (p.ids[0..p.n], p.probs[0..p.n]) |id, pr| overlap += @min(pr, q.prob(id));
    const rate = @as(f64, @floatFromInt(kept)) / @as(f64, @floatFromInt(n));
    try std.testing.expect(@abs(rate - overlap) < 0.005);
    const crate = @as(f64, @floatFromInt(coupled)) / @as(f64, @floatFromInt(n));
    if (std.c.getenv("SPEC_PROBE") != null) std.debug.print("spec chi2 {d:.2} (df 5) keep {d:.4} vs sum min {d:.4}; keyed match {d:.4}\n", .{ chi, rate, overlap, crate });
    if (std.c.getenv("SPEC_PROBE") != null) return error.Probe;
}

test "dist follows choose's cut: its draws are choose's kept set" {
    const gpa = std.testing.allocator;
    const values = [_]f32{ 5.0, 4.5, 4.4, 1.0, 0.5, -3.0, 4.5 };
    const ids = [_]u64{ 10, 11, 12, 13, 14, 15, 9 };
    const s: S.Sampling = .{ .seed = 1, .temperature = 0.7, .top_k = 5, .top_p = 0.95 };
    var d: Dist = .{};
    try std.testing.expect(try dist(gpa, &values, &ids, s, &d));
    var sum: f64 = 0;
    for (d.probs[0..d.n]) |x| sum += x;
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), sum, 1e-12);
    // every keyed draw lands in the kept set, and the kept set's draws match the distribution's support
    for (0..2000) |pos| {
        const t = try S.choose(gpa, &[_]f64{ 5.0, 4.5, 4.4, 1.0, 0.5, -3.0, 4.5 }, &ids, pos, s);
        try std.testing.expect(d.prob(t) > 0);
    }
    try std.testing.expect(!(try dist(gpa, &values, &ids, .{ .seed = 1, .temperature = 0.7, .top_k = 0 }, &d)));
}

test "recursive rejection over three i.i.d. siblings emits the target's distribution (chi-square)" {
    const p = mk(&.{ 3, 7, 11, 13, 17, 19 }, &.{ 0.40, 0.25, 0.15, 0.10, 0.06, 0.04 });
    const q = mk(&.{ 7, 3, 11, 23, 13 }, &.{ 0.45, 0.20, 0.20, 0.10, 0.05 });
    const seed: u64 = 777;
    const n: usize = 400_000;
    var counts: [6]usize = @splat(0);
    var kept: usize = 0;
    for (0..n) |pos| {
        // three siblings from q, each on its own key stream
        var xs: [3]u64 = undefined;
        for (&xs, 0..) |*x, i| x.* = drawFrom(&q, S.mix(seed ^ salt_draft ^ (@as(u64, i) << 32)), pos);
        const tok = switch (decideMulti(&p, &q, &xs, seed, pos)) {
            .keep => |i| blk: {
                kept += 1;
                break :blk xs[i];
            },
            .take => |t| t,
        };
        var hit = false;
        for (p.ids[0..p.n], 0..) |id, j| if (id == tok) {
            counts[j] += 1;
            hit = true;
        };
        try std.testing.expect(hit);
    }
    var chi: f64 = 0;
    for (counts, p.probs[0..6]) |c, pr| {
        const e = pr * @as(f64, @floatFromInt(n));
        const d = @as(f64, @floatFromInt(c)) - e;
        chi += d * d / e;
    }
    try std.testing.expect(chi < 25.0);
    const rate = @as(f64, @floatFromInt(kept)) / @as(f64, @floatFromInt(n));
    // three siblings keep more than one (0.65)
    try std.testing.expect(rate > 0.70);
    if (std.c.getenv("SPEC_PROBE") != null) {
        std.debug.print("multi chi2 {d:.2} (df 5) keep {d:.4}\n", .{ chi, rate });
        return error.Probe;
    }
}

test "the chi-square check has power: siblings each judged against the unchanged p miss the target" {
    const p = mk(&.{ 3, 7, 11, 13, 17, 19 }, &.{ 0.40, 0.25, 0.15, 0.10, 0.06, 0.04 });
    const q = mk(&.{ 7, 3, 11, 23, 13 }, &.{ 0.45, 0.20, 0.20, 0.10, 0.05 });
    const seed: u64 = 777;
    const n: usize = 400_000;
    var counts: [6]usize = @splat(0);
    for (0..n) |pos| {
        var tok: ?u64 = null;
        for (0..3) |i| {
            const x = drawFrom(&q, S.mix(seed ^ salt_draft ^ (@as(u64, i) << 32)), pos);
            if (decide(&p, &q, x, seed ^ (@as(u64, i) << 40), pos) == .keep) {
                tok = x;
                break;
            }
        }
        const t = tok orelse switch (decide(&p, &q, 23, seed, pos)) {
            .keep => unreachable,
            .take => |r| r,
        };
        for (p.ids[0..p.n], 0..) |id, j| if (id == t) {
            counts[j] += 1;
        };
    }
    var chi: f64 = 0;
    for (counts, p.probs[0..6]) |c, pr| {
        const e = pr * @as(f64, @floatFromInt(n));
        const d = @as(f64, @floatFromInt(c)) - e;
        chi += d * d / e;
    }
    try std.testing.expect(chi > 100.0);
}
