//! Borrowed media state travels through the main pass, verification and the MTP head without another owner.
const cuda = @import("cuda");
const shared = @import("qwen_vision");
const wt = @import("cuda_prompt_wt.zig");
pub const State = struct {
    image: shared.media.Media,
    prep: cuda.Function,
    pool: cuda.Function,
    /// The prep kernel for fp8 caches (--kv-cache fp8); null: bf16 caches.
    prep8: ?cuda.Function = null,
    pub fn view(s: *const State) View {
        return .{ .image = s.image.view(), .prep = s.prep, .pool = s.pool, .prep8 = s.prep8 };
    }
    pub fn deinit(s: *State) void {
        s.image.deinit();
    }
};
pub const View = struct {
    image: shared.media.View,
    prep: cuda.Function,
    pool: cuda.Function,
    prep8: ?cuda.Function = null,
    pub fn attn(v: View, stream: cuda.Stream, proj: u64, pos: u64, qw: u64, kw: u64, iw: u64, inv: u64, q: u64, kc: u64, vc: u64, iq: u64, ikc: u64, eps: f32, rows: usize) !void {
        return v.attnWith(v.prep, stream, proj, pos, qw, kw, iw, inv, q, kc, vc, iq, ikc, eps, rows);
    }
    /// attn into fp8 caches (cuda_kv.zig); the same arguments and grid.
    pub fn attnKv8(v: View, stream: cuda.Stream, proj: u64, pos: u64, qw: u64, kw: u64, iw: u64, inv: u64, q: u64, kc: u64, vc: u64, iq: u64, ikc: u64, eps: f32, rows: usize) !void {
        return v.attnWith(v.prep8 orelse return error.Fp8KvUnsupported, stream, proj, pos, qw, kw, iw, inv, q, kc, vc, iq, ikc, eps, rows);
    }
    fn attnWith(v: View, prep: cuda.Function, stream: cuda.Stream, proj: u64, pos: u64, qw: u64, kw: u64, iw: u64, inv: u64, q: u64, kc: u64, vc: u64, iq: u64, ikc: u64, eps: f32, rows: usize) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ proj, pos, qw, kw, iw, inv, q, kc, vc, iq, ikc }) |x| a.add(x);
        a.add(eps);
        for ([_]usize{ wt.proj_n, wt.q_heads, wt.kv_heads, wt.index_heads, wt.index_dim, wt.half }) |x| a.add(@as(i32, @intCast(x)));
        a.add(v.image.rope);
        a.add(@as(i32, @intCast(v.image.n)));
        a.add(v.image.delta);
        try cuda.launch.launch(prep, .{ .grid = .{ .x = @intCast(rows), .y = wt.q_heads + wt.kv_heads + wt.index_heads + 1, .z = 1 }, .block = .{ .x = 256, .y = 1, .z = 1 } }, stream, &a);
    }
    pub fn pooled(v: View, stream: cuda.Stream, ikc: u64, out: u64, pos: u64, w: u64, inv: u64, eps: f32, rows: usize) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ ikc, out, pos, w, inv }) |x| a.add(x);
        a.add(eps);
        for ([_]usize{ rows, wt.index_dim, wt.half, wt.index_ratio }) |x| a.add(@as(i32, @intCast(x)));
        a.add(v.image.rope);
        a.add(@as(i32, @intCast(v.image.n)));
        a.add(v.image.delta);
        try cuda.launch.launch(v.pool, .{ .grid = .{ .x = @intCast(rows / wt.index_ratio + 2), .y = 1, .z = 1 }, .block = .{ .x = wt.index_dim, .y = 1, .z = 1 } }, stream, &a);
    }
};

pub fn inject(view: ?View, rows: anytype, stream: cuda.Stream, out: u64, pos: usize, n: usize) !void {
    if (rows.len == 0) {
        if (view) |m| try m.image.inject(stream, out, pos, n, wt.streams);
    } else {
        var row0: usize = 0;
        for (rows) |l| {
            if (l.seq.media) |m| try m.image.inject(stream, out + row0 * wt.wide * 2, l.pos, l.rows, wt.streams);
            row0 += l.rows;
        }
    }
}

// The head embeds the next token, one logical row past the hidden state it conditions on.
pub fn injectNext(view: ?View, stream: cuda.Stream, out: u64, pos: usize, rows: usize) !void {
    if (view) |m| try m.image.inject(stream, out, pos + 1, rows, wt.streams);
}
test {
    _ = @import("cuda_media_test.zig");
}
