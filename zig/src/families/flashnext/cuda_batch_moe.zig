//! The shared pass's MoE write-back: the slots' fp32 sum, the adds and the kept branch.
const cuda = @import("cuda");
const wt = @import("cuda_prompt_wt.zig");
const batch = @import("cuda_batch.zig");
const LayerKeep = batch.LayerKeep;
const dims = wt.dims;
const slots = wt.slots;
const experts_n = wt.experts_n;

/// After a layer's experts, when something adds to or keeps its output: the slots' fp32 sum, adds, kept branch.
pub fn moeAdds(self: anytype, layer: usize, keep: LayerKeep) !void {
    const add = self.addAt(layer);
    if (add == null and keep.moe_branch == 0) return;
    const s = self.s;
    const n = self.rows;
    var a: cuda.Args = .{};
    a.add(s.y);
    a.add(s.wts);
    a.add(s.moe_f32);
    a.add(batch.cint(dims));
    a.add(batch.cint(slots));
    try cuda.launch.launch(self.k.moe_sum, .{ .grid = .{ .x = @intCast(n), .y = dims / 256 }, .block = .{ .x = 256 } }, self.stream, &a);
    if (add) |m| try m.add(m.ctx, layer, s.mixed, s.picks, s.wts, s.moe_f32, n);
    if (keep.moe_branch != 0) try self.sumSlices(s.moe_f32, keep.moe_branch, true, n * dims, 1);
    self.pending_f32 = true;
}

/// The native NVFP4 experts of the rows into s.y: prompt chunks on the shared-memory tile, else the decode kernel.
pub fn nativeRouted(self: anytype, native: anytype, plan: cuda.grouped.Plan) !void {
    const s = self.s;
    const n = self.rows;
    const pairs = n * slots;
    const e = experts_n + 1;
    const layer = native.experts.view();
    const up_in: cuda.experts.Rows = .{ .x = s.mixed, .stride = dims, .slots = slots };
    const down_in: cuda.experts.Rows = .{ .x = s.moe_act, .stride = layer.width };
    const gu = if (self.k.expert_tile) |*t| (if (t.cfg.applies(n, native.world)) t else null) else null;
    const dn = if (self.k.expert_tile) |*t| (if (t.cfg.appliesDown(n, native.world)) t else null) else null;
    const gu_pairs: usize = if (gu) |t| t.cfg.pairs else 16;
    try self.k.router.route(self.stream, s.picks, pairs, e, gu_pairs, plan);
    const gu_items = cuda.grouped.maxItems(pairs, e, gu_pairs);
    if (gu) |t| try t.run(self.stream, .up, up_in, layer, plan, s.moe_act, gu_items, experts_n) else try self.k.nvfp4.run(self.stream, .up, up_in, layer, plan, s.moe_act, gu_items, experts_n);
    const dn_pairs: usize = if (dn) |t| t.cfg.pairs else 16;
    if (dn_pairs != gu_pairs) try self.k.router.route(self.stream, s.picks, pairs, e, dn_pairs, plan);
    const dn_items = cuda.grouped.maxItems(pairs, e, dn_pairs);
    if (dn) |t| try t.run(self.stream, .down_f32, down_in, layer, plan, s.y, dn_items, experts_n) else try self.k.nvfp4.run(self.stream, .down_f32, down_in, layer, plan, s.y, dn_items, experts_n);
}
