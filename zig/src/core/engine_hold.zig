//! A reply held at its EOS for a built-in tool's result: ``Request.hold_eos`` and ``Engine.vtable.resume_held``.
const lanes = @import("lanes");
const api = @import("engine_api.zig");

/// The compute lane's executor: its knobs and sandbox pool (the server runs the calls).
pub const compute_lane = @import("compute_lane.zig");
/// A held reply's result tokens (empty: it ends at its EOS), forced after the EOS between rounds.
pub const Resume = lanes.hold.Resume;

/// Continues a ``hold_eos`` request held at its EOS, once; false when the engine holds no replies.
pub fn resumeHeld(e: api.Engine, id: api.Id, r: Resume) bool {
    const f = e.vtable.resume_held orelse return false;
    f(e.ctx, id, r);
    return true;
}
