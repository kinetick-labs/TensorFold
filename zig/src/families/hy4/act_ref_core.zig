//! `act_ref` for builds that also link core (which owns zig/src/core/act_ref.zig): the same functions through core.
pub const erf = @import("core").act_ref.erf;
