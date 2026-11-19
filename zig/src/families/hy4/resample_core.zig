//! `picture_resample` for builds that also link core (which owns zig/src/core/picture/resample.zig): the same
//! resize through core, so preprocess.zig reads one copy of the file.
const r = @import("core").picture.resample;
pub const resize = r.resize;
pub const Kind = r.Kind;
