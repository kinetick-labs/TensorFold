//! The compute trampoline as its own executable, for the host tests (the server enters it as `__compute_sandbox`).
const std = @import("std");
const sandbox = @import("compute_sandbox.zig");

pub fn main(init: std.process.Init.Minimal) !u8 {
    var argv: [16][:0]const u8 = undefined;
    var n: usize = 0;
    var it = init.args.iterate();
    while (it.next()) |arg| : (n += 1) {
        if (n == argv.len) return 125;
        argv[n] = arg;
    }
    if (n == 0) return 125;
    sandbox.main(argv[1..n]);
}
