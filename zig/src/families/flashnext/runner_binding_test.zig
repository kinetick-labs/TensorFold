//! Flash-Next's runner markup against the checkpoint's own chat template, when TF_FLASHNEXT_TEMPLATE names it.
const std = @import("std");
const binding = @import("runner_binding.zig").binding;

/// The checkpoint's template, from TF_FLASHNEXT_TEMPLATE (its chat_template.jinja; not vendored: Qwen's licence).
fn checkpointTemplate(a: std.mem.Allocator) !?[]u8 {
    const path = std.mem.span(std.c.getenv("TF_FLASHNEXT_TEMPLATE") orelse return null);
    var threaded: std.Io.Threaded = .init_single_threaded;
    return try std.Io.Dir.cwd().readFileAlloc(threaded.io(), path, a, .limited(1 << 20));
}

test "Flash Next runner: every marker is in the checkpoint's own chat template (TF_FLASHNEXT_TEMPLATE)" {
    const template = (try checkpointTemplate(std.testing.allocator)) orelse return error.SkipZigTest;
    defer std.testing.allocator.free(template);
    for (binding.family.quiet) |s| for (s) |m| try std.testing.expect(std.mem.indexOf(u8, template, m) != null);
    for (binding.family.never) |m| if (!std.mem.eql(u8, m, "<|endoftext|>"))
        try std.testing.expect(std.mem.indexOf(u8, template, m) != null);
    try std.testing.expect(std.mem.indexOf(u8, template, "'<|im_start|>assistant\\n' }}") != null);
}
