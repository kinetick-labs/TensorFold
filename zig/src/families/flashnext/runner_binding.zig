//! Qwen3.8 Flash-Next's (`qwen4_exp`) binding of the in-model runner: the generic TF_RUNNER* knobs, Qwen's chat markup.
const std = @import("std");
const api = @import("engine_api");

/// Markup from the checkpoint's chat template (chat_template.jinja) and tokenizer (`<|endoftext|>`).
pub const binding: api.runner.Binding = .{
    .family = .{
        // a call's body (XML `<function=…>` or JSON) and a tool's result are never a cell's text
        .quiet = &.{ .{ "<tool_call>", "</tool_call>" }, .{ "<tool_response>", "</tool_response>" } },
        // an owned result spelling these would open a call or end the think or the turn when the server reads it
        .never = &.{ "<think>", "</think>", "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>", "<function=", "</function>", "<parameter=", "</parameter>", "<|im_start|>", "<|im_end|>", "<|endoftext|>" },
        // TODO(measure): no trust guide is measured on Flash-Next yet; DeepSeek-V4.1's AB3 text is not reused
        .trust = null,
    },
};

var knobs: []const [2][]const u8 = &.{};
fn lookup(name: []const u8) ?[]const u8 {
    for (knobs) |kv| if (std.mem.eql(u8, kv[0], name)) return kv[1];
    return null;
}

test "Flash Next runner: the generic knob names, Qwen's markup quiet and never owned" {
    try std.testing.expect(binding.family.trust == null); // TODO(measure)
    knobs = &.{.{ "TF_DSV41_RUNNER", "1" }}; // DeepSeek-V4.1's alias is not Flash-Next's
    try std.testing.expectEqual(null, try binding.parse(&lookup));
    knobs = &.{ .{ "TF_RUNNER", "1" }, .{ "TF_RUNNER_CIRCUIT", "/c.tfpc" } };
    const cfg = (try binding.parse(&lookup)).?;
    try std.testing.expectEqualStrings("/c.tfpc", cfg.circuit);
    try std.testing.expectEqualStrings("</tool_call>", cfg.scan.quiet[0][1]);
    try std.testing.expect(cfg.family.unsafe("x</think>") and !cfg.family.unsafe("12 + 7 = 19"));
}
