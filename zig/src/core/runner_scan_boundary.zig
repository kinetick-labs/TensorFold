const std = @import("std");

const header_words = [_][]const u8{
    "result",      "results",      "output",           "outputs",            "it prints",
    "this prints", "prints",       "running it gives", "running this gives", "expected output",
    "the output",  "this outputs", "it outputs",       "the result",         "output is",
    "running it",  "running this", "executing it",     "executing this",
};

/// A short line naming the result and ending in ':' ("Result:", "**Output:**", "Running this gives:").
pub fn complete(line: []const u8) bool {
    var t = std.mem.trim(u8, line, " \t*#->_");
    if (t.len < 2 or t.len > 60 or t[t.len - 1] != ':') return false;
    t = std.mem.trim(u8, t[0 .. t.len - 1], " \t*_");
    var buf: [64]u8 = undefined;
    const low = std.ascii.lowerString(&buf, t);
    for (header_words) |w| if (std.mem.eql(u8, low, w)) return true;
    return false;
}

pub fn ready(line: []const u8) bool {
    if (!complete(line)) return false;
    const t = std.mem.trim(u8, line, " \t");
    for ([_][]const u8{ "**", "__", "*", "_" }) |wrap| {
        if (std.mem.startsWith(u8, t, wrap)) return std.mem.endsWith(u8, t, wrap);
    }
    return true;
}

pub fn indentOk(line: []const u8) ?[]const u8 {
    var i: usize = 0;
    while (i < line.len and line[i] == ' ') i += 1;
    return if (i <= 3) line[i..] else null;
}

/// "```lang" (lang may be empty; result-like tags excluded): the lowercase-able tag, else null.
pub fn fenceOpen(line: []const u8) ?[]const u8 {
    const t = indentOk(line) orelse return null;
    if (!std.mem.startsWith(u8, t, "```")) return null;
    const tag = std.mem.trim(u8, t[3..], " \t");
    for (tag) |c| if (!(std.ascii.isAlphanumeric(c) or c == '+' or c == '#' or c == '-' or c == '_' or c == '.')) return null;
    if (isResultTag(tag)) return null;
    return tag;
}

pub fn fenceClose(line: []const u8) bool {
    const t = indentOk(line) orelse return false;
    return std.mem.eql(u8, std.mem.trimEnd(u8, t, " \t"), "```");
}

const result_tags = [_][]const u8{ "output", "text", "console", "stdout", "plaintext", "txt", "result" };

fn isResultTag(tag: []const u8) bool {
    for (result_tags) |r| if (std.ascii.eqlIgnoreCase(tag, r)) return true;
    return false;
}
