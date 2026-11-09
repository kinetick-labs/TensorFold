//! Inline modular steps (`A mod M =`, `A % M =`) as owned cells: exact spans from bn-011 and S2 round 2.
const std = @import("std");
const api = @import("runner_api.zig");
const graph = @import("runner_graph.zig");
const route = @import("runner_graph_route.zig");
const frontend = @import("runner_frontend.zig");
const scan = @import("runner_scan.zig");
const t = std.testing;

/// The line up to the model's `=`, the cell code, and CPython's value of it (`print((A) % (M))`).
const Golden = struct { []const u8, []const u8, []const u8 };

/// bn-011: smoke r0-583d0cd875f6 (t3-evidence); S2 round 2: trace2-20261101T235839Z (s2-trace-ab2); then variations.
pub const goldens = [_]Golden{
    // bn-011 smoke r0-583d0cd875f6
    .{ "So 2ab B = 699,892,476,819,640 *1,000,000,000. Mod 1e18, only last? Since B=1e9, coefficient mod 1e9 matters for term mod 1e18. 2ab mod 1e9 = 476,819,640? Wait 699,892,476,819,640 mod 1e9 =", "699892476819640 mod (1*10^9)", "476819640" },
    .{ "Now add b^2 = 35,665,017,849,089,281. Sum = 512,484,657,849,089,281. This is <1e18? 512 quadrillion = 5.12e17, yes. So 3^64 mod 1e18 =", "3^64 mod (1*10^18)", "512484657849089281" },
    .{ "Double mod 1e9 = 1,871,323,234 mod 1e9 =", "1871323234 mod (1*10^9)", "871323234" },
    .{ "Add b^2: 720,952,607,109,096,961 + 871,323,234,000,000,000 =1,592,275,841,109,096,961. Mod 1e18 subtract 1e18 =592,275,841,109,096,961. So 3^128 mod 1e18 =", "3^128 mod (1*10^18)", "592275841109096961" },
    .{ "Add b^2: 11,902,146,899,435,521 +653,638,402,000,000,000 =665,540,548,899,435,521. This is <1e18. So 3^256 mod 1e18 =", "3^256 mod (1*10^18)", "665540548899435521" },
    .{ "So 3^400 mod 10^18 =", "3^400 mod 10^18", "192443283334088001" },
    .{ "Compute 262144*? R mod. Use R mod 2^18. Since 10^6= 1000000 mod 262144 =", "1000000 mod 262144", "213568" },
    .{ "Now 3^400 = 3^(256+128+16) mod 262144 =", "3^(256+128+16) mod 262144", "116033" },
    // S2 round 2 exact/bn-002
    .{ "N = 10^30 -1. Need q = floor((10^30 -1)/97). Since 10^30 mod97=", "10^30 mod 97", "85" },
    .{ "Let's double-check remainder using modular arithmetic. 10^3=1000 mod97=", "1000 mod 97", "30" },
    .{ "Let's double-check remainder using modular arithmetic. 10^3=1000 mod97=30. 10^6=900 mod97=", "900 mod 97", "27" },
    .{ "Let's double-check remainder using modular arithmetic. 10^3=1000 mod97=30. 10^6=900 mod97=27? 97*9=873 rem27. 10^12=27^2=729 mod97=", "729 mod 97", "50" },
    .{ "Let's double-check remainder using modular arithmetic. 10^3=1000 mod97=30. 10^6=900 mod97=27? 97*9=873 rem27. 10^12=27^2=729 mod97=50. 10^24=50^2=2500 mod97=", "2500 mod 97", "75" },
    .{ "Let's double-check remainder using modular arithmetic. 10^3=1000 mod97=30. 10^6=900 mod97=27? 97*9=873 rem27. 10^12=27^2=729 mod97=50. 10^24=50^2=2500 mod97=75? 97*25=2425 rem75. 10^30=10^24*10^6=75*27=2025 mod97=", "2025 mod 97", "85" },
    // S2 round 2 exact/bn-008
    .{ "[exact: 930336977^2 mod 1000000007 =", "930336977^2 mod 1000000007", "714810341" },
    .{ "[exact: 714810341^2 mod 1000000007 =", "714810341^2 mod 1000000007", "23859541" },
    // S2 round 2 exact/bn-010
    .{ "[exact: 12345678901234567890 mod 900000000090 =", "12345678901234567890 mod 900000000090", "0" },
    // S2 round 2 exact/bn-011
    .{ "a*89 = a*(90-1)=46,123,619,130 -512,484,657=45,611,134,473. times1000 =45,611,134,473,000. Mod1e9: 134,473,000? Let's compute 45,611,134,473,000 mod1e9 =", "45611134473000 mod (1*10^9)", "134473000" },
    .{ "Add first term 669,000,000 =1,326,819,201 mod1e9 =", "1326819201 mod (1*10^9)", "326819201" },
    .{ "Sum a1*96,961 mod1e9=892,608,000+584,466,628=1,477,074,628 mod1e9=", "1477074628 mod (1*10^9)", "477074628" },
    .{ "Add first term 732,000,000 =1,209,074,628 mod1e9=", "1209074628 mod (1*10^9)", "209074628" },
    .{ "Sum a2*435,521 mod1e9=990,835,000+575,713,161=1,566,548,161 mod1e9=", "1566548161 mod (1*10^9)", "566548161" },
    .{ "Add first term 210,000,000 =1,151,266,870 mod1e9=", "1151266870 mod (1*10^9)", "151266870" },
    .{ "Compute 3^20= 3486784401 mod1e9 =", "3486784401 mod (1*10^9)", "486784401" },
    .{ "2ab=2*486*784,401=972*784,401. 784,401*(1000-28)=784,401,000 -21,963,228=762,437,772. Term *B=762,437,772*1,000,000 mod1e9. Need coefficient mod1000: 762,437,772 mod1000=", "762437772 mod 1000", "772" },
    .{ "Add b^2 mod1e9: 284,928,801+772,000,000=1,056,928,801 mod1e9=", "1056928801 mod (1*10^9)", "56928801" },
    .{ "3^80 = 147808829414345923316083210206383297601? Last 9? Let's compute 3^40 mod1e9 =", "3^40 mod (1*10^9)", "56928801" },
    .{ "Sum=1,383,297,601 mod1e9=", "1383297601 mod (1*10^9)", "383297601" },
    .{ "Sum=1,383,297,601 mod1e9=383,297,601. So 3^80 mod1e9=", "3^80 mod (1*10^9)", "383297601" },
    .{ "Sum=932,355,201. So 3^160 mod1e9=", "3^160 mod (1*10^9)", "932355201" },
    .{ "Sum=831,750,401. So 3^320 mod1e9=", "3^320 mod (1*10^9)", "831750401" },
    .{ "Sum=534,710,014. mod1000=14? Wait for term *1e6 mod1e9, coefficient mod1000. 534,710,014 mod1000=", "534710014 mod 1000", "14" },
    .{ "[exact: 192443283334088001 mod 262144 =", "192443283334088001 mod 262144", "116033" },
    .{ "192443 mod262144 =", "192443 mod 262144", "192443" },
    .{ "[exact: 192443283334088001 mod 3814697265625 =", "192443283334088001 mod 3814697265625", "3250375103626" },
    // S2 round 2 exact/bn-019
    .{ "(314159265)(358979323)(846264338)(327950288)(419716939)(937510?) Wait need group 9 digits from right. Since 51 digits, first group has 51 mod 9 =", "51 mod 9", "6" },
    .{ "G0 indices 1-2? Since 17 mod 3 =", "17 mod 3", "2" },
    .{ "c2 = this mod M = 318,391,393? Wait 198,318,391,393 mod 1e9 =", "198318391393 mod (1*10^9)", "318391393" },
    .{ "Last 3 groups: 944|537|150|290? Wait 9 groups of 3 = 27 digits. 27 mod 3 =", "27 mod 3", "0" },
    .{ "c2 = mod M = 314,588,294? 251,314,588,294 mod 1e9 =", "251314588294 mod (1*10^9)", "314588294" },
    // S2 round 2 guide/bn-011
    .{ "First compute 3^400 mod 5^18 =", "3^400 mod 5^18", "3250375103626" },
    .{ "Alternative use modulus 5^18 and exponent 400. Since 3^400 = (3^20)^20. 3^20=3,486,784,401. Mod 5^18. Could compute powers of 3^20. 3^20 mod 5^18 =", "3^20 mod 5^18", "3486784401" },
    .{ "Group 9 digits: 515377520 | 732011331 | 036461129 | 765621272 | 702107522 | 001? Wait need group from right in 9s. Let's do carefully. Number length 48. 48 mod 9 =", "48 mod 9", "3" },
    .{ "[exact: 133600968182994384 mod 1000000000 =", "133600968182994384 mod 1000000000", "182994384" },
    .{ "Check mod 10^9: Our result low chunk = 334,088,001. Need 3^400 mod 1e9. We can compute from x=3^100 mod1e9 =", "3^100 mod (1*10^9)", "107522001" },
    // S2 round 2 guide/bn-015
    .{ "group2 = 18 mod 1e9 =", "18 mod (1*10^9)", "18" },
    .{ "+5=141. 141 mod9 =", "141 mod 9", "6" },
    .{ "M1=1,985,229,328 digit sum:1+9+8+5+2+2+9+3+2+8=49? 1+9=10+8=18+5=23+2=25+2=27+9=36+3=39+2=41+8=49 mod9=", "49 mod 9", "4" },
    .{ "M2=19,088,743 sum=1+9+0+8+8+7+4+3=40 mod9=", "40 mod 9", "4" },
    .{ "N mod9 = H*B^3 + M1*B^2 + M2*B + L mod9 =8*1 +4*7 +4*4 +8 =8+28+16+8=60 mod9=", "60 mod 9", "6" },
    .{ "Check last 8 hex digits 89ABCDEF decimal = 2,309,737,967. Our N mod 1e9 = 85,074,415? Wait N = N2*B + L. Mod 1e9, B=4,294,967,296 mod1e9=", "4294967296 mod (1*10^9)", "294967296" },
    // S2 round 2 guide/bn-019
    .{ "k0: value 15510. digit0 = 15510 mod 1000 =", "15510 mod 1000", "510" },
    // variation
    .{ "So 3^64 % 10^18 =", "3^64 mod 10^18", "512484657849089281" },
    .{ "1000000 % 262144 = ", "1000000 mod 262144", "213568" },
    .{ "7**123 % 1000000007 =", "7**123 mod 1000000007", "937329259" },
    .{ "Then 930,336,977^2 mod 1,000,000,007 =", "930336977^2 mod 1000000007", "714810341" },
    .{ "Check 12 × 34 mod 7 =", "12 * 34 mod 7", "2" },
    .{ "2**61 mod 10**9 =", "2**61 mod 10**9", "213693952" },
    .{ "x = 3^20 mod 1e9 =", "3^20 mod (1*10^9)", "486784401" },
    .{ "Square: 3^5 mod7 =", "3^5 mod 7", "5" },
    .{ "(2^10+1)*3 mod 1000 =", "(2^10+1)*3 mod 1000", "75" },
    .{ "So 3^400 mod 5^18 = ", "3^400 mod 5^18", "3250375103626" },
};

/// Lines no modular cell may open; the model keeps writing.
const refused = [_][]const u8{
    // prose mod
    "Need compute exact mod 10^18 =",
    "the residue mod 7 =",
    "Remod 7 =",
    "mod 7 =",
    "x mod 7 =",
    "3^256 mod M =",
    "R mod 5^18 =",
    "So 3^16 mod =",
    "a mod 1e9 =",
    // a sum or difference outside parentheses, a sign, a decimal, a gap, a glued word or a list
    "2 + 3 mod 7 =",
    "12-20 mod 7 =",
    "-5 mod 3 =",
    "3.5 mod 2 =",
    "3 4 mod 7 =",
    "x3 mod 7 =",
    "f(2) mod 7 =",
    "1, 2 mod 7 =",
    "1,23 mod 7 =",
    "(3+4 mod 7 =",
    "3^ mod 7 =",
    // % in Python string formatting, or a percent
    "\"%d\" % 42 =",
    "'%5.2f' % 3 =",
    "print('%s%%' % 50) =",
    "50% * 2 =",
    "s = \"%d\" % 7 =",
    // a modulus of 0 (or a leading zero), a product or a name as modulus
    "17 mod 0 =",
    "17 % 0 =",
    "5 mod 0^3 =",
    "5 mod 0e5 =",
    "5 mod 007 =",
    "5 mod 0,000 =",
    "5 mod 3*4 =",
    "5 mod M5 =",
    // congruences and comparisons
    "16 ≡ 7 mod 9 =",
    "2^32 ≡ 2^2 mod 9 =",
    "x < 3^4 mod 7 =",
    "3^4 mod 7 ==",
    "3^4 mod 7 <=",
    "3^4 mod 7 = 4",
};

/// The cell a line closes at its `=` with inline math on, or null.
fn cellOf(a: std.mem.Allocator, line: []const u8, cfg: scan.Config) !?[]u8 {
    var s: scan.Scanner = .{ .cfg = cfg };
    defer s.deinit(a);
    const e = try s.feed(a, line, true);
    const c = e.closed orelse return null;
    if (c.form != .inline_expr or e.trigger == null or e.trigger.?.inline_space == null) return null;
    return try a.dupe(u8, c.code);
}

test "runner scan: inline mod steps from replies open cells with their literal code" {
    for (goldens) |g| {
        const got = try cellOf(t.allocator, g[0], .{ .inline_math = true }) orelse {
            std.debug.print("no cell: {s}\n", .{g[0]});
            return error.TestExpectedInline;
        };
        defer t.allocator.free(got);
        try t.expectEqualStrings(g[1], got);
    }
    for (refused) |line| if (try cellOf(t.allocator, line, .{ .inline_math = true })) |got| {
        defer t.allocator.free(got);
        std.debug.print("opened, must not: {s} -> {s}\n", .{ line, got });
        return error.TestUnexpectedInline;
    };
}

test "runner scan: inline mod is off without inline math or its knob, inside a fence, and after the value" {
    const a = t.allocator;
    try t.expectEqual(@as(?[]u8, null), try cellOf(a, "1000000 mod 262144 =", .{}));
    try t.expectEqual(@as(?[]u8, null), try cellOf(a, "1000000 mod 262144 =", .{ .inline_math = true, .inline_mod = false }));
    // the modulus alone is never the cell, knob on or off (live bn-011 had `3^400 mod 10^18 = 1000000000000000000`)
    for ([_][]const u8{ "So 3^400 mod 10^18 =", "x mod 10^18 =", "x % 10^18 =", "R mod 2^18 =" }) |line| {
        try t.expectEqual(@as(?[]u8, null), try cellOf(a, line, .{ .inline_math = true, .inline_mod = false }));
    }
    var f: scan.Scanner = .{ .cfg = .{ .inline_math = true } };
    defer f.deinit(a);
    _ = try f.feed(a, "```python\n", true);
    try t.expect((try f.feed(a, "x = 3**64 % 10**18 =", true)).trigger == null);
    var g: scan.Scanner = .{ .cfg = .{ .inline_math = true } };
    defer g.deinit(a);
    try t.expect((try g.feed(a, "1000000 mod 262144 = 213568", true)).trigger == null);
}

test "runner scan: a Python-float reply marks an e-notation modulus, not a literal one" {
    const a = t.allocator;
    for ([_]struct { []const u8, bool }{ .{ "3^64 mod 1e18 =", true }, .{ "3^64 mod 10^18 =", false } }) |c| {
        var s: scan.Scanner = .{ .cfg = .{ .inline_math = true } };
        defer s.deinit(a);
        _ = try s.feed(a, "x = float(n)\n", true);
        const e = try s.feed(a, c[0], true);
        try t.expectEqual(c[1], e.closed.?.own_context);
    }
}

fn bankPath() ![]const u8 {
    const p = std.c.getenv("TF_RUNNER_GRAPH_BANK") orelse return error.SkipZigTest;
    return std.mem.span(p);
}

test "runner proof: inline mod lines scan to cells that run on the graph bank and print CPython's value (TF_RUNNER_GRAPH_BANK)" {
    var bank = try graph.Bank.load(t.allocator, try bankPath());
    defer bank.deinit();
    var context: frontend.Context = .{ .io = t.io, .compiler = .{ .clang = "/nonexistent-inline-mod-clang" } };
    var routed: route.Routed = .{ .inner = context.frontend(), .bank = &bank };
    const f = routed.frontend();
    for (goldens) |g| {
        // end to end: the scanner's own cell for the reply line, not the golden's expected code
        const code = try cellOf(t.allocator, g[0], .{ .inline_math = true }) orelse return error.TestExpectedInline;
        defer t.allocator.free(code);
        const result = try f.translate(f.ctx, t.allocator, .{ .form = .inline_expr, .lang = "python", .code = code });
        defer if (result == .program) t.allocator.free(result.program.bytes);
        if (result != .program or result.program.kind != .graph) {
            std.debug.print("not on the graph: {s}: {any}\n", .{ code, result });
            return error.TestExpectedGraph;
        }
        var machine = try @import("runner_graph_machine.zig").Machine.boot(t.allocator, &bank, result.program.bytes);
        defer machine.deinit();
        var out: std.ArrayList(u8) = .empty;
        defer out.deinit(t.allocator);
        const end = try machine.run(4_000_000, &out, null);
        try t.expect(end.end == .halted);
        try t.expectEqualStrings(g[2], std.mem.trimEnd(u8, out.items, "\n"));
    }
}

/// T4W6 mul30-14 rc reply 5 (census rank6 cells 57, 59–67, 70; rank9 54, 56): each follows the fenced product below.
const mod_prior = [_]Golden{
    .{ "+8=128; +6=134; +6=140. 140 mod9 =", "140 mod 9", "5" },
    .{ "+9=145; +2=147. 147 mod9 =", "147 mod 9", "3" },
    .{ "147 mod9 = 3 (144). B mod9=3. Product mod9 = 5*3=15 mod9=", "15 mod 9", "6" },
    .{ "455992: sum digits 4+5+5+9+9+2= 34; 34 mod9=", "34 mod 9", "7" },
    .{ "065324469: sum=0+6+5+3+2+4+4+6+9=39; 39 mod9=", "39 mod 9", "3" },
    .{ "389066179: sum=3+8+9+0+6+6+1+7+9=49; 49 mod9=", "49 mod 9", "4" },
    .{ "825281039: sum=8+2+5+2+8+1+0+3+9=38; 38 mod9=", "38 mod 9", "2" },
    .{ "054396652: sum=0+5+4+3+9+6+6+5+2=40; 40 mod9=", "40 mod 9", "4" },
    .{ "721075268: sum=7+2+1+0+7+5+2+6+8=38; 38 mod9=", "38 mod 9", "2" },
    .{ "148093472: sum=1+4+8+0+9+3+4+7+2=38; 38 mod9=", "38 mod 9", "2" },
    .{ "Sum mods = 7+3+4+2+4+2+2 = 24; 24 mod9=", "24 mod 9", "6" },
    .{ "Check mod 11 maybe. Since 10^9 mod11 =", "10^9 mod 11", "10" },
    .{ "Alternating sum = d0 - d1 + d2 - d3 + d4 - d5 + d6 ≡ 10 -1 +2 -0 +7 -1 +9 = 26. 26 mod11 =", "26 mod 11", "4" },
};

const prior_fence = "print(831680992073052982499463120866 * 548277608446792630955630629392)\n";

test "runner graph route: only a literal-only inline value is closed (R5)" {
    for ([_][]const u8{ "140 mod 9", "10^9 mod 11", "(3+4)*5", "3^64 mod (1*10^18)" }) |c| try t.expect(route.closedLiteral(c));
    for ([_][]const u8{ "R mod 9", "A*3", "5 % 3", "1.5*2", "10/2", "pow(10, 9, 11)", "1e9 mod 7", "", "mod" }) |c|
        try t.expect(!route.closedLiteral(c));
}

fn same(x: api.Translation, y: api.Translation) bool {
    if (x == .abstain) return y == .abstain and x.abstain == y.abstain;
    return y == .program and x.program.kind == y.program.kind and std.mem.eql(u8, x.program.bytes, y.program.bytes);
}

test "runner proof: literal mod lines after an unrelated fenced cell run alone on the graph bank (R5, TF_RUNNER_GRAPH_BANK)" {
    const a = t.allocator;
    var bank = try graph.Bank.load(a, try bankPath());
    defer bank.deinit();
    var context: frontend.Context = .{ .io = t.io, .compiler = .{ .clang = "/nonexistent-inline-mod-clang" } };
    var routed: route.Routed = .{ .inner = context.frontend(), .bank = &bank };
    const f = routed.frontend();
    var quick: route.Quick = .{ .routed = &routed }; // TF_DSV41_RUNNER_SYNC_GRAPH's engine-thread route
    const q = quick.frontend();
    // the earlier cell as written, with its operand changed, and with more cells printing and binding after it
    const priors = [_][]const []const u8{
        &.{prior_fence},
        &.{"print(831680992073052982499463120867 * 548277608446792630955630629392)\n"},
        &.{ prior_fence, "print(7)\n", "x = 5\nprint(x * 3)\n" },
    };
    for (mod_prior) |g| {
        const code = try cellOf(a, g[0], .{ .inline_math = true }) orelse return error.TestExpectedInline;
        defer a.free(code);
        try t.expectEqualStrings(g[1], code);
        const alone = try f.translate(f.ctx, a, .{ .form = .inline_expr, .lang = "python", .code = code });
        defer if (alone == .program) a.free(alone.program.bytes);
        if (alone != .program or alone.program.kind != .graph) return error.TestExpectedGraph;
        for (priors) |p| {
            const with = try f.translate(f.ctx, a, .{ .form = .inline_expr, .lang = "python", .code = code, .prior = p });
            defer if (with == .program) a.free(with.program.bytes);
            try t.expect(same(alone, with));
            const synced = try q.translate(q.ctx, a, .{ .form = .inline_expr, .lang = "python", .code = code, .prior = p });
            defer if (synced == .program) a.free(synced.program.bytes);
            try t.expect(same(alone, synced));
        }
        var machine = try @import("runner_graph_machine.zig").Machine.boot(a, &bank, alone.program.bytes);
        defer machine.deinit();
        var out: std.ArrayList(u8) = .empty;
        defer out.deinit(a);
        const end = try machine.run(4_000_000, &out, null);
        try t.expect(end.end == .halted);
        try t.expectEqualStrings(g[2], std.mem.trimEnd(u8, out.items, "\n"));
    }
    // edge values keep their meaning with no earlier cell (zero modulus, a negative left side, an oversized power)
    for ([_][]const u8{ "5 mod 0", "(0-5) mod 3", "2^100000 mod 7", "3^400 mod 10^18" }) |code| {
        const x = try f.translate(f.ctx, a, .{ .form = .inline_expr, .lang = "python", .code = code });
        defer if (x == .program) a.free(x.program.bytes);
        const y = try f.translate(f.ctx, a, .{ .form = .inline_expr, .lang = "python", .code = code, .prior = &.{prior_fence} });
        defer if (y == .program) a.free(y.program.bytes);
        try t.expect(same(x, y));
    }
    // a value that reads or may read an earlier cell keeps the earlier-cell refusal: a name, a call it may rebind
    for ([_]api.Cell{
        .{ .form = .inline_expr, .lang = "python", .code = "A mod 9", .prior = &.{"A = 140\nprint(A)\n"} },
        .{ .form = .inline_expr, .lang = "python", .code = "pow(10, 9, 11)", .prior = &.{"def pow(a, b, c):\n    return 0\n"} },
    }) |c| {
        const r = try f.translate(f.ctx, a, c);
        defer if (r == .program) a.free(r.program.bytes);
        if (r == .program and r.program.kind == .graph) {
            std.debug.print("ran without its earlier cell: {s}\n", .{c.code});
            return error.TestUnexpectedGraph;
        }
    }
}
