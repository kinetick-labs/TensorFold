//! Form trace: binding recovery (T5-BINDING-RECOVERY.md R1–R4) on live lines; every wrong-intent shape still abstains.
const std = @import("std");
const trace = @import("runner_trace.zig");
const bind = @import("runner_trace_bind.zig");
const stated_test = @import("runner_trace_stated_test.zig");

const gpa = std.testing.allocator;

const all: bind.Rules = .{};
const none: bind.Rules = .{ .r1 = false, .r2 = false, .r3 = false, .r4 = false };
/// Each rule alone, all of them, and none: the configurations every hard guard holds under.
const configs = [_]bind.Rules{ .{ .r2 = false, .r3 = false, .r4 = false }, .{ .r1 = false, .r3 = false, .r4 = false }, .{ .r1 = false, .r2 = false, .r4 = false }, .{ .r1 = false, .r2 = false, .r3 = false }, all, none };

/// The last line's statement (source and frontend run) after `lines`, under `rules`; null if it abstains.
fn last(rules: bind.Rules, lines: []const []const u8) !?[2][]u8 {
    var t = trace.Tracer.init(gpa);
    defer t.deinit();
    t.rules = rules;
    var got: ?trace.Found = null;
    for (lines) |l| got = try t.line(l);
    const f = got orelse return null;
    return .{ try gpa.dupe(u8, f.source), try gpa.dupe(u8, f.run) };
}

/// Each line's statement and every name's binding after it, for a failure.
fn dump(rules: bind.Rules, lines: []const []const u8) !void {
    var t = trace.Tracer.init(gpa);
    defer t.deinit();
    t.rules = rules;
    for (lines) |l| {
        const f = try t.line(l);
        std.debug.print("  line: {s}\n    -> {s}\n   ", .{ l, if (f) |x| x.source else "-" });
        for (t.names.items, t.binds.items) |m, b| std.debug.print(" {s}:{s}", .{ m.name, @tagName(b.state) });
        std.debug.print("\n", .{});
    }
}

fn expectSource(rules: bind.Rules, lines: []const []const u8, want: []const u8) !void {
    const got = try last(rules, lines) orelse {
        std.debug.print("abstained: {s}\n", .{lines[lines.len - 1]});
        try dump(rules, lines);
        return error.TestExpectedStatement;
    };
    defer for (got) |g| gpa.free(g);
    try std.testing.expectEqualStrings(want, got[0]);
}

fn expectNone(rules: bind.Rules, lines: []const []const u8) !void {
    if (try last(rules, lines)) |got| {
        defer for (got) |g| gpa.free(g);
        std.debug.print("stated {s} from: {s} (rules {any})\n", .{ got[0], lines[lines.len - 1], rules });
        try dump(rules, lines);
        return error.TestUnexpectedStatement;
    }
}

const Golden = struct { rule: []const u8, lines: []const []const u8, want: []const u8 };

/// Live lines (condensed) whose correct claim the stale fix c17578b72 lost and T5's forecast credits to the rule.
pub const goldens = [_]Golden{
    // R1, big-number-017: approximate notes first, then plain exact declarations before the request
    .{ .rule = "R1", .lines = &.{ "AB=38,129,436,797,760 (~3.8129e13) CD=41,165,158,003,200 (~4.1165e13) Product ~1.5696e27.", "AB=38,129,436,797,760. CD=41,165,158,003,200. Compute P=AB*CD." }, .want = "(38129436797760)*(41165158003200)" },
    // R1, big-number-010: N and M first in a questioned paragraph
    .{ .rule = "R1", .lines = &.{ "with N=1234567890123456789? Wait a/10 = 1,234,567,890,123,456,789 (yes 19 digits), M=90,000,000,009.", "N = 1,234,567,890,123,456,789", "M = 90,000,000,009.", "Compute N mod M." }, .want = "(1234567890123456789)%(90000000009)" },
    // R1, big-number-002: an unresolved quotient formula first (`/` is not read), then the decimal value
    .{ .rule = "R1", .lines = &.{ "Need quotient Q = (10^30 -1 -84)/97 = (10^30 -85)/97.", "Q = 10,309,278,350,515,463,917,525,773,195. Compute Q*100 -3Q (since *97).", "Need compute Q*3." }, .want = "(10309278350515463917525773195)*3" },
    // R1, 0474: a doubted chain first, then `Yes L=…`
    .{ .rule = "R1", .lines = &.{ "Let L = A + 2q0 = 70,272,836,560,000,000,000 + 16,400,000,000 = 70,272,836,576,400,000,000? Wait A=70,272,836,560,000,000,000.", "2q0=16,400,000,000. Add =70,272,836,576,400,000,000. Yes L=70,272,836,576,400,000,000.", "Compute L*1442." }, .want = "(70272836576400000000)*1442" },
    // R1, 0626: a doubted hex chain first
    .{ .rule = "R1", .lines = &.{ "A=0xFEDCBA98 = 4275878552? Let's compute: 0xFE000000=4261412864.", "A=4,275,878,552. B=1,985,229,328.", "Now multiply by A=4,275,878,552. Need A * 2^96." }, .want = "(4275878552)*2**96" },
    // R1, big-number-019: Q doubted first; P defined once B and Q are exact
    .{ .rule = "R1", .lines = &.{ "Q=1,155,727,349,790,921,717,910,215,407,028? Proposed.", "Need parse Q = 1155727349790921717910215407028.", "B = 271,828,182,845,904,523,536.", "Need product P=B*Q." }, .want = "(271828182845904523536)*(1155727349790921717910215407028)" },
    // R1, 0079: N first an unresolved formula, then a doubted pipe layout, then `Yes N=…`; P states once N is exact
    .{ .rule = "R1", .lines = &.{ "Let N=M*B. Then P = N*10^10 + N.", "So N = 12 | 193263112 | 482853211 | 126352690? Wait groups high.", "Yes N=12,193,263,112,482,853,211,126,352,690.", "But easier P = N*10^10 + N." }, .want = "(12193263112482853211126352690)*10**10+(12193263112482853211126352690)" },
    // R1, big-number-001: `N=xy = …` (a symbolic middle term), then the plain value; P's definition states
    .{ .rule = "R1", .lines = &.{ "Let N=xy = 1,219,326,311,126,352,690 (19 digits? groups 1 +6*3=19 digits).", "N = 1,219,326,311,126,352,690. N*10^20 = 121,932,631,112,635,269,000,000,000,000,000,000,000?", "P = N*(10^20 + 2*10^10 +1)" }, .want = "(1219326311126352690)*(10**20+2*10**10+1)" },
    // R1 + R2, big-number-007: `A=a^2=…`, then `Let A = …`, then a `=> groups` restatement
    .{ .rule = "R1", .lines = &.{ "N^2 = A*(10^18 + 2,000,000,001), where A=a^2=975,461,057,789,971,041.", "Let A = 975,461,057,789,971,041.", "A=975,461,057,789,971,041 => groups [975,461,057,789,971,041] (6 groups).", "Need compute A*2,000,000,000 + A." }, .want = "(975461057789971041)*2000000000+(975461057789971041)" },
    // R2, R4 product-b: `-> groups [...]` after the unchanged value
    .{ .rule = "R2", .lines = &.{ "Let A = 98,765,432,109.", "A = 98,765,432,109 -> groups [98, 765, 432, 109]? From right: 98 | 765 | 432 | 109.", "Need compute A*345,678,901." }, .want = "(98765432109)*345678901" },
    // R2, R4 division: spaced groups with a digit count that counts them
    .{ .rule = "R2", .lines = &.{ "Let D = 271,828,183.", "groups: 314 159 265 358 979 323 846 264 (8 groups of 3) = 24 digits. D=271 828 183 (9 digits).", "Need compute D*1,155,727,349,135,756." }, .want = "(271828183)*1155727349135756" },
    // R2, 0112: `groups:` then the pipe layout
    .{ .rule = "R2", .lines = &.{ "Our result R=192,443,283,334,088,001.", "R groups base 1000: 192 | 443 | 283 | 334 | 088 | 001? Wait R=192,443,283,334,088,001 groups: 192,443,283,334,088,001 = 192|443|283|334|088|001 (6 groups).", "Need compute R mod 5^18." }, .want = "(192443283334088001)%5**18" },
    // R2, 0558: `=> groups high [...]`
    .{ .rule = "R2", .lines = &.{ "Let A=123456789012, B=987654321098.", "A=123,456,789,012 => groups high [123,456,789,012], low [12? Let's list low to high.", "Need compute A*345,678,902." }, .want = "(123456789012)*345678902" },
    // R3, 0633: `B = 2^64 = …` restates B; P's request reads H and B
    .{ .rule = "R3", .lines = &.{ "B = 18,446,744,073,709,551,616", "B = 2^64 = 18,446,744,073,709,551,616.", "H = 18,364,758,544,493,064,720.", "Need product P = H*B. Then add L." }, .want = "(18364758544493064720)*(18446744073709551616)" },
    // R3, 0655: hex/decimal chains (one ending a parenthetical), X0 = A = …, X1's `chunks […] = value` restatement
    .{ .rule = "R3", .lines = &.{ "A=0xFEDCBA98", "B=0x76543210", "Actually 0xFFFFFFFF - 0xFEDCBA98 = 0x01234567 = 19088743, so A=4294967295-19088743=4275878552). Yes.", "B=0x76543210 = 1985229328? Known 0x76543210 = 1985229328.", "Let X0 = A = 4,275,878,552. X1 = X0 * 2^32 + B.", "Now compute X1 = A * 2^32 + B.", "So X1 = chunks [493,064,720, 364,758,544, 18] = 18,364,758,544,493,064,720? Wait concatenate.", "Need compute X1*1000003." }, .want = "(((0xFEDCBA98))*2**32+(0x76543210))*1000003" },
    // R3, the same line's definitions with no stop between them (`A=… B=…`)
    .{ .rule = "R3", .lines = &.{ "Let's denote A=12345678901234567890 B=98765432109876543210", "Need compute B - A." }, .want = "(98765432109876543210)-(12345678901234567890)" },
    // R4 + R1, 0549: `B mod A = …?` assigns nothing to A; R doubted first, then declared
    .{ .rule = "R4", .lines = &.{ "A=12345678901234567890", "B=98765432109876543210", "So B mod A = 900,000,000,090? Wait B = 8A + R, R=900,000,000,090. Since R < A (12e18 vs 0.9e12), yes.", "Thus gcd(A,B)=gcd(A,R) where R=900,000,000,090.", "Now A=12,345,678,901,234,567,890. Need A mod R. R=900,000,000,090." }, .want = "(12345678901234567890)%(900000000090)" },
    // R1, big-number-011 s1: `base B=10^9:` opens a list, a sighting; `Let B=1,000,000,000.` binds B
    .{ .rule = "R1", .lines = &.{ "Let a = 621,272,702,107,522,001. Split into high H and low L base B=10^9:", "Let B=1,000,000,000.", "L=107,522,001. H=621,272,702.", "Compute H*L mod B." }, .want = "(621272702)*(107522001)%(1000000000)" },
    // R2, big-number-002 s2: `Q = (10^30 -85)/97` restates Q as an exact quotient
    .{ .rule = "R2", .lines = &.{ "Need quotient Q = (10^30 -1 -84)/97 = (10^30 -85)/97.", "Q = 10,309,278,350,515,463,917,525,773,195.", "Alternatively verify formula: (10^30 -1 -84)/97 = (10^30 -85)/97. Multiply by 97: 97Q =10^30 -85. Then +84 =10^30 -1 yes. Need Q exact if formula. Since Q computed from division groups likely correct. We can verify Q = (10^30 -85)/97. Is Q equal 10309278350515463917525773195? Multiply by 97:", "3Q groups: compute Q*3:" }, .want = "(10309278350515463917525773195)*3" },
};

test "runner trace bind: live recoveries (R1–R4) state the model's value, and none of them without the rules" {
    for (goldens) |g| {
        expectSource(all, g.lines, g.want) catch |e| {
            std.debug.print("golden {s}: {s}\n", .{ g.rule, g.want });
            return e;
        };
        try expectNone(none, g.lines);
    }
}

/// T5's frozen wrong-intent shapes (each in both STATED modes), condensed: each abstains under every configuration.
fn expectWrongIntentAbstains(cases: []const []const []const u8) !void {
    for (configs) |c| for (cases) |w| try expectNone(c, w);
}

test "runner trace bind: big-number-015 s2: the loop rule `N = N*256 + b` before `Start N=0` blocks N for good" {
    try expectWrongIntentAbstains(&.{
        &.{ "Number N = (((... bytes ...)*256 + next)).", "N = w1 * 2^96 + w2 * 2^64 + w3 *2^32 + w4.", "Process byte b: N = N*256 + b.", "Start N=0.", "Need N*256+118." },
        &.{ "Process byte b: N = N*256 + b.", "Start N=0.", "N = 0.", "Need compute N*256+118." },
    });
}

test "runner trace bind: trace3 s3: `Step3 N=` moves N, a label hides no reassignment, the old value never returns" {
    try expectWrongIntentAbstains(&.{
        &.{ "Let N=280,223,976,814,164. N*65536 = N*2^16.", "N=280,223,976,814,164 (~2.802e14).", "Step3 N=18,364,758,544,493,064,720. Compute N*65536." },
        &.{ "Let N=280,223,976,814,164.", "Step3 N=18,364,758,544,493,064,720.", "N = 280,223,976,814,164.", "Compute N*65536." },
    });
}

test "runner trace bind: 0038: `Let B=1e9` changes the base; 1eN is never read, so B is blocked, not rebound" {
    try expectWrongIntentAbstains(&.{
        &.{ "Long division base B=1000.", "Need compute long division base B=1e9.", "Q = 1155*B^3 + 727349790*B^2 + 921717910*B + 215407028." },
        &.{ "Long division base B=1000.", "Let B=1e9. D = d0?", "B = 1000.", "Q = 1155*B^3 + 727349790*B^2 + 921717910*B + 215407028." },
    });
}

test "runner trace bind: big-number-011 s3: M is only `M=1e18`; `R mod M = …` makes no modulus of the residue" {
    try expectWrongIntentAbstains(&.{
        &.{ "mod M (M=1e18): M-2000 = 999,999,999,999,998,000.", "Partial R= N1+N3+N5 = 0|253|566|317|402|000.", "Need compute -R mod M, with R=1,807,556,716,665,911,999. Mod M, R mod M =807,556,716,665,911,999 (subtract 1,000,000,000,000,000,000).", "Need compute -R mod M." },
        &.{ "mod M (M=1e18): M-2000 = 999,999,999,999,998,000.", "R=1,807,556,716,665,911,999. R mod M =807,556,716,665,911,999.", "Need compute R mod M." },
    });
}

test "runner trace bind: 0555: `Need N mod p. N=…` moves N after the request read it; neither value is stated" {
    try expectWrongIntentAbstains(&.{
        &.{ "Let's compute using base 1e9 maybe. N=4,852,937,748,780,900.", "Now mod p. Need N mod 1,000,000,007. N=81,333,145,593,191,556. Quotient ~81,333,145." },
        &.{ "A = 12345678901.", "B = A*3.", "Need compute B*1000003. A = 7." },
    });
}

test "runner trace bind: adversarial controls never state an old, doubted, unproved or future value" {
    const cases = [_][]const []const u8{
        // a pipe is never read (it may be OR); plain, it blocks for good; doubted, only a plain equal value clears it
        &.{ "Let D=12.", "D=1|2.", "D = 12.", "Need compute D*1000003." },
        &.{ "Let D=12.", "D=1|3?", "D = 13.", "Need compute D*1000003." },
        &.{ "Let D=12.", "D=1|2?", "Need compute D*1000003." },
        &.{ "Let D = 12345.", "D=12|345.", "Need compute D*1000003." },
        // unequal chains, hex or decimal
        &.{ "N=1=2.", "Need compute N*1000003." },
        &.{ "Let N = 5.", "N = 5 = 6.", "Need compute N*1000003." },
        &.{ "B = 0x10 = 17.", "Need compute B*1000003." },
        // approximation only, an unknown alias, a doubted chain: never a first binding
        &.{ "A = 12345678901 (≈1.2e10).", "Need compute A*1000003." },
        &.{ "AB=38,129,436,797,760 (~3.8129e13).", "Compute AB*41,165,158,003,200." },
        &.{ "X = Y.", "Need compute X*1000003." },
        &.{ "B=0x76543210 = 1985229328?", "Need compute B*1000003." },
        &.{ "Q = (10^30 - 85)/97.", "Q = 10,309,278,350,515,463,917,525,773,195?", "Need compute Q*3." },
        // a quotient proves only a held value it divides out to exactly; a list's opening value binds nothing
        &.{ "Q = 10,309,278,350,515,463,917,525,773,195.", "We can verify Q = (10^30 -86)/97.", "Need compute Q*3." },
        &.{ "Q = 10,309,278,350,515,463,917,525,773,196.", "We can verify Q = (10^30 -85)/97.", "Need compute Q*3." },
        &.{ "Split into high H and low L base B=10^9:", "Compute H*L mod B." },
        // self-reference and updates in place block for good, even back at the first literal
        &.{ "N = 5.", "N = N + 1.", "N = 5.", "Need compute N*1000003." },
        &.{ "Let N = 12345.", "N += 3.", "Need compute N*1000003." },
        &.{ "Let N = 12345.", "N++ then.", "Need compute N*1000003." },
        &.{ "Let N = 12345.", "N *= 2.", "N = 12345.", "Need compute N*1000003." },
        // a changed modulus or base, numeric or 1eN
        &.{ "Let M = 1,000,000,007.", "M = 998,244,353.", "Need compute M*1000003." },
        &.{ "Let M = 1,000,000,007.", "M = 1e9+7.", "M = 1,000,000,007.", "Need compute M*1000003." },
        // a definition after the request on the same line never supplies it
        &.{"Need compute N*1000003. N = 12345."},
        // a dependency moved: what read it is stale, restoring it restores nothing (a name in any equal piece counts)
        &.{ "A=12345678901.", "B=12345678901=A.", "C=B*3.", "A=7.", "Need compute C*1000003." },
        &.{ "A=12345678901.", "B=A=12345678901.", "C=B*3.", "A=7.", "Need compute C*1000003." },
        &.{ "A=12345678901.", "B=12345678901=A.", "A=7.", "Need compute B*1000003." },
        // an equal restatement of a held name reads its names too (T5-BINDING-ADB-REVIEW.md), quarantined or not
        &.{ "A=12345678901.", "B=12345678901.", "B=12345678901=A.", "C=B*3.", "A=7.", "Need compute C*1000003." },
        &.{ "A=12345678901.", "B=12345678901.", "B=A=12345678901.", "C=B*3.", "A=7.", "Need compute C*1000003." },
        &.{ "A=12345678901.", "B=12345678901.", "B=12345678901=A.", "A=7.", "Need compute B*1000003." },
        &.{ "A=12345678901.", "B=12345678901.", "B=Y=12345678901.", "B=12345678901=A.", "A=7.", "Need compute B*1000003." },
        &.{ "A=12345678901.", "B=A.", "B=Y=12345678901.", "B=12345678901.", "A=7.", "Need compute B*1000003." },
        // an unresolved piece or a dependency left unresolved never proves a restatement (T5 review fixtures)
        &.{ "B=12345678901.", "B=Y=12345678901.", "Need compute B*1000003." },
        &.{ "A=12345678901.", "B=12345678901.", "A=Y+1.", "B=A.", "Need compute B*1000003." },
        &.{ "A = 12345678901.", "B = A*3.", "A = 5.", "Need compute B*1000003." },
        &.{ "A = 12345678901.", "B = A*3.", "A = N*q + r.", "A = 12345678901.", "Need compute B*1000003." },
        &.{ "A = 12345678901.", "B = A*3.", "C = B + 1.", "A = 7.", "Need compute C*1000003." },
        // a quarantine clears only at a plain restatement of the same value
        &.{ "Let N = 12345678901.", "N = D*q + r.", "N = 12345678901?", "Need compute N*1000003." },
        &.{ "Let N = 12345678901.", "N = D*q + r.", "N = 12345678902.", "Need compute N*1000003." },
        // layout metadata and spaced groups prove a restatement, never a binding; groups need a true digit count
        &.{ "A = 98,765,432,109 -> groups [98, 765, 432, 109].", "Need compute A*345,678,901." },
        &.{ "Let D = 271,828,183.", "D=271 828 183.", "Need compute D*1000003." },
        &.{ "Let D = 271,828,183.", "D=271 828 183 (8 digits).", "Need compute D*1000003." },
        &.{ "Let D = 271,828,183.", "D=271 828 184 (9 digits).", "Need compute D*1000003." },
        // a right side that ends the line binds nothing yet
        &.{ "Let N =", "Need compute N*1000003." },
        // R4 binds nothing: the residue is no modulus
        &.{ "x mod M = 5.", "Need compute 7*M." },
    };
    for (cases) |c| try expectNone(all, c);
}

test "runner trace bind: what the guards keep: later declarations, restorations, a genuine definition beside R4" {
    try expectSource(all, &.{ "Need compute N*1000003. N = 12345.", "Need compute N*1000003." }, "(12345)*1000003");
    try expectSource(all, &.{ "A=12345678901.", "B=12345678901.", "B=12345678901=A.", "C=B*3.", "Need compute C*1000003." }, "((12345678901)*3)*1000003");
    try expectSource(all, &.{ "A=12345678901.", "B=12345678901.", "B=A=12345678901.", "C=B*3.", "Need compute C*1000003." }, "((12345678901)*3)*1000003");
    try expectSource(all, &.{ "A = 12345678901.", "B = A*3.", "A = N*q + r.", "A = 12345678901.", "Need compute A*1000003." }, "(12345678901)*1000003");
    try expectSource(all, &.{ "A = 12345678901.", "B = A*3.", "A = 5.", "B = 7.", "Need compute B*1000003." }, "(7)*1000003");
    try expectSource(all, &.{ "Let N = 12345678901.", "N = D*q + r.", "N = 12345678901.", "Need compute N*1000003." }, "(12345678901)*1000003");
    try expectSource(all, &.{ "Let A = 7. So B mod A = 3, M = 12345678901.", "Need compute M*1000003." }, "(12345678901)*1000003");
    try expectSource(all, &.{ "Let A = 12345678901 (≈1.2e10).", "A = 12345678901.", "Need compute A*1000003." }, "(12345678901)*1000003");
    try expectSource(all, &.{ "If A = 6 then fine.", "Let A = 12345.", "Need compute A*1000003." }, "(12345)*1000003");
    try expectSource(all, &.{ "Let N =", "N = 12345.", "Need compute N*1000003." }, "(12345)*1000003");
    try expectSource(all, &.{ "Let D=12.", "D=1|2?", "D = 12.", "Need compute D*1000003." }, "(12)*1000003");
    // the same value after the request moves nothing; after a moved request, the line's next statement is read
    try expectSource(all, &.{ "Let N = 12345.", "Need compute N*1000003. N = 12,345." }, "(12345)*1000003");
    // the permuted chain still binds when nothing it read moves
    try expectSource(all, &.{ "A=12345678901.", "B=12345678901=A.", "C=B*3.", "Need compute C*1000003." }, "((12345678901)*3)*1000003");
    try expectSource(all, &.{ "A=12345678901.", "B=A=12345678901.", "C=B*3.", "Need compute C*1000003." }, "(((12345678901))*3)*1000003");
    try expectSource(all, &.{ "A=12345678901234567890", "B=98765432109876543210", "Compute B mod A. B = 8*A? 8A = 98,765,431,209,876,543,120? Let's calculate A*8:" }, "(12345678901234567890)*8");
}

test "runner trace bind: a changed anchor changes the statement, never the old value" {
    // golden, its declaration line, the anchor, its change, the change's digits
    for ([_]struct { usize, usize, []const u8, []const u8, []const u8 }{
        .{ 0, 1, "38,129,436,797,760", "38,129,436,797,761", "38129436797761" },
        .{ 9, 1, "98,765,432,109", "98,765,432,119", "98765432119" },
        .{ 13, 2, "18,364,758,544,493,064,720", "18,364,758,544,493,064,721", "18364758544493064721" },
        .{ 16, 3, "900,000,000,090", "900,000,000,091", "900000000091" },
    }) |m| {
        const g = goldens[m[0]];
        const n = g.lines.len;
        var lines: [8][]const u8 = undefined;
        var owned: [8][]u8 = undefined;
        for (g.lines, 0..) |l, k| {
            owned[k] = try std.mem.replaceOwned(u8, gpa, l, m[2], m[3]);
            lines[k] = owned[k];
        }
        defer for (owned[0..n]) |o| gpa.free(o);
        const old_digits = try std.mem.replaceOwned(u8, gpa, m[2], ",", "");
        defer gpa.free(old_digits);
        const want = try std.mem.replaceOwned(u8, gpa, g.want, old_digits, m[4]);
        defer gpa.free(want);
        try expectSource(all, lines[0..n], want);
        // the old anchor bound, then the changed one declared: a numeric conflict blocks the name
        var conflict: [10][]const u8 = undefined;
        @memcpy(conflict[0..n], g.lines);
        conflict[n] = lines[m[1]];
        conflict[n + 1] = g.lines[n - 1];
        try expectNone(all, conflict[0 .. n + @as(usize, if (m[1] == n - 1) 1 else 2)]);
    }
}

test "runner proof: trace bind: each recovered statement runs on the bank and prints CPython's value (TF_RUNNER_GRAPH_BANK)" {
    const b = try stated_test.openBank() orelse return error.SkipZigTest;
    defer stated_test.closeBank(b);
    for (goldens) |g| {
        const got = (try last(all, g.lines)).?;
        defer for (got) |x| gpa.free(x);
        const want = try stated_test.python(got[1]);
        defer gpa.free(want);
        const bytes = try b.translate(gpa, got[1]);
        defer gpa.free(bytes);
        const out = try b.run(bytes);
        defer gpa.free(out);
        std.testing.expectEqualStrings(want, out) catch |e| {
            std.debug.print("bank value: {s}\n", .{got[1]});
            return e;
        };
    }
    // the bank computes the recovered request: one bound operand changed in the packet prints CPython's new value
    const got = (try last(all, goldens[16].lines)).?;
    defer for (got) |x| gpa.free(x);
    const bytes = try b.translate(gpa, got[1]);
    defer gpa.free(bytes);
    if (std.mem.count(u8, bytes, "\"900000000090\"") != 1) {
        std.debug.print("operand not one packet value: {s}\n", .{bytes});
        return error.TestExpectedInput;
    }
    const mutated = try std.mem.replaceOwned(u8, gpa, bytes, "\"900000000090\"", "\"900000000091\"");
    defer gpa.free(mutated);
    const want = try stated_test.python("(12345678901234567890)%(900000000091)");
    defer gpa.free(want);
    const out = try b.run(mutated);
    defer gpa.free(out);
    try std.testing.expectEqualStrings(want, out);
}
