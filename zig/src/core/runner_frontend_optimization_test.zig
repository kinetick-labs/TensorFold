const std = @import("std");
const compiler = @import("runner_frontend_program_compile.zig");
const blocks = @import("runner_frontend_python_blocks.zig");
const legacy = @import("runner_frontend_program.zig");
const t = std.testing;
const Env = std.process.Environ.Map;

const numeric_source =
    \\int main(void) {
    \\    int sum = 0;
    \\    for (int i = 1; i <= 4096; ++i) sum += i;
    \\    printf("%d\n", sum);
    \\    return 0;
    \\}
;
const numeric_output = "8390656\n";
const literal_source = "int main(void) { printf(\"retained literal output\\n\"); return 0; }";
const literal_output = "retained literal output\n";

// Refresh records do not increment Machine.steps; discard each record rather than accumulating trace tokens.
const streaming_reference =
    \\import sys
    \\from pathlib import Path
    \\sys.path.insert(0, str(Path(sys.argv[1]) / 'experimental/riscv'))
    \\from chipformer.rv import elf, ref
    \\machine = ref.Machine.from_program(elf.program(sys.argv[2]))
    \\records = 0
    \\subtractions = 0
    \\while not machine.halted and machine.steps < 4000000 and records < 4000000:
    \\    word = machine.rom[machine.anchor + machine.n]
    \\    before = machine.steps
    \\    machine.record()
    \\    if machine.steps != before and word & 0xfe00707f == 0x40000033: subtractions += 1
    \\    records += 1
    \\    assert len(machine.out) <= 4096, 'stdout limit'
    \\assert machine.halted and machine.fault is None and machine.exit_code == 0, (
    \\    machine.halted, machine.fault, machine.exit_code, machine.steps, records)
    \\assert 0 < machine.steps <= 4000000, machine.steps
    \\sys.stdout.buffer.write(f'{machine.steps} {subtractions}'.encode('ascii') + b'\n' + bytes(machine.out))
;

const Observation = struct {
    steps: u64,
    subtractions: u64,
    stdout: []u8,

    fn deinit(o: Observation) void {
        t.allocator.free(o.stdout);
    }
};

fn settings(env: *const Env, trusted: bool) !compiler.Config {
    const clang = env.get("TF_RUNNER_CLANG") orelse return error.SkipZigTest;
    _ = env.get("TF_RUNNER_QWENPUTER") orelse return error.SkipZigTest;
    return .{
        .clang = clang,
        .linker = env.get("TF_RUNNER_LLD"),
        .opaque_inputs = trusted,
        .step_budget = 0,
    };
}

fn build(source: []const u8, config: compiler.Config) ![]u8 {
    const bytes = (try compiler.compile(t.allocator, t.io, source, config)) orelse
        return error.UnexpectedCompilationAbstention;
    errdefer t.allocator.free(bytes);
    try t.expect(compiler.validElf(bytes));
    return bytes;
}

fn observe(env: *const Env, bytes: []const u8) !Observation {
    const root = env.get("TF_RUNNER_QWENPUTER") orelse return error.SkipZigTest;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const file = try tmp.dir.createFile(t.io, "optimization.elf", .{});
    defer file.close(t.io);
    try file.writeStreamingAll(t.io, bytes);
    const path = try tmp.dir.realPathFileAlloc(t.io, "optimization.elf", t.allocator);
    defer t.allocator.free(path);
    const result = try std.process.run(t.allocator, t.io, .{
        .argv = &.{ env.get("TF_RUNNER_REFERENCE_PYTHON") orelse "/usr/bin/python3", "-c", streaming_reference, root, path },
        .stdout_limit = .limited(8192),
        .stderr_limit = .limited(16384),
        .timeout = .{ .duration = .{ .raw = .fromSeconds(30), .clock = .awake } },
    });
    defer t.allocator.free(result.stdout);
    defer t.allocator.free(result.stderr);
    if (!result.term.success()) {
        std.debug.print("optimization RV32 reference failed: {s}\n", .{result.stderr});
        return error.ReferenceFailed;
    }
    const newline = std.mem.indexOfScalar(u8, result.stdout, '\n') orelse return error.InvalidReferenceReport;
    var fields = std.mem.splitScalar(u8, result.stdout[0..newline], ' ');
    const steps = std.fmt.parseInt(u64, fields.next() orelse return error.InvalidReferenceReport, 10) catch return error.InvalidReferenceReport;
    const subtractions = std.fmt.parseInt(u64, fields.next() orelse return error.InvalidReferenceReport, 10) catch return error.InvalidReferenceReport;
    if (fields.next() != null) return error.InvalidReferenceReport;
    if (steps == 0 or steps > 4000000) return error.InvalidReferenceReport;
    return .{ .steps = steps, .subtractions = subtractions, .stdout = try t.allocator.dupe(u8, result.stdout[newline + 1 ..]) };
}

// ceil(O0 / 8) avoids floating-point comparisons and multiplication overflow at the exact acceptance boundary.
pub fn assertStepFloor(o0_steps: u64, o1_steps: u64) !void {
    return stepFloor(o0_steps, o1_steps, 8);
}

fn stepFloor(o0_steps: u64, o1_steps: u64, divisor: u64) !void {
    if (o0_steps == 0 or o1_steps == 0 or divisor == 0) return error.InvalidStepCount;
    const minimum = o0_steps / divisor + @intFromBool(o0_steps % divisor != 0);
    if (o1_steps < minimum) return error.StepFloorRejected;
}

fn read(comptime T: type, bytes: []const u8, offset: usize) T {
    return std.mem.readInt(T, bytes[offset..][0..@sizeOf(T)], .little);
}

fn sectionBytes(bytes: []const u8, header: usize) ![]const u8 {
    const offset: usize = read(u32, bytes, header + 16);
    const size: usize = read(u32, bytes, header + 20);
    if (offset > bytes.len or size > bytes.len - offset) return error.InvalidElfSections;
    return bytes[offset .. offset + size];
}

// Inspect named ELF sections rather than searching text, symbols, writable input data or the whole ELF file.
pub fn assertNoOutputInRodata(bytes: []const u8, expected: []const u8) !void {
    if (expected.len == 0) return error.EmptyExpectedOutput;
    if (bytes.len < 52 or !std.mem.eql(u8, bytes[0..4], "\x7fELF") or bytes[4] != 1 or bytes[5] != 1)
        return error.InvalidElfSections;
    const table: usize = read(u32, bytes, 32);
    const count: usize = read(u16, bytes, 48);
    const names_index: usize = read(u16, bytes, 50);
    if (read(u16, bytes, 46) != 40 or count == 0 or names_index >= count or
        table > bytes.len or count * 40 > bytes.len - table) return error.InvalidElfSections;
    const names = try sectionBytes(bytes, table + names_index * 40);
    for (0..count) |i| {
        const header = table + i * 40;
        const name_at: usize = read(u32, bytes, header);
        if (name_at >= names.len) return error.InvalidElfSections;
        const end = std.mem.indexOfScalarPos(u8, names, name_at, 0) orelse return error.InvalidElfSections;
        const name = names[name_at..end];
        if (!std.mem.eql(u8, name, ".rodata") and !std.mem.startsWith(u8, name, ".rodata.") and
            !std.mem.eql(u8, name, ".srodata") and !std.mem.startsWith(u8, name, ".srodata.")) continue;
        if (read(u32, bytes, header + 4) != 1) return error.InvalidElfSections;
        const data = try sectionBytes(bytes, header);
        if (std.mem.indexOf(u8, data, expected) != null) return error.OutputInRodata;
    }
}

// Raw C goes through AST opacity; true is reserved for C emitted by trusted Python lowerings.
pub fn checkOptimization(env: *const Env, source: []const u8, expected: []const u8, trusted: bool) !void {
    return checkWithFloor(env, source, expected, trusted, 8);
}

pub fn checkCollectionsOptimization(env: *const Env, source: []const u8, expected: []const u8) !void {
    return checkWithStack(env, source, expected, true, 8, true);
}

fn checkWithFloor(env: *const Env, source: []const u8, expected: []const u8, trusted: bool, divisor: u64) !void {
    return checkWithStack(env, source, expected, trusted, divisor, false);
}

fn checkWithStack(env: *const Env, source: []const u8, expected: []const u8, trusted: bool, divisor: u64, collection_stack: bool) !void {
    var config = try settings(env, trusted);
    config.collection_stack = collection_stack;
    config.optimize = .O0;
    const baseline = try build(source, config);
    defer t.allocator.free(baseline);
    config.optimize = .O1;
    const optimized = try build(source, config);
    defer t.allocator.free(optimized);
    const before = try observe(env, baseline);
    defer before.deinit();
    const after = try observe(env, optimized);
    defer after.deinit();
    try t.expectEqualStrings(expected, before.stdout);
    try t.expectEqualStrings(expected, after.stdout);
    try t.expectEqualStrings(before.stdout, after.stdout);
    std.debug.print("opaque optimization: O0={d} O1={d} output={s}\n", .{ before.steps, after.steps, expected });
    try assertNoOutputInRodata(baseline, expected);
    try assertNoOutputInRodata(optimized, expected);
    stepFloor(before.steps, after.steps, divisor) catch |err| {
        std.debug.print("optimization step floor rejected O0={d}, O1={d}\n", .{ before.steps, after.steps });
        return err;
    };
}

test "raw C opaque O1 preserves numeric work and keeps literal stdout out of rodata" {
    var env = try t.environ.createMap(t.allocator);
    defer env.deinit();
    try checkOptimization(&env, numeric_source, numeric_output, false);
    try checkOptimization(&env, literal_source, literal_output, false);
}

test "trusted Python blocks retain numeric list work across O0 and O1" {
    var env = try t.environ.createMap(t.allocator);
    defer env.deinit();
    _ = try settings(&env, true);
    const source = (try blocks.lowerPython(t.allocator, "values = [7, 11, 13, 17]\nprint(sum(values))")) orelse
        return error.UnexpectedLoweringAbstention;
    defer t.allocator.free(source);
    try checkOptimization(&env, source, "48\n", true);
}

test "trusted Python blocks keep literal text inputs opaque at O1" {
    var env = try t.environ.createMap(t.allocator);
    defer env.deinit();
    _ = try settings(&env, true);
    const source = (try blocks.lowerPython(t.allocator, "print('retained literal output')")) orelse
        return error.UnexpectedLoweringAbstention;
    defer t.allocator.free(source);
    try checkOptimization(&env, source, literal_output, true);
}

test "legacy generator sum retains circuit steps across O0 and O1" {
    var env = try t.environ.createMap(t.allocator);
    defer env.deinit();
    _ = try settings(&env, true);
    const body = (try legacy.lowerPython(t.allocator, "print(sum(i*i for i in range(512)))")) orelse
        return error.UnexpectedLoweringAbstention;
    defer t.allocator.free(body);
    const source = try compiler.wrapBody(t.allocator, body);
    defer t.allocator.free(source);
    try checkOptimization(&env, source, "44608256\n", true);
}

test "numeric folding bypass has exact stdout but fails the executed step floor" {
    var env = try t.environ.createMap(t.allocator);
    defer env.deinit();
    var config = try settings(&env, false);
    config.optimize = .O0;
    const baseline = try build(numeric_source, config);
    defer t.allocator.free(baseline);
    config.optimize = .O1;
    config.opaque_inputs = true;
    const folded = try build(numeric_source, config);
    defer t.allocator.free(folded);
    const before = try observe(&env, baseline);
    defer before.deinit();
    const after = try observe(&env, folded);
    defer after.deinit();
    try t.expectEqualStrings(numeric_output, before.stdout);
    try t.expectEqualStrings(before.stdout, after.stdout);
    try assertNoOutputInRodata(folded, numeric_output);
    std.debug.print("folded negative: O0={d} O1={d}\n", .{ before.steps, after.steps });
    try t.expectError(error.StepFloorRejected, assertStepFloor(before.steps, after.steps));
}

test "literal stdout bypass executes correctly but fails the real ELF rodata check" {
    var env = try t.environ.createMap(t.allocator);
    defer env.deinit();
    var config = try settings(&env, true);
    config.optimize = .O1;
    const folded = try build(literal_source, config);
    defer t.allocator.free(folded);
    const execution = try observe(&env, folded);
    defer execution.deinit();
    try t.expectEqualStrings(literal_output, execution.stdout);
    try t.expectError(error.OutputInRodata, assertNoOutputInRodata(folded, literal_output));
}

test "actual model prime factorial and Fibonacci cells retain circuit work at O1" {
    var env = try t.environ.createMap(t.allocator);
    defer env.deinit();
    _ = try settings(&env, true);
    const saved = @import("runner_frontend_blocks_test.zig");
    for ([_][2][]const u8{
        .{ saved.prime, "100th prime: 541\nSum of first 100 primes: 24133\n" },
        .{ saved.factorial, "0! = 1\n1! = 1\n5! = 120\n12! = 479001600\n" },
    }) |cell| {
        const source = (try blocks.lowerPython(t.allocator, cell[0])) orelse return error.UnexpectedLoweringAbstention;
        defer t.allocator.free(source);
        try checkWithFloor(&env, source, cell[1], true, 32);
    }
    try checkOptimization(&env, @import("runner_frontend_cwide_test.zig").observed_fibonacci, "F(0) = 0\nF(1) = 1\nF(10) = 55\nF(40) = 102334155\n", false);
    try checkOptimization(&env, @import("runner_frontend_budget_test.zig").observed_gcd, "gcd(462, 1071) = 21\n", false);
}

test "step floors retain calibrated function inlining headroom and reject folded work" {
    try stepFloor(563814, 30666, 32);
    try t.expectError(error.StepFloorRejected, stepFloor(57922, 248, 32));
    try assertStepFloor(800, 100);
    try t.expectError(error.StepFloorRejected, assertStepFloor(801, 100));
    try t.expectError(error.InvalidStepCount, stepFloor(1, 1, 0));
}

fn replaceDataWord(bytes: []u8, from: i32, to: i32) !void {
    const table: usize = read(u32, bytes, 32);
    const count: usize = read(u16, bytes, 48);
    const names_index: usize = read(u16, bytes, 50);
    if (read(u16, bytes, 46) != 40 or names_index >= count or table > bytes.len or
        count > (bytes.len - table) / 40) return error.InvalidElfSections;
    const names = try sectionBytes(bytes, table + names_index * 40);
    var matches: usize = 0;
    for (0..count) |i| {
        const header = table + i * 40;
        const name_at: usize = read(u32, bytes, header);
        if (name_at >= names.len) return error.InvalidElfSections;
        const end = std.mem.indexOfScalarPos(u8, names, name_at, 0) orelse return error.InvalidElfSections;
        if (!std.mem.eql(u8, names[name_at..end], ".data")) continue;
        const data = try sectionBytes(bytes, header);
        const offset: usize = read(u32, bytes, header + 16);
        var j: usize = 0;
        while (j + 4 <= data.len) : (j += 4) {
            if (read(i32, data, j) != from) continue;
            std.mem.writeInt(i32, bytes[offset + j ..][0..4], to, .little);
            matches += 1;
        }
    }
    try t.expectEqual(@as(usize, 1), matches);
}

test "optimized C and actual prime cells read mutable ELF inputs instead of folded answers" {
    var env = try t.environ.createMap(t.allocator);
    defer env.deinit();
    var config = try settings(&env, false);
    config.optimize = .O1;
    const c = try build(numeric_source, config);
    defer t.allocator.free(c);
    try replaceDataWord(c, 4096, 1000);
    const c_result = try observe(&env, c);
    defer c_result.deinit();
    try t.expectEqualStrings("500500\n", c_result.stdout);
    const source = (try blocks.lowerPython(t.allocator, @import("runner_frontend_blocks_test.zig").prime)) orelse
        return error.UnexpectedLoweringAbstention;
    defer t.allocator.free(source);
    config.opaque_inputs = true;
    const py = try build(source, config);
    defer t.allocator.free(py);
    try replaceDataWord(py, 100, 10);
    const py_result = try observe(&env, py);
    defer py_result.deinit();
    try t.expectEqualStrings("100th prime: 29\nSum of first 100 primes: 129\n", py_result.stdout);
}

test "implicit static and automatic array zeros remain runtime inputs under O1" {
    var env = try t.environ.createMap(t.allocator);
    defer env.deinit();
    for ([_][]const u8{
        "static const int values[512]; int main(void){int n=0;for(int i=0;i<512;++i)n+=values[i];printf(\"%d\\n\",n);return 0;}",
        "int main(void){int values[512]={0};int n=0;for(int i=1;i<512;++i)n+=values[i];printf(\"%d\\n\",n);return 0;}",
    }) |source| try checkOptimization(&env, source, "0\n", false);
}

test "zero-array aliases and explicit casts cannot turn runtime inputs into answers" {
    var env = try t.environ.createMap(t.allocator);
    defer env.deinit();
    for ([_][]const u8{
        "static const int values[512]; static int sum(const int *p){int n=0;for(int i=0;i<512;++i)n+=p[i];return n;} int main(void){printf(\"%d\\n\",sum(values));return 0;}",
        "static const int values[512]; int main(void){const int *p=(const int *)values;int n=0;for(int i=0;i<512;++i)n+=p[i];printf(\"%d\\n\",n);return 0;}",
    }) |source| try checkOptimization(&env, source, "0\n", false);
}

test "pointer-array input restoration preserves pointer types and runtime dereferences" {
    var env = try t.environ.createMap(t.allocator);
    defer env.deinit();
    const source = "static int value=7;static int *items[2]={&value,&value};int main(void){int n=0;for(int i=0;i<32;++i)n+=*items[i%2];printf(\"%d\\n\",n);return 0;}";
    try checkOptimization(&env, source, "224\n", false);
}

test "global aggregate arithmetic runs after ELF input loads instead of static folding" {
    var env = try t.environ.createMap(t.allocator);
    defer env.deinit();
    const source = "static int values[2][3]={{2*3,7},{11}};int main(void){printf(\"%d\\n\",values[0][0]+values[0][1]+values[1][0]+values[1][1]);return 0;}";
    try checkOptimization(&env, source, "24\n", false);
    var config = try settings(&env, false);
    config.optimize = .O1;
    const bytes = try build(source, config);
    defer t.allocator.free(bytes);
    try replaceDataWord(bytes, 2, 4);
    const result = try observe(&env, bytes);
    defer result.deinit();
    try t.expectEqualStrings("30\n", result.stdout);
}

fn subtractionWork(o: Observation) error{ComputationFolded}!void {
    if (o.subtractions < 512) return error.ComputationFolded;
}

test "C Python blocks and legacy arithmetic identities execute their subtractions" {
    var env = try t.environ.createMap(t.allocator);
    defer env.deinit();
    const c = "int main(void){int x=7,n=0;for(int i=0;i<512;++i)n+=x-x;printf(\"%d\\n\",n);return 0;}";
    const py = "x = 7\nn = 0\nfor i in range(512):\n    n += x-x\nprint(n)";
    const structured = (try blocks.lowerPython(t.allocator, py)) orelse return error.UnexpectedLoweringAbstention;
    defer t.allocator.free(structured);
    const body = (try legacy.lowerPython(t.allocator, py)) orelse return error.UnexpectedLoweringAbstention;
    defer t.allocator.free(body);
    const old = try compiler.wrapBody(t.allocator, body);
    defer t.allocator.free(old);
    for ([_][]const u8{ c, structured, old }, 0..) |source, i| {
        try checkWithFloor(&env, source, "0\n", i != 0, if (i == 1) 32 else 8);
        var config = try settings(&env, i != 0);
        config.optimize = .O1;
        const bytes = try build(source, config);
        defer t.allocator.free(bytes);
        const result = try observe(&env, bytes);
        defer result.deinit();
        try subtractionWork(result);
    }
    var config = try settings(&env, true);
    config.optimize = .O1;
    const folded = try build(c, config);
    defer t.allocator.free(folded);
    const result = try observe(&env, folded);
    defer result.deinit();
    try t.expectEqualStrings("0\n", result.stdout);
    try t.expectError(error.ComputationFolded, subtractionWork(result));
}
