const std = @import("std");
const log = std.log;
const math = std.math;
const testing = std.testing;
const assert = std.debug.assert;
const expect = std.testing.expect;

const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;

test {
    std.testing.refAllDeclsRecursive(@This());
}

pub const Register = enum(u7) {
    // zig fmt: off
    rax, rcx, rdx, rbx, rsp, rbp, rsi, rdi,
    r8, r9, r10, r11, r12, r13, r14, r15,

    eax, ecx, edx, ebx, esp, ebp, esi, edi,
    r8d, r9d, r10d, r11d, r12d, r13d, r14d, r15d,

    ax, cx, dx, bx, sp, bp, si, di,
    r8w, r9w, r10w, r11w, r12w, r13w, r14w, r15w,

    al, cl, dl, bl, spl, bpl, sil, dil,
    r8b, r9b, r10b, r11b, r12b, r13b, r14b, r15b,

    ah, ch, dh, bh,

    ymm0, ymm1, ymm2,  ymm3,  ymm4,  ymm5,  ymm6,  ymm7,
    ymm8, ymm9, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15,

    xmm0, xmm1, xmm2,  xmm3,  xmm4,  xmm5,  xmm6,  xmm7,
    xmm8, xmm9, xmm10, xmm11, xmm12, xmm13, xmm14, xmm15,

    mm0, mm1, mm2, mm3, mm4, mm5, mm6, mm7,

    st0, st1, st2, st3, st4, st5, st6, st7,

    es, cs, ss, ds, fs, gs,

    none,
    // zig fmt: on

    pub const Class = enum {
        general_purpose,
        segment,
        x87,
        mmx,
        sse,
    };

    pub fn class(reg: Register) Class {
        return switch (@intFromEnum(reg)) {
            // zig fmt: off
            @intFromEnum(Register.rax)  ... @intFromEnum(Register.r15)   => .general_purpose,
            @intFromEnum(Register.eax)  ... @intFromEnum(Register.r15d)  => .general_purpose,
            @intFromEnum(Register.ax)   ... @intFromEnum(Register.r15w)  => .general_purpose,
            @intFromEnum(Register.al)   ... @intFromEnum(Register.r15b)  => .general_purpose,
            @intFromEnum(Register.ah)   ... @intFromEnum(Register.bh)    => .general_purpose,

            @intFromEnum(Register.ymm0) ... @intFromEnum(Register.ymm15) => .sse,
            @intFromEnum(Register.xmm0) ... @intFromEnum(Register.xmm15) => .sse,
            @intFromEnum(Register.mm0)  ... @intFromEnum(Register.mm7)   => .mmx,
            @intFromEnum(Register.st0)  ... @intFromEnum(Register.st7)   => .x87,

            @intFromEnum(Register.es)   ... @intFromEnum(Register.gs)    => .segment,

            else => unreachable,
            // zig fmt: on
        };
    }

    pub fn id(reg: Register) u6 {
        const base = switch (@intFromEnum(reg)) {
            // zig fmt: off
            @intFromEnum(Register.rax)  ... @intFromEnum(Register.r15)   => @intFromEnum(Register.rax),
            @intFromEnum(Register.eax)  ... @intFromEnum(Register.r15d)  => @intFromEnum(Register.eax),
            @intFromEnum(Register.ax)   ... @intFromEnum(Register.r15w)  => @intFromEnum(Register.ax),
            @intFromEnum(Register.al)   ... @intFromEnum(Register.r15b)  => @intFromEnum(Register.al),
            @intFromEnum(Register.ah)   ... @intFromEnum(Register.bh)    => @intFromEnum(Register.ah) - 4,

            @intFromEnum(Register.ymm0) ... @intFromEnum(Register.ymm15) => @intFromEnum(Register.ymm0) - 16,
            @intFromEnum(Register.xmm0) ... @intFromEnum(Register.xmm15) => @intFromEnum(Register.xmm0) - 16,
            @intFromEnum(Register.mm0)  ... @intFromEnum(Register.mm7)   => @intFromEnum(Register.mm0) - 32,
            @intFromEnum(Register.st0)  ... @intFromEnum(Register.st7)   => @intFromEnum(Register.st0) - 40,

            @intFromEnum(Register.es)   ... @intFromEnum(Register.gs)    => @intFromEnum(Register.es) - 48,

            else => unreachable,
            // zig fmt: on
        };
        return @as(u6, @intCast(@intFromEnum(reg) - base));
    }

    pub fn bitSize(reg: Register) u64 {
        return switch (@intFromEnum(reg)) {
            // zig fmt: off
            @intFromEnum(Register.rax)  ... @intFromEnum(Register.r15)   => 64,
            @intFromEnum(Register.eax)  ... @intFromEnum(Register.r15d)  => 32,
            @intFromEnum(Register.ax)   ... @intFromEnum(Register.r15w)  => 16,
            @intFromEnum(Register.al)   ... @intFromEnum(Register.r15b)  => 8,
            @intFromEnum(Register.ah)   ... @intFromEnum(Register.bh)    => 8,

            @intFromEnum(Register.ymm0) ... @intFromEnum(Register.ymm15) => 256,
            @intFromEnum(Register.xmm0) ... @intFromEnum(Register.xmm15) => 128,
            @intFromEnum(Register.mm0)  ... @intFromEnum(Register.mm7)   => 64,
            @intFromEnum(Register.st0)  ... @intFromEnum(Register.st7)   => 80,

            @intFromEnum(Register.es)   ... @intFromEnum(Register.gs)    => 16,

            else => unreachable,
            // zig fmt: on
        };
    }

    pub fn isExtended(reg: Register) bool {
        return switch (@intFromEnum(reg)) {
            // zig fmt: off
            @intFromEnum(Register.r8)  ... @intFromEnum(Register.r15)    => true,
            @intFromEnum(Register.r8d) ... @intFromEnum(Register.r15d)   => true,
            @intFromEnum(Register.r8w) ... @intFromEnum(Register.r15w)   => true,
            @intFromEnum(Register.r8b) ... @intFromEnum(Register.r15b)   => true,

            @intFromEnum(Register.ymm8) ... @intFromEnum(Register.ymm15) => true,
            @intFromEnum(Register.xmm8) ... @intFromEnum(Register.xmm15) => true,

            else => false,
            // zig fmt: on
        };
    }

    pub fn enc(reg: Register) u4 {
        const base = switch (@intFromEnum(reg)) {
            // zig fmt: off
            @intFromEnum(Register.rax)  ... @intFromEnum(Register.r15)   => @intFromEnum(Register.rax),
            @intFromEnum(Register.eax)  ... @intFromEnum(Register.r15d)  => @intFromEnum(Register.eax),
            @intFromEnum(Register.ax)   ... @intFromEnum(Register.r15w)  => @intFromEnum(Register.ax),
            @intFromEnum(Register.al)   ... @intFromEnum(Register.r15b)  => @intFromEnum(Register.al),
            @intFromEnum(Register.ah)   ... @intFromEnum(Register.bh)    => @intFromEnum(Register.ah) - 4,

            @intFromEnum(Register.ymm0) ... @intFromEnum(Register.ymm15) => @intFromEnum(Register.ymm0),
            @intFromEnum(Register.xmm0) ... @intFromEnum(Register.xmm15) => @intFromEnum(Register.xmm0),
            @intFromEnum(Register.mm0)  ... @intFromEnum(Register.mm7)   => @intFromEnum(Register.mm0),
            @intFromEnum(Register.st0)  ... @intFromEnum(Register.st7)   => @intFromEnum(Register.st0),

            @intFromEnum(Register.es)   ... @intFromEnum(Register.gs)    => @intFromEnum(Register.es),

            else => unreachable,
            // zig fmt: on
        };
        return @as(u4, @truncate(@intFromEnum(reg) - base));
    }

    pub fn lowEnc(reg: Register) u3 {
        return @as(u3, @truncate(reg.enc()));
    }

    pub fn toBitSize(reg: Register, bit_size: u64) Register {
        return switch (bit_size) {
            8 => reg.to8(),
            16 => reg.to16(),
            32 => reg.to32(),
            64 => reg.to64(),
            128 => reg.to128(),
            256 => reg.to256(),
            else => unreachable,
        };
    }

    fn gpBase(reg: Register) u7 {
        assert(reg.class() == .general_purpose);
        return switch (@intFromEnum(reg)) {
            // zig fmt: off
            @intFromEnum(Register.rax)  ... @intFromEnum(Register.r15)   => @intFromEnum(Register.rax),
            @intFromEnum(Register.eax)  ... @intFromEnum(Register.r15d)  => @intFromEnum(Register.eax),
            @intFromEnum(Register.ax)   ... @intFromEnum(Register.r15w)  => @intFromEnum(Register.ax),
            @intFromEnum(Register.al)   ... @intFromEnum(Register.r15b)  => @intFromEnum(Register.al),
            @intFromEnum(Register.ah)   ... @intFromEnum(Register.bh)    => @intFromEnum(Register.ah) - 4,
            else => unreachable,
            // zig fmt: on
        };
    }

    pub fn to64(reg: Register) Register {
        return @as(Register, @enumFromInt(@intFromEnum(reg) - reg.gpBase() + @intFromEnum(Register.rax)));
    }

    pub fn to32(reg: Register) Register {
        return @as(Register, @enumFromInt(@intFromEnum(reg) - reg.gpBase() + @intFromEnum(Register.eax)));
    }

    pub fn to16(reg: Register) Register {
        return @as(Register, @enumFromInt(@intFromEnum(reg) - reg.gpBase() + @intFromEnum(Register.ax)));
    }

    pub fn to8(reg: Register) Register {
        return @as(Register, @enumFromInt(@intFromEnum(reg) - reg.gpBase() + @intFromEnum(Register.al)));
    }

    fn sseBase(reg: Register) u7 {
        assert(reg.class() == .sse);
        return switch (@intFromEnum(reg)) {
            @intFromEnum(Register.ymm0)...@intFromEnum(Register.ymm15) => @intFromEnum(Register.ymm0),
            @intFromEnum(Register.xmm0)...@intFromEnum(Register.xmm15) => @intFromEnum(Register.xmm0),
            else => unreachable,
        };
    }

    pub fn to256(reg: Register) Register {
        return @as(Register, @enumFromInt(@intFromEnum(reg) - reg.sseBase() + @intFromEnum(Register.ymm0)));
    }

    pub fn to128(reg: Register) Register {
        return @as(Register, @enumFromInt(@intFromEnum(reg) - reg.sseBase() + @intFromEnum(Register.xmm0)));
    }
};

test "Register id - different classes" {
    try expect(Register.al.id() == Register.ax.id());
    try expect(Register.ah.id() == Register.spl.id());
    try expect(Register.ax.id() == Register.eax.id());
    try expect(Register.eax.id() == Register.rax.id());

    try expect(Register.ymm0.id() == 0b10000);
    try expect(Register.ymm0.id() != Register.rax.id());
    try expect(Register.xmm0.id() == Register.ymm0.id());
    try expect(Register.xmm0.id() != Register.mm0.id());
    try expect(Register.mm0.id() != Register.st0.id());

    try expect(Register.es.id() == 0b110000);
}

test "Register enc - different classes" {
    try expect(Register.al.enc() == Register.ax.enc());
    try expect(Register.ax.enc() == Register.eax.enc());
    try expect(Register.eax.enc() == Register.rax.enc());
    try expect(Register.ymm0.enc() == Register.rax.enc());
    try expect(Register.xmm0.enc() == Register.ymm0.enc());
    try expect(Register.es.enc() == Register.rax.enc());
}

test "Register classes" {
    try expect(Register.r11.class() == .general_purpose);
    try expect(Register.ymm11.class() == .sse);
    try expect(Register.mm3.class() == .mmx);
    try expect(Register.st3.class() == .x87);
    try expect(Register.fs.class() == .segment);
}

pub const FrameIndex = enum(u32) {
    // This index refers to the start of the arguments passed to this function
    args_frame,
    // This index refers to the return address pushed by a `call` and popped by a `ret`.
    ret_addr,
    // This index refers to the base pointer pushed in the prologue and popped in the epilogue.
    base_ptr,
    // This index refers to the entire stack frame.
    stack_frame,
    // This index refers to the start of the call frame for arguments passed to called functions
    call_frame,
    // Other indices are used for local variable stack slots
    _,

    pub const named_count = @typeInfo(FrameIndex).@"enum".fields.len;

    pub fn isNamed(fi: FrameIndex) bool {
        return @intFromEnum(fi) < named_count;
    }

    pub fn format(
        fi: FrameIndex,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) @TypeOf(writer).Error!void {
        try writer.writeAll("FrameIndex");
        if (fi.isNamed()) {
            try writer.writeByte('.');
            try writer.writeAll(@tagName(fi));
        } else {
            try writer.writeByte('(');
            try std.fmt.formatType(@intFromEnum(fi), fmt, options, writer, 0);
            try writer.writeByte(')');
        }
    }
};

pub const Base = union(enum) {
    none,
    reg: Register,
    frame_index: FrameIndex,

    pub const Tag = @typeInfo(Base).@"union".tag_type.?;

    pub fn register(reg: Register) Base {
        return .{ .reg = reg };
    }

    pub fn frame(fi: FrameIndex) Base {
        return .{ .frame_index = fi };
    }

    pub fn isExtended(self: Base) bool {
        return switch (self) {
            .none, .frame_index => false, // neither rsp nor rbp are extended
            .reg => |reg| reg.isExtended(),
        };
    }
};

pub const ScaleIndex = struct {
    scale: u4,
    index: Register,

    const none = ScaleIndex{ .scale = 0, .index = undefined };
};

pub const PtrSize = enum {
    byte,
    word,
    dword,
    qword,
    tbyte,
    xword,
    yword,
    zword,

    pub fn fromSize(size: u32) PtrSize {
        return switch (size) {
            1...1 => .byte,
            2...2 => .word,
            3...4 => .dword,
            5...8 => .qword,
            9...16 => .xword,
            17...32 => .yword,
            33...64 => .zword,
            else => unreachable,
        };
    }

    pub fn fromBitSize(bit_size: u64) PtrSize {
        return switch (bit_size) {
            8 => .byte,
            16 => .word,
            32 => .dword,
            64 => .qword,
            80 => .tbyte,
            128 => .xword,
            256 => .yword,
            512 => .zword,
            else => unreachable,
        };
    }

    pub fn bitSize(s: PtrSize) u64 {
        return switch (s) {
            .byte => 8,
            .word => 16,
            .dword => 32,
            .qword => 64,
            .tbyte => 80,
            .xword => 128,
            .yword => 256,
            .zword => 512,
        };
    }
};

pub const Sib = struct {
    ptr_size: PtrSize,
    base: Base,
    scale_index: ScaleIndex,
    disp: i32,
};

pub const Rip = struct {
    ptr_size: PtrSize,
    disp: i32,
};

pub const Moffs = struct {
    seg: Register,
    offset: u64,
};

pub const Memory = union(enum) {
    m_sib: Sib,
    m_rip: Rip,
    m_moffs: Moffs,

    pub fn moffs(reg: Register, offset: u64) Memory {
        assert(reg.class() == .segment);
        return .{ .m_moffs = .{ .seg = reg, .offset = offset } };
    }

    pub fn ptr_offset(disp: i32, offset: Register) Memory {
        return .{ .m_sib = .{
            .base = .register(offset),
            .disp = disp,
            .ptr_size = .qword,
            .scale_index = ScaleIndex.none,
        } };
    }

    pub fn sib(ptr_size: PtrSize, args: struct {
        disp: i32 = 0,
        base: Base = .none,
        scale_index: ?ScaleIndex = null,
    }) Memory {
        if (args.scale_index) |si| assert(std.math.isPowerOfTwo(si.scale));
        return .{ .m_sib = .{
            .base = args.base,
            .disp = args.disp,
            .ptr_size = ptr_size,
            .scale_index = if (args.scale_index) |si| si else ScaleIndex.none,
        } };
    }

    pub fn rip(ptr_size: PtrSize, disp: i32) Memory {
        return .{ .m_rip = .{ .ptr_size = ptr_size, .disp = disp } };
    }

    pub fn isSegmentRegister(mem: Memory) bool {
        return switch (mem) {
            .m_moffs => true,
            .m_rip => false,
            .m_sib => |s| switch (s.base) {
                .none, .frame_index => false,
                .reg => |reg| reg.class() == .segment,
            },
        };
    }

    pub fn base(mem: Memory) Base {
        return switch (mem) {
            .m_moffs => |m| .{ .reg = m.seg },
            .m_sib => |s| s.base,
            .m_rip => .none,
        };
    }

    pub fn scaleIndex(mem: Memory) ?ScaleIndex {
        return switch (mem) {
            .m_moffs, .m_rip => null,
            .m_sib => |s| if (s.scale_index.scale > 0) s.scale_index else null,
        };
    }

    pub fn bitSize(mem: Memory) u64 {
        return switch (mem) {
            .m_rip => |r| r.ptr_size.bitSize(),
            .m_sib => |s| s.ptr_size.bitSize(),
            .m_moffs => 64,
        };
    }
};

pub const Immediate = union(enum) {
    signed: i32,
    unsigned: u64,

    pub fn u(x: u64) Immediate {
        return .{ .unsigned = x };
    }

    pub fn s(x: i32) Immediate {
        return .{ .signed = x };
    }

    pub fn asUnsigned(imm: Immediate, bit_size: u64) u64 {
        return switch (imm) {
            .signed => |x| switch (bit_size) {
                1, 8 => @as(u8, @bitCast(@as(i8, @intCast(x)))),
                16 => @as(u16, @bitCast(@as(i16, @intCast(x)))),
                32, 64 => @as(u32, @bitCast(x)),
                else => unreachable,
            },
            .unsigned => |x| switch (bit_size) {
                1, 8 => @as(u8, @intCast(x)),
                16 => @as(u16, @intCast(x)),
                32 => @as(u32, @intCast(x)),
                64 => x,
                else => unreachable,
            },
        };
    }
};

pub const Prefix = enum(u3) {
    none,
    lock,
    rep,
    repe,
    repz,
    repne,
    repnz,
};

pub const Operand = union(enum) {
    none,
    reg: Register,
    mem: Memory,
    imm: Immediate,

    pub fn im(x: anytype) Operand {
        return .{ .imm = .{ .unsigned = @as(std.meta.Int(.unsigned, @bitSizeOf(@TypeOf(x))), @bitCast(x)) } };
    }

    pub const rax = Operand.register(.rax);
    pub const rcx = Operand.register(.rcx);
    pub const rdx = Operand.register(.rdx);
    pub const rbx = Operand.register(.rbx);
    pub const rsp = Operand.register(.rsp);
    pub const rbp = Operand.register(.rbp);
    pub const rsi = Operand.register(.rsi);
    pub const rdi = Operand.register(.rdi);
    pub const r8 = Operand.register(.r8);
    pub const r9 = Operand.register(.r9);
    pub const r10 = Operand.register(.r10);
    pub const r11 = Operand.register(.r11);
    pub const r12 = Operand.register(.r12);
    pub const r13 = Operand.register(.r13);
    pub const r14 = Operand.register(.r14);
    pub const r15 = Operand.register(.r15);

    pub const eax = Operand.register(.eax);
    pub const ecx = Operand.register(.ecx);
    pub const edx = Operand.register(.edx);
    pub const ebx = Operand.register(.ebx);
    pub const esp = Operand.register(.esp);
    pub const ebp = Operand.register(.ebp);
    pub const esi = Operand.register(.esi);
    pub const edi = Operand.register(.edi);
    pub const r8d = Operand.register(.r8d);
    pub const r9d = Operand.register(.r9d);
    pub const r10d = Operand.register(.r10d);
    pub const r11d = Operand.register(.r11d);
    pub const r12d = Operand.register(.r12d);
    pub const r13d = Operand.register(.r13d);
    pub const r14d = Operand.register(.r14d);
    pub const r15d = Operand.register(.r15d);

    pub const ax = Operand.register(.ax);
    pub const cx = Operand.register(.cx);
    pub const dx = Operand.register(.dx);
    pub const bx = Operand.register(.bx);
    pub const sp = Operand.register(.sp);
    pub const bp = Operand.register(.bp);
    pub const si = Operand.register(.si);
    pub const di = Operand.register(.di);
    pub const r8w = Operand.register(.r8w);
    pub const r9w = Operand.register(.r9w);
    pub const r10w = Operand.register(.r10w);
    pub const r11w = Operand.register(.r11w);
    pub const r12w = Operand.register(.r12w);
    pub const r13w = Operand.register(.r13w);
    pub const r14w = Operand.register(.r14w);
    pub const r15w = Operand.register(.r15w);

    pub const al = Operand.register(.al);
    pub const cl = Operand.register(.cl);
    pub const dl = Operand.register(.dl);
    pub const bl = Operand.register(.bl);
    pub const spl = Operand.register(.spl);
    pub const bpl = Operand.register(.bpl);
    pub const sil = Operand.register(.sil);
    pub const dil = Operand.register(.dil);
    pub const r8b = Operand.register(.r8b);
    pub const r9b = Operand.register(.r9b);
    pub const r10b = Operand.register(.r10b);
    pub const r11b = Operand.register(.r11b);
    pub const r12b = Operand.register(.r12b);
    pub const r13b = Operand.register(.r13b);
    pub const r14b = Operand.register(.r14b);
    pub const r15b = Operand.register(.r15b);

    pub const ah = Operand.register(.ah);
    pub const ch = Operand.register(.ch);
    pub const dh = Operand.register(.dh);
    pub const bh = Operand.register(.bh);

    pub const ymm0 = Operand.register(.ymm0);
    pub const ymm1 = Operand.register(.ymm1);
    pub const ymm2 = Operand.register(.ymm2);
    pub const ymm3 = Operand.register(.ymm3);
    pub const ymm4 = Operand.register(.ymm4);
    pub const ymm5 = Operand.register(.ymm5);
    pub const ymm6 = Operand.register(.ymm6);
    pub const ymm7 = Operand.register(.ymm7);
    pub const ymm8 = Operand.register(.ymm8);
    pub const ymm9 = Operand.register(.ymm9);
    pub const ymm10 = Operand.register(.ymm10);
    pub const ymm11 = Operand.register(.ymm11);
    pub const ymm12 = Operand.register(.ymm12);
    pub const ymm13 = Operand.register(.ymm13);
    pub const ymm14 = Operand.register(.ymm14);
    pub const ymm15 = Operand.register(.ymm15);

    pub const xmm0 = Operand.register(.xmm0);
    pub const xmm1 = Operand.register(.xmm1);
    pub const xmm2 = Operand.register(.xmm2);
    pub const xmm3 = Operand.register(.xmm3);
    pub const xmm4 = Operand.register(.xmm4);
    pub const xmm5 = Operand.register(.xmm5);
    pub const xmm6 = Operand.register(.xmm6);
    pub const xmm7 = Operand.register(.xmm7);
    pub const xmm8 = Operand.register(.xmm8);
    pub const xmm9 = Operand.register(.xmm9);
    pub const xmm10 = Operand.register(.xmm10);
    pub const xmm11 = Operand.register(.xmm11);
    pub const xmm12 = Operand.register(.xmm12);
    pub const xmm13 = Operand.register(.xmm13);
    pub const xmm14 = Operand.register(.xmm14);
    pub const xmm15 = Operand.register(.xmm15);

    pub const mm0 = Operand.register(.mm0);
    pub const mm1 = Operand.register(.mm1);
    pub const mm2 = Operand.register(.mm2);
    pub const mm3 = Operand.register(.mm3);
    pub const mm4 = Operand.register(.mm4);
    pub const mm5 = Operand.register(.mm5);
    pub const mm6 = Operand.register(.mm6);
    pub const mm7 = Operand.register(.mm7);

    pub const st0 = Operand.register(.st0);
    pub const st1 = Operand.register(.st1);
    pub const st2 = Operand.register(.st2);
    pub const st3 = Operand.register(.st3);
    pub const st4 = Operand.register(.st4);
    pub const st5 = Operand.register(.st5);
    pub const st6 = Operand.register(.st6);
    pub const st7 = Operand.register(.st7);

    pub const es = Operand.register(.es);
    pub const cs = Operand.register(.cs);
    pub const ss = Operand.register(.ss);
    pub const ds = Operand.register(.ds);
    pub const fs = Operand.register(.fs);
    pub const gs = Operand.register(.gs);

    pub fn ptr_offset(r: Register, offset: i32) Operand {
        return .{ .mem = Memory.ptr_offset(offset, r) };
    }

    pub fn register(r: Register) Operand {
        return .{ .reg = r };
    }

    pub fn memory(mem: Memory) Operand {
        return .{ .mem = mem };
    }

    pub fn immediate(imm: Immediate) Operand {
        return .{ .imm = imm };
    }

    /// Returns the bitsize of the operand.
    pub fn bitSize(op: Operand) u64 {
        return switch (op) {
            .none => unreachable,
            .reg => |reg| reg.bitSize(),
            .mem => |mem| mem.bitSize(),
            .imm => unreachable,
        };
    }

    /// Returns true if the operand is a segment register.
    /// Asserts the operand is either register or memory.
    pub fn isSegmentRegister(op: Operand) bool {
        return switch (op) {
            .none => unreachable,
            .reg => |reg| reg.class() == .segment,
            .mem => |mem| mem.isSegmentRegister(),
            .imm => unreachable,
        };
    }

    pub fn isBaseExtended(op: Operand) bool {
        return switch (op) {
            .none, .imm => false,
            .reg => |reg| reg.isExtended(),
            .mem => |mem| mem.base().isExtended(),
        };
    }

    pub fn isIndexExtended(op: Operand) bool {
        return switch (op) {
            .none, .reg, .imm => false,
            .mem => |mem| if (mem.scaleIndex()) |x| x.index.isExtended() else false,
        };
    }

    fn format(
        op: Operand,
        comptime unused_format_string: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = op;
        _ = unused_format_string;
        _ = options;
        _ = writer;
        @compileError("do not format Operand directly; use fmtPrint() instead");
    }

    pub fn fmtPrint(op: Operand, enc_op: Op) std.fmt.Formatter(fmtOperand) {
        return .{ .data = .{ .op = op, .enc_op = enc_op } };
    }
};

const FormatContext = struct {
    op: Operand,
    enc_op: Op,
};

fn fmtOperand(
    ctx: FormatContext,
    comptime unused_format_string: []const u8,
    options: std.fmt.FormatOptions,
    writer: anytype,
) @TypeOf(writer).Error!void {
    _ = unused_format_string;
    _ = options;
    const op = ctx.op;
    const enc_op = ctx.enc_op;
    switch (op) {
        .none => {},
        .reg => |reg| try writer.writeAll(@tagName(reg)),
        .mem => |mem| switch (mem) {
            .m_rip => |rip| {
                try writer.print("{s} ptr [rip", .{@tagName(rip.ptr_size)});
                if (rip.disp != 0) try writer.print(" {c} 0x{x}", .{
                    @as(u8, if (rip.disp < 0) '-' else '+'),
                    @abs(rip.disp),
                });
                try writer.writeByte(']');
            },
            .m_sib => |sib| {
                try writer.print("{s} ptr ", .{@tagName(sib.ptr_size)});

                if (mem.isSegmentRegister()) {
                    return writer.print("{s}:0x{x}", .{ @tagName(sib.base.reg), sib.disp });
                }

                try writer.writeByte('[');

                var any = false;
                switch (sib.base) {
                    .none => {},
                    .reg => |reg| {
                        try writer.print("{s}", .{@tagName(reg)});
                        any = true;
                    },
                    .frame_index => |frame| {
                        try writer.print("{}", .{frame});
                        any = true;
                    },
                }
                if (mem.scaleIndex()) |si| {
                    if (any) try writer.writeAll(" + ");
                    try writer.print("{s} * {d}", .{ @tagName(si.index), si.scale });
                    any = true;
                }
                if (sib.disp != 0 or !any) {
                    if (any)
                        try writer.print(" {c} ", .{@as(u8, if (sib.disp < 0) '-' else '+')})
                    else if (sib.disp < 0)
                        try writer.writeByte('-');
                    try writer.print("0x{x}", .{@abs(sib.disp)});
                    any = true;
                }

                try writer.writeByte(']');
            },
            .m_moffs => |moffs| try writer.print("{s}:0x{x}", .{
                @tagName(moffs.seg),
                moffs.offset,
            }),
        },
        .imm => |imm| try writer.print("0x{x}", .{imm.asUnsigned(enc_op.immBitSize())}),
    }
}

pub const Instruction = struct {
    prefix: Prefix = .none,
    encoding: Encoding,
    ops: [4]Operand = .{.none} ** 4,

    pub fn new(prefix: Prefix, mnemonic: Mnemonic, ops: []const Operand) error{BadEncoding}!Instruction {
        const encoding = Encoding.findByMnemonic(prefix, mnemonic, ops) orelse {
            if (@inComptime()) {
                @compileLog("no encoding found for: ", prefix, mnemonic, ops);
            } else {
                log.err("no encoding found for: {s} {s} {s} {s} {s} {s}{s}", .{ @tagName(prefix), @tagName(mnemonic), @tagName(if (ops.len > 0) Op.fromOperand(ops[0]) else .none), @tagName(if (ops.len > 1) Op.fromOperand(ops[1]) else .none), @tagName(if (ops.len > 2) Op.fromOperand(ops[2]) else .none), @tagName(if (ops.len > 3) Op.fromOperand(ops[3]) else .none), if (ops.len > 3) " ...?" else "" });
            }
            return error.BadEncoding;
        };

        if (@inComptime()) {
            // @compileLog("selected encoding", .{encoding});
        } else {
            log.debug("selected encoding: {}", .{encoding});
        }

        var inst = Instruction{
            .prefix = prefix,
            .encoding = encoding,
            .ops = [1]Operand{.none} ** 4,
        };
        @memcpy(inst.ops[0..ops.len], ops);
        return inst;
    }

    pub fn format(
        inst: Instruction,
        comptime unused_format_string: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) @TypeOf(writer).Error!void {
        _ = unused_format_string;
        _ = options;
        if (inst.prefix != .none) try writer.print("{s} ", .{@tagName(inst.prefix)});
        try writer.print("{s}", .{@tagName(inst.encoding.mnemonic)});
        for (inst.ops, inst.encoding.data.ops, 0..) |op, enc, i| {
            if (op == .none) break;
            if (i > 0) try writer.writeByte(',');
            try writer.writeByte(' ');
            try writer.print("{}", .{op.fmtPrint(enc)});
        }
    }

    pub fn encode(inst: Instruction, writer: anytype, comptime opts: EncoderOptions) !void {
        const e = Encoder(@TypeOf(writer), opts){ .writer = writer };
        const enc = inst.encoding;
        const data = enc.data;

        if (data.mode.isVex()) {
            try inst.encodeVexPrefix(e);
            const opc = inst.encoding.opcode();
            try e.opcode_1byte(opc[opc.len - 1]);
        } else {
            try inst.encodeLegacyPrefixes(e);
            try inst.encodeMandatoryPrefix(e);
            try inst.encodeRexPrefix(e);
            try inst.encodeOpcode(e);
        }

        switch (data.op_en) {
            .np, .o => {},
            .i, .d => try encodeImm(inst.ops[0].imm, data.ops[0], e),
            .zi, .oi => try encodeImm(inst.ops[1].imm, data.ops[1], e),
            .fd => try e.imm64(inst.ops[1].mem.m_moffs.offset),
            .td => try e.imm64(inst.ops[0].mem.m_moffs.offset),
            else => {
                const mem_op = switch (data.op_en) {
                    .m, .mi, .m1, .mc, .mr, .mri, .mrc, .mvr => inst.ops[0],
                    .rm, .rmi, .rm0, .vmi => inst.ops[1],
                    .rvm, .rvmr, .rvmi => inst.ops[2],
                    else => unreachable,
                };
                switch (mem_op) {
                    .reg => |reg| {
                        const rm = switch (data.op_en) {
                            .m, .mi, .m1, .mc, .vmi => enc.modRmExt(),
                            .mr, .mri, .mrc => inst.ops[1].reg.lowEnc(),
                            .rm, .rmi, .rm0, .rvm, .rvmr, .rvmi => inst.ops[0].reg.lowEnc(),
                            .mvr => inst.ops[2].reg.lowEnc(),
                            else => unreachable,
                        };
                        try e.modRm_direct(rm, reg.lowEnc());
                    },
                    .mem => |mem| {
                        const op = switch (data.op_en) {
                            .m, .mi, .m1, .mc, .vmi => .none,
                            .mr, .mri, .mrc => inst.ops[1],
                            .rm, .rmi, .rm0, .rvm, .rvmr, .rvmi => inst.ops[0],
                            .mvr => inst.ops[2],
                            else => unreachable,
                        };
                        try encodeMemory(enc, mem, op, e);
                    },
                    else => unreachable,
                }

                switch (data.op_en) {
                    .mi => try encodeImm(inst.ops[1].imm, data.ops[1], e),
                    .rmi, .mri, .vmi => try encodeImm(inst.ops[2].imm, data.ops[2], e),
                    .rvmr => try e.imm8(@as(u8, inst.ops[3].reg.enc()) << 4),
                    .rvmi => try encodeImm(inst.ops[3].imm, data.ops[3], e),
                    else => {},
                }
            },
        }
    }

    fn encodeOpcode(inst: Instruction, e: anytype) !void {
        const opcode = inst.encoding.opcode();
        const first = @intFromBool(inst.encoding.mandatoryPrefix() != null);
        const final = opcode.len - 1;
        for (opcode[first..final]) |byte| try e.opcode_1byte(byte);
        switch (inst.encoding.data.op_en) {
            .o, .oi => try e.opcode_withReg(opcode[final], inst.ops[0].reg.lowEnc()),
            else => try e.opcode_1byte(opcode[final]),
        }
    }

    fn encodeLegacyPrefixes(inst: Instruction, e: anytype) !void {
        const enc = inst.encoding;
        const data = enc.data;
        const op_en = data.op_en;

        var legacy = LegacyPrefixes{};

        switch (inst.prefix) {
            .none => {},
            .lock => legacy.prefix_f0 = true,
            .repne, .repnz => legacy.prefix_f2 = true,
            .rep, .repe, .repz => legacy.prefix_f3 = true,
        }

        switch (data.mode) {
            .short, .rex_short => legacy.set16BitOverride(),
            else => {},
        }

        const segment_override: ?Register = switch (op_en) {
            .i, .zi, .o, .oi, .d, .np => null,
            .fd => inst.ops[1].mem.base().reg,
            .td => inst.ops[0].mem.base().reg,
            .rm, .rmi, .rm0 => if (inst.ops[1].isSegmentRegister())
                switch (inst.ops[1]) {
                    .reg => |reg| reg,
                    .mem => |mem| mem.base().reg,
                    else => unreachable,
                }
            else
                null,
            .m, .mi, .m1, .mc, .mr, .mri, .mrc => if (inst.ops[0].isSegmentRegister())
                switch (inst.ops[0]) {
                    .reg => |reg| reg,
                    .mem => |mem| mem.base().reg,
                    else => unreachable,
                }
            else
                null,
            .vmi, .rvm, .rvmr, .rvmi, .mvr => unreachable,
        };
        if (segment_override) |seg| {
            legacy.setSegmentOverride(seg);
        }

        try e.legacyPrefixes(legacy);
    }

    fn encodeRexPrefix(inst: Instruction, e: anytype) !void {
        const op_en = inst.encoding.data.op_en;

        var rex = Rex{};
        rex.present = inst.encoding.data.mode == .rex;
        rex.w = inst.encoding.data.mode == .long;

        switch (op_en) {
            .np, .i, .zi, .fd, .td, .d => {},
            .o, .oi => rex.b = inst.ops[0].reg.isExtended(),
            .m, .mi, .m1, .mc, .mr, .rm, .rmi, .mri, .mrc, .rm0 => {
                const r_op = switch (op_en) {
                    .rm, .rmi, .rm0 => inst.ops[0],
                    .mr, .mri, .mrc => inst.ops[1],
                    else => .none,
                };
                rex.r = r_op.isBaseExtended();

                const b_x_op = switch (op_en) {
                    .rm, .rmi, .rm0 => inst.ops[1],
                    .m, .mi, .m1, .mc, .mr, .mri, .mrc => inst.ops[0],
                    else => unreachable,
                };
                rex.b = b_x_op.isBaseExtended();
                rex.x = b_x_op.isIndexExtended();
            },
            .vmi, .rvm, .rvmr, .rvmi, .mvr => unreachable,
        }

        try e.rex(rex);
    }

    fn encodeVexPrefix(inst: Instruction, e: anytype) !void {
        const op_en = inst.encoding.data.op_en;
        const opc = inst.encoding.opcode();
        const mand_pre = inst.encoding.mandatoryPrefix();

        var vex = Vex{};

        vex.w = inst.encoding.data.mode.isLong();

        switch (op_en) {
            .np, .i, .zi, .fd, .td, .d => {},
            .o, .oi => vex.b = inst.ops[0].reg.isExtended(),
            .m, .mi, .m1, .mc, .mr, .rm, .rmi, .mri, .mrc, .rm0, .vmi, .rvm, .rvmr, .rvmi, .mvr => {
                const r_op = switch (op_en) {
                    .rm, .rmi, .rm0, .rvm, .rvmr, .rvmi => inst.ops[0],
                    .mr, .mri, .mrc => inst.ops[1],
                    .mvr => inst.ops[2],
                    .m, .mi, .m1, .mc, .vmi => .none,
                    else => unreachable,
                };
                vex.r = r_op.isBaseExtended();

                const b_x_op = switch (op_en) {
                    .rm, .rmi, .rm0, .vmi => inst.ops[1],
                    .m, .mi, .m1, .mc, .mr, .mri, .mrc, .mvr => inst.ops[0],
                    .rvm, .rvmr, .rvmi => inst.ops[2],
                    else => unreachable,
                };
                vex.b = b_x_op.isBaseExtended();
                vex.x = b_x_op.isIndexExtended();
            },
        }

        vex.l = inst.encoding.data.mode.isVecLong();

        vex.p = if (mand_pre) |mand| switch (mand) {
            0x66 => .@"66",
            0xf2 => .f2,
            0xf3 => .f3,
            else => unreachable,
        } else .none;

        const leading: usize = if (mand_pre) |_| 1 else 0;
        assert(opc[leading] == 0x0f);
        vex.m = switch (opc[leading + 1]) {
            else => .@"0f",
            0x38 => .@"0f38",
            0x3a => .@"0f3a",
        };

        switch (op_en) {
            else => {},
            .vmi => vex.v = inst.ops[0].reg,
            .rvm, .rvmr, .rvmi => vex.v = inst.ops[1].reg,
        }

        try e.vex(vex);
    }

    fn encodeMandatoryPrefix(inst: Instruction, e: anytype) !void {
        const prefix = inst.encoding.mandatoryPrefix() orelse return;
        try e.opcode_1byte(prefix);
    }

    fn encodeMemory(encoding: Encoding, mem: Memory, operand: Operand, e: anytype) !void {
        const operand_enc = switch (operand) {
            .reg => |reg| reg.lowEnc(),
            .none => encoding.modRmExt(),
            else => unreachable,
        };

        switch (mem) {
            .m_moffs => unreachable,
            .m_sib => |sib| switch (sib.base) {
                .none => {
                    try e.modRm_SIBDisp0(operand_enc);
                    if (mem.scaleIndex()) |si| {
                        const scale = math.log2_int(u4, si.scale);
                        try e.sib_scaleIndexDisp32(scale, si.index.lowEnc());
                    } else {
                        try e.sib_disp32();
                    }
                    try e.disp32(sib.disp);
                },
                .reg => |base| if (base.class() == .segment) {
                    // TODO audit this wrt SIB
                    try e.modRm_SIBDisp0(operand_enc);
                    if (mem.scaleIndex()) |si| {
                        const scale = math.log2_int(u4, si.scale);
                        try e.sib_scaleIndexDisp32(scale, si.index.lowEnc());
                    } else {
                        try e.sib_disp32();
                    }
                    try e.disp32(sib.disp);
                } else {
                    assert(base.class() == .general_purpose);
                    const dst = base.lowEnc();
                    const src = operand_enc;
                    if (dst == 4 or mem.scaleIndex() != null) {
                        if (sib.disp == 0 and dst != 5) {
                            try e.modRm_SIBDisp0(src);
                            if (mem.scaleIndex()) |si| {
                                const scale = math.log2_int(u4, si.scale);
                                try e.sib_scaleIndexBase(scale, si.index.lowEnc(), dst);
                            } else {
                                try e.sib_base(dst);
                            }
                        } else if (math.cast(i8, sib.disp)) |_| {
                            try e.modRm_SIBDisp8(src);
                            if (mem.scaleIndex()) |si| {
                                const scale = math.log2_int(u4, si.scale);
                                try e.sib_scaleIndexBaseDisp8(scale, si.index.lowEnc(), dst);
                            } else {
                                try e.sib_baseDisp8(dst);
                            }
                            try e.disp8(@as(i8, @truncate(sib.disp)));
                        } else {
                            try e.modRm_SIBDisp32(src);
                            if (mem.scaleIndex()) |si| {
                                const scale = math.log2_int(u4, si.scale);
                                try e.sib_scaleIndexBaseDisp32(scale, si.index.lowEnc(), dst);
                            } else {
                                try e.sib_baseDisp32(dst);
                            }
                            try e.disp32(sib.disp);
                        }
                    } else {
                        if (sib.disp == 0 and dst != 5) {
                            try e.modRm_indirectDisp0(src, dst);
                        } else if (math.cast(i8, sib.disp)) |_| {
                            try e.modRm_indirectDisp8(src, dst);
                            try e.disp8(@as(i8, @truncate(sib.disp)));
                        } else {
                            try e.modRm_indirectDisp32(src, dst);
                            try e.disp32(sib.disp);
                        }
                    }
                },
                .frame_index => if (@TypeOf(e).options.allow_frame_loc) {
                    try e.modRm_indirectDisp32(operand_enc, undefined);
                    try e.disp32(undefined);
                } else return error.CannotEncode,
            },
            .m_rip => |rip| {
                try e.modRm_RIPDisp32(operand_enc);
                try e.disp32(rip.disp);
            },
        }
    }

    fn encodeImm(imm: Immediate, kind: Op, e: anytype) !void {
        const raw = imm.asUnsigned(kind.immBitSize());
        switch (kind.immBitSize()) {
            8 => try e.imm8(@as(u8, @intCast(raw))),
            16 => try e.imm16(@as(u16, @intCast(raw))),
            32 => try e.imm32(@as(u32, @intCast(raw))),
            64 => try e.imm64(raw),
            else => unreachable,
        }
    }
};

pub const LegacyPrefixes = packed struct {
    /// LOCK
    prefix_f0: bool = false,
    /// REPNZ, REPNE, REP, Scalar Double-precision
    prefix_f2: bool = false,
    /// REPZ, REPE, REP, Scalar Single-precision
    prefix_f3: bool = false,

    /// CS segment override or Branch not taken
    prefix_2e: bool = false,
    /// SS segment override
    prefix_36: bool = false,
    /// ES segment override
    prefix_26: bool = false,
    /// FS segment override
    prefix_64: bool = false,
    /// GS segment override
    prefix_65: bool = false,

    /// Branch taken
    prefix_3e: bool = false,

    /// Address size override (enables 16 bit address size)
    prefix_67: bool = false,

    /// Operand size override (enables 16 bit operation)
    prefix_66: bool = false,

    padding: u5 = 0,

    pub fn setSegmentOverride(self: *LegacyPrefixes, reg: Register) void {
        assert(reg.class() == .segment);
        switch (reg) {
            .cs => self.prefix_2e = true,
            .ss => self.prefix_36 = true,
            .es => self.prefix_26 = true,
            .fs => self.prefix_64 = true,
            .gs => self.prefix_65 = true,
            .ds => {},
            else => unreachable,
        }
    }

    pub fn set16BitOverride(self: *LegacyPrefixes) void {
        self.prefix_66 = true;
    }
};

pub const EncoderOptions = struct { allow_frame_loc: bool = false };

fn Encoder(comptime T: type, comptime opts: EncoderOptions) type {
    return struct {
        writer: T,

        const Self = @This();
        pub const options = opts;

        // --------
        // Prefixes
        // --------

        /// Encodes legacy prefixes
        pub fn legacyPrefixes(self: Self, prefixes: LegacyPrefixes) !void {
            if (@as(u16, @bitCast(prefixes)) != 0) {
                // Hopefully this path isn't taken very often, so we'll do it the slow way for now

                // LOCK
                if (prefixes.prefix_f0) try self.writer.writeByte(0xf0);
                // REPNZ, REPNE, REP, Scalar Double-precision
                if (prefixes.prefix_f2) try self.writer.writeByte(0xf2);
                // REPZ, REPE, REP, Scalar Single-precision
                if (prefixes.prefix_f3) try self.writer.writeByte(0xf3);

                // CS segment override or Branch not taken
                if (prefixes.prefix_2e) try self.writer.writeByte(0x2e);
                // DS segment override
                if (prefixes.prefix_36) try self.writer.writeByte(0x36);
                // ES segment override
                if (prefixes.prefix_26) try self.writer.writeByte(0x26);
                // FS segment override
                if (prefixes.prefix_64) try self.writer.writeByte(0x64);
                // GS segment override
                if (prefixes.prefix_65) try self.writer.writeByte(0x65);

                // Branch taken
                if (prefixes.prefix_3e) try self.writer.writeByte(0x3e);

                // Operand size override
                if (prefixes.prefix_66) try self.writer.writeByte(0x66);

                // Address size override
                if (prefixes.prefix_67) try self.writer.writeByte(0x67);
            }
        }

        /// Use 16 bit operand size
        ///
        /// Note that this flag is overridden by REX.W, if both are present.
        pub fn prefix16BitMode(self: Self) !void {
            try self.writer.writeByte(0x66);
        }

        /// Encodes a REX prefix byte given all the fields
        ///
        /// Use this byte whenever you need 64 bit operation,
        /// or one of reg, index, r/m, base, or opcode-reg might be extended.
        ///
        /// See struct `Rex` for a description of each field.
        pub fn rex(self: Self, fields: Rex) !void {
            if (!fields.present and !fields.isSet()) return;

            var byte: u8 = 0b0100_0000;

            if (fields.w) byte |= 0b1000;
            if (fields.r) byte |= 0b0100;
            if (fields.x) byte |= 0b0010;
            if (fields.b) byte |= 0b0001;

            try self.writer.writeByte(byte);
        }

        /// Encodes a VEX prefix given all the fields
        ///
        /// See struct `Vex` for a description of each field.
        pub fn vex(self: Self, fields: Vex) !void {
            if (fields.is3Byte()) {
                try self.writer.writeByte(0b1100_0100);

                try self.writer.writeByte(
                    @as(u8, ~@intFromBool(fields.r)) << 7 |
                        @as(u8, ~@intFromBool(fields.x)) << 6 |
                        @as(u8, ~@intFromBool(fields.b)) << 5 |
                        @as(u8, @intFromEnum(fields.m)) << 0,
                );

                try self.writer.writeByte(
                    @as(u8, @intFromBool(fields.w)) << 7 |
                        @as(u8, ~fields.v.enc()) << 3 |
                        @as(u8, @intFromBool(fields.l)) << 2 |
                        @as(u8, @intFromEnum(fields.p)) << 0,
                );
            } else {
                try self.writer.writeByte(0b1100_0101);
                try self.writer.writeByte(
                    @as(u8, ~@intFromBool(fields.r)) << 7 |
                        @as(u8, ~fields.v.enc()) << 3 |
                        @as(u8, @intFromBool(fields.l)) << 2 |
                        @as(u8, @intFromEnum(fields.p)) << 0,
                );
            }
        }

        // ------
        // Opcode
        // ------

        /// Encodes a 1 byte opcode
        pub fn opcode_1byte(self: Self, opcode: u8) !void {
            try self.writer.writeByte(opcode);
        }

        /// Encodes a 2 byte opcode
        ///
        /// e.g. IMUL has the opcode 0x0f 0xaf, so you use
        ///
        /// encoder.opcode_2byte(0x0f, 0xaf);
        pub fn opcode_2byte(self: Self, prefix: u8, opcode: u8) !void {
            try self.writer.writeAll(&.{ prefix, opcode });
        }

        /// Encodes a 3 byte opcode
        ///
        /// e.g. MOVSD has the opcode 0xf2 0x0f 0x10
        ///
        /// encoder.opcode_3byte(0xf2, 0x0f, 0x10);
        pub fn opcode_3byte(self: Self, prefix_1: u8, prefix_2: u8, opcode: u8) !void {
            try self.writer.writeAll(&.{ prefix_1, prefix_2, opcode });
        }

        /// Encodes a 1 byte opcode with a reg field
        ///
        /// Remember to add a REX prefix byte if reg is extended!
        pub fn opcode_withReg(self: Self, opcode: u8, reg: u3) !void {
            assert(opcode & 0b111 == 0);
            try self.writer.writeByte(opcode | reg);
        }

        // ------
        // ModR/M
        // ------

        /// Construct a ModR/M byte given all the fields
        ///
        /// Remember to add a REX prefix byte if reg or rm are extended!
        pub fn modRm(self: Self, mod: u2, reg_or_opx: u3, rm: u3) !void {
            try self.writer.writeByte(@as(u8, mod) << 6 | @as(u8, reg_or_opx) << 3 | rm);
        }

        /// Construct a ModR/M byte using direct r/m addressing
        /// r/m effective address: r/m
        ///
        /// Note reg's effective address is always just reg for the ModR/M byte.
        /// Remember to add a REX prefix byte if reg or rm are extended!
        pub fn modRm_direct(self: Self, reg_or_opx: u3, rm: u3) !void {
            try self.modRm(0b11, reg_or_opx, rm);
        }

        /// Construct a ModR/M byte using indirect r/m addressing
        /// r/m effective address: [r/m]
        ///
        /// Note reg's effective address is always just reg for the ModR/M byte.
        /// Remember to add a REX prefix byte if reg or rm are extended!
        pub fn modRm_indirectDisp0(self: Self, reg_or_opx: u3, rm: u3) !void {
            assert(rm != 4 and rm != 5);
            try self.modRm(0b00, reg_or_opx, rm);
        }

        /// Construct a ModR/M byte using indirect SIB addressing
        /// r/m effective address: [SIB]
        ///
        /// Note reg's effective address is always just reg for the ModR/M byte.
        /// Remember to add a REX prefix byte if reg or rm are extended!
        pub fn modRm_SIBDisp0(self: Self, reg_or_opx: u3) !void {
            try self.modRm(0b00, reg_or_opx, 0b100);
        }

        /// Construct a ModR/M byte using RIP-relative addressing
        /// r/m effective address: [RIP + disp32]
        ///
        /// Note reg's effective address is always just reg for the ModR/M byte.
        /// Remember to add a REX prefix byte if reg or rm are extended!
        pub fn modRm_RIPDisp32(self: Self, reg_or_opx: u3) !void {
            try self.modRm(0b00, reg_or_opx, 0b101);
        }

        /// Construct a ModR/M byte using indirect r/m with a 8bit displacement
        /// r/m effective address: [r/m + disp8]
        ///
        /// Note reg's effective address is always just reg for the ModR/M byte.
        /// Remember to add a REX prefix byte if reg or rm are extended!
        pub fn modRm_indirectDisp8(self: Self, reg_or_opx: u3, rm: u3) !void {
            assert(rm != 4);
            try self.modRm(0b01, reg_or_opx, rm);
        }

        /// Construct a ModR/M byte using indirect SIB with a 8bit displacement
        /// r/m effective address: [SIB + disp8]
        ///
        /// Note reg's effective address is always just reg for the ModR/M byte.
        /// Remember to add a REX prefix byte if reg or rm are extended!
        pub fn modRm_SIBDisp8(self: Self, reg_or_opx: u3) !void {
            try self.modRm(0b01, reg_or_opx, 0b100);
        }

        /// Construct a ModR/M byte using indirect r/m with a 32bit displacement
        /// r/m effective address: [r/m + disp32]
        ///
        /// Note reg's effective address is always just reg for the ModR/M byte.
        /// Remember to add a REX prefix byte if reg or rm are extended!
        pub fn modRm_indirectDisp32(self: Self, reg_or_opx: u3, rm: u3) !void {
            assert(rm != 4);
            try self.modRm(0b10, reg_or_opx, rm);
        }

        /// Construct a ModR/M byte using indirect SIB with a 32bit displacement
        /// r/m effective address: [SIB + disp32]
        ///
        /// Note reg's effective address is always just reg for the ModR/M byte.
        /// Remember to add a REX prefix byte if reg or rm are extended!
        pub fn modRm_SIBDisp32(self: Self, reg_or_opx: u3) !void {
            try self.modRm(0b10, reg_or_opx, 0b100);
        }

        // ---
        // SIB
        // ---

        /// Construct a SIB byte given all the fields
        ///
        /// Remember to add a REX prefix byte if index or base are extended!
        pub fn sib(self: Self, scale: u2, index: u3, base: u3) !void {
            try self.writer.writeByte(@as(u8, scale) << 6 | @as(u8, index) << 3 | base);
        }

        /// Construct a SIB byte with scale * index + base, no frills.
        /// r/m effective address: [base + scale * index]
        ///
        /// Remember to add a REX prefix byte if index or base are extended!
        pub fn sib_scaleIndexBase(self: Self, scale: u2, index: u3, base: u3) !void {
            assert(base != 5);

            try self.sib(scale, index, base);
        }

        /// Construct a SIB byte with scale * index + disp32
        /// r/m effective address: [scale * index + disp32]
        ///
        /// Remember to add a REX prefix byte if index or base are extended!
        pub fn sib_scaleIndexDisp32(self: Self, scale: u2, index: u3) !void {
            // scale is actually ignored
            // index = 4 means no index if and only if we haven't extended the register
            // TODO enforce this
            // base = 5 means no base, if mod == 0.
            try self.sib(scale, index, 5);
        }

        /// Construct a SIB byte with just base
        /// r/m effective address: [base]
        ///
        /// Remember to add a REX prefix byte if index or base are extended!
        pub fn sib_base(self: Self, base: u3) !void {
            assert(base != 5);

            // scale is actually ignored
            // index = 4 means no index
            try self.sib(0, 4, base);
        }

        /// Construct a SIB byte with just disp32
        /// r/m effective address: [disp32]
        ///
        /// Remember to add a REX prefix byte if index or base are extended!
        pub fn sib_disp32(self: Self) !void {
            // scale is actually ignored
            // index = 4 means no index
            // base = 5 means no base, if mod == 0.
            try self.sib(0, 4, 5);
        }

        /// Construct a SIB byte with scale * index + base + disp8
        /// r/m effective address: [base + scale * index + disp8]
        ///
        /// Remember to add a REX prefix byte if index or base are extended!
        pub fn sib_scaleIndexBaseDisp8(self: Self, scale: u2, index: u3, base: u3) !void {
            try self.sib(scale, index, base);
        }

        /// Construct a SIB byte with base + disp8, no index
        /// r/m effective address: [base + disp8]
        ///
        /// Remember to add a REX prefix byte if index or base are extended!
        pub fn sib_baseDisp8(self: Self, base: u3) !void {
            // scale is ignored
            // index = 4 means no index
            try self.sib(0, 4, base);
        }

        /// Construct a SIB byte with scale * index + base + disp32
        /// r/m effective address: [base + scale * index + disp32]
        ///
        /// Remember to add a REX prefix byte if index or base are extended!
        pub fn sib_scaleIndexBaseDisp32(self: Self, scale: u2, index: u3, base: u3) !void {
            try self.sib(scale, index, base);
        }

        /// Construct a SIB byte with base + disp32, no index
        /// r/m effective address: [base + disp32]
        ///
        /// Remember to add a REX prefix byte if index or base are extended!
        pub fn sib_baseDisp32(self: Self, base: u3) !void {
            // scale is ignored
            // index = 4 means no index
            try self.sib(0, 4, base);
        }

        // -------------------------
        // Trivial (no bit fiddling)
        // -------------------------

        /// Encode an 8 bit displacement
        ///
        /// It is sign-extended to 64 bits by the cpu.
        pub fn disp8(self: Self, disp: i8) !void {
            try self.writer.writeByte(@as(u8, @bitCast(disp)));
        }

        /// Encode an 32 bit displacement
        ///
        /// It is sign-extended to 64 bits by the cpu.
        pub fn disp32(self: Self, disp: i32) !void {
            try self.writer.writeInt(i32, disp, .little);
        }

        /// Encode an 8 bit immediate
        ///
        /// It is sign-extended to 64 bits by the cpu.
        pub fn imm8(self: Self, imm: u8) !void {
            try self.writer.writeByte(imm);
        }

        /// Encode an 16 bit immediate
        ///
        /// It is sign-extended to 64 bits by the cpu.
        pub fn imm16(self: Self, imm: u16) !void {
            try self.writer.writeInt(u16, imm, .little);
        }

        /// Encode an 32 bit immediate
        ///
        /// It is sign-extended to 64 bits by the cpu.
        pub fn imm32(self: Self, imm: u32) !void {
            try self.writer.writeInt(u32, imm, .little);
        }

        /// Encode an 64 bit immediate
        ///
        /// It is sign-extended to 64 bits by the cpu.
        pub fn imm64(self: Self, imm: u64) !void {
            try self.writer.writeInt(u64, imm, .little);
        }
    };
}

pub const Rex = struct {
    w: bool = false,
    r: bool = false,
    x: bool = false,
    b: bool = false,
    present: bool = false,

    pub fn isSet(rex: Rex) bool {
        return rex.w or rex.r or rex.x or rex.b;
    }
};

pub const Vex = struct {
    w: bool = false,
    r: bool = false,
    x: bool = false,
    b: bool = false,
    l: bool = false,
    p: enum(u2) {
        none = 0b00,
        @"66" = 0b01,
        f3 = 0b10,
        f2 = 0b11,
    } = .none,
    m: enum(u5) {
        @"0f" = 0b0_0001,
        @"0f38" = 0b0_0010,
        @"0f3a" = 0b0_0011,
        _,
    } = .@"0f",
    v: Register = .ymm0,

    pub fn is3Byte(vex: Vex) bool {
        return vex.w or vex.x or vex.b or vex.m != .@"0f";
    }
};

pub const Assembler = struct {
    it: Tokenizer,

    const Tokenizer = struct {
        input: []const u8,
        pos: usize = 0,

        const Error = error{InvalidToken};

        const Token = struct {
            id: Id,
            start: usize,
            end: usize,

            const Id = enum {
                eof,

                space,
                new_line,

                colon,
                comma,
                open_br,
                close_br,
                plus,
                minus,
                star,

                string,
                numeral,
            };
        };

        const Iterator = struct {};

        fn next(it: *Tokenizer) !Token {
            var result = Token{
                .id = .eof,
                .start = it.pos,
                .end = it.pos,
            };

            var state: enum {
                start,
                space,
                new_line,
                string,
                numeral,
                numeral_hex,
            } = .start;

            while (it.pos < it.input.len) : (it.pos += 1) {
                const ch = it.input[it.pos];
                switch (state) {
                    .start => switch (ch) {
                        ',' => {
                            result.id = .comma;
                            it.pos += 1;
                            break;
                        },
                        ':' => {
                            result.id = .colon;
                            it.pos += 1;
                            break;
                        },
                        '[' => {
                            result.id = .open_br;
                            it.pos += 1;
                            break;
                        },
                        ']' => {
                            result.id = .close_br;
                            it.pos += 1;
                            break;
                        },
                        '+' => {
                            result.id = .plus;
                            it.pos += 1;
                            break;
                        },
                        '-' => {
                            result.id = .minus;
                            it.pos += 1;
                            break;
                        },
                        '*' => {
                            result.id = .star;
                            it.pos += 1;
                            break;
                        },
                        ' ', '\t' => state = .space,
                        '\n', '\r' => state = .new_line,
                        'a'...'z', 'A'...'Z' => state = .string,
                        '0'...'9' => state = .numeral,
                        else => return error.InvalidToken,
                    },

                    .space => switch (ch) {
                        ' ', '\t' => {},
                        else => {
                            result.id = .space;
                            break;
                        },
                    },

                    .new_line => switch (ch) {
                        '\n', '\r', ' ', '\t' => {},
                        else => {
                            result.id = .new_line;
                            break;
                        },
                    },

                    .string => switch (ch) {
                        'a'...'z', 'A'...'Z', '0'...'9' => {},
                        else => {
                            result.id = .string;
                            break;
                        },
                    },

                    .numeral => switch (ch) {
                        'x' => state = .numeral_hex,
                        '0'...'9' => {},
                        else => {
                            result.id = .numeral;
                            break;
                        },
                    },

                    .numeral_hex => switch (ch) {
                        'a'...'f' => {},
                        '0'...'9' => {},
                        else => {
                            result.id = .numeral;
                            break;
                        },
                    },
                }
            }

            if (it.pos >= it.input.len) {
                switch (state) {
                    .string => result.id = .string,
                    .numeral, .numeral_hex => result.id = .numeral,
                    else => {},
                }
            }

            result.end = it.pos;
            return result;
        }

        fn seekTo(it: *Tokenizer, pos: usize) void {
            it.pos = pos;
        }
    };

    pub fn init(input: []const u8) Assembler {
        return .{
            .it = Tokenizer{ .input = input },
        };
    }

    pub fn assemble(as: *Assembler, writer: anytype) !void {
        while (try as.next()) |parsed_inst| {
            const inst = try Instruction.new(.none, parsed_inst.mnemonic, &parsed_inst.ops);
            try inst.encode(writer, .{});
        }
    }

    const ParseResult = struct {
        mnemonic: Mnemonic,
        ops: [4]Operand,
    };

    const ParseError = error{
        UnexpectedToken,
        InvalidMnemonic,
        InvalidOperand,
        InvalidRegister,
        InvalidPtrSize,
        InvalidMemoryOperand,
        InvalidScaleIndex,
    } || Tokenizer.Error || std.fmt.ParseIntError;

    fn next(as: *Assembler) ParseError!?ParseResult {
        try as.skip(2, .{ .space, .new_line });
        const mnemonic_tok = as.expect(.string) catch |err| switch (err) {
            error.UnexpectedToken => return if (try as.peek() == .eof) null else err,
            else => return err,
        };
        const mnemonic = mnemonicFromString(as.source(mnemonic_tok)) orelse
            return error.InvalidMnemonic;
        try as.skip(1, .{.space});

        const rules = .{
            .{},
            .{.register},
            .{.memory},
            .{.immediate},
            .{ .register, .register },
            .{ .register, .memory },
            .{ .memory, .register },
            .{ .register, .immediate },
            .{ .memory, .immediate },
            .{ .register, .register, .immediate },
            .{ .register, .memory, .immediate },
        };

        const pos = as.it.pos;
        inline for (rules) |rule| {
            var ops = [4]Operand{ .none, .none, .none, .none };
            if (as.parseOperandRule(rule, &ops)) {
                return .{
                    .mnemonic = mnemonic,
                    .ops = ops,
                };
            } else |_| {
                as.it.seekTo(pos);
            }
        }

        return error.InvalidOperand;
    }

    fn source(as: *Assembler, token: Tokenizer.Token) []const u8 {
        return as.it.input[token.start..token.end];
    }

    fn peek(as: *Assembler) Tokenizer.Error!Tokenizer.Token.Id {
        const pos = as.it.pos;
        const next_tok = try as.it.next();
        const id = next_tok.id;
        as.it.seekTo(pos);
        return id;
    }

    fn expect(as: *Assembler, id: Tokenizer.Token.Id) ParseError!Tokenizer.Token {
        const next_tok_id = try as.peek();
        if (next_tok_id == id) return as.it.next();
        return error.UnexpectedToken;
    }

    fn skip(as: *Assembler, comptime num: comptime_int, tok_ids: [num]Tokenizer.Token.Id) Tokenizer.Error!void {
        outer: while (true) {
            const pos = as.it.pos;
            const next_tok = try as.it.next();
            inline for (tok_ids) |tok_id| {
                if (next_tok.id == tok_id) continue :outer;
            }
            as.it.seekTo(pos);
            break;
        }
    }

    fn mnemonicFromString(bytes: []const u8) ?Mnemonic {
        const ti = @typeInfo(Mnemonic).@"enum";
        inline for (ti.fields) |field| {
            if (std.mem.eql(u8, bytes, field.name)) {
                return @field(Mnemonic, field.name);
            }
        }
        return null;
    }

    fn parseOperandRule(as: *Assembler, rule: anytype, ops: *[4]Operand) ParseError!void {
        inline for (rule, 0..) |cond, i| {
            comptime assert(i < 4);
            if (i > 0) {
                _ = try as.expect(.comma);
                try as.skip(1, .{.space});
            }
            if (@typeInfo(@TypeOf(cond)) != .enum_literal) {
                @compileError("invalid condition in the rule: " ++ @typeName(@TypeOf(cond)));
            }
            switch (cond) {
                .register => {
                    const reg_tok = try as.expect(.string);
                    const reg = registerFromString(as.source(reg_tok)) orelse
                        return error.InvalidOperand;
                    ops[i] = .{ .reg = reg };
                },
                .memory => {
                    const mem = try as.parseMemory();
                    ops[i] = .{ .mem = mem };
                },
                .immediate => {
                    const is_neg = if (as.expect(.minus)) |_| true else |_| false;
                    const imm_tok = try as.expect(.numeral);
                    const imm: Immediate = if (is_neg) blk: {
                        const imm = try std.fmt.parseInt(i32, as.source(imm_tok), 0);
                        break :blk .{ .signed = imm * -1 };
                    } else .{ .unsigned = try std.fmt.parseInt(u64, as.source(imm_tok), 0) };
                    ops[i] = .{ .imm = imm };
                },
                else => @compileError("unhandled enum literal " ++ @tagName(cond)),
            }
            try as.skip(1, .{.space});
        }

        try as.skip(1, .{.space});
        const tok = try as.it.next();
        switch (tok.id) {
            .new_line, .eof => {},
            else => return error.InvalidOperand,
        }
    }

    fn registerFromString(bytes: []const u8) ?Register {
        const ti = @typeInfo(Register).@"enum";
        inline for (ti.fields) |field| {
            if (std.mem.eql(u8, bytes, field.name)) {
                return @field(Register, field.name);
            }
        }
        return null;
    }

    fn parseMemory(as: *Assembler) ParseError!Memory {
        const ptr_size: ?PtrSize = blk: {
            const pos = as.it.pos;
            const ptr_size = as.parsePtrSize() catch |err| switch (err) {
                error.UnexpectedToken => {
                    as.it.seekTo(pos);
                    break :blk null;
                },
                else => return err,
            };
            break :blk ptr_size;
        };

        try as.skip(1, .{.space});

        // Supported rules and orderings.
        const rules = .{
            .{ .open_br, .base, .close_br }, // [ base ]
            .{ .open_br, .base, .plus, .disp, .close_br }, // [ base + disp ]
            .{ .open_br, .base, .minus, .disp, .close_br }, // [ base - disp ]
            .{ .open_br, .disp, .plus, .base, .close_br }, // [ disp + base ]
            .{ .open_br, .base, .plus, .index, .close_br }, // [ base + index ]
            .{ .open_br, .base, .plus, .index, .star, .scale, .close_br }, // [ base + index * scale ]
            .{ .open_br, .index, .star, .scale, .plus, .base, .close_br }, // [ index * scale + base ]
            .{ .open_br, .base, .plus, .index, .star, .scale, .plus, .disp, .close_br }, // [ base + index * scale + disp ]
            .{ .open_br, .base, .plus, .index, .star, .scale, .minus, .disp, .close_br }, // [ base + index * scale - disp ]
            .{ .open_br, .index, .star, .scale, .plus, .base, .plus, .disp, .close_br }, // [ index * scale + base + disp ]
            .{ .open_br, .index, .star, .scale, .plus, .base, .minus, .disp, .close_br }, // [ index * scale + base - disp ]
            .{ .open_br, .disp, .plus, .index, .star, .scale, .plus, .base, .close_br }, // [ disp + index * scale + base ]
            .{ .open_br, .disp, .plus, .base, .plus, .index, .star, .scale, .close_br }, // [ disp + base + index * scale ]
            .{ .open_br, .base, .plus, .disp, .plus, .index, .star, .scale, .close_br }, // [ base + disp + index * scale ]
            .{ .open_br, .base, .minus, .disp, .plus, .index, .star, .scale, .close_br }, // [ base - disp + index * scale ]
            .{ .open_br, .base, .plus, .disp, .plus, .scale, .star, .index, .close_br }, // [ base + disp + scale * index ]
            .{ .open_br, .base, .minus, .disp, .plus, .scale, .star, .index, .close_br }, // [ base - disp + scale * index ]
            .{ .open_br, .rip, .plus, .disp, .close_br }, // [ rip + disp ]
            .{ .open_br, .rip, .minus, .disp, .close_br }, // [ rig - disp ]
            .{ .base, .colon, .disp }, // seg:disp
        };

        const pos = as.it.pos;
        inline for (rules) |rule| {
            if (as.parseMemoryRule(rule)) |res| {
                if (res.rip) {
                    if (res.base != .none or res.scale_index != null or res.offset != null)
                        return error.InvalidMemoryOperand;
                    return Memory.rip(ptr_size orelse .qword, res.disp orelse 0);
                }
                switch (res.base) {
                    .none => {},
                    .reg => |base| {
                        if (res.rip)
                            return error.InvalidMemoryOperand;
                        if (res.offset) |offset| {
                            if (res.scale_index != null or res.disp != null)
                                return error.InvalidMemoryOperand;
                            return Memory.moffs(base, offset);
                        }
                        return Memory.sib(ptr_size orelse .qword, .{
                            .base = .{ .reg = base },
                            .scale_index = res.scale_index,
                            .disp = res.disp orelse 0,
                        });
                    },
                    .frame_index => unreachable,
                }
                return error.InvalidMemoryOperand;
            } else |_| {
                as.it.seekTo(pos);
            }
        }

        return error.InvalidOperand;
    }

    const MemoryParseResult = struct {
        rip: bool = false,
        base: Memory.Base = .none,
        scale_index: ?Memory.ScaleIndex = null,
        disp: ?i32 = null,
        offset: ?u64 = null,
    };

    fn parseMemoryRule(as: *Assembler, rule: anytype) ParseError!MemoryParseResult {
        var res: MemoryParseResult = .{};
        inline for (rule, 0..) |cond, i| {
            if (@typeInfo(@TypeOf(cond)) != .enum_literal) {
                @compileError("unsupported condition type in the rule: " ++ @typeName(@TypeOf(cond)));
            }
            switch (cond) {
                .open_br, .close_br, .plus, .minus, .star, .colon => {
                    _ = try as.expect(cond);
                },
                .base => {
                    const tok = try as.expect(.string);
                    res.base = .{ .reg = registerFromString(as.source(tok)) orelse return error.InvalidMemoryOperand };
                },
                .rip => {
                    const tok = try as.expect(.string);
                    if (!std.mem.eql(u8, as.source(tok), "rip")) return error.InvalidMemoryOperand;
                    res.rip = true;
                },
                .index => {
                    const tok = try as.expect(.string);
                    const index = registerFromString(as.source(tok)) orelse
                        return error.InvalidMemoryOperand;
                    if (res.scale_index) |*si| {
                        si.index = index;
                    } else {
                        res.scale_index = .{ .scale = 1, .index = index };
                    }
                },
                .scale => {
                    const tok = try as.expect(.numeral);
                    const scale = try std.fmt.parseInt(u2, as.source(tok), 0);
                    if (res.scale_index) |*si| {
                        si.scale = scale;
                    } else {
                        res.scale_index = .{ .scale = scale, .index = undefined };
                    }
                },
                .disp => {
                    const tok = try as.expect(.numeral);
                    const is_neg = blk: {
                        if (i > 0) {
                            if (rule[i - 1] == .minus) break :blk true;
                        }
                        break :blk false;
                    };
                    if (std.fmt.parseInt(i32, as.source(tok), 0)) |disp| {
                        res.disp = if (is_neg) -1 * disp else disp;
                    } else |err| switch (err) {
                        error.Overflow => {
                            if (is_neg) return err;
                            switch (res.base) {
                                .none => {},
                                .reg => |base| if (base.class() != .segment) return err,
                                .frame_index => unreachable,
                            }
                            const offset = try std.fmt.parseInt(u64, as.source(tok), 0);
                            res.offset = offset;
                        },
                        else => return err,
                    }
                },
                else => @compileError("unhandled operand output type: " ++ @tagName(cond)),
            }
            try as.skip(1, .{.space});
        }
        return res;
    }

    fn parsePtrSize(as: *Assembler) ParseError!PtrSize {
        const size = try as.expect(.string);
        try as.skip(1, .{.space});
        const ptr = try as.expect(.string);

        const size_raw = as.source(size);
        const ptr_raw = as.source(ptr);
        const len = size_raw.len + ptr_raw.len + 1;
        var buf: ["qword ptr".len]u8 = undefined;
        if (len > buf.len) return error.InvalidPtrSize;

        for (size_raw, 0..) |c, i| {
            buf[i] = std.ascii.toLower(c);
        }
        buf[size_raw.len] = ' ';
        for (ptr_raw, 0..) |c, i| {
            buf[size_raw.len + i + 1] = std.ascii.toLower(c);
        }

        const slice = buf[0..len];
        if (std.mem.eql(u8, slice, "qword ptr")) return .qword;
        if (std.mem.eql(u8, slice, "dword ptr")) return .dword;
        if (std.mem.eql(u8, slice, "word ptr")) return .word;
        if (std.mem.eql(u8, slice, "byte ptr")) return .byte;
        if (std.mem.eql(u8, slice, "tbyte ptr")) return .tbyte;
        return error.InvalidPtrSize;
    }
};

pub const Disassembler = struct {
    pub const Error = error{
        EndOfStream,
        LegacyPrefixAfterRex,
        UnknownOpcode,
        Todo,
    };

    code: []const u8,
    pos: usize = 0,

    pub fn init(code: []const u8) Disassembler {
        return .{ .code = code };
    }

    pub fn next(dis: *Disassembler) Error!?Instruction {
        const prefixes = dis.parsePrefixes() catch |err| switch (err) {
            error.EndOfStream => return null,
            else => |e| return e,
        };

        const enc = try dis.parseEncoding(prefixes) orelse return error.UnknownOpcode;
        switch (enc.data.op_en) {
            .np => return inst(enc, .{}),
            .d, .i => {
                const imm = try dis.parseImm(enc.data.ops[0]);
                return inst(enc, .{
                    .op1 = .{ .imm = imm },
                });
            },
            .zi => {
                const imm = try dis.parseImm(enc.data.ops[1]);
                return inst(enc, .{
                    .op1 = .{ .reg = Register.rax.toBitSize(enc.data.ops[0].regBitSize()) },
                    .op2 = .{ .imm = imm },
                });
            },
            .o, .oi => {
                const reg_low_enc = @as(u3, @truncate(dis.code[dis.pos - 1]));
                const op2: Operand = if (enc.data.op_en == .oi) .{
                    .imm = try dis.parseImm(enc.data.ops[1]),
                } else .none;
                return inst(enc, .{
                    .op1 = .{ .reg = parseGpRegister(reg_low_enc, prefixes.rex.b, prefixes.rex, enc.data.ops[0].regBitSize()) },
                    .op2 = op2,
                });
            },
            .m, .mi, .m1, .mc => {
                const modrm = try dis.parseModRmByte();
                const act_enc = Encoding.findByOpcode(enc.opcode(), .{
                    .legacy = prefixes.legacy,
                    .rex = prefixes.rex,
                }, modrm.op1) orelse return error.UnknownOpcode;
                const sib = if (modrm.sib()) try dis.parseSibByte() else null;

                if (modrm.direct()) {
                    const op2: Operand = switch (act_enc.data.op_en) {
                        .mi => .{ .imm = try dis.parseImm(act_enc.data.ops[1]) },
                        .m1 => .{ .imm = Immediate.u(1) },
                        .mc => .{ .reg = .cl },
                        .m => .none,
                        else => unreachable,
                    };
                    return inst(act_enc, .{
                        .op1 = .{ .reg = parseGpRegister(modrm.op2, prefixes.rex.b, prefixes.rex, act_enc.data.ops[0].regBitSize()) },
                        .op2 = op2,
                    });
                }

                const disp = try dis.parseDisplacement(modrm, sib);
                const op2: Operand = switch (act_enc.data.op_en) {
                    .mi => .{ .imm = try dis.parseImm(act_enc.data.ops[1]) },
                    .m1 => .{ .imm = Immediate.u(1) },
                    .mc => .{ .reg = .cl },
                    .m => .none,
                    else => unreachable,
                };

                if (modrm.rip()) {
                    return inst(act_enc, .{
                        .op1 = .{ .mem = Memory.rip(PtrSize.fromBitSize(act_enc.data.ops[0].memBitSize()), disp) },
                        .op2 = op2,
                    });
                }

                const scale_index = if (sib) |info| info.scaleIndex(prefixes.rex) else null;
                const base = if (sib) |info|
                    info.baseReg(modrm, prefixes)
                else
                    parseGpRegister(modrm.op2, prefixes.rex.b, prefixes.rex, 64);
                return inst(act_enc, .{
                    .op1 = .{ .mem = Memory.sib(PtrSize.fromBitSize(act_enc.data.ops[0].memBitSize()), .{
                        .base = if (base) |base_reg| .{ .reg = base_reg } else .none,
                        .scale_index = scale_index,
                        .disp = disp,
                    }) },
                    .op2 = op2,
                });
            },
            .fd => {
                const seg = segmentRegister(prefixes.legacy);
                const offset = try dis.parseOffset();
                return inst(enc, .{
                    .op1 = .{ .reg = Register.rax.toBitSize(enc.data.ops[0].regBitSize()) },
                    .op2 = .{ .mem = Memory.moffs(seg, offset) },
                });
            },
            .td => {
                const seg = segmentRegister(prefixes.legacy);
                const offset = try dis.parseOffset();
                return inst(enc, .{
                    .op1 = .{ .mem = Memory.moffs(seg, offset) },
                    .op2 = .{ .reg = Register.rax.toBitSize(enc.data.ops[1].regBitSize()) },
                });
            },
            .mr, .mri, .mrc => {
                const modrm = try dis.parseModRmByte();
                const sib = if (modrm.sib()) try dis.parseSibByte() else null;
                const src_bit_size = enc.data.ops[1].regBitSize();

                if (modrm.direct()) {
                    return inst(enc, .{
                        .op1 = .{ .reg = parseGpRegister(modrm.op2, prefixes.rex.b, prefixes.rex, enc.data.ops[0].regBitSize()) },
                        .op2 = .{ .reg = parseGpRegister(modrm.op1, prefixes.rex.x, prefixes.rex, src_bit_size) },
                    });
                }

                const dst_bit_size = enc.data.ops[0].memBitSize();
                const disp = try dis.parseDisplacement(modrm, sib);
                const op3: Operand = switch (enc.data.op_en) {
                    .mri => .{ .imm = try dis.parseImm(enc.data.ops[2]) },
                    .mrc => .{ .reg = .cl },
                    .mr => .none,
                    else => unreachable,
                };

                if (modrm.rip()) {
                    return inst(enc, .{
                        .op1 = .{ .mem = Memory.rip(PtrSize.fromBitSize(dst_bit_size), disp) },
                        .op2 = .{ .reg = parseGpRegister(modrm.op1, prefixes.rex.r, prefixes.rex, src_bit_size) },
                        .op3 = op3,
                    });
                }

                const scale_index = if (sib) |info| info.scaleIndex(prefixes.rex) else null;
                const base = if (sib) |info|
                    info.baseReg(modrm, prefixes)
                else
                    parseGpRegister(modrm.op2, prefixes.rex.b, prefixes.rex, 64);
                return inst(enc, .{
                    .op1 = .{ .mem = Memory.sib(PtrSize.fromBitSize(dst_bit_size), .{
                        .base = if (base) |base_reg| .{ .reg = base_reg } else .none,
                        .scale_index = scale_index,
                        .disp = disp,
                    }) },
                    .op2 = .{ .reg = parseGpRegister(modrm.op1, prefixes.rex.r, prefixes.rex, src_bit_size) },
                    .op3 = op3,
                });
            },
            .rm, .rmi => {
                const modrm = try dis.parseModRmByte();
                const sib = if (modrm.sib()) try dis.parseSibByte() else null;
                const dst_bit_size = enc.data.ops[0].regBitSize();

                if (modrm.direct()) {
                    const op3: Operand = switch (enc.data.op_en) {
                        .rm => .none,
                        .rmi => .{ .imm = try dis.parseImm(enc.data.ops[2]) },
                        else => unreachable,
                    };
                    return inst(enc, .{
                        .op1 = .{ .reg = parseGpRegister(modrm.op1, prefixes.rex.x, prefixes.rex, dst_bit_size) },
                        .op2 = .{ .reg = parseGpRegister(modrm.op2, prefixes.rex.b, prefixes.rex, enc.data.ops[1].regBitSize()) },
                        .op3 = op3,
                    });
                }

                const src_bit_size = if (enc.data.ops[1] == .m) dst_bit_size else enc.data.ops[1].memBitSize();
                const disp = try dis.parseDisplacement(modrm, sib);
                const op3: Operand = switch (enc.data.op_en) {
                    .rmi => .{ .imm = try dis.parseImm(enc.data.ops[2]) },
                    .rm => .none,
                    else => unreachable,
                };

                if (modrm.rip()) {
                    return inst(enc, .{
                        .op1 = .{ .reg = parseGpRegister(modrm.op1, prefixes.rex.r, prefixes.rex, dst_bit_size) },
                        .op2 = .{ .mem = Memory.rip(PtrSize.fromBitSize(src_bit_size), disp) },
                        .op3 = op3,
                    });
                }

                const scale_index = if (sib) |info| info.scaleIndex(prefixes.rex) else null;
                const base = if (sib) |info|
                    info.baseReg(modrm, prefixes)
                else
                    parseGpRegister(modrm.op2, prefixes.rex.b, prefixes.rex, 64);
                return inst(enc, .{
                    .op1 = .{ .reg = parseGpRegister(modrm.op1, prefixes.rex.r, prefixes.rex, dst_bit_size) },
                    .op2 = .{ .mem = Memory.sib(PtrSize.fromBitSize(src_bit_size), .{
                        .base = if (base) |base_reg| .{ .reg = base_reg } else .none,
                        .scale_index = scale_index,
                        .disp = disp,
                    }) },
                    .op3 = op3,
                });
            },
            .rm0, .vmi, .rvm, .rvmr, .rvmi, .mvr => unreachable, // TODO
        }
    }

    fn inst(encoding: Encoding, args: struct {
        prefix: Prefix = .none,
        op1: Operand = .none,
        op2: Operand = .none,
        op3: Operand = .none,
        op4: Operand = .none,
    }) Instruction {
        const i = Instruction{ .encoding = encoding, .prefix = args.prefix, .ops = .{
            args.op1,
            args.op2,
            args.op3,
            args.op4,
        } };
        return i;
    }

    const Prefixes = struct {
        legacy: LegacyPrefixes = .{},
        rex: Rex = .{},
        // TODO add support for VEX prefix
    };

    fn parsePrefixes(dis: *Disassembler) !Prefixes {
        const rex_prefix_mask: u4 = 0b0100;
        var stream = std.io.fixedBufferStream(dis.code[dis.pos..]);
        const reader = stream.reader();

        var res: Prefixes = .{};

        while (true) {
            const next_byte = try reader.readByte();
            dis.pos += 1;

            switch (next_byte) {
                0xf0, 0xf2, 0xf3, 0x2e, 0x36, 0x26, 0x64, 0x65, 0x3e, 0x66, 0x67 => {
                    // Legacy prefix
                    if (res.rex.present) return error.LegacyPrefixAfterRex;
                    switch (next_byte) {
                        0xf0 => res.legacy.prefix_f0 = true,
                        0xf2 => res.legacy.prefix_f2 = true,
                        0xf3 => res.legacy.prefix_f3 = true,
                        0x2e => res.legacy.prefix_2e = true,
                        0x36 => res.legacy.prefix_36 = true,
                        0x26 => res.legacy.prefix_26 = true,
                        0x64 => res.legacy.prefix_64 = true,
                        0x65 => res.legacy.prefix_65 = true,
                        0x3e => res.legacy.prefix_3e = true,
                        0x66 => res.legacy.prefix_66 = true,
                        0x67 => res.legacy.prefix_67 = true,
                        else => unreachable,
                    }
                },
                else => {
                    if (rex_prefix_mask == @as(u4, @truncate(next_byte >> 4))) {
                        // REX prefix
                        res.rex.w = next_byte & 0b1000 != 0;
                        res.rex.r = next_byte & 0b100 != 0;
                        res.rex.x = next_byte & 0b10 != 0;
                        res.rex.b = next_byte & 0b1 != 0;
                        res.rex.present = true;
                        continue;
                    }

                    // TODO VEX prefix

                    dis.pos -= 1;
                    break;
                },
            }
        }

        return res;
    }

    fn parseEncoding(dis: *Disassembler, prefixes: Prefixes) !?Encoding {
        const o_mask: u8 = 0b1111_1000;

        var opcode: [3]u8 = .{ 0, 0, 0 };
        var stream = std.io.fixedBufferStream(dis.code[dis.pos..]);
        const reader = stream.reader();

        comptime var opc_count = 0;
        inline while (opc_count < 3) : (opc_count += 1) {
            const byte = try reader.readByte();
            opcode[opc_count] = byte;
            dis.pos += 1;

            if (byte == 0x0f) {
                // Multi-byte opcode
            } else if (opc_count > 0) {
                // Multi-byte opcode
                if (Encoding.findByOpcode(opcode[0 .. opc_count + 1], .{
                    .legacy = prefixes.legacy,
                    .rex = prefixes.rex,
                }, null)) |mnemonic| {
                    return mnemonic;
                }
            } else {
                // Single-byte opcode
                if (Encoding.findByOpcode(opcode[0..1], .{
                    .legacy = prefixes.legacy,
                    .rex = prefixes.rex,
                }, null)) |mnemonic| {
                    return mnemonic;
                } else {
                    // Try O* encoding
                    return Encoding.findByOpcode(&.{opcode[0] & o_mask}, .{
                        .legacy = prefixes.legacy,
                        .rex = prefixes.rex,
                    }, null);
                }
            }
        }
        return null;
    }

    fn parseGpRegister(low_enc: u3, is_extended: bool, rex: Rex, bit_size: u64) Register {
        const reg_id: u4 = @as(u4, @intCast(@intFromBool(is_extended))) << 3 | low_enc;
        const reg = @as(Register, @enumFromInt(reg_id)).toBitSize(bit_size);
        return switch (reg) {
            .spl => if (rex.present or rex.isSet()) .spl else .ah,
            .dil => if (rex.present or rex.isSet()) .dil else .bh,
            .bpl => if (rex.present or rex.isSet()) .bpl else .ch,
            .sil => if (rex.present or rex.isSet()) .sil else .dh,
            else => reg,
        };
    }

    fn parseImm(dis: *Disassembler, kind: Op) !Immediate {
        var stream = std.io.fixedBufferStream(dis.code[dis.pos..]);
        var creader = std.io.countingReader(stream.reader());
        const reader = creader.reader();
        const imm = switch (kind) {
            .imm8s, .rel8 => Immediate.s(try reader.readInt(i8, .little)),
            .imm16s, .rel16 => Immediate.s(try reader.readInt(i16, .little)),
            .imm32s, .rel32 => Immediate.s(try reader.readInt(i32, .little)),
            .imm8 => Immediate.u(try reader.readInt(u8, .little)),
            .imm16 => Immediate.u(try reader.readInt(u16, .little)),
            .imm32 => Immediate.u(try reader.readInt(u32, .little)),
            .imm64 => Immediate.u(try reader.readInt(u64, .little)),
            else => unreachable,
        };
        dis.pos += creader.bytes_read;
        return imm;
    }

    fn parseOffset(dis: *Disassembler) !u64 {
        var stream = std.io.fixedBufferStream(dis.code[dis.pos..]);
        const reader = stream.reader();
        const offset = try reader.readInt(u64, .little);
        dis.pos += 8;
        return offset;
    }

    const ModRm = packed struct {
        mod: u2,
        op1: u3,
        op2: u3,

        inline fn direct(self: ModRm) bool {
            return self.mod == 0b11;
        }

        inline fn rip(self: ModRm) bool {
            return self.mod == 0 and self.op2 == 0b101;
        }

        inline fn sib(self: ModRm) bool {
            return !self.direct() and self.op2 == 0b100;
        }
    };

    fn parseModRmByte(dis: *Disassembler) !ModRm {
        if (dis.code[dis.pos..].len == 0) return error.EndOfStream;
        const modrm_byte = dis.code[dis.pos];
        dis.pos += 1;
        const mod: u2 = @as(u2, @truncate(modrm_byte >> 6));
        const op1: u3 = @as(u3, @truncate(modrm_byte >> 3));
        const op2: u3 = @as(u3, @truncate(modrm_byte));
        return ModRm{ .mod = mod, .op1 = op1, .op2 = op2 };
    }

    fn segmentRegister(prefixes: LegacyPrefixes) Register {
        if (prefixes.prefix_2e) return .cs;
        if (prefixes.prefix_36) return .ss;
        if (prefixes.prefix_26) return .es;
        if (prefixes.prefix_64) return .fs;
        if (prefixes.prefix_65) return .gs;
        return .ds;
    }

    const Sib = packed struct {
        scale: u2,
        index: u3,
        base: u3,

        fn scaleIndex(self: Disassembler.Sib, rex: Rex) ?ScaleIndex {
            if (self.index == 0b100 and !rex.x) return null;
            return .{
                .scale = @as(u4, 1) << self.scale,
                .index = parseGpRegister(self.index, rex.x, rex, 64),
            };
        }

        fn baseReg(self: Disassembler.Sib, modrm: ModRm, prefixes: Prefixes) ?Register {
            if (self.base == 0b101 and modrm.mod == 0) {
                if (self.scaleIndex(prefixes.rex)) |_| return null;
                return segmentRegister(prefixes.legacy);
            }
            return parseGpRegister(self.base, prefixes.rex.b, prefixes.rex, 64);
        }
    };

    fn parseSibByte(dis: *Disassembler) !Disassembler.Sib {
        if (dis.code[dis.pos..].len == 0) return error.EndOfStream;
        const sib_byte = dis.code[dis.pos];
        dis.pos += 1;
        const scale: u2 = @as(u2, @truncate(sib_byte >> 6));
        const index: u3 = @as(u3, @truncate(sib_byte >> 3));
        const base: u3 = @as(u3, @truncate(sib_byte));
        return Disassembler.Sib{ .scale = scale, .index = index, .base = base };
    }

    fn parseDisplacement(dis: *Disassembler, modrm: ModRm, sib: ?Disassembler.Sib) !i32 {
        var stream = std.io.fixedBufferStream(dis.code[dis.pos..]);
        var creader = std.io.countingReader(stream.reader());
        const reader = creader.reader();
        const disp = disp: {
            if (sib) |info| {
                if (info.base == 0b101 and modrm.mod == 0) {
                    break :disp try reader.readInt(i32, .little);
                }
            }
            if (modrm.rip()) {
                break :disp try reader.readInt(i32, .little);
            }
            break :disp switch (modrm.mod) {
                0b00 => 0,
                0b01 => try reader.readInt(i8, .little),
                0b10 => try reader.readInt(i32, .little),
                0b11 => unreachable,
            };
        };
        dis.pos += creader.bytes_read;
        return disp;
    }
};

pub const Mnemonic = enum {
    // zig fmt: off
        // General-purpose
        adc, add, @"and",
        bsf, bsr, bswap, bt, btc, btr, bts,
        call, cbw, cdq, cdqe,
        cmova, cmovae, cmovb, cmovbe, cmovc, cmove, cmovg, cmovge, cmovl, cmovle, cmovna,
        cmovnae, cmovnb, cmovnbe, cmovnc, cmovne, cmovng, cmovnge, cmovnl, cmovnle, cmovno,
        cmovnp, cmovns, cmovnz, cmovo, cmovp, cmovpe, cmovpo, cmovs, cmovz,
        cmp,
        cmps, cmpsb, cmpsd, cmpsq, cmpsw,
        cmpxchg, cmpxchg8b, cmpxchg16b,
        cqo, cwd, cwde,
        div,
        endbr64,
        hlt,
        idiv, imul, int3,
        ja, jae, jb, jbe, jc, jrcxz, je, jg, jge, jl, jle, jna, jnae, jnb, jnbe,
        jnc, jne, jng, jnge, jnl, jnle, jno, jnp, jns, jnz, jo, jp, jpe, jpo, js, jz,
        jmp,
        lea, leave, lfence,
        lods, lodsb, lodsd, lodsq, lodsw,
        lzcnt,
        mfence, mov, movbe,
        movs, movsb, movsd, movsq, movsw,
        movsx, movsxd, movzx, mul,
        neg, nop, not,
        @"or",
        pop, popcnt, push,
        rcl, rcr, ret, rol, ror,
        sal, sar, sbb,
        scas, scasb, scasd, scasq, scasw,
        shl, shld, shr, shrd, sub, syscall,
        seta, setae, setb, setbe, setc, sete, setg, setge, setl, setle, setna, setnae,
        setnb, setnbe, setnc, setne, setng, setnge, setnl, setnle, setno, setnp, setns,
        setnz, seto, setp, setpe, setpo, sets, setz,
        sfence,
        stos, stosb, stosd, stosq, stosw,
        @"test", tzcnt,
        ud2,
        xadd, xchg, xor,
        // X87
        fisttp, fld,
        // MMX
        movd, movq,
        packssdw, packsswb, packuswb,
        paddb, paddd, paddq, paddsb, paddsw, paddusb, paddusw, paddw,
        pand, pandn, por, pxor,
        pmulhw, pmullw,
        psubb, psubd, psubq, psubsb, psubsw, psubusb, psubusw, psubw,
        // SSE
        addps, addss,
        andps,
        andnps,
        cmpps, cmpss,
        cvtpi2ps, cvtps2pi, cvtsi2ss, cvtss2si, cvttps2pi, cvttss2si,
        divps, divss,
        maxps, maxss,
        minps, minss,
        movaps, movhlps, movlhps,
        movss, movups,
        mulps, mulss,
        orps,
        pextrw, pinsrw,
        pmaxsw, pmaxub, pminsw, pminub,
        shufps,
        sqrtps, sqrtss,
        subps, subss,
        ucomiss,
        xorps,
        // SSE2
        addpd, addsd,
        andpd,
        andnpd,
        cmppd, //cmpsd,
        cvtdq2pd, cvtdq2ps, cvtpd2dq, cvtpd2pi, cvtpd2ps, cvtpi2pd,
        cvtps2dq, cvtps2pd, cvtsd2si, cvtsd2ss, cvtsi2sd, cvtss2sd,
        cvttpd2dq, cvttpd2pi, cvttps2dq, cvttsd2si,
        divpd, divsd,
        maxpd, maxsd,
        minpd, minsd,
        movapd,
        movdqa, movdqu,
        //movsd,
        movupd,
        mulpd, mulsd,
        orpd,
        pshufhw, pshuflw,
        psrld, psrlq, psrlw,
        punpckhbw, punpckhdq, punpckhqdq, punpckhwd,
        punpcklbw, punpckldq, punpcklqdq, punpcklwd,
        shufpd,
        sqrtpd, sqrtsd,
        subpd, subsd,
        ucomisd,
        xorpd,
        // SSE3
        movddup, movshdup, movsldup,
        // SSE4.1
        blendpd, blendps, blendvpd, blendvps,
        extractps,
        insertps,
        packusdw,
        pextrb, pextrd, pextrq,
        pinsrb, pinsrd, pinsrq,
        pmaxsb, pmaxsd, pmaxud, pmaxuw, pminsb, pminsd, pminud, pminuw,
        pmulld,
        roundpd, roundps, roundsd, roundss,
        // AVX
        vaddpd, vaddps, vaddsd, vaddss,
        vandnpd, vandnps, vandpd, vandps,
        vblendpd, vblendps, vblendvpd, vblendvps,
        vbroadcastf128, vbroadcastsd, vbroadcastss,
        vcmppd, vcmpps, vcmpsd, vcmpss,
        vcvtdq2pd, vcvtdq2ps, vcvtpd2dq, vcvtpd2ps,
        vcvtps2dq, vcvtps2pd, vcvtsd2si, vcvtsd2ss,
        vcvtsi2sd, vcvtsi2ss, vcvtss2sd, vcvtss2si,
        vcvttpd2dq, vcvttps2dq, vcvttsd2si, vcvttss2si,
        vdivpd, vdivps, vdivsd, vdivss,
        vextractf128, vextractps,
        vinsertf128, vinsertps,
        vmaxpd, vmaxps, vmaxsd, vmaxss,
        vminpd, vminps, vminsd, vminss,
        vmovapd, vmovaps,
        vmovd,
        vmovddup,
        vmovdqa, vmovdqu,
        vmovhlps, vmovlhps,
        vmovq,
        vmovsd,
        vmovshdup, vmovsldup,
        vmovss,
        vmovupd, vmovups,
        vmulpd, vmulps, vmulsd, vmulss,
        vorpd, vorps,
        vpackssdw, vpacksswb, vpackusdw, vpackuswb,
        vpaddb, vpaddd, vpaddq, vpaddsb, vpaddsw, vpaddusb, vpaddusw, vpaddw,
        vpand, vpandn,
        vpextrb, vpextrd, vpextrq, vpextrw,
        vpinsrb, vpinsrd, vpinsrq, vpinsrw,
        vpmaxsb, vpmaxsd, vpmaxsw, vpmaxub, vpmaxud, vpmaxuw,
        vpminsb, vpminsd, vpminsw, vpminub, vpminud, vpminuw,
        vpmulhw, vpmulld, vpmullw,
        vpor,
        vpshufhw, vpshuflw,
        vpsrld, vpsrlq, vpsrlw,
        vpsubb, vpsubd, vpsubq, vpsubsb, vpsubsw, vpsubusb, vpsubusw, vpsubw,
        vpunpckhbw, vpunpckhdq, vpunpckhqdq, vpunpckhwd,
        vpunpcklbw, vpunpckldq, vpunpcklqdq, vpunpcklwd,
        vpxor,
        vroundpd, vroundps, vroundsd, vroundss,
        vshufpd, vshufps,
        vsqrtpd, vsqrtps, vsqrtsd, vsqrtss,
        vsubpd, vsubps, vsubsd, vsubss,
        vxorpd, vxorps,
        // F16C
        vcvtph2ps, vcvtps2ph,
        // FMA
        vfmadd132pd, vfmadd213pd, vfmadd231pd,
        vfmadd132ps, vfmadd213ps, vfmadd231ps,
        vfmadd132sd, vfmadd213sd, vfmadd231sd,
        vfmadd132ss, vfmadd213ss, vfmadd231ss,
        // zig fmt: on
};

pub const OpEn = enum {
    // zig fmt: off
        np,
        o, oi,
        i, zi,
        d, m,
        fd, td,
        m1, mc, mi, mr, rm,
        rmi, mri, mrc,
        rm0, vmi, rvm, rvmr, rvmi, mvr,
        // zig fmt: on
};

pub const Op = enum {
    // zig fmt: off
        none,
        o16, o32, o64,
        unity,
        imm8, imm16, imm32, imm64,
        imm8s, imm16s, imm32s,
        al, ax, eax, rax,
        cl,
        r8, r16, r32, r64,
        rm8, rm16, rm32, rm64,
        r32_m8, r32_m16, r64_m16,
        m8, m16, m32, m64, m80, m128, m256,
        rel8, rel16, rel32,
        m,
        moffs,
        sreg,
        st, mm, mm_m64,
        xmm0, xmm, xmm_m32, xmm_m64, xmm_m128,
        ymm, ymm_m256,
        // zig fmt: on

    pub fn fromOperand(operand: Operand) Op {
        return switch (operand) {
            .none => .none,

            .reg => |reg| switch (reg.class()) {
                .general_purpose => if (reg.to64() == .rax)
                    switch (reg) {
                        .al => .al,
                        .ax => .ax,
                        .eax => .eax,
                        .rax => .rax,
                        else => unreachable,
                    }
                else if (reg == .cl)
                    .cl
                else switch (reg.bitSize()) {
                    8 => .r8,
                    16 => .r16,
                    32 => .r32,
                    64 => .r64,
                    else => unreachable,
                },
                .segment => .sreg,
                .x87 => .st,
                .mmx => .mm,
                .sse => if (reg == .xmm0)
                    .xmm0
                else switch (reg.bitSize()) {
                    128 => .xmm,
                    256 => .ymm,
                    else => unreachable,
                },
            },

            .mem => |mem| switch (mem) {
                .m_moffs => .moffs,
                .m_sib, .m_rip => switch (mem.bitSize()) {
                    8 => .m8,
                    16 => .m16,
                    32 => .m32,
                    64 => .m64,
                    80 => .m80,
                    128 => .m128,
                    256 => .m256,
                    else => unreachable,
                },
            },

            .imm => |imm| switch (imm) {
                .signed => |x| if (x == 1)
                    .unity
                else if (math.cast(i8, x)) |_|
                    .imm8s
                else if (math.cast(i16, x)) |_|
                    .imm16s
                else
                    .imm32s,
                .unsigned => |x| if (x == 1)
                    .unity
                else if (math.cast(i8, x)) |_|
                    .imm8s
                else if (math.cast(u8, x)) |_|
                    .imm8
                else if (math.cast(i16, x)) |_|
                    .imm16s
                else if (math.cast(u16, x)) |_|
                    .imm16
                else if (math.cast(i32, x)) |_|
                    .imm32s
                else if (math.cast(u32, x)) |_|
                    .imm32
                else
                    .imm64,
            },
        };
    }

    pub fn immBitSize(op: Op) u64 {
        return switch (op) {
            .none, .o16, .o32, .o64, .moffs, .m, .sreg => unreachable,
            .al, .cl, .r8, .rm8, .r32_m8 => unreachable,
            .ax, .r16, .rm16 => unreachable,
            .eax, .r32, .rm32, .r32_m16 => unreachable,
            .rax, .r64, .rm64, .r64_m16 => unreachable,
            .st, .mm, .mm_m64 => unreachable,
            .xmm0, .xmm, .xmm_m32, .xmm_m64, .xmm_m128 => unreachable,
            .ymm, .ymm_m256 => unreachable,
            .m8, .m16, .m32, .m64, .m80, .m128, .m256 => unreachable,
            .unity => 1,
            .imm8, .imm8s, .rel8 => 8,
            .imm16, .imm16s, .rel16 => 16,
            .imm32, .imm32s, .rel32 => 32,
            .imm64 => 64,
        };
    }

    pub fn regBitSize(op: Op) u64 {
        return switch (op) {
            .none, .o16, .o32, .o64, .moffs, .m, .sreg => unreachable,
            .unity, .imm8, .imm8s, .imm16, .imm16s, .imm32, .imm32s, .imm64 => unreachable,
            .rel8, .rel16, .rel32 => unreachable,
            .m8, .m16, .m32, .m64, .m80, .m128, .m256 => unreachable,
            .al, .cl, .r8, .rm8 => 8,
            .ax, .r16, .rm16 => 16,
            .eax, .r32, .rm32, .r32_m8, .r32_m16 => 32,
            .rax, .r64, .rm64, .r64_m16, .mm, .mm_m64 => 64,
            .st => 80,
            .xmm0, .xmm, .xmm_m32, .xmm_m64, .xmm_m128 => 128,
            .ymm, .ymm_m256 => 256,
        };
    }

    pub fn memBitSize(op: Op) u64 {
        return switch (op) {
            .none, .o16, .o32, .o64, .moffs, .m, .sreg => unreachable,
            .unity, .imm8, .imm8s, .imm16, .imm16s, .imm32, .imm32s, .imm64 => unreachable,
            .rel8, .rel16, .rel32 => unreachable,
            .al, .cl, .r8, .ax, .r16, .eax, .r32, .rax, .r64 => unreachable,
            .st, .mm, .xmm0, .xmm, .ymm => unreachable,
            .m8, .rm8, .r32_m8 => 8,
            .m16, .rm16, .r32_m16, .r64_m16 => 16,
            .m32, .rm32, .xmm_m32 => 32,
            .m64, .rm64, .mm_m64, .xmm_m64 => 64,
            .m80 => 80,
            .m128, .xmm_m128 => 128,
            .m256, .ymm_m256 => 256,
        };
    }

    pub fn isSigned(op: Op) bool {
        return switch (op) {
            .unity, .imm8, .imm16, .imm32, .imm64 => false,
            .imm8s, .imm16s, .imm32s => true,
            else => unreachable,
        };
    }

    pub fn isUnsigned(op: Op) bool {
        return !op.isSigned();
    }

    pub fn isRegister(op: Op) bool {
        // zig fmt: off
            return switch (op) {
                .cl,
                .al, .ax, .eax, .rax,
                .r8, .r16, .r32, .r64,
                .rm8, .rm16, .rm32, .rm64,
                .r32_m8, .r32_m16, .r64_m16,
                .st, .mm, .mm_m64,
                .xmm0, .xmm, .xmm_m32, .xmm_m64, .xmm_m128,
                .ymm, .ymm_m256,
                => true,
                else => false,
            };
            // zig fmt: on
    }

    pub fn isImmediate(op: Op) bool {
        // zig fmt: off
            return switch (op) {
                .imm8, .imm16, .imm32, .imm64,
                .imm8s, .imm16s, .imm32s,
                .rel8, .rel16, .rel32,
                .unity,
                =>  true,
                else => false,
            };
            // zig fmt: on
    }

    pub fn isMemory(op: Op) bool {
        // zig fmt: off
            return switch (op) {
                .rm8, .rm16, .rm32, .rm64,
                .r32_m8, .r32_m16, .r64_m16,
                .m8, .m16, .m32, .m64, .m80, .m128, .m256,
                .m,
                .mm_m64,
                .xmm_m32, .xmm_m64, .xmm_m128,
                .ymm_m256,
                =>  true,
                else => false,
            };
            // zig fmt: on
    }

    pub fn isSegmentRegister(op: Op) bool {
        return switch (op) {
            .moffs, .sreg => true,
            else => false,
        };
    }

    pub fn class(op: Op) Register.Class {
        return switch (op) {
            else => unreachable,
            .al, .ax, .eax, .rax, .cl => .general_purpose,
            .r8, .r16, .r32, .r64 => .general_purpose,
            .rm8, .rm16, .rm32, .rm64 => .general_purpose,
            .r32_m8, .r32_m16, .r64_m16 => .general_purpose,
            .sreg => .segment,
            .st => .x87,
            .mm, .mm_m64 => .mmx,
            .xmm0, .xmm, .xmm_m32, .xmm_m64, .xmm_m128 => .sse,
            .ymm, .ymm_m256 => .sse,
        };
    }

    /// Given an operand `op` checks if `target` is a subset for the purposes of the encoding.
    pub fn isSubset(op: Op, target: Op) bool {
        switch (op) {
            .m, .o16, .o32, .o64 => unreachable,
            .moffs, .sreg => return op == target,
            .none => switch (target) {
                .o16, .o32, .o64, .none => return true,
                else => return false,
            },
            else => {
                if (op.isRegister() and target.isRegister()) {
                    return switch (target) {
                        .cl, .al, .ax, .eax, .rax, .xmm0 => op == target,
                        else => op.class() == target.class() and op.regBitSize() == target.regBitSize(),
                    };
                }
                if (op.isMemory() and target.isMemory()) {
                    switch (target) {
                        .m => return true,
                        else => return op.memBitSize() == target.memBitSize(),
                    }
                }
                if (op.isImmediate() and target.isImmediate()) {
                    switch (target) {
                        .imm64 => if (op.immBitSize() <= 64) return true,
                        .imm32s, .rel32 => if (op.immBitSize() < 32 or (op.immBitSize() == 32 and op.isSigned()))
                            return true,
                        .imm32 => if (op.immBitSize() <= 32) return true,
                        .imm16s, .rel16 => if (op.immBitSize() < 16 or (op.immBitSize() == 16 and op.isSigned()))
                            return true,
                        .imm16 => if (op.immBitSize() <= 16) return true,
                        .imm8s, .rel8 => if (op.immBitSize() < 8 or (op.immBitSize() == 8 and op.isSigned()))
                            return true,
                        .imm8 => if (op.immBitSize() <= 8) return true,
                        else => {},
                    }
                    return op == target;
                }
                return false;
            },
        }
    }
};

pub const Mode = enum {
    // zig fmt: off
        none,
        short, long,
        rex, rex_short,
        vex_128_w0, vex_128_w1, vex_128_wig,
        vex_256_w0, vex_256_w1, vex_256_wig,
        vex_lig_w0, vex_lig_w1, vex_lig_wig,
        vex_lz_w0,  vex_lz_w1,  vex_lz_wig,
        // zig fmt: on

    pub fn isShort(mode: Mode) bool {
        return switch (mode) {
            .short, .rex_short => true,
            else => false,
        };
    }

    pub fn isLong(mode: Mode) bool {
        return switch (mode) {
            .long,
            .vex_128_w1,
            .vex_256_w1,
            .vex_lig_w1,
            .vex_lz_w1,
            => true,
            else => false,
        };
    }

    pub fn isRex(mode: Mode) bool {
        return switch (mode) {
            else => false,
            .rex, .rex_short => true,
        };
    }

    pub fn isVex(mode: Mode) bool {
        return switch (mode) {
            // zig fmt: off
                else => false,
                .vex_128_w0, .vex_128_w1, .vex_128_wig,
                .vex_256_w0, .vex_256_w1, .vex_256_wig,
                .vex_lig_w0, .vex_lig_w1, .vex_lig_wig,
                .vex_lz_w0,  .vex_lz_w1,  .vex_lz_wig,
                => true,
                // zig fmt: on
        };
    }

    pub fn isVecLong(mode: Mode) bool {
        return switch (mode) {
            // zig fmt: off
                else => unreachable,
                .vex_128_w0, .vex_128_w1, .vex_128_wig,
                .vex_lig_w0, .vex_lig_w1, .vex_lig_wig,
                .vex_lz_w0,  .vex_lz_w1,  .vex_lz_wig,
                => false,
                .vex_256_w0, .vex_256_w1, .vex_256_wig,
                => true,
                // zig fmt: on
        };
    }
};

pub const Feature = enum {
    none,
    avx,
    avx2,
    bmi,
    f16c,
    fma,
    lzcnt,
    movbe,
    popcnt,
    sse,
    sse2,
    sse3,
    sse4_1,
    x87,
};

pub const Encoding = struct {
    mnemonic: Mnemonic,
    data: Data,

    const Data = struct {
        op_en: OpEn,
        ops: [4]Op,
        opc_len: u3,
        opc: [7]u8,
        modrm_ext: u3,
        mode: Mode,
        feature: Feature,
    };

    pub fn findByMnemonic(
        prefix: Prefix,
        mnemonic: Mnemonic,
        ops: []const Operand,
    ) ?Encoding {
        var input_ops = [1]Op{.none} ** 4;
        for (input_ops[0..ops.len], ops) |*input_op, op| input_op.* = Op.fromOperand(op);

        const rex_required = for (ops) |op| switch (op) {
            .reg => |r| switch (r) {
                .spl, .bpl, .sil, .dil => break true,
                else => {},
            },
            else => {},
        } else false;
        const rex_invalid = for (ops) |op| switch (op) {
            .reg => |r| switch (r) {
                .ah, .bh, .ch, .dh => break true,
                else => {},
            },
            else => {},
        } else false;
        const rex_extended = for (ops) |op| {
            if (op.isBaseExtended() or op.isIndexExtended()) break true;
        } else false;

        if ((rex_required or rex_extended) and rex_invalid) return null;

        var shortest_enc: ?Encoding = null;
        var shortest_len: ?usize = null;
        next: for (mnemonic_to_encodings_map[@intFromEnum(mnemonic)]) |data| {
            switch (data.mode) {
                .none, .short => if (rex_required) continue,
                .rex, .rex_short => if (!rex_required) continue,
                else => {},
            }
            for (input_ops, data.ops) |input_op, data_op|
                if (!input_op.isSubset(data_op)) continue :next;

            const enc = Encoding{ .mnemonic = mnemonic, .data = data };
            if (shortest_enc) |previous_shortest_enc| {
                const len = estimateInstructionLength(prefix, enc, ops);
                const previous_shortest_len = shortest_len orelse
                    estimateInstructionLength(prefix, previous_shortest_enc, ops);
                if (len < previous_shortest_len) {
                    shortest_enc = enc;
                    shortest_len = len;
                } else shortest_len = previous_shortest_len;
            } else shortest_enc = enc;
        }
        return shortest_enc;
    }

    /// Returns first matching encoding by opcode.
    pub fn findByOpcode(opc: []const u8, prefixes: struct {
        legacy: LegacyPrefixes,
        rex: Rex,
    }, modrm_ext: ?u3) ?Encoding {
        for (mnemonic_to_encodings_map, 0..) |encs, mnemonic_int| for (encs) |data| {
            const enc = Encoding{ .mnemonic = @as(Mnemonic, @enumFromInt(mnemonic_int)), .data = data };
            if (modrm_ext) |ext| if (ext != data.modrm_ext) continue;
            if (!std.mem.eql(u8, opc, enc.opcode())) continue;
            if (prefixes.rex.w) {
                if (!data.mode.isLong()) continue;
            } else if (prefixes.rex.present and !prefixes.rex.isSet()) {
                if (!data.mode.isRex()) continue;
            } else if (prefixes.legacy.prefix_66) {
                if (!data.mode.isShort()) continue;
            } else {
                if (data.mode.isShort()) continue;
            }
            return enc;
        };
        return null;
    }

    pub fn opcode(encoding: *const Encoding) []const u8 {
        return encoding.data.opc[0..encoding.data.opc_len];
    }

    pub fn mandatoryPrefix(encoding: *const Encoding) ?u8 {
        const prefix = encoding.data.opc[0];
        return switch (prefix) {
            0x66, 0xf2, 0xf3 => prefix,
            else => null,
        };
    }

    pub fn modRmExt(encoding: Encoding) u3 {
        return switch (encoding.data.op_en) {
            .m, .mi, .m1, .mc, .vmi => encoding.data.modrm_ext,
            else => unreachable,
        };
    }

    pub fn format(
        encoding: Encoding,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = options;
        _ = fmt;

        var opc = encoding.opcode();
        if (encoding.data.mode.isVex()) {
            try writer.writeAll("VEX.");

            try writer.writeAll(switch (encoding.data.mode) {
                .vex_128_w0, .vex_128_w1, .vex_128_wig => "128",
                .vex_256_w0, .vex_256_w1, .vex_256_wig => "256",
                .vex_lig_w0, .vex_lig_w1, .vex_lig_wig => "LIG",
                .vex_lz_w0, .vex_lz_w1, .vex_lz_wig => "LZ",
                else => unreachable,
            });

            switch (opc[0]) {
                else => {},
                0x66, 0xf3, 0xf2 => {
                    try writer.print(".{X:0>2}", .{opc[0]});
                    opc = opc[1..];
                },
            }

            try writer.print(".{}", .{std.fmt.fmtSliceHexUpper(opc[0 .. opc.len - 1])});
            opc = opc[opc.len - 1 ..];

            try writer.writeAll(".W");
            try writer.writeAll(switch (encoding.data.mode) {
                .vex_128_w0, .vex_256_w0, .vex_lig_w0, .vex_lz_w0 => "0",
                .vex_128_w1, .vex_256_w1, .vex_lig_w1, .vex_lz_w1 => "1",
                .vex_128_wig, .vex_256_wig, .vex_lig_wig, .vex_lz_wig => "IG",
                else => unreachable,
            });

            try writer.writeByte(' ');
        } else if (encoding.data.mode.isLong()) try writer.writeAll("REX.W + ");
        for (opc) |byte| try writer.print("{x:0>2} ", .{byte});

        switch (encoding.data.op_en) {
            .np, .fd, .td, .i, .zi, .d => {},
            .o, .oi => {
                const tag = switch (encoding.data.ops[0]) {
                    .r8 => "rb",
                    .r16 => "rw",
                    .r32 => "rd",
                    .r64 => "rd",
                    else => unreachable,
                };
                try writer.print("+{s} ", .{tag});
            },
            .m, .mi, .m1, .mc, .vmi => try writer.print("/{d} ", .{encoding.modRmExt()}),
            .mr, .rm, .rmi, .mri, .mrc, .rm0, .rvm, .rvmr, .rvmi, .mvr => try writer.writeAll("/r "),
        }

        switch (encoding.data.op_en) {
            .i, .d, .zi, .oi, .mi, .rmi, .mri, .vmi, .rvmi => {
                const op = switch (encoding.data.op_en) {
                    .i, .d => encoding.data.ops[0],
                    .zi, .oi, .mi => encoding.data.ops[1],
                    .rmi, .mri, .vmi => encoding.data.ops[2],
                    .rvmi => encoding.data.ops[3],
                    else => unreachable,
                };
                const tag = switch (op) {
                    .imm8, .imm8s => "ib",
                    .imm16, .imm16s => "iw",
                    .imm32, .imm32s => "id",
                    .imm64 => "io",
                    .rel8 => "cb",
                    .rel16 => "cw",
                    .rel32 => "cd",
                    else => unreachable,
                };
                try writer.print("{s} ", .{tag});
            },
            .rvmr => try writer.writeAll("/is4 "),
            .np, .fd, .td, .o, .m, .m1, .mc, .mr, .rm, .mrc, .rm0, .rvm, .mvr => {},
        }

        try writer.print("{s} ", .{@tagName(encoding.mnemonic)});

        for (encoding.data.ops) |op| switch (op) {
            .none, .o16, .o32, .o64 => break,
            else => try writer.print("{s} ", .{@tagName(op)}),
        };

        const op_en = switch (encoding.data.op_en) {
            .zi => .i,
            else => |op_en| op_en,
        };
        try writer.print("{s}", .{@tagName(op_en)});
    }

    fn estimateInstructionLength(prefix: Prefix, encoding: Encoding, ops: []const Operand) usize {
        var inst = Instruction{
            .prefix = prefix,
            .encoding = encoding,
            .ops = [1]Operand{.none} ** 4,
        };
        @memcpy(inst.ops[0..ops.len], ops);

        var cwriter = std.io.countingWriter(std.io.null_writer);
        inst.encode(cwriter.writer(), .{ .allow_frame_loc = true }) catch unreachable; // Not allowed to fail here unless OOM.
        return @as(usize, @intCast(cwriter.bytes_written));
    }

    const mnemonic_to_encodings_map = init: {
        @setEvalBranchQuota(5_000);
        const mnemonic_count = @typeInfo(Mnemonic).@"enum".fields.len;
        var mnemonic_map: [mnemonic_count][]Data = .{&.{}} ** mnemonic_count;
        const encodings = struct {
            const modrm_ext = u3;

            pub const Entry = struct { Mnemonic, OpEn, []const Op, []const u8, modrm_ext, Mode, Feature };

            // TODO move this into a .zon file when Zig is capable of importing .zon files
            // zig fmt: off
            pub const table = [_]Entry{
                // General-purpose
                .{ .adc, .zi, &.{ .al,   .imm8   }, &.{ 0x14 }, 0, .none,  .none },
                .{ .adc, .zi, &.{ .ax,   .imm16  }, &.{ 0x15 }, 0, .short, .none },
                .{ .adc, .zi, &.{ .eax,  .imm32  }, &.{ 0x15 }, 0, .none,  .none },
                .{ .adc, .zi, &.{ .rax,  .imm32s }, &.{ 0x15 }, 0, .long,  .none },
                .{ .adc, .mi, &.{ .rm8,  .imm8   }, &.{ 0x80 }, 2, .none,  .none },
                .{ .adc, .mi, &.{ .rm8,  .imm8   }, &.{ 0x80 }, 2, .rex,   .none },
                .{ .adc, .mi, &.{ .rm16, .imm16  }, &.{ 0x81 }, 2, .short, .none },
                .{ .adc, .mi, &.{ .rm32, .imm32  }, &.{ 0x81 }, 2, .none,  .none },
                .{ .adc, .mi, &.{ .rm64, .imm32s }, &.{ 0x81 }, 2, .long,  .none },
                .{ .adc, .mi, &.{ .rm16, .imm8s  }, &.{ 0x83 }, 2, .short, .none },
                .{ .adc, .mi, &.{ .rm32, .imm8s  }, &.{ 0x83 }, 2, .none,  .none },
                .{ .adc, .mi, &.{ .rm64, .imm8s  }, &.{ 0x83 }, 2, .long,  .none },
                .{ .adc, .mr, &.{ .rm8,  .r8     }, &.{ 0x10 }, 0, .none,  .none },
                .{ .adc, .mr, &.{ .rm8,  .r8     }, &.{ 0x10 }, 0, .rex,   .none },
                .{ .adc, .mr, &.{ .rm16, .r16    }, &.{ 0x11 }, 0, .short, .none },
                .{ .adc, .mr, &.{ .rm32, .r32    }, &.{ 0x11 }, 0, .none,  .none },
                .{ .adc, .mr, &.{ .rm64, .r64    }, &.{ 0x11 }, 0, .long,  .none },
                .{ .adc, .rm, &.{ .r8,   .rm8    }, &.{ 0x12 }, 0, .none,  .none },
                .{ .adc, .rm, &.{ .r8,   .rm8    }, &.{ 0x12 }, 0, .rex,   .none },
                .{ .adc, .rm, &.{ .r16,  .rm16   }, &.{ 0x13 }, 0, .short, .none },
                .{ .adc, .rm, &.{ .r32,  .rm32   }, &.{ 0x13 }, 0, .none,  .none },
                .{ .adc, .rm, &.{ .r64,  .rm64   }, &.{ 0x13 }, 0, .long,  .none },

                .{ .add, .zi, &.{ .al,   .imm8   }, &.{ 0x04 }, 0, .none,  .none },
                .{ .add, .zi, &.{ .ax,   .imm16  }, &.{ 0x05 }, 0, .short, .none },
                .{ .add, .zi, &.{ .eax,  .imm32  }, &.{ 0x05 }, 0, .none,  .none },
                .{ .add, .zi, &.{ .rax,  .imm32s }, &.{ 0x05 }, 0, .long,  .none },
                .{ .add, .mi, &.{ .rm8,  .imm8   }, &.{ 0x80 }, 0, .none,  .none },
                .{ .add, .mi, &.{ .rm8,  .imm8   }, &.{ 0x80 }, 0, .rex,   .none },
                .{ .add, .mi, &.{ .rm16, .imm16  }, &.{ 0x81 }, 0, .short, .none },
                .{ .add, .mi, &.{ .rm32, .imm32  }, &.{ 0x81 }, 0, .none,  .none },
                .{ .add, .mi, &.{ .rm64, .imm32s }, &.{ 0x81 }, 0, .long,  .none },
                .{ .add, .mi, &.{ .rm16, .imm8s  }, &.{ 0x83 }, 0, .short, .none },
                .{ .add, .mi, &.{ .rm32, .imm8s  }, &.{ 0x83 }, 0, .none,  .none },
                .{ .add, .mi, &.{ .rm64, .imm8s  }, &.{ 0x83 }, 0, .long,  .none },
                .{ .add, .mr, &.{ .rm8,  .r8     }, &.{ 0x00 }, 0, .none,  .none },
                .{ .add, .mr, &.{ .rm8,  .r8     }, &.{ 0x00 }, 0, .rex,   .none },
                .{ .add, .mr, &.{ .rm16, .r16    }, &.{ 0x01 }, 0, .short, .none },
                .{ .add, .mr, &.{ .rm32, .r32    }, &.{ 0x01 }, 0, .none,  .none },
                .{ .add, .mr, &.{ .rm64, .r64    }, &.{ 0x01 }, 0, .long,  .none },
                .{ .add, .rm, &.{ .r8,   .rm8    }, &.{ 0x02 }, 0, .none,  .none },
                .{ .add, .rm, &.{ .r8,   .rm8    }, &.{ 0x02 }, 0, .rex,   .none },
                .{ .add, .rm, &.{ .r16,  .rm16   }, &.{ 0x03 }, 0, .short, .none },
                .{ .add, .rm, &.{ .r32,  .rm32   }, &.{ 0x03 }, 0, .none,  .none },
                .{ .add, .rm, &.{ .r64,  .rm64   }, &.{ 0x03 }, 0, .long,  .none },

                .{ .@"and", .zi, &.{ .al,   .imm8   }, &.{ 0x24 }, 0, .none,  .none },
                .{ .@"and", .zi, &.{ .ax,   .imm16  }, &.{ 0x25 }, 0, .short, .none },
                .{ .@"and", .zi, &.{ .eax,  .imm32  }, &.{ 0x25 }, 0, .none,  .none },
                .{ .@"and", .zi, &.{ .rax,  .imm32s }, &.{ 0x25 }, 0, .long,  .none },
                .{ .@"and", .mi, &.{ .rm8,  .imm8   }, &.{ 0x80 }, 4, .none,  .none },
                .{ .@"and", .mi, &.{ .rm8,  .imm8   }, &.{ 0x80 }, 4, .rex,   .none },
                .{ .@"and", .mi, &.{ .rm16, .imm16  }, &.{ 0x81 }, 4, .short, .none },
                .{ .@"and", .mi, &.{ .rm32, .imm32  }, &.{ 0x81 }, 4, .none,  .none },
                .{ .@"and", .mi, &.{ .rm64, .imm32s }, &.{ 0x81 }, 4, .long,  .none },
                .{ .@"and", .mi, &.{ .rm16, .imm8s  }, &.{ 0x83 }, 4, .short, .none },
                .{ .@"and", .mi, &.{ .rm32, .imm8s  }, &.{ 0x83 }, 4, .none,  .none },
                .{ .@"and", .mi, &.{ .rm64, .imm8s  }, &.{ 0x83 }, 4, .long,  .none },
                .{ .@"and", .mr, &.{ .rm8,  .r8     }, &.{ 0x20 }, 0, .none,  .none },
                .{ .@"and", .mr, &.{ .rm8,  .r8     }, &.{ 0x20 }, 0, .rex,   .none },
                .{ .@"and", .mr, &.{ .rm16, .r16    }, &.{ 0x21 }, 0, .short, .none },
                .{ .@"and", .mr, &.{ .rm32, .r32    }, &.{ 0x21 }, 0, .none,  .none },
                .{ .@"and", .mr, &.{ .rm64, .r64    }, &.{ 0x21 }, 0, .long,  .none },
                .{ .@"and", .rm, &.{ .r8,   .rm8    }, &.{ 0x22 }, 0, .none,  .none },
                .{ .@"and", .rm, &.{ .r8,   .rm8    }, &.{ 0x22 }, 0, .rex,   .none },
                .{ .@"and", .rm, &.{ .r16,  .rm16   }, &.{ 0x23 }, 0, .short, .none },
                .{ .@"and", .rm, &.{ .r32,  .rm32   }, &.{ 0x23 }, 0, .none,  .none },
                .{ .@"and", .rm, &.{ .r64,  .rm64   }, &.{ 0x23 }, 0, .long,  .none },

                .{ .bsf, .rm, &.{ .r16, .rm16 }, &.{ 0x0f, 0xbc }, 0, .short, .none },
                .{ .bsf, .rm, &.{ .r32, .rm32 }, &.{ 0x0f, 0xbc }, 0, .none,  .none },
                .{ .bsf, .rm, &.{ .r64, .rm64 }, &.{ 0x0f, 0xbc }, 0, .long,  .none },

                .{ .bsr, .rm, &.{ .r16, .rm16 }, &.{ 0x0f, 0xbd }, 0, .short, .none },
                .{ .bsr, .rm, &.{ .r32, .rm32 }, &.{ 0x0f, 0xbd }, 0, .none,  .none },
                .{ .bsr, .rm, &.{ .r64, .rm64 }, &.{ 0x0f, 0xbd }, 0, .long,  .none },

                .{ .bswap, .o, &.{ .r32 }, &.{ 0x0f, 0xc8 }, 0, .none, .none },
                .{ .bswap, .o, &.{ .r64 }, &.{ 0x0f, 0xc8 }, 0, .long, .none },

                .{ .bt, .mr, &.{ .rm16, .r16  }, &.{ 0x0f, 0xa3 }, 0, .short, .none },
                .{ .bt, .mr, &.{ .rm32, .r32  }, &.{ 0x0f, 0xa3 }, 0, .none,  .none },
                .{ .bt, .mr, &.{ .rm64, .r64  }, &.{ 0x0f, 0xa3 }, 0, .long,  .none },
                .{ .bt, .mi, &.{ .rm16, .imm8 }, &.{ 0x0f, 0xba }, 4, .short, .none },
                .{ .bt, .mi, &.{ .rm32, .imm8 }, &.{ 0x0f, 0xba }, 4, .none,  .none },
                .{ .bt, .mi, &.{ .rm64, .imm8 }, &.{ 0x0f, 0xba }, 4, .long,  .none },

                .{ .btc, .mr, &.{ .rm16, .r16  }, &.{ 0x0f, 0xbb }, 0, .short, .none },
                .{ .btc, .mr, &.{ .rm32, .r32  }, &.{ 0x0f, 0xbb }, 0, .none,  .none },
                .{ .btc, .mr, &.{ .rm64, .r64  }, &.{ 0x0f, 0xbb }, 0, .long,  .none },
                .{ .btc, .mi, &.{ .rm16, .imm8 }, &.{ 0x0f, 0xba }, 7, .short, .none },
                .{ .btc, .mi, &.{ .rm32, .imm8 }, &.{ 0x0f, 0xba }, 7, .none,  .none },
                .{ .btc, .mi, &.{ .rm64, .imm8 }, &.{ 0x0f, 0xba }, 7, .long,  .none },

                .{ .btr, .mr, &.{ .rm16, .r16  }, &.{ 0x0f, 0xb3 }, 0, .short, .none },
                .{ .btr, .mr, &.{ .rm32, .r32  }, &.{ 0x0f, 0xb3 }, 0, .none,  .none },
                .{ .btr, .mr, &.{ .rm64, .r64  }, &.{ 0x0f, 0xb3 }, 0, .long,  .none },
                .{ .btr, .mi, &.{ .rm16, .imm8 }, &.{ 0x0f, 0xba }, 6, .short, .none },
                .{ .btr, .mi, &.{ .rm32, .imm8 }, &.{ 0x0f, 0xba }, 6, .none,  .none },
                .{ .btr, .mi, &.{ .rm64, .imm8 }, &.{ 0x0f, 0xba }, 6, .long,  .none },

                .{ .bts, .mr, &.{ .rm16, .r16  }, &.{ 0x0f, 0xab }, 0, .short, .none },
                .{ .bts, .mr, &.{ .rm32, .r32  }, &.{ 0x0f, 0xab }, 0, .none,  .none },
                .{ .bts, .mr, &.{ .rm64, .r64  }, &.{ 0x0f, 0xab }, 0, .long,  .none },
                .{ .bts, .mi, &.{ .rm16, .imm8 }, &.{ 0x0f, 0xba }, 5, .short, .none },
                .{ .bts, .mi, &.{ .rm32, .imm8 }, &.{ 0x0f, 0xba }, 5, .none,  .none },
                .{ .bts, .mi, &.{ .rm64, .imm8 }, &.{ 0x0f, 0xba }, 5, .long,  .none },

                // This is M encoding according to Intel, but D makes more sense here.
                .{ .call, .d, &.{ .rel32 }, &.{ 0xe8 }, 0, .none, .none },
                .{ .call, .m, &.{ .rm64  }, &.{ 0xff }, 2, .none, .none },

                .{ .cbw,  .np, &.{ .o16 }, &.{ 0x98 }, 0, .short, .none },
                .{ .cwde, .np, &.{ .o32 }, &.{ 0x98 }, 0, .none,  .none },
                .{ .cdqe, .np, &.{ .o64 }, &.{ 0x98 }, 0, .long,  .none },

                .{ .cwd, .np, &.{ .o16 }, &.{ 0x99 }, 0, .short, .none },
                .{ .cdq, .np, &.{ .o32 }, &.{ 0x99 }, 0, .none,  .none },
                .{ .cqo, .np, &.{ .o64 }, &.{ 0x99 }, 0, .long,  .none },

                .{ .cmova,   .rm, &.{ .r16, .rm16 }, &.{ 0x0f, 0x47 }, 0, .short, .none },
                .{ .cmova,   .rm, &.{ .r32, .rm32 }, &.{ 0x0f, 0x47 }, 0, .none,  .none },
                .{ .cmova,   .rm, &.{ .r64, .rm64 }, &.{ 0x0f, 0x47 }, 0, .long,  .none },
                .{ .cmovae,  .rm, &.{ .r16, .rm16 }, &.{ 0x0f, 0x43 }, 0, .short, .none },
                .{ .cmovae,  .rm, &.{ .r32, .rm32 }, &.{ 0x0f, 0x43 }, 0, .none,  .none },
                .{ .cmovae,  .rm, &.{ .r64, .rm64 }, &.{ 0x0f, 0x43 }, 0, .long,  .none },
                .{ .cmovb,   .rm, &.{ .r16, .rm16 }, &.{ 0x0f, 0x42 }, 0, .short, .none },
                .{ .cmovb,   .rm, &.{ .r32, .rm32 }, &.{ 0x0f, 0x42 }, 0, .none,  .none },
                .{ .cmovb,   .rm, &.{ .r64, .rm64 }, &.{ 0x0f, 0x42 }, 0, .long,  .none },
                .{ .cmovbe,  .rm, &.{ .r16, .rm16 }, &.{ 0x0f, 0x46 }, 0, .short, .none },
                .{ .cmovbe,  .rm, &.{ .r32, .rm32 }, &.{ 0x0f, 0x46 }, 0, .none,  .none },
                .{ .cmovbe,  .rm, &.{ .r64, .rm64 }, &.{ 0x0f, 0x46 }, 0, .long,  .none },
                .{ .cmovc,   .rm, &.{ .r16, .rm16 }, &.{ 0x0f, 0x42 }, 0, .short, .none },
                .{ .cmovc,   .rm, &.{ .r32, .rm32 }, &.{ 0x0f, 0x42 }, 0, .none,  .none },
                .{ .cmovc,   .rm, &.{ .r64, .rm64 }, &.{ 0x0f, 0x42 }, 0, .long,  .none },
                .{ .cmove,   .rm, &.{ .r16, .rm16 }, &.{ 0x0f, 0x44 }, 0, .short, .none },
                .{ .cmove,   .rm, &.{ .r32, .rm32 }, &.{ 0x0f, 0x44 }, 0, .none,  .none },
                .{ .cmove,   .rm, &.{ .r64, .rm64 }, &.{ 0x0f, 0x44 }, 0, .long,  .none },
                .{ .cmovg,   .rm, &.{ .r16, .rm16 }, &.{ 0x0f, 0x4f }, 0, .short, .none },
                .{ .cmovg,   .rm, &.{ .r32, .rm32 }, &.{ 0x0f, 0x4f }, 0, .none,  .none },
                .{ .cmovg,   .rm, &.{ .r64, .rm64 }, &.{ 0x0f, 0x4f }, 0, .long,  .none },
                .{ .cmovge,  .rm, &.{ .r16, .rm16 }, &.{ 0x0f, 0x4d }, 0, .short, .none },
                .{ .cmovge,  .rm, &.{ .r32, .rm32 }, &.{ 0x0f, 0x4d }, 0, .none,  .none },
                .{ .cmovge,  .rm, &.{ .r64, .rm64 }, &.{ 0x0f, 0x4d }, 0, .long,  .none },
                .{ .cmovl,   .rm, &.{ .r16, .rm16 }, &.{ 0x0f, 0x4c }, 0, .short, .none },
                .{ .cmovl,   .rm, &.{ .r32, .rm32 }, &.{ 0x0f, 0x4c }, 0, .none,  .none },
                .{ .cmovl,   .rm, &.{ .r64, .rm64 }, &.{ 0x0f, 0x4c }, 0, .long,  .none },
                .{ .cmovle,  .rm, &.{ .r16, .rm16 }, &.{ 0x0f, 0x4e }, 0, .short, .none },
                .{ .cmovle,  .rm, &.{ .r32, .rm32 }, &.{ 0x0f, 0x4e }, 0, .none,  .none },
                .{ .cmovle,  .rm, &.{ .r64, .rm64 }, &.{ 0x0f, 0x4e }, 0, .long,  .none },
                .{ .cmovna,  .rm, &.{ .r16, .rm16 }, &.{ 0x0f, 0x46 }, 0, .short, .none },
                .{ .cmovna,  .rm, &.{ .r32, .rm32 }, &.{ 0x0f, 0x46 }, 0, .none,  .none },
                .{ .cmovna,  .rm, &.{ .r64, .rm64 }, &.{ 0x0f, 0x46 }, 0, .long,  .none },
                .{ .cmovnae, .rm, &.{ .r16, .rm16 }, &.{ 0x0f, 0x42 }, 0, .short, .none },
                .{ .cmovnae, .rm, &.{ .r32, .rm32 }, &.{ 0x0f, 0x42 }, 0, .none,  .none },
                .{ .cmovnae, .rm, &.{ .r64, .rm64 }, &.{ 0x0f, 0x42 }, 0, .long,  .none },
                .{ .cmovnb,  .rm, &.{ .r16, .rm16 }, &.{ 0x0f, 0x43 }, 0, .short, .none },
                .{ .cmovnb,  .rm, &.{ .r32, .rm32 }, &.{ 0x0f, 0x43 }, 0, .none,  .none },
                .{ .cmovnb,  .rm, &.{ .r64, .rm64 }, &.{ 0x0f, 0x43 }, 0, .long,  .none },
                .{ .cmovnbe, .rm, &.{ .r16, .rm16 }, &.{ 0x0f, 0x47 }, 0, .short, .none },
                .{ .cmovnbe, .rm, &.{ .r32, .rm32 }, &.{ 0x0f, 0x47 }, 0, .none,  .none },
                .{ .cmovnbe, .rm, &.{ .r64, .rm64 }, &.{ 0x0f, 0x47 }, 0, .long,  .none },
                .{ .cmovnc,  .rm, &.{ .r16, .rm16 }, &.{ 0x0f, 0x43 }, 0, .short, .none },
                .{ .cmovnc,  .rm, &.{ .r32, .rm32 }, &.{ 0x0f, 0x43 }, 0, .none,  .none },
                .{ .cmovnc,  .rm, &.{ .r64, .rm64 }, &.{ 0x0f, 0x43 }, 0, .long,  .none },
                .{ .cmovne,  .rm, &.{ .r16, .rm16 }, &.{ 0x0f, 0x45 }, 0, .short, .none },
                .{ .cmovne,  .rm, &.{ .r32, .rm32 }, &.{ 0x0f, 0x45 }, 0, .none,  .none },
                .{ .cmovne,  .rm, &.{ .r64, .rm64 }, &.{ 0x0f, 0x45 }, 0, .long,  .none },
                .{ .cmovng,  .rm, &.{ .r16, .rm16 }, &.{ 0x0f, 0x4e }, 0, .short, .none },
                .{ .cmovng,  .rm, &.{ .r32, .rm32 }, &.{ 0x0f, 0x4e }, 0, .none,  .none },
                .{ .cmovng,  .rm, &.{ .r64, .rm64 }, &.{ 0x0f, 0x4e }, 0, .long,  .none },
                .{ .cmovnge, .rm, &.{ .r16, .rm16 }, &.{ 0x0f, 0x4c }, 0, .short, .none },
                .{ .cmovnge, .rm, &.{ .r32, .rm32 }, &.{ 0x0f, 0x4c }, 0, .none,  .none },
                .{ .cmovnge, .rm, &.{ .r64, .rm64 }, &.{ 0x0f, 0x4c }, 0, .long,  .none },
                .{ .cmovnl,  .rm, &.{ .r16, .rm16 }, &.{ 0x0f, 0x4d }, 0, .short, .none },
                .{ .cmovnl,  .rm, &.{ .r32, .rm32 }, &.{ 0x0f, 0x4d }, 0, .none,  .none },
                .{ .cmovnl,  .rm, &.{ .r64, .rm64 }, &.{ 0x0f, 0x4d }, 0, .long,  .none },
                .{ .cmovnle, .rm, &.{ .r16, .rm16 }, &.{ 0x0f, 0x4f }, 0, .short, .none },
                .{ .cmovnle, .rm, &.{ .r32, .rm32 }, &.{ 0x0f, 0x4f }, 0, .none,  .none },
                .{ .cmovnle, .rm, &.{ .r64, .rm64 }, &.{ 0x0f, 0x4f }, 0, .long,  .none },
                .{ .cmovno,  .rm, &.{ .r16, .rm16 }, &.{ 0x0f, 0x41 }, 0, .short, .none },
                .{ .cmovno,  .rm, &.{ .r32, .rm32 }, &.{ 0x0f, 0x41 }, 0, .none,  .none },
                .{ .cmovno,  .rm, &.{ .r64, .rm64 }, &.{ 0x0f, 0x41 }, 0, .long,  .none },
                .{ .cmovnp,  .rm, &.{ .r16, .rm16 }, &.{ 0x0f, 0x4b }, 0, .short, .none },
                .{ .cmovnp,  .rm, &.{ .r32, .rm32 }, &.{ 0x0f, 0x4b }, 0, .none,  .none },
                .{ .cmovnp,  .rm, &.{ .r64, .rm64 }, &.{ 0x0f, 0x4b }, 0, .long,  .none },
                .{ .cmovns,  .rm, &.{ .r16, .rm16 }, &.{ 0x0f, 0x49 }, 0, .short, .none },
                .{ .cmovns,  .rm, &.{ .r32, .rm32 }, &.{ 0x0f, 0x49 }, 0, .none,  .none },
                .{ .cmovns,  .rm, &.{ .r64, .rm64 }, &.{ 0x0f, 0x49 }, 0, .long,  .none },
                .{ .cmovnz,  .rm, &.{ .r16, .rm16 }, &.{ 0x0f, 0x45 }, 0, .short, .none },
                .{ .cmovnz,  .rm, &.{ .r32, .rm32 }, &.{ 0x0f, 0x45 }, 0, .none,  .none },
                .{ .cmovnz,  .rm, &.{ .r64, .rm64 }, &.{ 0x0f, 0x45 }, 0, .long,  .none },
                .{ .cmovo,   .rm, &.{ .r16, .rm16 }, &.{ 0x0f, 0x40 }, 0, .short, .none },
                .{ .cmovo,   .rm, &.{ .r32, .rm32 }, &.{ 0x0f, 0x40 }, 0, .none,  .none },
                .{ .cmovo,   .rm, &.{ .r64, .rm64 }, &.{ 0x0f, 0x40 }, 0, .long,  .none },
                .{ .cmovp,   .rm, &.{ .r16, .rm16 }, &.{ 0x0f, 0x4a }, 0, .short, .none },
                .{ .cmovp,   .rm, &.{ .r32, .rm32 }, &.{ 0x0f, 0x4a }, 0, .none,  .none },
                .{ .cmovp,   .rm, &.{ .r64, .rm64 }, &.{ 0x0f, 0x4a }, 0, .long,  .none },
                .{ .cmovpe,  .rm, &.{ .r16, .rm16 }, &.{ 0x0f, 0x4a }, 0, .short, .none },
                .{ .cmovpe,  .rm, &.{ .r32, .rm32 }, &.{ 0x0f, 0x4a }, 0, .none,  .none },
                .{ .cmovpe,  .rm, &.{ .r64, .rm64 }, &.{ 0x0f, 0x4a }, 0, .long,  .none },
                .{ .cmovpo,  .rm, &.{ .r16, .rm16 }, &.{ 0x0f, 0x4b }, 0, .short, .none },
                .{ .cmovpo,  .rm, &.{ .r32, .rm32 }, &.{ 0x0f, 0x4b }, 0, .none,  .none },
                .{ .cmovpo,  .rm, &.{ .r64, .rm64 }, &.{ 0x0f, 0x4b }, 0, .long,  .none },
                .{ .cmovs,   .rm, &.{ .r16, .rm16 }, &.{ 0x0f, 0x48 }, 0, .short, .none },
                .{ .cmovs,   .rm, &.{ .r32, .rm32 }, &.{ 0x0f, 0x48 }, 0, .none,  .none },
                .{ .cmovs,   .rm, &.{ .r64, .rm64 }, &.{ 0x0f, 0x48 }, 0, .long,  .none },
                .{ .cmovz,   .rm, &.{ .r16, .rm16 }, &.{ 0x0f, 0x44 }, 0, .short, .none },
                .{ .cmovz,   .rm, &.{ .r32, .rm32 }, &.{ 0x0f, 0x44 }, 0, .none,  .none },
                .{ .cmovz,   .rm, &.{ .r64, .rm64 }, &.{ 0x0f, 0x44 }, 0, .long,  .none },

                .{ .cmp, .zi, &.{ .al,   .imm8   }, &.{ 0x3c }, 0, .none,  .none },
                .{ .cmp, .zi, &.{ .ax,   .imm16  }, &.{ 0x3d }, 0, .short, .none },
                .{ .cmp, .zi, &.{ .eax,  .imm32  }, &.{ 0x3d }, 0, .none,  .none },
                .{ .cmp, .zi, &.{ .rax,  .imm32s }, &.{ 0x3d }, 0, .long,  .none },
                .{ .cmp, .mi, &.{ .rm8,  .imm8   }, &.{ 0x80 }, 7, .none,  .none },
                .{ .cmp, .mi, &.{ .rm8,  .imm8   }, &.{ 0x80 }, 7, .rex,   .none },
                .{ .cmp, .mi, &.{ .rm16, .imm16  }, &.{ 0x81 }, 7, .short, .none },
                .{ .cmp, .mi, &.{ .rm32, .imm32  }, &.{ 0x81 }, 7, .none,  .none },
                .{ .cmp, .mi, &.{ .rm64, .imm32s }, &.{ 0x81 }, 7, .long,  .none },
                .{ .cmp, .mi, &.{ .rm16, .imm8s  }, &.{ 0x83 }, 7, .short, .none },
                .{ .cmp, .mi, &.{ .rm32, .imm8s  }, &.{ 0x83 }, 7, .none,  .none },
                .{ .cmp, .mi, &.{ .rm64, .imm8s  }, &.{ 0x83 }, 7, .long,  .none },
                .{ .cmp, .mr, &.{ .rm8,  .r8     }, &.{ 0x38 }, 0, .none,  .none },
                .{ .cmp, .mr, &.{ .rm8,  .r8     }, &.{ 0x38 }, 0, .rex,   .none },
                .{ .cmp, .mr, &.{ .rm16, .r16    }, &.{ 0x39 }, 0, .short, .none },
                .{ .cmp, .mr, &.{ .rm32, .r32    }, &.{ 0x39 }, 0, .none,  .none },
                .{ .cmp, .mr, &.{ .rm64, .r64    }, &.{ 0x39 }, 0, .long,  .none },
                .{ .cmp, .rm, &.{ .r8,   .rm8    }, &.{ 0x3a }, 0, .none,  .none },
                .{ .cmp, .rm, &.{ .r8,   .rm8    }, &.{ 0x3a }, 0, .rex,   .none },
                .{ .cmp, .rm, &.{ .r16,  .rm16   }, &.{ 0x3b }, 0, .short, .none },
                .{ .cmp, .rm, &.{ .r32,  .rm32   }, &.{ 0x3b }, 0, .none,  .none },
                .{ .cmp, .rm, &.{ .r64,  .rm64   }, &.{ 0x3b }, 0, .long,  .none },

                .{ .cmps,  .np, &.{ .m8,   .m8   }, &.{ 0xa6 }, 0, .none,  .none },
                .{ .cmps,  .np, &.{ .m16,  .m16  }, &.{ 0xa7 }, 0, .short, .none },
                .{ .cmps,  .np, &.{ .m32,  .m32  }, &.{ 0xa7 }, 0, .none,  .none },
                .{ .cmps,  .np, &.{ .m64,  .m64  }, &.{ 0xa7 }, 0, .long,  .none },

                .{ .cmpsb, .np, &.{}, &.{ 0xa6 }, 0, .none,  .none },
                .{ .cmpsw, .np, &.{}, &.{ 0xa7 }, 0, .short, .none },
                .{ .cmpsd, .np, &.{}, &.{ 0xa7 }, 0, .none,  .none },
                .{ .cmpsq, .np, &.{}, &.{ 0xa7 }, 0, .long,  .none },

                .{ .cmpxchg, .mr, &.{ .rm8,  .r8  }, &.{ 0x0f, 0xb0 }, 0, .none,  .none },
                .{ .cmpxchg, .mr, &.{ .rm8,  .r8  }, &.{ 0x0f, 0xb0 }, 0, .rex,   .none },
                .{ .cmpxchg, .mr, &.{ .rm16, .r16 }, &.{ 0x0f, 0xb1 }, 0, .short, .none },
                .{ .cmpxchg, .mr, &.{ .rm32, .r32 }, &.{ 0x0f, 0xb1 }, 0, .none,  .none },
                .{ .cmpxchg, .mr, &.{ .rm64, .r64 }, &.{ 0x0f, 0xb1 }, 0, .long,  .none },

                .{ .cmpxchg8b,  .m, &.{ .m64  }, &.{ 0x0f, 0xc7 }, 1, .none, .none },
                .{ .cmpxchg16b, .m, &.{ .m128 }, &.{ 0x0f, 0xc7 }, 1, .long, .none },

                .{ .div, .m, &.{ .rm8  }, &.{ 0xf6 }, 6, .none,  .none },
                .{ .div, .m, &.{ .rm8  }, &.{ 0xf6 }, 6, .rex,   .none },
                .{ .div, .m, &.{ .rm16 }, &.{ 0xf7 }, 6, .short, .none },
                .{ .div, .m, &.{ .rm32 }, &.{ 0xf7 }, 6, .none,  .none },
                .{ .div, .m, &.{ .rm64 }, &.{ 0xf7 }, 6, .long,  .none },

                .{ .endbr64, .np, &.{}, &.{ 0xf3, 0x0f, 0x1e, 0xfa }, 0, .none, .none },

                .{ .hlt, .np, &.{}, &.{ 0xf4 }, 0, .none, .none },

                .{ .idiv, .m, &.{ .rm8  }, &.{ 0xf6 }, 7, .none,  .none },
                .{ .idiv, .m, &.{ .rm8  }, &.{ 0xf6 }, 7, .rex,   .none },
                .{ .idiv, .m, &.{ .rm16 }, &.{ 0xf7 }, 7, .short, .none },
                .{ .idiv, .m, &.{ .rm32 }, &.{ 0xf7 }, 7, .none,  .none },
                .{ .idiv, .m, &.{ .rm64 }, &.{ 0xf7 }, 7, .long,  .none },

                .{ .imul, .m,   &.{ .rm8                 }, &.{ 0xf6       }, 5, .none,  .none },
                .{ .imul, .m,   &.{ .rm8                 }, &.{ 0xf6       }, 5, .rex,   .none },
                .{ .imul, .m,   &.{ .rm16,               }, &.{ 0xf7       }, 5, .short, .none },
                .{ .imul, .m,   &.{ .rm32,               }, &.{ 0xf7       }, 5, .none,  .none },
                .{ .imul, .m,   &.{ .rm64,               }, &.{ 0xf7       }, 5, .long,  .none },
                .{ .imul, .rm,  &.{ .r16,  .rm16,        }, &.{ 0x0f, 0xaf }, 0, .short, .none },
                .{ .imul, .rm,  &.{ .r32,  .rm32,        }, &.{ 0x0f, 0xaf }, 0, .none,  .none },
                .{ .imul, .rm,  &.{ .r64,  .rm64,        }, &.{ 0x0f, 0xaf }, 0, .long,  .none },
                .{ .imul, .rmi, &.{ .r16,  .rm16, .imm8s }, &.{ 0x6b       }, 0, .short, .none },
                .{ .imul, .rmi, &.{ .r32,  .rm32, .imm8s }, &.{ 0x6b       }, 0, .none,  .none },
                .{ .imul, .rmi, &.{ .r64,  .rm64, .imm8s }, &.{ 0x6b       }, 0, .long,  .none },
                .{ .imul, .rmi, &.{ .r16,  .rm16, .imm16 }, &.{ 0x69       }, 0, .short, .none },
                .{ .imul, .rmi, &.{ .r32,  .rm32, .imm32 }, &.{ 0x69       }, 0, .none,  .none },
                .{ .imul, .rmi, &.{ .r64,  .rm64, .imm32 }, &.{ 0x69       }, 0, .long,  .none },

                .{ .int3, .np, &.{}, &.{ 0xcc }, 0, .none, .none },

                .{ .ja,    .d, &.{ .rel32 }, &.{ 0x0f, 0x87 }, 0, .none, .none },
                .{ .jae,   .d, &.{ .rel32 }, &.{ 0x0f, 0x83 }, 0, .none, .none },
                .{ .jb,    .d, &.{ .rel32 }, &.{ 0x0f, 0x82 }, 0, .none, .none },
                .{ .jbe,   .d, &.{ .rel32 }, &.{ 0x0f, 0x86 }, 0, .none, .none },
                .{ .jc,    .d, &.{ .rel32 }, &.{ 0x0f, 0x82 }, 0, .none, .none },
                .{ .jrcxz, .d, &.{ .rel32 }, &.{ 0xe3       }, 0, .none, .none },
                .{ .je,    .d, &.{ .rel32 }, &.{ 0x0f, 0x84 }, 0, .none, .none },
                .{ .jg,    .d, &.{ .rel32 }, &.{ 0x0f, 0x8f }, 0, .none, .none },
                .{ .jge,   .d, &.{ .rel32 }, &.{ 0x0f, 0x8d }, 0, .none, .none },
                .{ .jl,    .d, &.{ .rel32 }, &.{ 0x0f, 0x8c }, 0, .none, .none },
                .{ .jle,   .d, &.{ .rel32 }, &.{ 0x0f, 0x8e }, 0, .none, .none },
                .{ .jna,   .d, &.{ .rel32 }, &.{ 0x0f, 0x86 }, 0, .none, .none },
                .{ .jnae,  .d, &.{ .rel32 }, &.{ 0x0f, 0x82 }, 0, .none, .none },
                .{ .jnb,   .d, &.{ .rel32 }, &.{ 0x0f, 0x83 }, 0, .none, .none },
                .{ .jnbe,  .d, &.{ .rel32 }, &.{ 0x0f, 0x87 }, 0, .none, .none },
                .{ .jnc,   .d, &.{ .rel32 }, &.{ 0x0f, 0x83 }, 0, .none, .none },
                .{ .jne,   .d, &.{ .rel32 }, &.{ 0x0f, 0x85 }, 0, .none, .none },
                .{ .jng,   .d, &.{ .rel32 }, &.{ 0x0f, 0x8e }, 0, .none, .none },
                .{ .jnge,  .d, &.{ .rel32 }, &.{ 0x0f, 0x8c }, 0, .none, .none },
                .{ .jnl,   .d, &.{ .rel32 }, &.{ 0x0f, 0x8d }, 0, .none, .none },
                .{ .jnle,  .d, &.{ .rel32 }, &.{ 0x0f, 0x8f }, 0, .none, .none },
                .{ .jno,   .d, &.{ .rel32 }, &.{ 0x0f, 0x81 }, 0, .none, .none },
                .{ .jnp,   .d, &.{ .rel32 }, &.{ 0x0f, 0x8b }, 0, .none, .none },
                .{ .jns,   .d, &.{ .rel32 }, &.{ 0x0f, 0x89 }, 0, .none, .none },
                .{ .jnz,   .d, &.{ .rel32 }, &.{ 0x0f, 0x85 }, 0, .none, .none },
                .{ .jo,    .d, &.{ .rel32 }, &.{ 0x0f, 0x80 }, 0, .none, .none },
                .{ .jp,    .d, &.{ .rel32 }, &.{ 0x0f, 0x8a }, 0, .none, .none },
                .{ .jpe,   .d, &.{ .rel32 }, &.{ 0x0f, 0x8a }, 0, .none, .none },
                .{ .jpo,   .d, &.{ .rel32 }, &.{ 0x0f, 0x8b }, 0, .none, .none },
                .{ .js,    .d, &.{ .rel32 }, &.{ 0x0f, 0x88 }, 0, .none, .none },
                .{ .jz,    .d, &.{ .rel32 }, &.{ 0x0f, 0x84 }, 0, .none, .none },

                .{ .jmp, .d, &.{ .rel32 }, &.{ 0xe9 }, 0, .none, .none },
                .{ .jmp, .m, &.{ .rm64  }, &.{ 0xff }, 4, .none, .none },

                .{ .lea, .rm, &.{ .r16, .m }, &.{ 0x8d }, 0, .short, .none },
                .{ .lea, .rm, &.{ .r32, .m }, &.{ 0x8d }, 0, .none,  .none },
                .{ .lea, .rm, &.{ .r64, .m }, &.{ 0x8d }, 0, .long,  .none },

                .{ .leave, .np, &.{}, &.{ 0xc9 }, 0, .none, .none },

                .{ .lfence, .np, &.{}, &.{ 0x0f, 0xae, 0xe8 }, 0, .none, .none },

                .{ .lods,  .np, &.{ .m8  }, &.{ 0xac }, 0, .none,  .none },
                .{ .lods,  .np, &.{ .m16 }, &.{ 0xad }, 0, .short, .none },
                .{ .lods,  .np, &.{ .m32 }, &.{ 0xad }, 0, .none,  .none },
                .{ .lods,  .np, &.{ .m64 }, &.{ 0xad }, 0, .long,  .none },

                .{ .lodsb, .np, &.{}, &.{ 0xac }, 0, .none,  .none },
                .{ .lodsw, .np, &.{}, &.{ 0xad }, 0, .short, .none },
                .{ .lodsd, .np, &.{}, &.{ 0xad }, 0, .none,  .none },
                .{ .lodsq, .np, &.{}, &.{ 0xad }, 0, .long,  .none },

                .{ .lzcnt, .rm, &.{ .r16, .rm16 }, &.{ 0xf3, 0x0f, 0xbd }, 0, .short, .lzcnt },
                .{ .lzcnt, .rm, &.{ .r32, .rm32 }, &.{ 0xf3, 0x0f, 0xbd }, 0, .none,  .lzcnt },
                .{ .lzcnt, .rm, &.{ .r64, .rm64 }, &.{ 0xf3, 0x0f, 0xbd }, 0, .long,  .lzcnt },

                .{ .mfence, .np, &.{}, &.{ 0x0f, 0xae, 0xf0 }, 0, .none, .none },

                .{ .mov, .mr, &.{ .rm8,     .r8      }, &.{ 0x88 }, 0, .none,  .none },
                .{ .mov, .mr, &.{ .rm8,     .r8      }, &.{ 0x88 }, 0, .rex,   .none },
                .{ .mov, .mr, &.{ .rm16,    .r16     }, &.{ 0x89 }, 0, .short, .none },
                .{ .mov, .mr, &.{ .rm32,    .r32     }, &.{ 0x89 }, 0, .none,  .none },
                .{ .mov, .mr, &.{ .rm64,    .r64     }, &.{ 0x89 }, 0, .long,  .none },
                .{ .mov, .rm, &.{ .r8,      .rm8     }, &.{ 0x8a }, 0, .none,  .none },
                .{ .mov, .rm, &.{ .r8,      .rm8     }, &.{ 0x8a }, 0, .rex,   .none },
                .{ .mov, .rm, &.{ .r16,     .rm16    }, &.{ 0x8b }, 0, .short, .none },
                .{ .mov, .rm, &.{ .r32,     .rm32    }, &.{ 0x8b }, 0, .none,  .none },
                .{ .mov, .rm, &.{ .r64,     .rm64    }, &.{ 0x8b }, 0, .long,  .none },
                .{ .mov, .mr, &.{ .rm16,    .sreg    }, &.{ 0x8c }, 0, .short, .none },
                .{ .mov, .mr, &.{ .r32_m16, .sreg    }, &.{ 0x8c }, 0, .none,  .none },
                .{ .mov, .mr, &.{ .r64_m16, .sreg    }, &.{ 0x8c }, 0, .long,  .none },
                .{ .mov, .rm, &.{ .sreg,    .rm16    }, &.{ 0x8e }, 0, .short, .none },
                .{ .mov, .rm, &.{ .sreg,    .r32_m16 }, &.{ 0x8e }, 0, .none,  .none },
                .{ .mov, .rm, &.{ .sreg,    .r64_m16 }, &.{ 0x8e }, 0, .long,  .none },
                .{ .mov, .fd, &.{ .al,      .moffs   }, &.{ 0xa0 }, 0, .none,  .none },
                .{ .mov, .fd, &.{ .ax,      .moffs   }, &.{ 0xa1 }, 0, .short, .none },
                .{ .mov, .fd, &.{ .eax,     .moffs   }, &.{ 0xa1 }, 0, .none,  .none },
                .{ .mov, .fd, &.{ .rax,     .moffs   }, &.{ 0xa1 }, 0, .long,  .none },
                .{ .mov, .td, &.{ .moffs,   .al      }, &.{ 0xa2 }, 0, .none,  .none },
                .{ .mov, .td, &.{ .moffs,   .ax      }, &.{ 0xa3 }, 0, .short, .none },
                .{ .mov, .td, &.{ .moffs,   .eax     }, &.{ 0xa3 }, 0, .none,  .none },
                .{ .mov, .td, &.{ .moffs,   .rax     }, &.{ 0xa3 }, 0, .long,  .none },
                .{ .mov, .oi, &.{ .r8,      .imm8    }, &.{ 0xb0 }, 0, .none,  .none },
                .{ .mov, .oi, &.{ .r8,      .imm8    }, &.{ 0xb0 }, 0, .rex,   .none },
                .{ .mov, .oi, &.{ .r16,     .imm16   }, &.{ 0xb8 }, 0, .short, .none },
                .{ .mov, .oi, &.{ .r32,     .imm32   }, &.{ 0xb8 }, 0, .none,  .none },
                .{ .mov, .oi, &.{ .r64,     .imm64   }, &.{ 0xb8 }, 0, .long,  .none },
                .{ .mov, .mi, &.{ .rm8,     .imm8    }, &.{ 0xc6 }, 0, .none,  .none },
                .{ .mov, .mi, &.{ .rm8,     .imm8    }, &.{ 0xc6 }, 0, .rex,   .none },
                .{ .mov, .mi, &.{ .rm16,    .imm16   }, &.{ 0xc7 }, 0, .short, .none },
                .{ .mov, .mi, &.{ .rm32,    .imm32   }, &.{ 0xc7 }, 0, .none,  .none },
                .{ .mov, .mi, &.{ .rm64,    .imm32s  }, &.{ 0xc7 }, 0, .long,  .none },

                .{ .movbe, .rm, &.{ .r16, .m16 }, &.{ 0x0f, 0x38, 0xf0 }, 0, .short, .movbe },
                .{ .movbe, .rm, &.{ .r32, .m32 }, &.{ 0x0f, 0x38, 0xf0 }, 0, .none,  .movbe },
                .{ .movbe, .rm, &.{ .r64, .m64 }, &.{ 0x0f, 0x38, 0xf0 }, 0, .long,  .movbe },
                .{ .movbe, .mr, &.{ .m16, .r16 }, &.{ 0x0f, 0x38, 0xf1 }, 0, .short, .movbe },
                .{ .movbe, .mr, &.{ .m32, .r32 }, &.{ 0x0f, 0x38, 0xf1 }, 0, .none,  .movbe },
                .{ .movbe, .mr, &.{ .m64, .r64 }, &.{ 0x0f, 0x38, 0xf1 }, 0, .long,  .movbe },

                .{ .movs,  .np, &.{ .m8,  .m8  }, &.{ 0xa4 }, 0, .none,  .none },
                .{ .movs,  .np, &.{ .m16, .m16 }, &.{ 0xa5 }, 0, .short, .none },
                .{ .movs,  .np, &.{ .m32, .m32 }, &.{ 0xa5 }, 0, .none,  .none },
                .{ .movs,  .np, &.{ .m64, .m64 }, &.{ 0xa5 }, 0, .long,  .none },

                .{ .movsb, .np, &.{}, &.{ 0xa4 }, 0, .none,  .none },
                .{ .movsw, .np, &.{}, &.{ 0xa5 }, 0, .short, .none },
                .{ .movsd, .np, &.{}, &.{ 0xa5 }, 0, .none,  .none },
                .{ .movsq, .np, &.{}, &.{ 0xa5 }, 0, .long,  .none },

                .{ .movsx, .rm, &.{ .r16, .rm8  }, &.{ 0x0f, 0xbe }, 0, .short,     .none },
                .{ .movsx, .rm, &.{ .r16, .rm8  }, &.{ 0x0f, 0xbe }, 0, .rex_short, .none },
                .{ .movsx, .rm, &.{ .r32, .rm8  }, &.{ 0x0f, 0xbe }, 0, .none,      .none },
                .{ .movsx, .rm, &.{ .r32, .rm8  }, &.{ 0x0f, 0xbe }, 0, .rex,       .none },
                .{ .movsx, .rm, &.{ .r64, .rm8  }, &.{ 0x0f, 0xbe }, 0, .long,      .none },
                .{ .movsx, .rm, &.{ .r32, .rm16 }, &.{ 0x0f, 0xbf }, 0, .none,      .none },
                .{ .movsx, .rm, &.{ .r32, .rm16 }, &.{ 0x0f, 0xbf }, 0, .rex,       .none },
                .{ .movsx, .rm, &.{ .r64, .rm16 }, &.{ 0x0f, 0xbf }, 0, .long,      .none },

                // This instruction is discouraged.
                .{ .movsxd, .rm, &.{ .r32, .rm32 }, &.{ 0x63 }, 0, .none, .none },
                .{ .movsxd, .rm, &.{ .r64, .rm32 }, &.{ 0x63 }, 0, .long, .none },

                .{ .movzx, .rm, &.{ .r16, .rm8  }, &.{ 0x0f, 0xb6 }, 0, .short,     .none },
                .{ .movzx, .rm, &.{ .r16, .rm8  }, &.{ 0x0f, 0xb6 }, 0, .rex_short, .none },
                .{ .movzx, .rm, &.{ .r32, .rm8  }, &.{ 0x0f, 0xb6 }, 0, .none,      .none },
                .{ .movzx, .rm, &.{ .r32, .rm8  }, &.{ 0x0f, 0xb6 }, 0, .rex,       .none },
                .{ .movzx, .rm, &.{ .r64, .rm8  }, &.{ 0x0f, 0xb6 }, 0, .long,      .none },
                .{ .movzx, .rm, &.{ .r32, .rm16 }, &.{ 0x0f, 0xb7 }, 0, .none,      .none },
                .{ .movzx, .rm, &.{ .r32, .rm16 }, &.{ 0x0f, 0xb7 }, 0, .rex,       .none },
                .{ .movzx, .rm, &.{ .r64, .rm16 }, &.{ 0x0f, 0xb7 }, 0, .long,      .none },

                .{ .mul, .m, &.{ .rm8  }, &.{ 0xf6 }, 4, .none,  .none },
                .{ .mul, .m, &.{ .rm8  }, &.{ 0xf6 }, 4, .rex,   .none },
                .{ .mul, .m, &.{ .rm16 }, &.{ 0xf7 }, 4, .short, .none },
                .{ .mul, .m, &.{ .rm32 }, &.{ 0xf7 }, 4, .none,  .none },
                .{ .mul, .m, &.{ .rm64 }, &.{ 0xf7 }, 4, .long,  .none },

                .{ .neg, .m, &.{ .rm8  }, &.{ 0xf6 }, 3, .none,  .none },
                .{ .neg, .m, &.{ .rm8  }, &.{ 0xf6 }, 3, .rex,   .none },
                .{ .neg, .m, &.{ .rm16 }, &.{ 0xf7 }, 3, .short, .none },
                .{ .neg, .m, &.{ .rm32 }, &.{ 0xf7 }, 3, .none,  .none },
                .{ .neg, .m, &.{ .rm64 }, &.{ 0xf7 }, 3, .long,  .none },

                .{ .nop, .np, &.{}, &.{ 0x90 }, 0, .none, .none },

                .{ .not, .m, &.{ .rm8  }, &.{ 0xf6 }, 2, .none,  .none },
                .{ .not, .m, &.{ .rm8  }, &.{ 0xf6 }, 2, .rex,   .none },
                .{ .not, .m, &.{ .rm16 }, &.{ 0xf7 }, 2, .short, .none },
                .{ .not, .m, &.{ .rm32 }, &.{ 0xf7 }, 2, .none,  .none },
                .{ .not, .m, &.{ .rm64 }, &.{ 0xf7 }, 2, .long,  .none },

                .{ .@"or", .zi, &.{ .al,   .imm8   }, &.{ 0x0c }, 0, .none,  .none },
                .{ .@"or", .zi, &.{ .ax,   .imm16  }, &.{ 0x0d }, 0, .short, .none },
                .{ .@"or", .zi, &.{ .eax,  .imm32  }, &.{ 0x0d }, 0, .none,  .none },
                .{ .@"or", .zi, &.{ .rax,  .imm32s }, &.{ 0x0d }, 0, .long,  .none },
                .{ .@"or", .mi, &.{ .rm8,  .imm8   }, &.{ 0x80 }, 1, .none,  .none },
                .{ .@"or", .mi, &.{ .rm8,  .imm8   }, &.{ 0x80 }, 1, .rex,   .none },
                .{ .@"or", .mi, &.{ .rm16, .imm16  }, &.{ 0x81 }, 1, .short, .none },
                .{ .@"or", .mi, &.{ .rm32, .imm32  }, &.{ 0x81 }, 1, .none,  .none },
                .{ .@"or", .mi, &.{ .rm64, .imm32s }, &.{ 0x81 }, 1, .long,  .none },
                .{ .@"or", .mi, &.{ .rm16, .imm8s  }, &.{ 0x83 }, 1, .short, .none },
                .{ .@"or", .mi, &.{ .rm32, .imm8s  }, &.{ 0x83 }, 1, .none,  .none },
                .{ .@"or", .mi, &.{ .rm64, .imm8s  }, &.{ 0x83 }, 1, .long,  .none },
                .{ .@"or", .mr, &.{ .rm8,  .r8     }, &.{ 0x08 }, 0, .none,  .none },
                .{ .@"or", .mr, &.{ .rm8,  .r8     }, &.{ 0x08 }, 0, .rex,   .none },
                .{ .@"or", .mr, &.{ .rm16, .r16    }, &.{ 0x09 }, 0, .short, .none },
                .{ .@"or", .mr, &.{ .rm32, .r32    }, &.{ 0x09 }, 0, .none,  .none },
                .{ .@"or", .mr, &.{ .rm64, .r64    }, &.{ 0x09 }, 0, .long,  .none },
                .{ .@"or", .rm, &.{ .r8,   .rm8    }, &.{ 0x0a }, 0, .none,  .none },
                .{ .@"or", .rm, &.{ .r8,   .rm8    }, &.{ 0x0a }, 0, .rex,   .none },
                .{ .@"or", .rm, &.{ .r16,  .rm16   }, &.{ 0x0b }, 0, .short, .none },
                .{ .@"or", .rm, &.{ .r32,  .rm32   }, &.{ 0x0b }, 0, .none,  .none },
                .{ .@"or", .rm, &.{ .r64,  .rm64   }, &.{ 0x0b }, 0, .long,  .none },

                .{ .pop, .o, &.{ .r16  }, &.{ 0x58 }, 0, .short, .none },
                .{ .pop, .o, &.{ .r64  }, &.{ 0x58 }, 0, .none,  .none },
                .{ .pop, .m, &.{ .rm16 }, &.{ 0x8f }, 0, .short, .none },
                .{ .pop, .m, &.{ .rm64 }, &.{ 0x8f }, 0, .none,  .none },

                .{ .popcnt, .rm, &.{ .r16, .rm16 }, &.{ 0xf3, 0x0f, 0xb8 }, 0, .short, .popcnt },
                .{ .popcnt, .rm, &.{ .r32, .rm32 }, &.{ 0xf3, 0x0f, 0xb8 }, 0, .none,  .popcnt },
                .{ .popcnt, .rm, &.{ .r64, .rm64 }, &.{ 0xf3, 0x0f, 0xb8 }, 0, .long,  .popcnt },

                .{ .push, .o, &.{ .r16   }, &.{ 0x50 }, 0, .short, .none },
                .{ .push, .o, &.{ .r64   }, &.{ 0x50 }, 0, .none,  .none },
                .{ .push, .m, &.{ .rm16  }, &.{ 0xff }, 6, .short, .none },
                .{ .push, .m, &.{ .rm64  }, &.{ 0xff }, 6, .none,  .none },
                .{ .push, .i, &.{ .imm8  }, &.{ 0x6a }, 0, .none,  .none },
                .{ .push, .i, &.{ .imm16 }, &.{ 0x68 }, 0, .short, .none },
                .{ .push, .i, &.{ .imm32 }, &.{ 0x68 }, 0, .none,  .none },

                .{ .ret, .np, &.{}, &.{ 0xc3 }, 0, .none, .none },

                .{ .rcl, .m1, &.{ .rm8,  .unity }, &.{ 0xd0 }, 2, .none,  .none },
                .{ .rcl, .m1, &.{ .rm8,  .unity }, &.{ 0xd0 }, 2, .rex,   .none },
                .{ .rcl, .mc, &.{ .rm8,  .cl    }, &.{ 0xd2 }, 2, .none,  .none },
                .{ .rcl, .mc, &.{ .rm8,  .cl    }, &.{ 0xd2 }, 2, .rex,   .none },
                .{ .rcl, .mi, &.{ .rm8,  .imm8  }, &.{ 0xc0 }, 2, .none,  .none },
                .{ .rcl, .mi, &.{ .rm8,  .imm8  }, &.{ 0xc0 }, 2, .rex,   .none },
                .{ .rcl, .m1, &.{ .rm16, .unity }, &.{ 0xd1 }, 2, .short, .none },
                .{ .rcl, .mc, &.{ .rm16, .cl    }, &.{ 0xd3 }, 2, .short, .none },
                .{ .rcl, .mi, &.{ .rm16, .imm8  }, &.{ 0xc1 }, 2, .short, .none },
                .{ .rcl, .m1, &.{ .rm32, .unity }, &.{ 0xd1 }, 2, .none,  .none },
                .{ .rcl, .m1, &.{ .rm64, .unity }, &.{ 0xd1 }, 2, .long,  .none },
                .{ .rcl, .mc, &.{ .rm32, .cl    }, &.{ 0xd3 }, 2, .none,  .none },
                .{ .rcl, .mc, &.{ .rm64, .cl    }, &.{ 0xd3 }, 2, .long,  .none },
                .{ .rcl, .mi, &.{ .rm32, .imm8  }, &.{ 0xc1 }, 2, .none,  .none },
                .{ .rcl, .mi, &.{ .rm64, .imm8  }, &.{ 0xc1 }, 2, .long,  .none },

                .{ .rcr, .m1, &.{ .rm8,  .unity }, &.{ 0xd0 }, 3, .none,  .none },
                .{ .rcr, .m1, &.{ .rm8,  .unity }, &.{ 0xd0 }, 3, .rex,   .none },
                .{ .rcr, .mc, &.{ .rm8,  .cl    }, &.{ 0xd2 }, 3, .none,  .none },
                .{ .rcr, .mc, &.{ .rm8,  .cl    }, &.{ 0xd2 }, 3, .rex,   .none },
                .{ .rcr, .mi, &.{ .rm8,  .imm8  }, &.{ 0xc0 }, 3, .none,  .none },
                .{ .rcr, .mi, &.{ .rm8,  .imm8  }, &.{ 0xc0 }, 3, .rex,   .none },
                .{ .rcr, .m1, &.{ .rm16, .unity }, &.{ 0xd1 }, 3, .short, .none },
                .{ .rcr, .mc, &.{ .rm16, .cl    }, &.{ 0xd3 }, 3, .short, .none },
                .{ .rcr, .mi, &.{ .rm16, .imm8  }, &.{ 0xc1 }, 3, .short, .none },
                .{ .rcr, .m1, &.{ .rm32, .unity }, &.{ 0xd1 }, 3, .none,  .none },
                .{ .rcr, .m1, &.{ .rm64, .unity }, &.{ 0xd1 }, 3, .long,  .none },
                .{ .rcr, .mc, &.{ .rm32, .cl    }, &.{ 0xd3 }, 3, .none,  .none },
                .{ .rcr, .mc, &.{ .rm64, .cl    }, &.{ 0xd3 }, 3, .long,  .none },
                .{ .rcr, .mi, &.{ .rm32, .imm8  }, &.{ 0xc1 }, 3, .none,  .none },
                .{ .rcr, .mi, &.{ .rm64, .imm8  }, &.{ 0xc1 }, 3, .long,  .none },

                .{ .rol, .m1, &.{ .rm8,  .unity }, &.{ 0xd0 }, 0, .none,  .none },
                .{ .rol, .m1, &.{ .rm8,  .unity }, &.{ 0xd0 }, 0, .rex,   .none },
                .{ .rol, .mc, &.{ .rm8,  .cl    }, &.{ 0xd2 }, 0, .none,  .none },
                .{ .rol, .mc, &.{ .rm8,  .cl    }, &.{ 0xd2 }, 0, .rex,   .none },
                .{ .rol, .mi, &.{ .rm8,  .imm8  }, &.{ 0xc0 }, 0, .none,  .none },
                .{ .rol, .mi, &.{ .rm8,  .imm8  }, &.{ 0xc0 }, 0, .rex,   .none },
                .{ .rol, .m1, &.{ .rm16, .unity }, &.{ 0xd1 }, 0, .short, .none },
                .{ .rol, .mc, &.{ .rm16, .cl    }, &.{ 0xd3 }, 0, .short, .none },
                .{ .rol, .mi, &.{ .rm16, .imm8  }, &.{ 0xc1 }, 0, .short, .none },
                .{ .rol, .m1, &.{ .rm32, .unity }, &.{ 0xd1 }, 0, .none,  .none },
                .{ .rol, .m1, &.{ .rm64, .unity }, &.{ 0xd1 }, 0, .long,  .none },
                .{ .rol, .mc, &.{ .rm32, .cl    }, &.{ 0xd3 }, 0, .none,  .none },
                .{ .rol, .mc, &.{ .rm64, .cl    }, &.{ 0xd3 }, 0, .long,  .none },
                .{ .rol, .mi, &.{ .rm32, .imm8  }, &.{ 0xc1 }, 0, .none,  .none },
                .{ .rol, .mi, &.{ .rm64, .imm8  }, &.{ 0xc1 }, 0, .long,  .none },

                .{ .ror, .m1, &.{ .rm8,  .unity }, &.{ 0xd0 }, 1, .none,  .none },
                .{ .ror, .m1, &.{ .rm8,  .unity }, &.{ 0xd0 }, 1, .rex,   .none },
                .{ .ror, .mc, &.{ .rm8,  .cl    }, &.{ 0xd2 }, 1, .none,  .none },
                .{ .ror, .mc, &.{ .rm8,  .cl    }, &.{ 0xd2 }, 1, .rex,   .none },
                .{ .ror, .mi, &.{ .rm8,  .imm8  }, &.{ 0xc0 }, 1, .none,  .none },
                .{ .ror, .mi, &.{ .rm8,  .imm8  }, &.{ 0xc0 }, 1, .rex,   .none },
                .{ .ror, .m1, &.{ .rm16, .unity }, &.{ 0xd1 }, 1, .short, .none },
                .{ .ror, .mc, &.{ .rm16, .cl    }, &.{ 0xd3 }, 1, .short, .none },
                .{ .ror, .mi, &.{ .rm16, .imm8  }, &.{ 0xc1 }, 1, .short, .none },
                .{ .ror, .m1, &.{ .rm32, .unity }, &.{ 0xd1 }, 1, .none,  .none },
                .{ .ror, .m1, &.{ .rm64, .unity }, &.{ 0xd1 }, 1, .long,  .none },
                .{ .ror, .mc, &.{ .rm32, .cl    }, &.{ 0xd3 }, 1, .none,  .none },
                .{ .ror, .mc, &.{ .rm64, .cl    }, &.{ 0xd3 }, 1, .long,  .none },
                .{ .ror, .mi, &.{ .rm32, .imm8  }, &.{ 0xc1 }, 1, .none,  .none },
                .{ .ror, .mi, &.{ .rm64, .imm8  }, &.{ 0xc1 }, 1, .long,  .none },

                .{ .sal, .m1, &.{ .rm8,  .unity }, &.{ 0xd0 }, 4, .none,  .none },
                .{ .sal, .m1, &.{ .rm8,  .unity }, &.{ 0xd0 }, 4, .rex,   .none },
                .{ .sal, .m1, &.{ .rm16, .unity }, &.{ 0xd1 }, 4, .short, .none },
                .{ .sal, .m1, &.{ .rm32, .unity }, &.{ 0xd1 }, 4, .none,  .none },
                .{ .sal, .m1, &.{ .rm64, .unity }, &.{ 0xd1 }, 4, .long,  .none },
                .{ .sal, .mc, &.{ .rm8,  .cl    }, &.{ 0xd2 }, 4, .none,  .none },
                .{ .sal, .mc, &.{ .rm8,  .cl    }, &.{ 0xd2 }, 4, .rex,   .none },
                .{ .sal, .mc, &.{ .rm16, .cl    }, &.{ 0xd3 }, 4, .short, .none },
                .{ .sal, .mc, &.{ .rm32, .cl    }, &.{ 0xd3 }, 4, .none,  .none },
                .{ .sal, .mc, &.{ .rm64, .cl    }, &.{ 0xd3 }, 4, .long,  .none },
                .{ .sal, .mi, &.{ .rm8,  .imm8  }, &.{ 0xc0 }, 4, .none,  .none },
                .{ .sal, .mi, &.{ .rm8,  .imm8  }, &.{ 0xc0 }, 4, .rex,   .none },
                .{ .sal, .mi, &.{ .rm16, .imm8  }, &.{ 0xc1 }, 4, .short, .none },
                .{ .sal, .mi, &.{ .rm32, .imm8  }, &.{ 0xc1 }, 4, .none,  .none },
                .{ .sal, .mi, &.{ .rm64, .imm8  }, &.{ 0xc1 }, 4, .long,  .none },

                .{ .sar, .m1, &.{ .rm8,  .unity }, &.{ 0xd0 }, 7, .none,  .none },
                .{ .sar, .m1, &.{ .rm8,  .unity }, &.{ 0xd0 }, 7, .rex,   .none },
                .{ .sar, .m1, &.{ .rm16, .unity }, &.{ 0xd1 }, 7, .short, .none },
                .{ .sar, .m1, &.{ .rm32, .unity }, &.{ 0xd1 }, 7, .none,  .none },
                .{ .sar, .m1, &.{ .rm64, .unity }, &.{ 0xd1 }, 7, .long,  .none },
                .{ .sar, .mc, &.{ .rm8,  .cl    }, &.{ 0xd2 }, 7, .none,  .none },
                .{ .sar, .mc, &.{ .rm8,  .cl    }, &.{ 0xd2 }, 7, .rex,   .none },
                .{ .sar, .mc, &.{ .rm16, .cl    }, &.{ 0xd3 }, 7, .short, .none },
                .{ .sar, .mc, &.{ .rm32, .cl    }, &.{ 0xd3 }, 7, .none,  .none },
                .{ .sar, .mc, &.{ .rm64, .cl    }, &.{ 0xd3 }, 7, .long,  .none },
                .{ .sar, .mi, &.{ .rm8,  .imm8  }, &.{ 0xc0 }, 7, .none,  .none },
                .{ .sar, .mi, &.{ .rm8,  .imm8  }, &.{ 0xc0 }, 7, .rex,   .none },
                .{ .sar, .mi, &.{ .rm16, .imm8  }, &.{ 0xc1 }, 7, .short, .none },
                .{ .sar, .mi, &.{ .rm32, .imm8  }, &.{ 0xc1 }, 7, .none,  .none },
                .{ .sar, .mi, &.{ .rm64, .imm8  }, &.{ 0xc1 }, 7, .long,  .none },

                .{ .sbb, .zi, &.{ .al,   .imm8   }, &.{ 0x1c }, 0, .none,  .none },
                .{ .sbb, .zi, &.{ .ax,   .imm16  }, &.{ 0x1d }, 0, .short, .none },
                .{ .sbb, .zi, &.{ .eax,  .imm32  }, &.{ 0x1d }, 0, .none,  .none },
                .{ .sbb, .zi, &.{ .rax,  .imm32s }, &.{ 0x1d }, 0, .long,  .none },
                .{ .sbb, .mi, &.{ .rm8,  .imm8   }, &.{ 0x80 }, 3, .none,  .none },
                .{ .sbb, .mi, &.{ .rm8,  .imm8   }, &.{ 0x80 }, 3, .rex,   .none },
                .{ .sbb, .mi, &.{ .rm16, .imm16  }, &.{ 0x81 }, 3, .short, .none },
                .{ .sbb, .mi, &.{ .rm32, .imm32  }, &.{ 0x81 }, 3, .none,  .none },
                .{ .sbb, .mi, &.{ .rm64, .imm32s }, &.{ 0x81 }, 3, .long,  .none },
                .{ .sbb, .mi, &.{ .rm16, .imm8s  }, &.{ 0x83 }, 3, .short, .none },
                .{ .sbb, .mi, &.{ .rm32, .imm8s  }, &.{ 0x83 }, 3, .none,  .none },
                .{ .sbb, .mi, &.{ .rm64, .imm8s  }, &.{ 0x83 }, 3, .long,  .none },
                .{ .sbb, .mr, &.{ .rm8,  .r8     }, &.{ 0x18 }, 0, .none,  .none },
                .{ .sbb, .mr, &.{ .rm8,  .r8     }, &.{ 0x18 }, 0, .rex,   .none },
                .{ .sbb, .mr, &.{ .rm16, .r16    }, &.{ 0x19 }, 0, .short, .none },
                .{ .sbb, .mr, &.{ .rm32, .r32    }, &.{ 0x19 }, 0, .none,  .none },
                .{ .sbb, .mr, &.{ .rm64, .r64    }, &.{ 0x19 }, 0, .long,  .none },
                .{ .sbb, .rm, &.{ .r8,   .rm8    }, &.{ 0x1a }, 0, .none,  .none },
                .{ .sbb, .rm, &.{ .r8,   .rm8    }, &.{ 0x1a }, 0, .rex,   .none },
                .{ .sbb, .rm, &.{ .r16,  .rm16   }, &.{ 0x1b }, 0, .short, .none },
                .{ .sbb, .rm, &.{ .r32,  .rm32   }, &.{ 0x1b }, 0, .none,  .none },
                .{ .sbb, .rm, &.{ .r64,  .rm64   }, &.{ 0x1b }, 0, .long,  .none },

                .{ .scas,  .np, &.{ .m8  }, &.{ 0xae }, 0, .none,  .none },
                .{ .scas,  .np, &.{ .m16 }, &.{ 0xaf }, 0, .short, .none },
                .{ .scas,  .np, &.{ .m32 }, &.{ 0xaf }, 0, .none,  .none },
                .{ .scas,  .np, &.{ .m64 }, &.{ 0xaf }, 0, .long,  .none },

                .{ .scasb, .np, &.{}, &.{ 0xae }, 0, .none,  .none },
                .{ .scasw, .np, &.{}, &.{ 0xaf }, 0, .short, .none },
                .{ .scasd, .np, &.{}, &.{ 0xaf }, 0, .none,  .none },
                .{ .scasq, .np, &.{}, &.{ 0xaf }, 0, .long,  .none },

                .{ .seta,   .m, &.{ .rm8 }, &.{ 0x0f, 0x97 }, 0, .none, .none },
                .{ .seta,   .m, &.{ .rm8 }, &.{ 0x0f, 0x97 }, 0, .rex,  .none },
                .{ .setae,  .m, &.{ .rm8 }, &.{ 0x0f, 0x93 }, 0, .none, .none },
                .{ .setae,  .m, &.{ .rm8 }, &.{ 0x0f, 0x93 }, 0, .rex,  .none },
                .{ .setb,   .m, &.{ .rm8 }, &.{ 0x0f, 0x92 }, 0, .none, .none },
                .{ .setb,   .m, &.{ .rm8 }, &.{ 0x0f, 0x92 }, 0, .rex,  .none },
                .{ .setbe,  .m, &.{ .rm8 }, &.{ 0x0f, 0x96 }, 0, .none, .none },
                .{ .setbe,  .m, &.{ .rm8 }, &.{ 0x0f, 0x96 }, 0, .rex,  .none },
                .{ .setc,   .m, &.{ .rm8 }, &.{ 0x0f, 0x92 }, 0, .none, .none },
                .{ .setc,   .m, &.{ .rm8 }, &.{ 0x0f, 0x92 }, 0, .rex,  .none },
                .{ .sete,   .m, &.{ .rm8 }, &.{ 0x0f, 0x94 }, 0, .none, .none },
                .{ .sete,   .m, &.{ .rm8 }, &.{ 0x0f, 0x94 }, 0, .rex,  .none },
                .{ .setg,   .m, &.{ .rm8 }, &.{ 0x0f, 0x9f }, 0, .none, .none },
                .{ .setg,   .m, &.{ .rm8 }, &.{ 0x0f, 0x9f }, 0, .rex,  .none },
                .{ .setge,  .m, &.{ .rm8 }, &.{ 0x0f, 0x9d }, 0, .none, .none },
                .{ .setge,  .m, &.{ .rm8 }, &.{ 0x0f, 0x9d }, 0, .rex,  .none },
                .{ .setl,   .m, &.{ .rm8 }, &.{ 0x0f, 0x9c }, 0, .none, .none },
                .{ .setl,   .m, &.{ .rm8 }, &.{ 0x0f, 0x9c }, 0, .rex,  .none },
                .{ .setle,  .m, &.{ .rm8 }, &.{ 0x0f, 0x9e }, 0, .none, .none },
                .{ .setle,  .m, &.{ .rm8 }, &.{ 0x0f, 0x9e }, 0, .rex,  .none },
                .{ .setna,  .m, &.{ .rm8 }, &.{ 0x0f, 0x96 }, 0, .none, .none },
                .{ .setna,  .m, &.{ .rm8 }, &.{ 0x0f, 0x96 }, 0, .rex,  .none },
                .{ .setnae, .m, &.{ .rm8 }, &.{ 0x0f, 0x92 }, 0, .none, .none },
                .{ .setnae, .m, &.{ .rm8 }, &.{ 0x0f, 0x92 }, 0, .rex,  .none },
                .{ .setnb,  .m, &.{ .rm8 }, &.{ 0x0f, 0x93 }, 0, .none, .none },
                .{ .setnb,  .m, &.{ .rm8 }, &.{ 0x0f, 0x93 }, 0, .rex,  .none },
                .{ .setnbe, .m, &.{ .rm8 }, &.{ 0x0f, 0x97 }, 0, .none, .none },
                .{ .setnbe, .m, &.{ .rm8 }, &.{ 0x0f, 0x97 }, 0, .rex,  .none },
                .{ .setnc,  .m, &.{ .rm8 }, &.{ 0x0f, 0x93 }, 0, .none, .none },
                .{ .setnc,  .m, &.{ .rm8 }, &.{ 0x0f, 0x93 }, 0, .rex,  .none },
                .{ .setne,  .m, &.{ .rm8 }, &.{ 0x0f, 0x95 }, 0, .none, .none },
                .{ .setne,  .m, &.{ .rm8 }, &.{ 0x0f, 0x95 }, 0, .rex,  .none },
                .{ .setng,  .m, &.{ .rm8 }, &.{ 0x0f, 0x9e }, 0, .none, .none },
                .{ .setng,  .m, &.{ .rm8 }, &.{ 0x0f, 0x9e }, 0, .rex,  .none },
                .{ .setnge, .m, &.{ .rm8 }, &.{ 0x0f, 0x9c }, 0, .none, .none },
                .{ .setnge, .m, &.{ .rm8 }, &.{ 0x0f, 0x9c }, 0, .rex,  .none },
                .{ .setnl,  .m, &.{ .rm8 }, &.{ 0x0f, 0x9d }, 0, .none, .none },
                .{ .setnl,  .m, &.{ .rm8 }, &.{ 0x0f, 0x9d }, 0, .rex,  .none },
                .{ .setnle, .m, &.{ .rm8 }, &.{ 0x0f, 0x9f }, 0, .none, .none },
                .{ .setnle, .m, &.{ .rm8 }, &.{ 0x0f, 0x9f }, 0, .rex,  .none },
                .{ .setno,  .m, &.{ .rm8 }, &.{ 0x0f, 0x91 }, 0, .none, .none },
                .{ .setno,  .m, &.{ .rm8 }, &.{ 0x0f, 0x91 }, 0, .rex,  .none },
                .{ .setnp,  .m, &.{ .rm8 }, &.{ 0x0f, 0x9b }, 0, .none, .none },
                .{ .setnp,  .m, &.{ .rm8 }, &.{ 0x0f, 0x9b }, 0, .rex,  .none },
                .{ .setns,  .m, &.{ .rm8 }, &.{ 0x0f, 0x99 }, 0, .none, .none },
                .{ .setns,  .m, &.{ .rm8 }, &.{ 0x0f, 0x99 }, 0, .rex,  .none },
                .{ .setnz,  .m, &.{ .rm8 }, &.{ 0x0f, 0x95 }, 0, .none, .none },
                .{ .setnz,  .m, &.{ .rm8 }, &.{ 0x0f, 0x95 }, 0, .rex,  .none },
                .{ .seto,   .m, &.{ .rm8 }, &.{ 0x0f, 0x90 }, 0, .none, .none },
                .{ .seto,   .m, &.{ .rm8 }, &.{ 0x0f, 0x90 }, 0, .rex,  .none },
                .{ .setp,   .m, &.{ .rm8 }, &.{ 0x0f, 0x9a }, 0, .none, .none },
                .{ .setp,   .m, &.{ .rm8 }, &.{ 0x0f, 0x9a }, 0, .rex,  .none },
                .{ .setpe,  .m, &.{ .rm8 }, &.{ 0x0f, 0x9a }, 0, .none, .none },
                .{ .setpe,  .m, &.{ .rm8 }, &.{ 0x0f, 0x9a }, 0, .rex,  .none },
                .{ .setpo,  .m, &.{ .rm8 }, &.{ 0x0f, 0x9b }, 0, .none, .none },
                .{ .setpo,  .m, &.{ .rm8 }, &.{ 0x0f, 0x9b }, 0, .rex,  .none },
                .{ .sets,   .m, &.{ .rm8 }, &.{ 0x0f, 0x98 }, 0, .none, .none },
                .{ .sets,   .m, &.{ .rm8 }, &.{ 0x0f, 0x98 }, 0, .rex,  .none },
                .{ .setz,   .m, &.{ .rm8 }, &.{ 0x0f, 0x94 }, 0, .none, .none },
                .{ .setz,   .m, &.{ .rm8 }, &.{ 0x0f, 0x94 }, 0, .rex,  .none },

                .{ .sfence, .np, &.{}, &.{ 0x0f, 0xae, 0xf8 }, 0, .none, .none },

                .{ .shl, .m1, &.{ .rm8,  .unity }, &.{ 0xd0 }, 4, .none,  .none },
                .{ .shl, .m1, &.{ .rm8,  .unity }, &.{ 0xd0 }, 4, .rex,   .none },
                .{ .shl, .m1, &.{ .rm16, .unity }, &.{ 0xd1 }, 4, .short, .none },
                .{ .shl, .m1, &.{ .rm32, .unity }, &.{ 0xd1 }, 4, .none,  .none },
                .{ .shl, .m1, &.{ .rm64, .unity }, &.{ 0xd1 }, 4, .long,  .none },
                .{ .shl, .mc, &.{ .rm8,  .cl    }, &.{ 0xd2 }, 4, .none,  .none },
                .{ .shl, .mc, &.{ .rm8,  .cl    }, &.{ 0xd2 }, 4, .rex,   .none },
                .{ .shl, .mc, &.{ .rm16, .cl    }, &.{ 0xd3 }, 4, .short, .none },
                .{ .shl, .mc, &.{ .rm32, .cl    }, &.{ 0xd3 }, 4, .none,  .none },
                .{ .shl, .mc, &.{ .rm64, .cl    }, &.{ 0xd3 }, 4, .long,  .none },
                .{ .shl, .mi, &.{ .rm8,  .imm8  }, &.{ 0xc0 }, 4, .none,  .none },
                .{ .shl, .mi, &.{ .rm8,  .imm8  }, &.{ 0xc0 }, 4, .rex,   .none },
                .{ .shl, .mi, &.{ .rm16, .imm8  }, &.{ 0xc1 }, 4, .short, .none },
                .{ .shl, .mi, &.{ .rm32, .imm8  }, &.{ 0xc1 }, 4, .none,  .none },
                .{ .shl, .mi, &.{ .rm64, .imm8  }, &.{ 0xc1 }, 4, .long,  .none },

                .{ .shld, .mri, &.{ .rm16, .r16, .imm8 }, &.{ 0x0f, 0xa4 }, 0, .short, .none },
                .{ .shld, .mrc, &.{ .rm16, .r16, .cl   }, &.{ 0x0f, 0xa5 }, 0, .short, .none },
                .{ .shld, .mri, &.{ .rm32, .r32, .imm8 }, &.{ 0x0f, 0xa4 }, 0, .none,  .none },
                .{ .shld, .mri, &.{ .rm64, .r64, .imm8 }, &.{ 0x0f, 0xa4 }, 0, .long,  .none },
                .{ .shld, .mrc, &.{ .rm32, .r32, .cl   }, &.{ 0x0f, 0xa5 }, 0, .none,  .none },
                .{ .shld, .mrc, &.{ .rm64, .r64, .cl   }, &.{ 0x0f, 0xa5 }, 0, .long,  .none },

                .{ .shr, .m1, &.{ .rm8,  .unity }, &.{ 0xd0 }, 5, .none,  .none },
                .{ .shr, .m1, &.{ .rm8,  .unity }, &.{ 0xd0 }, 5, .rex,   .none },
                .{ .shr, .m1, &.{ .rm16, .unity }, &.{ 0xd1 }, 5, .short, .none },
                .{ .shr, .m1, &.{ .rm32, .unity }, &.{ 0xd1 }, 5, .none,  .none },
                .{ .shr, .m1, &.{ .rm64, .unity }, &.{ 0xd1 }, 5, .long,  .none },
                .{ .shr, .mc, &.{ .rm8,  .cl    }, &.{ 0xd2 }, 5, .none,  .none },
                .{ .shr, .mc, &.{ .rm8,  .cl    }, &.{ 0xd2 }, 5, .rex,   .none },
                .{ .shr, .mc, &.{ .rm16, .cl    }, &.{ 0xd3 }, 5, .short, .none },
                .{ .shr, .mc, &.{ .rm32, .cl    }, &.{ 0xd3 }, 5, .none,  .none },
                .{ .shr, .mc, &.{ .rm64, .cl    }, &.{ 0xd3 }, 5, .long,  .none },
                .{ .shr, .mi, &.{ .rm8,  .imm8  }, &.{ 0xc0 }, 5, .none,  .none },
                .{ .shr, .mi, &.{ .rm8,  .imm8  }, &.{ 0xc0 }, 5, .rex,   .none },
                .{ .shr, .mi, &.{ .rm16, .imm8  }, &.{ 0xc1 }, 5, .short, .none },
                .{ .shr, .mi, &.{ .rm32, .imm8  }, &.{ 0xc1 }, 5, .none,  .none },
                .{ .shr, .mi, &.{ .rm64, .imm8  }, &.{ 0xc1 }, 5, .long,  .none },

                .{ .shrd, .mri, &.{ .rm16, .r16, .imm8 }, &.{ 0x0f, 0xac }, 0, .short, .none },
                .{ .shrd, .mrc, &.{ .rm16, .r16, .cl   }, &.{ 0x0f, 0xad }, 0, .short, .none },
                .{ .shrd, .mri, &.{ .rm32, .r32, .imm8 }, &.{ 0x0f, 0xac }, 0, .none,  .none },
                .{ .shrd, .mri, &.{ .rm64, .r64, .imm8 }, &.{ 0x0f, 0xac }, 0, .long,  .none },
                .{ .shrd, .mrc, &.{ .rm32, .r32, .cl   }, &.{ 0x0f, 0xad }, 0, .none,  .none },
                .{ .shrd, .mrc, &.{ .rm64, .r64, .cl   }, &.{ 0x0f, 0xad }, 0, .long,  .none },

                .{ .stos,  .np, &.{ .m8  }, &.{ 0xaa }, 0, .none,  .none },
                .{ .stos,  .np, &.{ .m16 }, &.{ 0xab }, 0, .short, .none },
                .{ .stos,  .np, &.{ .m32 }, &.{ 0xab }, 0, .none,  .none },
                .{ .stos,  .np, &.{ .m64 }, &.{ 0xab }, 0, .long,  .none },

                .{ .stosb, .np, &.{}, &.{ 0xaa }, 0, .none,  .none },
                .{ .stosw, .np, &.{}, &.{ 0xab }, 0, .short, .none },
                .{ .stosd, .np, &.{}, &.{ 0xab }, 0, .none,  .none },
                .{ .stosq, .np, &.{}, &.{ 0xab }, 0, .long,  .none },

                .{ .sub, .zi, &.{ .al,   .imm8   }, &.{ 0x2c }, 0, .none,  .none },
                .{ .sub, .zi, &.{ .ax,   .imm16  }, &.{ 0x2d }, 0, .short, .none },
                .{ .sub, .zi, &.{ .eax,  .imm32  }, &.{ 0x2d }, 0, .none,  .none },
                .{ .sub, .zi, &.{ .rax,  .imm32s }, &.{ 0x2d }, 0, .long,  .none },
                .{ .sub, .mi, &.{ .rm8,  .imm8   }, &.{ 0x80 }, 5, .none,  .none },
                .{ .sub, .mi, &.{ .rm8,  .imm8   }, &.{ 0x80 }, 5, .rex,   .none },
                .{ .sub, .mi, &.{ .rm16, .imm16  }, &.{ 0x81 }, 5, .short, .none },
                .{ .sub, .mi, &.{ .rm32, .imm32  }, &.{ 0x81 }, 5, .none,  .none },
                .{ .sub, .mi, &.{ .rm64, .imm32s }, &.{ 0x81 }, 5, .long,  .none },
                .{ .sub, .mi, &.{ .rm16, .imm8s  }, &.{ 0x83 }, 5, .short, .none },
                .{ .sub, .mi, &.{ .rm32, .imm8s  }, &.{ 0x83 }, 5, .none,  .none },
                .{ .sub, .mi, &.{ .rm64, .imm8s  }, &.{ 0x83 }, 5, .long,  .none },
                .{ .sub, .mr, &.{ .rm8,  .r8     }, &.{ 0x28 }, 0, .none,  .none },
                .{ .sub, .mr, &.{ .rm8,  .r8     }, &.{ 0x28 }, 0, .rex,   .none },
                .{ .sub, .mr, &.{ .rm16, .r16    }, &.{ 0x29 }, 0, .short, .none },
                .{ .sub, .mr, &.{ .rm32, .r32    }, &.{ 0x29 }, 0, .none,  .none },
                .{ .sub, .mr, &.{ .rm64, .r64    }, &.{ 0x29 }, 0, .long,  .none },
                .{ .sub, .rm, &.{ .r8,   .rm8    }, &.{ 0x2a }, 0, .none,  .none },
                .{ .sub, .rm, &.{ .r8,   .rm8    }, &.{ 0x2a }, 0, .rex,   .none },
                .{ .sub, .rm, &.{ .r16,  .rm16   }, &.{ 0x2b }, 0, .short, .none },
                .{ .sub, .rm, &.{ .r32,  .rm32   }, &.{ 0x2b }, 0, .none,  .none },
                .{ .sub, .rm, &.{ .r64,  .rm64   }, &.{ 0x2b }, 0, .long,  .none },

                .{ .syscall, .np, &.{}, &.{ 0x0f, 0x05 }, 0, .none, .none },

                .{ .@"test", .zi, &.{ .al,   .imm8   }, &.{ 0xa8 }, 0, .none,  .none },
                .{ .@"test", .zi, &.{ .ax,   .imm16  }, &.{ 0xa9 }, 0, .short, .none },
                .{ .@"test", .zi, &.{ .eax,  .imm32  }, &.{ 0xa9 }, 0, .none,  .none },
                .{ .@"test", .zi, &.{ .rax,  .imm32s }, &.{ 0xa9 }, 0, .long,  .none },
                .{ .@"test", .mi, &.{ .rm8,  .imm8   }, &.{ 0xf6 }, 0, .none,  .none },
                .{ .@"test", .mi, &.{ .rm8,  .imm8   }, &.{ 0xf6 }, 0, .rex,   .none },
                .{ .@"test", .mi, &.{ .rm16, .imm16  }, &.{ 0xf7 }, 0, .short, .none },
                .{ .@"test", .mi, &.{ .rm32, .imm32  }, &.{ 0xf7 }, 0, .none,  .none },
                .{ .@"test", .mi, &.{ .rm64, .imm32s }, &.{ 0xf7 }, 0, .long,  .none },
                .{ .@"test", .mr, &.{ .rm8,  .r8     }, &.{ 0x84 }, 0, .none,  .none },
                .{ .@"test", .mr, &.{ .rm8,  .r8     }, &.{ 0x84 }, 0, .rex,   .none },
                .{ .@"test", .mr, &.{ .rm16, .r16    }, &.{ 0x85 }, 0, .short, .none },
                .{ .@"test", .mr, &.{ .rm32, .r32    }, &.{ 0x85 }, 0, .none,  .none },
                .{ .@"test", .mr, &.{ .rm64, .r64    }, &.{ 0x85 }, 0, .long,  .none },

                .{ .tzcnt, .rm, &.{ .r16, .rm16 }, &.{ 0xf3, 0x0f, 0xbc }, 0, .short, .bmi },
                .{ .tzcnt, .rm, &.{ .r32, .rm32 }, &.{ 0xf3, 0x0f, 0xbc }, 0, .none,  .bmi },
                .{ .tzcnt, .rm, &.{ .r64, .rm64 }, &.{ 0xf3, 0x0f, 0xbc }, 0, .long,  .bmi },

                .{ .ud2, .np, &.{}, &.{ 0x0f, 0x0b }, 0, .none, .none },

                .{ .xadd, .mr, &.{ .rm8,  .r8  }, &.{ 0x0f, 0xc0 }, 0, .none,  .none },
                .{ .xadd, .mr, &.{ .rm8,  .r8  }, &.{ 0x0f, 0xc0 }, 0, .rex,   .none },
                .{ .xadd, .mr, &.{ .rm16, .r16 }, &.{ 0x0f, 0xc1 }, 0, .short, .none },
                .{ .xadd, .mr, &.{ .rm32, .r32 }, &.{ 0x0f, 0xc1 }, 0, .none,  .none },
                .{ .xadd, .mr, &.{ .rm64, .r64 }, &.{ 0x0f, 0xc1 }, 0, .long,  .none },

                .{ .xchg, .o,  &.{ .ax,   .r16  }, &.{ 0x90 }, 0, .short, .none },
                .{ .xchg, .o,  &.{ .r16,  .ax   }, &.{ 0x90 }, 0, .short, .none },
                .{ .xchg, .o,  &.{ .eax,  .r32  }, &.{ 0x90 }, 0, .none,  .none },
                .{ .xchg, .o,  &.{ .rax,  .r64  }, &.{ 0x90 }, 0, .long,  .none },
                .{ .xchg, .o,  &.{ .r32,  .eax  }, &.{ 0x90 }, 0, .none,  .none },
                .{ .xchg, .o,  &.{ .r64,  .rax  }, &.{ 0x90 }, 0, .long,  .none },
                .{ .xchg, .mr, &.{ .rm8,  .r8   }, &.{ 0x86 }, 0, .none,  .none },
                .{ .xchg, .mr, &.{ .rm8,  .r8   }, &.{ 0x86 }, 0, .rex,   .none },
                .{ .xchg, .rm, &.{ .r8,   .rm8  }, &.{ 0x86 }, 0, .none,  .none },
                .{ .xchg, .rm, &.{ .r8,   .rm8  }, &.{ 0x86 }, 0, .rex,   .none },
                .{ .xchg, .mr, &.{ .rm16, .r16  }, &.{ 0x87 }, 0, .short, .none },
                .{ .xchg, .rm, &.{ .r16,  .rm16 }, &.{ 0x87 }, 0, .short, .none },
                .{ .xchg, .mr, &.{ .rm32, .r32  }, &.{ 0x87 }, 0, .none,  .none },
                .{ .xchg, .mr, &.{ .rm64, .r64  }, &.{ 0x87 }, 0, .long,  .none },
                .{ .xchg, .rm, &.{ .r32,  .rm32 }, &.{ 0x87 }, 0, .none,  .none },
                .{ .xchg, .rm, &.{ .r64,  .rm64 }, &.{ 0x87 }, 0, .long,  .none },

                .{ .xor, .zi, &.{ .al,   .imm8   }, &.{ 0x34 }, 0, .none,  .none },
                .{ .xor, .zi, &.{ .ax,   .imm16  }, &.{ 0x35 }, 0, .short, .none },
                .{ .xor, .zi, &.{ .eax,  .imm32  }, &.{ 0x35 }, 0, .none,  .none },
                .{ .xor, .zi, &.{ .rax,  .imm32s }, &.{ 0x35 }, 0, .long,  .none },
                .{ .xor, .mi, &.{ .rm8,  .imm8   }, &.{ 0x80 }, 6, .none,  .none },
                .{ .xor, .mi, &.{ .rm8,  .imm8   }, &.{ 0x80 }, 6, .rex,   .none },
                .{ .xor, .mi, &.{ .rm16, .imm16  }, &.{ 0x81 }, 6, .short, .none },
                .{ .xor, .mi, &.{ .rm32, .imm32  }, &.{ 0x81 }, 6, .none,  .none },
                .{ .xor, .mi, &.{ .rm64, .imm32s }, &.{ 0x81 }, 6, .long,  .none },
                .{ .xor, .mi, &.{ .rm16, .imm8s  }, &.{ 0x83 }, 6, .short, .none },
                .{ .xor, .mi, &.{ .rm32, .imm8s  }, &.{ 0x83 }, 6, .none,  .none },
                .{ .xor, .mi, &.{ .rm64, .imm8s  }, &.{ 0x83 }, 6, .long,  .none },
                .{ .xor, .mr, &.{ .rm8,  .r8     }, &.{ 0x30 }, 0, .none,  .none },
                .{ .xor, .mr, &.{ .rm8,  .r8     }, &.{ 0x30 }, 0, .rex,   .none },
                .{ .xor, .mr, &.{ .rm16, .r16    }, &.{ 0x31 }, 0, .short, .none },
                .{ .xor, .mr, &.{ .rm32, .r32    }, &.{ 0x31 }, 0, .none,  .none },
                .{ .xor, .mr, &.{ .rm64, .r64    }, &.{ 0x31 }, 0, .long,  .none },
                .{ .xor, .rm, &.{ .r8,   .rm8    }, &.{ 0x32 }, 0, .none,  .none },
                .{ .xor, .rm, &.{ .r8,   .rm8    }, &.{ 0x32 }, 0, .rex,   .none },
                .{ .xor, .rm, &.{ .r16,  .rm16   }, &.{ 0x33 }, 0, .short, .none },
                .{ .xor, .rm, &.{ .r32,  .rm32   }, &.{ 0x33 }, 0, .none,  .none },
                .{ .xor, .rm, &.{ .r64,  .rm64   }, &.{ 0x33 }, 0, .long,  .none },

                // X87
                .{ .fisttp, .m, &.{ .m16 }, &.{ 0xdf }, 1, .none, .x87 },
                .{ .fisttp, .m, &.{ .m32 }, &.{ 0xdb }, 1, .none, .x87 },
                .{ .fisttp, .m, &.{ .m64 }, &.{ 0xdd }, 1, .none, .x87 },

                .{ .fld, .m, &.{ .m32 }, &.{ 0xd9 }, 0, .none, .x87 },
                .{ .fld, .m, &.{ .m64 }, &.{ 0xdd }, 0, .none, .x87 },
                .{ .fld, .m, &.{ .m80 }, &.{ 0xdb }, 5, .none, .x87 },

                // SSE
                .{ .addps, .rm, &.{ .xmm, .xmm_m128 }, &.{ 0x0f, 0x58 }, 0, .none, .sse },

                .{ .addss, .rm, &.{ .xmm, .xmm_m32 }, &.{ 0xf3, 0x0f, 0x58 }, 0, .none, .sse },

                .{ .andnps, .rm, &.{ .xmm, .xmm_m128 }, &.{ 0x0f, 0x55 }, 0, .none, .sse },

                .{ .andps, .rm, &.{ .xmm, .xmm_m128 }, &.{ 0x0f, 0x54 }, 0, .none, .sse },

                .{ .cmpps, .rmi, &.{ .xmm, .xmm_m128, .imm8 }, &.{ 0x0f, 0xc2 }, 0, .none, .sse },

                .{ .cmpss, .rmi, &.{ .xmm, .xmm_m32, .imm8 }, &.{ 0xf3, 0x0f, 0xc2 }, 0, .none, .sse },

                .{ .cvtpi2ps, .rm, &.{ .xmm, .mm_m64 }, &.{ 0x0f, 0x2a }, 0, .none, .sse },

                .{ .cvtps2pi, .rm, &.{ .mm, .xmm_m64 }, &.{ 0x0f, 0x2d }, 0, .none, .sse },

                .{ .cvtsi2ss, .rm, &.{ .xmm, .rm32 }, &.{ 0xf3, 0x0f, 0x2a }, 0, .none, .sse },
                .{ .cvtsi2ss, .rm, &.{ .xmm, .rm64 }, &.{ 0xf3, 0x0f, 0x2a }, 0, .long, .sse },

                .{ .cvtss2si, .rm, &.{ .r32, .xmm_m32 }, &.{ 0xf3, 0x0f, 0x2d }, 0, .none, .sse },
                .{ .cvtss2si, .rm, &.{ .r64, .xmm_m32 }, &.{ 0xf3, 0x0f, 0x2d }, 0, .long, .sse },

                .{ .cvttps2pi, .rm, &.{ .mm, .xmm_m64 }, &.{ 0x0f, 0x2c }, 0, .none, .sse },

                .{ .cvttss2si, .rm, &.{ .r32, .xmm_m32 }, &.{ 0xf3, 0x0f, 0x2c }, 0, .none, .sse },
                .{ .cvttss2si, .rm, &.{ .r64, .xmm_m32 }, &.{ 0xf3, 0x0f, 0x2c }, 0, .long, .sse },

                .{ .divps, .rm, &.{ .xmm, .xmm_m128 }, &.{ 0x0f, 0x5e }, 0, .none, .sse },

                .{ .divss, .rm, &.{ .xmm, .xmm_m32 }, &.{ 0xf3, 0x0f, 0x5e }, 0, .none, .sse },

                .{ .maxps, .rm, &.{ .xmm, .xmm_m128 }, &.{ 0x0f, 0x5f }, 0, .none, .sse },

                .{ .maxss, .rm, &.{ .xmm, .xmm_m32 }, &.{ 0xf3, 0x0f, 0x5f }, 0, .none, .sse },

                .{ .minps, .rm, &.{ .xmm, .xmm_m128 }, &.{ 0x0f, 0x5d }, 0, .none, .sse },

                .{ .minss, .rm, &.{ .xmm, .xmm_m32 }, &.{ 0xf3, 0x0f, 0x5d }, 0, .none, .sse },

                .{ .movaps, .rm, &.{ .xmm,      .xmm_m128 }, &.{ 0x0f, 0x28 }, 0, .none, .sse },
                .{ .movaps, .mr, &.{ .xmm_m128, .xmm      }, &.{ 0x0f, 0x29 }, 0, .none, .sse },

                .{ .movhlps, .rm, &.{ .xmm, .xmm }, &.{ 0x0f, 0x12 }, 0, .none, .sse },

                .{ .movlhps, .rm, &.{ .xmm, .xmm }, &.{ 0x0f, 0x16 }, 0, .none, .sse },

                .{ .movss, .rm, &.{ .xmm,     .xmm_m32 }, &.{ 0xf3, 0x0f, 0x10 }, 0, .none, .sse },
                .{ .movss, .mr, &.{ .xmm_m32, .xmm     }, &.{ 0xf3, 0x0f, 0x11 }, 0, .none, .sse },

                .{ .movups, .rm, &.{ .xmm,      .xmm_m128 }, &.{ 0x0f, 0x10 }, 0, .none, .sse },
                .{ .movups, .mr, &.{ .xmm_m128, .xmm      }, &.{ 0x0f, 0x11 }, 0, .none, .sse },

                .{ .mulps, .rm, &.{ .xmm, .xmm_m128 }, &.{ 0x0f, 0x59 }, 0, .none, .sse },

                .{ .mulss, .rm, &.{ .xmm, .xmm_m32 }, &.{ 0xf3, 0x0f, 0x59 }, 0, .none, .sse },

                .{ .orps, .rm, &.{ .xmm, .xmm_m128 }, &.{ 0x0f, 0x56 }, 0, .none, .sse },

                .{ .shufps, .rmi, &.{ .xmm, .xmm_m128, .imm8 }, &.{ 0x0f, 0xc6 }, 0, .none, .sse },

                .{ .sqrtps, .rm, &.{ .xmm, .xmm_m128 }, &.{ 0x0f, 0x51 }, 0, .none, .sse },

                .{ .sqrtss, .rm, &.{ .xmm, .xmm_m32 }, &.{ 0xf3, 0x0f, 0x51 }, 0, .none, .sse },

                .{ .subps, .rm, &.{ .xmm, .xmm_m128 }, &.{ 0x0f, 0x5c }, 0, .none, .sse },

                .{ .subss, .rm, &.{ .xmm, .xmm_m32 }, &.{ 0xf3, 0x0f, 0x5c }, 0, .none, .sse },

                .{ .ucomiss, .rm, &.{ .xmm, .xmm_m32 }, &.{ 0x0f, 0x2e }, 0, .none, .sse },

                .{ .xorps, .rm, &.{ .xmm, .xmm_m128 }, &.{ 0x0f, 0x57 }, 0, .none, .sse },

                // SSE2
                .{ .addpd, .rm, &.{ .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0x58 }, 0, .none, .sse2 },

                .{ .addsd, .rm, &.{ .xmm, .xmm_m64 }, &.{ 0xf2, 0x0f, 0x58 }, 0, .none, .sse2 },

                .{ .andnpd, .rm, &.{ .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0x55 }, 0, .none, .sse2 },

                .{ .andpd, .rm, &.{ .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0x54 }, 0, .none, .sse2 },

                .{ .cmppd, .rmi, &.{ .xmm, .xmm_m128, .imm8 }, &.{ 0x66, 0x0f, 0xc2 }, 0, .none, .sse2 },

                .{ .cmpsd, .rmi, &.{ .xmm, .xmm_m64, .imm8 }, &.{ 0xf2, 0x0f, 0xc2 }, 0, .none, .sse2 },

                .{ .cvtdq2pd, .rm, &.{ .xmm, .xmm_m64 }, &.{ 0xf3, 0x0f, 0xe6 }, 0, .none, .sse2 },

                .{ .cvtdq2ps, .rm, &.{ .xmm, .xmm_m128 }, &.{ 0x0f, 0x5b }, 0, .none, .sse2 },

                .{ .cvtpd2dq, .rm, &.{ .xmm, .xmm_m128 }, &.{ 0xf2, 0x0f, 0xe6 }, 0, .none, .sse2 },

                .{ .cvtpd2pi, .rm, &.{ .mm, .xmm_m128 }, &.{ 0x66, 0x0f, 0x2d }, 0, .none, .sse2 },

                .{ .cvtpd2ps, .rm, &.{ .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0x5a }, 0, .none, .sse2 },

                .{ .cvtpi2pd, .rm, &.{ .xmm, .mm_m64 }, &.{ 0x66, 0x0f, 0x2a }, 0, .none, .sse2 },

                .{ .cvtps2dq, .rm, &.{ .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0x5b }, 0, .none, .sse2 },

                .{ .cvtps2pd, .rm, &.{ .xmm, .xmm_m64 }, &.{ 0x0f, 0x5a }, 0, .none, .sse2 },

                .{ .cvtsd2si, .rm, &.{ .r32, .xmm_m64 }, &.{ 0xf2, 0x0f, 0x2d }, 0, .none, .sse2 },
                .{ .cvtsd2si, .rm, &.{ .r64, .xmm_m64 }, &.{ 0xf2, 0x0f, 0x2d }, 0, .long, .sse2 },

                .{ .cvtsd2ss, .rm, &.{ .xmm, .xmm_m64 }, &.{ 0xf2, 0x0f, 0x5a }, 0, .none, .sse2 },

                .{ .cvtsi2sd, .rm, &.{ .xmm, .rm32 }, &.{ 0xf2, 0x0f, 0x2a }, 0, .none, .sse2 },
                .{ .cvtsi2sd, .rm, &.{ .xmm, .rm64 }, &.{ 0xf2, 0x0f, 0x2a }, 0, .long, .sse2 },

                .{ .cvtss2sd, .rm, &.{ .xmm, .xmm_m32 }, &.{ 0xf3, 0x0f, 0x5a }, 0, .none, .sse2 },

                .{ .cvttpd2dq, .rm, &.{ .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0xe6 }, 0, .none, .sse2 },

                .{ .cvttpd2pi, .rm, &.{ .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0x2c }, 0, .none, .sse2 },

                .{ .cvttps2dq, .rm, &.{ .xmm, .xmm_m128 }, &.{ 0xf3, 0x0f, 0x5b }, 0, .none, .sse2 },

                .{ .cvttsd2si, .rm, &.{ .r32, .xmm_m64 }, &.{ 0xf2, 0x0f, 0x2c }, 0, .none, .sse2 },
                .{ .cvttsd2si, .rm, &.{ .r64, .xmm_m64 }, &.{ 0xf2, 0x0f, 0x2c }, 0, .long, .sse2 },

                .{ .divpd, .rm, &.{ .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0x5e }, 0, .none, .sse2 },

                .{ .divsd, .rm, &.{ .xmm, .xmm_m64 }, &.{ 0xf2, 0x0f, 0x5e }, 0, .none, .sse2 },

                .{ .maxpd, .rm, &.{ .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0x5f }, 0, .none, .sse2 },

                .{ .maxsd, .rm, &.{ .xmm, .xmm_m64 }, &.{ 0xf2, 0x0f, 0x5f }, 0, .none, .sse2 },

                .{ .minpd, .rm, &.{ .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0x5d }, 0, .none, .sse2 },

                .{ .minsd, .rm, &.{ .xmm, .xmm_m64 }, &.{ 0xf2, 0x0f, 0x5d }, 0, .none, .sse2 },

                .{ .movapd, .rm, &.{ .xmm,      .xmm_m128 }, &.{ 0x66, 0x0f, 0x28 }, 0, .none, .sse2 },
                .{ .movapd, .mr, &.{ .xmm_m128, .xmm      }, &.{ 0x66, 0x0f, 0x29 }, 0, .none, .sse2 },

                .{ .movd, .rm, &.{ .xmm,  .rm32 }, &.{ 0x66, 0x0f, 0x6e }, 0, .none, .sse2 },
                .{ .movq, .rm, &.{ .xmm,  .rm64 }, &.{ 0x66, 0x0f, 0x6e }, 0, .long, .sse2 },
                .{ .movd, .mr, &.{ .rm32, .xmm  }, &.{ 0x66, 0x0f, 0x7e }, 0, .none, .sse2 },
                .{ .movq, .mr, &.{ .rm64, .xmm  }, &.{ 0x66, 0x0f, 0x7e }, 0, .long, .sse2 },

                .{ .movdqa, .rm, &.{ .xmm,      .xmm_m128 }, &.{ 0x66, 0x0f, 0x6f }, 0, .none, .sse2 },
                .{ .movdqa, .mr, &.{ .xmm_m128, .xmm      }, &.{ 0x66, 0x0f, 0x7f }, 0, .none, .sse2 },

                .{ .movdqu, .rm, &.{ .xmm,      .xmm_m128 }, &.{ 0xf3, 0x0f, 0x6f }, 0, .none, .sse2 },
                .{ .movdqu, .mr, &.{ .xmm_m128, .xmm      }, &.{ 0xf3, 0x0f, 0x7f }, 0, .none, .sse2 },

                .{ .movq, .rm, &.{ .xmm,     .xmm_m64 }, &.{ 0xf3, 0x0f, 0x7e }, 0, .none, .sse2 },
                .{ .movq, .mr, &.{ .xmm_m64, .xmm     }, &.{ 0x66, 0x0f, 0xd6 }, 0, .none, .sse2 },

                .{ .movupd, .rm, &.{ .xmm,      .xmm_m128 }, &.{ 0x66, 0x0f, 0x10 }, 0, .none, .sse2 },
                .{ .movupd, .mr, &.{ .xmm_m128, .xmm      }, &.{ 0x66, 0x0f, 0x11 }, 0, .none, .sse2 },

                .{ .mulpd, .rm, &.{ .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0x59 }, 0, .none, .sse2 },

                .{ .mulsd, .rm, &.{ .xmm, .xmm_m64 }, &.{ 0xf2, 0x0f, 0x59 }, 0, .none, .sse2 },

                .{ .orpd, .rm, &.{ .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0x56 }, 0, .none, .sse2 },

                .{ .packsswb, .rm, &.{ .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0x63 }, 0, .none, .sse2 },
                .{ .packssdw, .rm, &.{ .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0x6b }, 0, .none, .sse2 },

                .{ .packuswb, .rm, &.{ .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0x67 }, 0, .none, .sse2 },

                .{ .paddb, .rm, &.{ .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0xfc }, 0, .none, .sse2 },
                .{ .paddw, .rm, &.{ .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0xfd }, 0, .none, .sse2 },
                .{ .paddd, .rm, &.{ .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0xfe }, 0, .none, .sse2 },
                .{ .paddq, .rm, &.{ .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0xd4 }, 0, .none, .sse2 },

                .{ .paddsb, .rm, &.{ .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0xec }, 0, .none, .sse2 },
                .{ .paddsw, .rm, &.{ .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0xed }, 0, .none, .sse2 },

                .{ .paddusb, .rm, &.{ .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0xdc }, 0, .none, .sse2 },
                .{ .paddusw, .rm, &.{ .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0xdd }, 0, .none, .sse2 },

                .{ .pand, .rm, &.{ .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0xdb }, 0, .none, .sse2 },

                .{ .pandn, .rm, &.{ .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0xdf }, 0, .none, .sse2 },

                .{ .pextrw, .rmi, &.{ .r32, .xmm, .imm8 }, &.{ 0x66, 0x0f, 0xc5 }, 0, .none, .sse2 },

                .{ .pinsrw, .rmi, &.{ .xmm, .r32_m16, .imm8 }, &.{ 0x66, 0x0f, 0xc4 }, 0, .none, .sse2 },

                .{ .pmaxsw, .rm, &.{ .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0xee }, 0, .none, .sse2 },

                .{ .pmaxub, .rm, &.{ .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0xde }, 0, .none, .sse2 },

                .{ .pminsw, .rm, &.{ .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0xea }, 0, .none, .sse2 },

                .{ .pminub, .rm, &.{ .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0xda }, 0, .none, .sse2 },

                .{ .pmulhw, .rm, &.{ .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0xe5 }, 0, .none, .sse2 },

                .{ .pmullw, .rm, &.{ .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0xd5 }, 0, .none, .sse2 },

                .{ .por, .rm, &.{ .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0xeb }, 0, .none, .sse2 },

                .{ .pshufhw, .rmi, &.{ .xmm, .xmm_m128, .imm8 }, &.{ 0xf3, 0x0f, 0x70 }, 0, .none, .sse2 },

                .{ .pshuflw, .rmi, &.{ .xmm, .xmm_m128, .imm8 }, &.{ 0xf2, 0x0f, 0x70 }, 0, .none, .sse2 },

                .{ .psrlw, .rm, &.{ .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0xd1 }, 0, .none, .sse2 },
                .{ .psrlw, .mi, &.{ .xmm, .imm8     }, &.{ 0x66, 0x0f, 0x71 }, 2, .none, .sse2 },
                .{ .psrld, .rm, &.{ .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0xd2 }, 0, .none, .sse2 },
                .{ .psrld, .mi, &.{ .xmm, .imm8     }, &.{ 0x66, 0x0f, 0x72 }, 2, .none, .sse2 },
                .{ .psrlq, .rm, &.{ .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0xd3 }, 0, .none, .sse2 },
                .{ .psrlq, .mi, &.{ .xmm, .imm8     }, &.{ 0x66, 0x0f, 0x73 }, 2, .none, .sse2 },

                .{ .psubb, .rm, &.{ .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0xf8 }, 0, .none, .sse2 },
                .{ .psubw, .rm, &.{ .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0xf9 }, 0, .none, .sse2 },
                .{ .psubd, .rm, &.{ .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0xfa }, 0, .none, .sse2 },

                .{ .psubsb, .rm, &.{ .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0xe8 }, 0, .none, .sse2 },
                .{ .psubsw, .rm, &.{ .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0xe9 }, 0, .none, .sse2 },

                .{ .psubq, .rm, &.{ .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0xfb }, 0, .none, .sse2 },

                .{ .psubusb, .rm, &.{ .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0xd8 }, 0, .none, .sse2 },
                .{ .psubusw, .rm, &.{ .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0xd9 }, 0, .none, .sse2 },

                .{ .punpckhbw,  .rm, &.{ .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0x68 }, 0, .none, .sse2 },
                .{ .punpckhwd,  .rm, &.{ .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0x69 }, 0, .none, .sse2 },
                .{ .punpckhdq,  .rm, &.{ .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0x6a }, 0, .none, .sse2 },
                .{ .punpckhqdq, .rm, &.{ .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0x6d }, 0, .none, .sse2 },

                .{ .punpcklbw,  .rm, &.{ .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0x60 }, 0, .none, .sse2 },
                .{ .punpcklwd,  .rm, &.{ .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0x61 }, 0, .none, .sse2 },
                .{ .punpckldq,  .rm, &.{ .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0x62 }, 0, .none, .sse2 },
                .{ .punpcklqdq, .rm, &.{ .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0x6c }, 0, .none, .sse2 },

                .{ .pxor, .rm, &.{ .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0xef }, 0, .none, .sse2 },

                .{ .shufpd, .rmi, &.{ .xmm, .xmm_m128, .imm8 }, &.{ 0x66, 0x0f, 0xc6 }, 0, .none, .sse2 },

                .{ .sqrtpd, .rm, &.{ .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0x51 }, 0, .none, .sse2 },

                .{ .sqrtsd, .rm, &.{ .xmm, .xmm_m64 }, &.{ 0xf2, 0x0f, 0x51 }, 0, .none, .sse2 },

                .{ .subpd, .rm, &.{ .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0x5c }, 0, .none, .sse2 },

                .{ .subsd, .rm, &.{ .xmm, .xmm_m64 }, &.{ 0xf2, 0x0f, 0x5c }, 0, .none, .sse2 },

                .{ .movsd, .rm, &.{ .xmm,     .xmm_m64 }, &.{ 0xf2, 0x0f, 0x10 }, 0, .none, .sse2 },
                .{ .movsd, .mr, &.{ .xmm_m64, .xmm     }, &.{ 0xf2, 0x0f, 0x11 }, 0, .none, .sse2 },

                .{ .ucomisd, .rm, &.{ .xmm, .xmm_m64 }, &.{ 0x66, 0x0f, 0x2e }, 0, .none, .sse2 },

                .{ .xorpd, .rm, &.{ .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0x57 }, 0, .none, .sse2 },

                // SSE3
                .{ .movddup, .rm, &.{ .xmm, .xmm_m64 }, &.{ 0xf2, 0x0f, 0x12 }, 0, .none, .sse3 },

                .{ .movshdup, .rm, &.{ .xmm, .xmm_m128 }, &.{ 0xf3, 0x0f, 0x16 }, 0, .none, .sse3 },

                .{ .movsldup, .rm, &.{ .xmm, .xmm_m128 }, &.{ 0xf3, 0x0f, 0x12 }, 0, .none, .sse3 },

                // SSE4.1
                .{ .blendpd, .rmi, &.{ .xmm, .xmm_m128, .imm8 }, &.{ 0x66, 0x0f, 0x3a, 0x0d }, 0, .none, .sse4_1 },

                .{ .blendps, .rmi, &.{ .xmm, .xmm_m128, .imm8 }, &.{ 0x66, 0x0f, 0x3a, 0x0c }, 0, .none, .sse4_1 },

                .{ .blendvpd, .rm0, &.{ .xmm, .xmm_m128, .xmm0 }, &.{ 0x66, 0x0f, 0x38, 0x15 }, 0, .none, .sse4_1 },

                .{ .blendvps, .rm0, &.{ .xmm, .xmm_m128, .xmm0 }, &.{ 0x66, 0x0f, 0x38, 0x14 }, 0, .none, .sse4_1 },

                .{ .extractps, .mri, &.{ .rm32, .xmm, .imm8 }, &.{ 0x66, 0x0f, 0x3a, 0x17 }, 0, .none, .sse4_1 },

                .{ .insertps, .rmi, &.{ .xmm, .xmm_m32, .imm8 }, &.{ 0x66, 0x0f, 0x3a, 0x21 }, 0, .none, .sse4_1 },

                .{ .packusdw, .rm, &.{ .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0x38, 0x2b }, 0, .none, .sse4_1 },

                .{ .pextrb, .mri, &.{ .r32_m8, .xmm, .imm8 }, &.{ 0x66, 0x0f, 0x3a, 0x14 }, 0, .none, .sse4_1 },
                .{ .pextrd, .mri, &.{ .rm32,   .xmm, .imm8 }, &.{ 0x66, 0x0f, 0x3a, 0x16 }, 0, .none, .sse4_1 },
                .{ .pextrq, .mri, &.{ .rm64,   .xmm, .imm8 }, &.{ 0x66, 0x0f, 0x3a, 0x16 }, 0, .long, .sse4_1 },

                .{ .pextrw, .mri, &.{ .r32_m16, .xmm, .imm8 }, &.{ 0x66, 0x0f, 0x3a, 0x15 }, 0, .none, .sse4_1 },

                .{ .pinsrb, .rmi, &.{ .xmm, .r32_m8, .imm8 }, &.{ 0x66, 0x0f, 0x3a, 0x20 }, 0, .none, .sse4_1 },
                .{ .pinsrd, .rmi, &.{ .xmm, .rm32,   .imm8 }, &.{ 0x66, 0x0f, 0x3a, 0x22 }, 0, .none, .sse4_1 },
                .{ .pinsrq, .rmi, &.{ .xmm, .rm64,   .imm8 }, &.{ 0x66, 0x0f, 0x3a, 0x22 }, 0, .long, .sse4_1 },

                .{ .pmaxsb, .rm, &.{ .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0x38, 0x3c }, 0, .none, .sse4_1 },
                .{ .pmaxsd, .rm, &.{ .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0x38, 0x3d }, 0, .none, .sse4_1 },

                .{ .pmaxuw, .rm, &.{ .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0x38, 0x3e }, 0, .none, .sse4_1 },

                .{ .pmaxud, .rm, &.{ .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0x38, 0x3f }, 0, .none, .sse4_1 },

                .{ .pminsb, .rm, &.{ .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0x38, 0x38 }, 0, .none, .sse4_1 },
                .{ .pminsd, .rm, &.{ .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0x38, 0x39 }, 0, .none, .sse4_1 },

                .{ .pminuw, .rm, &.{ .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0x38, 0x3a }, 0, .none, .sse4_1 },

                .{ .pminud, .rm, &.{ .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0x38, 0x3b }, 0, .none, .sse4_1 },

                .{ .pmulld, .rm, &.{ .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0x38, 0x40 }, 0, .none, .sse4_1 },

                .{ .roundpd, .rmi, &.{ .xmm, .xmm_m128, .imm8 }, &.{ 0x66, 0x0f, 0x3a, 0x09 }, 0, .none, .sse4_1 },

                .{ .roundps, .rmi, &.{ .xmm, .xmm_m128, .imm8 }, &.{ 0x66, 0x0f, 0x3a, 0x08 }, 0, .none, .sse4_1 },

                .{ .roundsd, .rmi, &.{ .xmm, .xmm_m64, .imm8 }, &.{ 0x66, 0x0f, 0x3a, 0x0b }, 0, .none, .sse4_1 },

                .{ .roundss, .rmi, &.{ .xmm, .xmm_m32, .imm8 }, &.{ 0x66, 0x0f, 0x3a, 0x0a }, 0, .none, .sse4_1 },

                // AVX
                .{ .vaddpd, .rvm, &.{ .xmm, .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0x58 }, 0, .vex_128_wig, .avx },
                .{ .vaddpd, .rvm, &.{ .ymm, .ymm, .ymm_m256 }, &.{ 0x66, 0x0f, 0x58 }, 0, .vex_256_wig, .avx },

                .{ .vaddps, .rvm, &.{ .xmm, .xmm, .xmm_m128 }, &.{ 0x0f, 0x58 }, 0, .vex_128_wig, .avx },
                .{ .vaddps, .rvm, &.{ .ymm, .ymm, .ymm_m256 }, &.{ 0x0f, 0x58 }, 0, .vex_256_wig, .avx },

                .{ .vaddsd, .rvm, &.{ .xmm, .xmm, .xmm_m64 }, &.{ 0xf2, 0x0f, 0x58 }, 0, .vex_lig_wig, .avx },

                .{ .vaddss, .rvm, &.{ .xmm, .xmm, .xmm_m32 }, &.{ 0xf3, 0x0f, 0x58 }, 0, .vex_lig_wig, .avx },

                .{ .vandnpd, .rvm, &.{ .xmm, .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0x55 }, 0, .vex_128_wig, .avx },
                .{ .vandnpd, .rvm, &.{ .ymm, .ymm, .ymm_m256 }, &.{ 0x66, 0x0f, 0x55 }, 0, .vex_256_wig, .avx },

                .{ .vandnps, .rvm, &.{ .xmm, .xmm, .xmm_m128 }, &.{ 0x0f, 0x55 }, 0, .vex_128_wig, .avx },
                .{ .vandnps, .rvm, &.{ .ymm, .ymm, .ymm_m256 }, &.{ 0x0f, 0x55 }, 0, .vex_256_wig, .avx },

                .{ .vandpd, .rvm, &.{ .xmm, .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0x54 }, 0, .vex_128_wig, .avx },
                .{ .vandpd, .rvm, &.{ .ymm, .ymm, .ymm_m256 }, &.{ 0x66, 0x0f, 0x54 }, 0, .vex_256_wig, .avx },

                .{ .vandps, .rvm, &.{ .xmm, .xmm, .xmm_m128 }, &.{ 0x0f, 0x54 }, 0, .vex_128_wig, .avx },
                .{ .vandps, .rvm, &.{ .ymm, .ymm, .ymm_m256 }, &.{ 0x0f, 0x54 }, 0, .vex_256_wig, .avx },

                .{ .vblendpd, .rvmi, &.{ .xmm, .xmm, .xmm_m128, .imm8 }, &.{ 0x66, 0x0f, 0x3a, 0x0d }, 0, .vex_128_wig, .avx },
                .{ .vblendpd, .rvmi, &.{ .ymm, .ymm, .ymm_m256, .imm8 }, &.{ 0x66, 0x0f, 0x3a, 0x0d }, 0, .vex_256_wig, .avx },

                .{ .vblendps, .rvmi, &.{ .xmm, .xmm, .xmm_m128, .imm8 }, &.{ 0x66, 0x0f, 0x3a, 0x0c }, 0, .vex_128_wig, .avx },
                .{ .vblendps, .rvmi, &.{ .ymm, .ymm, .ymm_m256, .imm8 }, &.{ 0x66, 0x0f, 0x3a, 0x0c }, 0, .vex_256_wig, .avx },

                .{ .vblendvpd, .rvmr, &.{ .xmm, .xmm, .xmm_m128, .xmm }, &.{ 0x66, 0x0f, 0x3a, 0x4b }, 0, .vex_128_w0, .avx },
                .{ .vblendvpd, .rvmr, &.{ .ymm, .ymm, .ymm_m256, .ymm }, &.{ 0x66, 0x0f, 0x3a, 0x4b }, 0, .vex_256_w0, .avx },

                .{ .vblendvps, .rvmr, &.{ .xmm, .xmm, .xmm_m128, .xmm }, &.{ 0x66, 0x0f, 0x3a, 0x4a }, 0, .vex_128_w0, .avx },
                .{ .vblendvps, .rvmr, &.{ .ymm, .ymm, .ymm_m256, .ymm }, &.{ 0x66, 0x0f, 0x3a, 0x4a }, 0, .vex_256_w0, .avx },

                .{ .vbroadcastss,   .rm, &.{ .xmm, .m32  }, &.{ 0x66, 0x0f, 0x38, 0x18 }, 0, .vex_128_w0, .avx },
                .{ .vbroadcastss,   .rm, &.{ .ymm, .m32  }, &.{ 0x66, 0x0f, 0x38, 0x18 }, 0, .vex_256_w0, .avx },
                .{ .vbroadcastsd,   .rm, &.{ .ymm, .m64  }, &.{ 0x66, 0x0f, 0x38, 0x19 }, 0, .vex_256_w0, .avx },
                .{ .vbroadcastf128, .rm, &.{ .ymm, .m128 }, &.{ 0x66, 0x0f, 0x38, 0x1a }, 0, .vex_256_w0, .avx },

                .{ .vcmppd, .rvmi, &.{ .xmm, .xmm, .xmm_m128, .imm8 }, &.{ 0x66, 0x0f, 0xc2 }, 0, .vex_128_wig, .avx },
                .{ .vcmppd, .rvmi, &.{ .ymm, .ymm, .ymm_m256, .imm8 }, &.{ 0x66, 0x0f, 0xc2 }, 0, .vex_256_wig, .avx },

                .{ .vcmpps, .rvmi, &.{ .xmm, .xmm, .xmm_m128, .imm8 }, &.{ 0x0f, 0xc2 }, 0, .vex_128_wig, .avx },
                .{ .vcmpps, .rvmi, &.{ .ymm, .ymm, .ymm_m256, .imm8 }, &.{ 0x0f, 0xc2 }, 0, .vex_256_wig, .avx },

                .{ .vcmpsd, .rvmi, &.{ .xmm, .xmm, .xmm_m64, .imm8 }, &.{ 0xf2, 0x0f, 0xc2 }, 0, .vex_lig_wig, .avx },

                .{ .vcmpss, .rvmi, &.{ .xmm, .xmm, .xmm_m32, .imm8 }, &.{ 0xf3, 0x0f, 0xc2 }, 0, .vex_lig_wig, .avx },

                .{ .vcvtdq2pd, .rm, &.{ .xmm, .xmm_m64  }, &.{ 0xf3, 0x0f, 0xe6 }, 0, .vex_128_wig, .avx },
                .{ .vcvtdq2pd, .rm, &.{ .ymm, .xmm_m128 }, &.{ 0xf3, 0x0f, 0xe6 }, 0, .vex_256_wig, .avx },

                .{ .vcvtdq2ps, .rm, &.{ .xmm, .xmm_m128 }, &.{ 0x0f, 0x5b }, 0, .vex_128_wig, .avx },
                .{ .vcvtdq2ps, .rm, &.{ .ymm, .ymm_m256 }, &.{ 0x0f, 0x5b }, 0, .vex_256_wig, .avx },

                .{ .vcvtpd2dq, .rm, &.{ .xmm, .xmm_m128 }, &.{ 0xf2, 0x0f, 0xe6 }, 0, .vex_128_wig, .avx },
                .{ .vcvtpd2dq, .rm, &.{ .xmm, .ymm_m256 }, &.{ 0xf2, 0x0f, 0xe6 }, 0, .vex_256_wig, .avx },

                .{ .vcvtpd2ps, .rm, &.{ .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0x5a }, 0, .vex_128_wig, .avx },
                .{ .vcvtpd2ps, .rm, &.{ .xmm, .ymm_m256 }, &.{ 0x66, 0x0f, 0x5a }, 0, .vex_256_wig, .avx },

                .{ .vcvtps2dq, .rm, &.{ .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0x5b }, 0, .vex_128_wig, .avx },
                .{ .vcvtps2dq, .rm, &.{ .ymm, .ymm_m256 }, &.{ 0x66, 0x0f, 0x5b }, 0, .vex_256_wig, .avx },

                .{ .vcvtps2pd, .rm, &.{ .xmm, .xmm_m64  }, &.{ 0x0f, 0x5a }, 0, .vex_128_wig, .avx },
                .{ .vcvtps2pd, .rm, &.{ .ymm, .xmm_m128 }, &.{ 0x0f, 0x5a }, 0, .vex_256_wig, .avx },

                .{ .vcvtsd2si, .rm, &.{ .r32, .xmm_m64 }, &.{ 0xf2, 0x0f, 0x2d }, 0, .vex_lig_w0, .sse2 },
                .{ .vcvtsd2si, .rm, &.{ .r64, .xmm_m64 }, &.{ 0xf2, 0x0f, 0x2d }, 0, .vex_lig_w1, .sse2 },

                .{ .vcvtsd2ss, .rvm, &.{ .xmm, .xmm, .xmm_m64 }, &.{ 0xf2, 0x0f, 0x5a }, 0, .vex_lig_wig, .avx },

                .{ .vcvtsi2sd, .rvm, &.{ .xmm, .xmm, .rm32 }, &.{ 0xf2, 0x0f, 0x2a }, 0, .vex_lig_w0, .avx },
                .{ .vcvtsi2sd, .rvm, &.{ .xmm, .xmm, .rm64 }, &.{ 0xf2, 0x0f, 0x2a }, 0, .vex_lig_w1, .avx },

                .{ .vcvtsi2ss, .rvm, &.{ .xmm, .xmm, .rm32 }, &.{ 0xf3, 0x0f, 0x2a }, 0, .vex_lig_w0, .avx },
                .{ .vcvtsi2ss, .rvm, &.{ .xmm, .xmm, .rm64 }, &.{ 0xf3, 0x0f, 0x2a }, 0, .vex_lig_w1, .avx },

                .{ .vcvtss2sd, .rvm, &.{ .xmm, .xmm, .xmm_m32 }, &.{ 0xf3, 0x0f, 0x5a }, 0, .vex_lig_wig, .avx },

                .{ .vcvtss2si, .rm, &.{ .r32, .xmm_m32 }, &.{ 0xf3, 0x0f, 0x2d }, 0, .vex_lig_w0, .avx },
                .{ .vcvtss2si, .rm, &.{ .r64, .xmm_m32 }, &.{ 0xf3, 0x0f, 0x2d }, 0, .vex_lig_w1, .avx },

                .{ .vcvttpd2dq, .rm, &.{ .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0xe6 }, 0, .vex_128_wig, .avx },
                .{ .vcvttpd2dq, .rm, &.{ .xmm, .ymm_m256 }, &.{ 0x66, 0x0f, 0xe6 }, 0, .vex_256_wig, .avx },

                .{ .vcvttps2dq, .rm, &.{ .xmm, .xmm_m128 }, &.{ 0xf3, 0x0f, 0x5b }, 0, .vex_128_wig, .avx },
                .{ .vcvttps2dq, .rm, &.{ .ymm, .ymm_m256 }, &.{ 0xf3, 0x0f, 0x5b }, 0, .vex_256_wig, .avx },

                .{ .vcvttsd2si, .rm, &.{ .r32, .xmm_m64 }, &.{ 0xf2, 0x0f, 0x2c }, 0, .vex_lig_w0, .sse2 },
                .{ .vcvttsd2si, .rm, &.{ .r64, .xmm_m64 }, &.{ 0xf2, 0x0f, 0x2c }, 0, .vex_lig_w1, .sse2 },

                .{ .vcvttss2si, .rm, &.{ .r32, .xmm_m32 }, &.{ 0xf3, 0x0f, 0x2c }, 0, .vex_lig_w0, .avx },
                .{ .vcvttss2si, .rm, &.{ .r64, .xmm_m32 }, &.{ 0xf3, 0x0f, 0x2c }, 0, .vex_lig_w1, .avx },

                .{ .vdivpd, .rvm, &.{ .xmm, .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0x5e }, 0, .vex_128_wig, .avx },
                .{ .vdivpd, .rvm, &.{ .ymm, .ymm, .ymm_m256 }, &.{ 0x66, 0x0f, 0x5e }, 0, .vex_256_wig, .avx },

                .{ .vdivps, .rvm, &.{ .xmm, .xmm, .xmm_m128 }, &.{ 0x0f, 0x5e }, 0, .vex_128_wig, .avx },
                .{ .vdivps, .rvm, &.{ .ymm, .ymm, .ymm_m256 }, &.{ 0x0f, 0x5e }, 0, .vex_256_wig, .avx },

                .{ .vdivsd, .rvm, &.{ .xmm, .xmm, .xmm_m64 }, &.{ 0xf2, 0x0f, 0x5e }, 0, .vex_lig_wig, .avx },

                .{ .vdivss, .rvm, &.{ .xmm, .xmm, .xmm_m32 }, &.{ 0xf3, 0x0f, 0x5e }, 0, .vex_lig_wig, .avx },

                .{ .vextractf128, .mri, &.{ .xmm_m128, .ymm, .imm8 }, &.{ 0x66, 0x0f, 0x3a, 0x19 }, 0, .vex_256_w0, .avx },

                .{ .vextractps, .mri, &.{ .rm32, .xmm, .imm8 }, &.{ 0x66, 0x0f, 0x3a, 0x17 }, 0, .vex_128_wig, .avx },

                .{ .vinsertf128, .rvmi, &.{ .ymm, .ymm, .xmm_m128, .imm8 }, &.{ 0x66, 0x0f, 0x3a, 0x18 }, 0, .vex_256_w0, .avx },

                .{ .vinsertps, .rvmi, &.{ .xmm, .xmm, .xmm_m32, .imm8 }, &.{ 0x66, 0x0f, 0x3a, 0x21 }, 0, .vex_128_wig, .avx },

                .{ .vmaxpd, .rvm, &.{ .xmm, .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0x5f }, 0, .vex_128_wig, .avx },
                .{ .vmaxpd, .rvm, &.{ .ymm, .ymm, .ymm_m256 }, &.{ 0x66, 0x0f, 0x5f }, 0, .vex_256_wig, .avx },

                .{ .vmaxps, .rvm, &.{ .xmm, .xmm, .xmm_m128 }, &.{ 0x0f, 0x5f }, 0, .vex_128_wig, .avx },
                .{ .vmaxps, .rvm, &.{ .ymm, .ymm, .ymm_m256 }, &.{ 0x0f, 0x5f }, 0, .vex_256_wig, .avx },

                .{ .vmaxsd, .rvm, &.{ .xmm, .xmm, .xmm_m64 }, &.{ 0xf2, 0x0f, 0x5f }, 0, .vex_lig_wig, .avx },

                .{ .vmaxss, .rvm, &.{ .xmm, .xmm, .xmm_m32 }, &.{ 0xf3, 0x0f, 0x5f }, 0, .vex_lig_wig, .avx },

                .{ .vminpd, .rvm, &.{ .xmm, .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0x5d }, 0, .vex_128_wig, .avx },
                .{ .vminpd, .rvm, &.{ .ymm, .ymm, .ymm_m256 }, &.{ 0x66, 0x0f, 0x5d }, 0, .vex_256_wig, .avx },

                .{ .vminps, .rvm, &.{ .xmm, .xmm, .xmm_m128 }, &.{ 0x0f, 0x5d }, 0, .vex_128_wig, .avx },
                .{ .vminps, .rvm, &.{ .ymm, .ymm, .ymm_m256 }, &.{ 0x0f, 0x5d }, 0, .vex_256_wig, .avx },

                .{ .vminsd, .rvm, &.{ .xmm, .xmm, .xmm_m64 }, &.{ 0xf2, 0x0f, 0x5d }, 0, .vex_lig_wig, .avx },

                .{ .vminss, .rvm, &.{ .xmm, .xmm, .xmm_m32 }, &.{ 0xf3, 0x0f, 0x5d }, 0, .vex_lig_wig, .avx },

                .{ .vmovapd, .rm, &.{ .xmm,      .xmm_m128 }, &.{ 0x66, 0x0f, 0x28 }, 0, .vex_128_wig, .avx },
                .{ .vmovapd, .mr, &.{ .xmm_m128, .xmm      }, &.{ 0x66, 0x0f, 0x29 }, 0, .vex_128_wig, .avx },
                .{ .vmovapd, .rm, &.{ .ymm,      .ymm_m256 }, &.{ 0x66, 0x0f, 0x28 }, 0, .vex_256_wig, .avx },
                .{ .vmovapd, .mr, &.{ .ymm_m256, .ymm      }, &.{ 0x66, 0x0f, 0x29 }, 0, .vex_256_wig, .avx },

                .{ .vmovaps, .rm, &.{ .xmm,      .xmm_m128 }, &.{ 0x0f, 0x28 }, 0, .vex_128_wig, .avx },
                .{ .vmovaps, .mr, &.{ .xmm_m128, .xmm      }, &.{ 0x0f, 0x29 }, 0, .vex_128_wig, .avx },
                .{ .vmovaps, .rm, &.{ .ymm,      .ymm_m256 }, &.{ 0x0f, 0x28 }, 0, .vex_256_wig, .avx },
                .{ .vmovaps, .mr, &.{ .ymm_m256, .ymm      }, &.{ 0x0f, 0x29 }, 0, .vex_256_wig, .avx },

                .{ .vmovd, .rm, &.{ .xmm,  .rm32 }, &.{ 0x66, 0x0f, 0x6e }, 0, .vex_128_w0, .avx },
                .{ .vmovq, .rm, &.{ .xmm,  .rm64 }, &.{ 0x66, 0x0f, 0x6e }, 0, .vex_128_w1, .avx },
                .{ .vmovd, .mr, &.{ .rm32, .xmm  }, &.{ 0x66, 0x0f, 0x7e }, 0, .vex_128_w0, .avx },
                .{ .vmovq, .mr, &.{ .rm64, .xmm  }, &.{ 0x66, 0x0f, 0x7e }, 0, .vex_128_w1, .avx },

                .{ .vmovddup, .rm, &.{ .xmm, .xmm_m64  }, &.{ 0xf2, 0x0f, 0x12 }, 0, .vex_128_wig, .avx },
                .{ .vmovddup, .rm, &.{ .ymm, .ymm_m256 }, &.{ 0xf2, 0x0f, 0x12 }, 0, .vex_256_wig, .avx },

                .{ .vmovdqa, .rm, &.{ .xmm,      .xmm_m128 }, &.{ 0x66, 0x0f, 0x6f }, 0, .vex_128_wig, .avx },
                .{ .vmovdqa, .mr, &.{ .xmm_m128, .xmm      }, &.{ 0x66, 0x0f, 0x7f }, 0, .vex_128_wig, .avx },
                .{ .vmovdqa, .rm, &.{ .ymm,      .ymm_m256 }, &.{ 0x66, 0x0f, 0x6f }, 0, .vex_256_wig, .avx },
                .{ .vmovdqa, .mr, &.{ .ymm_m256, .ymm      }, &.{ 0x66, 0x0f, 0x7f }, 0, .vex_256_wig, .avx },

                .{ .vmovdqu, .rm, &.{ .xmm,      .xmm_m128 }, &.{ 0xf3, 0x0f, 0x6f }, 0, .vex_128_wig, .avx },
                .{ .vmovdqu, .mr, &.{ .xmm_m128, .xmm      }, &.{ 0xf3, 0x0f, 0x7f }, 0, .vex_128_wig, .avx },
                .{ .vmovdqu, .rm, &.{ .ymm,      .ymm_m256 }, &.{ 0xf3, 0x0f, 0x6f }, 0, .vex_256_wig, .avx },
                .{ .vmovdqu, .mr, &.{ .ymm_m256, .ymm      }, &.{ 0xf3, 0x0f, 0x7f }, 0, .vex_256_wig, .avx },

                .{ .vmovhlps, .rvm, &.{ .xmm, .xmm, .xmm }, &.{ 0x0f, 0x12 }, 0, .vex_128_wig, .avx },

                .{ .vmovlhps, .rvm, &.{ .xmm, .xmm, .xmm }, &.{ 0x0f, 0x16 }, 0, .vex_128_wig, .avx },

                .{ .vmovq, .rm, &.{ .xmm,     .xmm_m64 }, &.{ 0xf3, 0x0f, 0x7e }, 0, .vex_128_wig, .avx },
                .{ .vmovq, .mr, &.{ .xmm_m64, .xmm     }, &.{ 0x66, 0x0f, 0xd6 }, 0, .vex_128_wig, .avx },

                .{ .vmovsd, .rvm, &.{ .xmm, .xmm, .xmm }, &.{ 0xf2, 0x0f, 0x10 }, 0, .vex_lig_wig, .avx },
                .{ .vmovsd, .rm,  &.{       .xmm, .m64 }, &.{ 0xf2, 0x0f, 0x10 }, 0, .vex_lig_wig, .avx },
                .{ .vmovsd, .mvr, &.{ .xmm, .xmm, .xmm }, &.{ 0xf2, 0x0f, 0x11 }, 0, .vex_lig_wig, .avx },
                .{ .vmovsd, .mr,  &.{       .m64, .xmm }, &.{ 0xf2, 0x0f, 0x11 }, 0, .vex_lig_wig, .avx },

                .{ .vmovshdup, .rm, &.{ .xmm, .xmm_m128 }, &.{ 0xf3, 0x0f, 0x16 }, 0, .vex_128_wig, .avx },
                .{ .vmovshdup, .rm, &.{ .ymm, .ymm_m256 }, &.{ 0xf3, 0x0f, 0x16 }, 0, .vex_256_wig, .avx },

                .{ .vmovsldup, .rm, &.{ .xmm, .xmm_m128 }, &.{ 0xf3, 0x0f, 0x12 }, 0, .vex_128_wig, .avx },
                .{ .vmovsldup, .rm, &.{ .ymm, .ymm_m256 }, &.{ 0xf3, 0x0f, 0x12 }, 0, .vex_256_wig, .avx },

                .{ .vmovss, .rvm, &.{ .xmm, .xmm, .xmm }, &.{ 0xf3, 0x0f, 0x10 }, 0, .vex_lig_wig, .avx },
                .{ .vmovss, .rm,  &.{       .xmm, .m32 }, &.{ 0xf3, 0x0f, 0x10 }, 0, .vex_lig_wig, .avx },
                .{ .vmovss, .mvr, &.{ .xmm, .xmm, .xmm }, &.{ 0xf3, 0x0f, 0x11 }, 0, .vex_lig_wig, .avx },
                .{ .vmovss, .mr,  &.{       .m32, .xmm }, &.{ 0xf3, 0x0f, 0x11 }, 0, .vex_lig_wig, .avx },

                .{ .vmovupd, .rm, &.{ .xmm,      .xmm_m128 }, &.{ 0x66, 0x0f, 0x10 }, 0, .vex_128_wig, .avx },
                .{ .vmovupd, .mr, &.{ .xmm_m128, .xmm      }, &.{ 0x66, 0x0f, 0x11 }, 0, .vex_128_wig, .avx },
                .{ .vmovupd, .rm, &.{ .ymm,      .ymm_m256 }, &.{ 0x66, 0x0f, 0x10 }, 0, .vex_256_wig, .avx },
                .{ .vmovupd, .mr, &.{ .ymm_m256, .ymm      }, &.{ 0x66, 0x0f, 0x11 }, 0, .vex_256_wig, .avx },

                .{ .vmovups, .rm, &.{ .xmm,      .xmm_m128 }, &.{ 0x0f, 0x10 }, 0, .vex_128_wig, .avx },
                .{ .vmovups, .mr, &.{ .xmm_m128, .xmm      }, &.{ 0x0f, 0x11 }, 0, .vex_128_wig, .avx },
                .{ .vmovups, .rm, &.{ .ymm,      .ymm_m256 }, &.{ 0x0f, 0x10 }, 0, .vex_256_wig, .avx },
                .{ .vmovups, .mr, &.{ .ymm_m256, .ymm      }, &.{ 0x0f, 0x11 }, 0, .vex_256_wig, .avx },

                .{ .vmulpd, .rvm, &.{ .xmm, .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0x59 }, 0, .vex_128_wig, .avx },
                .{ .vmulpd, .rvm, &.{ .ymm, .ymm, .ymm_m256 }, &.{ 0x66, 0x0f, 0x59 }, 0, .vex_256_wig, .avx },

                .{ .vmulps, .rvm, &.{ .xmm, .xmm, .xmm_m128 }, &.{ 0x0f, 0x59 }, 0, .vex_128_wig, .avx },
                .{ .vmulps, .rvm, &.{ .ymm, .ymm, .ymm_m256 }, &.{ 0x0f, 0x59 }, 0, .vex_256_wig, .avx },

                .{ .vmulsd, .rvm, &.{ .xmm, .xmm, .xmm_m64 }, &.{ 0xf2, 0x0f, 0x59 }, 0, .vex_lig_wig, .avx },

                .{ .vmulss, .rvm, &.{ .xmm, .xmm, .xmm_m32 }, &.{ 0xf3, 0x0f, 0x59 }, 0, .vex_lig_wig, .avx },

                .{ .vorpd, .rvm, &.{ .xmm, .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0x56 }, 0, .vex_128_wig, .avx },
                .{ .vorpd, .rvm, &.{ .ymm, .ymm, .ymm_m256 }, &.{ 0x66, 0x0f, 0x56 }, 0, .vex_256_wig, .avx },

                .{ .vorps, .rvm, &.{ .xmm, .xmm, .xmm_m128 }, &.{ 0x0f, 0x56 }, 0, .vex_128_wig, .avx },
                .{ .vorps, .rvm, &.{ .ymm, .ymm, .ymm_m256 }, &.{ 0x0f, 0x56 }, 0, .vex_256_wig, .avx },

                .{ .vpacksswb, .rvm, &.{ .xmm, .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0x63 }, 0, .vex_128_wig, .avx },
                .{ .vpackssdw, .rvm, &.{ .xmm, .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0x6b }, 0, .vex_128_wig, .avx },

                .{ .vpackusdw, .rvm, &.{ .xmm, .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0x38, 0x2b }, 0, .vex_128_wig, .avx },

                .{ .vpackuswb, .rvm, &.{ .xmm, .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0x67 }, 0, .vex_128_wig, .avx },

                .{ .vpaddb, .rvm, &.{ .xmm, .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0xfc }, 0, .vex_128_wig, .avx },
                .{ .vpaddw, .rvm, &.{ .xmm, .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0xfd }, 0, .vex_128_wig, .avx },
                .{ .vpaddd, .rvm, &.{ .xmm, .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0xfe }, 0, .vex_128_wig, .avx },
                .{ .vpaddq, .rvm, &.{ .xmm, .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0xd4 }, 0, .vex_128_wig, .avx },

                .{ .vpaddsb, .rvm, &.{ .xmm, .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0xec }, 0, .vex_128_wig, .avx },
                .{ .vpaddsw, .rvm, &.{ .xmm, .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0xed }, 0, .vex_128_wig, .avx },

                .{ .vpaddusb, .rvm, &.{ .xmm, .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0xdc }, 0, .vex_128_wig, .avx },
                .{ .vpaddusw, .rvm, &.{ .xmm, .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0xdd }, 0, .vex_128_wig, .avx },

                .{ .vpand, .rvm, &.{ .xmm, .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0xdb }, 0, .vex_128_wig, .avx },

                .{ .vpandn, .rvm, &.{ .xmm, .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0xdf }, 0, .vex_128_wig, .avx },

                .{ .vpextrb, .mri, &.{ .r32_m8, .xmm, .imm8 }, &.{ 0x66, 0x0f, 0x3a, 0x14 }, 0, .vex_128_w0, .avx },
                .{ .vpextrd, .mri, &.{ .rm32,   .xmm, .imm8 }, &.{ 0x66, 0x0f, 0x3a, 0x16 }, 0, .vex_128_w0, .avx },
                .{ .vpextrq, .mri, &.{ .rm64,   .xmm, .imm8 }, &.{ 0x66, 0x0f, 0x3a, 0x16 }, 0, .vex_128_w1, .avx },

                .{ .vpextrw, .rmi, &.{ .r32,     .xmm, .imm8 }, &.{ 0x66, 0x0f,       0x15 }, 0, .vex_128_wig, .avx },
                .{ .vpextrw, .mri, &.{ .r32_m16, .xmm, .imm8 }, &.{ 0x66, 0x0f, 0x3a, 0x15 }, 0, .vex_128_wig, .avx },

                .{ .vpinsrb, .rmi, &.{ .xmm, .r32_m8, .imm8 }, &.{ 0x66, 0x0f, 0x3a, 0x20 }, 0, .vex_128_w0, .avx },
                .{ .vpinsrd, .rmi, &.{ .xmm, .rm32,   .imm8 }, &.{ 0x66, 0x0f, 0x3a, 0x22 }, 0, .vex_128_w0, .avx },
                .{ .vpinsrq, .rmi, &.{ .xmm, .rm64,   .imm8 }, &.{ 0x66, 0x0f, 0x3a, 0x22 }, 0, .vex_128_w1, .avx },

                .{ .vpinsrw, .rvmi, &.{ .xmm, .xmm, .r32_m16, .imm8 }, &.{ 0x66, 0x0f, 0xc4 }, 0, .vex_128_wig, .avx },

                .{ .vpmaxsb, .rvm, &.{ .xmm, .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0x38, 0x3c }, 0, .vex_128_wig, .avx },
                .{ .vpmaxsw, .rvm, &.{ .xmm, .xmm, .xmm_m128 }, &.{ 0x66, 0x0f,       0xee }, 0, .vex_128_wig, .avx },
                .{ .vpmaxsd, .rvm, &.{ .xmm, .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0x38, 0x3d }, 0, .vex_128_wig, .avx },

                .{ .vpmaxub, .rvm, &.{ .xmm, .xmm, .xmm_m128 }, &.{ 0x66, 0x0f,       0xde }, 0, .vex_128_wig, .avx },
                .{ .vpmaxuw, .rvm, &.{ .xmm, .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0x38, 0x3e }, 0, .vex_128_wig, .avx },

                .{ .vpmaxud, .rvm, &.{ .xmm, .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0x38, 0x3f }, 0, .vex_128_wig, .avx },

                .{ .vpminsb, .rvm, &.{ .xmm, .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0x38, 0x38 }, 0, .vex_128_wig, .avx },
                .{ .vpminsw, .rvm, &.{ .xmm, .xmm, .xmm_m128 }, &.{ 0x66, 0x0f,       0xea }, 0, .vex_128_wig, .avx },
                .{ .vpminsd, .rvm, &.{ .xmm, .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0x38, 0x39 }, 0, .vex_128_wig, .avx },

                .{ .vpminub, .rvm, &.{ .xmm, .xmm, .xmm_m128 }, &.{ 0x66, 0x0f,       0xda }, 0, .vex_128_wig, .avx },
                .{ .vpminuw, .rvm, &.{ .xmm, .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0x38, 0x3a }, 0, .vex_128_wig, .avx },

                .{ .vpminud, .rvm, &.{ .xmm, .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0x38, 0x3b }, 0, .vex_128_wig, .avx },

                .{ .vpmulhw, .rvm, &.{ .xmm, .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0xe5 }, 0, .vex_128_wig, .avx },

                .{ .vpmulld, .rvm, &.{ .xmm, .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0x38, 0x40 }, 0, .vex_128_wig, .avx },

                .{ .vpmullw, .rvm, &.{ .xmm, .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0xd5 }, 0, .vex_128_wig, .avx },

                .{ .vpor, .rvm, &.{ .xmm, .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0xeb }, 0, .vex_128_wig, .avx },

                .{ .vpsrlw, .rvm, &.{ .xmm, .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0xd1 }, 0, .vex_128_wig, .avx },
                .{ .vpsrlw, .vmi, &.{ .xmm, .xmm, .imm8     }, &.{ 0x66, 0x0f, 0x71 }, 2, .vex_128_wig, .avx },
                .{ .vpsrld, .rvm, &.{ .xmm, .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0xd2 }, 0, .vex_128_wig, .avx },
                .{ .vpsrld, .vmi, &.{ .xmm, .xmm, .imm8     }, &.{ 0x66, 0x0f, 0x72 }, 2, .vex_128_wig, .avx },
                .{ .vpsrlq, .rvm, &.{ .xmm, .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0xd3 }, 0, .vex_128_wig, .avx },
                .{ .vpsrlq, .vmi, &.{ .xmm, .xmm, .imm8     }, &.{ 0x66, 0x0f, 0x73 }, 2, .vex_128_wig, .avx },

                .{ .vpsubb, .rvm, &.{ .xmm, .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0xf8 }, 0, .vex_128_wig, .avx },
                .{ .vpsubw, .rvm, &.{ .xmm, .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0xf9 }, 0, .vex_128_wig, .avx },
                .{ .vpsubd, .rvm, &.{ .xmm, .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0xfa }, 0, .vex_128_wig, .avx },

                .{ .vpsubsb, .rvm, &.{ .xmm, .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0xe8 }, 0, .vex_128_wig, .avx },
                .{ .vpsubsw, .rvm, &.{ .xmm, .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0xe9 }, 0, .vex_128_wig, .avx },

                .{ .vpsubq, .rvm, &.{ .xmm, .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0xfb }, 0, .vex_128_wig, .avx },

                .{ .vpsubusb, .rvm, &.{ .xmm, .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0xd8 }, 0, .vex_128_wig, .avx },
                .{ .vpsubusw, .rvm, &.{ .xmm, .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0xd9 }, 0, .vex_128_wig, .avx },

                .{ .vpunpckhbw,  .rvm, &.{ .xmm, .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0x68 }, 0, .vex_128_wig, .avx },
                .{ .vpunpckhwd,  .rvm, &.{ .xmm, .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0x69 }, 0, .vex_128_wig, .avx },
                .{ .vpunpckhdq,  .rvm, &.{ .xmm, .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0x6a }, 0, .vex_128_wig, .avx },
                .{ .vpunpckhqdq, .rvm, &.{ .xmm, .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0x6d }, 0, .vex_128_wig, .avx },

                .{ .vpunpcklbw,  .rvm, &.{ .xmm, .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0x60 }, 0, .vex_128_wig, .avx },
                .{ .vpunpcklwd,  .rvm, &.{ .xmm, .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0x61 }, 0, .vex_128_wig, .avx },
                .{ .vpunpckldq,  .rvm, &.{ .xmm, .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0x62 }, 0, .vex_128_wig, .avx },
                .{ .vpunpcklqdq, .rvm, &.{ .xmm, .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0x6c }, 0, .vex_128_wig, .avx },

                .{ .vpxor, .rvm, &.{ .xmm, .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0xef }, 0, .vex_128_wig, .avx },

                .{ .vroundpd, .rmi, &.{ .xmm, .xmm_m128, .imm8 }, &.{ 0x66, 0x0f, 0x3a, 0x09 }, 0, .vex_128_wig, .avx },
                .{ .vroundpd, .rmi, &.{ .ymm, .ymm_m256, .imm8 }, &.{ 0x66, 0x0f, 0x3a, 0x09 }, 0, .vex_256_wig, .avx },

                .{ .vroundps, .rmi, &.{ .xmm, .xmm_m128, .imm8 }, &.{ 0x66, 0x0f, 0x3a, 0x08 }, 0, .vex_128_wig, .avx },
                .{ .vroundps, .rmi, &.{ .ymm, .ymm_m256, .imm8 }, &.{ 0x66, 0x0f, 0x3a, 0x08 }, 0, .vex_256_wig, .avx },

                .{ .vroundsd, .rvmi, &.{ .xmm, .xmm, .xmm_m64, .imm8 }, &.{ 0x66, 0x0f, 0x3a, 0x0b }, 0, .vex_lig_wig, .avx },

                .{ .vroundss, .rvmi, &.{ .xmm, .xmm, .xmm_m32, .imm8 }, &.{ 0x66, 0x0f, 0x3a, 0x0a }, 0, .vex_lig_wig, .avx },

                .{ .vshufpd, .rvmi, &.{ .xmm, .xmm, .xmm_m128, .imm8 }, &.{ 0x66, 0x0f, 0xc6 }, 0, .vex_128_wig, .avx },
                .{ .vshufpd, .rvmi, &.{ .ymm, .ymm, .ymm_m256, .imm8 }, &.{ 0x66, 0x0f, 0xc6 }, 0, .vex_256_wig, .avx },

                .{ .vshufps, .rvmi, &.{ .xmm, .xmm, .xmm_m128, .imm8 }, &.{ 0x0f, 0xc6 }, 0, .vex_128_wig, .avx },
                .{ .vshufps, .rvmi, &.{ .ymm, .ymm, .ymm_m256, .imm8 }, &.{ 0x0f, 0xc6 }, 0, .vex_256_wig, .avx },

                .{ .vsqrtpd, .rm, &.{ .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0x51 }, 0, .vex_128_wig, .avx },
                .{ .vsqrtpd, .rm, &.{ .ymm, .ymm_m256 }, &.{ 0x66, 0x0f, 0x51 }, 0, .vex_256_wig, .avx },

                .{ .vsqrtps, .rm, &.{ .xmm, .xmm_m128 }, &.{ 0x0f, 0x51 }, 0, .vex_128_wig, .avx },
                .{ .vsqrtps, .rm, &.{ .ymm, .ymm_m256 }, &.{ 0x0f, 0x51 }, 0, .vex_256_wig, .avx },

                .{ .vsqrtsd, .rvm, &.{ .xmm, .xmm, .xmm_m64 }, &.{ 0xf2, 0x0f, 0x51 }, 0, .vex_lig_wig, .avx },

                .{ .vsqrtss, .rvm, &.{ .xmm, .xmm, .xmm_m32 }, &.{ 0xf3, 0x0f, 0x51 }, 0, .vex_lig_wig, .avx },

                .{ .vsubpd, .rvm, &.{ .xmm, .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0x5c }, 0, .vex_128_wig, .avx },
                .{ .vsubpd, .rvm, &.{ .ymm, .ymm, .ymm_m256 }, &.{ 0x66, 0x0f, 0x5c }, 0, .vex_256_wig, .avx },

                .{ .vsubps, .rvm, &.{ .xmm, .xmm, .xmm_m128 }, &.{ 0x0f, 0x5c }, 0, .vex_128_wig, .avx },
                .{ .vsubps, .rvm, &.{ .ymm, .ymm, .ymm_m256 }, &.{ 0x0f, 0x5c }, 0, .vex_256_wig, .avx },

                .{ .vsubsd, .rvm, &.{ .xmm, .xmm, .xmm_m64 }, &.{ 0xf2, 0x0f, 0x5c }, 0, .vex_lig_wig, .avx },

                .{ .vsubss, .rvm, &.{ .xmm, .xmm, .xmm_m32 }, &.{ 0xf3, 0x0f, 0x5c }, 0, .vex_lig_wig, .avx },

                .{ .vxorpd, .rvm, &.{ .xmm, .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0x57 }, 0, .vex_128_wig, .avx },
                .{ .vxorpd, .rvm, &.{ .ymm, .ymm, .ymm_m256 }, &.{ 0x66, 0x0f, 0x57 }, 0, .vex_256_wig, .avx },

                .{ .vxorps, .rvm, &.{ .xmm, .xmm, .xmm_m128 }, &.{ 0x0f, 0x57 }, 0, .vex_128_wig, .avx },
                .{ .vxorps, .rvm, &.{ .ymm, .ymm, .ymm_m256 }, &.{ 0x0f, 0x57 }, 0, .vex_256_wig, .avx },

                // F16C
                .{ .vcvtph2ps, .rm, &.{ .xmm, .xmm_m64  }, &.{ 0x66, 0x0f, 0x38, 0x13 }, 0, .vex_128_w0, .f16c },
                .{ .vcvtph2ps, .rm, &.{ .ymm, .xmm_m128 }, &.{ 0x66, 0x0f, 0x38, 0x13 }, 0, .vex_256_w0, .f16c },

                .{ .vcvtps2ph, .mri, &.{ .xmm_m64,  .xmm, .imm8 }, &.{ 0x66, 0x0f, 0x3a, 0x1d }, 0, .vex_128_w0, .f16c },
                .{ .vcvtps2ph, .mri, &.{ .xmm_m128, .ymm, .imm8 }, &.{ 0x66, 0x0f, 0x3a, 0x1d }, 0, .vex_256_w0, .f16c },

                // FMA
                .{ .vfmadd132pd, .rvm, &.{ .xmm, .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0x38, 0x98 }, 0, .vex_128_w1, .fma },
                .{ .vfmadd213pd, .rvm, &.{ .xmm, .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0x38, 0xa8 }, 0, .vex_128_w1, .fma },
                .{ .vfmadd231pd, .rvm, &.{ .xmm, .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0x38, 0xb8 }, 0, .vex_128_w1, .fma },
                .{ .vfmadd132pd, .rvm, &.{ .ymm, .ymm, .ymm_m256 }, &.{ 0x66, 0x0f, 0x38, 0x98 }, 0, .vex_256_w1, .fma },
                .{ .vfmadd213pd, .rvm, &.{ .ymm, .ymm, .ymm_m256 }, &.{ 0x66, 0x0f, 0x38, 0xa8 }, 0, .vex_256_w1, .fma },
                .{ .vfmadd231pd, .rvm, &.{ .ymm, .ymm, .ymm_m256 }, &.{ 0x66, 0x0f, 0x38, 0xb8 }, 0, .vex_256_w1, .fma },

                .{ .vfmadd132ps, .rvm, &.{ .xmm, .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0x38, 0x98 }, 0, .vex_128_w0, .fma },
                .{ .vfmadd213ps, .rvm, &.{ .xmm, .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0x38, 0xa8 }, 0, .vex_128_w0, .fma },
                .{ .vfmadd231ps, .rvm, &.{ .xmm, .xmm, .xmm_m128 }, &.{ 0x66, 0x0f, 0x38, 0xb8 }, 0, .vex_128_w0, .fma },
                .{ .vfmadd132ps, .rvm, &.{ .ymm, .ymm, .ymm_m256 }, &.{ 0x66, 0x0f, 0x38, 0x98 }, 0, .vex_256_w0, .fma },
                .{ .vfmadd213ps, .rvm, &.{ .ymm, .ymm, .ymm_m256 }, &.{ 0x66, 0x0f, 0x38, 0xa8 }, 0, .vex_256_w0, .fma },
                .{ .vfmadd231ps, .rvm, &.{ .ymm, .ymm, .ymm_m256 }, &.{ 0x66, 0x0f, 0x38, 0xb8 }, 0, .vex_256_w0, .fma },

                .{ .vfmadd132sd, .rvm, &.{ .xmm, .xmm, .xmm_m64 }, &.{ 0x66, 0x0f, 0x38, 0x99 }, 0, .vex_lig_w1, .fma },
                .{ .vfmadd213sd, .rvm, &.{ .xmm, .xmm, .xmm_m64 }, &.{ 0x66, 0x0f, 0x38, 0xa9 }, 0, .vex_lig_w1, .fma },
                .{ .vfmadd231sd, .rvm, &.{ .xmm, .xmm, .xmm_m64 }, &.{ 0x66, 0x0f, 0x38, 0xb9 }, 0, .vex_lig_w1, .fma },

                .{ .vfmadd132ss, .rvm, &.{ .xmm, .xmm, .xmm_m32 }, &.{ 0x66, 0x0f, 0x38, 0x99 }, 0, .vex_lig_w0, .fma },
                .{ .vfmadd213ss, .rvm, &.{ .xmm, .xmm, .xmm_m32 }, &.{ 0x66, 0x0f, 0x38, 0xa9 }, 0, .vex_lig_w0, .fma },
                .{ .vfmadd231ss, .rvm, &.{ .xmm, .xmm, .xmm_m32 }, &.{ 0x66, 0x0f, 0x38, 0xb9 }, 0, .vex_lig_w0, .fma },

                // AVX2
                .{ .vbroadcastss, .rm, &.{ .xmm, .xmm }, &.{ 0x66, 0x0f, 0x38, 0x18 }, 0, .vex_128_w0, .avx2 },
                .{ .vbroadcastss, .rm, &.{ .ymm, .xmm }, &.{ 0x66, 0x0f, 0x38, 0x18 }, 0, .vex_256_w0, .avx2 },
                .{ .vbroadcastsd, .rm, &.{ .ymm, .xmm }, &.{ 0x66, 0x0f, 0x38, 0x19 }, 0, .vex_256_w0, .avx2 },

                .{ .vpacksswb, .rvm, &.{ .ymm, .ymm, .ymm_m256 }, &.{ 0x66, 0x0f, 0x63 }, 0, .vex_256_wig, .avx2 },
                .{ .vpackssdw, .rvm, &.{ .ymm, .ymm, .ymm_m256 }, &.{ 0x66, 0x0f, 0x6b }, 0, .vex_256_wig, .avx2 },

                .{ .vpackusdw, .rvm, &.{ .ymm, .ymm, .ymm_m256 }, &.{ 0x66, 0x0f, 0x38, 0x2b }, 0, .vex_256_wig, .avx2 },

                .{ .vpackuswb, .rvm, &.{ .ymm, .ymm, .ymm_m256 }, &.{ 0x66, 0x0f, 0x67 }, 0, .vex_256_wig, .avx2 },

                .{ .vpaddb, .rvm, &.{ .ymm, .ymm, .ymm_m256 }, &.{ 0x66, 0x0f, 0xfc }, 0, .vex_256_wig, .avx2 },
                .{ .vpaddw, .rvm, &.{ .ymm, .ymm, .ymm_m256 }, &.{ 0x66, 0x0f, 0xfd }, 0, .vex_256_wig, .avx2 },
                .{ .vpaddd, .rvm, &.{ .ymm, .ymm, .ymm_m256 }, &.{ 0x66, 0x0f, 0xfe }, 0, .vex_256_wig, .avx2 },
                .{ .vpaddq, .rvm, &.{ .ymm, .ymm, .ymm_m256 }, &.{ 0x66, 0x0f, 0xd4 }, 0, .vex_256_wig, .avx2 },

                .{ .vpaddsb, .rvm, &.{ .ymm, .ymm, .ymm_m256 }, &.{ 0x66, 0x0f, 0xec }, 0, .vex_256_wig, .avx2 },
                .{ .vpaddsw, .rvm, &.{ .ymm, .ymm, .ymm_m256 }, &.{ 0x66, 0x0f, 0xed }, 0, .vex_256_wig, .avx2 },

                .{ .vpaddusb, .rvm, &.{ .ymm, .ymm, .ymm_m256 }, &.{ 0x66, 0x0f, 0xdc }, 0, .vex_256_wig, .avx2 },
                .{ .vpaddusw, .rvm, &.{ .ymm, .ymm, .ymm_m256 }, &.{ 0x66, 0x0f, 0xdd }, 0, .vex_256_wig, .avx2 },

                .{ .vpand, .rvm, &.{ .ymm, .ymm, .ymm_m256 }, &.{ 0x66, 0x0f, 0xdb }, 0, .vex_256_wig, .avx2 },

                .{ .vpandn, .rvm, &.{ .ymm, .ymm, .ymm_m256 }, &.{ 0x66, 0x0f, 0xdf }, 0, .vex_256_wig, .avx2 },

                .{ .vpmaxsb, .rvm, &.{ .ymm, .ymm, .ymm_m256 }, &.{ 0x66, 0x0f, 0x38, 0x3c }, 0, .vex_256_wig, .avx },
                .{ .vpmaxsw, .rvm, &.{ .ymm, .ymm, .ymm_m256 }, &.{ 0x66, 0x0f,       0xee }, 0, .vex_256_wig, .avx },
                .{ .vpmaxsd, .rvm, &.{ .ymm, .ymm, .ymm_m256 }, &.{ 0x66, 0x0f, 0x38, 0x3d }, 0, .vex_256_wig, .avx },

                .{ .vpmaxub, .rvm, &.{ .ymm, .ymm, .ymm_m256 }, &.{ 0x66, 0x0f,       0xde }, 0, .vex_256_wig, .avx },
                .{ .vpmaxuw, .rvm, &.{ .ymm, .ymm, .ymm_m256 }, &.{ 0x66, 0x0f, 0x38, 0x3e }, 0, .vex_256_wig, .avx },

                .{ .vpmaxud, .rvm, &.{ .ymm, .ymm, .ymm_m256 }, &.{ 0x66, 0x0f, 0x38, 0x3f }, 0, .vex_256_wig, .avx },

                .{ .vpminsb, .rvm, &.{ .ymm, .ymm, .ymm_m256 }, &.{ 0x66, 0x0f, 0x38, 0x38 }, 0, .vex_256_wig, .avx },
                .{ .vpminsw, .rvm, &.{ .ymm, .ymm, .ymm_m256 }, &.{ 0x66, 0x0f,       0xea }, 0, .vex_256_wig, .avx },
                .{ .vpminsd, .rvm, &.{ .ymm, .ymm, .ymm_m256 }, &.{ 0x66, 0x0f, 0x38, 0x39 }, 0, .vex_256_wig, .avx },

                .{ .vpminub, .rvm, &.{ .ymm, .ymm, .ymm_m256 }, &.{ 0x66, 0x0f,       0xda }, 0, .vex_256_wig, .avx },
                .{ .vpminuw, .rvm, &.{ .ymm, .ymm, .ymm_m256 }, &.{ 0x66, 0x0f, 0x38, 0x3a }, 0, .vex_256_wig, .avx },

                .{ .vpminud, .rvm, &.{ .ymm, .ymm, .ymm_m256 }, &.{ 0x66, 0x0f, 0x38, 0x3b }, 0, .vex_256_wig, .avx },

                .{ .vpmulhw, .rvm, &.{ .ymm, .ymm, .ymm_m256 }, &.{ 0x66, 0x0f, 0xe5 }, 0, .vex_256_wig, .avx },

                .{ .vpmulld, .rvm, &.{ .ymm, .ymm, .ymm_m256 }, &.{ 0x66, 0x0f, 0x38, 0x40 }, 0, .vex_256_wig, .avx },

                .{ .vpmullw, .rvm, &.{ .ymm, .ymm, .ymm_m256 }, &.{ 0x66, 0x0f, 0xd5 }, 0, .vex_256_wig, .avx },

                .{ .vpor, .rvm, &.{ .ymm, .ymm, .ymm_m256 }, &.{ 0x66, 0x0f, 0xeb }, 0, .vex_256_wig, .avx2 },

                .{ .vpsrlw, .rvm, &.{ .ymm, .ymm, .xmm_m128 }, &.{ 0x66, 0x0f, 0xd1 }, 0, .vex_256_wig, .avx2 },
                .{ .vpsrlw, .vmi, &.{ .ymm, .ymm, .imm8     }, &.{ 0x66, 0x0f, 0x71 }, 2, .vex_256_wig, .avx2 },
                .{ .vpsrld, .rvm, &.{ .ymm, .ymm, .xmm_m128 }, &.{ 0x66, 0x0f, 0xd2 }, 0, .vex_256_wig, .avx2 },
                .{ .vpsrld, .vmi, &.{ .ymm, .ymm, .imm8     }, &.{ 0x66, 0x0f, 0x72 }, 2, .vex_256_wig, .avx2 },
                .{ .vpsrlq, .rvm, &.{ .ymm, .ymm, .xmm_m128 }, &.{ 0x66, 0x0f, 0xd3 }, 0, .vex_256_wig, .avx2 },
                .{ .vpsrlq, .vmi, &.{ .ymm, .ymm, .imm8     }, &.{ 0x66, 0x0f, 0x73 }, 2, .vex_256_wig, .avx2 },

                .{ .vpsubb, .rvm, &.{ .ymm, .ymm, .ymm_m256 }, &.{ 0x66, 0x0f, 0xf8 }, 0, .vex_256_wig, .avx2 },
                .{ .vpsubw, .rvm, &.{ .ymm, .ymm, .ymm_m256 }, &.{ 0x66, 0x0f, 0xf9 }, 0, .vex_256_wig, .avx2 },
                .{ .vpsubd, .rvm, &.{ .ymm, .ymm, .ymm_m256 }, &.{ 0x66, 0x0f, 0xfa }, 0, .vex_256_wig, .avx2 },

                .{ .vpsubsb, .rvm, &.{ .ymm, .ymm, .ymm_m256 }, &.{ 0x66, 0x0f, 0xe8 }, 0, .vex_256_wig, .avx2 },
                .{ .vpsubsw, .rvm, &.{ .ymm, .ymm, .ymm_m256 }, &.{ 0x66, 0x0f, 0xe9 }, 0, .vex_256_wig, .avx2 },

                .{ .vpsubq, .rvm, &.{ .ymm, .ymm, .ymm_m256 }, &.{ 0x66, 0x0f, 0xfb }, 0, .vex_256_wig, .avx2 },

                .{ .vpsubusb, .rvm, &.{ .ymm, .ymm, .ymm_m256 }, &.{ 0x66, 0x0f, 0xd8 }, 0, .vex_256_wig, .avx2 },
                .{ .vpsubusw, .rvm, &.{ .ymm, .ymm, .ymm_m256 }, &.{ 0x66, 0x0f, 0xd9 }, 0, .vex_256_wig, .avx2 },

                .{ .vpunpckhbw,  .rvm, &.{ .ymm, .ymm, .ymm_m256 }, &.{ 0x66, 0x0f, 0x68 }, 0, .vex_256_wig, .avx2 },
                .{ .vpunpckhwd,  .rvm, &.{ .ymm, .ymm, .ymm_m256 }, &.{ 0x66, 0x0f, 0x69 }, 0, .vex_256_wig, .avx2 },
                .{ .vpunpckhdq,  .rvm, &.{ .ymm, .ymm, .ymm_m256 }, &.{ 0x66, 0x0f, 0x6a }, 0, .vex_256_wig, .avx2 },
                .{ .vpunpckhqdq, .rvm, &.{ .ymm, .ymm, .ymm_m256 }, &.{ 0x66, 0x0f, 0x6d }, 0, .vex_256_wig, .avx2 },

                .{ .vpunpcklbw,  .rvm, &.{ .ymm, .ymm, .ymm_m256 }, &.{ 0x66, 0x0f, 0x60 }, 0, .vex_256_wig, .avx2 },
                .{ .vpunpcklwd,  .rvm, &.{ .ymm, .ymm, .ymm_m256 }, &.{ 0x66, 0x0f, 0x61 }, 0, .vex_256_wig, .avx2 },
                .{ .vpunpckldq,  .rvm, &.{ .ymm, .ymm, .ymm_m256 }, &.{ 0x66, 0x0f, 0x62 }, 0, .vex_256_wig, .avx2 },
                .{ .vpunpcklqdq, .rvm, &.{ .ymm, .ymm, .ymm_m256 }, &.{ 0x66, 0x0f, 0x6c }, 0, .vex_256_wig, .avx2 },

                .{ .vpxor, .rvm, &.{ .ymm, .ymm, .ymm_m256 }, &.{ 0x66, 0x0f, 0xef }, 0, .vex_256_wig, .avx2 },
            };
            // zig fmt: on
        };
        for (encodings.table) |entry| mnemonic_map[@intFromEnum(entry[0])].len += 1;
        var data_storage: [encodings.table.len]Data = undefined;
        var storage_i: usize = 0;
        for (&mnemonic_map) |*value| {
            value.ptr = data_storage[storage_i..].ptr;
            storage_i += value.len;
        }
        var mnemonic_i: [mnemonic_count]usize = .{0} ** mnemonic_count;
        const ops_len = @typeInfo(std.meta.FieldType(Data, .ops)).array.len;
        const opc_len = @typeInfo(std.meta.FieldType(Data, .opc)).array.len;
        for (encodings.table) |entry| {
            const i = &mnemonic_i[@intFromEnum(entry[0])];
            mnemonic_map[@intFromEnum(entry[0])][i.*] = .{
                .op_en = entry[1],
                .ops = (entry[2] ++ .{.none} ** (ops_len - entry[2].len)).*,
                .opc_len = entry[3].len,
                .opc = (entry[3] ++ .{undefined} ** (opc_len - entry[3].len)).*,
                .modrm_ext = entry[4],
                .mode = entry[5],
                .feature = entry[6],
            };
            i.* += 1;
        }
        const final_storage = data_storage;
        var final_map: [mnemonic_count][]const Data = .{&.{}} ** mnemonic_count;
        storage_i = 0;
        for (&final_map, mnemonic_map) |*value, wip_value| {
            value.ptr = final_storage[storage_i..].ptr;
            value.len = wip_value.len;
            storage_i += value.len;
        }
        break :init final_map;
    };
};

pub const DisassemblerOptions = struct {
    display: DisplayFlags = .{},
    base_address: enum { absolute, relative } = .absolute,
    address_style: enum { hex, dec } = .hex,
    sentinel: ?Mnemonic = null,

    pub const DisplayFlags = packed struct {
        function_address: bool = true,
        address: bool = false,
        bytes: bool = false,
        x64_intel_syntax: bool = true,
    };
};

fn showAddress(ptr: [*]const u8, opts: *const DisassemblerOptions, writer: anytype) !void {
    if (opts.display.function_address) {
        switch (opts.address_style) {
            .hex => try writer.print("[{x:0<16}]:", .{@intFromPtr(ptr)}),
            .dec => try writer.print("[{d: <16}]:", .{@intFromPtr(ptr)}),
        }
    }
}

pub fn disas(memory: []const u8, opts: DisassemblerOptions, writer: anytype) !void {
    var disassembler = Disassembler.init(memory);
    var lastPos = disassembler.pos;

    try writer.writeAll("\n");

    var options = opts;

    if (!(options.display.address or options.display.bytes or options.display.x64_intel_syntax)) {
        if (@inComptime()) {
            @compileLog("no display options enabled, defaulting to x64_intel_syntax");
        } else {
            log.warn("no display options enabled, defaulting to x64_intel_syntax", .{});
        }
        options.display.x64_intel_syntax = true;
    }

    const justSyntax = !(options.display.address or options.display.bytes);
    const justBytes = !(options.display.address or options.display.x64_intel_syntax);
    const justAddress = !(options.display.bytes or options.display.x64_intel_syntax);

    

    if (options.display.address) {
        if (justAddress) {
            try writer.writeAll("  ╿ ");
            try showAddress(memory.ptr, &options, writer);
        } else {
            try writer.writeAll("                    ╿ ");
            try showAddress(memory.ptr, &options, writer);
        }
    }
    if (options.display.bytes) {
        if (justBytes) {
            try writer.writeAll("  ╿ ");
            try showAddress(memory.ptr, &options, writer);
        } else if (options.display.x64_intel_syntax) {
            try writer.writeAll("                                    ╿");
        }
    }
    if (justSyntax) {
        try writer.writeAll("  ╿ ");
        try showAddress(memory.ptr, &options, writer);
    }

    try writer.writeAll("\n");

    if (options.display.address) {
        if (justAddress) try writer.writeAll("  ");
        try writer.writeAll("  address");
        if (!justAddress) try writer.writeAll("           │");
    }
    if (options.display.bytes) {
        if (justBytes) try writer.writeAll("  │");
        try writer.writeAll("  bytes");
        if (options.display.x64_intel_syntax) try writer.writeAll("                             │");
    }
    if (options.display.x64_intel_syntax) {
        if (justSyntax) try writer.writeAll("  │");
        try writer.writeAll("  x64-intel-syntax");
    }

    try writer.writeAll("\n");

    if (options.display.address) {
        if (justAddress) {
            try writer.writeAll("╾─┼");
        } else {
            try writer.writeAll("╾──");
        }

        try writer.writeAll("─────────────────");

        if (justAddress) {
            try writer.writeAll("╼");
        } else {
            try writer.writeAll("┼");
        }
    }
    if (options.display.bytes) {
        if (options.display.address) {
            try writer.writeAll("──");
        } else {
            try writer.writeAll("╾─");
        }
        if (!options.display.x64_intel_syntax) {
            try writer.writeAll("┼");
        }
        try writer.writeAll("──────────────────────────────────");
        if (options.display.x64_intel_syntax) {
            try writer.writeAll("┼");
        } else {
            try writer.writeAll("╼");
        }
    }
    if (options.display.x64_intel_syntax) {
        if (options.display.address or options.display.bytes) {
            try writer.writeAll("───────────────────────────────────╼");
        } else {
            try writer.writeAll("╾─┼──────────────────────────────────╼");
        }
    }

    try writer.writeAll("\n");

    if (options.display.address) {
        if (justAddress) {
            try writer.writeAll("  │");
        } else {
            try writer.writeAll("                    │");
        }
    }
    if (options.display.bytes) {
        if (justBytes) {
            try writer.writeAll("  │");
        } else if (options.display.x64_intel_syntax) {
            try writer.writeAll("                                    │");
        }
    }
    if (justSyntax and options.display.x64_intel_syntax) try writer.writeAll("  │");

    try writer.writeAll("\n");

    const base = switch (options.base_address) {
        .absolute => @intFromPtr(memory.ptr),
        .relative => 0,
    };

    while (try disassembler.next()) |instr| {
        if (options.display.address) {
            if (justAddress) {
                try writer.writeAll("  │");
            }

            switch (options.address_style) {
                .hex => try writer.print("  {x:0<16}", .{base + lastPos}),
                .dec => try writer.print("  {d: <16}", .{base + lastPos}),
            }

            if (!justAddress) try writer.writeAll("  │");
        }

        if (options.display.bytes) {
            if (justBytes) try writer.writeAll("  │");

            var i: u120 = 0;
            const bytes = memory[lastPos..disassembler.pos];
            for (0..@min(bytes.len, @sizeOf(u120))) |index| {
                i |= @as(u120, bytes[index]) << @as(std.meta.Int(.unsigned, math.log2(@bitSizeOf(u120))), @intCast(index * 8));
            }

            try writer.print("  {x: <32}", .{i});
            if (options.display.x64_intel_syntax) try writer.writeAll("  │");
        }

        if (options.display.x64_intel_syntax) {
            if (justSyntax) try writer.writeAll("  │");
            try writer.print("  {}", .{instr});
        }

        try writer.writeAll("\n");

        lastPos = disassembler.pos;

        if (options.sentinel) |x| {
            if (instr.encoding.mnemonic == x) break;
        }
    }

    if (options.display.address) {
        if (justAddress) {
            try writer.writeAll("  ╽");
        } else {
            try writer.writeAll("                    ╽");
        }
    }
    if (options.display.bytes) {
        if (justBytes) {
            try writer.writeAll("  ╽");
        } else if (options.display.x64_intel_syntax) {
            try writer.writeAll("                                    ╽");
        }
    }
    if (justSyntax) try writer.writeAll("  ╽");

    try writer.writeAll("\n");
}
