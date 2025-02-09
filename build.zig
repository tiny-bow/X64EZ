const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    _ = b.addModule("X64EZ", .{
        .root_source_file = b.path("src/X64EZ.zig"),
        .target = target,
        .optimize = optimize,
    });
}
