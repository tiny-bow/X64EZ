const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    _ = b.addModule("r64", .{
        .root_source_file = b.path("src/r64.zig"),
        .target = target,
        .optimize = optimize,
    });
}
