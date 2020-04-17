"""
Tests that run a compute shader and validate the outcome.
With this we can validate arithmetic, control flow etc.
"""


import ctypes

import python_shader
from python_shader import f32, i32, ivec2, ivec3, ivec4, vec2, vec3, vec4, Array  # noqa

import wgpu.backends.rs  # noqa
from wgpu.utils import compute_with_buffers

import pytest
from testutils import can_use_wgpu_lib, iters_equal
from testutils import validate_module, run_test_and_print_new_hashes


def test_index():
    @python2shader_and_validate
    def compute_shader(
        index: ("input", "GlobalInvocationId", i32), data2: ("buffer", 1, Array(i32)),
    ):
        data2[index] = index

    skip_if_no_wgpu()

    inp_arrays = {}
    out_arrays = {1: ctypes.c_int32 * 20}
    out = compute_with_buffers(inp_arrays, out_arrays, compute_shader)

    assert iters_equal(out[1], range(20))


def test_copy():
    @python2shader_and_validate
    def compute_shader(
        index: ("input", "GlobalInvocationId", i32),
        data1: ("buffer", 0, Array(i32)),
        data2: ("buffer", 1, Array(i32)),
    ):
        data2[index] = data1[index]

    skip_if_no_wgpu()

    inp_arrays = {0: (ctypes.c_int32 * 20)(*range(20))}
    out_arrays = {1: ctypes.c_int32 * 20}
    out = compute_with_buffers(inp_arrays, out_arrays, compute_shader)

    assert iters_equal(out[1], range(20))


def test_copy_vec2():
    @python2shader_and_validate
    def compute_shader(
        index: ("input", "GlobalInvocationId", i32),
        data1: ("buffer", 0, Array(vec2)),
        data2: ("buffer", 1, Array(vec2)),
        data3: ("buffer", 2, Array(ivec2)),
    ):
        data2[index] = data1[index].xy
        data3[index] = ivec2(index, index)

    skip_if_no_wgpu()

    inp_arrays = {0: (ctypes.c_float * 60)(*range(60))}
    out_arrays = {1: ctypes.c_float * 60, 2: ctypes.c_int32 * 60}
    out = compute_with_buffers(inp_arrays, out_arrays, compute_shader, n=30)

    assert iters_equal(out[1], range(60))
    assert iters_equal(out[2][0::2], range(30))
    assert iters_equal(out[2][1::2], range(30))


def test_copy_vec3():
    @python2shader_and_validate
    def compute_shader(
        index: ("input", "GlobalInvocationId", i32),
        data1: ("buffer", 0, Array(vec3)),
        data2: ("buffer", 1, Array(vec3)),
        data3: ("buffer", 2, Array(ivec3)),
    ):
        data2[index] = data1[index].xyz
        data3[index] = ivec3(index, index, index)

    # # Equivalent shader in GLSL
    # compute_shader = python_shader.dev.glsl2spirv("""
    #     #version 450
    #     layout(std430 , set=0, binding=0) buffer Foo1 { vec3[] data1; };
    #     layout(std430 , set=0, binding=1) buffer Foo2 { vec3[] data2; };
    #     layout(std430 , set=0, binding=2) buffer Foo3 { ivec3[] data3; };
    #
    #     void main() {
    #         uint index = gl_GlobalInvocationID.x;
    #         data2[index] = data1[index];
    #         data3[index] = ivec3(index, index, index);
    #     }
    # """, "compute")

    skip_if_no_wgpu()

    inp_arrays = {0: (ctypes.c_float * 60)(*range(60))}
    out_arrays = {1: ctypes.c_float * 60, 2: ctypes.c_int32 * 60}
    out = compute_with_buffers(inp_arrays, out_arrays, compute_shader, n=20)

    # NOPE, buffers alignments are rounded to vec4 ...
    # https://stackoverflow.com/questions/38172696/

    # assert iters_equal(out[1], range(60))
    assert iters_equal(out[1][0::4], range(0, 60, 4))
    assert iters_equal(out[1][1::4], range(1, 60, 4))
    assert iters_equal(out[1][2::4], range(2, 60, 4))
    # Depending on your driver, this might or might not work
    align_ok = iters_equal(out[1][3::4], range(3, 60, 4))
    align_fail = iters_equal(out[1][3::4], [0 for i in range(3, 60, 4)])
    assert align_ok or align_fail

    if align_ok:
        assert iters_equal(out[2][0::3], range(20))
        assert iters_equal(out[2][1::3], range(20))
        assert iters_equal(out[2][2::3], range(20))
    if align_fail:
        assert iters_equal(out[2][0::4], range(15))
        assert iters_equal(out[2][1::4], range(15))
        assert iters_equal(out[2][2::4], range(15))
        assert iters_equal(out[2][3::4], [0 for i in range(15)])


def test_copy_vec4():
    @python2shader_and_validate
    def compute_shader(
        index: ("input", "GlobalInvocationId", i32),
        data1: ("buffer", 0, Array(vec4)),
        data2: ("buffer", 1, Array(vec4)),
        data3: ("buffer", 2, Array(ivec4)),
    ):
        data2[index] = data1[index].xyzw
        data3[index] = ivec4(index, index, index, index)

    skip_if_no_wgpu()

    inp_arrays = {0: (ctypes.c_float * 60)(*range(60))}
    out_arrays = {1: ctypes.c_float * 60, 2: ctypes.c_int32 * 60}
    out = compute_with_buffers(inp_arrays, out_arrays, compute_shader, n=15)

    assert iters_equal(out[1], range(60))
    assert iters_equal(out[2][0::4], range(15))
    assert iters_equal(out[2][1::4], range(15))
    assert iters_equal(out[2][2::4], range(15))
    assert iters_equal(out[2][3::4], range(15))


# %% Utils for this module


def python2shader_and_validate(func):
    m = python_shader.python2shader(func)
    assert m.input is func
    validate_module(m, HASHES)
    return m


def skip_if_no_wgpu():
    if not can_use_wgpu_lib:
        raise pytest.skip(msg="SpirV validated, but not run (cannot use wgpu)")


HASHES = {
    "test_index.compute_shader": ("5b4be829de8c83e5", "e677b4cc6992630a"),
    "test_copy.compute_shader": ("7b03b3564a72be3c", "46f084870ce2681b"),
    "test_copy_vec2.compute_shader": ("5aa1d35b110a2318", "e12816b3c5a511a4"),
    "test_copy_vec3.compute_shader": ("91d4532ab3aec94c", "648a9966f2cd20fc"),
    "test_copy_vec4.compute_shader": ("5c4494844138daf8", "828169151df9a719"),
}


if __name__ == "__main__":
    run_test_and_print_new_hashes(globals())