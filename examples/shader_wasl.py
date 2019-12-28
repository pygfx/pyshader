"""
Compile a wasl shader to SpirV, validate it, and show the SpirV disassembly.

WASL is experimental, currently broken, and probably being deprecated :)
"""


from spirv.wasl import wasl2spirv

vertex_code = """
fn main (
    index: input i32 VertexId,  # VertexID or VertexIndex
    pos: output vec4 Position,
) {

    positions = Array(
        vec2(+0.0, -0.5),
        vec2(+0.5, +0.5),
        vec2(-0.5, +0.5),
    )

    pos = vec4(positions[index], 0.0, 1.0)
}
""".lstrip()

module = wasl2spirv(vertex_code, "vert")

module.validate()
print(module.disassble())
