"""
Compile a glsl shader to SpirV, validate it, and show the SpirV disassembly.

Note that you need to have spir-v tools installed for this to work.
"""

from spirv.glsl import glsl2spirv


vertex_code = """
#version 450

out gl_PerVertex {
    vec4 gl_Position;
};

const vec2 positions[3] = vec2[3](
    vec2(0.0, -0.5),
    vec2(0.5, 0.5),
    vec2(-0.5, 0.5)
);

void main() {
    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);

}

""".lstrip()


module = glsl2spirv(vertex_code, "vert")

module.validate()
print(module.disassble())
