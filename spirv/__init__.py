"""
Tools to work with SpirV in Python, including a Python to SpirV compiler.
"""

__version__ = "0.2.0"


from ._module import SpirVModule
from .raw import bytes2spirv, file2spirv
from .py import python2spirv
# from .glsl import glsl2spirv
# from .wasl import wasl2spirv

from ._types import vec2, vec3, vec4  # todo: and more
