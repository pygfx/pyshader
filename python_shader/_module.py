from ._generator_bc import Bytecode2SpirVGenerator


class ShaderModule:
    """ Representation of a shader module. Is basically a wrapper around the
    source input and the bytes representing the actual SpirV code.
    """

    def __init__(self, input, bytecode, description):
        self._input = input
        self._bytecode = bytecode
        self._description = description

    def __repr__(self):
        return f"<ShaderModule {self._description} at 0x{hex(id(self))}>"

    @property
    def description(self):
        """ The shaders's (source) description.
        """
        return self._description

    @property
    def input(self):
        """ The input used to produce this SpirV module.
        """
        return self._input

    def to_bytecode(self):
        """ Get the bytecode representing this shader module.
        Note that the bytecode is not yet part of the public API; it can change.
        """
        return self._bytecode

    def to_spirv(self):
        """ Get the binary representation of the SpirV module (bytes).
        """
        # Store the generator - great during dev
        self.gen = Bytecode2SpirVGenerator()
        self.gen.convert(self._bytecode)
        return self.gen.dump()  # bytes