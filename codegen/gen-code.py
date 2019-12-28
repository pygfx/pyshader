import subprocess

def blacken(src, ll=88):
    p = subprocess.Popen(
        ["black", "-l", str(ll), "-"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    p.stdin.write(src.encode())
    p.stdin.close()
    result = p.stdout.read().decode()
    log = p.stderr.read().decode()
    if "error" in log.lower():
        raise RuntimeError(log)
    return result


# %% Generate dis.py


peamble = '''
"""
Python opcodes

Taken from https://github.com/beeware/batavia/blob/master/batavia/modules/dis.js
which is Copyright Russell Keith-Magee, and licensed under the 3-clause BSD.
"""

# NOTE: THIS CODE IS AUTOGENERATED; DO NOT EDIT
'''


# Read js version
js = open("dis.js", "rb").read().decode()

# Process and generate lines
pylines = []
for line in js.splitlines():
    line = line.rstrip().replace("//", "#").replace(".push(", ".append(")
    line2 = line.lstrip()
    line3 = line2.split("#")[0]
    indent = line[: len(line) - len(line2)]
    if line2.startswith(("/*", "*", "module.exports", "CO_GENERATOR:")):
        continue
    elif ": {" in line3 or ": [" in line3:
        name, _, rest = line2.partition(":")
        line = f"{indent}{name} = {rest.strip().strip(', ')}"
    elif line.startswith("var dis ="):
        line = "class dis:"
    elif line2 == "}":
        continue
    elif line.startswith("function "):
        line = line.replace("function ", "def ").replace("{", ":")
        line = line.replace(" :", ":")
    elif line.startswith("for (var op = 0; op < 256; op++)"):
        line = "for op in range(256):"
    elif line.count("+ op +"):
        line = line.replace("+ op +", "+ str(op) +")
    pylines.append(line.rstrip())

code = "\n".join(pylines).replace("\n\n\n\n", "\n\n").replace("\n\n\n", "\n\n")
code = peamble.strip() + "\n\n\n" + code.strip() + "\n"
code = blacken(code)

with open("../spirv/_dis.py", "wb") as f:

    f.write(code.encode())


# %% Generate _spirv_constants.py

preamble = '''
"""
All the SpirV constants. Generated by the Python file provided by Khronos at
https://raw.githubusercontent.com/KhronosGroup/SPIRV-Headers/master/include/spirv/unified1/spirv.py
"""

# NOTE: THIS CODE IS AUTOGENERATED; DO NOT EDIT

builtins = {}

class Enum(int):
    """ Enum (integer) with a meaningfull repr. """
    def __new__(cls, name, value):
        base = int.__new__(cls, value)
        base.name = name
        if name.startswith("BuiltIn_"):
            builtins[name[8:]] = base
        return base
    def __repr__(self):
        return self.name
'''


# Get dict
ns = {}
exec(open("spirv.py", "rb").read().decode(), ns, ns)
spv = ns["spv"]

# Process and generate lines
pylines = []
for key, val in spv.items():
    if isinstance(val, dict):
        pylines.append("")
        for subkey, val in val.items():
            fullkey = subkey if key.startswith("Op") else key + "_" + subkey
            pylines.append(f"{fullkey} = Enum({fullkey!r}, {val!r})")
    else:
        rval = (
            hex(val) if key in ("MagicNumber", "Version", "OpCodeMask") else repr(val)
        )
        pylines.append(f"{key} = Enum({key!r}, {rval})")

code = preamble.strip() + "\n\n\n" + "\n".join(pylines) + "\n"
code = blacken(code)

with open("../spirv/_spirv_constants.py", "wb") as f:
    f.write(code.encode())
