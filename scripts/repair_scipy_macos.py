"""Repair malformed SciPy Mach-O headers rejected by newer macOS loaders.

Run with the notebook's Python interpreter. Original libraries are preserved as
*.so.before-macos-repair. Reinstalling SciPy replaces this local workaround.
Upstream report: https://github.com/scipy/scipy/issues/25635
"""

import importlib.util
import os
from pathlib import Path
import platform
import shutil
import struct
import subprocess
import tempfile


def repair(path):
    data = bytearray(path.read_bytes())
    # Only handle little-endian 64-bit Mach-O files, as in Apple Silicon wheels.
    if struct.unpack_from("<I", data)[0] != 0xFEEDFACF:
        raise RuntimeError(f"Unsupported Mach-O format: {path}")
    commands = struct.unpack_from("<I", data, 16)[0]
    position = 32
    changed = False
    for _ in range(commands):
        command, size = struct.unpack_from("<II", data, position)
        if command == 0x19:  # LC_SEGMENT_64
            sections = struct.unpack_from("<I", data, position + 64)[0]
            for index in range(sections):
                section = position + 72 + 80 * index
                name = data[section:section + 16].split(b"\0")[0]
                flags = struct.unpack_from("<I", data, section + 64)[0]
                offset = struct.unpack_from("<I", data, section + 48)[0]
                if name == b"__thread_bss" and flags & 0xFF == 0x12 and offset:
                    # S_THREAD_LOCAL_ZEROFILL has no bytes in the file.
                    struct.pack_into("<I", data, section + 48, 0)
                    changed = True
        position += size
    if not changed:
        return

    backup = path.with_suffix(path.suffix + ".before-macos-repair")
    if backup.exists():
        raise RuntimeError(f"Backup already exists; inspect before retrying: {backup}")
    # Replace atomically rather than modifying a possible uv cache hardlink.
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(data)
    try:
        shutil.copymode(path, temporary)
        subprocess.run(
            ["codesign", "--force", "--sign", "-", str(temporary)], check=True
        )
        shutil.copy2(path, backup)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Repaired: {path.name}; backup: {backup.name}")


if __name__ == "__main__":
    if platform.system() != "Darwin":
        raise SystemExit("This workaround applies only to macOS.")
    scipy = importlib.util.find_spec("scipy")
    if scipy is None:
        raise SystemExit("SciPy is not installed in this Python environment.")
    folder = Path(scipy.origin).parent
    libraries = sorted(folder.rglob("*.so"))
    if not libraries:
        raise SystemExit(f"No SciPy libraries found in {folder}")
    for library in libraries:
        repair(library)
