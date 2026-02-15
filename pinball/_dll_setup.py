"""Windows DLL search directory setup — imported early by ``__init__.py``.

Python 3.8+ no longer searches PATH for DLL dependencies of extension
modules (.pyd) or ctypes-loaded DLLs.  We call ``os.add_dll_directory()``
so that runtime libraries such as ``libopenblas.dll`` and
``libgfortran-5.dll`` can be found.

For wheel installs this is a no-op because ``delvewheel`` bundles the
required DLLs directly into the package.  It is essential, however, for
editable installs and CI where DLLs live in external directories.
"""

import os
import sys


def _setup_windows_dll_dirs() -> None:
    if sys.platform != "win32":
        return

    added: set[str] = set()

    # 1. Package directory + vendored-DLL directories
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    # delvewheel puts vendored DLLs in  <package>.libs/  (sibling of pkg_dir)
    site_libs = os.path.join(os.path.dirname(pkg_dir), "pinball.libs")
    for d in (pkg_dir, os.path.join(pkg_dir, ".libs"), site_libs):
        if os.path.isdir(d) and d not in added:
            try:
                os.add_dll_directory(d)
                added.add(d)
            except OSError:
                pass

    # 2. PATH entries (CI / dev installs: C:\openblas\bin, MinGW bin, …)
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        entry = entry.strip()
        if entry and os.path.isdir(entry) and entry not in added:
            try:
                os.add_dll_directory(entry)
                added.add(entry)
            except OSError:
                pass


_setup_windows_dll_dirs()
