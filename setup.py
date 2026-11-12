"""Builds the optional Backstitch backend, pathrag._backstitch.

    BACKSTITCH_DIR=~/code/Backstitch python setup.py build_ext --inplace

Backstitch is not part of this repo. Point BACKSTITCH_DIR at a local
checkout of it (default: ../Backstitch next to this repo). Only its index
core, repair and pipeline sources are compiled. Without a checkout, nothing
is built and pathrag runs on its FAISS backends alone.
"""
import os
import platform
from pathlib import Path

from setuptools import setup

ROOT = Path(__file__).resolve().parent
BACKSTITCH = Path(os.environ.get("BACKSTITCH_DIR", ROOT.parent / "Backstitch")).expanduser()
BGR = BACKSTITCH / "src"
CORE = ["index.cpp", "prune.cpp", "dataset_io.cpp", "repair/locate.cpp", "repair/salc.cpp",
        "repair/wolverine.cpp", "repair/repairer.cpp", "pipeline/engine.cpp"]

ext_modules, cmdclass = [], {}
if (BGR / "bgr/index.cpp").exists():
    from pybind11.setup_helpers import Pybind11Extension, build_ext

    flags = ["-O3", "-DNDEBUG"]
    flags.append("-mcpu=native" if platform.machine() in ("arm64", "aarch64") else "-march=native")
    ext_modules = [Pybind11Extension(
        "pathrag._backstitch",
        ["bindings/pybackstitch.cpp"] + [str(BGR / "bgr" / f) for f in CORE],
        include_dirs=[str(BGR)],
        cxx_std=20,
        extra_compile_args=flags,
        extra_link_args=["-pthread"],
    )]
    cmdclass = {"build_ext": build_ext}
else:
    print(f"no Backstitch checkout at {BACKSTITCH}: building without the backstitch backend")

setup(name="pathrag", packages=["pathrag"], ext_modules=ext_modules, cmdclass=cmdclass)
