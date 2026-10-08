"""Build the optional exact feature encoder: python setup_native.py build_ext --inplace."""

import sys

from setuptools import Extension, setup

setup(
    name="windup-native-encoder",
    version="0.1.0",
    packages=[],
    ext_modules=[
        Extension(
            "_windup_native",
            sources=["native/encoder.c"],
            depends=[
                "native/blake2/blake2b-ref.c",
                "native/blake2/blake2.h",
                "native/blake2/blake2-impl.h",
            ],
            include_dirs=["native/blake2"],
            extra_compile_args=["/O2"] if sys.platform == "win32" else ["-O3"],
        )
    ],
)
