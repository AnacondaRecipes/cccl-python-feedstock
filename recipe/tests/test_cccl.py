"""Smoke tests for the cccl-python package.

These run on CPU-only CI machines, so nothing here needs a GPU or a driver.
Checks that cannot be evaluated without the CUDA bindings are skipped.
"""

import sys
from importlib.resources import files
from pathlib import Path

import cuda.cccl
import cuda.compute


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def check_version(expected):
    if expected is None:
        return
    check(
        cuda.cccl.__version__ == expected,
        f"cuda.cccl.__version__ is {cuda.cccl.__version__!r}, expected {expected!r}",
    )


def check_bundled_headers():
    include_dir = files("cuda.cccl.headers.include")
    for relpath in ("cub/version.cuh", "thrust/version.h", "cuda/version"):
        header = include_dir
        # Join one component at a time: not every Traversable implementation
        # accepts multiple arguments before Python 3.11.
        for part in relpath.split("/"):
            header = header / part
        check(header.is_file(), f"bundled CCCL header is missing: {relpath}")


def check_bundled_libraries():
    package_dir = Path(cuda.compute.__file__).parent
    libraries = [
        path
        for pattern in ("cu*/cccl/libcccl.c.parallel.*", "cu*/cccl/cccl.c.parallel.dll")
        for path in package_dir.glob(pattern)
        if path.is_file() and path.suffix in (".so", ".dll", ".dylib")
    ]
    check(
        libraries,
        f"bundled cccl.c.parallel library is missing from {package_dir}",
    )
    extensions = [
        path
        for pattern in ("cu*/_bindings_impl.*.so", "cu*/_bindings_impl.*.pyd")
        for path in package_dir.glob(pattern)
        if path.is_file()
    ]
    check(
        extensions,
        f"compiled _bindings_impl extension is missing from {package_dir}",
    )


def check_build_backend():
    from cuda.compute._build_info import USING_V2

    check(USING_V2 is False, "this build should use the v1 (NVRTC) backend")


def check_binding_smoke_tests():
    import numpy as np

    from cuda.compute import _bindings

    if not getattr(_bindings, "_BINDINGS_AVAILABLE", False):
        print("CUDA bindings unavailable (no driver?); skipping binding-level checks")
        return

    from cuda.compute import gpu_struct
    from cuda.compute._utils import sanitize_identifier

    check(
        sanitize_identifier("<lambda>") == "_lambda_",
        "sanitize_identifier() no longer turns '<lambda>' into '_lambda_'",
    )

    try:
        gpu_struct({"field name": np.float32})
    except ValueError:
        pass
    else:
        raise AssertionError("gpu_struct() accepted an invalid field name")


def main():
    expected_version = sys.argv[1] if len(sys.argv) > 1 else None
    check_version(expected_version)
    check_bundled_headers()
    check_bundled_libraries()
    check_build_backend()
    check_binding_smoke_tests()
    print("cccl-python smoke tests passed")


if __name__ == "__main__":
    main()
