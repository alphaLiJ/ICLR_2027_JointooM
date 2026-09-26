from __future__ import annotations

import ctypes
from pathlib import Path


def test_lacam_library_is_built_outside_the_source_tree():
    from real_expert_alg.lacam import build_lacam_library, lacam_library_path

    library = build_lacam_library()
    expected = lacam_library_path()
    source = Path(__file__).parents[2] / "src" / "real_expert_alg" / "lacam"

    assert library == expected
    assert library.is_file()
    assert source not in library.parents
    assert not (source / "CMakeCache.txt").exists()
    assert not (source / "liblacam.so").exists()
    ctypes.CDLL(str(library))
