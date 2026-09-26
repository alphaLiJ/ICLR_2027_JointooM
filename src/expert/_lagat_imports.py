from __future__ import annotations

import os
import sys


def repo_root_from_module_file(module_file: str) -> str:
    resolved = os.path.realpath(module_file)
    return os.path.dirname(os.path.dirname(os.path.dirname(resolved)))


def _lagat_candidates(project_root: str) -> list[str]:
    return [
        os.path.join(project_root, "third_party", "lagat"),
        os.path.join(project_root, "other", "lagat"),
    ]


def ensure_lagat_on_path(project_root: str) -> str:
    for candidate in _lagat_candidates(project_root):
        if os.path.isdir(candidate):
            if candidate not in sys.path:
                sys.path.insert(0, candidate)
            return candidate

    searched = ", ".join(_lagat_candidates(project_root))
    raise ModuleNotFoundError(
        f"Could not locate the lagat checkout needed for MAGAT+ imports. Searched: {searched}"
    )
