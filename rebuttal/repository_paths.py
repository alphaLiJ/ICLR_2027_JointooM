"""Repository-local source paths and configurable experiment inputs."""
import os
from importlib import metadata
import importlib
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
os.environ.setdefault('JOINTLOOM_REPOSITORY', str(REPO))
ARTIFACTS = Path(os.environ.get('JOINTLOOM_ARTIFACTS', REPO / 'artifacts/rebuttal')).expanduser()
PYTHON = Path(os.environ.get('JOINTLOOM_PYTHON', sys.executable)).expanduser()


def input_path(variable, default):
    return Path(os.environ.get(variable, default)).expanduser()


def add_paths(*paths):
    """Earlier arguments take precedence; each experiment uses its own process."""
    for path in reversed(paths):
        value = str(path)
        if value in sys.path:
            sys.path.remove(value)
        sys.path.insert(0, value)


def installed_pogema(version):
    """Keep spawned workers on the experiment's installed Pogema version."""
    distribution = metadata.distribution('pogema')
    if distribution.version != version:
        raise RuntimeError(f'This workflow uses pogema=={version}; installed: {distribution.version}')
    location = Path(distribution.locate_file('')).resolve()
    add_paths(location)
    module = importlib.import_module('pogema')
    expected = (location / 'pogema').resolve()
    if expected not in Path(module.__file__).resolve().parents:
        raise RuntimeError('Start this workflow in a fresh Python process using its installed Pogema package')
    return location


def source_paths(native=None, upstream=None, extra=()):
    paths = [native] if native is not None else []
    if upstream is not None:
        paths.append(upstream)
    paths.extend(extra)
    paths.extend([REPO / 'src', REPO / 'experiments/runners',
                  REPO / 'third_party/lagat', REPO / 'src/real_expert_alg', REPO])
    add_paths(*paths)
