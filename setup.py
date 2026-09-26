import os
import site

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch


def _resolve_map_profile() -> tuple[int, int]:
    env_profile = os.environ.get('MAPF_CUDA_MAP_PROFILE', '').strip().lower()
    if not env_profile:
        return 128, 128

    if 'x' not in env_profile:
        raise ValueError(
            "MAPF_CUDA_MAP_PROFILE must look like '<width>x<height>', "
            f"got {env_profile!r}"
        )

    width_str, height_str = env_profile.split('x', 1)
    try:
        width = int(width_str)
        height = int(height_str)
    except ValueError as exc:
        raise ValueError(
            "MAPF_CUDA_MAP_PROFILE must contain integer width/height, "
            f"got {env_profile!r}"
        ) from exc

    if width <= 0 or height <= 0:
        raise ValueError(
            f"MAPF_CUDA_MAP_PROFILE must be positive, got {env_profile!r}"
        )
    if height % 32 != 0:
        raise ValueError(
            "MAPF_CUDA_MAP_PROFILE height must be a multiple of 32 so the "
            f"compressed column layout stays integral, got {env_profile!r}"
        )

    return width, height


def _map_profile_defines(map_w: int, map_h: int) -> list[str]:
    return [
        f'-DMAP_W={map_w}',
        f'-DMAP_H={map_h}',
    ]


def _resolve_cuda_arch() -> str:
    # Allow manual override for remote builds or cross-compilation.
    env_arch = os.environ.get('MAPF_CUDA_ARCH', '').strip()
    if env_arch:
        return env_arch

    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        return f'{major}{minor}'

    return '89'


def _collect_nvidia_wheel_paths(kind: str) -> list[str]:
    paths: list[str] = []
    for root in site.getsitepackages():
        nvidia_root = os.path.join(root, 'nvidia')
        if not os.path.isdir(nvidia_root):
            continue

        for package in sorted(os.listdir(nvidia_root)):
            candidate = os.path.join(nvidia_root, package, kind)
            if os.path.isdir(candidate):
                paths.append(candidate)

    # Keep ordering stable while removing duplicates.
    return list(dict.fromkeys(paths))


cuda_arch = _resolve_cuda_arch()
map_w, map_h = _resolve_map_profile()
map_profile_defines = _map_profile_defines(map_w, map_h)
extra_include_dirs = _collect_nvidia_wheel_paths('include')
extra_library_dirs = _collect_nvidia_wheel_paths('lib')

cxx_args = [
    '-O3',
    '-std=c++17',
    '-march=native',
    '-fopenmp',
    *map_profile_defines,
]

nvcc_args = [
    '-O3',
    '-std=c++17',
    '-gencode', f'arch=compute_{cuda_arch},code=sm_{cuda_arch}',
    *map_profile_defines,
    '--use_fast_math',          
    '--expt-relaxed-constexpr', 
    '-Xptxas', '-O3',           
    '--disable-warnings',      
]

setup(
    name='mapf-cuda-system',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    ext_modules=[
        CUDAExtension(
            name='grid_world_cpp',
            sources=[
                'src/cuda_backend/bind.cpp',
                'src/cuda_backend/128_agents_map_step.cu',
                'src/cuda_backend/128_agents_map_stateless_step.cu',
                'src/cuda_backend/energy_map.cu',
                'src/cuda_backend/grid_world_cuda.cu',
                'src/cuda_backend/mapf_gpt_cuda.cu',
                'src/cuda_backend/grid_world_simulator.cpp',
                'src/cuda_backend/mapf_gpt_builder.cpp',
                'src/cuda_backend/print_tensor.cpp',
            ],
            include_dirs=extra_include_dirs,
            library_dirs=extra_library_dirs,
            extra_compile_args={
                'cxx': cxx_args,
                'nvcc': nvcc_args
            },
            extra_link_args=['-lgomp'] 
        )
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(use_ninja=True)
    }
)
