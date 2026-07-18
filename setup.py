import glob
import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Absolute path so the -I flag resolves correctly under torch's ninja build
# (which runs from build/temp, not the project root).
CSRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")

# CUTLASS header-only library path (third_party submodule)
CUTLASS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party", "cutlass")


def get_cuda_arch_flags():
    """Auto-detect the current GPU's compute capability for nvcc (-gencode)."""
    arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if arch:
        return None  # respect an explicit override
    try:
        import torch

        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability(0)
            return [f"-gencode=arch=compute_{cap[0]}{cap[1]},code=sm_{cap[0]}{cap[1]}"]
    except Exception:
        pass
    return None


setup(
    name="cuda-gemm",
    version="0.1.0",
    description="A minimalist PyTorch extension for custom CUDA GEMM kernels.",
    ext_modules=[
        CUDAExtension(
            name="cuda_gemm",
            # Auto-collect all C++/CUDA sources under csrc/ so new kernel
            # variants can be dropped in without editing this file.
            # recursive=True lets `**` match the top level (csrc/ops.cpp) too.
            sources=sorted(
                glob.glob("csrc/**/*.cpp", recursive=True)
                + glob.glob("csrc/**/*.cu", recursive=True)
            ),
            include_dirs=[
                CSRC_DIR,
                os.path.join(CUTLASS_DIR, "include"),  # CUTLASS header-only
            ],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", *(get_cuda_arch_flags() or [])],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.9",
    zip_safe=False,
)
