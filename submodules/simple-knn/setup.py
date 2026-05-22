#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

cxx_compiler_flags = ["-std=c++17"]
nvcc_flags = [
    "-std=c++17",
    "--forward-unknown-to-host-compiler",
    "-Xcompiler", "-fno-builtin",
    "-Xcompiler", "-U__STRICT_ANSI__",
    "-Xcompiler", "-D_GNU_SOURCE",
    "-Xcompiler", "-D_DEFAULT_SOURCE",
]

if os.name == 'nt':
    cxx_compiler_flags.append("/wd4624")

setup(
    name="simple_knn",
    ext_modules=[
        CUDAExtension(
            name="simple_knn._C",
            sources=[
            "spatial.cu", 
            "simple_knn.cu",
            "ext.cpp"],
            extra_compile_args={"nvcc": nvcc_flags, "cxx": cxx_compiler_flags})
        ],
    cmdclass={
        'build_ext': BuildExtension.with_options(use_ninja=False)
    }
)
