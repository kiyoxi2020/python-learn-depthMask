from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

INSTALL_REQUIREMENTS = []

ext_modules=[
    CUDAExtension('reconstructPrevDepth.cuda',[
        'reconstructPrevDepth/cuda/reconstructPrevDepth.cpp',
        'reconstructPrevDepth/cuda/reconstructPrevDepth_kernel.cu',
    ]),
]

setup(
    description="cuda implement of fsr reconstructPrevDepth",
    name="reconstructPrevDepth_cuda",
    test_suite='setup.test_all',
    packages=['reconstructPrevDepth.cuda'],
    install_requires=INSTALL_REQUIREMENTS,
    ext_modules=ext_modules,
    cmdclass = {'build_ext': BuildExtension}
)