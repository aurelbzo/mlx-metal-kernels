from setuptools import setup
from mlx import extension


setup(
    ext_modules=[extension.CMakeExtension("mlx_cpp_projection._ext")],
    cmdclass={"build_ext": extension.CMakeBuild},
    packages=["mlx_cpp_projection"],
    zip_safe=False,
)
