from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name="cpp_exts",
    ext_modules=[
        CppExtension(
            name="cpp_exts._C",
            sources=["src/cpp_exts/rans.cpp"],
            extra_compile_args={"cxx": ["-DPy_LIMITED_API=0x03090000"]},
            py_limited_api=True
        )
    ],
    cmdclass={'build_ext': BuildExtension},
    options={"bdist_wheel": {"py_limited_api": "cp39"}}
)