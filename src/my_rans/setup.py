# from setuptools import setup, find_packages
# from torch.utils.cpp_extension import CppExtension, BuildExtension

# # --- Debug Flags ---
# # Default flags for GCC/Clang
# compile_args = ['-g', '-O0'] # Add debug symbols, disable optimization
# link_args = ['-g']

# # --- Extension Configuration ---
# ext_modules = [
#     CppExtension(
#         # Rename the extension module to avoid conflict with package name
#         name='my_rans',
#         sources=['rans_ops.cpp'],
#         extra_compile_args=compile_args,
#         extra_link_args=link_args,
#         py_limited_api=True
#     )
# ]

# setup(
#     name='my_rans',
#     version='0.1',
#     packages=find_packages(), # Finds the 'my_rans' Python package directory
#     ext_modules=ext_modules,
#     cmdclass={'build_ext': BuildExtension},
#     # Keep wheel options consistent with py_limited_api
#     options={'bdist_wheel': {'py_limited_api': 'cp39'}} # Or your target base Python e.g. 'cp310'
# )

from setuptools import setup, find_packages
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='my_rans',
    version='0.1',
    packages=find_packages(),
    ext_modules=[
        CppExtension(
            name='my_rans',
            sources=['rans_ops.cpp'],
            extra_compile_args={'cxx': ['-DPy_LIMITED_API=0x03090000']},
            py_limited_api=True
        )
    ],
    cmdclass={'build_ext': BuildExtension},
    options={'bdist_wheel': {'py_limited_api': 'cp39'}}
)