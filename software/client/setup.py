from setuptools import setup, find_packages
# Find packages gives me an import error but its runs fine

setup(
    name='client',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'zeroconf',
        'flask',
    ],
)