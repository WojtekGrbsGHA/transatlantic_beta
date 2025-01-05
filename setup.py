"""
<transatlantic> Package
"""


import setuptools
from setuptools.command.sdist import sdist
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy as np
import os.path
import glob
import os
import sys
import re

PATH_ALG1 = "./transatlantic_beta/algorithms/src/algorithm1/source/"
PATH_ALG2 = "./transatlantic_beta/algorithms/src/algorithm2/source/"
PATH_ALG3 = "./transatlantic_beta/algorithms/src/algorithm3/source/"

DEST_PATH_ALG1 = "transatlantic_beta.algorithms.src.algorithm1.binaries."
DEST_PATH_ALG2 = "transatlantic_beta.algorithms.src.algorithm2.binaries."
DEST_PATH_ALG3 = "transatlantic_beta.algorithms.src.algorithm3.binaries."

ALG1_MODULES = (
    "mstl",
)
ALG2_MODULES = (
    "mknn",
    "mknn_bfs",
    "mknn_connect",
    "mknn_lp",
)
ALG3_MODULES = (
    "tdbscan",
    "postprocess",
)

alg1_module_list = [PATH_ALG1 + mod + ".pyx" for mod in ALG1_MODULES]
alg2_module_list = [PATH_ALG2 + mod + ".pyx" for mod in ALG2_MODULES]
alg3_module_list = [PATH_ALG3 + mod + ".pyx" for mod in ALG3_MODULES]


alg1_extensions_list = [setuptools.Extension(DEST_PATH_ALG1 + module, [pyx_module]) for module,pyx_module
                         in zip(ALG1_MODULES, alg1_module_list)]

alg2_extensions_list = [setuptools.Extension(DEST_PATH_ALG2 + module, [pyx_module]) for module,pyx_module
                         in zip(ALG2_MODULES, alg2_module_list)]

alg3_extensions_list = [setuptools.Extension(DEST_PATH_ALG3 + module, [pyx_module]) for module,pyx_module
                         in zip(ALG3_MODULES, alg3_module_list)]


extensions_list = [] + alg1_extensions_list + alg2_extensions_list + alg3_extensions_list

# cython_modules = {
#     "genieclust.internal": [
#         os.path.join("genieclust", "internal.pyx")
#     ],
#     "genieclust.compare_partitions": [
#         os.path.join("genieclust", "compare_partitions.pyx")
#     ],
#     "genieclust.cluster_validity": [
#         os.path.join("genieclust", "cluster_validity.pyx")
#     ],
#     "genieclust.inequality": [
#         os.path.join("genieclust", "inequality.pyx")
#     ],
#     "genieclust.tools": [
#         os.path.join("genieclust", "tools.pyx")
#     ]
# }


class genieclust_sdist(sdist):
    def run(self):
        cythonize(extensions_list, 
                  language_level="3",)
        sdist.run(self)


class genieclust_build_ext(build_ext):
    def build_extensions(self):

        # This code is adapted from
        # scikit-learn/sklearn/_build_utils/openmp_helpers.py
        # (version last updated on 13 Nov 2019; 9876f74)
        # See https://github.com/scikit-learn/scikit-learn and https://scikit-learn.org/.

        if hasattr(self.compiler, 'compiler'):
            compiler = self.compiler.compiler[0]
        else:
            compiler = self.compiler.__class__.__name__

        # if sys.platform == "win32" and ('icc' in compiler or 'icl' in compiler):
        #     for e in self.extensions:
        #         e.extra_compile_args += ['/Qopenmp', '/Qstd=c++11']
        #         e.extra_link_args += ['/Qopenmp']
        # elif sys.platform == "win32":
        #     for e in self.extensions:
        #         e.extra_compile_args += ['/openmp']
        #         e.extra_link_args += ['/openmp']
        # if sys.platform == "darwin" and ('icc' in compiler or 'icl' in compiler):
        #     for e in self.extensions:
        #         e.extra_compile_args += ['-openmp', '-std=c++11']
        #         e.extra_link_args += ['-openmp']
        # elif sys.platform == "darwin":  # and 'openmp' in os.getenv('CPPFLAGS', ''):
        #     for e in self.extensions:
        #         e.extra_compile_args += ['-std=c++11']
        #     pass
        # elif sys.platform == "linux":
        #     # Default flag for GCC and clang:
        #     for e in self.extensions:
        #         e.extra_compile_args += ['-fopenmp', '-std=c++11']
        #         e.extra_link_args += ['-fopenmp']
        # else:
        #     pass

        build_ext.build_extensions(self)


ext_kwargs = dict(
    include_dirs=[np.get_include(), "src/", "../src/"],
    language="c++",
    depends=glob.glob(os.path.join("src", "c_*.h")) +
            glob.glob(os.path.join("genieclust", "*.pxd")),
)


with open("README.md", "r") as fh:
    long_description = fh.read()

# with open("genieclust/__init__.py", "r") as fh:
#     __version__ = re.search("(?m)^\\s*__version__\\s*=\\s*[\"']([0-9.]+)[\"']", fh.read())
#     if __version__ is None:
#         raise ValueError("the package version could not be read")
#     __version__ = __version__.group(1)

__version__ = "0.0.dev1"
setuptools.setup(
    name="transatlantic_beta",
    license="GNU Affero General Public License v3",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "numpy",
        "scipy",
        "Cython", 
        "matplotlib",
        "scikit-learn",
        "networkx",
      ],
    download_url="https://google.com/",
    url="https://google.com/",

    project_urls={
        "Documentation":      "https://google.com/",
    },
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3 :: Only",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Development Status :: 5 - Production/Stable",
        "Topic :: Scientific/Engineering",
    ],
    cmdclass={
        "sdist": genieclust_sdist,
        "build_ext": genieclust_build_ext
    },
    packages=setuptools.find_packages(where="./transatlantic_beta/"),  # Automatically find packages in the current directory
    package_dir={"": "transatlantic_beta"},  # Map the root directory of packages to "package/"
    ext_modules=extensions_list,
    include_dirs=[np.get_include()]
)
