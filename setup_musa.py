from __future__ import annotations

# setup_musa.py - MUSA (Moore Threads) variant of pycuda setup.
# Builds pymusa extension against the MUSA Driver API instead of CUDA.
#
# Usage:
#   python setup_musa.py build_ext --inplace

import os
import sys
from os.path import dirname, join, normpath


def search_on_path(filenames):
    """Find file on system path."""
    from os import environ, pathsep
    from os.path import abspath, exists

    search_path = environ["PATH"]
    paths = search_path.split(pathsep)
    for path in paths:
        for filename in filenames:
            if exists(join(path, filename)):
                return abspath(join(path, filename))


def get_config_schema():
    from aksetup_helper import (
        BoostLibraries,
        ConfigSchema,
        IncludeDir,
        Libraries,
        LibraryDir,
        Option,
        StringListOption,
        Switch,
        make_boost_base_options,
    )

    mcc_path = search_on_path(["mcc", "mcc.exe"])
    if mcc_path is None:
        print("WARNING: mcc not in path. May need to set MUSA_INC_DIR for installation to succeed.")
        # Default MUSA SDK installation path
        musa_root_default = "/usr/local/musa"
    else:
        musa_root_default = normpath(join(dirname(mcc_path), ".."))

    cxxflags_default = []
    ldflags_default = []

    default_lib_dirs = [
        "${MUSA_ROOT}/lib",
        "${MUSA_ROOT}/lib64",
        "${MUSA_ROOT}/lib/stubs",
    ]

    return ConfigSchema(
        [*make_boost_base_options(),
            Switch("USE_SHIPPED_BOOST", True, "Use included Boost library"),
            BoostLibraries("python"),
            BoostLibraries("thread"),
            Switch("MUSA_TRACE", False, "Enable MUSA API tracing"),
            Option("MUSA_ROOT", default=musa_root_default,
                   help="Path to the MUSA toolkit"),
            Option("MUSA_PRETEND_VERSION",
                   help="Assumed MUSA version, in the form 3010 for 3.1."),
            IncludeDir("MUSA", None),
            Switch("MUSA_ENABLE_GL", False, "Enable MUSA GL interoperability"),
            Switch("MUSA_ENABLE_MURAND", True, "Enable MURAND library"),
            LibraryDir("MUSADRV", default_lib_dirs),
            Libraries("MUSADRV", ["musa"]),
            LibraryDir("MURAND", default_lib_dirs),
            Libraries("MURAND", ["murand"]),
            StringListOption("CXXFLAGS", cxxflags_default,
                             help="Any extra C++ compiler options to include"),
            StringListOption("LDFLAGS", ldflags_default,
                             help="Any extra linker options to include")]
    )


def main():
    from aksetup_helper import (
        ExtensionUsingNumpy,
        NumpyBuildExtCommand,
        check_git_submodules,
        get_config,
        hack_distutils,
        set_up_shipped_boost_if_requested,
        setup,
    )

    check_git_submodules()

    hack_distutils()
    conf = get_config(get_config_schema())

    EXTRA_SOURCES, EXTRA_DEFINES = set_up_shipped_boost_if_requested("pymusa", conf)

    EXTRA_DEFINES["PYGPU_PACKAGE"] = "pymusa"
    EXTRA_DEFINES["PYGPU_PYCUDA"] = "1"
    EXTRA_DEFINES["PYGPU_PYMUSA"] = "1"

    LIBRARY_DIRS = conf["BOOST_LIB_DIR"] + conf["MUSADRV_LIB_DIR"]
    LIBRARIES = (
        conf["BOOST_PYTHON_LIBNAME"]
        + conf["BOOST_THREAD_LIBNAME"]
        + conf["MUSADRV_LIBNAME"]
    )

    if not conf["MUSA_INC_DIR"] and conf["MUSA_ROOT"]:
        conf["MUSA_INC_DIR"] = [join(conf["MUSA_ROOT"], "include")]

    if conf["MUSA_TRACE"]:
        EXTRA_DEFINES["MUSAPP_TRACE_MUSA"] = 1

    if conf["MUSA_PRETEND_VERSION"]:
        EXTRA_DEFINES["MUSAPP_PRETEND_MUSA_VERSION"] = conf["MUSA_PRETEND_VERSION"]

    # Point to MUSA-ported C++ source files
    INCLUDE_DIRS = ["src/musa_cpp"] + conf["BOOST_INC_DIR"]
    if conf["MUSA_INC_DIR"]:
        INCLUDE_DIRS += conf["MUSA_INC_DIR"]

    conf["USE_MUSA"] = True

    if conf["MUSA_ENABLE_GL"]:
        EXTRA_SOURCES.append("src/musa_wrapper/wrap_cudagl.cpp")
        EXTRA_DEFINES["HAVE_GL"] = 1

    if conf["MUSA_ENABLE_MURAND"]:
        EXTRA_DEFINES["HAVE_CURAND"] = 1  # Keep for compatibility with wrap_curand.cpp
        EXTRA_SOURCES.extend(["src/musa_wrapper/wrap_curand.cpp"])
        LIBRARIES.extend(conf["MURAND_LIBNAME"])
        LIBRARY_DIRS.extend(conf["MURAND_LIB_DIR"])

    ver_dic = {}
    exec(
        compile(open("pycuda/__init__.py").read(), "pycuda/__init__.py", "exec"),
        ver_dic,
    )

    setup(
        name="pycuda",  # Keep package name for compatibility
        version=ver_dic["VERSION_TEXT"],
        description="Python wrapper for Moore Threads MUSA (CUDA-compatible API)",
        long_description=open("README.rst").read(),
        author="Andreas Kloeckner",
        author_email="inform@tiker.net",
        license="MIT",
        url="http://mathema.tician.de/software/pycuda",
        classifiers=[
            "Environment :: Console",
            "Development Status :: 5 - Production/Stable",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: MIT License",
            "Natural Language :: English",
            "Programming Language :: C++",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3",
            "Topic :: Scientific/Engineering",
        ],
        packages=["pycuda", "pycuda.gl", "pycuda.sparse", "pycuda.compyte"],
        python_requires="~=3.8",
        install_requires=[
            "pytools>=2011.2",
            "platformdirs>=2.2.0",
            "mako",
        ],
        test_requires=[
            "pytest>=2",
        ],
        ext_package="pycuda",
        ext_modules=[
            ExtensionUsingNumpy(
                "_driver",
                ["src/musa_cpp/cuda.cpp",
                    "src/musa_cpp/bitlog.cpp",
                    "src/musa_wrapper/wrap_cudadrv.cpp",
                    "src/musa_wrapper/mempool.cpp",
                    *EXTRA_SOURCES],
                include_dirs=INCLUDE_DIRS,
                library_dirs=LIBRARY_DIRS,
                libraries=LIBRARIES,
                define_macros=list(EXTRA_DEFINES.items()),
                extra_compile_args=conf["CXXFLAGS"],
                extra_link_args=conf["LDFLAGS"],
            ),
            ExtensionUsingNumpy(
                "_pvt_struct",
                ["src/musa_wrapper/_pvt_struct_v3.cpp"],
                extra_compile_args=conf["CXXFLAGS"],
                extra_link_args=conf["LDFLAGS"],
            ),
        ],
        cmdclass={"build_ext": NumpyBuildExtCommand},
        include_package_data=True,
        package_data={
            "pycuda": [
                "cuda/*.hpp",
            ]
        },
        zip_safe=False,
    )


if __name__ == "__main__":
    main()
