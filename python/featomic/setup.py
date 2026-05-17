import glob
import os
import pathlib
import subprocess
import sys

import packaging
from setuptools import Extension, setup
from setuptools.command.bdist_egg import bdist_egg
from setuptools.command.bdist_wheel import bdist_wheel
from setuptools.command.build_ext import build_ext
from setuptools.command.sdist import sdist


ROOT = pathlib.Path(__file__).parent.resolve()
FEATOMIC_SRC = (ROOT / ".." / ".." / "featomic").resolve()

FEATOMIC_BUILD_TYPE = os.environ.get("FEATOMIC_BUILD_TYPE", "release")
if FEATOMIC_BUILD_TYPE not in ["debug", "release"]:
    raise Exception(
        f"invalid build type passed: '{FEATOMIC_BUILD_TYPE}',"
        "expected 'debug' or 'release'"
    )

FEATOMIC_TORCH_SRC = (ROOT / ".." / "featomic_torch").resolve()


class universal_wheel(bdist_wheel):
    # When building the wheel, the `wheel` package assumes that if we have a
    # binary extension then we are linking to `libpython.so`; and thus the wheel
    # is only usable with a single python version. This is not the case for
    # here, and the wheel will be compatible with any Python >=3. This is
    # tracked in https://github.com/pypa/wheel/issues/185, but until then we
    # manually override the wheel tag.
    def get_tag(self):
        tag = bdist_wheel.get_tag(self)
        # tag[2:] contains the os/arch tags, we want to keep them
        return ("py3", "none") + tag[2:]


class cmake_ext(build_ext):
    """Build the native library using cmake."""

    def run(self):
        """Run cmake build and install the resulting library."""
        import metatensor

        source_dir = FEATOMIC_SRC
        build_dir = ROOT / "build" / "cmake-build"
        install_dir = pathlib.Path(self.build_lib).resolve() / "featomic"

        try:
            os.mkdir(build_dir)
        except OSError:
            pass

        cmake_options = [
            f"-DCMAKE_INSTALL_PREFIX={install_dir}",
            "-DCMAKE_INSTALL_LIBDIR=lib",
            f"-DCMAKE_BUILD_TYPE={FEATOMIC_BUILD_TYPE}",
            f"-DCMAKE_PREFIX_PATH={metatensor.utils.cmake_prefix_path}",
            "-DFEATOMIC_INSTALL_BOTH_STATIC_SHARED=OFF",
            "-DBUILD_SHARED_LIBS=ON",
            "-DEXTRA_RUST_FLAGS=-Cstrip=symbols",
        ]

        if "CARGO" in os.environ:
            cmake_options.append(f"-DCARGO_EXE={os.environ['CARGO']}")

        # Handle cross-compilation by detecting cibuildwheels environnement
        # variables
        if sys.platform.startswith("darwin"):
            # ARCHFLAGS is set by cibuildwheels
            ARCHFLAGS = os.environ.get("ARCHFLAGS")
            if ARCHFLAGS is not None:
                archs = filter(
                    lambda u: bool(u),
                    ARCHFLAGS.strip().split("-arch "),
                )
                archs = list(archs)
                assert len(archs) == 1
                arch = archs[0].strip()

                if arch == "x86_64":
                    cmake_options.append("-DRUST_BUILD_TARGET=x86_64-apple-darwin")
                elif arch == "arm64":
                    cmake_options.append("-DRUST_BUILD_TARGET=aarch64-apple-darwin")
                else:
                    raise ValueError(f"unknown arch: {arch}")

        elif sys.platform.startswith("linux"):
            # we set RUST_BUILD_TARGET in our custom docker image
            RUST_BUILD_TARGET = os.environ.get("RUST_BUILD_TARGET")
            if RUST_BUILD_TARGET is not None:
                cmake_options.append(f"-DRUST_BUILD_TARGET={RUST_BUILD_TARGET}")

        elif sys.platform.startswith("win32"):
            # CARGO_BUILD_TARGET is set by cibuildwheels
            CARGO_BUILD_TARGET = os.environ.get("CARGO_BUILD_TARGET")
            if CARGO_BUILD_TARGET is not None:
                cmake_options.append(f"-DRUST_BUILD_TARGET={CARGO_BUILD_TARGET}")

        else:
            raise ValueError(f"unknown platform: {sys.platform}")

        subprocess.run(
            ["cmake", source_dir, *cmake_options],
            cwd=build_dir,
            check=True,
        )
        subprocess.run(
            ["cmake", "--build", build_dir, "--parallel", "--target", "install"],
            check=True,
        )


class bdist_egg_disabled(bdist_egg):
    """Disabled version of bdist_egg

    Prevents setup.py install performing setuptools' default easy_install,
    which it should never ever do.
    """

    def run(self):
        sys.exit(
            "Aborting implicit building of eggs. "
            + "Use `pip install .` or `python setup.py bdist_wheel && pip "
            + "uninstall featomic -y && pip install dist/featomic-*.whl` "
            + "to install from source."
        )


class sdist_generate_data(sdist):
    """
    Create a sdist with an additional generated files:
        - `git_version_info`
        - `featomic-cxx-*.tar.gz`
    """

    def run(self):
        n_commits, git_hash = git_version_info()
        with open("git_version_info", "w") as fd:
            fd.write(f"{n_commits}\n{git_hash}\n")

        generate_cxx_tar()

        # run original sdist
        super().run()

        os.unlink("git_version_info")
        for path in glob.glob("featomic-cxx-*.tar.gz"):
            os.unlink(path)


def generate_cxx_tar():
    script = ROOT / ".." / ".." / "scripts" / "package-featomic.sh"
    assert script.exists()

    try:
        output = subprocess.run(
            ["bash", "--version"],
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            encoding="utf8",
        )
    except Exception as e:
        raise RuntimeError("could not run `bash`, is it installed?") from e

    output = subprocess.run(
        ["bash", script, os.getcwd()],
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
        encoding="utf8",
    )
    if output.returncode != 0:
        stderr = output.stderr
        stdout = output.stdout
        raise RuntimeError(
            "failed to collect C++ sources for Python sdist\n"
            f"stdout:\n {stdout}\n\nstderr:\n {stderr}"
        )


def get_rust_version():
    # read version from Cargo.toml
    with open(FEATOMIC_SRC / "Cargo.toml") as fd:
        for line in fd:
            if line.startswith("version"):
                _, version = line.split(" = ")
                # remove quotes
                version = version[1:-2]
                # take the first version in the file, this should be the right
                # version
                break

    return version


def git_version_info():
    """
    If git is available and we are building from a checkout, get the number of commits
    since the last tag & full hash of the code. Otherwise, this always returns (0, "").
    """
    TAG_PREFIX = "featomic-v"

    if (ROOT / "git_version_info").exists():
        # we are building from a sdist, without git available, but the git
        # version was recorded in the `git_version_info` file
        with open(ROOT / "git_version_info") as fd:
            n_commits = int(fd.readline().strip())
            git_hash = fd.readline().strip()
    else:
        script = ROOT / ".." / ".." / "scripts" / "git-version-info.py"
        assert script.exists()

        output = subprocess.run(
            [sys.executable, script, TAG_PREFIX],
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            encoding="utf8",
        )

        if output.returncode != 0:
            raise Exception(
                "failed to get git version info.\n"
                f"stdout: {output.stdout}\n"
                f"stderr: {output.stderr}\n"
            )
        elif output.stderr:
            print(output.stderr, file=sys.stderr)
            n_commits = 1
            git_hash = "no-git-hash"
        else:
            lines = output.stdout.splitlines()
            n_commits = int(lines[0].strip())
            git_hash = lines[1].strip()

    return n_commits, git_hash


def create_version_number(version):
    version = packaging.version.parse(version)

    n_commits, git_hash = git_version_info()
    if n_commits != 0:
        # `n_commits` will be non zero only if we have commits since the last tag. This
        # mean we are in a pre-release of the next version. So we increase either the
        # minor version number or the release candidate number (if we are closing up on
        # a release)
        if version.pre is not None:
            assert version.pre[0] == "rc"
            pre = ("rc", version.pre[1] + 1)
            release = version.release
        else:
            major, minor, _ = version.release
            release = (major, minor + 1, 0)
            pre = None

        version = version.__replace__(
            release=release,
            pre=pre,
            dev=n_commits,
            local=git_hash,
        )

    return str(version)


if __name__ == "__main__":
    if not FEATOMIC_SRC.exists():
        # we are building from a sdist, which should include featomic Rust
        # sources as a tarball
        tarballs = glob.glob("featomic-*.tar.gz", root_dir=ROOT)

        if not len(tarballs) == 1:
            raise RuntimeError(
                "expected a single 'featomic-*.tar.gz' file containing "
                "featomic Rust sources. remove all files and re-run "
                "scripts/package-featomic.sh"
            )

        FEATOMIC_SRC = pathlib.Path(tarballs[0]).resolve()
        subprocess.run(
            ["cmake", "-E", "tar", "xf", FEATOMIC_SRC],
            cwd=ROOT,
            check=True,
        )

        FEATOMIC_SRC = pathlib.Path(".".join(str(FEATOMIC_SRC).split(".")[:-2]))

    with open(ROOT / "AUTHORS") as fd:
        authors = fd.read().splitlines()

    extras_require = {}

    # when packaging a sdist for release, we should never use local dependencies
    FEATOMIC_NO_LOCAL_DEPS = os.environ.get("FEATOMIC_NO_LOCAL_DEPS", "0") == "1"
    if not FEATOMIC_NO_LOCAL_DEPS and FEATOMIC_TORCH_SRC.exists():
        # we are building from a git checkout
        extras_require["torch"] = f"featomic-torch @ {FEATOMIC_TORCH_SRC.as_uri()}"
    else:
        # we are building from a sdist/installing from a wheel
        extras_require["torch"] = "featomic-torch"

    setup(
        version=create_version_number(get_rust_version()),
        author=", ".join(authors),
        extras_require=extras_require,
        ext_modules=[
            # only declare the extension, it is built & copied as required by cmake
            # in the build_ext command
            Extension(name="featomic", sources=[]),
        ],
        cmdclass={
            "build_ext": cmake_ext,
            "bdist_egg": bdist_egg if "bdist_egg" in sys.argv else bdist_egg_disabled,
            "bdist_wheel": universal_wheel,
            "sdist": sdist_generate_data,
        },
        package_data={
            "featomic": [
                "featomic/lib/*",
                "featomic/include/*",
            ]
        },
    )
