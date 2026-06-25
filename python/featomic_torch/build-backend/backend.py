# this is a custom Python build backend wrapping setuptool's to add a build-time
# dependencies to featomic, using the local version if it exists, and otherwise
# falling back to the one on PyPI.
import os
import pathlib

from setuptools import build_meta


ROOT = pathlib.Path(__file__).parent.parent.resolve()
FEATOMIC_SRC = (ROOT / ".." / "featomic").resolve()
FORCED_FEATOMIC_VERSION = os.environ.get("FEATOMIC_TORCH_BUILD_WITH_FEATOMIC_VERSION")

FEATOMIC_NO_LOCAL_DEPS = os.environ.get("FEATOMIC_NO_LOCAL_DEPS", "0") == "1"

if FORCED_FEATOMIC_VERSION is not None:
    # force a specific version for metatensor-core, this is used when checking the build
    # from a sdist on a non-released version
    FEATOMIC_DEP = f"featomic =={FORCED_FEATOMIC_VERSION}"

elif not FEATOMIC_NO_LOCAL_DEPS and FEATOMIC_SRC.exists():
    # we are building from a git checkout
    FEATOMIC_DEP = f"featomic @ {FEATOMIC_SRC.as_uri()}"
else:
    # we are building from a sdist
    FEATOMIC_DEP = "featomic >=0.6.6,<0.7"

FORCED_TORCH_VERSION = os.environ.get("FEATOMIC_TORCH_BUILD_WITH_TORCH_VERSION")
if FORCED_TORCH_VERSION is not None:
    TORCH_DEP = f"torch =={FORCED_TORCH_VERSION}"
else:
    TORCH_DEP = "torch >=2.1"


get_requires_for_build_sdist = build_meta.get_requires_for_build_sdist
prepare_metadata_for_build_wheel = build_meta.prepare_metadata_for_build_wheel
build_wheel = build_meta.build_wheel
build_sdist = build_meta.build_sdist


def get_requires_for_build_wheel(config_settings=None):
    defaults = build_meta.get_requires_for_build_wheel(config_settings)
    return defaults + [
        "cmake",
        TORCH_DEP,
        "metatensor-torch >=0.10.0,<0.11",
        "metatomic-torch >=0.1.15,<0.2",
        FEATOMIC_DEP,
    ]
