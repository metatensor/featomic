# this is a custom Python build backend wrapping setuptool's to add a build-time
# dependencies to featomic, using the local version if it exists, and otherwise
# falling back to the one on PyPI.
import os
import uuid

from setuptools import build_meta


ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
FEATOMIC_SRC = os.path.realpath(os.path.join(ROOT, "..", "featomic"))
FORCED_FEATOMIC_VERSION = os.environ.get("FEATOMIC_TORCH_BUILD_WITH_FEATOMIC_VERSION")


if FORCED_FEATOMIC_VERSION is not None:
    # force a specific version for metatensor-core, this is used when checking the build
    # from a sdist on a non-released version
    FEATOMIC_DEP = f"featomic =={FORCED_FEATOMIC_VERSION}"

elif os.path.exists(FEATOMIC_SRC):
    # we are building from a git checkout

    # add a random uuid to the file url to prevent pip from using a cached
    # wheel for metatensor-core, and force it to re-build from scratch
    uuid = uuid.uuid4()
    FEATOMIC_DEP = f"featomic @ file://{FEATOMIC_SRC}?{uuid}"
else:
    # we are building from a sdist
    FEATOMIC_DEP = "featomic >=0.1.0.dev0,<0.2.0"


get_requires_for_build_sdist = build_meta.get_requires_for_build_sdist
prepare_metadata_for_build_wheel = build_meta.prepare_metadata_for_build_wheel
build_wheel = build_meta.build_wheel
build_sdist = build_meta.build_sdist


def get_requires_for_build_wheel(config_settings=None):
    defaults = build_meta.get_requires_for_build_wheel(config_settings)
    return defaults + [
        "torch >= 1.12",
        "metatensor-torch >=0.6.0,<0.7.0",
        FEATOMIC_DEP,
    ]
