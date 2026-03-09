#!/usr/bin/env bash

set -eux

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)
cd "$ROOT_DIR"

TMP_DIR="$1"
rm -rf "$TMP_DIR"/dist

# check building sdist from a checkout, and wheel from the sdist
python -m build python/featomic --outdir "$TMP_DIR"/dist

# get the version of featomic we just built
FEATOMIC_VERSION=$(basename "$(find "$TMP_DIR"/dist -name "featomic-*.tar.gz")" | cut -d - -f 2)
FEATOMIC_VERSION=${FEATOMIC_VERSION%.tar.gz}

# for featomic-torch, we need a pre-built version of featomic, so
# we use the one we just generated and make it available to pip via PIP_FIND_LINKS

# Get absolute path to the dist directory
# We use python here to ensure we get a valid absolute path regardless of platform/shell
DIST_DIR=$(python -c "import os, sys; print(os.path.abspath(sys.argv[1]))" "$TMP_DIR/dist")

# add the dist directory to the set of find links
# PIP_FIND_LINKS allows pip to find packages in a flat directory
export PIP_FIND_LINKS="file://$DIST_DIR ${PIP_FIND_LINKS:-}"

# force featomic-torch to use a specific featomic version when building
export FEATOMIC_TORCH_BUILD_WITH_FEATOMIC_VERSION="$FEATOMIC_VERSION"

# build featomic-torch, using featomic from `PIP_FIND_LINKS`
# for the sdist => wheel build.
python -m build python/featomic_torch --outdir "$TMP_DIR/dist"
