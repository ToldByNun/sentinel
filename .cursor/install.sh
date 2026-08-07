#!/usr/bin/env bash
# Idempotent Cloud Agent bootstrap for the Sentinel C++/CUDA LM framework.
# The CUDA 13 toolkit, GNU toolchain, Ninja and Python dev headers are provided
# by the base environment snapshot; this script (re)builds the repo from source.
set -euo pipefail

# CUDA toolchain (from the base snapshot) — make it explicit so the build does
# not depend on shell profile sourcing.
export CUDA_HOME=/usr/local/cuda-13.0
export CUDA_PATH="${CUDA_HOME}"
export PATH="${CUDA_HOME}/bin:${PATH}"

echo "==> Toolchain"
nvcc --version | tail -2
cmake --version | head -1

# Single GPU arch (sm_75) keeps builds fast; the fat binary is a release-time
# concern. Pin the GNU toolchain so nvcc's host compiler and the C++ OpenMP
# flags stay consistent (the base image's default 'c++' is clang).
CMAKE_TOOLCHAIN_ARGS=(
  -DCMAKE_C_COMPILER=gcc
  -DCMAKE_CXX_COMPILER=g++
  -DCMAKE_CUDA_HOST_COMPILER=g++
  -DSENTINEL_CUDA_ARCHITECTURES=75
)

echo "==> Configuring + building C++/CUDA library, demo and examples"
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release "${CMAKE_TOOLCHAIN_ARGS[@]}"
cmake --build build --parallel

echo "==> Python bindings (editable install into .venv)"
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
. .venv/bin/activate
python -m pip install --upgrade pip -q
python -m pip install -q "scikit-build-core>=0.10" "nanobind>=2.0" cmake ninja
python -m pip install -e . --no-build-isolation \
  -C cmake.define.SENTINEL_CUDA_ARCHITECTURES=75 \
  -C cmake.define.CMAKE_C_COMPILER=gcc \
  -C cmake.define.CMAKE_CXX_COMPILER=g++ \
  -C cmake.define.CMAKE_CUDA_HOST_COMPILER=g++

echo "==> Sanity import"
python -c "import sentinel as S; print('sentinel import OK; cuda_available =', S.cuda_available())"

echo "==> Install complete"
