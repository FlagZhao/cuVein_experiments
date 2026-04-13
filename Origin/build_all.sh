#!/usr/bin/env bash
# Build all applications under Origin/ for a target GPU architecture.
#
# Usage: ./build_all.sh [SM_VERSION]
#   SM_VERSION  CUDA SM version without the "sm_" prefix (default: 89)
#
# Examples:
#   ./build_all.sh        # builds for sm_89 (RTX 4000 series)
#   ./build_all.sh 80     # builds for sm_80 (A100)
#   ./build_all.sh 90     # builds for sm_90 (H100)

set -euo pipefail

SM="${1:-89}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Detect CUDA installation
CUDA_DIR="${CUDA_DIR:-/usr/local/cuda}"
if [ ! -d "$CUDA_DIR" ]; then
    for d in /usr/local/cuda-*/; do
        CUDA_DIR="${d%/}"
        break
    done
fi
if [ ! -f "$CUDA_DIR/bin/nvcc" ]; then
    echo "ERROR: nvcc not found under CUDA_DIR=$CUDA_DIR" >&2
    exit 1
fi
export CUDA_DIR

PASS=()
FAIL=()

run_build() {
    local label="$1"; shift
    echo ""
    echo "========== Building: $label (sm_${SM}) =========="
    if "$@"; then
        PASS+=("$label")
    else
        echo "WARNING: $label build failed (exit $?)" >&2
        FAIL+=("$label")
    fi
}

# ---------------------------------------------------------------------------
# XSBench/cuda  (SM_VERSION make variable)
# ---------------------------------------------------------------------------
run_build "XSBench/cuda" \
    make -C "$SCRIPT_DIR/XSBench/cuda" SM_VERSION="$SM"

# ---------------------------------------------------------------------------
# LULESH/cuda/src  (bare nvcc; pass NVCC and FLAGS on command line)
# ---------------------------------------------------------------------------
run_build "LULESH/cuda" \
    make -C "$SCRIPT_DIR/LULESH/cuda/src" release \
         NVCC="${CUDA_DIR}/bin/nvcc -ccbin /usr/bin/gcc" \
         FLAGS="-arch=sm_${SM} -lineinfo" \
         RFLAGS="-O3 -DNDEBUG"

# ---------------------------------------------------------------------------
# rodinia_3.1/cuda — uses common/make.config for CUDA_DIR
# ---------------------------------------------------------------------------
RODINIA_ARCH="-arch=sm_${SM}"

# backprop: uses NVCC_FLAGS
run_build "rodinia/backprop" \
    make -C "$SCRIPT_DIR/rodinia_3.1/cuda/backprop" \
         CUDA_DIR="${CUDA_DIR}" \
         NVCC_FLAGS="-I${CUDA_DIR}/include -O2 -lineinfo ${RODINIA_ARCH}"

# bfs: inject arch via KERNEL_DIM
run_build "rodinia/bfs" \
    make -C "$SCRIPT_DIR/rodinia_3.1/cuda/bfs" release \
         CUDA_DIR="${CUDA_DIR}" \
         KERNEL_DIM="${RODINIA_ARCH}"

# hotspot: uses KERNEL_DIM
run_build "rodinia/hotspot" \
    make -C "$SCRIPT_DIR/rodinia_3.1/cuda/hotspot" release \
         CUDA_DIR="${CUDA_DIR}" \
         KERNEL_DIM="${RODINIA_ARCH}"

# srad_v1: bare `nvcc` call in rule — shadow nvcc on PATH with a wrapper
SRAD1_DIR="$SCRIPT_DIR/rodinia_3.1/cuda/srad/srad_v1"
SRAD1_TMPBIN="$(mktemp -d)"
cat > "$SRAD1_TMPBIN/nvcc" <<WRAP
#!/usr/bin/env bash
args=()
skip_next=false
for arg in "\$@"; do
    if \$skip_next; then
        skip_next=false
        continue
    fi
    case "\$arg" in
        -arch) args+=("-arch" "sm_${SM}"); skip_next=true ;;
        -arch=sm_*) args+=("-arch=sm_${SM}") ;;
        sm_[0-9]*) ;;
        *) args+=("\$arg") ;;
    esac
done
exec "${CUDA_DIR}/bin/nvcc" "\${args[@]}"
WRAP
chmod +x "$SRAD1_TMPBIN/nvcc"
run_build "rodinia/srad_v1" \
    env PATH="$SRAD1_TMPBIN:$PATH" make -C "$SRAD1_DIR" \
        CUDA_DIR="${CUDA_DIR}"
rm -rf "$SRAD1_TMPBIN"

# ---------------------------------------------------------------------------
# pasta  (CMake + CUDA, CUDA_ARCH_BIN variable)
# ---------------------------------------------------------------------------
PASTA_SRC="$SCRIPT_DIR/pasta"
PASTA_BUILD="$PASTA_SRC/build"
run_build "pasta" bash -c "
    cmake -S '$PASTA_SRC' -B '$PASTA_BUILD' \
        -DUSE_ICC=OFF \
        -DCMAKE_C_COMPILER=gcc \
        -DCMAKE_CXX_COMPILER=g++ \
        -DUSE_OPENMP=ON \
        -DUSE_CUDA=ON \
        -DCUDA_ARCH_BIN='$SM' \
        -DCUDA_rt_LIBRARY=/usr/lib/x86_64-linux-gnu/librt.so.1 \
        && cmake --build '$PASTA_BUILD' --parallel \$(nproc)
"

# ---------------------------------------------------------------------------
# llama.cpp  (CMake + CUDA, GGML_CUDA=ON)
# ---------------------------------------------------------------------------
LLAMA_SRC="$SCRIPT_DIR/llama.cpp"
LLAMA_BUILD="$LLAMA_SRC/build_sm${SM}"
run_build "llama.cpp" bash -c "
    cmake -S '$LLAMA_SRC' -B '$LLAMA_BUILD' \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES='$SM' \
        -DLLAMA_CURL=OFF \
        && cmake --build '$LLAMA_BUILD' --parallel \$(nproc)
"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "=========================================="
echo " Build summary  (target: sm_${SM})"
echo "=========================================="
if [ ${#PASS[@]} -gt 0 ]; then
    echo "PASSED (${#PASS[@]}):"
    for t in "${PASS[@]}"; do echo "  [OK]  $t"; done
fi
if [ ${#FAIL[@]} -gt 0 ]; then
    echo "FAILED (${#FAIL[@]}):"
    for t in "${FAIL[@]}"; do echo "  [!!]  $t"; done
    exit 1
fi
echo "All builds succeeded."
