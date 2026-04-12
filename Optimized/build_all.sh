#!/usr/bin/env bash
# Build all applications under Optimized/ for a target GPU architecture.
#
# Usage: ./build_all.sh [SM_VERSION]
#   SM_VERSION  CUDA SM version without the "sm_" prefix (default: 89)
#
# Examples:
#   ./build_all.sh        # builds for sm_89 (RTX 4000 series)
#   ./build_all.sh 80     # builds for sm_80 (A100)
#   ./build_all.sh 86     # builds for sm_86 (RTX 3000 series)
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
# LULESH/cuda/src  (FLAGS make variable carries -arch)
# ---------------------------------------------------------------------------
run_build "LULESH/cuda" \
    make -C "$SCRIPT_DIR/LULESH/cuda/src" release \
         FLAGS="-arch=sm_${SM} -lineinfo" \
         RFLAGS="-O3 -DNDEBUG"

# ---------------------------------------------------------------------------
# rodinia/cuda — each benchmark uses ../../common/make.config for CUDA_DIR.
# Arch is injected via KERNEL_DIM (used as extra nvcc flags in several
# Makefiles) or NVCC_FLAGS where the variable is referenced.
# ---------------------------------------------------------------------------
RODINIA_ARCH="-arch=sm_${SM}"

# backprop: uses NVCC_FLAGS
run_build "rodinia/cuda/backprop" \
    make -C "$SCRIPT_DIR/rodinia/cuda/backprop" \
         NVCC_FLAGS="-I${CUDA_DIR}/include -O2 -lineinfo ${RODINIA_ARCH}"

# bfs: no arch flag in original Makefile; inject via KERNEL_DIM (appended to CC call)
run_build "rodinia/cuda/bfs" \
    make -C "$SCRIPT_DIR/rodinia/cuda/bfs" release \
         KERNEL_DIM="${RODINIA_ARCH}"

# hotspot: uses KERNEL_DIM
run_build "rodinia/cuda/hotspot" \
    make -C "$SCRIPT_DIR/rodinia/cuda/hotspot" release \
         KERNEL_DIM="${RODINIA_ARCH}"


# srad_v1: arch is hardcoded as a literal in the makefile rule AND the rule
# calls `nvcc` directly (not $(CC)), so we shadow nvcc on PATH with a wrapper.
SRAD1_DIR="$SCRIPT_DIR/rodinia/cuda/srad/srad_v1"
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
run_build "rodinia/cuda/srad_v1" \
    env PATH="$SRAD1_TMPBIN:$PATH" make -C "$SRAD1_DIR"
rm -rf "$SRAD1_TMPBIN"

# srad_v2: top-level Makefile delegates with bare `make` (not $(MAKE)),
# so command-line variables don't propagate automatically. Build directly.
run_build "rodinia/cuda/srad_v2" \
    make -C "$SCRIPT_DIR/rodinia/cuda/srad/srad_v2" release \
         KERNEL_DIM="${RODINIA_ARCH}"

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
