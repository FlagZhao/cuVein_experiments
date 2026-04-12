#!/bin/bash
# run_all.sh
# Profiles every Optimized/ application with ncu (Nsight Compute).
# Produces one .ncu-rep file per run and saves them all.
#
# Usage: bash run_all.sh [RUNS] [NCU_EXTRA_FLAGS]
#   RUNS            number of ncu runs per app (default: 10)
#   NCU_EXTRA_FLAGS additional flags forwarded to every ncu invocation
#
# Examples:
#   bash run_all.sh
#   bash run_all.sh 5
#   bash run_all.sh 10 "--set full"

set -euo pipefail

RUNS=${1:-10}
NCU_EXTRA=${2:-}

export PATH=/usr/local/cuda/bin:${PATH}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
RESULTS_DIR="${SCRIPT_DIR}/ncu_results_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}"
LOG_FILE="${RESULTS_DIR}/run.log"

# ─── application definitions ────────────────────────────────────────────────
# add_app NAME WORKDIR BINARY "ARGS" ["ENV_VARS"]

NAMES=(); WORKDIRS=(); BINARIES=(); ARGS_LIST=(); ENVS=()

add_app() {
    NAMES+=("$1"); WORKDIRS+=("$2"); BINARIES+=("$3")
    ARGS_LIST+=("$4"); ENVS+=("${5:-}")
}

OPT="${SCRIPT_DIR}"
DATA="${OPT}/rodinia/data"

add_app "rodinia/backprop"          "${OPT}/rodinia/cuda/backprop"          "./backprop"             "65536"
add_app "rodinia/bfs"               "${OPT}/rodinia/cuda/bfs"               "./bfs"                  "${DATA}/bfs/graph1MW_6.txt"
add_app "rodinia/hotspot"           "${OPT}/rodinia/cuda/hotspot"           "./hotspot"              "512 2 2 ${DATA}/hotspot/temp_512 ${DATA}/hotspot/power_512 output.out"
add_app "rodinia/srad_v1"           "${OPT}/rodinia/cuda/srad/srad_v1"      "./srad"                 "100 0.5 502 458"
add_app "rodinia/srad_v2"           "${OPT}/rodinia/cuda/srad/srad_v2"      "./srad"                 "2048 2048 0 127 0 127 0.5 2"
add_app "LULESH"                    "${OPT}/LULESH/cuda/src"                "./lulesh"               "-s 100"
add_app "XSBench"                   "${OPT}/XSBench/cuda"                   "./XSBench"              "-s large -m event -l 1700000"
add_app "llama.cpp/soft_max"        "${OPT}/llama.cpp/build/bin"            "./test-backend-ops"     "test -o SOFT_MAX"      "CUDA_VISIBLE_DEVICES=0"
add_app "llama.cpp/rms_norm"        "${OPT}/llama.cpp/build/bin"            "./test-backend-ops"     "test -o RMS_NORM"      "CUDA_VISIBLE_DEVICES=0"
add_app "llama.cpp/rms_norm_back"   "${OPT}/llama.cpp/build/bin"            "./test-backend-ops"     "test -o RMS_NORM_BACK" "CUDA_VISIBLE_DEVICES=0"
add_app "llama.cpp/norm"            "${OPT}/llama.cpp/build/bin"            "./test-backend-ops"     "test -o NORM"          "CUDA_VISIBLE_DEVICES=0"

# ─── main ───────────────────────────────────────────────────────────────────

echo "============================================================" | tee    "${LOG_FILE}"
echo "  ncu profiling run  (${RUNS} runs per app)"                 | tee -a "${LOG_FILE}"
echo "  Started:     $(date)"                                       | tee -a "${LOG_FILE}"
echo "  Results dir: ${RESULTS_DIR}"                                | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"

for idx in "${!NAMES[@]}"; do
    name="${NAMES[$idx]}"
    workdir="${WORKDIRS[$idx]}"
    binary="${BINARIES[$idx]}"
    args="${ARGS_LIST[$idx]}"
    env_vars="${ENVS[$idx]}"

    echo "" | tee -a "${LOG_FILE}"
    echo "[ ${name} ]" | tee -a "${LOG_FILE}"

    if [ ! -d "${workdir}" ]; then
        echo "  SKIP — workdir not found: ${workdir}" | tee -a "${LOG_FILE}"
        continue
    fi
    if [ ! -f "${workdir}/${binary#./}" ]; then
        echo "  SKIP — binary not found: ${workdir}/${binary#./}" | tee -a "${LOG_FILE}"
        continue
    fi

    app_dir="${RESULTS_DIR}/${name//\//_}"
    mkdir -p "${app_dir}"

    read -ra arg_arr <<< "${args}"
    read -ra env_arr <<< "${env_vars}"
    env_prefix=(${env_arr[@]+"env" "${env_arr[@]}"})

    pushd "${workdir}" >/dev/null

    for ((i=1; i<=RUNS; i++)); do
        report="${app_dir}/run_$(printf '%02d' ${i})"
        printf "  run %2d/%d  ->  %s.ncu-rep\n" "${i}" "${RUNS}" "${report}" | tee -a "${LOG_FILE}"

        # shellcheck disable=SC2086
        "${env_prefix[@]+"${env_prefix[@]}"}" \
            ncu ${NCU_EXTRA} \
                --set full \
                -o "${report}" \
                --force-overwrite \
                "${binary}" "${arg_arr[@]+"${arg_arr[@]}"}" \
            >> "${LOG_FILE}" 2>&1
    done

    echo "  done — reports in ${app_dir}/" | tee -a "${LOG_FILE}"
    popd >/dev/null
done

echo "" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"
echo "  Done.  Results: ${RESULTS_DIR}"                            | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"
