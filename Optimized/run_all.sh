#!/bin/bash
# run_all.sh
# Measures avg end-to-end runtime of every Optimized/ application in two modes:
#   1. Original (no profiling)
#   2. PC dependency analysis (cuVein / accelprof)
#
# Usage: bash run_all.sh [RUNS]
#   RUNS: number of timed runs per mode (default: 10)

set -euo pipefail

RUNS=${1:-10}

ACCEL_PROF_HOME=/home/yzhao62/opt/Accelprof/cuVein/AccelProf
export ACCEL_PROF_HOME
export PATH=${ACCEL_PROF_HOME}/bin:/usr/local/cuda/bin:${PATH}
export LD_LIBRARY_PATH=/home/yzhao62/anaconda3/envs/cuVein/lib:/usr/local/cuda/compute-sanitizer:${LD_LIBRARY_PATH:-}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)

# Persistent output directory for accelprof result files and the summary log
RESULTS_DIR="${SCRIPT_DIR}/results_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}"

LOG_FILE="${RESULTS_DIR}/summary.log"

# ─── helpers ────────────────────────────────────────────────────────────────

measure_ms() {
    local start end
    start=$(date +%s%N)
    "$@" >/dev/null 2>&1
    end=$(date +%s%N)
    echo $(( (end - start) / 1000000 ))
}

run_and_collect() {
    local n=$1; shift
    local total=0 ms i
    for ((i=1; i<=n; i++)); do
        ms=$(measure_ms "$@")
        echo "${ms}"
        total=$((total + ms))
    done
    echo "avg=$(( total / n ))"
}

calc_stderr() {
    local avg=$1; shift
    local vals=("$@")
    local n=${#vals[@]}
    if (( n < 2 )); then echo "N/A"; return; fi
    local sum_sq="0"
    for v in "${vals[@]}"; do
        sum_sq="${sum_sq} + (${v} - ${avg})^2"
    done
    echo "scale=1; sqrt((${sum_sq}) / (${n} - 1)) / sqrt(${n})" | bc -l
}

# ─── application definitions ────────────────────────────────────────────────
# add_app NAME WORKDIR BINARY "ARGS" ["ENV_VARS"]

NAMES=(); WORKDIRS=(); BINARIES=(); ARGS_LIST=(); ENVS=()

add_app() {
    NAMES+=("$1"); WORKDIRS+=("$2"); BINARIES+=("$3")
    ARGS_LIST+=("$4"); ENVS+=("${5:-}")
}

OPT="${SCRIPT_DIR}"
DATA="${OPT}/rodinia/data"

add_app "rodinia/backprop"          "${OPT}/rodinia/cuda/backprop"          "./backprop"                "65536"
add_app "rodinia/bfs"               "${OPT}/rodinia/cuda/bfs"               "./bfs"                     "${DATA}/bfs/graph1MW_6.txt"
add_app "rodinia/hotspot"           "${OPT}/rodinia/cuda/hotspot"           "./hotspot"                 "512 2 2 ${DATA}/hotspot/temp_512 ${DATA}/hotspot/power_512 output.out"
add_app "rodinia/particlefilter"    "${OPT}/rodinia/cuda/particlefilter"    "./particlefilter_naive"    "-x 128 -y 128 -z 10 -np 1000"
add_app "rodinia/srad_v1"           "${OPT}/rodinia/cuda/srad/srad_v1"      "./srad"                    "100 0.5 502 458"
add_app "rodinia/srad_v2"           "${OPT}/rodinia/cuda/srad/srad_v2"      "./srad"                    "2048 2048 0 127 0 127 0.5 2"
add_app "LULESH"                    "${OPT}/LULESH/cuda/src"                "./lulesh"                  "-s 100"
add_app "XSBench"                   "${OPT}/XSBench/cuda"                   "./XSBench"                 "-s large -m event -l 1700000"
add_app "llama.cpp/soft_max"        "${OPT}/llama.cpp/build/bin"            "./test-backend-ops"        "test -o SOFT_MAX"      "CUDA_VISIBLE_DEVICES=0"
add_app "llama.cpp/rms_norm"        "${OPT}/llama.cpp/build/bin"            "./test-backend-ops"        "test -o RMS_NORM"      "CUDA_VISIBLE_DEVICES=0"
add_app "llama.cpp/rms_norm_back"   "${OPT}/llama.cpp/build/bin"            "./test-backend-ops"        "test -o RMS_NORM_BACK" "CUDA_VISIBLE_DEVICES=0"
add_app "llama.cpp/norm"            "${OPT}/llama.cpp/build/bin"            "./test-backend-ops"        "test -o NORM"          "CUDA_VISIBLE_DEVICES=0"

# ─── main ───────────────────────────────────────────────────────────────────

echo "============================================================"  | tee "${LOG_FILE}"
echo "  cuVein Overhead Benchmark  (${RUNS} runs per mode)"        | tee -a "${LOG_FILE}"
echo "  Started:  $(date)"                                          | tee -a "${LOG_FILE}"
echo "  Results:  ${RESULTS_DIR}"                                   | tee -a "${LOG_FILE}"
echo "============================================================"  | tee -a "${LOG_FILE}"

for idx in "${!NAMES[@]}"; do
    name="${NAMES[$idx]}"
    workdir="${WORKDIRS[$idx]}"
    binary="${BINARIES[$idx]}"
    args="${ARGS_LIST[$idx]}"
    env_vars="${ENVS[$idx]}"

    # Skip if workdir or binary doesn't exist yet
    if [ ! -d "${workdir}" ]; then
        echo "" | tee -a "${LOG_FILE}"
        echo "[ ${name} ]  SKIP — workdir not found: ${workdir}" | tee -a "${LOG_FILE}"
        continue
    fi
    if [ ! -f "${workdir}/${binary#./}" ]; then
        echo "" | tee -a "${LOG_FILE}"
        echo "[ ${name} ]  SKIP — binary not found: ${workdir}/${binary#./}" | tee -a "${LOG_FILE}"
        continue
    fi

    # Per-app accelprof output directory (persistent)
    app_out="${RESULTS_DIR}/${name//\//_}"
    mkdir -p "${app_out}"

    read -ra arg_arr  <<< "${args}"
    read -ra env_arr  <<< "${env_vars}"
    env_prefix=(${env_arr[@]+"env" "${env_arr[@]}"})

    orig_cmd=("${env_prefix[@]+"${env_prefix[@]}"}" "${binary}" "${arg_arr[@]+"${arg_arr[@]}"}")
    pc_dep_cmd=("${env_prefix[@]+"${env_prefix[@]}"}" accelprof -t pc_dependency_analysis -o "${app_out}" "${binary}" "${arg_arr[@]+"${arg_arr[@]}"}")

    pushd "${workdir}" >/dev/null

    echo "" | tee -a "${LOG_FILE}"
    echo "[ ${name} ]" | tee -a "${LOG_FILE}"

    mapfile -t orig_raw   < <(run_and_collect "${RUNS}" "${orig_cmd[@]}")
    mapfile -t pc_dep_raw < <(run_and_collect "${RUNS}" "${pc_dep_cmd[@]}")

    orig_avg="${orig_raw[-1]#avg=}";     unset "orig_raw[-1]"
    pc_dep_avg="${pc_dep_raw[-1]#avg=}"; unset "pc_dep_raw[-1]"

    orig_stderr=$(calc_stderr   "${orig_avg}"   "${orig_raw[@]}")
    pc_dep_stderr=$(calc_stderr "${pc_dep_avg}" "${pc_dep_raw[@]}")

    for ((i=0; i<RUNS; i++)); do
        printf "  run %2d:  orig=%6d ms  pc_dependency=%6d ms\n" \
            $((i+1)) "${orig_raw[$i]}" "${pc_dep_raw[$i]}" \
            | tee -a "${LOG_FILE}"
    done

    printf "  avg:     orig=%6d ms (±%s ms)  pc_dependency=%6d ms (±%s ms)  " \
        "${orig_avg}" "${orig_stderr}" "${pc_dep_avg}" "${pc_dep_stderr}" \
        | tee -a "${LOG_FILE}"
    printf "pc_dependency/orig=%sx\n" \
        "$(echo "scale=2; ${pc_dep_avg} / ${orig_avg}" | bc)" \
        | tee -a "${LOG_FILE}"

    popd >/dev/null
done

echo "" | tee -a "${LOG_FILE}"
echo "============================================================"  | tee -a "${LOG_FILE}"
echo "  Done."                                                        | tee -a "${LOG_FILE}"
echo "  Summary log:     ${LOG_FILE}"                                 | tee -a "${LOG_FILE}"
echo "  accelprof files: ${RESULTS_DIR}/<app>/"                      | tee -a "${LOG_FILE}"
echo "============================================================"  | tee -a "${LOG_FILE}"
