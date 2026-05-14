#!/usr/bin/env bash
set -euo pipefail

GENERATIONS="${GENERATIONS:-10}"
NEUTRON_COUNTS="${NEUTRON_COUNTS:-100 200 300 400 500 600 700 800 900 1000 1250 1500 1750 2000 2500 3000 4000 5000 6000 7000 8000 9000 10000 25000 50000 75000 100000}"
BATCH_SIZES="${BATCH_SIZES:-1 5 10 25 50 100}"
RESULT_DIR="${RESULT_DIR:-results/perf}"
TARGET="${TARGET:-./transport_sim}"
SRUN="${SRUN:-srun -G 1 -n 1}"

mkdir -p "$RESULT_DIR"

summary="$RESULT_DIR/summary.csv"
printf "neutrons,generations,batch_size,wall_seconds,gpu_seconds,completed_generations,interactions,scattering,capture,fission,fission_bank_sites,queue_overflow,fission_bank_overflow,lost_no_surface,log\n" > "$summary"

make clean
make

extract_value() {
    local pattern="$1"
    local file="$2"
    awk -F'= *' -v pat="$pattern" '$0 ~ pat {gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2}' "$file" | tail -n 1
}

for neutrons in $NEUTRON_COUNTS; do
    for batch_size in $BATCH_SIZES; do
        log="$RESULT_DIR/N${neutrons}_G${GENERATIONS}_B${batch_size}.log"
        echo "Running neutrons=$neutrons generations=$GENERATIONS batch_size=$batch_size"

        set +e
        { /usr/bin/time -f "wall_seconds = %e" $SRUN "$TARGET" "$neutrons" "$GENERATIONS" "$batch_size"; } > "$log" 2>&1
        status=$?
        set -e

        wall_seconds="$(extract_value "wall_seconds" "$log")"
        gpu_seconds="$(extract_value "GPU Timed Section Seconds" "$log")"
        completed="$(extract_value "Completed Generations" "$log")"
        interactions="$(extract_value "Number of Interactions" "$log")"
        scattering="$(extract_value "Number of Scattering Events" "$log")"
        capture="$(extract_value "Number of Capture Events" "$log")"
        fission="$(extract_value "Number of Fission Events" "$log")"
        fission_bank="$(extract_value "Number of Fission Bank Sites" "$log")"
        queue_overflow="$(extract_value "Queue Overflowed Particles" "$log")"
        fission_overflow="$(extract_value "Fission Bank Overflowed Particles" "$log")"
        lost_no_surface="$(extract_value "Lost With No Surface" "$log")"

        printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
            "$neutrons" "$GENERATIONS" "$batch_size" "${wall_seconds:-NA}" \
            "${gpu_seconds:-NA}" "${completed:-NA}" "${interactions:-NA}" "${scattering:-NA}" \
            "${capture:-NA}" "${fission:-NA}" "${fission_bank:-NA}" \
            "${queue_overflow:-NA}" "${fission_overflow:-NA}" \
            "${lost_no_surface:-NA}" "$log" >> "$summary"

        if [[ "$status" -ne 0 ]]; then
            echo "Run failed with status $status; see $log"
        fi
    done
done

echo "Wrote $summary"
