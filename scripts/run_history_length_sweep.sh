#!/usr/bin/env bash
set -euo pipefail

GENERATIONS="${GENERATIONS:-10}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NEUTRON_COUNTS="${NEUTRON_COUNTS:-100 200 300 400 500 600 700 800 900 1000 1250 1500 1750 2000 2500 3000 4000 5000 6000 7000 8000 9000 10000 25000 50000 75000 100000}"
RESULT_DIR="${RESULT_DIR:-results/history_lengths}"
TARGET="${TARGET:-./transport_sim}"
SRUN="${SRUN:-srun -G 1 -n 1}"

mkdir -p "$RESULT_DIR"

summary="$RESULT_DIR/summary.csv"
printf "neutrons,generations,batch_size,csv,log\n" > "$summary"

make clean
make PROFILE_HISTORY_LENGTHS=1

for neutrons in $NEUTRON_COUNTS; do
    csv="$RESULT_DIR/history_moves_N${neutrons}_G${GENERATIONS}_B${BATCH_SIZE}.csv"
    log="$RESULT_DIR/history_moves_N${neutrons}_G${GENERATIONS}_B${BATCH_SIZE}.log"

    echo "Running neutrons=$neutrons generations=$GENERATIONS batch_size=$BATCH_SIZE"
    $SRUN "$TARGET" "$neutrons" "$GENERATIONS" "$BATCH_SIZE" "$csv" > "$log" 2>&1

    printf "%s,%s,%s,%s,%s\n" \
        "$neutrons" "$GENERATIONS" "$BATCH_SIZE" "$csv" "$log" >> "$summary"
done

echo "Wrote $summary"
