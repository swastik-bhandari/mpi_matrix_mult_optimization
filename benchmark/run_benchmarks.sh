#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR/.."
SRC="$ROOT/src"
RESULTS="$ROOT/results"

mkdir -p "$RESULTS"

echo "Compiling..."
mpicxx -O3 -march=native -o "$SRC/naive_matmul" "$SRC/naive_matmul.cpp"
mpicxx -O3 -march=native -o "$SRC/cannon_matmul" "$SRC/cannon_matmul.cpp"
echo "Done."

SIZES=(256 512 1024)
PROCS=(1 4 9 16)

RESULT_FILE="$RESULTS/raw_timings.csv"
echo "algorithm,n,procs,time_s" > "$RESULT_FILE"

echo ""
echo "Running benchmarks..."
echo "====================="

for n in "${SIZES[@]}"; do
    for p in "${PROCS[@]}"; do
        echo -n "  naive  n=$n p=$p ... "
        output=$(mpirun --oversubscribe -np "$p" "$SRC/naive_matmul" "$n" 2>/dev/null || echo "SKIP")
        if [[ "$output" != "SKIP" && -n "$output" ]]; then
            algo=$(echo "$output" | awk '{print $1}')
            time_val=$(echo "$output" | awk '{print $4}')
            echo "$algo,$n,$p,$time_val" >> "$RESULT_FILE"
            echo "$time_val s"
        else
            echo "skipped (n not divisible by p)"
        fi

        if [[ "$p" == "1" || "$p" == "4" || "$p" == "9" || "$p" == "16" ]]; then
            q=$(echo "sqrt($p)" | bc)
            if (( q * q == p )); then
                echo -n "  cannon n=$n p=$p ... "
                output=$(mpirun --oversubscribe -np "$p" "$SRC/cannon_matmul" "$n" 2>/dev/null || echo "SKIP")
                if [[ "$output" != "SKIP" && -n "$output" ]]; then
                    algo=$(echo "$output" | awk '{print $1}')
                    time_val=$(echo "$output" | awk '{print $4}')
                    echo "$algo,$n,$p,$time_val" >> "$RESULT_FILE"
                    echo "$time_val s"
                else
                    echo "skipped"
                fi
            fi
        fi
    done
done

echo ""
echo "Results saved to $RESULT_FILE"
echo "Run: python3 benchmark/plot_results.py to generate charts"
