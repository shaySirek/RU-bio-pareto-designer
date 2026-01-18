#!/bin/bash

BASE_BIO="bio_data/zea_mays_genes"
BASE_RESULTS="designer_results"
FSM_ID="db_fsm"

while getopts "m:s:f:" opt; do
  case $opt in
    m) MOTIF_ID="$OPTARG" ;;
    s) SEQ_ID="$OPTARG" ;;
    f) FSM_ID="$OPTARG" ;;
    *) exit 1 ;;
  esac
done

if [[ -z "$MOTIF_ID" || -z "$SEQ_ID" ]]; then
    exit 1
fi

TARGET_PATH="${BASE_BIO}/${SEQ_ID}.txt"
RESULTS_DIR="${BASE_RESULTS}/${MOTIF_ID}/${SEQ_ID}/runs/${FSM_ID}"

if [[ ! -f "$TARGET_PATH" || ! -d "$RESULTS_DIR" ]]; then
    exit 1
fi

TARGET_SEQ=$(tr -d '*[:space:]' < "$TARGET_PATH")
TARGET_LEN=${#TARGET_SEQ}

echo "Motif ID:      $MOTIF_ID"
echo "Sequence ID:   $SEQ_ID"
echo "Target Length: $TARGET_LEN"
echo "========================================"

for solution in "$RESULTS_DIR"/*_sequence.txt; do
    [[ -e "$solution" ]] || continue

    SOLUTION_SEQ=$(tr -d '*[:space:]' < "$solution")
    SOLUTION_LEN=${#SOLUTION_SEQ}
    
    echo "Solution $(basename "$solution")"

    if [[ $TARGET_LEN -ne $SOLUTION_LEN ]]; then
        echo "[ERROR: Length Mismatch! Target: $TARGET_LEN, Solution: $SOLUTION_LEN]"
        echo "----------------------------------------"
        continue
    fi

    for (( i=0; i<$TARGET_LEN; i++ )); do
        char_target=${TARGET_SEQ:i:1}
        char_solution=${SOLUTION_SEQ:i:1}

        if [[ "$char_target" != "$char_solution" ]]; then
            printf "  Pos %-5d : %s -> %s\n" $((i+1)) "$char_target" "$char_solution"
        fi
    done
    echo "----------------------------------------"
done