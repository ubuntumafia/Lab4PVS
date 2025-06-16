#!/bin/bash
if [ $# -ne 5 ]; then
  echo "Usage: $0 <path/to/task4_cuda> <rows> <cols> <operation> <block_size>"
  echo "Operations: 0-add, 1-sub, 2-mul, 3-div"
  exit 1
fi

PROG=$1
ROWS=$2
COLS=$3
OP=$4
BLOCK_SIZE=$5
N=$((ROWS * COLS))

if [ ! -x "$PROG" ]; then
  echo "Error: cannot execute '$PROG'"
  exit 1
fi

if (( N < 100000 )); then
  echo "Error: Total elements must be >= 100000"
  exit 1
fi

SUM=0.0
REPEATS=10

for i in $(seq 1 $REPEATS); do
  T=$("$PROG" "$ROWS" "$COLS" "$OP" "$BLOCK_SIZE")
  
  if ! [[ $T =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    echo "Error: program output is not a number: '$T'"
    exit 1
  fi
  
  SUM=$(awk "BEGIN{ printf \"%.10f\", $SUM + $T }")
done

AVG=$(awk "BEGIN{ printf \"%.6f\", $SUM / $REPEATS }")

echo "Rows: $ROWS, Cols: $COLS, Block: ${BLOCK_SIZE}x${BLOCK_SIZE}, Op: $OP"
echo "Average time over $REPEATS runs: $AVG s"