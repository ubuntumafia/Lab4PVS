#!/bin/bash

if [ $# -ne 2 ]; then
  echo "Использование: $0 ARRAY_SIZE NUM_RUNS"
  exit 1
fi

ARRAY_SIZE=$1
NUM_RUNS=$2
total_time=0

for ((i=1; i<=NUM_RUNS; i++)); do
  line=$(./seq_merge_sort "$ARRAY_SIZE" | grep "Время выполнения")
  t=$(awk '{print $3}' <<< "$line")
  total_time=$(awk -v acc="$total_time" -v dt="$t" 'BEGIN{printf "%.9f", acc + dt}')
done

avg_time=$(awk -v acc="$total_time" -v n="$NUM_RUNS" 'BEGIN{printf "%.9f", acc / n}')
echo "Размер массива: $ARRAY_SIZE"
echo "Число запусков: $NUM_RUNS"
echo "Среднее время: $avg_time секунд"
