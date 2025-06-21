#!/bin/bash

if [ $# -ne 2 ]; then
  echo "Использование: $0 ARRAY_SIZE THREADS_PER_BLOCK"
  exit 1
fi

ARRAY_SIZE=$1
THREADS_PER_BLOCK=$2
NUM_RUNS=100
total_time=0

echo "Запуск CUDA-программы с размером массива $ARRAY_SIZE и $THREADS_PER_BLOCK потоков на блок (100 раз)"

for ((i=1; i<=NUM_RUNS; i++)); do
  line=$(./array_ops_cuda "$ARRAY_SIZE" "$THREADS_PER_BLOCK" | grep "Время выполнения")
  t=$(awk '{print $4}' <<< "$line")
  total_time=$(awk -v acc="$total_time" -v dt="$t" 'BEGIN{printf "%.9f", acc + dt}')
done

avg_time=$(awk -v acc="$total_time" -v n="$NUM_RUNS" 'BEGIN{printf "%.9f", acc / n}')
echo "------------------------------------------"
echo "Среднее время выполнения: $avg_time секунд"
