#!/usr/bin/env bash

# Launch nvvp
nvvp -vm /usr/lib/jvm/java-8-openjdk-amd64/jre/bin/java &

# Collect traces
# LD_LIBRARY_PATH=/usr/local/cuda-11.0/extras/CUPTI/lib64
# nvprof -f --output-profile profile2.nvprof python3.9 run.py --run-id test --rounds 5 --num-clients-per-round 8 -b 32 --dataset cifar100_fl --model resnet18 --deterministic -li 1  --threadpool-for-local-opt 8
