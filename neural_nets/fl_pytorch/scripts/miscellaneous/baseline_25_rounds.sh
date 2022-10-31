#!/usr/bin/env bash

cd ./../

cmdline="python3.9 run.py --run-id test --rounds 25 --num-clients-per-round 8 -b 32 --dataset cifar100_fl --model resnet18 --deterministic -li 1 --eval-async-threads 1 --threadpool-for-local-opt 8"

echo "Profiled launched command line \"${cmdline}\""

NVIDIA_LIBS_PROFILE=""
NVPROF_PROFILE=""

if [[ ${NVIDIA_LIBS_PROFILE:-} != "" ]];then 
  export CUBLAS_LOGINFO_DBG=1
  export CUBLAS_LOGDEST_DBG=logging_cublas_calls.log

  export CUDNN_LOGINFO_DBG=1
  export CUDNN_LOGDEST_DBG=logging_cudnn_calls.log

  echo "NVIDIA Libs profiling is [ON]"
else
  echo "NVIDIA Libs profiling is [OFF]"
fi


if [ -d ./../check_points/ ]; then
    rm -r ./../check_points/
fi

STARTTIME_SEC=$(date +%s)

if [[ ${NVPROF_PROFILE:-} != "" ]];then
  echo "NVIDIA nvprof profiling is [ON]"
  /usr/local/cuda-11.0/bin/nvprof --print-gpu-trace --log-file nvprof_summary.log ${cmdline}
  #nvprof --log-file nvprof_summary.log ${cmdline}
else
  echo "NVIDIA nvprof profiling is [OFF]"
  ${cmdline}
fi


ENDTIME_SEC=$(date +%s)

echo "Total exec time of \"${cmdline}\""
echo "    $((ENDTIME_SEC-STARTTIME_SEC)) seconds."
