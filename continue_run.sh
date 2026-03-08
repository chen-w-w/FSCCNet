#!/bin/bash



EPOCHS=200
GPU_ID=0
CONTINUE_FOLDER="FSCCNET_combined 2025.12.20--16-36-33"



export CUDA_VISIBLE_DEVICES=$GPU_ID

CMD="python main.py continue \
  --folder \"./runs/${CONTINUE_FOLDER}\" \
  --epochs ${EPOCHS} "

echo "运行命令: $CMD"
eval $CMD

