#!/bin/bash

DATA_DIR="/root/autodl-tmp/COCO"
BATCH_SIZE=16
EPOCHS=50
EXPERIMENT_NAME="FSCCNET_Ablation_three"
GPU_ID=0
MESSAGE=64
NOISE="Jpeg(50)+gaussian(3,2)+dropout(0.5,0.5)+cropout((0.49, 0.49), (0.49, 0.49))+Gaussian_noise(0.0,0.10)"
#NOISE="Jpeg(50)"
#NOISE="gaussian(3,2)" # (k,σ)
#NOISE="dropout(0.5,0.5)"
#NOISE="resize(0.5,1.5)"
#NOISE="crop((0.25, 0.25), (0.25, 0.25))"
#NOISE="cropout((0.25, 0.25), (0.25, 0.25))"
#NOISE="Gaussian_noise(0.0,0.10)"
#NOISE="sp(0.05)"
#NOISE="Median_filter(3)"
#NOISE="Adjust_brightness(1.1)"
#NOISE="Adjust_contrast(1.5)"
#NOISE="Adjust_hue(0.2)"
#NOISE="Adjust_saturation(15)"
#NOISE="grid_crop(0.3)"
#NOISE="grid_crop(0.3)"


# ============ 启动命令 ============
export CUDA_VISIBLE_DEVICES=$GPU_ID

CMD="python main.py new \
  --name ${EXPERIMENT_NAME} \
  --data-dir ${DATA_DIR} \
  --batch-size ${BATCH_SIZE} \
  --epochs ${EPOCHS} \
  --message ${MESSAGE}"

if [ ! -z "$NOISE" ]; then
  CMD="$CMD --noise \"$NOISE\""
fi

echo "运行命令: $CMD"
eval $CMD


