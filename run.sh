#! /bin/bash
DATASHAPE=256
CROPSHAPE=224

#NETWORK=resnet-50
#LOADMODEL=models/resnet-50
NETWORK=inception-v3
LOADMODEL=models/Inception-V3

CLASSES=1000
#CLASSES=$(find ./data/train -mindepth 1 -type d | wc -l)
echo found number of classes: $CLASSES
SAMPLES=$(wc -l < "data/imagenet_train.lst")
echo found number of training samples: $SAMPLES
BATCHSIZE=256

python train.py \
        --data-dir ./data \
        --network ${NETWORK} \
        --num-classes ${CLASSES} \
        --num-examples ${SAMPLES} \
        --batch-size ${BATCHSIZE} \
        --save-model-prefix checkpoints/$1 \
        --model-load-epoch 0 \
        --save-log-prefix log/$1 \
        --gpus $2
        # --load-model-prefix ${LOADMODEL} \
