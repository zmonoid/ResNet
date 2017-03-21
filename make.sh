cd data
PREFIX=imagenet
SHAPE=480
# python im2rec.py fish train test --list True --train-ratio 0.9 --recursive True
~/svn/mxnet/bin/im2rec ${PREFIX}_train.lst train/ train_${SHAPE}.rec resize=${SHAPE} quality=90
~/svn/mxnet/bin/im2rec ${PREFIX}_val.lst train/ val_${SHAPE}.rec resize=${SHAPE} quality=90
~/svn/mxnet/bin/im2rec ${PREFIX}_test.lst test/ test_${SHAPE}.rec resize=${SHAPE} quality=90

#~/mxnet/bin/im2rec ${PREFIX}_${index}_train.lst train/ train_${SHAPE}_${index}.rec resize=${SHAPE} quality=100
#~/mxnet/bin/im2rec ${PREFIX}_${index}_val.lst train/ val_${SHAPE}_${index}.rec resize=${SHAPE} quality=100

#SHAPE=227
#~/mxnet/bin/im2rec ${PREFIX}_train.lst train/ train_${SHAPE}.rec resize=${SHAPE} quality=100
#~/mxnet/bin/im2rec ${PREFIX}_val.lst train/ val_${SHAPE}.rec resize=${SHAPE} quality=100
# 
#SHAPE=224
#~/mxnet/bin/im2rec ${PREFIX}_train.lst train/ train_${SHAPE}.rec resize=${SHAPE} quality=100
#~/mxnet/bin/im2rec ${PREFIX}_val.lst train/ val_${SHAPE}.rec resize=${SHAPE} quality=100
#~/mxnet/bin/im2rec ${PREFIX}_test.lst test/ test_${SHAPE}.rec resize=${SHAPE} quality=100
