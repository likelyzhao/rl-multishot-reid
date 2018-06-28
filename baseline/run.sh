#!/usr/bin/env bash
cd /workspace/mnt/group/video/zhaozhijian/rl-multishot-reid/baseline
bs=64
epochs=1
sets=0
gpus=$1
data_dir=$4
mode='SAVE'
case $2 in
  iLiDS-VID)
    base=ilds_$3_$sets
    num_id=150
    train_set=image_valid$sets
    valid_set=image_test$sets
    ;;
  PRID-2011)
    base=prid_$3_$sets
    num_id=100
    train_set=image_valid$sets
    valid_set=image_test$sets
    ;;
  MARS)
    base=mars_$3
    num_id=624
    train_set=image_train
    test_set=image_valid
    ;;
  *)
    echo "No valid dataset"
    exit
    ;;
esac

case $3 in
  alexnet)
    python2 baseline.py --gpus $gpus --data-dir $data_dir \
        --num-id $num_id --batch-size $bs \
        --train-file $train_set --test-file $test_set \
        --lr 1e-4 --num-epoches $epochs --mode $mode \
        --network alexnet --model-load-prefix alexnet --model-load-epoch 1
    ;;
  inception-bn)
     python2 baseline.py --gpus $gpus --data-dir $data_dir \
        --num-id $num_id --batch-size $bs \
        --train-file $train_set --test-file $test_set \
        --lr 1e-2 --num-epoches $epochs --mode $mode  --lmnn #--lsoftmax
    ;;
  *)
    echo "No valid basenet"
    exit
    ;;
esac
