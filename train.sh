#!/usr/bin/env bash

mkdir checkpoints

python -u main.py \
    -shuffle \
    -train_record \
    -model spearkerresnet14 \
    -data_dir ./data/data \
    -train_list ./data/train.txt \
    -test_list ./data/test.txt \
    -save_path checkpoints \
    -output_classes 512 \
    -n_epochs 120 \
    -learn_rate 0.003 \
    -batch_size 64 \
    -workers 6 \
    -nGPU 1 \
    -decay 30 \
    -softmax \
2>&1 | tee train.log
