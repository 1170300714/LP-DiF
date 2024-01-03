#!/bin/bash
for ((i = 0 ; i < 20 ; i++))
do
    if [ $i -eq 0 ]; then
        python train.py \
            --root data \
            --seed 1 \
            --trainer LP_DiF \
            --dataset-config-file configs/datasets/cub200_wo_base.yaml \
            --config-file configs/trainers/LP_DiF/vit_b16.yaml \
            --output-dir output/LP_DiF/cub200_wo_base/session0 \
            TRAINER.TASK_ID 0 
    else
        j=$(($i-1))
        python train.py \
            --root data \
            --seed 1 \
            --trainer LP_DiF \
            --dataset-config-file configs/datasets/cub200_wo_base.yaml \
            --config-file configs/trainers/LP_DiF/vit_b16.yaml \
            --output-dir output/LP_DiF/cub200_wo_base/session${i} \
            --model-dir output/LP_DiF/cub200_wo_base/session${j} \
            TRAINER.TASK_ID ${i} 
    fi
done