#!/bin/bash
for ((i = 0 ; i < 21 ; i++))
do
    if [ $i -eq 0 ]; then
        python train.py \
            --root data \
            --seed 1 \
            --trainer LP_DiF \
            --dataset-config-file configs/datasets/sun397.yaml \
            --config-file configs/trainers/LP_DiF/vit_b16.yaml \
            --output-dir output/LP_DiF/sun397/session0 \
            TRAINER.TASK_ID 0 
    else
        j=$(($i-1))
        python train.py \
            --root data \
            --seed 1 \
            --trainer LP_DiF \
            --dataset-config-file configs/datasets/sun397.yaml \
            --config-file configs/trainers/LP_DiF/vit_b16.yaml \
            --output-dir output/LP_DiF/sun397/session${i} \
            --model-dir output/LP_DiF/sun397/session${j} \
            TRAINER.TASK_ID ${i} 
    fi
done