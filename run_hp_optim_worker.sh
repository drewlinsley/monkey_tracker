#!/bin/bash

# read -p "Enter the ID of the gpu you want to use: "  gpu
# echo "Starting worker on gpu $gpu."

while true
    do
        CUDA_VISIBLE_DEVICES=$1 python train_and_eval_joints.py --hp_optim
        ret=$?
        if [ $ret -ne 0 ]; then
            echo "Worker is finished."
        fi 
    done
