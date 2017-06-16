#!/usr/bin/env bash

arr=(rForeArm lHand rHand lMid1 rMid1 lMid3 rMid3)

for i in rForeArm lHand rHand lMid1 rMid1 lMid3 rMid3
do
   CUDA_VISIBLE_DEVICES=3 python train_and_eval_joints.py --which_joint=$i
done
