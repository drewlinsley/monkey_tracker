#!/usr/bin/env bash

arr=(
	lEye neck abdomen lShldr rShldr lForeArm rForeArm
	lHand rHand lMid1 rMid1 lMid3 rMid3 lThigh rThigh
	lShin rShin lFoot rFoot lToe rToe lToeMid3 rToeMid3
	)

for i in "${arr[@]}"
do
   CUDA_VISIBLE_DEVICES=0 python train_and_eval_joints.py --which_joint="$i"
done
