#!/bin/bash

#SBATCH -p gpu --gres=gpu:1
#SBATCH -J HICPLUS
#SBATCH -n 1
#SBATCH -t 80:00:00
#SBATCH --mem=24G

hicplus train -i /users/tdefrosc/data/tdefrosc/coco_herbarium/GSE63525_K562_combined.hic -r 20 -c 19 -o model_epochnumber.model
