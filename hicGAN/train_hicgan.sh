#!/bin/bash

#SBATCH -p gpu --gres=gpu:1
#SBATCH -J HICGAN
#SBATCH -n 1
#SBATCH -t 80:00:00
#SBATCH --mem=24G
python run_hicGAN.py 0 checkpoint/k562 graph/k562 k562
