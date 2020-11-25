#!/bin/bash

#SBATCH -J HICGAN_CPU
#SBATCH -n 8
#SBATCH -t 168:00:00
#SBATCH --mem=24G
python run_hicGAN.py 0 checkpoint_cpu/k562 graph_cpu/k562 k562
