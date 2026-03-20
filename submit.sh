#!/bin/bash
#SBATCH --partition=gpu-batch
#SBATCH --account=sci-lippert
#SBATCH --cpus-per-task=8
#SBATCH --gpus=a100:1
#SBATCH --mem=32G
#SBATCH --time=20:00:00

cd SkinLesionSegmentation
conda activate image_segmentation
pixi run wandb agent jakob-fanselow-hasso-plattner-institut/SkinLesionSegmentation/bx28yiqm
