#!/bin/sh
#SBATCH -J FLLA-Net
#SBATCH -o /home/xiaridehehe/ownProgram/ChangeViT/result/out/train.out.%j
#SBATCH -e /home/xiaridehehe/ownProgram/ChangeViT/result/err/train.err.%j
#SBATCH --partition=gpu-l20
#SBATCH -w gpu58
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
source activate UNetPlus
python trainDLAR_Net.py
