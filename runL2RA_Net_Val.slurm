#!/bin/sh
#SBATCH -J ReMML2RL_Net_Val
#SBATCH -o /home/xiaridehehe/ownProgram/ChangeViT/result/out/train.out.%j
#SBATCH -e /home/xiaridehehe/ownProgram/ChangeViT/result/err/train.err.%j
#SBATCH --partition=gpu-a10
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
source activate UNetPlus
python trainL2RA_Net_TSNE.py --model_path_classifier /home/xiaridehehe/ownProgram/ReMM/Weight/L2RA-Ablation/weight0.1-1/classifier1_ckt.pth

# /home/xiaridehehe/ownProgram/ReMM/Weight/L2RA-Ablation/weight0.1-1/classifier1_ckt.pth
