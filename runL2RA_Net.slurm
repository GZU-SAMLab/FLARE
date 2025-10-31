#!/bin/sh
#SBATCH -J ReMML2RLLayer2
#SBATCH -o /home/xiaridehehe/ownProgram/ChangeViT/result/out/train.out.%j
#SBATCH -e /home/xiaridehehe/ownProgram/ChangeViT/result/err/train.err.%j
#SBATCH --partition=gpu-l20
#SBATCH -w gpu59
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
source activate UNetPlus
python trainL2RA_Net_Test.py --epochs 300 --batch_size 16 \
    --model_path_lesion /home/xiaridehehe/ownProgram/ReMM/Weight/Unet++Lesion_1_151.pth \
    --model_path_leaf /home/xiaridehehe/ownProgram/ReMM/Weight/Unet++_Leaf_124.pth

#    --model_path_lesion /home/xiaridehehe/ownProgram/ReMM/Weight/DLAR_4301/Unet++_1_151.pth \
#    --model_path_leaf /home/xiaridehehe/ownProgram/ReMM/Weight/Unet++_Leaf_124.pth