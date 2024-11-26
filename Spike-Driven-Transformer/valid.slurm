#!/bin/bash
#SBATCH -p gpupar
#SBATCH --cpus-per-task 8
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --nodelist gpu01
#SBATCH -o ./slurm%A_mix.log
#SBATCH -e ./slurm%A_mix.log

# Validate trained model on imagetnet-1k
export CUDA_VISIBLE_DEVICES=4

# For pytorch > 2.0.0
# torchrun --nproc_per_node=1 --master_port 29521 firing_num.py -c conf/imagenet/8_768_300E_t4.yml --model sdt --spike-mode lif --resume checkpoints/8_768.pth.tar --no-resume-opt -data-dir /data0/share/datasets/ImageNet -vb 16
torchrun --nproc_per_node=1 --master_port 29521 firing_num.py -c conf/imagenet/8_768_300E_t4.yml --model sdt --spike-mode lif --resume checkpoints/8_768.pth.tar --no-resume-opt -data-dir /data0/share/datasets/ImageNet -vb 16 --large-valid

# For 1.10.0 <= pytorch < 2.0.0
# python -m torch.distributed.launch --nproc_per_node=1 --master_port 29521 firing_num.py -c conf/imagenet/8_768_300E_t4.yml --model sdt --spike-mode lif --resume checkpoints/8_768.pth.tar --no-resume-opt -data-dir /data0/share/datasets/ImageNet -vb 16
# python -m torch.distributed.launch --nproc_per_node=1 --master_port 29521 firing_num.py -c conf/imagenet/8_768_300E_t4.yml --model sdt --spike-mode lif --resume checkpoints/8_768.pth.tar --no-resume-opt -data-dir /data0/share/datasets/ImageNet -vb 16 --large-valid
