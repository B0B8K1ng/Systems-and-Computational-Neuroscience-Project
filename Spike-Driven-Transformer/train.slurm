#!/bin/bash
#SBATCH -p gpupar
#SBATCH --cpus-per-task 8
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --nodelist gpu01
#SBATCH -o ./slurm%A_mix.log
#SBATCH -e ./slurm%A_mix.log

# Run train.py on imagetnet-1k
torchrun --nproc_per_node=1 --master_port 29501 train.py -c conf/imagenet/8_768_300E_t4.yml --model sdt --spike-mode lif -data-dir /data0/share/datasets/ImageNet
