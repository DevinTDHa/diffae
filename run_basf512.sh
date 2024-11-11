#!/bin/sh
#SBATCH --job-name=dae_basf512
#SBATCH --partition=gpu-2d
#SBATCH --constraint=80gb
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=2
#SBATCH --output=logs/job-%j.out
#SBATCH --workdir=/home/tha/diffae

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

cd 
apptainer run \
    -B ~/datasets/squashed/basf_resize512.sqfs:/data/basf:image-src=/ \
    --nv ~/apptainers/thesis.sif \
    python run_basf512.py
