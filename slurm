#!/bin/sh

#SBATCH --job-name=test_script
#SBATCH --output=log_slurm/%j.log
#SBATCH --error=log_slurm/%j.err

#SBATCH --nodes=3
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=long
#SBATCH --nodelist=xgph11,xgph12,xgph13
#SBATCH --time=03-00:00:00  # days-hh:mm:ss

source /home/z/zhiwen/.bashrc
sh train_spec_ref.sh