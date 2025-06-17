#!/bin/bash
#SBATCH --job-name=training_ricciardi
#SBATCH --output=train.out
#SBATCH --error=train.err
#SBATCH --account=cvcs2025
#SBATCH --partition=all_usr_prod
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

module load cuda/11.8
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate nic

base_dir=/work/cvcs2025/garagnani_napolitano_ricciardi/nic
dataset_dir=$base_dir/dataset

python3 $base_dir/src/extractor/arrow/train.py --info_file $dataset_dir/arrow/train.json --images_dir $dataset_dir/arrow/train --patch_size 64 --n_epochs 10 --output $base_dir/test.pth