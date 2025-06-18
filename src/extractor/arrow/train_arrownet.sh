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
#SBATCH --mem=64G

module load cuda/12.6.3-none-none

base_dir=/work/cvcs2025/garagnani_napolitano_ricciardi/nic
dataset_dir=$base_dir/dataset

export PYTHONPATH=$base_dir

python3 $base_dir/src/extractor/arrow/train_test.py --train_info_file $dataset_dir/train.json --train_images_dir $dataset_dir/train \
--test_info_file $dataset_dir/test.json --test_images_dir $dataset_dir/test \
--patch_size 64 --n_epochs 10 \
--batch_size 8 --output $base_dir/test.pth