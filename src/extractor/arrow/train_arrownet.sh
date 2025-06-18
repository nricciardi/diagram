#!/bin/bash
#SBATCH --job-name=model_patch128_epoch20_batch8_sigma5_ricciardi
#SBATCH --output=model_patch128_epoch20_batch8_sigma5.out
#SBATCH --error=model_patch128_epoch20_batch8_sigma5.err
#SBATCH --account=cvcs2025
#SBATCH --partition=all_usr_prod
#SBATCH --ntasks=1
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

module load cuda/12.6.3-none-none

base_dir=/work/cvcs2025/garagnani_napolitano_ricciardi/nic
dataset_dir=$base_dir/dataset

export PYTHONPATH=$base_dir

python3 $base_dir/src/extractor/arrow/train_test.py --train_info_file $dataset_dir/train.json --train_images_dir $dataset_dir/train \
--test_info_file $dataset_dir/test.json --test_images_dir $dataset_dir/test \
--patch_size 128 --n_epochs 20 \
--batch_size 8 --output $base_dir/model_patch128_epoch20_batch8_sigma5.pth \
--sigma 5