export DATASET_PATH=/home/zwyan/3d_cv/data/hypernerf/raw/single-vrig-chicken/
export EXPERIMENT_PATH=experiments/s_chicken_spec_exp01_base
CUDA_VISIBLE_DEVICES=0 python train.py \
    --base_folder $EXPERIMENT_PATH \
    --gin_bindings="data_dir='$DATASET_PATH'" \
    --gin_configs configs/test_local.gin