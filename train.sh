export DATASET_PATH=/hdd/zhiwen/data/hypernerf/raw/americano/
export EXPERIMENT_PATH=experiments/spec_exp01_base
CUDA_VISIBLE_DEVICES=2 python train.py \
    --base_folder $EXPERIMENT_PATH \
    --gin_bindings="data_dir='$DATASET_PATH'" \
    --gin_configs configs/test_local.gin