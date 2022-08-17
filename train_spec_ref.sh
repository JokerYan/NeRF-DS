#export DATASET_PATH=/hdd/zhiwen/data/hypernerf/raw/americano/
export DATASET_PATH=/home/zwyan/3d_cv/data/hypernerf/raw/americano/
export EXPERIMENT_PATH=experiments/spec_exp18
CUDA_VISIBLE_DEVICES=0 python train.py \
    --base_folder $EXPERIMENT_PATH \
    --gin_bindings="data_dir='$DATASET_PATH'" \
    --gin_configs configs/test_local_spec_ref.gin