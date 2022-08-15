export DATASET_PATH=/home/zwyan/3d_cv/data/hypernerf/raw/vrig-chicken/
export EXPERIMENT_PATH=experiments/chicken_spec_exp01_base
CUDA_VISIBLE_DEVICES=0 python train.py \
    --base_folder $EXPERIMENT_PATH \
    --gin_bindings="data_dir='$DATASET_PATH'" \
    --gin_configs configs/test_local.gin

export EXPERIMENT_PATH=experiments/chicken_spec_exp01
CUDA_VISIBLE_DEVICES=0 python train.py \
    --base_folder $EXPERIMENT_PATH \
    --gin_bindings="data_dir='$DATASET_PATH'" \
    --gin_configs configs/test_local_spec_r.gin

export EXPERIMENT_PATH=experiments/chicken_spec_exp02
CUDA_VISIBLE_DEVICES=0 python train.py \
    --base_folder $EXPERIMENT_PATH \
    --gin_bindings="data_dir='$DATASET_PATH'" \
    --gin_configs configs/test_local_spec_s.gin

export EXPERIMENT_PATH=experiments/chicken_spec_exp03
CUDA_VISIBLE_DEVICES=0 python train.py \
    --base_folder $EXPERIMENT_PATH \
    --gin_bindings="data_dir='$DATASET_PATH'" \
    --gin_configs configs/test_local_spec_ref.gin

export DATASET_PATH=/home/zwyan/3d_cv/data/hypernerf/raw/americano/
export EXPERIMENT_PATH=experiments/spec_exp13_base
CUDA_VISIBLE_DEVICES=0 python train.py \
    --base_folder $EXPERIMENT_PATH \
    --gin_bindings="data_dir='$DATASET_PATH'" \
    --gin_configs configs/test_local.gin

export DATASET_PATH=/home/zwyan/3d_cv/data/hypernerf/raw/americano/
export EXPERIMENT_PATH=experiments/spec_exp11
CUDA_VISIBLE_DEVICES=0 python train.py \
    --base_folder $EXPERIMENT_PATH \
    --gin_bindings="data_dir='$DATASET_PATH'" \
    --gin_configs configs/test_local_spec_ref.gin