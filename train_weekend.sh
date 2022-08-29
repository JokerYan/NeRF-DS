export DATASET_PATH=/home/zwyan/3d_cv/data/hypernerf/raw/aluminium-sheet-5/
export EXPERIMENT_PATH=experiments/as5_exp01_base
CUDA_VISIBLE_DEVICES=0 python train.py \
    --base_folder $EXPERIMENT_PATH \
    --gin_bindings="data_dir='$DATASET_PATH'" \
    --gin_configs configs/test_local.gin

export DATASET_PATH=/home/zwyan/3d_cv/data/hypernerf/raw/aluminium-sheet-5/
export EXPERIMENT_PATH=experiments/as5_exp02
CUDA_VISIBLE_DEVICES=0 python train.py \
    --base_folder $EXPERIMENT_PATH \
    --gin_bindings="data_dir='$DATASET_PATH'" \
    --gin_configs configs/test_local_spec_hc.gin

export DATASET_PATH=/home/zwyan/3d_cv/data/hypernerf/raw/aluminium-sheet-5/
export EXPERIMENT_PATH=experiments/as5_exp01
CUDA_VISIBLE_DEVICES=0 python train.py \
    --base_folder $EXPERIMENT_PATH \
    --gin_bindings="data_dir='$DATASET_PATH'" \
    --gin_configs configs/test_local_spec_ref.gin



export DATASET_PATH=/home/zwyan/3d_cv/data/hypernerf/raw/white-board-6/
export EXPERIMENT_PATH=experiments/wb6_exp01_base
CUDA_VISIBLE_DEVICES=0 python train.py \
    --base_folder $EXPERIMENT_PATH \
    --gin_bindings="data_dir='$DATASET_PATH'" \
    --gin_configs configs/test_local.gin

export DATASET_PATH=/home/zwyan/3d_cv/data/hypernerf/raw/white-board-6/
export EXPERIMENT_PATH=experiments/wb6_exp02
CUDA_VISIBLE_DEVICES=0 python train.py \
    --base_folder $EXPERIMENT_PATH \
    --gin_bindings="data_dir='$DATASET_PATH'" \
    --gin_configs configs/test_local_spec_hc.gin

export DATASET_PATH=/home/zwyan/3d_cv/data/hypernerf/raw/white-board-6/
export EXPERIMENT_PATH=experiments/wb6_exp01
CUDA_VISIBLE_DEVICES=0 python train.py \
    --base_folder $EXPERIMENT_PATH \
    --gin_bindings="data_dir='$DATASET_PATH'" \
    --gin_configs configs/test_local_spec_ref.gin