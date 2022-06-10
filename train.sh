export DATASET_PATH=dataset/espresso
export EXPERIMENT_PATH=experiments
python train.py \
    --base_folder $EXPERIMENT_PATH \
    --gin_bindings="data_dir='$DATASET_PATH'" \
    --gin_configs configs/test_local.gin