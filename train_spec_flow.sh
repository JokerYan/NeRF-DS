export DATASET_PATH=/home/zwyan/3d_cv/data/hypernerf/raw/bell-3_qualitative/
export EXPERIMENT_PATH=experiments/b3_q_ref_exp02

CUDA_VISIBLE_DEVICES=0 python train_flow.py \
    --base_folder $EXPERIMENT_PATH \
    --gin_bindings="data_dir='$DATASET_PATH'" \
    --gin_configs configs/test_local_spec_ref.gin
#    --gin_bindings="ExperimentConfig.image_scale = 1"