#export DATASET_PATH=/home/zwyan/3d_cv/data/hypernerf/raw/008_bell_04_novel_view/
#export EXPERIMENT_PATH=experiments/008_b04_nv_ref_exp01
export DATASET_PATH=/home/zwyan/3d_cv/data/hypernerf/raw/009_bell_05_novel_view/
export EXPERIMENT_PATH=experiments/009_b05_nv_ref_exp01

CUDA_VISIBLE_DEVICES=0 python train_flow.py \
    --base_folder $EXPERIMENT_PATH \
    --gin_bindings="data_dir='$DATASET_PATH'" \
    --gin_configs configs/test_local_spec_ref.gin \
    --gin_bindings="ExperimentConfig.image_scale = 1"