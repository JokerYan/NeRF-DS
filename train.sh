#export DATASET_PATH=/hdd/zhiwen/data/hypernerf/raw/americano/
#export DATASET_PATH=/home/zwyan/3d_cv/data/hypernerf/raw/americano/
export DATASET_PATH=/home/zwyan/3d_cv/data/hypernerf/raw/white-board-6/
#export DATASET_PATH=/home/zwyan/3d_cv/data/hypernerf/raw/aluminium-sheet-5/
export EXPERIMENT_PATH=experiments/wb6_exp01_base
CUDA_VISIBLE_DEVICES=0 python train.py \
    --base_folder $EXPERIMENT_PATH \
    --gin_bindings="data_dir='$DATASET_PATH'" \
    --gin_configs configs/test_local.gin
#    --gin_configs configs/hypernerf_vrig_ap_2d_test.gin