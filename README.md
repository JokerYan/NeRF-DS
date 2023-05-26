# NeRF-DS: Neural Radiance Fields for Dynamic Specular Objects
[[Project Page](https://jokeryan.github.io/projects/nerf-ds/)]

This is the code for ["NeRF-DS: Neural Radiance Fields for Dynamic Specular Objects"](http://arxiv.org/abs/2303.14435).

The code is just released, mostly as a placeholder first. More details and clean-ups will be done soon.

```bash
python train.py \
    --base_folder ./experiments/bell_novel_view_ds_exp01 \
    --gin_configs ./configs/nerf_ds.gin \
    --gin_bindings data_dir=\'/home/zwyan/3d_cv/data/hypernerf/raw/bell_novel_view\'
```

```bash
python render.py \
    --base_folder ./experiments/bell_novel_view_ds_exp01 \
    --data_dir /home/zwyan/3d_cv/data/hypernerf/raw/bell_novel_view \
    --interval 1 \
    --chunk_size 4096
```