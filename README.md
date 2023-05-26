# NeRF-DS: Neural Radiance Fields for Dynamic Specular Objects
[[Project Page](https://jokeryan.github.io/projects/nerf-ds/)]

This is the code for ["NeRF-DS: Neural Radiance Fields for Dynamic Specular Objects"](http://arxiv.org/abs/2303.14435).

### Remarks
This repo has recently been cleaned up. 
If you encounter any problem or have any question, please kindly submit an issue.

Also, due to the slow training and rendering of HyperNeRF from the last generation, this repo might be considered as too slow nowadays.
You might want to consider applying the technique proposed in our paper to a more recent NeRF implementation, 
such as [TiNeuVox](https://github.com/hustvl/TiNeuVox) or [K-Planes](https://github.com/sarafridov/K-Planes).

### Environment
The required packages can be installed by `pip install -r requirements.txt`. 
However, some of the packages could have some compatibility issues over time, 
so we provide an exact environment file in `requirements_exact.txt` for reference.

### Data
The data structure is the same as [HyperNeRF](https://github.com/google/hypernerf), 
except for the foreground mask of the moving objects in the training views. 
You should reuse the mask from the camera registration process.
You can refer to the data released for a more detailed file structure.
```
-- exp_dir
 |-- camera
 |-- resized_mask       # training view masks, foreground as 0
 |-- rgb
 |-- train_camera
 |-- vrig_camera        # test cameras
 |-- dataset.json
 |-- metadata.json
 |-- points.npy         # not required if not using background loss
 |-- scene.json
```

### Train
The model can be trained with the following command:
```bash
python train.py \
    --base_folder $EXPERIMENT_DIR \
    --gin_configs ./configs/nerf_ds.gin \
    --gin_bindings data_dir=\'$DATA_DIR' 
```
The `$EXPERIMENT_DIR` is the directory to save the experiment results, 
and the `$DATA_DIR` is the directory of the training data. 

### Render
The trained model can be trained with the following command:
```bash
python render.py \
    --base_folder ./experiments/bell_novel_view_ds_exp01 \
    --data_dir /home/zwyan/3d_cv/data/hypernerf/raw/bell_novel_view \
    --interval 1 \
    --chunk_size 4096
```
The `interval` is the interval of frames to render.
The `chunk_size` is the batch size of rays for rendering.

If our paper is useful for your research, please consider citing:
```
@inproceedings{yan2023nerf,
  title={NeRF-DS: Neural Radiance Fields for Dynamic Specular Objects},
  author={Yan, Zhiwen and Li, Chen and Lee, Gim Hee},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={8285--8295},
  year={2023}
}
```