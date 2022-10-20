pipeline_settings = {
  'base': {
    'exp01': ["ExperimentConfig.image_scale = 1"]
  },
  'ref': {
    'exp01': ["ExperimentConfig.image_scale = 1"]
  },
  'bone': {
    'exp01': ["ExperimentConfig.image_scale = 1"]
  },
  'ms': {
    'exp28': ["ExperimentConfig.image_scale = 1",
              "NerfModel.use_predicted_mask = True",
              "NerfModel.use_3d_mask = True",
              "NerfModel.use_mask_in_rgb = False",
              "SpecularConfig.mask_ratio_schedule = {'type': 'constant', 'value': 1}",
              "MaskMLP.depth = 8",
              "MaskMLP.width = 128",
              "MaskMLP.output_activation = @jax.nn.relu",
             ],
    'exp29': ["ExperimentConfig.image_scale = 1",
              "NerfModel.use_predicted_mask = True",
              "NerfModel.use_3d_mask = True",
              "NerfModel.use_mask_in_rgb = False",
              "SpecularConfig.mask_ratio_schedule = {'type': 'constant', 'value': 1}",
              "MaskMLP.depth = 8",
              "MaskMLP.width = 128",
              "MaskMLP.output_activation = @jax.nn.relu",
              "NerfModel.use_mask_sharp_weights = True",
             ],
    'exp30': ["ExperimentConfig.image_scale = 1",
              "NerfModel.use_predicted_mask = True",
              "NerfModel.use_3d_mask = True",
              "NerfModel.use_mask_in_rgb = False",
              "SpecularConfig.mask_ratio_schedule = {'type': 'constant', 'value': 1}",
              "MaskMLP.depth = 8",
              "MaskMLP.width = 128",
              "MaskMLP.output_activation = @jax.nn.relu",
              "NerfModel.use_mask_sharp_weights = True",
              "NerfModel.use_x_in_rgb_condition = True",
             ],
    'exp31': ["ExperimentConfig.image_scale = 1",
              "NerfModel.use_predicted_mask = True",
              "NerfModel.use_3d_mask = True",
              "NerfModel.use_mask_in_rgb = False",
              "SpecularConfig.mask_ratio_schedule = {'type': 'constant', 'value': 1}",
              "MaskMLP.depth = 8",
              "MaskMLP.width = 128",
              "MaskMLP.output_activation = @jax.nn.relu",
              "NerfModel.use_mask_sharp_weights = True",
              "NerfModel.use_norm_voxel = True",
             ],
    'exp32': ["ExperimentConfig.image_scale = 1",
              "NerfModel.use_predicted_mask = True",
              "NerfModel.use_3d_mask = True",
              "NerfModel.use_mask_in_rgb = False",
              "SpecularConfig.mask_ratio_schedule = {'type': 'constant', 'value': 1}",
              "MaskMLP.depth = 8",
              "MaskMLP.width = 128",
              "MaskMLP.output_activation = @jax.nn.relu",
              "NerfModel.use_mask_sharp_weights = True",
              "NerfModel.use_delta_x_in_rgb_condition = True",
             ],
  }
}