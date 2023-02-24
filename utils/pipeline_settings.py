pipeline_settings = {
    'base': {
        'exp01': ["ExperimentConfig.image_scale = 1"],
        'exp02': ["ExperimentConfig.image_scale = 4"],
        'exp03': ["ExperimentConfig.image_scale = 4",
                  "TrainConfig.use_background_loss = False",
                  "NerfModel.hyper_embed_key = 'appearance'",
                  "NerfModel.use_rgb_condition = False"
                  ],
        'exp44': [
            "ExperimentConfig.image_scale = 2",
            "batch_size=512",
            "max_steps=2500000",
            "lr_decay_steps=2500000",
            "init_lr = 0.001",
            "final_lr = 0.0001",
            "NerfModel.num_coarse_samples = 128",
            "NerfModel.num_coarse_samples = 128",
        ],
        'exp45': [
            "ExperimentConfig.image_scale = 2",
            "batch_size=512",
            "max_steps=2500000",
            "lr_decay_steps=2500000",
            "init_lr = 0.001",
            "final_lr = 0.0001",
            "NerfModel.num_coarse_samples = 128",
            "NerfModel.num_coarse_samples = 128",
            "TrainConfig.hyper_alpha_schedule = %LONG_DELAYED_HYPER_ALPHA_SCHED",
            "warp_alpha_steps = 800000"
        ],
        'exp46': [
            "ExperimentConfig.image_scale = 2",
            "batch_size=512",
            "max_steps=2500000",
            "lr_decay_steps=2500000",
            "init_lr = 0.0001",
            "final_lr = 0.00001",
            "NerfModel.num_coarse_samples = 128",
            "NerfModel.num_coarse_samples = 128",
            "TrainConfig.hyper_alpha_schedule = %LONG_DELAYED_HYPER_ALPHA_SCHED",
            "warp_alpha_steps = 800000"
        ],
        'exp47': [
            "ExperimentConfig.image_scale = 2",
            "batch_size=512",
            "max_steps=2500000",
            "lr_decay_steps=2500000",
            "init_lr = 0.0003",
            "final_lr = 0.00003",
            "NerfModel.num_coarse_samples = 128",
            "NerfModel.num_coarse_samples = 128",
            "TrainConfig.hyper_alpha_schedule = %LONG_DELAYED_HYPER_ALPHA_SCHED",
            "warp_alpha_steps = 800000"
        ],
    },
    'nerfies': {
        'exp01': ["ExperimentConfig.image_scale = 1"]
    },
    'ref': {
        'exp01': ["ExperimentConfig.image_scale = 1"],
        'exp02': ["ExperimentConfig.image_scale = 1",
                  "NerfModel.norm_input_posenc = False"],
        'exp03': ["ExperimentConfig.image_scale = 1",
                  "NerfModel.norm_input_posenc = False"],
        'exp05': ["ExperimentConfig.image_scale = 2"],
        'exp44': [
            "ExperimentConfig.image_scale = 2",
            "batch_size=512",
            "max_steps=2500000",
            "lr_decay_steps=2500000",
            "init_lr = 0.001",
            "final_lr = 0.0001",
            "NerfModel.num_coarse_samples = 128",
            "NerfModel.num_coarse_samples = 128",
        ]
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
                  "TrainConfig.save_every = 5000",
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
        'exp33': ["ExperimentConfig.image_scale = 1",
                  "NerfModel.use_predicted_mask = True",
                  "NerfModel.use_3d_mask = True",
                  "NerfModel.use_mask_in_rgb = False",
                  "SpecularConfig.mask_ratio_schedule = {'type': 'constant', 'value': 1}",
                  "MaskMLP.depth = 8",
                  "MaskMLP.width = 128",
                  "MaskMLP.output_activation = @jax.nn.relu",
                  "NerfModel.use_hyper_for_sigma = False",
                  "NerfModel.use_hyper_for_rgb = True"
                  ],
        'exp34': ["ExperimentConfig.image_scale = 1",
                  "NerfModel.use_predicted_mask = True",
                  "NerfModel.use_3d_mask = True",
                  "NerfModel.use_mask_in_rgb = False",
                  "SpecularConfig.mask_ratio_schedule = {'type': 'constant', 'value': 1}",
                  "MaskMLP.depth = 8",
                  "MaskMLP.width = 128",
                  "MaskMLP.output_activation = @jax.nn.relu",
                  "NerfModel.use_rgb_sharp_weights = True",
                  """SpecularConfig.sharp_mask_std_schedule = {
                      'type': 'piecewise',
                      'schedules': [
                          (30000, ('exponential', 1, 0.1, 30000)),
                          (300000, ('constant', 0.1))
                      ]
                  }"""
                  ],
        'exp35': ["ExperimentConfig.image_scale = 1",
                  "NerfModel.use_predicted_mask = True",
                  "NerfModel.use_3d_mask = True",
                  "NerfModel.use_mask_in_rgb = False",
                  "SpecularConfig.mask_ratio_schedule = {'type': 'constant', 'value': 1}",
                  "MaskMLP.depth = 8",
                  "MaskMLP.width = 128",
                  "MaskMLP.output_activation = @jax.nn.relu",
                  "NerfModel.use_mask_sharp_weights = True",
                  "NerfModel.use_x_in_rgb_condition = True",
                  "NerfModel.window_x_in_rgb_condition = True"
                  ],
        'exp36': ["ExperimentConfig.image_scale = 1",
                  "NerfModel.use_predicted_mask = True",
                  "NerfModel.use_3d_mask = True",
                  "NerfModel.use_mask_in_rgb = False",
                  "SpecularConfig.mask_ratio_schedule = {'type': 'constant', 'value': 1}",
                  "MaskMLP.depth = 8",
                  "MaskMLP.width = 128",
                  "MaskMLP.output_activation = @jax.nn.relu",
                  "NerfModel.use_mask_sharp_weights = True",
                  "NerfModel.use_x_in_rgb_condition = True",
                  "NerfModel.window_x_in_rgb_condition = True",
                  "NerfModel.norm_supervision_type = 'warped'"
                  ],
        'exp37': ["ExperimentConfig.image_scale = 1",
                  "NerfModel.use_predicted_mask = True",
                  "NerfModel.use_3d_mask = True",
                  "NerfModel.use_mask_in_rgb = False",
                  "NerfModel.use_mask_embed = False",
                  "SpecularConfig.mask_ratio_schedule = {'type': 'constant', 'value': 1}",
                  "MaskMLP.depth = 8",
                  "MaskMLP.width = 128",
                  "MaskMLP.output_activation = @jax.nn.relu",
                  "NerfModel.use_mask_sharp_weights = True",
                  "NerfModel.use_x_in_rgb_condition = True",
                  "NerfModel.window_x_in_rgb_condition = True",
                  "NerfModel.norm_supervision_type = 'warped'"
                  ],
        'exp38': ["ExperimentConfig.image_scale = 1",
                  "NerfModel.use_predicted_mask = True",
                  "NerfModel.use_3d_mask = True",
                  "NerfModel.use_mask_in_rgb = False",
                  "SpecularConfig.mask_ratio_schedule = {'type': 'constant', 'value': 1}",
                  "MaskMLP.depth = 8",
                  "MaskMLP.width = 128",
                  "MaskMLP.output_activation = @jax.nn.relu",
                  "NerfModel.use_mask_sharp_weights = False",
                  "NerfModel.use_x_in_rgb_condition = True",
                  "NerfModel.window_x_in_rgb_condition = True",
                  "NerfModel.norm_supervision_type = 'warped'"
                  ],
        'exp39': ["ExperimentConfig.image_scale = 1",
                  "NerfModel.use_predicted_mask = True",
                  "NerfModel.use_3d_mask = True",
                  "NerfModel.use_mask_in_rgb = False",
                  "SpecularConfig.mask_ratio_schedule = {'type': 'constant', 'value': 1}",
                  "MaskMLP.depth = 8",
                  "MaskMLP.width = 128",
                  "MaskMLP.output_activation = @jax.nn.relu",
                  "NerfModel.use_mask_sharp_weights = True",
                  "NerfModel.use_x_in_rgb_condition = True",
                  "NerfModel.window_x_in_rgb_condition = True",
                  "NerfModel.norm_supervision_type = 'warped'",
                  """SpecularConfig.sharp_mask_std_schedule = {
                      'type': 'piecewise',
                      'schedules': [
                          (30000, ('exponential', 1, 0.1, 30000)),
                          (220000, ('constant', 0.1))
                      ]
                  }"""
                  ],
        'exp40': ["ExperimentConfig.image_scale = 1",
                  "NerfModel.use_predicted_mask = True",
                  "NerfModel.use_3d_mask = True",
                  "NerfModel.use_mask_in_rgb = False",
                  "SpecularConfig.mask_ratio_schedule = {'type': 'constant', 'value': 1}",
                  "MaskMLP.depth = 8",
                  "MaskMLP.width = 128",
                  "MaskMLP.output_activation = @jax.nn.relu",
                  "NerfModel.use_mask_sharp_weights = True",
                  "NerfModel.use_x_in_rgb_condition = True",
                  "NerfModel.window_x_in_rgb_condition = True",
                  "NerfModel.norm_supervision_type = 'warped'",
                  """SpecularConfig.sharp_mask_std_schedule = {
                      'type': 'piecewise',
                      'schedules': [
                          (30000, ('exponential', 1, 0.1, 30000)),
                          (220000, ('constant', 0.1))
                      ]
                  }"""
                  ],
        'exp41': ["ExperimentConfig.image_scale = 1",
                  "NerfModel.use_predicted_mask = True",
                  "NerfModel.use_3d_mask = True",
                  "NerfModel.use_mask_in_rgb = False",
                  "SpecularConfig.mask_ratio_schedule = {'type': 'constant', 'value': 1}",
                  "MaskMLP.depth = 8",
                  "MaskMLP.width = 128",
                  "MaskMLP.output_activation = @jax.nn.relu",
                  "NerfModel.use_mask_sharp_weights = True",
                  "NerfModel.use_x_in_rgb_condition = True",
                  "NerfModel.window_x_in_rgb_condition = True",
                  "NerfModel.norm_supervision_type = 'warped'",
                  """SpecularConfig.sharp_mask_std_schedule = {
                      'type': 'piecewise',
                      'schedules': [
                          (30000, ('exponential', 1, 0.1, 30000)),
                          (220000, ('constant', 0.1))
                      ]
                  }""",
                  "TrainConfig.use_background_loss = True"
                  ],
        'exp42': ["ExperimentConfig.image_scale = 4",
                  "NerfModel.use_predicted_mask = True",
                  "NerfModel.use_3d_mask = True",
                  "NerfModel.use_mask_in_rgb = False",
                  "SpecularConfig.mask_ratio_schedule = {'type': 'constant', 'value': 1}",
                  "MaskMLP.depth = 8",
                  "MaskMLP.width = 128",
                  "MaskMLP.output_activation = @jax.nn.relu",
                  "NerfModel.use_mask_sharp_weights = True",
                  "NerfModel.use_x_in_rgb_condition = True",
                  "NerfModel.window_x_in_rgb_condition = True",
                  "NerfModel.norm_supervision_type = 'warped'",
                  """SpecularConfig.sharp_mask_std_schedule = {
                      'type': 'piecewise',
                      'schedules': [
                          (30000, ('exponential', 1, 0.1, 30000)),
                          (220000, ('constant', 0.1))
                      ]
                  }"""
                  ],
        'exp43': ["ExperimentConfig.image_scale = 2",
                  "NerfModel.use_predicted_mask = True",
                  "NerfModel.use_3d_mask = True",
                  "NerfModel.use_mask_in_rgb = False",
                  "SpecularConfig.mask_ratio_schedule = {'type': 'constant', 'value': 1}",
                  "MaskMLP.depth = 8",
                  "MaskMLP.width = 128",
                  "MaskMLP.output_activation = @jax.nn.relu",
                  "NerfModel.use_mask_sharp_weights = True",
                  "NerfModel.use_x_in_rgb_condition = True",
                  "NerfModel.window_x_in_rgb_condition = True",
                  "NerfModel.norm_supervision_type = 'warped'",
                  """SpecularConfig.sharp_mask_std_schedule = {
                      'type': 'piecewise',
                      'schedules': [
                          (30000, ('exponential', 1, 0.1, 30000)),
                          (220000, ('constant', 0.1))
                      ]
                  }"""
                  ],
        'exp44': ["ExperimentConfig.image_scale = 2",
                  "batch_size=512",
                  "max_steps=2500000",
                  "lr_decay_steps=2500000",
                  "init_lr = 0.001",
                  "final_lr = 0.0001",
                  "NerfModel.num_coarse_samples = 128",
                  "NerfModel.num_coarse_samples = 128",
                  "NerfModel.use_predicted_mask = True",
                  "NerfModel.use_3d_mask = True",
                  "NerfModel.use_mask_in_rgb = False",
                  "SpecularConfig.mask_ratio_schedule = {'type': 'constant', 'value': 1}",
                  "MaskMLP.depth = 8",
                  "MaskMLP.width = 128",
                  "MaskMLP.output_activation = @jax.nn.relu",
                  "NerfModel.use_mask_sharp_weights = True",
                  "NerfModel.use_x_in_rgb_condition = True",
                  "NerfModel.window_x_in_rgb_condition = True",
                  "NerfModel.norm_supervision_type = 'warped'",
                  """SpecularConfig.sharp_mask_std_schedule = {
                      'type': 'piecewise',
                      'schedules': [
                          (30000, ('exponential', 1, 0.1, 30000)),
                          (220000, ('constant', 0.1))
                      ]
                  }""",
                  ],
        'exp50': ["ExperimentConfig.image_scale = 1",
                  "NerfModel.use_predicted_mask = True",
                  "NerfModel.use_3d_mask = True",
                  "NerfModel.use_mask_in_rgb = False",
                  "SpecularConfig.mask_ratio_schedule = {'type': 'constant', 'value': 1}",
                  "MaskMLP.depth = 8",
                  "MaskMLP.width = 128",
                  "MaskMLP.output_activation = @jax.nn.relu",
                  "NerfModel.use_mask_sharp_weights = True",
                  "NerfModel.use_x_in_rgb_condition = True",
                  "NerfModel.window_x_in_rgb_condition = False",
                  "NerfModel.norm_supervision_type = 'warped'",
                  """SpecularConfig.sharp_mask_std_schedule = {
                      'type': 'piecewise',
                      'schedules': [
                          (30000, ('exponential', 1, 0.1, 30000)),
                          (220000, ('constant', 0.1))
                      ]
                  }"""
                  ],
        'exp51': ["ExperimentConfig.image_scale = 1",
                  "NerfModel.use_predicted_mask = True",
                  "NerfModel.use_3d_mask = True",
                  "NerfModel.use_mask_in_rgb = False",
                  "SpecularConfig.mask_ratio_schedule = {'type': 'constant', 'value': 1}",
                  "MaskMLP.depth = 8",
                  "MaskMLP.width = 128",
                  "MaskMLP.output_activation = @jax.nn.relu",
                  "NerfModel.use_mask_sharp_weights = True",
                  "NerfModel.use_x_in_rgb_condition = True",
                  "NerfModel.window_x_in_rgb_condition = True",
                  """SpecularConfig.x_for_rgb_alpha_schedule = {
                      'type': 'piecewise',
                      'schedules': [
                          (0, ('constant', 0)),
                          (50000, ('linear', 0, 4.0, 50000)),
                          (150000, ('constant', 4.0))
                      ]
                  }""",
                  "NerfModel.norm_supervision_type = 'warped'",
                  """SpecularConfig.sharp_mask_std_schedule = {
                      'type': 'piecewise',
                      'schedules': [
                          (30000, ('exponential', 1, 0.1, 30000)),
                          (220000, ('constant', 0.1))
                      ]
                  }"""
                  ],
        'exp52': ["ExperimentConfig.image_scale = 1",
                  "NerfModel.use_predicted_mask = True",
                  "NerfModel.use_3d_mask = True",
                  "NerfModel.use_mask_in_rgb = False",
                  "SpecularConfig.mask_ratio_schedule = {'type': 'constant', 'value': 1}",
                  "MaskMLP.depth = 8",
                  "MaskMLP.width = 128",
                  "MaskMLP.output_activation = @jax.nn.relu",
                  "NerfModel.use_mask_sharp_weights = True",
                  "NerfModel.use_x_in_rgb_condition = True",
                  "NerfModel.window_x_in_rgb_condition = True",
                  """SpecularConfig.x_for_rgb_alpha_schedule = {
                      'type': 'piecewise',
                      'schedules': [
                          (10000, ('constant', 0)),
                          (50000, ('linear', 0, 4.0, 50000)),
                          (150000, ('constant', 4.0))
                      ]
                  }""",
                  "NerfModel.norm_supervision_type = 'warped'",
                  """SpecularConfig.sharp_mask_std_schedule = {
                      'type': 'piecewise',
                      'schedules': [
                          (30000, ('exponential', 1, 0.1, 30000)),
                          (220000, ('constant', 0.1))
                      ]
                  }"""
                  ],
        'exp53': ["ExperimentConfig.image_scale = 1",
                  "NerfModel.use_predicted_mask = True",
                  "NerfModel.use_3d_mask = True",
                  "NerfModel.use_mask_in_rgb = False",
                  "SpecularConfig.mask_ratio_schedule = {'type': 'constant', 'value': 1}",
                  "MaskMLP.depth = 8",
                  "MaskMLP.width = 128",
                  "MaskMLP.output_activation = @jax.nn.relu",
                  "NerfModel.use_mask_sharp_weights = True",
                  "NerfModel.use_x_in_rgb_condition = True",
                  "NerfModel.window_x_in_rgb_condition = True",
                  """SpecularConfig.x_for_rgb_alpha_schedule = {
                      'type': 'piecewise',
                      'schedules': [
                          (100000, ('constant', 0)),
                          (50000, ('linear', 0, 4.0, 50000)),
                          (150000, ('constant', 4.0))
                      ]
                  }""",
                  "NerfModel.norm_supervision_type = 'warped'",
                  """SpecularConfig.sharp_mask_std_schedule = {
                      'type': 'piecewise',
                      'schedules': [
                          (30000, ('exponential', 1, 0.1, 30000)),
                          (220000, ('constant', 0.1))
                      ]
                  }"""
                  ],
        'exp54': ["ExperimentConfig.image_scale = 1",
                  "NerfModel.use_predicted_mask = True",
                  "NerfModel.use_3d_mask = True",
                  "NerfModel.use_mask_in_rgb = False",
                  "SpecularConfig.mask_ratio_schedule = {'type': 'constant', 'value': 1}",
                  "MaskMLP.depth = 8",
                  "MaskMLP.width = 128",
                  "MaskMLP.output_activation = @jax.nn.relu",
                  "NerfModel.use_mask_sharp_weights = True",
                  "NerfModel.use_x_in_rgb_condition = False",
                  "NerfModel.norm_supervision_type = 'warped'",
                  """SpecularConfig.sharp_mask_std_schedule = {
                      'type': 'piecewise',
                      'schedules': [
                          (30000, ('exponential', 1, 0.1, 30000)),
                          (220000, ('constant', 0.1))
                      ]
                  }"""
                  ],
        'exp60': ["ExperimentConfig.image_scale = 1",
                  "NerfModel.use_predicted_mask = True",
                  "NerfModel.use_3d_mask = True",
                  "NerfModel.use_mask_in_rgb = False",
                  "SpecularConfig.mask_ratio_schedule = {'type': 'constant', 'value': 1}",
                  "MaskMLP.depth = 8",
                  "MaskMLP.width = 128",
                  "MaskMLP.output_activation = @jax.nn.relu",
                  "NerfModel.use_mask_sharp_weights = True",
                  "NerfModel.use_x_in_rgb_condition = True",
                  "NerfModel.window_x_in_rgb_condition = True",
                  "NerfModel.norm_supervision_type = 'warped'",
                  """SpecularConfig.sharp_mask_std_schedule = {
                      'type': 'piecewise',
                      'schedules': [
                          (30000, ('exponential', 1, 0.01, 30000)),
                          (220000, ('constant', 0.1))
                      ]
                  }"""
                  ],
        'exp61': ["ExperimentConfig.image_scale = 1",
                  "NerfModel.use_predicted_mask = True",
                  "NerfModel.use_3d_mask = True",
                  "NerfModel.use_mask_in_rgb = False",
                  "SpecularConfig.mask_ratio_schedule = {'type': 'constant', 'value': 1}",
                  "MaskMLP.depth = 8",
                  "MaskMLP.width = 128",
                  "MaskMLP.output_activation = @jax.nn.relu",
                  "NerfModel.use_mask_sharp_weights = True",
                  "NerfModel.use_x_in_rgb_condition = True",
                  "NerfModel.window_x_in_rgb_condition = True",
                  "NerfModel.norm_supervision_type = 'warped'",
                  """SpecularConfig.sharp_mask_std_schedule = {
                      'type': 'piecewise',
                      'schedules': [
                          (30000, ('exponential', 1, 0.03, 30000)),
                          (220000, ('constant', 0.1))
                      ]
                  }"""
                  ],
        'exp62': ["ExperimentConfig.image_scale = 1",
                  "NerfModel.use_predicted_mask = True",
                  "NerfModel.use_3d_mask = True",
                  "NerfModel.use_mask_in_rgb = False",
                  "SpecularConfig.mask_ratio_schedule = {'type': 'constant', 'value': 1}",
                  "MaskMLP.depth = 8",
                  "MaskMLP.width = 128",
                  "MaskMLP.output_activation = @jax.nn.relu",
                  "NerfModel.use_mask_sharp_weights = True",
                  "NerfModel.use_x_in_rgb_condition = True",
                  "NerfModel.window_x_in_rgb_condition = True",
                  "NerfModel.norm_supervision_type = 'warped'",
                  """SpecularConfig.sharp_mask_std_schedule = {
                      'type': 'piecewise',
                      'schedules': [
                          (30000, ('exponential', 1, 0.3, 30000)),
                          (220000, ('constant', 0.1))
                      ]
                  }"""
                  ],
        'exp63': ["ExperimentConfig.image_scale = 1",
                  "NerfModel.use_predicted_mask = True",
                  "NerfModel.use_3d_mask = True",
                  "NerfModel.use_mask_in_rgb = False",
                  "SpecularConfig.mask_ratio_schedule = {'type': 'constant', 'value': 1}",
                  "MaskMLP.depth = 8",
                  "MaskMLP.width = 128",
                  "MaskMLP.output_activation = @jax.nn.relu",
                  "NerfModel.use_mask_sharp_weights = False",
                  "NerfModel.use_x_in_rgb_condition = True",
                  "NerfModel.window_x_in_rgb_condition = True",
                  "NerfModel.norm_supervision_type = 'warped'",
                  ],
        'exp70': ["ExperimentConfig.image_scale = 1",
                  "NerfModel.use_predicted_mask = True",
                  "NerfModel.use_3d_mask = True",
                  "NerfModel.use_mask_in_rgb = False",
                  "SpecularConfig.mask_ratio_schedule = {'type': 'constant', 'value': 1}",
                  "MaskMLP.depth = 8",
                  "MaskMLP.width = 128",
                  "MaskMLP.output_activation = @jax.nn.relu",
                  "NerfModel.use_mask_sharp_weights = True",
                  "NerfModel.use_x_in_rgb_condition = True",
                  "NerfModel.window_x_in_rgb_condition = True",
                  "NerfModel.norm_supervision_type = 'canonical_unwarped'",
                  """SpecularConfig.sharp_mask_std_schedule = {
                      'type': 'piecewise',
                      'schedules': [
                          (30000, ('exponential', 1, 0.1, 30000)),
                          (220000, ('constant', 0.1))
                      ]
                  }"""
                  ],
        'exp71': ["ExperimentConfig.image_scale = 1",
                  "NerfModel.use_predicted_mask = True",
                  "NerfModel.use_3d_mask = True",
                  "NerfModel.use_mask_in_rgb = False",
                  "SpecularConfig.mask_ratio_schedule = {'type': 'constant', 'value': 1}",
                  "MaskMLP.depth = 8",
                  "MaskMLP.width = 128",
                  "MaskMLP.output_activation = @jax.nn.relu",
                  "NerfModel.use_mask_sharp_weights = True",
                  "NerfModel.use_x_in_rgb_condition = True",
                  "NerfModel.window_x_in_rgb_condition = True",
                  "NerfModel.norm_supervision_type = 'direct'",
                  """SpecularConfig.sharp_mask_std_schedule = {
                      'type': 'piecewise',
                      'schedules': [
                          (30000, ('exponential', 1, 0.1, 30000)),
                          (220000, ('constant', 0.1))
                      ]
                  }"""
                  ],
    },
    'mso': {
        'exp01': ["ExperimentConfig.image_scale = 1",
                  "NerfModel.use_predicted_mask = True",
                  "NerfModel.use_3d_mask = True",
                  "NerfModel.use_mask_in_rgb = False",
                  "SpecularConfig.mask_ratio_schedule = {'type': 'constant', 'value': 1}",
                  "MaskMLP.depth = 8",
                  "MaskMLP.width = 128",
                  "MaskMLP.output_activation = @jax.nn.relu",
                  "NerfModel.use_mask_sharp_weights = True",
                  """SpecularConfig.sharp_mask_std_schedule = {
                      'type': 'piecewise',
                      'schedules': [
                          (30000, ('exponential', 1, 0.1, 30000)),
                          (220000, ('constant', 0.1))
                      ]
                  }"""
                  ],
        'exp05': ["ExperimentConfig.image_scale = 2",
                  "NerfModel.use_predicted_mask = True",
                  "NerfModel.use_3d_mask = True",
                  "NerfModel.use_mask_in_rgb = False",
                  "SpecularConfig.mask_ratio_schedule = {'type': 'constant', 'value': 1}",
                  "MaskMLP.depth = 8",
                  "MaskMLP.width = 128",
                  "MaskMLP.output_activation = @jax.nn.relu",
                  "NerfModel.use_mask_sharp_weights = True",
                  """SpecularConfig.sharp_mask_std_schedule = {
                      'type': 'piecewise',
                      'schedules': [
                          (30000, ('exponential', 1, 0.1, 30000)),
                          (220000, ('constant', 0.1))
                      ]
                  }"""
                  ],
        'exp44': [
            "ExperimentConfig.image_scale = 2",
            "batch_size=512",
            "max_steps=2500000",
            "lr_decay_steps=2500000",
            "init_lr = 0.001",
            "final_lr = 0.0001",
            "NerfModel.num_coarse_samples = 128",
            "NerfModel.num_coarse_samples = 128",
            "NerfModel.use_predicted_mask = True",
            "NerfModel.use_3d_mask = True",
            "NerfModel.use_mask_in_rgb = False",
            "SpecularConfig.mask_ratio_schedule = {'type': 'constant', 'value': 1}",
            "MaskMLP.depth = 8",
            "MaskMLP.width = 128",
            "MaskMLP.output_activation = @jax.nn.relu",
            "NerfModel.use_mask_sharp_weights = True",
            """SpecularConfig.sharp_mask_std_schedule = {
                'type': 'piecewise',
                'schedules': [
                    (30000, ('exponential', 1, 0.1, 30000)),
                    (220000, ('constant', 0.1))
                ]
            }"""
        ]
    },
    're_ms': {
        'exp47': [
            "ExperimentConfig.image_scale = 2",
            "batch_size=512",
            "max_steps=2500000",
            "lr_decay_steps=2500000",
            "init_lr = 0.0003",
            "final_lr = 0.00003",
            "NerfModel.num_coarse_samples = 128",
            "NerfModel.num_coarse_samples = 128",
            "TrainConfig.hyper_alpha_schedule = %LONG_DELAYED_HYPER_ALPHA_SCHED",
            "warp_alpha_steps = 800000",
            "NerfModel.use_predicted_mask = True",
            "NerfModel.use_3d_mask = True",
            "NerfModel.use_mask_in_rgb = False",
            "SpecularConfig.mask_ratio_schedule = {'type': 'constant', 'value': 1}",
            "MaskMLP.depth = 8",
            "MaskMLP.width = 128",
            "MaskMLP.output_activation = @jax.nn.relu",
            "NerfModel.use_mask_sharp_weights = True",
            "NerfModel.use_x_in_rgb_condition = True",
            "NerfModel.window_x_in_rgb_condition = True",
            "NerfModel.norm_supervision_type = 'warped'",
            """SpecularConfig.sharp_mask_std_schedule = {
                'type': 'piecewise',
                'schedules': [
                    (30000, ('exponential', 1, 0.1, 30000)),
                    (220000, ('constant', 0.1))
                ]
            }"""
        ],
    }
}
