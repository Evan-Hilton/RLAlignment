import numpy as np
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.vec_env import SubprocVecEnv

from configs.telescope.basicTelescopeConfig import telescopeConfig
from src.environments.basic_image_env import BasicImageEnv

config = {
    "env_params": {
        "max_steps": 512,
        "n_panels": 1,
        "memory_time": 1,
        "telescope": telescopeConfig,

        "observation_space": spaces.Box(
            low=0,
            high=255,
            shape=(1, # same as memory time
                    telescopeConfig["img_size"], 
                    telescopeConfig["img_size"]),  # CHW for SB3 CNN
            dtype=np.uint8,
        ),

        "action_space": spaces.Box(
            low=-1,
            high=1,
            shape=(2,),
            dtype=np.float32,
        ),
    },

    "environment": BasicImageEnv,

    "train_config": {
        "policy": "CnnPolicy",
        "learning_rate": 1e-4,
        "n_steps": 512,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.1,
        "clip_range_vf": None,
        "normalize_advantage": True,
        "ent_coef": 0.001,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "use_sde": False,
        "sde_sample_freq": -1,
        "rollout_buffer_class": None,
        "rollout_buffer_kwargs": None,
        "target_kl": None,
        "stats_window_size": 100,
        "tensorboard_log": "runs/06-01-2026_onePanelImageCodeCheck/",
        "policy_kwargs": dict(
            net_arch=dict(
                pi=[256, 256],     # policy MLP
                vf=[256, 256]            # value MLP
            ),
            activation_fn=nn.ReLU,
        ),
        "verbose": 1,
        "seed": None,
        "device": "cpu",
        "_init_setup_model": True
    },
    "vec_env_cls": SubprocVecEnv,
    "n_training_envs": 8,

    "total_timesteps": 100_000,
    "model_save_path": "runs/06-01-2026_onePanelImageCodeCheck/agent"
}