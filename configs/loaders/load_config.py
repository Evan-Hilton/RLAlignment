import yaml
from gymnasium import spaces
from stable_baselines3.common.vec_env import SubprocVecEnv
import torch.nn as nn

from src.environments.basic_image_env import *

ENVIRONMENTS = {
    "BasicImageEnv": BasicImageEnv,
}
VEC_ENV_CLS = {
    "SubprocVecEnv": SubprocVecEnv
}

"""
    Takes in a path to a yaml file, and loads it by
    creating all necessary python objects for the config
    and initializes a python dictionary to act as a config
    and returns it.
"""
def load_experiment_config(path):
    experiment_dict = yaml_import(path)
    telescope_config = load_telescope_config(experiment_dict["telescope_config"])
    config = {
        "environment": ENVIRONMENTS[experiment_dict["environment"]],
        "env_params": {

        },
        "n_training_envs": experiment_dict["n_training_envs"],
        "vec_env_cls": VEC_ENV_CLS[experiment_dict["vec_env_cls"]],
        "train_config": {
            "policy": experiment_dict["policy"],
            "learning_rate": experiment_dict["learning_rate"],
            "n_steps": experiment_dict["n_steps"],
            "batch_size": experiment_dict["batch_size"],
            "n_epochs": experiment_dict["n_epochs"],
            "gamma": experiment_dict["gamma"],
            "gae_lambda": experiment_dict["gae_lambda"],
            "clip_range": experiment_dict["clip_range"],
            "clip_range_vf": experiment_dict["clip_range_vf"],
            "normalize_advantage": experiment_dict["normalize_advantage"],
            "ent_coef": experiment_dict["ent_coef"],
            "vf_coef": experiment_dict["vf_coef"],
            "max_grad_norm": experiment_dict["max_grad_norm"],
            "use_sde": experiment_dict["use_sde"],
            "sde_sample_freq": experiment_dict["sde_sample_freq"],
            "rollout_buffer_class": experiment_dict["rollout_buffer_class"],
            "rollout_buffer_kwargs": experiment_dict["rollout_buffer_kwargs"],
            "target_kl": experiment_dict["target_kl"],
            "stats_window_size": experiment_dict["stats_window_size"],
            "tensorboard_log": experiment_dict["tensorboard_log"],
            "policy_kwargs": dict(
                net_arch=dict(
                    pi=[256, 256],     # policy MLP
                    vf=[256, 256]            # value MLP
                ),
                activation_fn=nn.ReLU,
            ),
            "verbose": experiment_dict["verbose"],
            "seed": experiment_dict["seed"],
            "device": experiment_dict["device"],
            "_init_setup_model": experiment_dict["_init_setup_model"]
        },
        "total_timesteps": experiment_dict["total_timesteps"],
        "model_save_path": experiment_dict["model_save_path"]
    }

    return config

def load_telescope_config(path):
    return yaml_import(path)

def yaml_import(path):
    with open(
        path,
        "r"
    ) as f:
        config = yaml.safe_load(f)
    return config