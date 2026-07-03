import torch.nn as nn
import warnings
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_checker import check_env

from configs.loaders.load_config import load_experiment_config
from src.debug_callbacks.featureStatsCallback import *

# ================ CHECK ENVIRONMENT ================

config = load_experiment_config("configs/experiments/d07_03_26_fineAlignmentAllDictDiffImgDiffRewLargerActions.yaml")

env = config["environment"](config["env_params"])

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message=".*observation.*dtype.*np.uint8.*"
    )
    warnings.filterwarnings(
        "ignore",
        message=".*observation space.*not in \\[0, 255\\].*"
    )
    check_env(env, warn=True)

# ================ TRAIN MODEL ================

if __name__ == "__main__":
    kwargs = {"env_config": config["env_params"]}
    env = make_vec_env(
        config["environment"],
        n_envs=config["n_training_envs"],
        vec_env_cls=config["vec_env_cls"],
        env_kwargs=kwargs
    )

    ppoConfig = config["train_config"]
    model = PPO(
        env = env,
        **ppoConfig
    )

    model.learn(total_timesteps=config["total_timesteps"], callback=FeatureStatsCallback())
    model.save(config["model_save_path"])
    env.close()