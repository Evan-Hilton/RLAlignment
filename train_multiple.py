import torch.nn as nn
import warnings
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_checker import check_env

from configs.loaders.load_config import load_experiment_config

# ================ CHECK ENVIRONMENT ================

config = load_experiment_config("configs/experiments/d06_09_26_noImageOneCentroidObs.yaml")

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

def train_model(model_config):
    kwargs = {"env_config": model_config["env_params"]}
    env = make_vec_env(
        model_config["environment"],
        n_envs=model_config["n_training_envs"],
        vec_env_cls=model_config["vec_env_cls"],
        env_kwargs=kwargs
    )

    ppoConfig = model_config["train_config"]
    model = PPO(
        env = env,
        **ppoConfig
    )

    model.learn(total_timesteps=model_config["total_timesteps"])
    model.save(model_config["model_save_path"])
    env.close()

if __name__ == "__main__":
    for i in range(48):
        if i > 7 and i % 8 == 0:
            config["total_timesteps"] = 200_000 * i
            config["model_save_path"] = config["model_save_path"][:-1] + str(i)
            config["env_params"]["telescope"] = "configs/telescopes/basicMultiPanelP12Config" + str(i) + ".yaml"
            print("training " + str(i) + " panels model")
            print(config["total_timesteps"], config["model_save_path"], config["env_params"]["telescope"])
            train_model(config)