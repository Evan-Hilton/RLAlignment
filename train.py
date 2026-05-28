import torch.nn as nn
import warnings
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_checker import check_env
from src.environments.base_environment import base_env
from configs.experiments.d05_27_26_onePanelBasicImageConfig import config

# Check that the model parameters are defined correctly in accordance with SB3
env = base_env(config["env"])

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

# Train
if __name__ == "__main__":
    env = make_vec_env(
        base_env,
        n_envs=config["n_envs"],
        vec_env_cls=config["vec_env_cls"],
        #vec_env_cls=SubprocVecEnv, # recommended in the documentation for speeding up training
        env_kwargs=config["env"] # might be wrong
    )
    # env = VecNormalize(env, norm_reward=True, norm_obs=False) # normalize the reward so that gradient updates aren't clipped too much
    # ultimately, env wraps VecNormalize, which wraps SupprocVecEnv, which wraps MirrorEnvImageDetect

    ppoConfig = config["train_config"]
    model = PPO(
        env = env,
        **ppoConfig
    )

    model.learn(total_timesteps=ppoConfig["total_timesteps"])
    model.save(ppoConfig["model_save_path"])
    env.close()