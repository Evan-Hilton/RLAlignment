import torch.nn as nn
import warnings
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.env_checker import check_env
from Environment.environment import pSCT_environment


# Check that the model parameters are defined correctly in accordance with SB3
env = pSCT_environment()

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
        pSCT_environment,
        n_envs=8,
        vec_env_cls=SubprocVecEnv, # recommended in the documentation for speeding up training
    )
    # env = VecNormalize(env, norm_reward=True, norm_obs=False) # normalize the reward so that gradient updates aren't clipped too much
    # ultimately, env wraps VecNormalize, which wraps SupprocVecEnv, which wraps MirrorEnvImageDetect

    model = PPO(
        policy="CnnPolicy",
        env=env,
        device="cpu",
        policy_kwargs = dict(
            net_arch=dict(
                pi=[256, 256],     # policy MLP
                vf=[256, 256]            # value MLP
            ),
            activation_fn=nn.ReLU,
        ),
        learning_rate=1e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.001,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        normalize_advantage=True,
        tensorboard_log="./Models/One Panel/",
    )

    model.learn(total_timesteps=100_000)
    model.save("Models/One Panel/OnePanelModel")
    env.close()