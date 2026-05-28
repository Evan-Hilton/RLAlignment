import numpy as np
import gymnasium as gym
from gymnasium import spaces
from src.telescope import pSCT
from src.telescope.image_analyzer import image_analyzer

""" 
    Base class implementation of the environment which an RL
    agent would exist in to align a telescope. Includes general
    API in accordance with SB3 and is a little more specific on 
    its use of a telescope. However, this class does not provide
    enough specificity to actually train an agent, so training
    should only be done with inhereted classes.
"""
class BaseEnv(gym.Env):

    def __init__(self,
                 env_config=None):
        
        # bookkeeping
        self.step_count = 0
        self.max_steps = env_config["max_steps"]

        self.telescope = self.initialize_telescope(env_config["telescope"])

        self.observation_space = env_config["observation_space"]

        self.action_space = env_config["action_space"]
    
    # =================================== API ===================================
    
    """
        Run one time step simulation of the environment that the agent exists in.
        
        It should: apply the action supplied by the agent, make any necessary updates
        to the telescope, retrieve an observation, reward, and whether termination of
        the episode should happen and return this information. It should also keep
        track of whether the episode should time out.
    """
    def step(self, action):
        self.apply_action(action)

        self.update_telescope()

        observation = self.get_observation()

        reward = self.compute_reward()

        terminated = self.check_terminated()

        truncated = self.step_count >= self.max_steps
        self.step_count += 1

        return observation, reward, terminated, truncated, {}

    """
        Resets the environment to an initial random state. This includes all things
        needed to reset the simulation of the telescope, and also to return the 
        observation from being in the new random state.
    """
    def reset(self, *, seed=None, options=None):
        # bookkeeping
        self.step_count = 0
        self.telescope.reset()

        observation = self.get_observation()

        return observation, {}