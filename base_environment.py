import numpy as np
import gymnasium as gym
from gymnasium import spaces
from src.telescope import pSCT
from src.telescope.image_analyzer import image_analyzer

""" 
    This class represents how the agent interacts with the pSCT.
    This class should be used as the environment class for training 
    an agent to align the pSCT mirrors.

    This iteration of the environment holds a current panel, and 
    moves to the next panel every frame. The agent can only move the
    selected panel at any given time step. Basically only one panel
    moves every frame.
"""
class base_env(gym.Env):

    def __init__(self,
                 env_config=None):
        
        # bookkeeping
        self.step_count = 0
        self.max_steps = env_config["max_steps"]
        self.prev_cost = 0
        
        # panels
        self.P1s = [1111, 1112, 1113, 1114, 1211, 1212, 1213, 1214, 1311, 1312, 1313, 1314, 1411, 1412, 1413, 1414]
        self.n_panels = env_config["n_panels"]
        self.current_panel = 0 # the current panel that the agent is controlling. ranges from 0 to (n_panels - 1)
        
        # the pSCT telescope
        self.telescope = pSCT(env_config["telescope"])

        # image information
        self.memory_time = env_config["memory_time"] # n frames of memory in the feature extractor
        self.memory = None # see observation_space for dtype

        # Observation: single-channel image, unnormalized.
        self.observation_space = env_config["observation_space"]

        # Action: (rx, ry)
        self.action_space = env_config["action_space"]
    
    # =================================== API ===================================
    
    """
        -Run one timestep of the environment's dynamics using the agent actions.
        For each panel in the simulation, the agent chooses to update its position
        by providing a rotation rx, ry. 
        -Takes each rotation and updates the location of the panel's corresponding 
        image and renders it.
        -Calculates and returns the reward corresponding with the provided action
        
        params:
        action (numpy.ndarray with shape (1, 2)):        panel rotations. the nth panel is rotated rx, ry = action[n]

        returns:
        observation (numpy.ndarray with shape (img_size, img_size)):    the new environment state
        reward (Float):                                                 how beneficial the action was
        terminated (bool):                                              whether the agent reaches the terminal state
        truncated (bool):                                               whether the agent reaches a state that should cause the simulation to stop early
        info (dict):                                                    contains debugging information
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
        Resets the environment to an initial internal state, returning an initial observation and info.
        Places each panel's image to a new random location.

        params:
        seed (None):                Satisfies the API but we introduce our own PRNG (numpy random number generator) so keep as None
        options (optional dict):    optional debug information to include about how the environment is reset. RNG seed for example.

        returns:
        observation (numpy.ndarray with shape (n_panels, 2)):   the new environment state
        info (dict):                                            contains debugging information. Should be the same information as returned from step()
    """
    def reset(self, *, seed=None, options=None):
        # bookkeeping
        self.step_count = 0
        self.telescope.set_random_rotations()

        # set up the observation
        img = self.telescope.get_image(self.P1s[:self.n_panels])
        self.memory = np.zeros((self.memory_time, self.telescope.img_size, self.telescope.img_size), dtype=np.uint8)
        self.memory[:] = img

        # set up reward shaping
        detected_centroids = image_analyzer.get_centroid_locations(self.memory[0])
        self.prev_cost = self.cost_from_detected_centroids(detected_centroids)

        return self.memory, {}
    
    # ============================== Helper Functions ==============================

    def cost_from_detected_centroids(self, detected_fp_coords):
        d = detected_fp_coords - self.telescope.center[None, :]
        mean_r2 = float(np.mean(np.sqrt(np.sum(d**2, axis=1))))

        cost = mean_r2
        cost /= 200 # normalize the cost
        return cost

    def increment_memory(self, img):
        self.memory[1:] = self.memory[:-1] # shift all frames forward (ignoring first fram and overriding last frame)
        self.memory[0] = img