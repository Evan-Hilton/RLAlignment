import numpy as np
import gymnasium as gym
from gymnasium import spaces
from PSCT.pSCT import pSCT
from PSCT.image_analyzer import image_analyzer

""" 
    This class represents how the agent interacts with the pSCT.
    This class should be used as the environment class for training 
    an agent to align the pSCT mirrors.
"""
class pSCT_environment(gym.Env):

    def __init__(self,
                 n_panels = 2,
                 memory_time = 1):
        
        # bookkeeping
        self.step_count = 0
        self.max_steps = 512 # the maximum amount of time the agent is allowed to move for
        self.prev_cost = 0
        
        # panels
        self.P1s = [1111, 1112, 1113, 1114, 1211, 1212, 1213, 1214, 1311, 1312, 1313, 1314, 1411, 1412, 1413, 1414]
        self.n_panels = n_panels
        # discretize the action rotations into self.action_quant amount of discrete values
        # note that action_quant should be odd so that (action_quant - 1) / 2 maps to rotation = 0 (allow the agent to not move a panel)
        self.action_quant: int = 25 # if this is 25, then the agent can choose between 25 values to move the panels by. 0 and 25 represent maximum motion

        # the pSCT telescope
        self.telescope = pSCT(n_panels=self.n_panels)

        # image information
        self.memory_time = memory_time # n frames of memory in the cnn
        self.memory = None # see observation_space for dtype

        # Observation: single-channel image, unnormalized.
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.memory_time, self.telescope.img_size, self.telescope.img_size),  # CHW for SB3 CNN
            dtype=np.uint8,
        )

        # Action: (panel choice, rx, ry)
        # panel choice is a number between 0 and n_panels - 1. 
        # rx and ry are discretized to 25 unique values.
        self.action_space = spaces.MultiDiscrete([self.n_panels, self.action_quant, self.action_quant], dtype=np.uint8)
    
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
        # normalize actions given by the network. map [0, action_quant] -> [-1, 1]
        rotation_x = action[1] - ((self.action_quant - 1) / 2)          # action values surround zero
        rotation_x = rotation_x * 1.0 / ((self.action_quant - 1) / 2)   # action scaled between -1 and 1
        rotation_y = action[2] - ((self.action_quant - 1) / 2)          # action values surround zero
        rotation_y = rotation_y * 1.0 / ((self.action_quant - 1) / 2)   # action scaled between -1 and 1

        # rotate the panel
        self.telescope.rotate_panel(self.P1s[action[0]], rotation_x, rotation_y)

        # update memory - give the new observation to the memory
        self.increment_memory(self.telescope.get_image(self.P1s[:self.n_panels]))

        # calculate reward and reward shaping
        detected_centroids = image_analyzer.get_centroid_locations(self.memory[0])
        cost = self.cost_from_detected_centroids(detected_centroids)
        reward = -cost
        improve = self.prev_cost - cost
        reward += 0.5 * improve

        terminated = False
        if self.telescope.all_centroids_at_center():
            reward += 10
            terminated = True
        if self.telescope.any_centroid_outside_image():
            reward -= 35 # truncation penalty should be 5x-20x worse than average reward (currently at ~-0.4)
            terminated = True
        reward -= 0.1 # time penalty. incentivices fast solutions
        self.prev_cost = cost

        # bookkeeping
        truncated = self.step_count >= self.max_steps
        self.step_count += 1
        info = {"detected": detected_centroids}

        return self.memory, reward, terminated, truncated, info

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
        return cost

    def increment_memory(self, img):
        self.memory[1:] = self.memory[:-1] # shift all frames forward (ignoring first fram and overriding last frame)
        self.memory[0] = img