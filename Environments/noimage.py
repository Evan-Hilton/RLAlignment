import numpy as np
import gymnasium as gym
from gymnasium import spaces
from PSCT.pSCT import pSCT

""" 
    This class represents how the agent interacts with the pSCT.
    This class should be used as the environment class for training 
    an agent to align the pSCT mirrors.

    This class uses a simplified version of the pSCT that does not
    have mirrors but insead interacts directly with the true position
    of the centroids. This is done mainly to prove that the method will
    work for the image version of the pSCT and to drastically increase
    training efficiency, since there is a large bottleneck when creating
    images.

    This iteration specifically deals with an observation space that is the
    true centroid location for each centroid and which panel it'll control, 
    and an action space that is just an (rx, ry). A list of active panels 
    is maintained and incremented over every 'memory_time' amount of steps, 
    and the action from the agent rotates whatever panel is currently selected.
"""
class pSCT_environment(gym.Env):

    def __init__(self,
                 n_panels = 10,
                 memory_time = 1, # how many steps backward in time the agent can see
                 panel_switch_time = 1
                 ):
        
        # bookkeeping
        self.step_count = 0
        # self.max_steps = 512 # the maximum amount of time the agent is allowed to move for
        # self.max_steps = 1024 # the maximum amount of time the agent is allowed to move for
        self.max_steps = n_panels * panel_switch_time # the agent only gets to move a panel at a time. once it's done moving that panel, it better be aligned.
        self.prev_cost = 0
        
        # panels
        self.P1s = [1111, 1112, 1113, 1114, 1211, 1212, 1213, 1214, 1311, 1312, 1313, 1314, 1411, 1412, 1413, 1414]
        self.n_panels = n_panels
        # discretize the action rotations into self.action_quant amount of discrete values
        # note that action_quant should be odd so that (action_quant - 1) / 2 maps to rotation = 0 (allow the agent to not move a panel)
        self.current_panel = 0
        self.current_panel_time = 0 # how many times has the current panel been moved since we switched panels
        self.panel_switch_time = panel_switch_time # every panel_switch_time amount of frames, switch which panel we're controlling

        # the pSCT telescope
        self.telescope = pSCT(n_panels=self.n_panels)

        # image information
        self.memory_time = memory_time # m frames of memory in the observation
        self.memory = None # np array with shape: (self.memory_time, 2 * self.n_panels)

        # Observation: each true centroid location given by (x1, y1, x2, y2, ..., xn, yn). this is stacked
        # for each time step in the past that the agent has access to (see memory_time). this vector is then
        # flattened to provide a single array to pass as the observation.
        #
        # -1 is the far left of the screen, 1 is the far right of the screen
        self.observation_space = spaces.Box(
            low=-1,
            high=1,
            shape=(self.n_panels * 2 * self.memory_time + self.memory_time,), # +self.memory_time because we give the agent which panel it's about to control at each time step
            dtype=np.float32,
        )

        # Action: (panel choice, rx, ry)
        # panel choice is a number between 0 and n_panels - 1. 
        # rx and ry are discretized to 25 unique values.
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(2,),
            dtype=np.float32,
        )
        #self.action_space = spaces.MultiDiscrete([self.action_quant, self.action_quant], dtype=np.uint8)
    
    # =================================== API ===================================
    
    """
        -Run one timestep of the environment's dynamics using the agent actions.
        For each panel in the simulation, the agent chooses to update its position
        by providing a rotation rx, ry and a panel to apply that rotation to.
        -Calculates and returns the reward corresponding with the provided action
    """
    def step(self, action):
        # normalize actions given by the network. map [0, action_quant] -> [-1, 1]
        # rotation_x = action[0] - ((self.action_quant - 1) / 2)          # action values surround zero
        # rotation_x = rotation_x * 1.0 / ((self.action_quant - 1) / 2)   # action scaled between -1 and 1
        # rotation_y = action[1] - ((self.action_quant - 1) / 2)          # action values surround zero
        # rotation_y = rotation_y * 1.0 / ((self.action_quant - 1) / 2)   # action scaled between -1 and 1
        rotation_x = action[0]
        rotation_y = action[1]

        # rotate the panel
        self.telescope.rotate_panel(self.P1s[self.current_panel], rotation_x, rotation_y)
        if self.telescope.any_centroid_outside_image(): # if the action causes a panel to go outside the image, don't take the action
            self.telescope.rotate_panel(self.P1s[self.current_panel], -rotation_x, -rotation_y)
        
        # update panel control
        if self.current_panel_time >= self.panel_switch_time - 1:
            self.current_panel = (self.current_panel + 1) % self.n_panels
        self.current_panel_time = (self.current_panel_time + 1) % self.panel_switch_time

        # update memory - give the new observation to the memory. see self.observation_space to see why we do this
        single_step_obs = self.telescope.get_normalized_centroid_fp_coords_on_screen().reshape(-1)
        single_step_obs = np.append(single_step_obs, self.current_panel) # make sure panel is updated BEFORE this line call. we want the environment to tell the agent which panel it's ABOUT to control
        self.increment_memory(single_step_obs)

        # get the cost from this state
        cost = self.cost_from_detected_centroids(self.telescope.true_centroids) * 1.2 # 0 good, 1 bad
        
        # normal shaping
        # reward = -cost

        # improvement shaping
        improve = self.prev_cost - cost
        reward = improve * 10.0 * self.n_panels
        self.prev_cost = cost

        terminated = False
        if self.telescope.all_centroids_at_center(success_radius=15): # success
            reward += 10
            terminated = True
        
        # bookkeeping
        truncated = self.step_count >= self.max_steps - 1
        self.step_count += 1
        info = {}

        return self.memory.flatten(order='F'), reward, terminated, truncated, info

    """
        Resets the environment to an initial internal state, returning an initial observation and info.
        Places each panel's image to a new random location.
    """
    def reset(self, *, seed=None, options=None):
        # bookkeeping
        self.step_count = 0
        self.telescope.set_random_rotations()

        # set up the observation
        self.memory = np.zeros((self.memory_time, 2*self.n_panels + 1), dtype=np.float32) # +1 because... see self.obs_space
        single_step_obs = self.telescope.get_normalized_centroid_fp_coords_on_screen().reshape(-1) # get flattened normalized true centroid locations
        single_step_obs = np.append(single_step_obs, self.current_panel) # make sure panel is updated BEFORE this line call. we want the environment to tell the agent which panel it's ABOUT to control
        self.memory[:] = single_step_obs

        # set up reward shaping
        self.prev_cost = self.cost_from_detected_centroids(self.telescope.true_centroids)

        self.current_panel = 0

        return self.memory.flatten(order='F'), {}
    
    # ============================== Helper Functions ==============================

    """
        Computes how bad it is for the given centroid locations to be where they are.
        Returns a normalized number that is 0 (really good) or 1 (really bad)
    """
    def cost_from_detected_centroids(self, detected_fp_coords):
        d = detected_fp_coords - self.telescope.center[None, :]
        mean_r2 = float(np.mean(np.sum(d**2, axis=1)))
        mean_r2 = self.normalize_centroid_error(mean_r2)

        cost = mean_r2 / (8 * 50)
        #print(cost)
        return cost

    """
        A helper method to update the current memory buffer given a new input.
        Cycles the old memory buffer forward one space to leave room for the new observation.
    """
    def increment_memory(self, img):
        self.memory[1:] = self.memory[:-1] # shift all frames forward (ignoring first fram and overriding last frame)
        self.memory[0] = img

    """
        Normalizes a distance given by 'distance' in fp coordinates from a point
        to the center of the telescope to be between 0 (point at center) and 1 (point at a corner of the image)
    """
    def normalize_centroid_error(self, distance):
        # scale the centroid distance by the maximum distance away it can be (without truncating)
        x_max_fp, y_max_fp = self.telescope._uv_to_fp(0, 0) # a centroid at 0, 0 is at the top left of the screen (max dist it can be away from center)
        telescope_center_fp = self.telescope.center
        max_fp_distance = np.sqrt((x_max_fp - telescope_center_fp[0])**2 + (y_max_fp - telescope_center_fp[1])**2)
        return distance / max_fp_distance