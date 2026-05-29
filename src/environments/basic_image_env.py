import numpy as np

from src.environments.base_environment import BaseEnv
from src.telescope.pSCT import pSCT
from src.telescope.image_analyzer import ImageAnalyzer

"""
    BasicImageEnv is a subclass of BaseEnv which implements
    all required API not specified in BaseEnv. This class is
    meant to be an implementation of the environment given the
    following specifications:
        - uses image based observations from phase 1 of alignment
        - uses basic standard reward of mean squared error from
            distance of centroids from center
        - rotates only one panel at a time (cycles through P1s)
"""
class BasicImageEnv(BaseEnv):
    def __init__(self, env_config):
        super().__init__(env_config=env_config)

        self.env_config = env_config

        self.current_panel = 0
        self.n_panels = env_config["n_panels"]
        self.P1s = [1111, 1112, 1113, 1114, 1211, 1212, 1213, 1214, 1311, 1312, 1313, 1314, 1411, 1412, 1413, 1414]

    def initialize_telescope(self, telescope_config):
        return pSCT(telescope_config)

    """
        Rotates the currently selected panel by some
        amount specified by 'action'. 'action' should
        have two values: (x, y) rotation normalized
        between -1 and 1.

        Here, we also increment which panel is being moved
        every frame.
    """
    def apply_action(self, action):
        self.telescope.rotate_panel(self.P1s[self.current_panel], action[0], action[1])

        # NOTE: uses true centroid locations
        if self.telescope.any_centroid_outside_image():
            self.telescope.rotate_panel(self.P1s[self.current_panel], -action[0], -action[1])
        
        self.current_panel = (self.current_panel + 1) % self.n_panels

    def update_telescope(self):
        pass

    """
        Gets the current image seen by the telescope. The
        returned value is a 2d numpy array with values
        between 0-255 and has shape (img_size, img_size),
        where img_size is specified by the telescope
    """
    def get_observation(self):
        return self.telescope.get_image(self.P1s[:self.n_panels])

    def get_current_reward(self):
        detected_centroids = ImageAnalyzer.get_centroid_locations(self.current_telescope_image)
        d = detected_centroids - self.telescope.center[None, :]
        mean_r2 = float(np.mean(np.sqrt(np.sum(d**2, axis=1))))
        return mean_r2

    def check_terminated(self):
        pass