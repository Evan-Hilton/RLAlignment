import numpy as np

from src.environments.basic_image_env import BasicImageEnv
from src.telescope.pSCT import PSCT_P1
from src.telescope.image_analyzer import ImageAnalyzer

"""
    This class includes memory into the cnn as color channels.
    Basically, the current frame, previous frame, and a difference
    image (prev - current) are included in the observation
"""
class BasicImageTwoFrameMemoryEnv(BasicImageEnv):
    def __init__(self, env_config):
        self.env_config = env_config
        super().__init__(env_config)
        self.prev_frame = self.telescope.image
        self.observation = self.telescope.image
        self.difference = self.prev_frame - self.observation
        minmaxdiff = (self.difference.max() - self.difference.min()) if (self.difference.max() - self.difference.min()) != 0 else 1
        self.difference = ((self.difference - self.difference.min()) / minmaxdiff * 255).astype(np.uint8)
    
    def apply_action(self, action):
        super().apply_action(action)
        self.update_observation()

    def get_observation(self):
        return np.transpose(np.dstack((self.observation, self.prev_frame, self.difference)), (2, 0, 1)).astype(np.uint8)
    
    def update_observation(self):
        self.prev_frame = self.observation
        self.observation = self.telescope.image
        self.difference = self.prev_frame - self.observation
        self.difference = ((self.difference - self.difference.min()) / (self.difference.max() - self.difference.min()) * 255).astype(np.uint8)