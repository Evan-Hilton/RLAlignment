from gymnasium import spaces
import numpy as np

from src.environments.fine_alignment.naive_image_env import NaiveImageEnv

class StackedImageEnv(NaiveImageEnv):
    def __init__(self, env_config):
        super().__init__(env_config)
        self.observation_space = spaces.Box( # observation space
                                    low=0,
                                    high=255,
                                    shape=(2, # current frame and previous frame
                                            self.telescope.img_size, 
                                            self.telescope.img_size),  # CHW for SB3 CNN
                                    dtype=np.uint8,
                                )
        self.prev_frame = None
    
    def get_observation(self):
        current_telescope_view = self.telescope.image
        obs = np.stack((current_telescope_view, self.prev_frame), axis=0)
        self.prev_frame = current_telescope_view
        return obs.astype(np.uint8)
    
    def reset_telescope(self):
        obs = super().reset_telescope()
        self.prev_frame = self.telescope.image
        return obs