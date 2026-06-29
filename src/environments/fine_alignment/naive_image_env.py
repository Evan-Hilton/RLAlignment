from gymnasium import spaces
import numpy as np

from src.environments.base_environment import BaseEnv
from src.telescope.pSCT_P12 import PSCT_P12
from src.telescope.image_analyzer import ImageAnalyzer

class NaiveImageEnv(BaseEnv):
    def __init__(self, env_config):
        self.telescope = PSCT_P12(env_config["telescope"])
        self.current_panel = 0
        
        super().__init__(env_config["max_steps"], 
                            spaces.Box( # observation space
                                low=0,
                                high=255,
                                shape=(1, # same as memory time
                                        self.telescope.img_size, 
                                        self.telescope.img_size),  # CHW for SB3 CNN
                                dtype=np.uint8,
                            ),
                            spaces.Box(
                                low=-1,
                                high=1,
                                shape=(2,),
                                dtype=np.float32,
                            )
        )
    
    def apply_action(self, action):
        current_panel_id = self.telescope.panels[self.current_panel]
        self.telescope.rotate_panel(current_panel_id, [action[0], action[1]])

        if ImageAnalyzer.any_centroid_outside_image(self.telescope.center, self.telescope.init_scatter_pix, self.telescope.true_centroids):
            self.telescope.rotate_panel(current_panel_id, [-action[0], -action[1]])
        
        self.current_panel = (self.current_panel + 1) % len(self.telescope.panels)

        self.telescope.update()

    def get_observation(self):
        return self.telescope.image[None, :].astype(np.uint8)

    def get_current_reward(self, observation):
        reward = 0

        # calculate currently detected centroids
        detected_centroids = ImageAnalyzer._sep_detection(self.telescope.image, dict())

        # for each detected centroid, calculate the error
        for ID in detected_centroids['ID'].tolist():
            # add distance from center as an error
            x, y = detected_centroids['X_IMAGE'][ID], detected_centroids['Y_IMAGE'][ID]
            cx, cy = self.telescope.fp_to_uv(self.telescope.center[0], self.telescope.center[1])
            dist_cost = -np.linalg.norm(np.array([x - cx, y - cy]))
            reward -= dist_cost ** 2

            # add width/height of detected centroids as an error
            a, b = detected_centroids['A_IMAGE'][ID], detected_centroids['B_IMAGE'][ID]
            reward += -max(a, b) * 2 # interested in diameter
        
        return reward
    
    def check_terminated(self, observation):
        success = ImageAnalyzer.all_centroids_at_center(self.telescope.center, self.telescope.true_centroids, success_radius=5)
        return success, 100 if success else 0

    def reset_telescope(self):
        self.telescope.reset()