from gymnasium import spaces
import numpy as np

from src.environments.base_environment import BaseEnv
from src.telescope.pSCT_P12 import PSCT_P12
from src.telescope.image_analyzer import ImageAnalyzer

"""
    An environment of all panels in P1 of pSCT, where instead
    of evaluating the image seen by the telescope, we act
    based on the true locations of each centroid
"""
class NoImagePSCTP12Env(BaseEnv):
    def __init__(self, env_config):
        self.telescope = PSCT_P12(env_config["telescope"])
        super().__init__(env_config["max_steps"],
                         spaces.Box( # observation space
                             low=-1.0,
                             high=1.0,
                             shape=(self.telescope.n_panels * 2 + 1,), # +1 because we append which panel it's controlling
                             dtype=np.float32
                         ),
                         spaces.Box( # action space
                             low=-1,
                             high=1,
                             shape=(2,),
                             dtype=np.float32
                         ))
        self.current_panel = 0
        self.prev_cost = 0
        self.previous_centroids = self.telescope.true_centroids

    def apply_action(self, action):
        current_panel_id = self.telescope.panels[self.current_panel]
        self.telescope.rotate_panel(current_panel_id, [action[0], action[1]])

        if ImageAnalyzer.any_centroid_outside_image(self.telescope.center, self.telescope.init_scatter_pix, self.telescope.true_centroids):
            self.telescope.rotate_panel(current_panel_id, [-action[0], -action[1]])
        
        self.current_panel = (self.current_panel + 1) % len(self.telescope.panels)

    def get_observation(self):
        panel_index_indicator = self.current_panel * (1 / len(self.telescope.panels)) * 2 - 1
        return np.append(self.__normalize_centroid_error((self.telescope.true_centroids - self.telescope.center[None, :]).reshape(-1)), panel_index_indicator).astype(np.float32)

    def get_current_reward(self, observation):
        panel_idx = (self.current_panel - 1) % len(self.telescope.panels)

        old_error = np.linalg.norm(self.previous_centroids[panel_idx] - self.telescope.center)

        new_error = np.linalg.norm(self.telescope.true_centroids[panel_idx] - self.telescope.center)

        self.previous_centroids = self.telescope.true_centroids
        return float(old_error - new_error) - (new_error * 0.01)

    """
        Normalizes a distance given by 'distance' in fp coordinates from a point
        to the center of the telescope to be between 0 (point at center) and 1 (point at a corner of the image)
    """
    def __normalize_centroid_error(self, distance):
        # scale the centroid distance by the maximum distance away it can be (without truncating)
        x_max_fp, y_max_fp = self.telescope.uv_to_fp(0, 0) # a centroid at 0, 0 is at the top left of the screen (max dist it can be away from center)
        telescope_center_fp = self.telescope.center
        max_fp_distance = np.sqrt((x_max_fp - telescope_center_fp[0])**2 + (y_max_fp - telescope_center_fp[1])**2)
        return distance * (1 / max_fp_distance)
    
    def check_terminated(self, observation):
        success = ImageAnalyzer.all_centroids_at_center(self.telescope.center, self.telescope.true_centroids, success_radius=20)
        return success, 100 if success else 0

    def reset_telescope(self):
        self.telescope.reset()

        self.previous_centroids = self.telescope.true_centroids