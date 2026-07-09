from gymnasium import spaces
import numpy as np

from src.environments.no_image_psctp12_env import NoImagePSCTP12Env

"""
    The idea behind this class is that it behaves exactly like NoImagePSCTP12Env,
    except we change the observation space for this environment. We recognize that
    at any given time step, the agent is about to control one panel. If we give information
    about where that panel used to be, where it is now, the action that brought it from
    before to now, that should be enough information to know how to move it.
"""
class NoImageOneCentroidObs(NoImagePSCTP12Env):
    def __init__(self, env_config):
        super().__init__(env_config)
        self.observation_space = spaces.Box( # action space
                                    low=-1,
                                    high=1,
                                    shape=(2 + 2 + 2,), # earlier panel location, action, newest panel location
                                    dtype=np.float32
                                )
        self.old_centroid_positions = self.telescope.true_centroids # the old centroid positions
        self.old_actions = np.zeros_like(self.telescope.true_centroids, dtype=np.float32) # the actions between the old and new centroid positions
    
    def apply_action(self, action):
        self.old_actions[self.current_panel] = action
        super().apply_action(action)
        
    def get_observation(self):
        # we take the current panel, because we want to tell the agent what it's about to control next
        old_centroid_position = self.normalize_centroid_error(self.old_centroid_positions[self.current_panel] - self.telescope.center[None, :])
        old_action = self.old_actions[self.current_panel]
        new_centroid_position = self.normalize_centroid_error(self.telescope.true_centroids[self.current_panel] - self.telescope.center[None, :])

        observation = np.concatenate((old_centroid_position.flatten(), new_centroid_position.flatten(), old_action))
        self.old_centroid_positions[self.current_panel] = new_centroid_position

        return observation.astype(np.float32)
    
    def reset(self, *, seed=None, options=None):
        observation = super().reset()
        self.old_centroid_positions = self.telescope.true_centroids
        self.old_actions = np.zeros_like(self.telescope.true_centroids, dtype=np.float32)
        return observation