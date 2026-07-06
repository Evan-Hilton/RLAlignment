from src.environments.fine_alignment.all_dict_obs_env_diff_img_diff_rew import *

"""
    Exact same environment as AllDictObsEnvDiffImgDiffRew except
    we encode the currently controlled panel ID as a one hot vector
    instead of a normalized single value between 0 and 1.
    Make sure to use everythingDictWithOneHotPanelIDs feature extractor
"""
class AllDictObsEnvDiffImgDiffRewOneHotID(AllDictObsEnvDiffImgDiffRew):
    def __init__(self, env_config):
        super().__init__(env_config)
        self.observation_space = spaces.Dict({
            "current_image": spaces.Box( # observation space
                                    low=0,
                                    high=1,
                                    shape=(1, # current frame
                                            self.telescope.img_size, 
                                            self.telescope.img_size),  # CHW for SB3 CNN
                                    dtype=np.float32
                                ),
            "previous_images": spaces.Box( # observation space
                                    low=0,
                                    high=1,
                                    shape=(1, # previous frame before and after 
                                            self.telescope.img_size, 
                                            self.telescope.img_size),  # CHW for SB3 CNN
                                    dtype=np.float32
                                ),
            "previous_action": spaces.Box(
                                    low=-1,
                                    high=1,
                                    shape=(2,),
                                    dtype=np.float32
                                ),
            "current_panel_id": spaces.Box(
                                    low=0,
                                    high=1,
                                    shape=(self.telescope.n_panels,), # zero or one, basically just one hot encoding for the panel id
                                    dtype=np.uint8
                                ),
            "detected_centroids": spaces.Box(
                                    low=-1,
                                    high=1,
                                    shape=(self.telescope.n_panels, 8,), # 8 observables per centroid, n_panels amount of centroids
                                    dtype=np.float32
                                )
        })
    
    def get_observation(self):
        observation = super().get_observation()
        observation["current_panel_id"] = np.zeros(self.telescope.n_panels, dtype=np.uint8)
        observation["current_panel_id"][self.current_panel] = 1
        return observation