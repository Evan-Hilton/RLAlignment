from src.environments.fine_alignment.all_dict_obs_env import *

class AllDictObsEnvDiffImg(AllDictObsEnv):
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
                                    shape=(1,),
                                    dtype=np.float32
                                ),
            "detected_centroids": spaces.Box(
                                    low=-1,
                                    high=1,
                                    shape=(self.telescope.n_panels, 8,), # 8 observables per centroid, n_panels amount of centroids
                                    dtype=np.float32
                                )
        })
    
    def get_observation(self):
        obs = super().get_observation()
        obs["previous_images"] = obs["previous_images"][0] - obs["previous_images"][1]
        obs["previous_images"] -= np.min(obs["previous_images"])
        obs["previous_images"] /= (np.max(obs["previous_images"]) if np.max(obs["previous_images"]) > 0 else 1)
        obs["previous_images"] = obs["previous_images"][None, :].astype(np.float32)
        #print(np.array([self.current_panel / self.telescope.n_panels], dtype=np.float32)) # a value between 0 and 1)
        return obs