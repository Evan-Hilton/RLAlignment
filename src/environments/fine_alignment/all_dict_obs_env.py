from src.telescope.image_analyzer import ImageAnalyzer
from src.environments.fine_alignment.stacked_image_with_action_env import * 

"""
    This environment acts exactly like StackedImageWithActionEnv,
    but with the added parts to the observation space: 
        - id of panel agent is about to control
        - 8 observables of any detected centroids
"""
class AllDictObsEnv(StackedImageWithActionEnv):
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
                                    shape=(2, # previous frame before and after 
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

        self.success_radius = 10
    
    def get_observation(self):
        # detect all centroids and put them in a dataframe
        det_cet = ImageAnalyzer._sep_detection(self.telescope.image, dict())
        columns = [
                "X_IMAGE",
                "Y_IMAGE",
                "FLUX_ISO",
                "FLUX_MAX",
                "BACKGROUND",
                "A_IMAGE",
                "B_IMAGE",
                "THETA_IMAGE",
                ]
        detected = det_cet[columns].to_numpy()
        obs = np.zeros((self.telescope.n_panels, 8), dtype=np.float32)
        n_detected = len(detected)
        obs[:n_detected] = detected[:n_detected]

        # normalization
        obs[:, 0] /= self.telescope.img_size    # X_IMAGE
        obs[:, 1] /= self.telescope.img_size    # Y_IMAGE

        obs[:, 2] /= 5e+04                      # FLUX_ISO

        obs[:, 3] /= 255                        # FLUX_MAX
        obs[:, 4] /= 255                        # BACKGROUND

        obs[:, 5] /= 10                          # A_IMG
        obs[:, 6] /= 10                          # B_IMG

        obs[:, 7] /= 90                         # THETA_IMG

        observation = super().get_observation()
        observation["current_panel_id"] = np.array([self.current_panel / self.telescope.n_panels], dtype=np.float32) # a value between 0 and 1
        observation["detected_centroids"] = obs

        return observation
    
    def reset_telescope(self):
        return super().reset_telescope()