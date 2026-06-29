from gymnasium import spaces
import numpy as np

from src.environments.fine_alignment.naive_image_env import NaiveImageEnv

"""
    This environment uses a somewhat complicated observation space
    consisting of 3 things:
        - the current image seen by the telescope
        - the image seen by the telescope before and after the last time the current panel was moved
        - the action that caused the difference in images before and after the last time the current panel was moved
"""
class StackedImageWithActionEnv(NaiveImageEnv):
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
                                )
        })

        # for each panel, we store the last action that was applied to it, and the
        # before and after images from applying that action to the telescope.
        self.panel_information = {}
        for panel in self.telescope.panels:
            # note that we index the panel information with the panel's id and not its index
            self.panel_information[panel] = {"action": np.zeros(2),
                                             "prev_image": self.telescope.image,
                                             "after_image": self.telescope.image}
    
    def apply_action(self, action):
        # get the index of the currently controlled panel
        current_panel_id = self.telescope.panels[self.current_panel]

        self.panel_information[current_panel_id]["prev_image"] = self.telescope.image
        super().apply_action(action)
        # reset the current panel id because it get updated during apply action
        current_panel_id = self.telescope.panels[(self.current_panel - 1) % len(self.telescope.panels)]
        self.panel_information[current_panel_id]["after_image"] = self.telescope.image

        self.panel_information[current_panel_id]["action"] = action
    
    def get_observation(self):
        current_panel_id = self.telescope.panels[self.current_panel]
        return {
            "current_image": np.expand_dims(self.telescope.image, axis=0).astype(np.float32) / 255.0,
            "previous_images": np.stack([self.panel_information[current_panel_id]["prev_image"], self.panel_information[current_panel_id]["after_image"]], axis=0).astype(np.float32) / 255.0,
            "previous_action": self.panel_information[current_panel_id]["action"].astype(np.float32)
        }
    
    def reset_telescope(self):
        super().reset_telescope()
        self.panel_information = {}
        for panel in self.telescope.panels:
            # note that we index the panel information with the panel's id and not its index
            self.panel_information[panel] = {"action": np.zeros(2),
                                             "prev_image": self.telescope.image,
                                             "after_image": self.telescope.image}