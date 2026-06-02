from gymnasium import spaces
import numpy as np

from configs.experiments.d06_02_26_twoPanelBasicCNNConfig import config as two_panel_cnn_config
from src.environments.basic_image_memory_env import BasicImageTwoFrameMemoryEnv
from src.evaluation.multiple_channel_obs_debug_vis import render_agent_diag

config = two_panel_cnn_config
config["environment"] = BasicImageTwoFrameMemoryEnv
config["train_config"]["tensorboard_log"] = "runs/06-02-2026_twoPanelTwoFrameMemoryWithDiff/"
config["model_save_path"] = "runs/06-02-2026_twoPanelTwoFrameMemoryWithDiff/agent"

img_size = config["env_params"]["telescope"]["img_size"]
config["env_params"]["observation_space"] = spaces.Box(
            low=0,
            high=255,
            shape=(3, 
                    img_size, 
                    img_size),  # CHW for SB3 CNN
            dtype=np.uint8,
        )
config["diagnostic_vis"] = render_agent_diag