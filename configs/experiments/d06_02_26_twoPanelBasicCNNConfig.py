import yaml
from pathlib import Path
import torch.nn as nn

from configs.experiments.d06_01_26_onePanelBasicImageConfig import config as one_panel_config
from src.architectures.configurableCNN import ConfigurableCNN

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

cnn_config = load_yaml(
    "configs/architectures/basicCNN.yaml"
)

policy_kwargs = dict(

    features_extractor_class=ConfigurableCNN,

    features_extractor_kwargs=dict(
        cnn_config=cnn_config
    ),

    net_arch=dict(
        pi=[256],
        vf=[256]
    ),

    activation_fn=nn.ReLU,
)

config = one_panel_config
config["train_config"]["policy_kwargs"] = policy_kwargs
config["env_params"]["n_panels"] = 2