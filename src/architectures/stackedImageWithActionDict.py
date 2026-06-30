import torch
import yaml
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from src.architectures.configurableCNN import ConfigurableCNN

class StackedImageWithActionDict(BaseFeaturesExtractor):

    def __init__(self, observation_space, config):

        dictConfig = self.load_yaml(config)

        current_cnn_features = self.load_yaml(dictConfig["current_image_cnn"])["features_dim"]
        previous_cnn_features = self.load_yaml(dictConfig["previous_images_cnn"])["features_dim"]

        super().__init__(
            observation_space,
            features_dim=current_cnn_features + previous_cnn_features + observation_space["previous_action"].shape[0]
        )

        self.current_cnn = ConfigurableCNN(
            observation_space["current_image"],
            dictConfig["current_image_cnn"]
        )

        self.previous_cnn = ConfigurableCNN(
            observation_space["previous_images"],
            dictConfig["previous_images_cnn"]
        )
    
    def load_yaml(self, path):
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def forward(self, observations):

        current_features = self.current_cnn(
            observations["current_image"]
        )

        previous_features = self.previous_cnn(
            observations["previous_images"]
        )

        action = observations["previous_action"]

        features = torch.cat(
            (
                current_features,
                previous_features,
                action
            ),
            dim=1
        )

        return features