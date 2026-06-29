import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from src.architectures.configurableCNN import ConfigurableCNN

class StackedImageWithActionDict(BaseFeaturesExtractor):

    def __init__(self, observation_space, config):

        current_dim = config["current_image_cnn"]["features_dim"]
        previous_dim = config["previous_images_cnn"]["features_dim"]
        action_dim = observation_space["previous_action"].shape[0]

        super().__init__(
            observation_space,
            features_dim=current_dim + previous_dim + action_dim
        )

        self.current_cnn = ConfigurableCNN(
            observation_space["current_image"],
            config["current_image_cnn"]
        )

        self.previous_cnn = ConfigurableCNN(
            observation_space["previous_images"],
            config["previous_images_cnn"]
        )

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