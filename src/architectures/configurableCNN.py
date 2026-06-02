from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import torch
import torch.nn as nn

class ConfigurableCNN(BaseFeaturesExtractor):

    def __init__(
        self,
        observation_space,
        cnn_config,
    ):

        super().__init__(
            observation_space,
            features_dim=cnn_config["features_dim"]
        )

        in_channels = observation_space.shape[0]

        layers = []

        for layer_cfg in cnn_config["conv_layers"]:

            layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=layer_cfg["out_channels"],
                    kernel_size=layer_cfg["kernel_size"],
                    stride=layer_cfg["stride"],
                    padding=layer_cfg.get("padding", 0)
                )
            )

            layers.append(nn.ReLU())

            in_channels = layer_cfg["out_channels"]

        self.cnn = nn.Sequential(*layers)

        # determine flattened size automatically
        with torch.no_grad():

            sample = torch.as_tensor(
                observation_space.sample()[None]
            ).float()

            n_flatten = self.cnn(sample).flatten(1).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(
                n_flatten,
                cnn_config["features_dim"]
            ),
            nn.ReLU()
        )

    def forward(self, observations):

        x = self.cnn(observations)

        x = torch.flatten(x, start_dim=1)

        x = self.linear(x)

        return x