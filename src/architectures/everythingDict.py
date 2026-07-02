import torch
import torch.nn as nn
import yaml
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from src.architectures.configurableCNN import ConfigurableCNN

"""
    This feature extractor processes a lot of things:
        - cnn of currently seen telescope image
        - cnn of previous and current telescope image from before/after the last time the current panel was moved
        - passes previous action along
        - centroid encoder: basically a small mlp that takes every single centroid and passes them through the same network, then averages all the outputs into one vector
    All these features are concactenated to the end of a feature vector
"""
class EverythingDict(BaseFeaturesExtractor):

    def __init__(self, observation_space, config):

        dictConfig = self.load_yaml(config)

        current_cnn_features = self.load_yaml(dictConfig["current_image_cnn"])["features_dim"]
        previous_cnn_features = self.load_yaml(dictConfig["previous_images_cnn"])["features_dim"]
        self.centroid_features = dictConfig["centroid_features"]
        self.centroid_encoder_network_values = dictConfig["centroid_network"]
        features_dim=(
                current_cnn_features
                + previous_cnn_features
                + observation_space["previous_action"].shape[0]
                + observation_space["current_panel_id"].shape[0]
                + 2 * self.centroid_features
            )

        super().__init__(
            observation_space,
            features_dim=features_dim
        )

        self.current_cnn = ConfigurableCNN(
            observation_space["current_image"],
            dictConfig["current_image_cnn"]
        )

        self.previous_cnn = ConfigurableCNN(
            observation_space["previous_images"],
            dictConfig["previous_images_cnn"]
        )

        layers = []
        last_dim = observation_space["detected_centroids"].shape[-1]
        for hidden_dim in self.centroid_encoder_network_values:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, self.centroid_features))
        self.centroid_encoder = nn.Sequential(*layers)

        self.current_norm = nn.LayerNorm(current_cnn_features)
        self.previous_norm = nn.LayerNorm(previous_cnn_features)
        self.centroid_norm = nn.LayerNorm(self.centroid_features * 2)
        self.action_norm = nn.LayerNorm(observation_space["previous_action"].shape[0])
        self.id_norm = nn.LayerNorm(observation_space["current_panel_id"].shape[0])
        self.final_norm = nn.LayerNorm(features_dim)
    
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

        centroids = observations["detected_centroids"].float()
        B, N, F = centroids.shape

        x = centroids.view(B * N, F)
        encoded = self.centroid_encoder(x)
        encoded = encoded.view(B, N, -1)

        mean_features = encoded.mean(dim=1)
        max_features = encoded.max(dim=1).values

        centroid_features = torch.cat(
            (mean_features, max_features),
            dim=1
        )

        action = observations["previous_action"]
        id = observations["current_panel_id"]

        current_features = self.current_norm(current_features)
        previous_features = self.previous_norm(previous_features)
        centroid_features = self.centroid_norm(centroid_features)
        action = self.action_norm(action)
        id = self.id_norm(id)

        features = torch.cat(
            (
                current_features,
                previous_features,
                action,
                id,
                centroid_features,
            ),
            dim=1
        )

        features = self.final_norm(features)
        
        return features