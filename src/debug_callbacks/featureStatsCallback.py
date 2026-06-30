from stable_baselines3.common.callbacks import BaseCallback
import torch

class FeatureStatsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # get latest raw observation from env rollout
        obs = self.model._last_obs

        with torch.no_grad():
            obs_tensor, _ = self.model.policy.obs_to_tensor(obs)

            features = self.model.policy.features_extractor(obs_tensor)

            self.logger.record("features/mean", features.mean().item())
            self.logger.record("features/std", features.std().item())
            self.logger.record("features/max", features.max().item())
            self.logger.record("features/min", features.min().item())

        return True