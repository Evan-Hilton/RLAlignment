from stable_baselines3 import PPO

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

import matplotlib.pyplot as plt
from stable_baselines3.common.logger import Figure

from stable_baselines3.common.utils import explained_variance

class LoggingPPO(PPO):
    def __init__(self,
                 heatmap_frequency, 
                 policy, 
                 env, 
                 learning_rate = 0.0003, 
                 n_steps = 2048, 
                 batch_size = 64, 
                 n_epochs = 10, 
                 gamma = 0.99, 
                 gae_lambda = 0.95, 
                 clip_range = 0.2, 
                 clip_range_vf = None, 
                 normalize_advantage = True, 
                 ent_coef = 0, 
                 vf_coef = 0.5, 
                 max_grad_norm = 0.5, 
                 use_sde = False, 
                 sde_sample_freq = -1, 
                 rollout_buffer_class = None, 
                 rollout_buffer_kwargs = None, 
                 target_kl = None, 
                 stats_window_size = 100, 
                 tensorboard_log = None, 
                 policy_kwargs = None, 
                 verbose = 0, 
                 seed = None, 
                 device = "auto", 
                 _init_setup_model = True
            ):
        super().__init__(policy, env, learning_rate, n_steps, batch_size, n_epochs, gamma, gae_lambda, clip_range, clip_range_vf, normalize_advantage, ent_coef, vf_coef, max_grad_norm, use_sde, sde_sample_freq, rollout_buffer_class, rollout_buffer_kwargs, target_kl, stats_window_size, tensorboard_log, policy_kwargs, verbose, seed, device, _init_setup_model)

        self.heatmap_frequency = heatmap_frequency

        # =========== ChatGPT log update =============
        # Store history:
        # shape = (number of rollouts, n_epochs)
        self.policy_loss_history = []
        self.value_loss_history = []
        self.kl_history = []
        self.clip_fraction_history = []
        self.entropy_history = []
        # ========================

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        # =========== ChatGPT log update =============
        rollout_policy_losses = []
        rollout_value_losses = []
        rollout_kl = []
        rollout_clip_fraction = []
        rollout_entropy = []
        # ========================

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []

            # =========== ChatGPT log update =============
            # Per-epoch statistics (reset every optimization epoch)
            epoch_entropy_losses = []
            epoch_pg_losses = []
            epoch_value_losses = []
            epoch_clip_fractions = []
            epoch_approx_kl_divs = []
            epoch_grad_norms = []
            # ========================

            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                epoch_pg_losses.append(policy_loss.item())# =========== ChatGPT log update =============
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)
                epoch_clip_fractions.append(clip_fraction)# =========== ChatGPT log update =============

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())
                epoch_value_losses.append(value_loss.item())# =========== ChatGPT log update =============

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())
                epoch_entropy_losses.append(entropy_loss.item())# =========== ChatGPT log update =============

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)
                    epoch_approx_kl_divs.append(approx_kl_div)# =========== ChatGPT log update =============

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

                # =========== ChatGPT log update =============
                grad_sq = 0.0

                for p in self.policy.parameters():
                    if p.grad is not None:
                        grad_sq += p.grad.detach().pow(2).sum().item()

                grad_norm = np.sqrt(grad_sq)

                epoch_grad_norms.append(grad_norm)
                # ========================
            
            # =========== ChatGPT log update =============
            # Mean over minibatches in this optimization epoch
            policy_epoch_loss = np.mean(epoch_pg_losses)
            value_epoch_loss = np.mean(epoch_value_losses)
            kl_epoch = np.mean(epoch_approx_kl_divs)
            clip_epoch = np.mean(epoch_clip_fractions)
            entropy_epoch = np.mean(epoch_entropy_losses)

            rollout_policy_losses.append(policy_epoch_loss)
            rollout_value_losses.append(value_epoch_loss)
            rollout_kl.append(kl_epoch)
            rollout_clip_fraction.append(clip_epoch)
            rollout_entropy.append(entropy_epoch)
            # ==============================

            self._n_updates += 1
            if not continue_training:
                break
        
        # =========== ChatGPT log update =============
        self.policy_loss_history.append(rollout_policy_losses)
        self.value_loss_history.append(rollout_value_losses)
        self.kl_history.append(rollout_kl)
        self.clip_fraction_history.append(rollout_clip_fraction)
        self.entropy_history.append(rollout_entropy)

        if self._n_updates % self.heatmap_frequency == 0:
            self.log_heatmap(
                self.policy_loss_history,
                "Policy Loss",
                "heatmaps/policy_loss"
            )
            self.log_heatmap(
                self.value_loss_history,
                "Value Loss",
                "heatmaps/value_loss"
            )
            self.log_heatmap(
                self.kl_history,
                "Approx KL",
                "heatmaps/approx_kl"
            )
            self.log_heatmap(
                self.clip_fraction_history,
                "Clip Fraction",
                "heatmaps/clip_fraction"
            )
            self.log_heatmap(
                self.entropy_history,
                "Entropy",
                "heatmaps/entropy"
            )
        # ========================

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
    
    def log_heatmap(self, data, title, tag):

        data = np.array(data)

        fig, ax = plt.subplots(figsize=(8, 6))

        im = ax.imshow(
            data,
            aspect="auto",
            interpolation="nearest"
        )

        ax.set_xlabel("Optimization Epoch")
        ax.set_ylabel("PPO Update")

        ax.set_title(title)

        fig.colorbar(im, ax=ax)

        self.logger.record(
            tag,
            Figure(fig, close=True),
            exclude=("stdout", "log", "json")
        )

        plt.close(fig)