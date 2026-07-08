import os
from stable_baselines3.common.callbacks import BaseCallback


class ModelSaveCallback(BaseCallback):
    def __init__(
        self,
        save_freq,
        save_path,
        verbose=1
    ):
        super().__init__(verbose)

        self.save_freq = save_freq
        self.save_path = save_path

        os.makedirs(save_path, exist_ok=True)

        self.next_save = save_freq


    def _on_step(self) -> bool:
        current_steps = self.num_timesteps

        if current_steps >= self.next_save:

            save_file = os.path.join(
                self.save_path,
                f"model_{self.next_save}_steps"
            )

            self.model.save(save_file)

            if self.verbose:
                print(f"Saved model checkpoint: {save_file}")

            self.next_save += self.save_freq

        return True