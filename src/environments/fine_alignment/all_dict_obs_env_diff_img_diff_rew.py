from src.environments.fine_alignment.all_dict_obs_env_diff_img import *

class AllDictObsEnvDiffImgDiffRew(AllDictObsEnvDiffImg):
    def get_current_reward(self, observation):
        detected_centroids = observation["detected_centroids"][:, [0, 1]]
        d = detected_centroids - self.telescope.multiple_fp_to_uv(self.telescope.center[None, :])
        mean_r2 = float(np.mean(np.sqrt(np.sum(d**2, axis=1)))) * 2 # normalized distance from center (0 at center, 1 far away)

        mean_size = observation["detected_centroids"][:, [5, 6]]
        mean_size = np.max(mean_size, axis=0)
        mean_size = np.mean(mean_size)
        print(mean_size, mean_r2)

        return -mean_r2 - mean_size