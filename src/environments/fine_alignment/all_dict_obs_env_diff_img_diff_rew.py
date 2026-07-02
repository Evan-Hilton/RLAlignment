from src.environments.fine_alignment.all_dict_obs_env_diff_img import *

class AllDictObsEnvDiffImgDiffRew(AllDictObsEnvDiffImg):
    def get_current_reward(self, observation):
        detected_centroids = observation["detected_centroids"][:, [0, 1]]
        new_detected_centroids = []
        for centroid in detected_centroids:
            if not (centroid==0).all(): 
                new_detected_centroids.append([centroid[0], centroid[1]])
        detected_centroids = np.array(new_detected_centroids)
        d = self.telescope.multiple_uv_to_fp(detected_centroids * self.telescope.img_size) - self.telescope.center[None, :]
        mean_r2 = float(np.mean(np.sqrt(np.sum(d**2, axis=1))))
        mean_r2 /= self.telescope.init_scatter_pix

        mean_size = observation["detected_centroids"][:, [5, 6]]
        mean_size = np.max(mean_size, axis=0)
        mean_size = np.mean(mean_size)

        return -mean_r2 - mean_size