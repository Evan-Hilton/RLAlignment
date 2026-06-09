import numpy as np
import yaml
from scipy.ndimage import gaussian_filter

from src.telescope.telescope import Telescope

"""
    This class defines a simulation of a pSCT telescope object.
    A pSCT telescope represents the current state of the simulated telescope.
"""
class PSCT_P1(Telescope):
    def __init__(self, config):
        telescope_config = self.unload_config(config)
        super().__init__(telescope_config)
        self.rng = np.random.RandomState(67)

        # panel information
        self.P1s = [1111, 1112, 1113, 1114, 1211, 1212, 1213, 1214, 1311, 1312, 1313, 1314, 1411, 1412, 1413, 1414]
        self.total_panels = len(self.P1s)
        self.n_panels = telescope_config["n_panels"]
        self.true_centroids = np.zeros((self.total_panels, 2), dtype=float) # (total # of panels, 2) these are in fp coords
        self.base_offsets = np.zeros((self.total_panels, 2)) # same shape as true centroids, also in fp coords
        self.rx_ry = None
        self.center = np.asarray(telescope_config["center_fp"])
        
        # rotation information
        self.action_scale = telescope_config["action_scale"]
        self.M_RxRy_inv = self.__load_all_rx_ry_matrices(respfile="src/telescope/P1_matrix.yml")

        # initial randomization information
        self.init_scatter_pix = telescope_config["init_scatter_pix"]
        self.init_rxry_scale = telescope_config["init_rxry_scale"]

        # image information
        self.bg_level = telescope_config["bg_level"]
        self.read_noise = telescope_config["read_noise"]
        self.img_fov_pix = telescope_config["img_fov_pix"]
        self.centroid_type = telescope_config["centroid_type"]
    
    """
        given a path to a telescope config file, it loads the file
        and sets all necessary items
    """
    def unload_config(self, path):
        with open(
            path,
            "r"
        ) as f:
            config = yaml.safe_load(f)
        return config

    """
        Rotates the panel specified by panel_id by an amount in x and y
        directions. rotation_x and rotation_y should be normalized
        between -1 and 1
    """
    def rotate_panel(self, panel_id: int, rotation_x, rotation_y):
        rotation_x *= self.action_scale
        rotation_y *= self.action_scale

        self.rx_ry[self.P1s.index(panel_id)] += [rotation_x, rotation_y]
        self.__compute_true_centroids()

    """
        Randomly misaligns every panel in the telescope
    """
    def reset(self):
        self.base_offsets = (self.rng.rand(self.total_panels, 2) - 0.5) * self.init_scatter_pix
        self.rx_ry = (self.rng.rand(self.total_panels, 2) - 0.5) * self.init_rxry_scale # (n_panels, 2)
        self.__compute_true_centroids()
        self.update(panel_ids=self.P1s[:self.n_panels])

    """
        Updates self.image to reflect the current image seen by the
        telescope. To only have certain panels appear in the image,
        specify them with panel_ids
    """
    def update(self, panel_ids=None):
        if panel_ids is None:
            panel_ids = self.P1s
        
        super().update(self.multiple_fp_to_uv(self.true_centroids[:self.n_panels]), centroid_type=self.centroid_type)

        # detector noise and background
        self.image += self.bg_level
        self.image += self.rng.normal(scale=self.read_noise, size=self.image.shape)
        self.normalize_image()

    # ========================== Helper Methods ==========================

    """
        Computes the true positions of each image created by each panel.
        Updates self.true_centroids to represent the new centroid positions (if panels have been moved).
        Everything is computed in focal plane coordinates
    """
    def __compute_true_centroids(self):
        centroids = np.zeros_like(self.true_centroids, dtype=float)
        for i, panel in enumerate(self.P1s):
            rx, ry = self.rx_ry[i]
            dx, dy = self.__calc_dx_dy(rx, ry, self.M_RxRy_inv[panel]) # focal plane coords
            base_xy = self.center + self.base_offsets[i]
            centroids[i] = base_xy + np.array([dx, dy])
        self.true_centroids = centroids
    
    """
        Uses the response matrix M_RxRy_inv to convert rotation coordinates to focal plane coordinates.
        The response matrix M_RxRy_inv should be given as the real response matrix obtained
        experimentally on a real telescope.
    """
    def __calc_dx_dy(self, rx, ry, M_RxRy_inv):
        M_RxRy = np.linalg.inv(M_RxRy_inv)
        dx, dy = np.matmul(M_RxRy, np.array([rx, ry]))
        return dx, dy
    
    """
        Helper method to load a file which hopefully contains all the response matrices
        for each panel.
    """
    def __load_all_rx_ry_matrices(self, respfile):
        with open(respfile) as f:
            respM_yaml = yaml.safe_load(f)
        return respM_yaml
    
    def fp_to_uv(self, x_fp, y_fp):
        """focal-plane pixels -> image pixels"""
        half = self.img_fov_pix / 2.0
        dx = x_fp - self.center[0]
        dy = y_fp - self.center[1]
        u = (dx + half) / (2 * half) * (self.img_size - 1)
        v = (dy + half) / (2 * half) * (self.img_size - 1)
        return u, v
    
    def multiple_fp_to_uv(self, coordinates):
        """computes fp_to_uv for the numpy array of coordinates.
            coordinates should have shape (2, n)"""
        ret = np.zeros_like(coordinates)
        for i, (xfp, yfp) in enumerate(coordinates):
            u, v = self.fp_to_uv(xfp, yfp)
            ret[i] = (u, v)
        return ret
    
    def uv_to_fp(self, u, v):
        """image pixels -> focal-plane pixels"""
        half = self.img_fov_pix / 2.0
        dx = (u / (self.img_size - 1)) * (2 * half) - half
        dy = (v / (self.img_size - 1)) * (2 * half) - half
        x_fp = self.center[0] + dx
        y_fp = self.center[1] + dy
        return x_fp, y_fp
    
    """
        loops over each gaussian to be added to the image and adds it. This uses the optimized
        helper method add_gaussian() which adds a gaussian to the image.
    """
    def __add_gaussians_batch(self, img, params):
        for g in params:
            x0, y0, A, s_r, s_t, p = g
            img = self.__add_gaussian(img, x0, y0, A, s_r, s_t, p)
        return img
    
    """
        For optimization purposes, we only consider a small box around where
        we generate the gaussian. Values far away from the gaussian are basically
        zero, so theres no point wasting computation time calculating multiple exponentials
        that will inevitably be very small
    """
    def __add_gaussian(self, img, x0, y0, A, s_r, s_t, p):
        
        H, W = img.shape

        # cutoff
        k = 3.5
        R = int(np.ceil(k * max(s_r, s_t)))

        x_min = max(0, int(x0 - R))
        x_max = min(W, int(x0 + R + 1))
        y_min = max(0, int(y0 - R))
        y_max = min(H, int(y0 + R + 1))

        # local grid
        xs = np.arange(x_min, x_max)
        ys = np.arange(y_min, y_max)
        X, Y = np.meshgrid(xs, ys)

        # shift
        c = X - x0
        d = Y - y0

        # rotation
        cp = np.cos(p)
        sp = np.sin(p)

        a = c * cp + d * sp
        b = -c * sp + d * cp

        G = A * np.exp(
            -0.5 * ((a / s_r) ** 2 + (b / s_t) ** 2)
        )

        img[y_min:y_max, x_min:x_max] += G

        return img