import numpy as np
import yaml
from scipy.ndimage import gaussian_filter

from src.telescope.telescope import Telescope

"""
    This class defines a simulation of a pSCT telescope object.
    A pSCT telescope represents the current state of the simulated telescope.
"""
class PSCT_P1(Telescope):
    def __init__(self, telescope_config):
        super().__init__(telescope_config)
        self.rng = np.random.RandomState(67)

        # panel information
        self.P1s = [1111, 1112, 1113, 1114, 1211, 1212, 1213, 1214, 1311, 1312, 1313, 1314, 1411, 1412, 1413, 1414]
        self.total_panels = len(self.P1s)
        self.true_centroids = np.zeros((self.total_panels, 2), dtype=float) # (total # of panels, 2) these are in fp coords
        self.base_offsets = np.zeros((self.total_panels, 2)) # same shape as true centroids, also in fp coords
        self.rx_ry = None
        self.center = telescope_config["center_fp"]
        
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

        # update telescope and state
        self.state["centroid_locations"] = self.true_centroids
        self.state["base_offsets"] = self.base_offsets
        self.state["center"] = self.center

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

    """
        Updates self.image to reflect the current image seen by the
        telescope. To only have certain panels appear in the image,
        specify them with panel_ids
    """
    def update(self, panel_ids=None):
        if panel_ids is None:
            panel_ids = self.P1s
        n_panels = len(panel_ids)

        # initialize the image
        img = np.zeros((self.img_size, self.img_size), float)
        params = np.zeros((n_panels, 6), float)

        # set up the paramaters for each centroid
        mask = np.isin(np.asarray(self.P1s), np.asarray(panel_ids))
        centroids = self.true_centroids[mask]
        for i, (x_fp, y_fp) in enumerate(centroids):
            u, v = self.__fp_to_uv(x_fp, y_fp)
            centerX, centerY = self.__fp_to_uv(self.center[0], self.center[1])
            r0 = np.sqrt((u-centerX)**2+(v-centerY)**2)
            params[i][0] = u
            params[i][1] = v
            params[i][2] = 255 # the height of each centroid
            params[i][3] = (np.tanh(0.17 * r0 - 3)) + 2
            params[i][4] = 0.006 * r0 + 1.00495
            params[i][5] = np.arctan((v - centerY) / (u - centerX)) if (u - centerX) != 0 else np.pi / 2
            
        img = self.__add_gaussians_batch(img, params)

        # detector noise and background
        img += self.bg_level
        img += self.rng.normal(scale=self.read_noise, size=img.shape)
        #img = gaussian_filter(img, sigma=1)
        
        # normalize to [0,255] (adding background noise might have put values above 255)
        img = img - img.min()
        if img.max() > 0:
            img = 255.0 * img / img.max()
        
        self.image = img

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
    
    def __fp_to_uv(self, x_fp, y_fp):
        """focal-plane pixels -> image pixels"""
        half = self.img_fov_pix / 2.0
        dx = x_fp - self.center[0]
        dy = y_fp - self.center[1]
        u = (dx + half) / (2 * half) * (self.img_size - 1)
        v = (dy + half) / (2 * half) * (self.img_size - 1)
        return u, v
    
    def __uv_to_fp(self, u, v):
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