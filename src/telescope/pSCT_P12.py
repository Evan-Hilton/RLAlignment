import numpy as np
import yaml
from scipy.ndimage import gaussian_filter

from src.telescope.telescope import Telescope
from src.telescope.panel_rotation_corrections import *

"""
    This class defines a simulation of a pSCT telescope object.
    A pSCT telescope represents the current state of the simulated telescope.
    This telescope has P1 and P2 mirror segments, and optionally simulates
    tube dragging
"""
class PSCT_P12(Telescope):
    def __init__(self, config):
        telescope_config = self.unload_config(config)
        super().__init__(telescope_config)
        self.rng = np.random.RandomState(67)

        # panel information
        self.n_panels = telescope_config["n_panels"] if "panels" not in telescope_config else len(telescope_config["panels"])
        self.P1s = [1111, 1112, 1113, 1114, 1211, 1212, 1213, 1214, 1311, 1312, 1313, 1314, 1411, 1412, 1413, 1414]
        self.P2s = [1221, 1222, 1223, 1224, 1225, 1226, 1227, 1228, 1321, 1322, 1323, 1324, 1325, 1326, 1327, 1328, 1421, 1422, 1423, 1424, 1425, 1426, 1427, 1428, 1121, 1122, 1123, 1124, 1125, 1126, 1127, 1128]
        self.Ps = self.P1s + self.P2s
        self.panels = self.Ps[:self.n_panels] if "panels" not in telescope_config else telescope_config["panels"]
        self.true_centroids = np.zeros((self.n_panels, 2), dtype=float) # (total # of panels, 2) these are in fp coords, and ordered the same as self.panels
        self.base_offsets = np.zeros((self.n_panels, 2)) # same shape as true centroids, also in fp coords
        self.rx_ry = None # same shape as true centroids, describes the rotation of each panel in rx, ry
        self.center = np.asarray(telescope_config["center_fp"])
        
        # rotation information
        self.action_scale = telescope_config["action_scale"]
        self.action_noise_scale = telescope_config["action_noise_scale"]
        self.M_RxRy_invP1 = self.__load_all_rx_ry_matrices(respfile="src/telescope/P1_matrix.yaml")
        self.M_RxRy_invP2 = self.__load_all_rx_ry_matrices(respfile="src/telescope/P2_matrix.yaml")
        self.M_RxRy_inv = self.M_RxRy_invP1 | self.M_RxRy_invP2
        self.tube_dragging_scale = telescope_config["tube_dragging_scale"]

        # initial randomization information
        self.init_scatter_pix = telescope_config["init_scatter_pix"]
        self.init_rxry_scale = telescope_config["init_rxry_scale"]

        # image information
        self.bg_level = telescope_config["bg_level"]
        self.read_noise = telescope_config["read_noise"]
        self.img_fov_pix = telescope_config["img_fov_pix"]
        self.centroid_type = telescope_config["centroid_type"]

        self.reset()
    
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
    def rotate_panel(self, panel_id: int, rotation):
        add_rotation_noise(rotation, self.action_noise_scale)

        rotation_x = self.action_scale * rotation[0]
        rotation_y = self.action_scale * rotation[1]
        self.rx_ry[self.panels.index(panel_id)] += [rotation_x, rotation_y]

        for pid, rotation in drag_tubes(panel_id, (rotation_x, rotation_y), self.tube_dragging_scale).items():
            if pid in self.panels:
                rx = rotation[0]
                ry = rotation[1]
                self.rx_ry[self.panels.index(pid)] += [rx, ry]
        
        self.__compute_true_centroids()

    """
        Randomly misaligns every panel in the telescope
    """
    def reset(self):
        n_panels = len(self.panels)
        self.base_offsets = (self.rng.rand(n_panels, 2) - 0.5) * self.init_scatter_pix
        self.rx_ry = (self.rng.rand(n_panels, 2) - 0.5) * self.init_rxry_scale # (n_panels, 2)
        self.__compute_true_centroids()
        self.update()

    """
        Updates self.image to reflect the current image seen by the
        telescope. only centroids created by panels in self.panels 
        will appear in the image
    """
    def update(self):
        super().update(self.multiple_fp_to_uv(self.true_centroids), centroid_type=self.centroid_type)

        # detector noise and background
        self.image += self.bg_level
        self.image += self.rng.normal(scale=self.read_noise, size=self.image.shape)
        if "scale" in self.config:
            if self.config["scale"] == "normalize":
                self.rescale_image()
            else:
                self.clip_image()
        else:
            self.clip_image()

    # ========================== Helper Methods ==========================

    """
        Computes the true positions of each image created by each panel.
        Updates self.true_centroids to represent the new centroid positions (if panels have been moved).
        Everything is computed in focal plane coordinates
    """
    def __compute_true_centroids(self):
        centroids = np.zeros_like(self.true_centroids, dtype=float)
        for i, panel in enumerate(self.panels):
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
    
    def multiple_uv_to_fp(self, coordinates):
        """computes uv_to_fp for the numpy array of coordinates.
            coordinates should have shape (2, n)"""
        ret = np.zeros_like(coordinates)
        for i, (xuv, yuv) in enumerate(coordinates):
            fpx, fpy = self.uv_to_fp(xuv, yuv)
            ret[i] = (fpx, fpy)
        return ret