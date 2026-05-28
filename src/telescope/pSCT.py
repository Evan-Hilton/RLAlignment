import numpy as np
import yaml

"""
    This class defines a simulation of a pSCT telescope object.
    A pSCT telescope represents the current state of the simulated telescope.
"""
class pSCT:
    """
        Default pSCT constructor.
    """
    def __init__(self, telescope_config):
        
        self.rng = np.random.RandomState(67)
        
        # image information
        self.center = telescope_config["center_fp"]
        self.img_size = telescope_config["img_size"]

        # centroid creation
        self.init_scatter_pix = telescope_config["init_scatter_pix"]
        self.init_rxry_scale = telescope_config["init_rxry_scale"]
        self.img_fov_pix = telescope_config["img_fov_pix"]
        self.M_RxRy_inv = self.load_all_rx_ry_matrices(respfile="PSCT/P1_matrix.yml")
        self.last_detected_fp = None
        self.P1s = [1111, 1112, 1113, 1114, 1211, 1212, 1213, 1214, 1311, 1312, 1313, 1314, 1411, 1412, 1413, 1414]
        self.P2s = []

        # background noise
        self.bg_level = 6
        self.read_noise = 3

        # centroid locations
        self.base_offsets = None
        self.true_centroids = None # (total number of panels, 2)
        self.rx_ry = None

        # rotation information
        #self.action_scale = 0.05
        self.action_scale = 0.75

    """
        Returns the image currently seen by the pSCT camera of an on-axis star.
        panel_ids: a list of all panel_ids that are active in this telescope
        The returned image is bound in the usual range: [0, 255]. dtype=np.int8
        The returned image is a numpy array with shape (img_size, img_size).
    """
    def get_image(self, panel_ids):
        n_panels = len(panel_ids)

        # initialize the image
        img = np.zeros((self.img_size, self.img_size), float)
        params = np.zeros((n_panels, 6), float)

        # set up the paramaters for each centroid
        for i, (x_fp, y_fp) in enumerate(self.true_centroids):
            u, v = self.fp_to_uv(x_fp, y_fp)
            centerX, centerY = self.fp_to_uv(self.center[0], self.center[1])
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

        # normalize to [0,255] (adding background noise might have put values above 255)
        img = img - img.min()
        if img.max() > 0:
            img = 255.0 * img / img.max()
        
        return img
    
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

    """
        Rotate the panel specified by 'panel_id' by rx and ry
        panel_id: integer representation of the panel's id. ex: 1121
        rotation: the amount to rotate the panel by. rotation_x in one 
                  direction, rotation_y in another orthogonal direction.
                  rotations should be normalized between [-1, 1]. A value of 1 means
                  move the panel as much as it should be allowed to in that 
                  direction.

        Throws: ValueError: shape mismatch if rotation is not (1, 2)
    """
    def rotate_panel(self, panel_id: int, rotation_x, rotation_y):
        rotation_x *= self.action_scale
        rotation_y *= self.action_scale

        self.rx_ry[self.P1s.index(panel_id)] += [rotation_x, rotation_y]
        self.__compute_true_centroids()

    """
        Randomly misaligns every panel in the telescope
    """
    def set_random_rotations(self):
        self.base_offsets = (self.rng.rand(self.n_panels, 2) - 0.5) * self.init_scatter_pix
        self.rx_ry = (self.rng.rand(self.n_panels, 2) - 0.5) * self.init_rxry_scale # (n_panels, 2)
        self.__compute_true_centroids()

    """
        Computes the true positions of each image created by each panel.
        Updates self.true_centroids to represent the new centroid positions (if panels have been moved).
        Everything is computed in focal plane coordinates
    """
    def __compute_true_centroids(self):
        centroids = np.zeros((self.n_panels, 2), float)
        for i, panel in enumerate(self.P1s[:self.n_panels]):
            rx, ry = self.rx_ry[i]
            dx, dy = self.calc_dx_dy(rx, ry, self.M_RxRy_inv[panel]) # focal plane coords
            base_xy = self.center + self.base_offsets[i]
            centroids[i] = base_xy + np.array([dx, dy])
        self.true_centroids = centroids

    """
        Uses the response matrix M_RxRy_inv to convert rotation coordinates to focal plane coordinates.
        The response matrix M_RxRy_inv should be given as the real response matrix obtained
        experimentally on a real telescope.
    """
    def calc_dx_dy(self, rx, ry, M_RxRy_inv):
        M_RxRy = np.linalg.inv(M_RxRy_inv)
        dx, dy = np.matmul(M_RxRy, np.array([rx, ry]))
        return dx, dy
    
    """
        converts given focal plane coordinates to pixel coordinates
    """
    def fp_to_uv(self, x_fp, y_fp):
        """focal-plane pixels -> image pixels"""
        half = self.img_fov_pix / 2.0
        dx = x_fp - self.center[0]
        dy = y_fp - self.center[1]
        u = (dx + half) / (2 * half) * (self.img_size - 1)
        v = (dy + half) / (2 * half) * (self.img_size - 1)
        return u, v
    
    """
        converts given pixel coordinates to focal plane coordinates
    """
    def uv_to_fp(self, u, v):
        """image pixels -> focal-plane pixels"""
        half = self.img_fov_pix / 2.0
        dx = (u / (self.img_size - 1)) * (2 * half) - half
        dy = (v / (self.img_size - 1)) * (2 * half) - half
        x_fp = self.center[0] + dx
        y_fp = self.center[1] + dy
        return x_fp, y_fp

    """
        specifies whether the current telescope has any of the created centroids
        outside the detectable area.
        Returns: true if any centroid is outside the image, false otherwise

        NOTE: this method might use the true centroids. true centroid locations are not accessible
        on a real telescope, so use this method with caution
    """
    def any_centroid_outside_image(self):
        for (fx, fy) in self.true_centroids:
            if ((np.abs(fx - self.center[0]) > self.init_scatter_pix / 2) or (np.abs(fy - self.center[1]) > self.init_scatter_pix / 2)):
                return True
        return False

    """
        specifies whether the current telescope has all of the centroids at
        the center of the detected image.

        Returns: true if all the detected centroids are at the center of the screen, false otherwise

        NOTE: this method might use the true centroids. true centroid locations are not accessible
        on a real telescope, so use this method with caution
    """
    def all_centroids_at_center(self, centroid_locations=None, success_radius=5):
        # if no location is given, use the true location
        if centroid_locations is None:
            centroid_locations = self.true_centroids
        
        # find distance of each centroid to the center
        d = centroid_locations - self.center[None, :]
        r = np.sqrt(np.sum(d**2, axis=1))

        # success if the CLOSEST n_panels detections are all within radius
        return not bool(np.any(r > success_radius))


    """
        Helper method to load a file which hopefully contains all the response matrices
        for each panel.
    """
    def load_all_rx_ry_matrices(self, respfile, verbose=False):
        with open(respfile) as f:
            respM_yaml = yaml.safe_load(f)
        return respM_yaml
    
    """
        Returns the normalized true_centroids values. The values are normalized
        between -1 and 1, where the top left of the detector screen has a value of
        (-1, -1) and the bottom right of the detector screen has a value (1, 1)
    """
    def get_normalized_centroid_fp_coords_on_screen(self):
        # get screen coordinates of the corners of the screen
        left_of_screen, top_of_screen = self.uv_to_fp(0, 0)
        right_of_screen, bottom_of_screen = self.uv_to_fp(self.img_size, self.img_size)

        # normalize the true centroids to be between the values found above
        normalized_centroids = np.empty_like(self.true_centroids, dtype=np.float32)
        normalized_centroids[:, 0] = 2 * (self.true_centroids[:, 0] - left_of_screen) / (right_of_screen - left_of_screen) - 1
        normalized_centroids[:, 1] = 2 * (self.true_centroids[:, 1] - top_of_screen) / (bottom_of_screen - top_of_screen) - 1

        return normalized_centroids