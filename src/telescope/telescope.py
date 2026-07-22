import numpy as np
import yaml

"""
    This class defines what a telescope should be
    in the context of CTAO gamma ray telescopes.
    A telescope should be able to:
        - Return an image of how it currently sees a bright, on-axis star
        - Rotate any panel by any amount
    
    This is only a base class for telescopes. Only derived
    classes should be used for simulation.
"""
class Telescope:
    """
        Default pSCT constructor.
    """
    def __init__(self, telescope_config):
        self.config = telescope_config
        self.img_size = telescope_config["img_size"]
        self.image = np.zeros((self.img_size, self.img_size))
        self.centroid_flux_max = telescope_config["centroid_flux_max"]

        # for image generation
        self.Y, self.X = np.mgrid[
            0:self.img_size,
            0:self.img_size
        ]

    """
        panel_ids is just a list of all the ids of the panels you
        want to rotate
    """
    def rotate_panel(self, panel_id: int, rotation):
        raise NotImplementedError

    """
        sets the panels to new random rotations
    """
    def reset(self):
        raise NotImplementedError
    
    """
        Updates the telescope so any changes are reflected properly in the
        telescope state. For example, this method is in charge of updating
        the simulated image of an on-axis star
    """
    def update(self, img_centroid_locations=None, centroid_type="gaussian"):
        if img_centroid_locations is None:
            self.image = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        elif centroid_type == "gaussian":
            self.image = self.create_gaussian_image(img_centroid_locations)
        elif centroid_type == "poly":
            self.image = self.create_poly_image(img_centroid_locations)
    
    """
        if any pixel values are outside 0, 255, they are clipped
    """
    def clip_image(self):
        self.image = np.clip(self.image, 0, 255)
    
    """
        if any pixel values are below zero, then all pixels are
        incremented until no pixel values are below zero. Then,
        the image is rescaled until all pixel values are less than 255
        but maintain the same shape
    """
    def rescale_image(self):
        self.image -= self.image.min()
        self.image *= 254 * (1 / self.image.max())
    
    """
        creates a new image which contains all centroids represented by gaussian
        distributions, who are paramaterized by the function specified in the 
        telescope config.
    """
    def create_gaussian_image(self, centroids):
        image = np.zeros((self.img_size, self.img_size), dtype=np.float32)

        center = self.img_size * 0.5

        for x0, y0 in centroids:
            # distance from image center
            dx_center = x0 - center
            dy_center = y0 - center

            R = np.sqrt(dx_center**2 + dy_center**2)

            sigma_r = self.centroid_spread_radial(R)
            sigma_t = self.centroid_spread_tangential(R)

            theta = np.arctan2(dy_center, dx_center)

            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)

            # Evaluate only inside a local bounding box
            radius = int(np.ceil(4.0 * max(sigma_r, sigma_t)))

            xmin = max(0, int(x0 - radius))
            xmax = min(self.img_size, int(x0 + radius + 1))

            ymin = max(0, int(y0 - radius))
            ymax = min(self.img_size, int(y0 + radius + 1))

            X = self.X[ymin:ymax, xmin:xmax]
            Y = self.Y[ymin:ymax, xmin:xmax]

            dx = X - x0
            dy = Y - y0

            u = dx * cos_theta + dy * sin_theta
            v = -dx * sin_theta + dy * cos_theta

            patch = np.exp(
                -0.5 * (
                    u * u / (sigma_r * sigma_r)
                    + v * v / (sigma_t * sigma_t)
                )
            )

            image[ymin:ymax, xmin:xmax] += patch * self.centroid_flux_max
        
        return image

    """
        creates a new image which contains all centroids represented by polynomial
        distributions, who are paramaterized by the function specified in the 
        telescope config.
    """
    def create_poly_image(self, centroids):
        image = np.zeros(
            (self.img_size, self.img_size),
            dtype=np.float32
        )

        center = self.img_size * 0.5

        FWHM_SCALE = 2.18 # np.sqrt(2 * np.log(2)) / np.sqrt(1 - 1 / np.sqrt(2))

        for x0, y0 in centroids:

            # Polar coordinates relative to image center
            dx_center = x0 - center
            dy_center = y0 - center

            R = np.sqrt(dx_center**2 + dy_center**2)

            sigma_r = self.centroid_spread_radial(R)
            sigma_t = self.centroid_spread_tangential(R)

            a = FWHM_SCALE * sigma_r
            b = FWHM_SCALE * sigma_t

            theta = np.arctan2(dy_center, dx_center)

            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)

            # Quartic kernel is exactly zero outside the ellipse,
            # so only evaluate inside the bounding box.
            radius = int(np.ceil(max(a, b)))

            xmin = max(0, int(x0 - radius))
            xmax = min(self.img_size, int(x0 + radius + 1))

            ymin = max(0, int(y0 - radius))
            ymax = min(self.img_size, int(y0 + radius + 1))

            X = self.X[ymin:ymax, xmin:xmax]
            Y = self.Y[ymin:ymax, xmin:xmax]

            dx = X - x0
            dy = Y - y0

            u = dx * cos_theta + dy * sin_theta
            v = -dx * sin_theta + dy * cos_theta

            r_e2 = (u * u) / (a * a) + (v * v) / (b * b)

            patch = np.zeros_like(r_e2, dtype=np.float32)

            mask = r_e2 < 1.0

            patch[mask] = (1.0 - r_e2[mask])**2

            image[ymin:ymax, xmin:xmax] += patch * self.centroid_flux_max

        return image

    """
        Given the distance of a centroid from the center of the screen, this function
        calculates and returns the correct distribution width along the radial direction.
        distance should be given in pixels, and this function assumes the center is at
        the center of the screen
    """
    def centroid_spread_radial(self, distance):
        return self.config["sigma_r_center"] + self.config["sigma_r_max"] * distance / (self.img_size * 0.5)
    
    """
        Given the distance of a centroid from the center of the screen, this function
        calculates and returns the correct distribution width along the tangential direction.
        distance should be given in pixels, and this function assumes the center is at
        the center of the screen
    """
    def centroid_spread_tangential(self, distance):
        return self.config["sigma_theta_center"] + self.config["sigma_theta_max"] * distance / (self.img_size * 0.5)