import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter, label, center_of_mass

"""
    This class acts as a collection of static methods
    which define useful tools to analyze images produced
    by the pSCT.
"""
class ImageAnalyzer():
    """
        Gets the image x, y focal plane coordinates of any detected centroids
    """
    def get_centroid_locations(image):
        pts_uv = ImageAnalyzer._detect_centroids_uv(image)
        if len(pts_uv) == 0:
            return np.zeros((0, 2), float)

        x_fp, y_fp = ImageAnalyzer._uv_to_fp(pts_uv[:, 0], pts_uv[:, 1])
        pts_fp = np.vstack([x_fp, y_fp]).T
        return pts_fp
    
    """
        converts given pixel coordinates to focal plane coordinates
    """
    def _uv_to_fp(u, v):
        """image pixels -> focal-plane pixels"""
        img_fov_pix = 600.0
        img_size = 128
        center = np.array([1612.2804, 1024.4423])

        half = img_fov_pix / 2.0
        dx = (u / (img_size - 1)) * (2 * half) - half
        dy = (v / (img_size - 1)) * (2 * half) - half
        x_fp = center[0] + dx
        y_fp = center[1] + dy
        return x_fp, y_fp
    
    """
        given the image, this function finds the centroid of each gaussian and
        stores and returns the pixel coordinate of each. 
    """
    def _detect_centroids_uv(img):
        """
        Simple detector:
          1) smooth
          2) threshold (robust sigma)
          3) local maxima
          4) connected-component COM to merge plateaus/blobs
        Returns centroids in image pixel coords (u,v).
        """
        det_smooth_sigma = 1.2
        det_thresh_sigma = 8.0
        img_size = 128
        det_max_peaks = 64
        det_merge_radius_pix = 2.0


        sm = gaussian_filter(img, det_smooth_sigma)

        # robust background/sigma from median + MAD
        med = np.median(sm)
        mad = np.median(np.abs(sm - med))
        sigma = 1.4826 * mad if mad > 0 else np.std(sm) + 1e-9

        thr = med + det_thresh_sigma * sigma
        mask = sm > thr

        if not np.any(mask):
            return np.zeros((0, 2), float)

        # local maxima among a 3x3 neighborhood
        mx = (sm == maximum_filter(sm, size=3)) & mask

        # label maxima regions (plateaus)
        lab, nlab = label(mx)
        if nlab == 0:
            return np.zeros((0, 2), float)

        com = center_of_mass(sm, lab, np.arange(1, nlab + 1))
        # com is list of (v,u) because array indexing is (row,col)
        pts = np.array([(u, v) for (v, u) in com], float)

        # keep strongest peaks if too many
        if pts.shape[0] > det_max_peaks:
            # score by sm at nearest integer pixel
            ui = np.clip(np.round(pts[:, 0]).astype(int), 0, img_size - 1)
            vi = np.clip(np.round(pts[:, 1]).astype(int), 0, img_size - 1)
            score = sm[vi, ui]
            keep = np.argsort(score)[-det_max_peaks:]
            pts = pts[keep]

        # optional merge close detections
        if pts.shape[0] >= 2 and det_merge_radius_pix > 0:
            pts = ImageAnalyzer._merge_close_points(pts, det_merge_radius_pix)
        
        return pts
    
    """
        used as a helper function for _detect_centroids_uv(). If two detected peaks
        are too close together, they are merged into one detected peak.
    """
    def _merge_close_points(pts, r):
        keep = []
        used = np.zeros(len(pts), dtype=bool)
        for i in range(len(pts)):
            if used[i]:
                continue
            d = np.sqrt(np.sum((pts - pts[i]) ** 2, axis=1))
            grp = np.where(d <= r)[0]
            used[grp] = True
            keep.append(np.mean(pts[grp], axis=0))
        return np.array(keep, float)
    
    """
        specifies whether the current telescope has all of the centroids at
        the center of the detected image.

        Returns: true if all the detected centroids are at the center of the screen, false otherwise
    """
    def all_centroids_at_center(center, centroid_locations, success_radius=5):
        # find distance of each centroid to the center
        d = centroid_locations - center[None, :]
        r = np.sqrt(np.sum(d**2, axis=1))

        # success if the CLOSEST n_panels detections are all within radius
        return not bool(np.any(r > success_radius))
    
    """
        specifies whether the current telescope has any of the created centroids
        outside the detectable area. the center and screen_size are both given
        in fp coordinates
        Returns: true if any centroid is outside the image, false otherwise
    """
    def any_centroid_outside_image(center, screen_size, centroid_locations):
        for (fx, fy) in centroid_locations:
            if ((np.abs(fx - center[0]) > screen_size / 2) or (np.abs(fy - center[1]) > screen_size / 2)):
                return True
        return False