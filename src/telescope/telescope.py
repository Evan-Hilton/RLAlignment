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
        self.img_size = telescope_config["img_size"]
        self.image = np.zeros((self.img_size, self.img_size))

    """
        panel_ids is just a list of all the ids of the panels you
        want to rotate
    """
    def rotate_panels(self, panel_ids):
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
    def update(self):
        pass