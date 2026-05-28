from src.environments.base_environment import BaseEnv
from src.telescope.pSCT import pSCT

"""
    BasicImageEnv is a subclass of BaseEnv which implements
    all required API not specified in BaseEnv. This class is
    meant to be an implementation of the environment given the
    following specifications:
        - uses image based observations from phase 1 of alignment
        - uses basic standard reward of mean squared error from
            distance of centroids from center
        - rotates only one panel at a time
"""
class BasicImageEnv(BaseEnv):
    def __init__(self, env_config):
        super().__init__(env_config=env_config)

        self.env_config = env_config

        self.current_panel = 0
        self.n_panels = env_config["n_panels"]
        self.P1s = [1111, 1112, 1113, 1114, 1211, 1212, 1213, 1214, 1311, 1312, 1313, 1314, 1411, 1412, 1413, 1414]

    def initialize_telescope(self):
        return pSCT(self.env_config["telescope"])

    """
        Rotates the currently selected panel by action amount. 
        Here, action should be a 1D array with two elements,
        where the first element is the chosen rotation in the
        x-direction (normalized between -1 and 1), and the 2nd
        element is the chosen y-direction rotation, similarly
        normalized.

        Here, we also increment which panel is being moved
        every frame
    """
    def apply_action(self, action):
        self.telescope.rotate_panel(self.P1s[self.current_panel], action[0], action[1])
        self.current_panel = (self.current_panel + 1) % self.n_panels

    def update_telescope(self):
        pass

    def get_observation(self):
        pass

    def compute_reward(self):
        pass

    def check_terminated(self):
        pass

    def initialize_observation(self):
        pass