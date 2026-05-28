from src.environments.base_environment import BaseEnv

"""
    BasicImageEnv is a subclass of BaseEnv which implements
    all required API not specified in BaseEnv. This class is
    meant to be an implementation of the environment given the
    following specifications:
        - uses image based observations from phase 1 of alignment
        - uses basic standard reward of mean squared error from
            distance of centroids from center
        - 
"""
class BasicImageEnv(BaseEnv):
    def __init__():
        pass

    def apply_action(action):
        pass

    def update_telescope():
        pass

    def get_observation():
        pass

    def compute_reward():
        pass

    def check_terminated():
        pass

    def initialize_observation():
        pass