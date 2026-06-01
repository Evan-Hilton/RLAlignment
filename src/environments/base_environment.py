import gymnasium as gym

""" 
    Base class implementation of the environment which an RL
    agent would exist in to align a telescope. Includes general
    API in accordance with SB3 and is a little more specific on 
    its use of a telescope. However, this class does not provide
    enough specificity to actually train an agent, so training
    should only be done with inhereted classes.
"""
class BaseEnv(gym.Env):

    def __init__(self,
                 max_steps,
                 telescope_config,
                 observation_space,
                 action_space):
        
        # bookkeeping
        self.step_count = 0
        self.max_steps = max_steps

        self.telescope = None
        self.initialize_telescope(telescope_config) # should initialize telescope

        self.observation_space = observation_space

        self.action_space = action_space
    
    # =================================== API ===================================
    
    """
        Run one time step simulation of the environment that the agent exists in.
        
        It should: apply the action supplied by the agent, make any necessary updates
        to the telescope, retrieve an observation, reward, and whether termination of
        the episode should happen and return this information. It should also keep
        track of whether the episode should time out.
    """
    def step(self, action):
        self.apply_action(action)

        observation = self.get_observation()

        reward = self.get_current_reward(observation)

        terminated, termination_reward = self.check_terminated(observation)
        reward += termination_reward

        truncated = self.step_count >= self.max_steps - 1
        self.step_count += 1

        return observation, reward, terminated, truncated, {}

    """
        Resets the environment to an initial random state. This includes all things
        needed to reset the simulation of the telescope, and also to return the 
        observation from being in the new random state.
    """
    def reset(self, *, seed=None, options=None):
        # bookkeeping
        self.step_count = 0
        self.reset_telescope()

        observation = self.get_observation()

        return observation, {}