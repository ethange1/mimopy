import numpy as np

import gymnasium as gym

from ...array import Array
from ...channel import *
from ...network import Network


class Environment:
    """RL environment class."""

    def __init__(self, *args, **kwargs):
        self.name = "Environment"
        self.network = None
        self.target_links = []
        self.controlled_nodes = []
        self.max_amp_change = 0.1  # percentage change in amplitude
        self.max_phase_change = 36  # degrees
        self.tolerance = 1
        self.tolerance_decay = 0.1
        self.target = None
        self.best_meas = None
        self.meas_buffer_size = 1
        self.init_weights = None

    def __str__(self):
        return self.name

    @classmethod
    def from_network(cls, network):
        """Create an environment from a network."""
        env = cls()
        env.network = network
        return env

    def add_target_link(self, link):
        """Add a target link to the environment."""
        self.target_links.append(link)

    def remove_target_link(self, link):
        """Remove a target link from the environment."""
        self.target_links.remove(link)

    def add_controlled_node(self, node):
        """Add a controlled node to the environment."""
        self.controlled_nodes.append(node)

    def remove_controlled_node(self, node):
        """Remove a controlled node from the environment."""
        self.controlled_nodes.remove(node)

    @property
    def action_space(self):
        """Return the action space based on the controlled nodes."""
        # get the total number of antennas for all controlled nodes
        total_num_antennas = np.sum(
            [node.num_antennas for node in self.controlled_nodes]
        )
        amp_phase_change = np.array(
            [self.max_amp_change, self.max_phase_change * np.pi / 180], dtype=np.float32
        )
        # create the action space via outer product (2x1) x (1xN) -> (2xN)
        # N is the total number of controlled antennas across all nodes
        space = np.outer(amp_phase_change, np.ones(total_num_antennas))
        return gym.spaces.Box(low=-space, high=space, dtype=np.float16)

    @action_space.setter
    # read only property
    def action_space(self, value):
        raise AttributeError("action_space is read-only")

    @property
    def observation_space(self):
        """Return the observation space based on the target links."""
        total_num_antennas = np.sum(
            [node.num_antennas for node in self.controlled_nodes]
        )

        return gym.spaces.Dict(
            {
                "SINR": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(len(self.target_links),)
                ),
                "amp": gym.spaces.Box(low=0, high=np.inf, shape=(total_num_antennas,)),
                "phase": gym.spaces.Box(
                    low=-np.pi, high=np.pi, shape=(total_num_antennas,)
                ),
            }
        )

    @observation_space.setter
    # read only property
    def observation_space(self, value):
        raise AttributeError("observation_space is read-only")

    def _get_obs(self):
        return {
            "SINR": np.array(
                [self.network.get_sinr(link) for link in self.target_links]
            ),
            "R": np.array(
                [
                    self.network.get_spectral_efﬁciency(link)
                    for link in self.target_links
                ]
            ),
            "amp": np.concatenate(
                [np.abs(node.weights) for node in self.controlled_nodes]
            ),
            "phase": np.concatenate(
                [np.angle(node.weights) for node in self.controlled_nodes]
            ),
        }

    def _get_info(self):
        return {
            "snr_target": self.snr_target,
            "tolerance": self.tolerance,
        }

    def _update_weights(self, action):
        """Update the weights of the controlled nodes."""
        # split the action into amplitude and phase changes
        nums_antennas = [node.num_antennas for node in self.controlled_nodes]
        changes = np.split(
            action.reshape((2, -1)), np.cumsum(nums_antennas)[:-1], axis=1
        )
        for node, change in zip(self.controlled_nodes, changes):
            # clip the changes to the max values
            new_amp = np.clip(np.abs(node.weights) + change[0], 0, node.max_power)
            new_phase = np.angle(node.weights) + change[1]
            new_phase /= new_phase[0]  # normalize the phase to the first antenna
            node.set_weights(new_amp * np.exp(1j * new_phase))

    def _update_tolerance(self):
        self.tolerance *= 1 - self.tolerance_decay

    def _get_reward(self, meas):
        if self.best_meas is None:
            self.best_meas = meas
            return 0
        reward = meas - np.mean(self.best_meas)
        if reward > 0:
            self.best_meas.append(meas)
            if len(self.best_meas) > self.meas_buffer_size:
                self.best_meas.pop(0)
        return reward


    def step(self, action):
        self._update_weights(action)
        obs = self._get_obs()
        reward = self._get_reward()
        done = self._get_done()
        if done and self.tolerance_decay:
            self._update_tolerance()
        return obs, reward, done, self._get_info()
    
    def reset(self):
        self.best_meas = None
        for node in self.controlled_nodes:
            node.set_weights(np.ones(node.num_antennas))
