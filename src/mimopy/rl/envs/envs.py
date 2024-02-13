from collections import abc
from collections.abc import Iterable

import numpy as np
import gymnasium as gym

from ...array import Array
from ...channel import *
from ...network import Network


class Environment:
    """RL environment class."""

    def __init__(self, name=None, *args, **kwargs):
        self.name = name
        if name is None:
            self.name = self.__class__.__name__
        self.network = None
        self.target_links = []
        self.controlled_nodes = []
        self.max_amp_change = 0.1  # percentage change in amplitude
        self.max_phase_change = 36  # degrees
        self.tolerance = 1
        self.tolerance_decay = 0.1
        self.target = None
        self.best_meas = None
        self.metrics = "SINR"
        self.meas_buffer_size = 1
        self.init_weights = None

    def __str__(self):
        return self.name

    # Network related methods
    @classmethod
    def from_network(cls, network):
        """Create an environment from a network."""
        env = cls()
        env.network = network
        return env

    def add_target_link(self, links):
        """Add a target link to the environment."""
        if isinstance(links, Iterable):
            for link in links:
                if link not in self.target_links:
                    self.target_links.append(link)
        else:
            if links not in self.target_links:
                self.target_links.append(links)

    def remove_target_link(self, links):
        """Remove a target link from the environment."""
        if isinstance(links, Iterable):
            for link in links:
                self.target_links.remove(link)
        else:
            self.target_links.remove(links)

    def add_controlled_node(self, nodes):
        """Add a controlled node to the environment."""
        if isinstance(nodes, Iterable):
            for node in nodes:
                if node not in self.controlled_nodes:
                    self.controlled_nodes.append(node)
        else:
            if nodes not in self.controlled_nodes:
                self.controlled_nodes.append(nodes)

    def remove_controlled_node(self, nodes):
        """Remove a controlled node from the environment."""
        if isinstance(nodes, Iterable):
            for node in nodes:
                self.controlled_nodes.remove(node)
        else:
            self.controlled_nodes.remove(nodes)

    def plot(self, plane="xy", **kwargs):
        """Plot the environment and highlight the controlled nodes and target links."""
        plot_coords = {"xy": [0, 1], "yz": [1, 2], "xz": [0, 2]}[plane]
        fig, ax = self.network.plot(plane=plane, **kwargs)
        for node in self.controlled_nodes:
            node



    # Gym related methods
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
        meas = np.sum(obs[self.metrics])
        reward = self._get_reward(meas)
        done = self._get_done()
        if done and self.tolerance_decay:
            self._update_tolerance()
        return obs, reward, done, self._get_info()

    def reset(self):
        self.best_meas = None
        for node in self.controlled_nodes:
            node.set_weights(np.ones(node.num_antennas))
