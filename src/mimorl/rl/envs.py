from collections import abc
from collections.abc import Iterable

import numpy as np
from numpy import log10
from matplotlib import pyplot as plt
import gymnasium as gym

from ..antenna_array import AntennaArray
from ..channel import *
from ..network import Network


class MIMOEnv(gym.Env):
    """RL environment class."""

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, name=None):
        self.name = name
        if name is None:
            self.name = self.__class__.__name__
        self.network = Network()
        self.target_links = []
        self.controlled_nodes = []
        self.max_amp_change = 0.1  # percentage change in amplitude
        self.max_phase_change = 36  # degrees
        self.tolerance = 1
        self.tolerance_decay = 0.1
        self.tolerance_update_factor = 1.5  # factor by which trigger tolerance update
        self.metrics = "sinr"
        self.meas_buffer_size = 1  # number of measurements to keep
        self.target_meas = None
        self.best_meas = []
        self.best_weights = []
        self.metadata = {"render.modes": ["human"]}

    def __str__(self):
        return self.name

    @property
    def controlled_weights(self):
        return [node.weights for node in self.controlled_nodes]

    @property
    def target_links_sinr(self):
        return [self.network.sinr(link) for link in self.target_links]

    # Network related methods
    @classmethod
    def from_network(cls, network: Network, target_links, controlled_nodes):
        """Create an environment from a network."""
        env = cls()
        env.network = network
        # Target link and controlled nodes are needed, otherwise reset() will break
        env.add_target_links(target_links)
        env.add_controlled_nodes(controlled_nodes)
        env.reset()
        obs = env.get_obs()
        env.best_meas = [obs[env.metrics]]
        env.best_weights = [env.controlled_weights]
        return env

    def add_target_links(self, links):
        """Add target links to the environment."""
        if isinstance(links, Iterable):
            for link in links:
                if link not in self.target_links:
                    self.target_links.append(link)
        else:
            if links not in self.target_links:
                self.target_links.append(links)

    def remove_target_links(self, links):
        """Remove target links from the environment."""
        if isinstance(links, Iterable):
            for link in links:
                self.target_links.remove(link)
        else:
            self.target_links.remove(links)

    def add_controlled_nodes(self, nodes):
        """Add a controlled node to the environment."""
        if isinstance(nodes, Iterable):
            for node in nodes:
                if node not in self.controlled_nodes:
                    self.controlled_nodes.append(node)
        else:
            if nodes not in self.controlled_nodes:
                self.controlled_nodes.append(nodes)

    def remove_controlled_nodes(self, nodes):
        """Remove a controlled node from the environment."""
        if isinstance(nodes, Iterable):
            for node in nodes:
                self.controlled_nodes.remove(node)
        else:
            self.controlled_nodes.remove(nodes)

    def plot(self, plane="xy", label=False, **kwargs):
        """Plot the environment and highlight the controlled nodes and target links."""
        coord_idx = {"xy": [0, 1], "yz": [1, 2], "xz": [0, 2]}[plane]
        fig, ax = self.network.plot(plane=plane, label=label, **kwargs)
        for node in self.controlled_nodes:
            ax.scatter(
                *node.location[coord_idx], s=75, facecolors="b", label="Controlled Node"
            )
            # ax.legend()
        for link in self.target_links:
            ul_loc = link.tx.location[coord_idx]
            dl_loc = link.rx.location[coord_idx]
            ax.plot(
                [ul_loc[0], dl_loc[0]],
                [ul_loc[1], dl_loc[1]],
                "c-",
                label="Target Link",
            )
        ax.invert_yaxis()
        plt.show()

    def plot_3d(self, **kwargs):
        """Plot the environment in 3D and highlight the controlled nodes and target links."""
        fig, ax = self.network.plot_3d(**kwargs)
        for node in self.controlled_nodes:
            ax.scatter(*node.location, s=75, facecolors="b", label="Controlled Node")
            # ax.legend()
        for link in self.target_links:
            ax.plot(
                [link.tx.location[0], link.rx.location[0]],
                [link.tx.location[1], link.rx.location[1]],
                [link.tx.location[2], link.rx.location[2]],
                "c-",
                label="Target Link",
            )
            # ax.legend()
        plt.show()

    def plot_gain(self, best_weights=False, **kwargs):
        """Plot the beam pattern of the controlled nodes."""
        num_plots = len(self.controlled_nodes)
        num_cols = np.ceil(np.sqrt(num_plots)).astype(int)
        num_rows = np.ceil(num_plots / num_cols).astype(int)
        fig, axes = plt.subplots(
            num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows), **kwargs
        )
        for node, ax, weights in zip(
            self.controlled_nodes, np.ravel(axes), self.best_weights[-1]
        ):
            if best_weights:
                node.plot_gain(ax=ax, weights=weights)
            else:
                node.plot_gain(ax=ax)
            title = ax.get_title()
            ax.set_title(f"{node.name}: {title}")
        plt.tight_layout()
        plt.show()

    # Gym related methods
    @property
    def action_space(self):
        """Return the action space based on the controlled nodes. 
        The action space is a Box space with shape (2, total_num_antennas)"""
        # get the total number of antennas for all controlled nodes
        total_num_antennas = np.sum(
            [node.num_antennas for node in self.controlled_nodes], dtype=np.int8
        )
        amp_phase_change = np.array(
            [self.max_amp_change, self.max_phase_change * np.pi / 180]
        )
        # create the action space via outer product (2x1) x (1xN) -> (2xN)
        # N is the total number of controlled antennas across all nodes
        space = np.outer(amp_phase_change, np.ones(total_num_antennas))
        space = np.float16(space)
        return gym.spaces.Box(low=-space, high=space)

    @property
    def action_space_tuple(self):
        """Action space is a Tuple of Box spaces for each controlled node.
        The Box spaces are 2d arrays of dim (2, num_antennas) for amplitude and phase changes
        """
        space_list = []
        change = np.array([self.max_amp_change, self.max_phase_change * np.pi / 180])
        for node in self.controlled_nodes:
            space = np.outer(change, np.ones(node.num_antennas)).reshape(2, -1)
            space = np.float16(space)
            space = gym.spaces.Box(low=-space, high=space)
            space_list.append(space)
        return gym.spaces.Tuple(space_list)

    @action_space.setter
    # read only property
    def action_space(self, value):
        raise AttributeError("action_space is read-only")

    @property
    def observation_space(self):
        """Return the observation space based on the target links."""
        total_num_antennas = np.sum(
            [node.num_antennas for node in self.controlled_nodes], dtype=np.int8
        )

        return gym.spaces.Dict(
            {
                "sinr": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(len(self.target_links),),
                    dtype=np.float16,
                ),
                "spectral_effeciency": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(len(self.target_links),),
                    dtype=np.float16,
                ),
                "gain": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(len(self.target_links),),
                    dtype=np.float16,
                ),
                "amp": gym.spaces.Box(low=0, high=np.inf, shape=(total_num_antennas,)),
                "phase": gym.spaces.Box(
                    low=-np.pi,
                    high=np.pi,
                    shape=(total_num_antennas,),
                    dtype=np.float16,
                ),
            }
        )

    @observation_space.setter
    # read only property
    def observation_space(self, value):
        raise AttributeError("observation_space is read-only")

    # ========================================================================
    # Update the following methods for the custom environment
    # ========================================================================

    def update_weights(self, action, **kwargs):
        """Update the weights of the controlled nodes."""
        # split the action into amplitude and phase changes
        nums_antennas = [node.num_antennas for node in self.controlled_nodes]
        changes = np.split(
            action.reshape((2, -1)), np.cumsum(nums_antennas)[:-1], axis=1
        )
        for node, change in zip(self.controlled_nodes, changes):
            # clip the changes to the max values
            new_amp = np.clip(np.abs(node.weights) + change[0], 0, node.power)
            new_phase = np.angle(node.weights) + change[1]
            new_phase -= new_phase[0]  # normalize the phase to the first antenna
            node.set_weights(new_amp * np.exp(1j * new_phase), **kwargs)

    def update_tolerance(self):
        self.tolerance *= 1 - self.tolerance_decay

    def get_reward(self, meas):
        """Calcuate the reward

        Args:
            meas (np.ndarray): The measurements from the environment

        Returns:
            reward (float): The reward
            is_best (bool): Whether the reward should be recorded as the best."""
        raise NotImplementedError("get_reward() must be implemented in a subclass.")
        reward = np.mean(meas) - np.mean(self.best_meas)
        return reward, reward > 0

    # def process_meas(self, meas) -> float:
    #     """Process the measurements. Called in step()."""
    #     return np.mean(meas)

    # ========================================================================
    # Environment related methods
    # ========================================================================

    def update_reward(self, meas):
        # proc_meas = self.process_meas(meas)
        if len(self.best_meas) == 0:
            self.best_meas.append(meas)
            self.best_weights.append(self.controlled_weights)
            return 0
        reward, is_best = self.get_reward(meas)
        if is_best:
            self.best_meas.append(meas)
            self.best_weights.append(self.controlled_weights)
            if len(self.best_meas) > self.meas_buffer_size:
                self.best_meas.pop(0)
                self.best_weights.pop(0)
        return reward

    def get_obs(self):
        return {
            "sinr": [self.network.sinr(link) for link in self.target_links],
            "spectral_effeciency": [
                self.network.spectral_efficiency(link) for link in self.target_links
            ],
            "gain": [self.network.bf_gain(link) for link in self.target_links],
            "amp": np.concatenate(
                [np.abs(node.weights) for node in self.controlled_nodes]
            ),
            "phase": np.concatenate(
                [np.angle(node.weights) for node in self.controlled_nodes]
            ),
        }

    def get_info(self):
        return {
            "target_meas": self.target_meas,
            "tolerance": self.tolerance,
            "best_meas": self.best_meas[-1],
            "best_weights": self.best_weights[-1],
            "controlled_weights": self.controlled_weights,
            # "processed_meas": self.process_meas(self.best_meas[-1]),
        }

    def get_done(self):
        dist = self.target_meas - np.mean(self.best_meas)
        if dist < self.tolerance * self.tolerance_update_factor:
            self.update_tolerance()
        return dist < self.tolerance

    def step(self, action):
        self.update_weights(action)
        obs = self.get_obs()
        reward = self.update_reward(obs[self.metrics])
        done = self.get_done()
        return obs, reward, done, False, self.get_info()

    def reset(self, **kwargs):
        for node in self.controlled_nodes:
            node.set_weights(np.ones(node.num_antennas))
        obs = self.get_obs()
        self.best_meas = [obs[self.metrics]]
        self.best_weights = [self.controlled_weights]
        return obs, self.get_info()

    def render(self, mode="human"):
        # if mode == "human":
        plt.close("all")
        self.plot(dpi=150)
        self.plot_gain(True, dpi=300)
        plt.show()
        return None
