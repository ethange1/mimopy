import functools

import numpy as np
from numpy import log10, log2
import numpy.linalg as LA
import matplotlib.pyplot as plt

from pettingzoo import AECEnv, ParallelEnv
from gymnasium import spaces

from ..antenna_array import AntennaArray
from ..channels import *
from ..network import Network


class MIMOMAEnv(AECEnv):
    """mimo multi-agent environment"""

    metadata = {"render.modes": ["human"], "name": "MIMO-MA"}

    def __init__(self, network: Network):
        self.network = network
        self.possible_agents = list(range(len(network.target_nodes)))
        self.render_mode = "human"
        self.max_amp_change = 0.1
        self.max_phase_change = 36 * np.pi / 180
        # aliases
        self.net = self.network
        self.nodes = self.network.nodes
        self.links = self.network.links
        self.nodes_dict = self.network.nodes_dict
        self.target_nodes = self.network.target_nodes
        self.target_links = self.network.target_links

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: Any) -> spaces.Space:
        """Return the observation space with the entire state info."""
        # any critic has access to the entire state
        num_antennas = [self.target_nodes[a].Nr for a in self.possible_agents]
        num_links = len(self.target_links)
        state_dict = {
            "sinr": spaces.Box(low=-np.inf, high=100, shape=(num_links,)),
            "spectral_efficiency": spaces.Box(low=0, high=np.inf, shape=(num_links,)),
            "gain": spaces.Box(low=-np.inf, high=np.inf, shape=(num_links,)),
        }
        # state_dict has amp and phase for target_nodes
        for a in self.agents:
            power = self.target_nodes[a].power
            name = self.target_nodes[a].name
            state_dict[f"amp{a}_{name}"] = spaces.Box(
                low=0, high=power, shape=(num_antennas[a],)
            )
            state_dict[f"phase{a}_{name}"] = spaces.Box(
                low=-np.pi, high=np.pi, shape=(num_antennas[a],)
            )
        return spaces.Dict(state_dict)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: Any) -> spaces.Space:
        """Return the action space for the given agent with shape (len([amp, phase]), num_antennas)"""
        num_antennas = self.target_nodes[agent].Nr
        amp_phase_change = np.array([self.max_amp_change, self.max_phase_change])
        space = np.outer(amp_phase_change, np.ones(num_antennas))
        return spaces.Box(low=-space, high=space)

    def observe(self, agent):
        """Return the observation for the given agent."""
        state_dict = {
            "sinr": [self.net.sinr(l) for l in self.target_links],
            "spectral_efficiency": [
                self.net.spectral_efficiency(l) for l in self.target_links
            ],
            "gain": [self.net.gain(l) for l in self.target_links],
            "amp": [self.target_nodes[a].amp for a in self.agents],
            "phase": [self.target_nodes[a].phase for a in self.agents],
        }

    @property
    def observation(self):
        """Return the observation for all agents."""
        return {a: self.observe(a) for a in self.agents}
    
    def close(self):
        plt.close("all")

    def render(self, mode="human"):
        """Render the environment."""
        if mode == "human":
            plt.close("all")
            self.net.plot()
            self.net.plot_gain(True)
        else:
            raise NotImplementedError