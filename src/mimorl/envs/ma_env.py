import functools
import copy
from collections.abc import Iterable

import numpy as np
from numpy import log10, log2
import numpy.linalg as LA
import matplotlib.pyplot as plt

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from gymnasium import spaces

from ..antenna_array import AntennaArray
from ..channels import *
from ..network import Network


class MIMOMAEnv(AECEnv):
    """mimo multi-agent environment"""

    metadata = {"render.modes": ["human"], "name": "MIMO-MA"}

    def __init__(self, network: Network, target_meas: float | Iterable):
        # pettingzoo related
        self.possible_agents = list(range(len(network.target_nodes)))
        self.render_mode = "human"
        self.timestamp = None
        self.max_timestamp = 1000

        self.network = copy.deepcopy(network)
        # mimo aliases
        self.net = self.network
        self.nodes = self.network.nodes
        self.links = self.network.links
        self.nodes_dict = self.network.nodes_dict
        self.target_nodes = self.network.target_nodes
        self.target_links = self.network.target_links
        self.best_meas = None

        # mimo related
        self.max_amp_change = 0.1
        self.max_phase_change = 36 * np.pi / 180
        self.metrics = "sinr"
        self.meas_buffer_size = 1  # number of measurements to average over
        if isinstance(target_meas, Iterable):
            if len(target_meas) != len(self.target_links):
                raise ValueError(
                    f"target_meas must be of length {len(self.target_links)}"
                )
            self.target_meas = target_meas
        else:
            self.target_meas = [target_meas] * len(self.target_links)
        self.target_meas = np.asarray(self.target_meas)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: Any) -> spaces.Space:
        """Return the observation space with the entire state info."""
        # any critic has access to the entire state
        num_antennas = self.target_nodes[agent].Nr
        num_links = len(self.target_links)
        low = []
        high = []
        num_l = len(self.target_links)
        low.extend([-200] * num_l + [0] * num_l + [-200] * num_l)
        high.extend([200] * num_l + [200] * num_l + [200] * num_l)
        for a in self.agents:
            low.extend([0] * num_antennas)  # amp
            high.extend([self.target_nodes[a].power] * num_antennas)
            low.extend([-np.pi] * num_antennas)  # phase
            high.extend([np.pi] * num_antennas)
        return spaces.Box(low=np.asarray(low), high=np.asarray(high))

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: Any) -> spaces.Space:
        """Return the action space for the given agent with shape (len([amp, phase]), num_antennas)"""
        num_antennas = self.target_nodes[agent].Nr
        amp_phase_change = np.array([self.max_amp_change, self.max_phase_change])
        space = np.outer(amp_phase_change, np.ones(num_antennas))
        return spaces.Box(low=-space, high=space)

    def observe(self, agent):
        observations = np.concatenate(
            (self.sinrs, self.spectral_efficiencies, self.gains), axis=0
        )
        for a in self.agents:
            amp = self.target_nodes[a].amp
            phase = self.target_nodes[a].phase
            observations = np.concatenate([observations, amp, phase], axis=0)
        return observations.astype(np.float32)

    @property
    def state(self):
        return{
            "sinr": self.sinrs,
            "spectral_efficiency": self.spectral_efficiencies,
            "gain": self.gains,
            "target_meas": self.target_meas,
            "best_meas": self.best_meas,
            "best_weights": self.best_weights,
            "weights": [self.target_nodes[a].weights for a in self.agents],
            "amp": [self.target_nodes[a].amp for a in self.agents],
            "phase": [self.target_nodes[a].phase for a in self.agents],
        }

    def close(self):
        plt.close("all")

    def render(self, mode="human"):
        """Render the environment."""
        if mode == "human":
            plt.close("all")
            self.net.plot()
            self.net.plot_gain(self.best_weights)
        else:
            raise NotImplementedError

    def reset(self, seed=None, options=None):
        """
        Reset needs to initialize the following attributes
        - agents
        - rewards
        - _cumulative_rewards
        - terminations
        - truncations
        - infos
        - agent_selection
        And must set up the environment so that render(), step(), and observe()
        can be called without issues.
        Here it sets up the state dictionary which is used by step() and the observations dictionary which is used by step() and observe()
        """
        self.agents = self.possible_agents.copy()
        self.rewards = {a: 0 for a in self.agents}
        self._cumulative_rewards = {a: 0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.timestamp = 0

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

        # network reset
        for node in self.target_nodes:
            node.set_weights(1)
        obs = self.observe(self.agent_selection)
        self.best_weights = [self.target_nodes[a].weights for a in self.agents]
        self.best_meas = self.state[self.metrics]
        self.infos = {a: {} for a in self.agents}
        # return obs, self.infos

    def step(self, action):
        """
        step(action) takes in an action for the current agent (specified by
        agent_selection) and needs to update
        - rewards
        - _cumulative_rewards (accumulating the rewards)
        - terminations
        - truncations
        - infos
        - agent_selection (to the next agent)
        And any internal state used by observe() or render()
        """
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            # handles stepping an agent which is already dead accepts a None
            # action for the one agent, and moves the agent_selection to the
            # next dead agent, or if there are no more dead agents, to the next
            # live agent
            self._was_dead_step(action)
            return
        # the agent which stepped last had its _cumulative_rewards accounted for
        # (because it was returned by last()), so the _cumulative_rewards for this
        # agent should start again at 0
        self._cumulative_rewards[self.agent_selection] = 0

        self.infos = {a: {} for a in self.agents}
        action = np.asarray(action, dtype=np.float32)
        self.update_weights(self.agent_selection, action)
        # obs = self.observe(self.agent_selection)
        self.rewards[self.agent_selection] = self.get_reward()
        # truncate if the agent has reached the target_meas
        # self.truncations = {
        #     a: self.state[self.metrics][a] >= self.target_meas[a] for a in self.agents
        # }
        # terminate if the agent has reached step limit
        terminate = self.timestamp >= self.max_timestamp
        self.terminations = {a: terminate for a in self.agents}
        self._accumulate_rewards()

        # return obs, self.rewards, self.terminations, self.truncations, _infos

    # =========================================================================
    # Network updates
    # =========================================================================

    def update_weights(self, agent, action):
        """Update the weights of the given agent."""
        amp_change, phase_change = action.reshape(2, -1)
        node = self.target_nodes[agent]
        new_amp = np.clip(node.amp + amp_change, 0, node.power)
        new_phase = node.phase + phase_change
        new_phase -= new_phase[0]  # normalize phase to the first element
        node.set_weights(new_amp * np.exp(1j * new_phase))
        # return node.weights

    def get_reward(self):
        return np.mean(self.state[self.metrics])

    @property
    def sinrs(self):
        return np.asarray(
            [self.net.sinr(l) for l in self.target_links], dtype=np.float32
        )

    @property
    def spectral_efficiencies(self):
        return np.asarray(
            [self.net.spectral_efficiency(l) for l in self.target_links],
            dtype=np.float32,
        )

    @property
    def gains(self):
        return np.asarray(
            [self.net.gain(l) for l in self.target_links], dtype=np.float32
        )
