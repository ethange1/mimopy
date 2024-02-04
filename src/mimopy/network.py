from typing import Any
import numpy as np
from numpy import log10

from array import Array
from channel import *
from matplotlib import pyplot as plt


class Network:
    """Network class.

    Attributes
    ----------
        name (str): Network name.
        nodes (List): List of nodes in the network.
        channels (List): List of channels in the network.
        adj_matrix (Array): edge-node adjacency matrix of the network.
            The rows are edges (channel) and the cols are nodes.
    """

    def __init__(self, *args, **kwargs):
        self.name = "Network"
        self.channels = []
        self.nodes = dict()

    def __str__(self):
        return self.name

    def add_node(self, node: Array):
        """Add a node to the network."""
        if node not in self.nodes:
            node.name += f"_{len(self.nodes)}"
            self.nodes[node] = {"dl": [], "ul": []}

    def add_channel(self, channel):
        """Add a channel to the network."""
        channel.name += f"_{len(self.channels)}"
        self.channels.append(channel)
        self.add_node(channel.tx)
        self.nodes[channel.tx]["dl"].append((channel.rx, channel))
        self.add_node(channel.rx)
        self.nodes[channel.rx]["ul"].append((channel.tx, channel))

    def remove_node(self, node: Array):
        """Remove a node and all channels associated with it from the network."""
        if node in self.nodes:
            for _, channel in self.nodes[node]["dl"]:
                # the node is the tx; remove ul from channel.rx
                self.channels.remove(channel)
                self.nodes[channel.rx]["ul"].remove((node, channel))
            for _, channel in self.nodes[node]["ul"]:
                # the node is the rx; remove dl from channel.tx
                self.channels.remove(channel)
                self.nodes[channel.tx]["dl"].remove((node, channel))

    def remove_channel(self, channel):
        """Remove a channel from the network."""
        self.channels.remove(channel)
        self.nodes[channel.tx]["dl"].remove((channel.rx, channel))
        self.nodes[channel.rx]["ul"].remove((channel.tx, channel))

    def get_bf_gain(self, channel) -> float:
        """Get the beamforming gain of the channel in dB."""
        return channel.get_bf_gain()

    def get_snr(self, channel) -> float:
        """Get the signal-to-noise ratio (SNR) of the channel in dB."""
        return self.get_bf_gain(channel) - channel.get_bf_noise()

    def get_interference(self, channel) -> float:
        """Get the interference-to-noise ratio (INR) of the channel in dB."""
        # interference is the sum of bf gains of all other ul channels of the rx
        interference = 0
        for _, ch in self.nodes[channel.rx]["ul"]:
            if ch != channel:
                interference += ch.get_bf_gain()

    def get_inr(self, channel) -> float:
        """Get the interference-to-noise ratio (INR) of the channel in dB."""
        return self.get_interference(channel) - channel.get_bf_noise()

    def get_sinr(self, channel) -> float:
        """Get the signal-to-interference-plus-noise ratio (SINR) of the channel in dB."""
        return (
            self.get_snr(channel)
            - self.get_interference(channel)
            - channel.get_bf_noise()
        )
    
    def get_spectral_efﬁciency(self, channel) -> float:
        """Get the spectral efﬁciency of the channel in bps/Hz."""
        return log10(1 + 10 ** (self.get_sinr(channel) / 10))


    def plot_network(self, plane="xy", annotate=False, ax=None):
        """Plot the network."""

        # loop over all nodes
        # plot their position (as 'O') as well as the downlink (as '->')

        plot_coords = {"xy": [0, 1], "yz": [1, 2], "xz": [0, 2]}[plane]

        if ax is None:
            fig, ax = plt.subplots()
        for node, value in self.nodes.items():
            # plot nodes
            node_loc = node.get_location()[plot_coords]
            # ax.scatter(*node_loc[plot_coords], "o", label=node.name)
            ax.scatter(
                *node_loc, s=50, facecolors="r", label=node.name
            )
            if annotate:
                ax.annotate(
                    node.name,
                    node_loc[plot_coords],
                )
            # plot downlink
            for dl, channel in value["dl"]:
                dl_loc = dl.get_location()[plot_coords]
                ax.plot(
                    *np.array([node_loc, dl_loc]).T,
                    "k:",
                )
                ax.plot(
                    *(dl_loc + (node_loc - dl_loc) / 5),
                    "kD",
                    label=dl.name,
                )
                if annotate:
                    offset = np.random.uniform(dl_loc - node_loc) * 0.1
                    ax.annotate(
                        channel.name,
                        (dl_loc + node_loc) / 2
                        + np.random.uniform(dl_loc - node_loc) * 0.1,
                    )
        plt.xlabel(f"{plane[0]}-axis")
        plt.ylabel(f"{plane[1]}-axis")
        plt.title(f"{self.name} Network")
        if ax is None:
            plt.show()
        return fig, ax

    def plot_network_3d(self, ax=None, annotate=False, **kwargs):
        """Plot the network in 3D."""
        if ax is None:
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, **kwargs)
        for node, value in self.nodes.items():
            # plot nodes
            node_loc = node.get_location()
            ax.scatter(
                *node_loc, s=50, facecolors="r", label=node.name
            )
            if annotate:
                ax.text(*node_loc, node.name)
            # plot downlink
            for dl, channel in value["dl"]:
                dl_loc = dl.get_location()
                ax.plot(
                    *np.array([node_loc, dl_loc]).T,
                    "k:",
                )
                ax.plot(
                    *(dl_loc + (node_loc - dl_loc) / 5),
                    "b*",
                    label=dl.name,
                )
                if annotate:
                    ax.text(*(dl_loc + node_loc) / 2, channel.name)
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_zlabel("Z-axis")
        ax.set_title(f"{self.name}")
        plt.tight_layout()
        if ax is None:
            plt.show()
        return fig, ax
