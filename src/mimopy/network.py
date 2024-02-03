from typing import Any
import numpy as np
from numpy import log10

from array import Array
from channel import *


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

    def add_node(self, node:Array):
        """Add a node to the network."""
        if node not in self.nodes:
            node.name = f"Node{len(self.nodes)}"
            self.nodes[node] = {"dl": [], "ul": []}

    def add_channel(self, channel):
        """Add a channel to the network."""
        channel.name += f"{len(self.channels)}"
        self.channels.append(channel)
        self.add_node(channel.tx)
        self.nodes[channel.tx]["dl"].append((channel.rx, channel))
        self.add_node(channel.rx)
        self.nodes[channel.rx]["ul"].append((channel.tx, channel))

    def remove_node(self, node:Array):
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

    def get_sinr(self, node:Array) -> float:
        """Get the signal-to-interference-plus-noise ratio (SINR) of a node in the network."""
        