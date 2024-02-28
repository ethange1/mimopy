from typing import Any
from collections import abc
from collections.abc import Iterable

import numpy as np
from numpy import log10, log2

from .array import Array
from .channel import *
from matplotlib import pyplot as plt


class Network:
    """Network class.

    Attributes
    ----------
        name (str): Network name.
        nodes (List): List of nodes in the network.
        links (List): List of links in the network.
    """

    def __init__(self, name="Network", *args, **kwargs):
        self.name = name
        self.links = []
        self.nodes = dict()

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def add_nodes(self, node: Array):
        """Add a node to the network."""
        if node not in self.nodes:
            node.name += f"_{len(self.nodes)}"
            self.nodes[node] = {"dl": [], "ul": []}

    def _add_link(self, link: Channel):
        """Add a link to the network."""
        link.name += f"_{len(self.links)}"
        self.links.append(link)
        self.add_nodes(link.tx)
        self.nodes[link.tx]["dl"].append((link.rx, link))
        self.add_nodes(link.rx)
        self.nodes[link.rx]["ul"].append((link.tx, link))

    def add_links(self, links):
        """Add links to the network."""
        if isinstance(links, Iterable):
            for link in links:
                self._add_link(link)
        else:
            self._add_link(links)

    def _remove_node(self, node: Array):
        """Remove a node and all links associated with it from the network."""
        if node in self.nodes:
            for _, link in self.nodes[node]["dl"]:
                # the node is the tx; remove ul from link.rx
                self.links.remove(link)
                self.nodes[link.rx]["ul"].remove((node, link))
            for _, link in self.nodes[node]["ul"]:
                # the node is the rx; remove dl from link.tx
                self.links.remove(link)
                self.nodes[link.tx]["dl"].remove((node, link))

    def remove_nodes(self, nodes):
        """Remove nodes from the network."""
        if nodes.__iter__:
            for node in nodes:
                self._remove_node(node)
        else:
            self._remove_node(nodes)

    def _remove_link(self, link: Channel):
        """Remove a link from the network."""
        self.links.remove(link)
        self.nodes[link.tx]["dl"].remove((link.rx, link))
        self.nodes[link.rx]["ul"].remove((link.tx, link))

    def remove_links(self, links):
        """Remove links from the network."""
        if isinstance(links, Iterable):
            for link in links:
                self._remove_link(link)
        else:
            self._remove_link(links)

    # ===================================================================
    # Link measurement methods wrapper
    # ===================================================================
    def signal_power(self, link, linear=False) -> float:
        """Get the beamforming gain of the link in dB."""
        return link.signal_power_lin if linear else link.signal_power

    def bf_noise_power(self, link, linear=False) -> float:
        """Get the noise power after beamforming in dBm."""
        return link.bf_noise_power_lin if linear else link.bf_noise_power

    def snr(self, link, linear=False) -> float:
        """Get the signal-to-noise ratio (SNR) of the link."""
        return link.snr_lin if linear else link.snr

    # ===================================================================
    # Network measurement methods
    # ===================================================================
    def interference(self, link, linear=False) -> float:
        """Get the interference of the link."""
        # interference is the sum of bf gains of all other ul links of the rx
        interference_lin = 0
        for _, l in self.nodes[link.rx]["ul"]:
            if l != link:
                interference_lin += self.signal_power(l, linear=True)
        return interference_lin if linear else 10 * log10(interference_lin)

    def inr(self, link, linear=False) -> float:
        """Get the interference-to-noise ratio (INR) of the link in dB."""
        inr_lin = self.interference(link, linear=True) / self.bf_noise_power(
            link, linear=True
        )
        return inr_lin if linear else 10 * log10(inr_lin)

    def sinr(self, link, linear=False) -> float:
        """Get the signal-to-interference-plus-noise ratio (SINR) of the link in dB."""
        sinr_lin = self.signal_power(link, linear=True) / (
            self.interference(link, linear=True)
            + self.bf_noise_power(link, linear=True)
        )
        return sinr_lin if linear else 10 * log10(sinr_lin)

    def spectral_efﬁciency(self, link) -> float:
        """Get the spectral efﬁciency of the link in bps/Hz."""
        return float(log10(1 + 10 ** (self.get_sinr(link) / 10)))

    # ===================================================================
    # Plotting methods
    # ===================================================================
    def plot(self, plane="xy", show_label=False, ax=None, **kwargs):
        """Plot the network."""
        coord_idx = {"xy": [0, 1], "yz": [1, 2], "xz": [0, 2]}[plane]
        if ax is None:
            fig, ax = plt.subplots(**kwargs)
        for node, value in self.nodes.items():
            # plot nodes
            node_loc = node.location[coord_idx]
            # ax.scatter(*node_loc[coord_idx], "o", label=node.name)
            ax.scatter(*node_loc, s=70, facecolors="k", label=node.name)
            if show_label:
                ax.annotate(
                    node.name,
                    node_loc[coord_idx],
                )
            # plot downlink
            for dl, link in value["dl"]:
                dl_loc = dl.location[coord_idx]
                ax.plot(
                    *np.array([node_loc, dl_loc]).T,
                    "k:",
                )
                ax.plot(
                    *(dl_loc + (node_loc - dl_loc) / 5),
                    "m*",
                    label=dl.name,
                )
                if show_label:
                    offset = np.random.uniform(dl_loc - node_loc) * 0.1
                    ax.annotate(
                        link.name,
                        (dl_loc + node_loc) / 2
                        + np.random.uniform(dl_loc - node_loc) * 0.1,
                    )
        plt.xlabel(f"{plane[0]}-axis")
        plt.ylabel(f"{plane[1]}-axis")
        plt.title(f"{self.name}")
        if ax is None:
            plt.show()
        return fig, ax

    def plot_3d(self, ax=None, show_label=False, **kwargs):
        """Plot the network in 3D."""
        if ax is None:
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, **kwargs)
        for node, value in self.nodes.items():
            # plot nodes
            node_loc = node.location
            ax.scatter(*node_loc, s=70, facecolors="k", label=node.name)
            if show_label:
                ax.text(*node_loc, node.name)
            # plot downlink
            for dl, link in value["dl"]:
                dl_loc = dl.location
                ax.plot(
                    *np.array([node_loc, dl_loc]).T,
                    "k:",
                )
                ax.plot(
                    *(dl_loc + (node_loc - dl_loc) / 5),
                    "m*",
                    label=dl.name,
                )
                if show_label:
                    ax.text(*(dl_loc + node_loc) / 2, link.name)
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_zlabel("Z-axis")
        ax.set_title(f"{self.name}")
        plt.tight_layout()
        if ax is None:
            plt.show()
        return fig, ax