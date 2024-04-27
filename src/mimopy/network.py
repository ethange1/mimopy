from typing import Any
from collections.abc import Iterable

import numpy as np
from numpy import log10

from .antenna_array import AntennaArray
from .channels import Channel
import matplotlib.pyplot as plt


class Network:
    """Network class.

    Attributes
    ----------
        name (str): Network name.
        links (list): List of links in the network.
        nodes (tuple): Tuple of nodes in the network.
        connections (dict): Dictionary of connections in the network.
        target_links (list): List of target links in the network.
        target_nodes (list): List of target nodes in the network.
    """

    def __init__(self, name="Network", *args, **kwargs):
        self.name = name
        self.links = {}
        self.connections = {}
        self.target_links = []
        self.target_nodes = []

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    # ===================================================================
    # Links and Nodes
    # ===================================================================

    @property
    def nodes(self: Iterable[AntennaArray]):
        return {key.name: key for key in self.connections.keys()}

    @nodes.setter
    def nodes(self, _):
        raise AttributeError("Cannot set nodes directly. Use add_nodes() instead.")

    n = nodes
    l = property(lambda self: self.links)
    topology = property(lambda self: self.connections)

    def _add_node(self, node: AntennaArray):
        """Add a node to the network."""
        if node not in self.connections:
            self.connections[node] = {"dl": [], "ul": []}

    def add_nodes(self, nodes: Iterable[AntennaArray]):
        """Add nodes to the network."""
        if isinstance(nodes, Iterable):
            for node in nodes:
                self._add_node(node)
        else:
            self._add_node(nodes)

    def _add_link(self, link: Channel):
        """Add a link to the network."""
        # link.name = f"{len(self.links)}_" + link.name
        if link.name not in self.links and link not in self.links.values():
            self.links[link.name] = link
            self.add_nodes(link.tx)
            self.connections[link.tx]["dl"].append((link.rx, link))
            self.add_nodes(link.rx)
            self.connections[link.rx]["ul"].append((link.tx, link))

    def add_links(self, links: Iterable[Channel]):
        """Add links to the network."""
        if isinstance(links, Iterable):
            for link in links:
                self._add_link(link)
        else:
            self._add_link(links)

    def _remove_node(self, node: AntennaArray):
        """Remove a node and all links associated with it from the network."""
        if node in self.connections:
            for _, link in self.connections[node]["dl"]:
                # the node is the tx; remove ul from link.rx
                self.links.pop(link, None)
                self.connections[link.rx]["ul"].remove((node, link))
            for _, link in self.connections[node]["ul"]:
                # the node is the rx; remove dl from link.tx
                self.links.pop(link, None)
                self.connections[link.tx]["dl"].remove((node, link))

    def remove_nodes(self, nodes):
        """Remove nodes from the network."""
        if nodes.__iter__:
            for node in nodes:
                self._remove_node(node)
        else:
            self._remove_node(nodes)

    def _remove_link(self, link: Channel | str):
        """Remove a link from the network."""
        if isinstance(link, str):
            link = self.links[link]
        self.links.pop(link.name, None)
        self.connections[link.tx]["dl"].remove((link.rx, link))
        self.connections[link.rx]["ul"].remove((link.tx, link))

    def remove_links(self, links):
        """Remove links from the network."""
        if isinstance(links, Iterable):
            for link in links:
                self._remove_link(link)
        else:
            self._remove_link(links)

    # ===================================================================
    # Target Links and Nodes
    # ===================================================================

    @property
    def target_weights(self: Channel):
        return [node.weights for node in self.target_nodes]

    @target_weights.setter
    def target_weights(self, _):
        raise AttributeError(
            "Cannot set target_weights directly. Use add_target_nodes() instead."
        )

    def add_target_nodes(self, nodes):
        """Add target nodes to the network."""
        if isinstance(nodes, Iterable):
            for node in nodes:
                self._add_node(node)
                self.target_nodes.append(node)
        else:
            self._add_node(nodes)
            self.target_nodes.append(nodes)

    def remove_target_nodes(self, nodes):
        """Remove target nodes from the network."""
        if isinstance(nodes, Iterable):
            for node in nodes:
                self.target_nodes.remove(node)
        else:
            self.target_nodes.remove(nodes)

    def add_target_links(self, links):
        """Add target links to the network."""
        if isinstance(links, Iterable):
            for link in links:
                self._add_link(link)
                self.target_links.append(link)
        else:
            self._add_link(links)
            self.target_links.append(links)

    def remove_target_links(self, links):
        """Remove target links from the network."""
        if isinstance(links, Iterable):
            for link in links:
                self.target_links.remove(link)
        else:
            self.target_links.remove(links)

    def _is_target_link(self, link: Channel):
        return link in self.target_links

    def _is_target_node(self, node: AntennaArray):
        return node in self.target_nodes

    def is_target(self, obj: Any):
        if isinstance(obj, Channel):
            return self._is_target_link(obj)
        elif isinstance(obj, AntennaArray):
            return self._is_target_node(obj)
        else:
            raise TypeError(f"Object of type {type(obj)} is not supported.")

    # ===================================================================
    # Link measurement methods wrapper
    # ===================================================================

    def bf_gain(self, link: Channel | str = None, linear=False) -> float:
        """Get the beamforming gain of the link in dB."""
        if link is None:
            return {
                link: self.bf_gain(link, linear=linear) for link in self.links.values()
            }
        if isinstance(link, str):
            link = self.links[link]
        if isinstance(link, Iterable):
            return {link: self.snr(link, linear=linear) for link in link}
        return link.bf_gain_lin if linear else link.bf_gain

    gain = bf_gain

    def signal_power(self, link: Channel | str = None, linear=False) -> float:
        """Get the beamforming gain of the link in dB."""
        if link is None:
            return {
                link: self.signal_power(link, linear=linear)
                for link in self.links.values()
            }
        if isinstance(link, str):
            link = self.links[link]
        if isinstance(link, Iterable):
            return {link: self.snr(link, linear=linear) for link in link}
        return link.signal_power_lin if linear else link.signal_power

    def bf_noise_power(self, link: Channel | str = None, linear=False) -> float:
        """Get the noise power after beamforming in dBm."""
        if link is None:
            return {
                link: self.bf_noise_power(link, linear=linear)
                for link in self.links.values()
            }
        if isinstance(link, str):
            link = self.links[link]
        if isinstance(link, Iterable):
            return {link: self.snr(link, linear=linear) for link in link}
        return link.bf_noise_power_lin if linear else link.bf_noise_power

    def snr(self, link: Channel | str = None, linear=False) -> float:
        """Get the signal-to-noise ratio (SNR) of the link."""
        if link is None:
            return {link: self.snr(link, linear=linear) for link in self.links.values()}
        if isinstance(link, Iterable):
            return {link: self.snr(link, linear=linear) for link in link}
        if isinstance(link, str):
            link = self.links[link]
        return link.snr_lin if linear else link.snr

    # ===================================================================
    # Network measurement methods
    # ===================================================================
    def interference(self, link=None, linear=False) -> float:
        """Get the interference of the link."""
        # interference is the sum of bf gains of all other ul links of the rx
        if link is None:
            return {
                link: self.interference(link, linear=linear)
                for link in self.links.values()
            }
        if isinstance(link, str):
            link = self.links[link]
        interference_lin = 0
        for _, l in self.connections[link.rx]["ul"]:
            if l != link:
                interference_lin += self.signal_power(l, linear=True)
        return (
            interference_lin
            if linear
            else 10 * log10(interference_lin + np.finfo(float).tiny)
        )

    def inr(self, link=None, linear=False) -> float:
        """Get the interference-to-noise ratio (INR) of the link in dB."""
        if link is None:
            return {link: self.inr(link, linear=linear) for link in self.links.values()}
        if isinstance(link, str):
            link = self.links[link]
        inr_lin = self.interference(link, linear=True) / self.bf_noise_power(
            link, linear=True
        )
        return inr_lin if linear else 10 * log10(inr_lin + np.finfo(float).tiny)

    def sinr(self, link=None, linear=False) -> float:
        """Get the signal-to-interference-plus-noise ratio (SINR) of the link in dB."""
        if link is None:
            return {
                link: self.sinr(link, linear=linear) for link in self.links.values()
            }
        if isinstance(link, str):
            link = self.links[link]
        sinr_lin = self.signal_power(link, linear=True) / (
            self.interference(link, linear=True)
            + self.bf_noise_power(link, linear=True)
        )
        return sinr_lin if linear else 10 * log10(sinr_lin + np.finfo(float).tiny)

    def spectral_efﬁciency(self, link=None) -> float:
        """Get the spectral efﬁciency of the link in bps/Hz."""
        if link is None:
            return {link: self.spectral_efﬁciency(link) for link in self.links.values()}
        if isinstance(link, str):
            link = self.links[link]
        return float(log10(1 + 10 ** (self.sinr(link) / 10)))

    # ===================================================================
    # Plotting methods
    # ===================================================================
    def plot(self, labels=False, plane="xy", ax=None, **kwargs):
        """Plot the network."""
        coord_idx = {"xy": [0, 1], "yz": [1, 2], "xz": [0, 2]}[plane]
        if ax is None:
            _, ax = plt.subplots(**kwargs)
        for node, value in self.connections.items():
            # plot nodes
            node_loc = node.location[coord_idx]
            # ax.scatter(*node_loc[coord_idx], "o", label=node.name)
            style = "b" if self.is_target(node) else "k"
            ax.scatter(*node_loc, s=70, facecolors=style, label=node.name)
            if labels:
                ax.annotate(
                    node.name,
                    node_loc[coord_idx],
                )
            # plot downlink
            for dl, link in value["dl"]:
                dl_loc = dl.location[coord_idx]
                style = "c-" if self.is_target(link) else "k:"
                ax.plot(
                    *np.array([node_loc, dl_loc]).T,
                    style,
                )
                ax.plot(
                    *(dl_loc + (node_loc - dl_loc) / 5),
                    "m*",
                    label=dl.name,
                )
                if labels:
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

    def plot_3d(self, ax=None, labels=False, **kwargs):
        """Plot the network in 3D."""
        if ax is None:
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, **kwargs)
        for node, value in self.connections.items():
            # plot nodes
            node_loc = node.location
            style = "b" if self.is_target(node) else "k"
            ax.scatter(*node_loc, s=70, facecolors=style, label=node.name)
            if labels:
                ax.text(*node_loc, node.name)
            # plot downlink
            for dl, link in value["dl"]:
                dl_loc = dl.location
                style = "c-" if self.is_target(link) else "k:"
                ax.plot(
                    *np.array([node_loc, dl_loc]).T,
                    style,
                )
                ax.plot(
                    *(dl_loc + (node_loc - dl_loc) / 5),
                    "m*",
                    label=dl.name,
                )
                if labels:
                    ax.text(*(dl_loc + node_loc) / 2, link.name)
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_zlabel("Z-axis")
        ax.set_title(f"{self.name}")
        plt.tight_layout()
        if ax is None:
            plt.show()
        return fig, ax

    def plot_gain(self, polar=True, axes=None, weights=None, ylim=-20, **kwargs):
        """Plot the beam pattern of the controlled nodes."""
        num_plots = len(self.target_nodes)
        num_cols = np.ceil(np.sqrt(num_plots)).astype(int)
        num_rows = np.ceil(num_plots / num_cols).astype(int)
        if "figsize" not in kwargs:
            kwargs["figsize"] = (5 * num_cols, 5 * num_rows)
        if axes is None:
            if polar:
                fig, axes = plt.subplots(
                    num_rows, num_cols, subplot_kw={"polar": True}, **kwargs
                )
            else:
                fig, axes = plt.subplots(num_rows, num_cols, **kwargs)
        for i, (node, ax) in enumerate(zip(self.target_nodes, np.ravel(axes))):
            if weights is not None:
                node.plot_gain(ax=ax, weights=weights[i], polar=polar)
            else:
                node.plot_gain(ax=ax, polar=polar)
            title = ax.get_title()
            ax.set_title(f"{node.name}: {title}")
        if polar:
            for ax in axes:
                ax.set_ylim(bottom=ylim)
                ax.set_theta_zero_location("E")
                ax.set_theta_direction(1)
        if axes is None:
            plt.tight_layout()
            plt.show()
