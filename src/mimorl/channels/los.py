from .awgn import Channel
import numpy as np
from numpy import log10


class LosChannel(Channel):
    """Line-of-sight channel class.

    Unique Attributes
    -----------------
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_channel_energy_normalized = True
        self.realize()

    # @property
    # def bf_noise_lin(self):
    #     """Noise power after beamforming in linear scale."""
    #     w = self.rx.weights.reshape(1, -1)
    #     return np.linalg.norm(w) ** 2 * self.noise_power_lin

    # @property
    # def bf_noise(self):
    #     """Noise power after beamforming in dBm."""
    #     w = self.rx.weights.reshape(1, -1)
    #     return 10 * log10(np.linalg.norm(w) ** 2) + self.noise_power

    # def get_bf_noise(self, linear=False) -> float:
    #     """Get the noise power after beamforming in dBm."""
    #     w = self.rx.get_weights().reshape(1, -1)
    #     if linear:
    #         return np.linalg.norm(w) ** 2 * self.noise_power_lin
    #     return 10 * log10(np.linalg.norm(w) ** 2) + self.noise_power

    # def get_snr(self, linear=False) -> float:
    #     """Get the signal-to-noise ratio (SNR) of the channel. """
    #     if linear:
    #        return self.get_bf_gain(linear) / self.get_bf_noise(linear)
    #     return self.get_bf_gain() - self.get_bf_noise()

    def realize(self, az=None, el=None):
        """Realize the channel.
        
        Parameters
        ----------
        az, el: float, optional
            AoA/AoD. If not specified, the angles are
            calculated based on the relative position of the transmitter and receiver.
        """
        if az is None or el is None:
            _, az_new, el_new = self.get_relative_position(
                self.tx.array_center, self.rx.array_center
            )
            if az is None:
                az = az_new
            if el is None:
                el = el_new
        self.set_angular_separation()
        tx_response = self.tx.get_array_response(self.az, self.el)
        rx_response = self.rx.get_array_response(self.az + np.pi, self.el + np.pi)
        # H = np.outer(tx_response, rx_response).T
        H = np.outer(rx_response, tx_response.conj())
        self.channel_matrix = H

    @staticmethod
    def get_relative_position(loc1, loc2):
        """Returns the relative position (range, azimuth and elevation) between 2 locations.

        Parameters
        ----------
            loc1, loc2: array_like, shape (3,)
                Location of the 2 points.
        """
        loc1 = np.asarray(loc1).reshape(3)
        loc2 = np.asarray(loc2).reshape(3)
        dxyz = dx, dy, dz = loc2 - loc1
        range = np.linalg.norm(dxyz)
        az = np.arctan2(dy, dx)
        el = np.arcsin(dz / range)
        return range, az, el