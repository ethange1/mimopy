from .base import Channel
import numpy as np
from numpy import log10


class LosChannel(Channel):
    """Line-of-sight channel class.

    Attributes
    ----------
        name (str): Channel name.
        tx (Array): Transmit array.
        rx (Array): Receive array.
        num_antennas_tx (int): Number of transmit antennas.
        num_antennas_rx (int): Number of receive antennas.
        propagation_velocity (float): Propagation velocity in meters per second.
        carrier_frequency (float): Carrier frequency in Hertz.
        carrier_wavelength (float): Carrier wavelength in meters.
        normalized_channel_energy (float): Normalized channel energy.
        is_channel_energy_normalized (bool): Indicates if the channel energy is normalized.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.az = 0
        self.el = 0
        self.is_channel_energy_normalized = True
        self.realize()

    def __str__(self):
        return f"{self.name}"

    @property
    def bf_noise(self):
        """Noise power after beamforming in dBm."""
        w = self.rx.weights.reshape(1, -1)
        return 10 * log10(np.linalg.norm(w) ** 2) + self.noise_power
    
    @property
    def snr(self):
        """Signal-to-noise ratio (SNR) of the channel."""
        return self.bf_gain - self.bf_noise

    def get_bf_noise(self) -> float:
        """Get the noise power after beamforming in dBm."""
        w = self.rx.get_weights().reshape(1, -1)
        return 10 * log10(np.linalg.norm(w) ** 2) + self.noise_power

    def get_snr(self) -> float:
        """Get the signal-to-noise ratio (SNR) of the channel. """
        return self.get_bf_gain() - self.get_bf_noise()

    def realize(self):
        """Realize the channel."""
        self.set_angular_separation()
        tx_response = self.tx.get_array_response(self.az, self.el)
        rx_response = self.rx.get_array_response(self.az + np.pi, self.el + np.pi)
        # H = np.outer(tx_response, rx_response).T
        H = np.outer(rx_response, tx_response.conj())
        self.set_channel_matrix(H)

    @staticmethod
    def _get_relative_position(loc1, loc2):
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

    def set_angular_separation(self):
        """Set angular separation between transmit and receive arrays.

        Parameters
        ----------
            ax: Float, Optional
                Azimuth angle in degrees depending the transmitter
            el: Float, Optional
                Elevation angle in degrees depending the transmitter
        """

        atx_center = self.tx.array_center
        arx_center = self.rx.array_center
        _, az, el = self._get_relative_position(atx_center, arx_center)
        self.az = az
        self.el = el