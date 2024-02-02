from .base import BaseChannel
import numpy as np


class LoSChannel(BaseChannel):
    """Line-of-sight channel class.

    Attributes
    ----------
        name (str): Channel name.
        tx_array (Array): Transmit array.
        rx_array (Array): Receive array.
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
        self.name = "LoSChannel"
        self.az = 0
        self.el = 0
        # self.noise_power = 1e-3
        self.is_channel_energy_normalized = True

    @classmethod
    def initialize(cls, tx_array, rx_array):
        """Initialize the channel.

        Parameters
        ----------
            tx_array (Array): Transmit array.
            rx_array (Array): Receive array.
        """
        channel = cls()
        channel.set_arrays(tx_array, rx_array)
        channel.realize_channel()
        return channel

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

    def set_angular_separation(self, az=0, el=0):
        """Set angular separation between transmit and receive arrays.

        Parameters
        ----------
            ax: Float, Optional
                Azimuth angle in degrees depending the transmitter
            el: Float, Optional
                Elevation angle in degrees depending the transmitter
        """

        atx_center = self.tx_array.get_array_center()
        arx_center = self.rx_array.get_array_center()
        _, az, el = self._get_relative_position(atx_center, arx_center)
        self.az = az
        self.el = el

    # def set_noise_power(self, noise_power):
    #     """Set the noise power.

    #     Parameters
    #     ----------
    #         noise_power: Float
    #             Noise power.
    #     """
    #     self.noise_power = noise_power

    # def get_noise_power(self):
    #     """Get the noise power."""
    #     return self.noise_power

    def realize_channel(self):
        """Realize the channel.

        Parameters
        ----------
            *args, **kwargs: Arguments and keyword arguments to be passed to the channel realization method.
        """
        self.set_angular_separation()
        tx_response = self.tx_array.get_array_response(self.az, self.el)
        rx_response = self.rx_array.get_array_response(self.az + np.pi, self.el + np.pi)
        H = np.outer(tx_response, rx_response)
        self.set_channel_matrix(H)
