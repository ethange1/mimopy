from typing import Any
import numpy as np
from numpy import log10

# from ..array import Array


class Channel:
    """Base class for channels.

    Attributes
    ----------
        name (str): Channel name.
        tx (Array): Transmit array.
        rx (Array): Receive array.
        noise_power (float): Noise power in dBm.
        propagation_velocity (float): Propagation velocity in meters per second.
        carrier_frequency (float): Carrier frequency in Hertz.
        carrier_wavelength (float): Carrier wavelength in meters.
        normalized_channel_energy (float): Normalized channel energy.
        channel_energy_is_normalized (bool): Indicates if the channel energy is normalized.
    """

    def __init__(self, tx=None, rx=None, name = None, *args, **kwargs):
        # use class name as default name
        self.name = name
        if name is None:
            self.name = self.__class__.__name__
        self.tx = tx
        self.rx = rx
        self.channel_matrix = None
        self.noise_power = 0
        self.propagation_velocity = 299792458
        self.carrier_frequency = 1e9
        self.carrier_wavelength = self.propagation_velocity / self.carrier_frequency

        self.normalized_channel_energy = 1
        self.channel_energy_is_normalized = False

        # for key, value in kwargs.items():
        #     setattr(self, key, value)
        # if tx is not None and rx is not None:
        #     self.realize()

    def __str__(self):
        return self.name
    
    def __repr__(self):
        return self.name
    
    @property
    def ul(self):
        return self.tx
    
    @ul.setter
    def ul(self, tx):
        self.tx = tx
    
    @property
    def dl(self):
        return self.rx
    
    @dl.setter
    def dl(self, rx):
        self.rx = rx

    @property
    def noise_power_lin(self):
        return 10 ** (self.noise_power / 10)
    
    @noise_power_lin.setter
    def noise_power_lin(self, noise_power_lin):
        self.noise_power = 10 * log10(noise_power_lin)

    def set_arrays(self, tx, rx):
        """Set the transmit and receive arrays of the channel.

        Parameters
        ----------
            tx (Array): Transmit array.
            rx (Array): Receive array.
        """
        self.set_tx(tx)
        self.set_rx(rx)

    def set_carrier_frequency(self, carrier_frequency):
        """Set the carrier frequency of the channel.

        Parameters
        ----------
            carrier_frequency (float): Carrier frequency in Hertz.
        """
        self.carrier_frequency = carrier_frequency
        self.carrier_wavelength = self.propagation_velocity / carrier_frequency

    def set_carrier_wavelegnth(self, carrier_wavelength):
        """Set the carrier wavelength of the channel.

        Parameters
        ----------
            carrier_wavelength (float): Carrier wavelength in meters.

        Note: This method also updates the carrier frequency but not the propagation velocity.
        """
        self.carrier_wavelength = carrier_wavelength
        self.carrier_frequency = self.propagation_velocity / carrier_wavelength

    def set_propagation_velocity(self, propagation_velocity):
        """Set the propagation velocity of the channel.

        Parameters
        ----------
            propagation_velocity (float): Propagation velocity in meters per second.

        Note: This method also updates the carrier wavelength but not the carrier frequency.
        """
        self.propagation_velocity = propagation_velocity
        self.carrier_wavelength = propagation_velocity / self.carrier_frequency

    def _normalize_channel_energy(self, channel_matrix):
        """Normalize the power of the channel matrix to the normalized transmit power.

        Parameters
        ----------
            channel_matrix (np.ndarray): Channel matrix.

        Returns
        -------
            np.ndarray: Normalized channel matrix.
        """
        frobenius_norm = np.linalg.norm(channel_matrix, ord="fro")
        return channel_matrix * np.sqrt(self.normalized_channel_energy) / frobenius_norm

    def set_channel_matrix(self, channel_matrix):
        """Set the channel matrix of the channel.

        Parameters
        ----------
            channel_matrix (np.ndarray): Channel matrix.
        """
        if self.channel_energy_is_normalized:
            self.channel_matrix = self._normalize_channel_energy(channel_matrix)
        else:
            self.channel_matrix = channel_matrix

    def get_channel_matrix(self):
        """Get the channel matrix of the channel.

        Returns
        -------
            np.ndarray: Channel matrix.
        """
        return self.channel_matrix
    
    def get_noise_power(self, dBm=True):
        """Get the noise power of the channel.

        Parameters
        ----------
            dBm (bool): If True, returns the noise power in dBm. Otherwise, returns the noise power in linear scale.

        Returns
        -------
            float: Noise power.
        """
        if dBm:
            return self.noise_power
        else:
            return self.noise_power_lin

    def get_bf_gain(self) -> float:
        """Get the beamforming gain of the channel  in dBm."""
        f = self.tx.get_weights().reshape(-1, 1)
        H = self.get_channel_matrix()
        w = self.rx.get_weights().reshape(-1, 1)
        P = self.tx.power
        bf_gain = 10 * log10(P * np.abs(w.conj().T @ H @ f) ** 2)
        return float(bf_gain)

    def realize(self):
        """Realize the channel."""
        raise NotImplementedError
