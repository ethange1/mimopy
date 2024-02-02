import numpy as np
# from ..array import Array

class BaseChannel:
    """Base class for channels.
    
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
        channel_energy_is_normalized (bool): Indicates if the channel energy is normalized.
    """

    def __init__(self, *args, **kwargs):
        self.name = "BaseChannel"
        self.channel_matrix = None
        self.tx_array = None
        self.rx_array = None
        self.num_antennas_tx = 0
        self.num_antennas_rx = 0
        self.propagation_velocity = 299792458
        self.carrier_frequency = 1e9
        self.carrier_wavelength = self.propagation_velocity / self.carrier_frequency
        self.normalized_channel_energy = 1
        self.channel_energy_is_normalized = False

    def __str__(self):
        return self.name

    def set_tx_array(self, tx_array):
        """Set the transmit array of the channel.

        Parameters
        ----------
            tx_array (Array): Transmit array.
        """
        self.tx_array = tx_array
        self.num_antennas_tx = tx_array.num_antennas

    def set_rx_array(self, rx_array):
        """Set the receive array of the channel.

        Parameters
        ----------
            rx_array (Array): Receive array.
        """
        self.rx_array = rx_array
        self.num_antennas_rx = rx_array.num_antennas

    def set_arrays(self, tx_array, rx_array):
        """Set the transmit and receive arrays of the channel.

        Parameters
        ----------
            tx_array (Array): Transmit array.
            rx_array (Array): Receive array.
        """
        self.set_tx_array(tx_array)
        self.set_rx_array(rx_array)

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

    def set_name(self, name):
        """Set the name of the channel.

        Parameters
        ----------
            name (str): Name of the channel.
        """
        self.name = name

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
    
    @classmethod
    def initialize(cls, tx_array, rx_array):
        pass