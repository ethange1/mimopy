from typing import Any
import numpy as np
from numpy import log10, log2


# from ..array import Array
class Channel:
    """Base class for Channel.

    Attributes
    ----------
        name (str): Channel name. tx (Array): Transmit array. rx (Array):
        Receive array. noise_power (float): Noise power in dBm.
        propagation_velocity (float): Propagation velocity in meters per second.
        carrier_frequency (float): Carrier frequency in Hertz.
        carrier_wavelength (float): Carrier wavelength in meters.
        normalized_channel_energy (float): Normalized channel energy. If False
        then the channel matrix is not normalized.
    """

    def __init__(self, tx=None, rx=None, name=None, *args, **kwargs):
        # use class name as default name
        self.name = name
        if name is None:
            self.name = self.__class__.__name__
        self.tx = tx
        self.rx = rx
        self._channel_matrix = None
        self.noise_power = 0
        self._carrier_frequency = 1e9
        self._propagation_velocity = 299792458
        self._carrier_wavelength = self.propagation_velocity / self.carrier_frequency

        self.normalized_channel_energy = False

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

    # ========================================================
    # Channel matrix
    # ========================================================

    def realize(self):
        """Realize the channel."""
        raise NotImplementedError

    @property
    def channel_matrix(self):
        """Channel matrix, setting the channel matrix will normalize the channel
        energy."""
        return self._channel_matrix

    @channel_matrix.setter
    def channel_matrix(self, channel_matrix):
        if self.normalized_channel_energy:
            self._channel_matrix = (
                channel_matrix
                * np.sqrt(self.normalized_channel_energy)
                / np.linalg.norm(channel_matrix, ord="fro")
            )
        else:
            self._channel_matrix = channel_matrix

    # ========================================================
    # Measurements
    # ========================================================

    @property
    def noise_power_lin(self):
        return 10 ** (self.noise_power / 10)

    @noise_power_lin.setter
    def noise_power_lin(self, noise_power_lin):
        self.noise_power = 10 * log10(noise_power_lin)

    @property
    def bf_gain_lin(self) -> float:
        """Beamforming gain in linear scale."""
        f = self.tx.get_weights().reshape(-1, 1)
        H = self.channel_matrix
        w = self.rx.get_weights().reshape(-1, 1)
        return float(np.abs(w.conj().T @ H @ f) ** 2)
    
    @property
    def bf_gain(self) -> float:
        """Beamforming gain in dB."""
        return 10 * log10(self.bf_gain_lin)

    @property
    def signal_power_lin(self) -> float:
        """Signal power after beamforming in linear scale."""
        return self.tx.power * self.bf_gain_lin

    @property
    def signal_power(self) -> float:
        """Signal power after beamforming in dBm."""
        return 10 * log10(self.signal_power_lin)

    @property
    def bf_noise_power_lin(self):
        """Noise power after beamforming in linear scale."""
        w = self.rx.get_weights().reshape(1, -1)
        return float(np.linalg.norm(w) ** 2 * self.noise_power_lin)

    @property
    def bf_noise_power(self) -> float:
        """Noise power after beamforming in dBm."""
        return 10 * log10(self.bf_noise_power_lin)

    @property
    def snr_lin(self) -> float:
        """Signal-to-noise ratio (SNR) in linear scale."""
        return float(self.signal_power_lin / self.bf_noise_power_lin)

    @property
    def snr(self) -> float:
        """Signal-to-noise ratio (SNR) in dB."""
        return 10 * log10(self.snr_lin)

    @property
    def capacity(self) -> float:
        """Channel capacity in bps/Hz."""
        return log2(1 + self.snr_lin)

    # ========================================================
    # Skip Setters
    # ========================================================

    @signal_power_lin.setter
    def signal_power_lin(self, _):
        self._cant_be_set()

    @signal_power.setter
    def signal_power(self, _):
        self._cant_be_set()

    @bf_noise_power_lin.setter
    def bf_noise_power_lin(self, _):
        self._cant_be_set()

    @bf_noise_power.setter
    def bf_noise_power(self, _):
        self._cant_be_set()

    @snr_lin.setter
    def snr_lin(self, _):
        self._cant_be_set()

    @snr.setter
    def snr(self, _):
        self._cant_be_set()

    @capacity.setter
    def capacity(self, _):
        self._cant_be_set()

    @staticmethod
    def _cant_be_set():
        # raise warning
        raise Warning("This property can't be set, skipping...")

    # def get_bf_gain(self, linear=False) -> float:
    #     """Get the beamforming gain of the channel in dBm."""
    #     f = self.tx.get_weights().reshape(-1, 1)
    #     H = self.get_channel_matrix()
    #     w = self.rx.get_weights().reshape(-1, 1)
    #     P = self.tx.power
    #     gain = P * np.abs(w.conj().T @ H @ f) ** 2
    #     if linear:
    #         return gain
    #     return 10 * log10(gain)

    # ========================================================
    # Physical properties
    # ========================================================
    @property
    def carrier_frequency(self):
        """Carrier frequency in Hertz.
        Also update carrier wavelength when set."""
        return self._carrier_frequency

    @carrier_frequency.setter
    def carrier_frequency(self, carrier_frequency):
        self._carrier_frequency = carrier_frequency
        self._carrier_wavelength = self.propagation_velocity / carrier_frequency

    @property
    def propagation_velocity(self):
        """Propagation velocity in meters per second.
        Also update carrier wavelength when set."""
        return self._propagation_velocity

    @propagation_velocity.setter
    def propagation_velocity(self, propagation_velocity):
        self._propagation_velocity = propagation_velocity
        self._carrier_wavelength = propagation_velocity / self.carrier_frequency

    @property
    def carrier_wavelength(self):
        """Carrier wavelength in meters.
        Also update carrier frequency when set."""
        return self._carrier_wavelength

    @carrier_wavelength.setter
    def carrier_wavelength(self, carrier_wavelength):
        self._carrier_wavelength = carrier_wavelength
        self._carrier_frequency = self.propagation_velocity / carrier_wavelength