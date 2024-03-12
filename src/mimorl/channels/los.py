from .awgn import Channel
import numpy as np
from numpy import log10


class LosChannel(Channel):
    """Line-of-sight channel class.

    Unique Attributes
    -----------------
    aoa, aod: float, optional
        AoA/AoD. If not specified, the angles are
        calculated based on the relative position of the transmitter and receiver.
    """

    def __init__(self, tx, rx, normalize_channel_energy=True, *args, **kwargs):
        super().__init__(tx=tx, rx=rx, *args, **kwargs)
        self.normalize_channel_energy = normalize_channel_energy
        self._az = None
        self._el = None

    @property
    def aoa(self):
        return (self._az, self._el)

    @aoa.setter
    def aoa(self, _):
        raise Warning("Use realize() to set the AoA/AoD, ignoring the input.")

    aod = aoa

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
            az = az_new if az is None else az
            el = el_new if el is None else el
        self._az = az
        self._el = el
        tx_response = self.tx.get_array_response(self._az, self._el)
        rx_response = self.rx.get_array_response(self._az + np.pi, self._el + np.pi)
        # H = np.outer(tx_response, rx_response).T
        H = np.outer(rx_response, tx_response.conj())
        self.channel_matrix = H
        return self

