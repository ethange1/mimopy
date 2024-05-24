from .awgn import Channel
import numpy as np
from .path_loss import get_path_loss
from ..utils.geometry import get_relative_position


class LoS(Channel):
    """Line-of-sight channel class.

    Unique Attributes
    -----------------
    aoa, aod: float, optional
        AoA/AoD. If not specified, the angles are
        calculated based on the relative position of the transmitter and receiver.
    """

    def __init__(self, tx, rx, path_loss="no_loss", *args, **kwargs):
        super().__init__(tx, rx, path_loss, *args, **kwargs)

    @property
    def aoa(self):
        return get_relative_position(self.tx.array_center, self.rx.array_center)

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
        az, el = get_relative_position(self.tx.array_center, self.rx.array_center)
        tx_response = self.tx.get_array_response(az, el)
        rx_response = self.rx.get_array_response(az + np.pi, el + np.pi)
        # H = np.outer(tx_response, rx_response).T
        self.H = np.outer(rx_response, tx_response.conj())
        self.normalize_energy(self.energy)
        return self
