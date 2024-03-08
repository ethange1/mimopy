import numpy as np
from .awgn import Channel
from .los import LosChannel


class RicianChannel(Channel):
    """Rician channel class.

    Unique Attributes
    ----------
        K (float): Rician K-factor.
    """

    def __init__(self, K=10, los=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if los is None:
            self.los = LosChannel(tx=self.tx, rx=self.rx)
        else:
            self.los = los
        self.K = K

    def realize(self):
        """Realize the channel."""
        h_los = self.los.channel_matrix
        h_nlos = np.sqrt(1 / 2) * (
            np.random.randn(*h_los.shape) + 1j * np.random.randn(*h_los.shape)
        )
        self.channel_matrix = (
            np.sqrt(self.K / (self.K + 1)) * h_los + np.sqrt(1 / (self.K + 1)) * h_nlos
        )