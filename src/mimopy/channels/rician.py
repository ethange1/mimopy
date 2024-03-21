import numpy as np
from .awgn import Channel
from .los import LoS
from .spherical_wave import SphericalWave


class Rician(Channel):
    """Rician channel class.

    Unique Attributes
    ----------
        K (float): Rician K-factor.
        H_los (np.ndarray): Line-of-sight channel matrix.
        H_nlos (np.ndarray): Non-line-of-sight channel matrix.
    """

    def __init__(self, tx, rx, K=10, nearfield=False, los=None, *args, **kwargs):
        super().__init__(tx=tx, rx=rx, *args, **kwargs)
        if los is None:
            if nearfield:
                self.los = SphericalWave(tx=self.tx, rx=self.rx).realize()
            else:
                self.los = LoS(tx=self.tx, rx=self.rx).realize()
        else:
            self.los = los
        self.K = 10 ** (K / 10) # Convert K-factor to linear scale
        self.H_nlos = None

    def realize(self, seed=None):
        """Realize the channel. If random is True, the non-line-of-sight channel
        matrix is generated randomly."""
        self.H_los = self.los.channel_matrix
        # print("Generating random NLOS channel matrix.")
        np.random.seed(seed)
        self.H_nlos = np.sqrt(1 / 2) * (
            np.random.randn(*self.H_los.shape) + 1j * np.random.randn(*self.H_los.shape)
        )
        self.channel_matrix = (
            np.sqrt(self.K / (self.K + 1)) * self.H_los
            + np.sqrt(1 / (self.K + 1)) * self.H_nlos
        )
        return self
