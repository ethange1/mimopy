import numpy as np
from .awgn import Channel
from .los import LoS
from .spherical_wave import SphericalWave
from .rayleigh import Rayleigh


class Rician(Channel):
    """Rician channel class.

    Unique Attributes
    ----------
        K (float): Rician K-factor.
        H_los (np.ndarray): Line-of-sight channel matrix.
        H_nlos (np.ndarray): Non-line-of-sight channel matrix.
    """

    def __init__(self, tx, rx, K=10, nearfield=False, *args, **kwargs):
        super().__init__(tx=tx, rx=rx, *args, **kwargs)
        if nearfield:
            self.los = SphericalWave(tx=self.tx, rx=self.rx)
        else:
            self.los = LoS(tx=self.tx, rx=self.rx)
        self.K = 10 ** (K / 10)  # Convert K-factor to linear scale
        self.nlos = Rayleigh(tx=self.tx, rx=self.rx)

    def realize(self, seed=None, energy=None):
        """Realize the channel. If random is True, the non-line-of-sight channel
        matrix is generated randomly."""
        if energy is None:
            energy = self.tx.N * self.rx.N
        np.random.seed(seed)
        self.los.realize(energy=energy)
        self.nlos.realize(seed=seed, energy=energy)
        self.channel_matrix = (
            np.sqrt(self.K / (self.K + 1)) * self.los.H
            + np.sqrt(1 / (self.K + 1)) * self.nlos.H
        )
        return self
