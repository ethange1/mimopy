import numpy as np
from .awgn import Channel


class Rayleigh(Channel):
    """Rayleigh channel class.

    Unique Attributes
    -----------------
    seed: int, optional
        Seed for random number generator.
    """

    def __init__(self, tx, rx, *args, **kwargs):
        super().__init__(tx=tx, rx=rx, *args, **kwargs)

    def realize(self, seed=None, energy=None):
        """Realize the channel."""
        shape = (self.rx.N, self.tx.N)
        if seed is not None:
            np.random.seed(self.seed)
        self.channel_matrix = np.sqrt(1 / 2) * (
            np.random.randn(*shape) + 1j * np.random.randn(*shape)
        )
        if energy is None:
            energy = self.tx.N * self.rx.N
        self.normalize_energy(energy)
        return self
