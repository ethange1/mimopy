import numpy as np
from scipy.stats import nakagami
from .awgn import Channel

class Nakagami(Channel):
    """Nakagami channel class.
    
    Unique Attributes
    -----------------
    m: float, optional
        Nakagami fading parameter.
    
    """
    def __init__(self, tx, rx, m=1, path_loss="no_loss", *args, **kwargs):
        super().__init__(tx, rx, path_loss, *args, **kwargs)
        
        # Nakagami fading parameter
        self.m = m  

    def realize(self):
        
        # Following prior code
        np.random.seed(self.seed)
        energy = self._energy / self.tx.N / self.rx.N
        shape = (self.rx.N, self.tx.N)
        
        # Nakagami distribution for magnitudes
        magnitude = nakagami.rvs(self.m, size=shape) 
        
        # Uniform distribution for phases
        phase = np.random.rand(*shape) * 2 * np.pi 
        
        # Realize the channel matrix
        self.channel_matrix = np.sqrt(energy) * magnitude * np.exp(1j * phase)
        return self
    