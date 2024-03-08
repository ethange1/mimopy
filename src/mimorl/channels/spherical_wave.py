from .awgn import Channel

class SphericalWave(Channel):
    """Spherical wave channel."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "SphericalWave"
    
    def realize(self):
        pass