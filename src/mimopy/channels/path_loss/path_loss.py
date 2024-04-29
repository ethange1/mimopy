from abc import ABC, abstractmethod
from .. import Channel

class PathLoss(ABC):
    @abstractmethod
    def received_power(self, channel:Channel):
        """Return the received power at the receiver."""
        pass

class NoLoss(PathLoss):
    def __str__(self):
        return "no_fading"

    def received_power(self, channel:Channel):
        return channel.tx.power
    
    