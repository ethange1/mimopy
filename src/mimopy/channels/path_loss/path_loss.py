from abc import ABC, abstractmethod
from ..awgn import Channel


class PathLoss(ABC):
    @abstractmethod
    def received_power(self, channel: Channel):
        """Return the received power at the receiver."""
        pass


class NoLoss(PathLoss):
    def __str__(self):
        return "no_fading"

    def received_power(self, channel: Channel):
        return channel.tx.power


class ConstantLoss(PathLoss):
    def __init__(self, loss: float):
        self.loss = loss

    def __str__(self):
        return f"constant {self.loss} dB"

    def received_power(self, channel: Channel):
        return channel.tx.power / 10 ** (self.loss / 10)