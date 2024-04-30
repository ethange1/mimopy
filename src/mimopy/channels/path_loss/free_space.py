import numpy as np
from .path_loss import PathLoss
from ..awgn import Channel


class FreeSpaceLoss(PathLoss):
    """Free-space path loss model for isotropic antennas."""

    def __str__(self):
        return "Free-space path loss"

    def received_power(self, channel: Channel):
        """
        Calculate the received power at the receiver.

        P_{\mathrm{rx}}=\\frac{P_{\mathrm{tx}}}{4 \pi d^2} \cdot A_{\mathrm{eff}},
        A_{\mathrm{eff}}=\\frac{\lambda^2}{4 \pi}.
        """
        d = channel.get_relative_position(
            channel.tx.array_center, channel.rx.array_center
        )[0]
        # the calculation is based on normalized wavelength
        return channel.tx.power / (16 * np.pi **2 * d**2)
