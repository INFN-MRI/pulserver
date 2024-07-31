"""ScanLoop helper wrapper."""

__all__ = ["ScanLoop"]

import numpy as np


class ScanLoop:
    """
    Scan loop wrapper.

    Each line reports the actual (set-of) Pulseq blocks to be executed
    at a given step of the sequence, as well as their dynamic properties
    (scaling, rotation, phase/frequency offsets).
    """

    def __init__(self):
        self._blockname = []
        self._rfamp = []
        self._rfphase = []
        self._rfoff = []
        self._gxamp = []
        self._gyamp = []
        self._recphase = []
        self._delay = []
        self._rotmat = []

        self._eye = np.eye(3)

    def update(
        self,
        blockname,
        rfamp=1.0,
        rfphase=0.0,
        rfoff=0.0,
        gxamp=1.0,
        gyamp=1.0,
        gzamp=1.0,
        recphase=0.0,
        delay=0.0,
        rotmat=None,
    ):
        """
        Update dynamic scan loop.

        Parameters
        ----------
        blockname : str
            Block name.
        rfamp : float, optional
            RF scaling ``[0.0, 1.0]``. The default is ``1.0``.
        rfphase : float, optional
            RF phase in ``[rad]``. The default is ``0.0``.
        rfoff : float, optional
            RF frequency offset in ``[Hz]``. The default is ``0.0``.
        gxamp : float, optional
            Gx scaling ``[0.0, 1.0]``. The default is ``1.0``.
        gyamp : float, optional
            Gy scaling ``[0.0, 1.0]``. The default is ``1.0``.
        gzamp : float, optional
            Gz scaling ``[0.0, 1.0]``. The default is ``1.0``.
        recphase : float, optional
            ADC phase in ``[rad]``. The default is ``0.0``.
        delay : float, optional
            Extra time in ``[ms]``. The default is ``0.0``.
        rotmat : np.ndarray, optional
            Gradient rotation matrix. The default is ``None`` (Identity).

        """
        if rotmat is None:
            rotmat = self._eye

        # update
        self._blockname.append(blockname)
        self._rfamp.append(rfamp)
        self._rfphase.append(rfphase)
        self._rfoff.append(rfoff)
        self._gxamp.append(gxamp)
        self._gyamp.append(gyamp)
        self._gzamp.append(gzamp)
        self._recphase.append(recphase)
        self._delay.append(delay)
        self._rotmat.append(rotmat.ravel())
