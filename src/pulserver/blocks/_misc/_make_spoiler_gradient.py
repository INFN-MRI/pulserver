"""Spoiler gradient creation subroutines."""

__all__ = ["make_spoiler_gradient"]


from types import SimpleNamespace

import numpy as np
import pypulseq as pp


def make_spoiler_gradient(
    channel: str,
    system: pp.Opts,
    ncycles: int,
    fov: float,
    npix: int,
    duration: float | None = None,
) -> SimpleNamespace:
    """
    Prepare spoiler gradient with given dephasing across
    the given spatial length.

    Parameters
    ----------
    channel : str
        Phase encoding axis. Must be
        one between ``x``, ``y`` and `z``.
    ncycles : int
        Number of spoiling cycles per voxel.
    system : pypulseq.Opts
        System limits.
    fov : float
        Field of view in the spoiling direction in ``[mm]``.
    npix : int
        Matrix size in the spoiling direction.
    duration : float | None, optional
        Duration of spoiling gradient in ``[s]``.
        If not provided, use minimum duration
        given by area and system specs.
        The default is ``None`` (minumum duration).

    Returns
    -------
    SimpleNamespace
        Spoiling event on the specified axes.

    """
    # get axis
    if channel not in ["x", "y", "z"]:
        raise ValueError(f"Unrecognized channel {channel} - must be 'x', 'y', or 'z'.")

    # k space area
    dr = fov / npix

    # prepare phase encoding gradient lobe
    if duration:
        return pp.make_trapezoid(
            channel=channel,
            area=(ncycles * np.pi / dr),
            system=system,
            duration=duration,
        )

    return pp.make_trapezoid(
        channel=channel, area=(ncycles * np.pi / dr), system=system
    )
