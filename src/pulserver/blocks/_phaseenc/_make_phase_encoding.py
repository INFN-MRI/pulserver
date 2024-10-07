"""Phase encoding gradient creation subroutines."""

__all__ = ["make_phase_encoding"]

import pypulseq as pp


def make_phase_encoding(
    channel: str,
    system: pp.Opts,
    fov: float,
    npix: int,
    output_block: dict | None = None,
) -> dict:
    """
    Prepare phase encoding gradient for a given resolution.

    Parameters
    ----------
    channel : str
        Phase encoding axis. Must be
        one between ``x``, ``y`` and `z``.
    system : pypulseq.Opts
        System limits.
    fov : float
        Field of view in the readout direction in ``[mm]``.
    npix : int
        Matrix size in the readout direction
        If it is a scalar, assume square matrix.
    output_block : dict, optional
        If provided, the new phase encoding will be
        included in ``output_block`` dictionary.
        The selected channel must be empty.

    Returns
    -------
    phase_block : dict
        Readout block dictionary with the following keys:

        * g{channel} : SimpleNamespace
            Phase encoding event on the specified axes.

        If ``output_block`` is provided, its content will be
        included in ``phase_block``.

    """
    # get axis
    if channel not in ["x", "y", "z"]:
        raise ValueError(f"Unrecognized channel {channel} - must be 'x', 'y', or 'z'.")
    if output_block is not None and f"g{channel}" in output_block:
        raise ValueError(f"Channel {channel} already exists.")

    # unit conversion (mm -> m)
    fov *= 1e-3

    # k space area
    dr = fov / npix

    # prepare phase encoding gradient lobe
    gphase = pp.make_trapezoid(channel=channel, area=1 / dr, system=system)

    # prepare output
    if output_block is not None:
        output_block[f"g{channel}"] = gphase
        return output_block

    return {f"g{channel}": gphase}
