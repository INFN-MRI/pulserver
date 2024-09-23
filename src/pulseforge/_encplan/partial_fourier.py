"""Partial Fourier design routine."""

__all__ = ["partial_fourier"]

import warnings

import numpy as np


def partial_fourier(shape, undersampling):
    """
    Generate sampling pattern for Partial Fourier accelerated acquisition.

    Parameters
    ----------
    shape : int
        Image shape along partial fourier axis.
    undersampling :
        Target undersampling factor. Must be > 0.5 (suggested > 0.7)
        and <= 1 (=1: no PF).

    Returns
    -------
    mask : np.ndarray
        Partial Fourier (1D) sampling mask.

    """
    # check
    if undersampling > 1 or undersampling <= 0.5:
        raise ValueError(
            "undersampling must be greater than 0.5 and lower than 1, got"
            f" {undersampling}"
        )
    if undersampling == 1:
        warnings.warn("Undersampling factor set to 1 - no acceleration")
    if undersampling < 0.7:
        warnings.warn(
            f"Undersampling factor = {undersampling} < 0.7 - phase errors will"
            " likely occur."
        )

    # generate mask
    mask = np.ones(shape, dtype=np.float32)

    # cut mask
    edge = np.floor(np.asarray(shape) * np.asarray(undersampling))
    edge = int(edge)
    mask[edge:] = 0

    return mask
