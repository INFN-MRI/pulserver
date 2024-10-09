"""Slice reoreding subroutines."""

__all__ = ["sequential", "interleaved", "center_out"]


import numpy as np


def sequential(n_slices: int) -> np.ndarray:
    """
    Sequential slice acquisition table.

    Parameters
    ----------
    n_slices : int
        Number of slices.

    Returns
    -------
    np.ndarray
        Slice index table.

    """
    return np.arange(n_slices)


def interleaved(n_slices: int) -> np.ndarray:
    """
    Reorder the slice table so that all the even slices are acquired
    after the odd.

    Parameters
    ----------
    n_slices : int
        Number of slices.

    Returns
    -------
    np.ndarray
        Slice index table.

    """
    slice_idx = np.arange(n_slices)
    return np.concatenate((slice_idx[::2], slice_idx[1::2]))


def center_out(n_slices: int) -> np.ndarray:
    """
    Reorder the slice table so that slices are acquired
    from center to periphery.

    Parameters
    ----------
    n_slices : int
        Number of slices.

    Returns
    -------
    np.ndarray
        Slice index table.

    """
    slice_idx = np.arange(n_slices)

    # get ordering
    order = np.zeros_like(slice_idx)
    for n in range(int(n_slices // 2)):
        order[2 * n] = n_slices // 2 + n
        order[2 * n + 1] = n_slices // 2 - n - 1
    if n_slices % 2 != 0:
        order[-1] = n_slices - 1

    return slice_idx[order.astype(int)]
