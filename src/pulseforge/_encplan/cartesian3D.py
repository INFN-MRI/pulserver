"""Cartesian 3D phase encoding plan."""

__all__ = ["CaipirinhaSampling"]

import warnings

import numpy as np
import numba as nb

from .partial_fourier import partial_fourier

class CaipirinhaSampling:
    """
    Wrapper class for 3D CAIPIRINHA sampling.
    """
    
    def __init__(self, shape, accel=1, calib=None, shift=0, crop_corner=True, pf=1.0):
        """
        Generate regular sampling pattern for GRAPPA/ARC accelerated acquisition.

        Parameters
        ----------
        shape : int | Tuple[int] 
            Image shape along phase encoding dims ``(ny, nz)``.
            If scalar, assume equal size for ``y`` and ``z`` axes.
        accel : float | Tuple[float], optional
            Target acceleration factor along phase encoding dims ``(Ry, Rz)``. 
            Must be ``>= 1``. If scalar, assume acceleration over ``y``
            only. The default is ``1`` (no acceleration).
        calib : int | Tuple[int], optional
            Image shape along phase encoding dims ``(cy, cz)``.
            If scalar, assume equal size for ``y`` and ``z`` axes.
            The default is ``None`` (no calibration).
        shift : int, optional
            Caipirinha shift. The default is ``0`` (standard PI sampling).
        crop_corner : bool, optional
            Toggle whether to crop corners of k-space (elliptical sampling).
            The default is ``True``.
        pf : float, optional
            Partial Fourier undersampling along ``z`` axis.
            Ranges from ``0.5`` to ``1.0`` (no Partial Fourier acceleration).
            A minimum value of ``0.75`` is suggested. The default is ``1.0``.

        """
        # generate mask
        self._mask = _regular_mask(shape, accel, calib, shift, crop_corner)
        
        # apply partial fourier
        if pf < 1.0:
            pf_mask = partial_fourier(self._mask.shape[-1], pf)
            self._mask = self._mask.astype(np.float32)
            self._mask *= pf_mask
        
        # cast to bool
        self._mask = self._mask.astype(bool)
        
        # get scaling factor
        ny, nz = self._mask.shape
        
        # prepare grid
        iy, iz = np.mgrid[:ny, :nz]
            
        # rescale between (-1, 1)
        scale_y, scale_z = iy - ny // 2, iz - nz // 2
        scale_y, scale_z = scale_y / ny, scale_z / nz
        scale_y, scale_z = 2 * scale_y, 2 * scale_z
        
        # pick samples
        self._iy = iy[self._mask]
        self._iz = iz[self._mask]
        
        self._scale_y = scale_y[self._mask]
        self._scale_z = scale_z[self._mask]
        
        # number of pe steps
        self._nmax = len(scale_y)
        
    def __call__(self, idx):
        n = idx % self._nmax
        return self._scale_y[n], self._scale_z[n]
    
    def update_hdr(self, hdr, idx):
        n = idx % self._nmax
        iy, iz = self._iy[n], self._iz[n]

class PoissonSampling:
    pass

class ShufflePoissonSampling:
    pass

# %% local subroutines
def _regular_mask(shape, accel=1, calib=None, shift=0, crop_corner=True):
    """
    Generate regular sampling pattern for GRAPPA/ARC accelerated acquisition.

    Parameters
    ----------
    shape : int | Tuple[int] 
        Image shape along phase encoding dims ``(ny, nz)``.
        If scalar, assume equal size for ``y`` and ``z`` axes.
    accel : float | Tuple[float], optional
        Target acceleration factor along phase encoding dims ``(Ry, Rz)``. 
        Must be ``>= 1``. If scalar, assume acceleration over ``y``
        only. The default is ``1`` (no acceleration).
    calib : int | Tuple[int], optional
        Image shape along phase encoding dims ``(cy, cz)``.
        If scalar, assume equal size for ``y`` and ``z`` axes.
        The default is ``None`` (no calibration).
    shift : int, optional
        Caipirinha shift. The default is ``0`` (standard PI sampling).
    crop_corner : bool, optional
        Toggle whether to crop corners of k-space (elliptical sampling).
        The default is ``True``.

    Returns
    -------
    mask : np.ndarray
        Regular-grid sampling mask of shape ``(ny, nz)``.

    """
    if np.isscalar(shape):
        # assume square matrix (ky, kz)
        shape = [shape, shape]
    if np.isscalar(accel):
        # assume acceleration along a single axis
        accel = [accel, 1]
        
    # cast tuple to lists
    shape = list(shape)
    accel = list(accel)
        
    # define elliptical grid
    nz, ny = shape
    z, y = np.mgrid[:nz, :ny]
    y, z = abs(y - shape[-1] // 2), abs(z - shape[-2] // 2)
    r = np.sqrt((y / shape[-1]) ** 2 + (z / shape[-2]) ** 2) < 0.5

    # check
    if accel[0] < 1:
        raise ValueError(f"Ky acceleration must be >= 1, got {accel[0]}")
    if accel[1] < 1:
        raise ValueError(f"Kz acceleration must be >= 1, got {accel[1]}")
    if shift < 0:
        raise ValueError(f"CAPIRINHA shift must be positive, got {shift}")
    if shift > accel[1] - 1:
        raise ValueError(f"CAPIRINHA shift must be lower than Rz, got {shift}")

    # build mask
    rows, cols = np.mgrid[:nz, :ny]
    mask = (rows % accel[0] == 0) & (cols % accel[1] == 0)

    # CAPIRINHA shift
    if shift > 0:
        # first pad
        padsize0 = int(np.ceil(mask.shape[0] / accel[0]) * accel[0] - mask.shape[0])
        mask = np.pad(mask, ((0, padsize0), (0, 0)))
        nzp0, _ = mask.shape

        # first reshape
        mask = mask.reshape(nzp0 // accel[0], accel[0], ny)
        mask = mask.reshape(nzp0 // accel[0], accel[0] * ny)

        # second pad
        padsize1 = int(np.ceil(mask.shape[0] / accel[1]) * accel[1] - mask.shape[0])
        mask = np.pad(mask, ((0, padsize1), (0, 0)))
        nzp1, _ = mask.shape

        # second reshape
        mask = mask.reshape(nzp1 // accel[1], accel[1], accel[0] * ny)

        # perform shift
        for n in range(1, mask.shape[1]):
            actshift = n * shift
            mask[:, n, :] = np.roll(mask[:, n, :], actshift)

        # first reshape back
        mask = mask.reshape(nzp1, accel[0] * ny)
        mask = mask[:nzp0, :]

        # second reshape back
        mask = mask.reshape(nzp0 // accel[0], accel[0], ny)
        mask = mask.reshape(nzp0, ny)
        mask = mask[:nz, :]

    # re-insert calibration region
    if calib is not None:
        # broadcast
        if np.isscalar(calib):
            calib = [calib, calib]
            
        # cast tuple to list
        calib = list(calib)
            
        # reverse (cz, cy)
        calib.reverse()
        
        mask[
            shape[0] // 2 - calib[0] // 2 : shape[0] // 2 + calib[0] // 2,
            shape[1] // 2 - calib[1] // 2 : shape[1] // 2 + calib[1] // 2,
        ] = 1

    # crop corners
    if crop_corner:
        mask *= r

    return mask.T # (ny, nz)

def _poisson_disk(shape, accel, calib=None, crop_corner=True, seed=0, max_attempts=30, tol=0.1):
    """
    Generate variable-density Poisson-disc sampling pattern.

    The function generates a variable density Poisson-disc sampling
    mask with density proportional to :math:`1 / (1 + s |r|)`,
    where :math:`r` represents the k-space radius, and :math:`s`
    represents the slope. A binary search is performed on the slope :math:`s`
    such that the resulting acceleration factor is close to the
    prescribed acceleration factor `accel`. The parameter `tol`
    determines how much they can deviate.

    Parameters
    ----------
    shape : int | Tuple[int] 
        Image shape along phase encoding dims ``(ny, nz)``.
        If scalar, assume equal size for ``y`` and ``z`` axes.
    accel : float | Tuple[float], optional
        Target acceleration factor along phase encoding dims ``(Ry, Rz)``. 
        Must be ``>= 1``. If scalar, assume acceleration over ``y``
        only. The default is ``1`` (no acceleration).
    calib : int | Tuple[int], optional
        Image shape along phase encoding dims ``(cy, cz)``.
        If scalar, assume equal size for ``y`` and ``z`` axes.
        The default is ``None`` (no calibration).
    crop_corner : bool, optional
        Toggle whether to crop corners of k-space (elliptical sampling).
        The default is ``True``.
    seed : int, optional 
        Random seed. The default is ``0``.
    max_attempts : float, optional
        Maximum number of samples to reject in Poisson disc calculation.
        The default is ``30``.
    tol : float. optional
        Tolerance for how much the resulting acceleration can
        deviate form ``accel``. The default is ``0.1``.

    Returns
    -------
    mask : np.ndarray
        Poisson-disc sampling mask of shape ``(ny, nz)``.
    R : float
        Actual undersampling factor.

    References
    ----------
    Bridson, Robert. "Fast Poisson disk sampling in arbitrary dimensions."
    SIGGRAPH sketches. 2007.

    """
    if np.isscalar(shape):
        # assume square matrix (ky, kz)
        shape = [shape, shape, 1]
    if len(shape) == 2:
        shape = shape + [1]

    if calib is not None:
        if np.isscalar(calib):
            calib = [calib, calib]
            
        # find actual calibration size
        if shape[-1] > 1:
            calib = max(calib)
            calib = int(np.ceil(calib / shape[-1]) * shape[-1])
            calib = [calib, calib]
        
        # reverse (cz, cy)
        calib.reverse()

    if accel <= 1:
        raise ValueError(f"accel must be greater than 1, got {accel}")
        
    if seed is not None:
        rand_state = np.random.get_state()

    # define elliptical grid
    ny, nz, nt = shape
    z, y = np.mgrid[:nz, :ny]
    y, z = abs(y - shape[0] / 2), abs(z - shape[1] / 2)
    rdisk = 2 * np.sqrt((y / shape[0]) ** 2 + (z / shape[1]) ** 2)
    if nt == 1:
        r = rdisk[None, ...]
    else:
        t, z, y = np.mgrid[:nt, :nz, :ny]
        y, z, t = abs(y - shape[0] / 2), abs(z - shape[1] / 2), abs(t - shape[2] / 2)
        r = 2 * np.sqrt((y / shape[0]) ** 2 + (z / shape[1]) ** 2 + (t / shape[2]) ** 2)

    # calculate mask
    slope_max = max(ny, nz, nt)
    slope_min = 0
    while slope_min < slope_max:
        slope = (slope_max + slope_min) / 2
        radius_y = np.clip((1 + r * slope) * ny / max(ny, nz, nt), 1, None)
        radius_z = np.clip((1 + r * slope) * nz / max(ny, nz, nt), 1, None)
        radius_t = np.clip((1 + r * slope) * nt / max(ny, nz, nt), 1, None)
        mask = _poisson(shape[0], shape[1], shape[2], radius_y, radius_z, radius_t, max_attempts, seed)
        
        # re-insert calibration region
        mask = _insert_calibration(mask, calib)
        
        if crop_corner:
            mask *= rdisk < 1

        actual_accel = np.prod(shape) / np.sum(mask)

        if abs(actual_accel - accel) < tol:
            break
        if actual_accel < accel:
            slope_min = slope
        else:
            slope_max = slope

    if abs(actual_accel - accel) >= tol:
        warnings.warn(
            f"Cannot generate mask to satisfy accel={accel}"
            f" - actual acceleration will be {actual_accel}"
        )

    # prepare for output
    mask = mask.reshape(shape[2], shape[1], shape[0]).squeeze()
    
    if seed is not None:
        np.random.set_state(rand_state)

    return mask, actual_accel

#%% local utils
@nb.njit(cache=True, fastmath=True)  # pragma: no cover
def _poisson(ny, nz, nt, radius_z, radius_y, radius_t, max_attempts, seed=None):
    mask = np.zeros((nt, nz, ny), dtype=np.int32)

    if seed is not None:
        np.random.seed(int(seed))

    # initialize active list
    pys = np.empty(ny * nz * nt, np.int32)
    pzs = np.empty(ny * nz * nt, np.int32)
    pts = np.empty(ny * nz * nt, np.int32)
    pys[0] = np.random.randint(0, ny)
    pzs[0] = np.random.randint(0, nz)
    pts[0] = np.random.randint(0, nt)
    num_actives = 1

    while num_actives > 0:
        i = np.random.randint(0, num_actives)
        py = pys[i]
        pz = pzs[i]
        pt = pts[i]
        ry = radius_y[pt, pz, py]
        rz = radius_z[pt, pz, py]
        rt = radius_t[pt, pz, py]

        # Attempt to generate point
        done = False
        k = 0
        while not done and k < max_attempts:
            # Generate point randomly from r and 2 * r
            v = (np.random.random() * 3 + 1) ** 0.5
            phi = 2 * np.pi * np.random.random()
            theta = np.arccos(np.random.random() * 2 - 1)
            
            qy = py + v * ry * np.cos(phi) * np.sin(theta)
            qz = pz + v * rz * np.sin(phi) * np.sin(theta)
            qt = pt + v * rt * np.cos(theta)

            # Reject if outside grid or close to other points
            if qy >= 0 and qy < ny and qz >= 0 and qz < nz and qt >= 0 and qt < nt:
                starty = max(int(qy - ry), 0)
                endy = min(int(qy + ry + 1), ny)
                startz = max(int(qz - rz), 0)
                endz = min(int(qz + rz + 1), nz)
                startt = max(int(qt - rt), 0)
                endt = min(int(qt + rt + 1), nt)

                done = True
                for y in range(starty, endy):
                    for z in range(startz, endz):
                        for t in range(startt, endt):
                            if mask[t, z, y] == 1 and (
                                ((qy - y) / (radius_y[t, z, y])) ** 2
                                + ((qz - z) / (radius_z[t, z, y])) ** 2
                                + ((qt - t) / (radius_t[t, z, y])) ** 2
                                < 1
                            ):
                                done = False
                                break

            k += 1

        # Add point if done else remove from active list
        if done:
            pys[num_actives] = qy
            pzs[num_actives] = qz
            pts[num_actives] = qt
            mask[int(qt), int(qz), int(qy)] = 1
            num_actives += 1
        else:
            pys[i] = pys[num_actives - 1]
            pzs[i] = pzs[num_actives - 1]
            pts[i] = pts[num_actives - 1]
            num_actives -= 1

    return mask

def _insert_calibration(mask, calib):
    shape = mask.shape
    if calib is not None:
        calib_mask = np.zeros(shape[1:], dtype=int)
        
        # find center and edges
        y0, z0 = shape[1] // 2, shape[2] // 2
        dy, dz = calib[0] // 2, calib[1] // 2
        calib_mask[y0-dy:y0+dy, z0-dz:z0+dz] = 1
        
        # find indices and fill mask
        idx = np.where(calib_mask)
        idx = [i.reshape(shape[0], int(i.shape[0] / shape[0])) for i in idx]
        idx = nb.typed.List(idx)
        _fill_mask(mask, idx)
        
    return mask
        
@nb.njit(cache=True)
def _fill_mask(mask, idx):
    nframes = mask.shape[0]
    npts = idx[0].shape[-1]
    for n in range(nframes):
        for i in range(npts):
            mask[n, idx[0][n, i], idx[1][n, i]] = 1
