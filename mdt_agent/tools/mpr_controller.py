"""
MPR Controller - Multi-Planar Reformation for vascular cruising.

This module provides the `extract_orthogonal_slice` function for resampling a 3D
medical image along an arbitrary plane orthogonal to a vessel's local tangent.

Implementation (SimpleITK.ResampleImageFilter)
=============================================
SimpleITK resampling always operates in 3D. To produce a 2D reformatted slice:

  1. OUTPUT SIZE = (W, H, 1) — single-slice 3D volume
  2. OUTPUT DIRECTION matrix encodes the slice plane orientation in physical space:
     columns = [Nx Bx Tx; Ny By Ty; Nz Bz Tz]
     where (T, B, N) is the orthonormal tangent/binormal/normal frame
     This correctly orients the 2D plane perpendicular to T in physical space.
  3. OUTPUT ORIGIN = [0, 0, 0] (convenient choice)
  4. OUTPUT SPACING = input (x, y, z) spacing — z spacing preserved but unused
  5. TranslationTransform offset = physical center of the cruise point
     (maps output voxel (0,0,0) → physical point in input image)
  6. OUTPUT DIRECTION rotates physical space so the z-axis of the output
     volume aligns with the desired slice normal

Coordinate Conventions
---------------------
  - Internal (nibabel): (z, y, x) voxel indices
  - SimpleITK physical: (x, y, z) LPS+ DICOM world coordinates
  - Tangent vector input is (z, y, x) — converted to (x, y, z) for direction matrix
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional

try:
    import SimpleITK as sitk
    _HAS_SIMPLEITK = True
except ImportError:
    sitk = None
    _HAS_SIMPLEITK = False


def extract_orthogonal_slice(
    image_3d: "sitk.Image",
    point: Tuple[int, int, int],
    tangent_vector: Tuple[float, float, float],
    slice_size: Tuple[int, int] = (100, 100),
) -> Optional[np.ndarray]:
    """
    Extract a 2D reformatted slice perpendicular to a vessel tangent.

    Parameters
    ----------
    image_3d : sitk.Image
        3D medical image (e.g., CT scan loaded via sitk.ReadImage).
    point : Tuple[int, int, int]
        Voxel coordinates (z, y, x) of the centerline cruise point.
    tangent_vector : Tuple[float, float, float]
        Unit tangent vector (z, y, x) at that point.
        Computed as: normalize(coords[i+1] - coords[i]).
    slice_size : Tuple[int, int]
        Output slice dimensions in pixels (width, height).
        Default: (100, 100).

    Returns
    -------
    np.ndarray or None
        2D numpy array of shape (height, width).
        Returns None if SimpleITK is unavailable or resampling fails.

    Example
    -------
    >>> import SimpleITK as sitk
    >>> from mdt_agent.tools.centerline_tool import VesselCenterlineExtractor
    >>> from mdt_agent.tools.mpr_controller import extract_orthogonal_slice
    >>> img = sitk.ReadImage("pancreas_ct.nii.gz")
    >>> ext = VesselCenterlineExtractor()
    >>> coords = ext.extract(sitk.GetArrayFromImage(sma_nii))
    >>> tangent = tuple((coords[10] - coords[9]) / norm(coords[10] - coords[9]))
    >>> sl = extract_orthogonal_slice(img, coords[10], tangent)
    >>> print(sl.shape)
    (100, 100)
    """
    if not _HAS_SIMPLEITK or image_3d is None:
        return None

    # Normalize tangent (z, y, x)
    t = np.array(tangent_vector, dtype=float)
    norm = np.linalg.norm(t)
    if norm < 1e-6:
        return None
    t = t / norm

    # Convert from (z, y, x) to SimpleITK physical (x, y, z)
    t_sitk = np.array([t[2], t[1], t[0]], dtype=float)

    # Reference vector for Gram-Schmidt — prefer z-axis (SI direction in DICOM LPS+)
    ref = np.array([0.0, 0.0, 1.0], dtype=float)
    if np.abs(np.dot(t_sitk, ref)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=float)

    # Gram-Schmidt: B = T × ref (binormal), N = B × T (normal)
    b = np.cross(t_sitk, ref)
    norm_b = np.linalg.norm(b)
    if norm_b < 1e-6:
        return None
    b = b / norm_b
    n = np.cross(b, t_sitk)

    # Output direction matrix (row-major, 9 elements):
    # rows of [n|b|t] so that output z-axis = normal N
    # SimpleITK: direction = [Nx Bx Tx; Ny By Ty; Nz Bz Tz]
    direction = np.column_stack([n, b, t_sitk]).flatten().tolist()

    # Physical center of the cruise point in input image
    # point is (z, y, x); SimpleITK uses (x, y, z)
    idx = np.array([point[2], point[1], point[0]], dtype=float)
    spacing = np.array(image_3d.GetSpacing(), dtype=float)
    origin = np.array(image_3d.GetOrigin(), dtype=float)
    direction_mat = np.array(image_3d.GetDirection()).reshape(3, 3)
    phys_center = origin + direction_mat @ (idx * spacing)

    # Translation transform: maps output voxel (0,0,0) → phys_center in input image
    transform = sitk.TranslationTransform(3)
    transform.SetOffset(phys_center.tolist())

    # Output spacing: x, y match input; z is preserved (not used for single-slice)
    out_spacing = [
        float(image_3d.GetSpacing()[0]),
        float(image_3d.GetSpacing()[1]),
        float(image_3d.GetSpacing()[2]),
    ]
    # SimpleITK size order: (width, height, depth) = (x, y, z)
    out_size = [slice_size[0], slice_size[1], 1]

    resampler = sitk.ResampleImageFilter()
    resampler.SetTransform(transform)
    resampler.SetOutputSpacing(out_spacing)
    resampler.SetOutputOrigin([0.0, 0.0, 0.0])
    resampler.SetOutputDirection(direction)
    resampler.SetSize(out_size)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(-1024)  # HU air value for CT

    try:
        output = resampler.Execute(image_3d)
    except Exception:
        return None

    # SimpleITK array order: (z, y, x) → shape (1, H, W)
    arr = sitk.GetArrayFromImage(output)
    if arr.shape[0] == 1:
        arr = arr[0]  # squeeze to (H, W)
    return arr
