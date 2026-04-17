"""
MPR Controller - Multi-Planar Reformation for vascular cruising.

This module provides the `extract_orthogonal_slice` function for resampling a 3D
medical image along an arbitrary plane orthogonal to a vessel's local tangent.

Future Implementation Notes (SimpleITK.Euler3DTransform)
====================================================

To sample a 2D slice perpendicular to a tangent vector at a given point,
we need to construct a rigid transform that maps:

  - The slice plane's normal  →  the image's z-axis (LR axis)
  - The slice plane's tangent  →  the image's y-axis (PA axis)
  - The slice plane's binormal →  the image's x-axis (SI axis)

Euler3DTransform (rotation-only, no shear) is ideal here because:
  - It preserves distances and angles (rigid body motion)
  - It has 3 rotation parameters (θx, θy, θz) and 3 translation params
  - SimpleITK can compose it with ResampleImageFilter for fast resampling

Construction Recipe
-------------------
1. Define the tangent vector T = (tx, ty, tz) and binormal B = (bx, by, bz)
   at the current centerline point. Normal N = T × B (cross product).

2. Build the 3×3 rotation matrix R that maps these basis vectors
   from the standard basis to the plane basis:

       R = [N | B | T]   (columns)

   where N/B/T are unit vectors. This is the "orientation" part of the transform.

3. Create the Euler3DTransform:
   ```python
   import SimpleITK as sitk
   transform = sitk.Euler3DTransform()
   transform.SetMatrix(R.flatten())  # rotation matrix
   transform.SetTranslation(t)      # translation to center point
   ```

4. Resample the 3D image onto the 2D slice plane:
   ```python
   resampler = sitk.ResampleImageFilter()
   resampler.SetTransform(transform)
   resampler.SetOutputDirection(sitk.Vector(Direction3D * R.T))  # maintain pixel spacing
   resampler.SetOutputSpacing(spacing3d)                         # match 3D spacing
   resampler.SetSize(output_slice_size)                          # e.g., (100, 100)
   resampler.SetInterpolator(sitk.sitkLinear)
   slice_2d = resampler.Execute(image_3d)
   ```

5. Convert back to numpy for further processing:
   ```python
   import numpy as np
   slice_array = sitk.GetArrayFromImage(slice_2d)  # shape (H, W)
   ```

Spatial Convention
-----------------
- Image axes: (x=SI, y=PA, z=LR) — DICOM LPS+ convention
- Internal convention used here: (z=SI, y=PA, x=LR) — nibabel/NiBabel convention
- The tangent/binormal/normal vectors are in voxel index space (z, y, x)
- The transform maps between these spaces correctly

References
----------
- SimpleITK Euler3DTransform:
  https://itk.org/SimpleITKDoxygen/html/classitk_1_1Euler3DTransform.html
- ResampleImageFilter:
  https://itk.org/SimpleITKDoxygen/html/classitk_1_1ResampleImageFilter.html
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, List, Optional

# Optional dependency — will gracefully fail if SimpleITK is not installed
try:
    import SimpleITK as sitk
    _HAS_SIMPLEITK = True
except ImportError:
    sitk = None
    _HAS_SIMPLEITK = False


def extract_orthogonal_slice(
    image_3d: "sitk.Image",  # SimpleITK Image object
    point: Tuple[int, int, int],
    tangent_vector: Tuple[float, float, float],
    slice_size: Tuple[int, int] = (100, 100),
) -> Optional[np.ndarray]:
    """
    Extract a 2D reformatted slice perpendicular to a vessel tangent.

    Parameters
    ----------
    image_3d : sitk.Image
        3D medical image (e.g., CT scan loaded via SimpleITK).
    point : Tuple[int, int, int]
        Voxel coordinates (z, y, x) of the centerline point.
    tangent_vector : Tuple[float, float, float]
        Unit tangent vector (z, y, x) at that point.
        Computed as the normalized difference between successive
        centerline coordinates: T = normalize(p[i+1] - p[i]).
    slice_size : Tuple[int, int]
        Output slice dimensions in pixels (height, width).
        Default: (100, 100).

    Returns
    -------
    np.ndarray or None
        2D numpy array of shape `slice_size`, or None if SimpleITK
        is not available or the resampling fails.

    Notes
    -----
    This is a stub: it calls a helper that raises NotImplementedError
    until SimpleITK.ResampleImageFilter + Euler3DTransform are wired up.

    Planned implementation:
      1. Normalize the tangent vector → unit vector T
      2. Construct a reference vector R not parallel to T (e.g., [0, 0, 1])
      3. Compute: B = T × R (binormal), N = B × T (normal)
      4. Build 3×3 rotation matrix Rmat = [N | B | T] (columns)
      5. Create Euler3DTransform: rotation = Rmat, translation = point
      6. Configure ResampleImageFilter with size = slice_size
      7. Execute resampling and return result as numpy array

    Example
    -------
    >>> import SimpleITK as sitk
    >>> import numpy as np
    >>> img = sitk.ReadImage("pancreas_ct.nii.gz")
    >>> centerline_points = [(145, 203, 118), (144, 205, 120), ...]
    >>> tangents = [normalize(c2 - c1) for c1, c2 in zip(centerline_points[:-1], centerline_points[1:])]
    >>> slice_img = extract_orthogonal_slice(img, centerline_points[5], tangents[5])
    >>> print(slice_img.shape)
    (100, 100)
    """
    if not _HAS_SIMPLEITK or image_3d is None:
        return None

    # Normalize tangent vector
    t = np.array(tangent_vector, dtype=float)
    norm = np.linalg.norm(t)
    if norm < 1e-6:
        return None
    t = t / norm

    # Step 1: pick a reference vector not parallel to t
    reference = np.array([0.0, 0.0, 1.0])
    if np.abs(np.dot(t, reference)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])

    # Step 2: Gram-Schmidt → build orthonormal frame (T, B, N)
    b = np.cross(t, reference)
    norm_b = np.linalg.norm(b)
    if norm_b < 1e-6:
        return None
    b = b / norm_b
    n = np.cross(b, t)  # normal to the plane = T × B

    # Step 3: rotation matrix [N | B | T] columns → maps plane basis → image basis
    rmat = np.column_stack([n, b, t])

    # Step 4: Euler3DTransform
    transform = sitk.Euler3DTransform()
    transform.SetMatrix(rmat.flatten().tolist())
    transform.SetTranslation([float(point[2]), float(point[1]), float(point[0])])

    # Step 5: ResampleImageFilter
    resampler = sitk.ResampleImageFilter()
    resampler.SetTransform(transform)
    resampler.SetOutputSpacing(image_3d.GetSpacing())
    resampler.SetSize(list(slice_size[::-1]))  # SimpleITK uses (x, y) order
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(-1024)  # HU air value for CT

    output_slice = resampler.Execute(image_3d)
    return sitk.GetArrayFromImage(output_slice)
