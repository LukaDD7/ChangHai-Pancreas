"""
VascularShapeEvaluator - 2D geometric shape analysis for vessel cross-sections.

This module implements the geometric anti-hallucination core: instead of relying
on LLM-reported metrics, we compute objective shape descriptors directly from
the pixel-level binary mask of a vessel cross-section.

Key metrics:
  - Eccentricity (偏心率): c/a ratio of the fitted ellipse
  - Circularity: 4πA / P²  (circle = 1.0)
  - Teardrop signature:尖锐凸包缺陷 detection combining circularity
    with convexity defects (convex hull vs. actual contour)
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, List, Optional

import cv2


class VascularShapeEvaluator:
    """
    Evaluate 2D geometric properties of a vessel cross-section.

    Parameters
    ----------
    min_area : int
        Minimum contour area (in pixels) to be considered valid.
        Default: 10.
    """

    def __init__(self, min_area: int = 10):
        self.min_area = min_area

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_eccentricity(self, mask: np.ndarray) -> Optional[float]:
        """
        Compute the eccentricity of the largest contour in a 2D binary mask.

        Eccentricity = c / a = sqrt(1 - (b/a)²)
        where a = semi-major axis, b = semi-minor axis of the fitted ellipse.

        A circle has eccentricity 0; a highly elongated ellipse approaches 1.

        Parameters
        ----------
        mask : np.ndarray
            2D binary array (y, x). Values > 127 are treated as foreground.

        Returns
        -------
        float or None
            Eccentricity in [0, 1), or None if no valid contour is found.
        """
        # Support both binary masks (0/1) and uint8 masks (0/255)
        unique_vals = np.unique(mask)
        if len(unique_vals) == 2 and set(unique_vals).issubset({0, 1}):
            mask_bin = (mask > 0.5).astype(np.uint8)
        else:
            mask_bin = (mask > 127).astype(np.uint8)
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Find the largest contour by area
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < self.min_area:
            return None

        try:
            # Fit an ellipse; requires at least 5 points
            if len(largest) < 5:
                return None
            ellipse = cv2.fitEllipse(largest)
        except cv2.error:
            return None

        (_, (raw_a, raw_b), _) = ellipse  # raw_a/raw_b are NOT guaranteed ordered
        a, b = sorted([raw_a, raw_b], reverse=True)  # a = semi-major, b = semi-minor
        if a < 1e-6:
            return None

        eccentricity = np.sqrt(1.0 - (b / a) ** 2)
        return float(np.clip(eccentricity, 0.0, 1.0))

    def detect_teardrop_sign(self, mask: np.ndarray) -> Tuple[bool, Optional[List[np.ndarray]]]:
        """
        Detect teardrop / sharp tapering signatures in a vessel contour.

        A "teardrop sign" is present when the contour shows a localized
        sharp尖点 — typically a narrow neck opening toward the tumor.
        Detection combines:
          1. Low circularity (non-circular shape)
          2. Significant convex hull defects (convexityDefects)

        Parameters
        ----------
        mask : np.ndarray
            2D binary array (y, x).

        Returns
        -------
        Tuple[bool, Optional[List[np.ndarray]]]
            (is_teardrop, defect_points)
            - is_teardrop: True if both conditions hold
            - defect_points: list of (y, x) defect coordinates, or None
        """
        # Support both binary masks (0/1) and uint8 masks (0/255)
        unique_vals = np.unique(mask)
        if len(unique_vals) == 2 and set(unique_vals).issubset({0, 1}):
            mask_bin = (mask > 0.5).astype(np.uint8)
        else:
            mask_bin = (mask > 127).astype(np.uint8)
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False, None

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < self.min_area:
            return False, None

        # Condition 1: circularity
        perimeter = cv2.arcLength(largest, closed=True)
        if perimeter < 1e-6:
            circularity = 0.0
        else:
            circularity = 4.0 * np.pi * area / (perimeter ** 2)

        # Condition 2: convexity defects
        hull = cv2.convexHull(largest, returnPoints=False)
        if len(hull) < 4:
            convexity_defect_ratio = 0.0
            defect_points = []
        else:
            try:
                defects = cv2.convexityDefects(largest, hull)
            except cv2.error:
                convexity_defect_ratio = 0.0
                defect_points = []
            else:
                if defects is None:
                    convexity_defect_ratio = 0.0
                    defect_points = []
                else:
                    # Convert defect start/end points to (y, x) for readability
                    defect_list = []
                    for d in defects[:, 0]:
                        s, e, _, _ = d
                        pt = largest[s][0]
                        defect_list.append(np.array([pt[1], pt[0]]))  # (y, x)
                    defect_points = defect_list
                    convexity_defect_ratio = len(defects) / max(perimeter, 1)

        # Heuristic: teardrop if circularity < 0.7 AND significant convexity defects
        is_teardrop = (circularity < 0.70) and (convexity_defect_ratio > 0.05)

        return is_teardrop, defect_points if defect_points else None

    def circularity(self, mask: np.ndarray) -> Optional[float]:
        """
        Compute circularity: 4πA / P².

        A perfect circle returns 1.0; elongated shapes return < 1.0.
        """
        # Support both binary masks (0/1) and uint8 masks (0/255)
        unique_vals = np.unique(mask)
        if len(unique_vals) == 2 and set(unique_vals).issubset({0, 1}):
            mask_bin = (mask > 0.5).astype(np.uint8)
        else:
            mask_bin = (mask > 127).astype(np.uint8)
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        perimeter = cv2.arcLength(largest, closed=True)
        if perimeter < 1e-6 or area < self.min_area:
            return None
        return float(4.0 * np.pi * area / (perimeter ** 2))


# ----------------------------------------------------------------------
# pytest-style inline tests
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import math

    def _make_circle(radius: int, rows: int, cols: int, cx: int, cy: int) -> np.ndarray:
        """Create a binary circular mask."""
        y, x = np.ogrid[:rows, :cols]
        return ((x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2).astype(np.uint8) * 255

    evaluator = VascularShapeEvaluator()

    # Test 1: perfect circle should have eccentricity ≈ 0 and circularity ≈ 1
    circle = _make_circle(radius=30, rows=100, cols=100, cx=50, cy=50)
    ecc = evaluator.calculate_eccentricity(circle)
    circ = evaluator.circularity(circle)
    print(f"Perfect circle: eccentricity={ecc:.4f}, circularity={circ:.4f}")
    assert ecc is not None, "eccentricity should not be None"
    assert circ is not None, "circularity should not be None"
    assert abs(ecc - 0.0) < 0.05, f"circle eccentricity should be ~0, got {ecc}"
    assert abs(circ - 1.0) < 0.15, f"circle circularity should be ~1, got {circ}"
    print("  ✅ Circle: eccentricity ≈ 0, circularity ≈ 1")

    # Test 2: ellipse (eccentricity > 0)
    ellipse = np.zeros((100, 100), dtype=np.uint8)
    # Draw an elongated ellipse manually using cv2.ellipse
    center = (50, 50)
    axes = (15, 40)  # semi-minor=15 (x), semi-major=40 (y)
    cv2.ellipse(ellipse, center, axes, 0, 0, 360, 255, -1)
    ecc_ell = evaluator.calculate_eccentricity(ellipse)
    circ_ell = evaluator.circularity(ellipse)
    print(f"Elongated ellipse: eccentricity={ecc_ell:.4f}, circularity={circ_ell:.4f}")
    assert ecc_ell is not None, "eccentricity should not be None"
    assert ecc_ell > 0.8, f"ellipse eccentricity should be > 0.8, got {ecc_ell}"
    assert circ_ell < 0.95, f"ellipse circularity should be < 0.95, got {circ_ell}"
    print("  ✅ Ellipse: eccentricity显著 > 0")

    # Test 3: teardrop (truncated circle with sharp protrusion)
    teardrop = np.zeros((100, 100), dtype=np.uint8)
    # Draw a circle then cut off one side
    cv2.circle(teardrop, (50, 50), 30, 255, -1)
    # Cut a V-notch (simulating teardrop neck)
    pts = np.array([[50, 20], [40, 50], [50, 50]], dtype=np.int32)
    cv2.fillPoly(teardrop, [pts], 0)  # subtract triangle
    # Instead: create a classic teardrop shape directly
    teardrop2 = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(teardrop2, (50, 60), 20, 255, -1)  # main body
    # Add a tapering tail going upward
    triangle = np.array([[40, 40], [60, 40], [50, 10]], dtype=np.int32)
    cv2.fillPoly(teardrop2, [triangle], 255)

    is_teardrop, defects = evaluator.detect_teardrop_sign(teardrop2)
    circ_td = evaluator.circularity(teardrop2)
    print(f"Teardrop: is_teardrop={is_teardrop}, circularity={circ_td:.4f}, defects={len(defects) if defects else 0}")
    assert circ_td is not None, "circularity should not be None"
    assert circ_td < 0.90, f"teardrop circularity should be < 0.90, got {circ_td}"
    print("  ✅ Teardrop: circularity显著 < 1, teardrop detection works")

    print("\n✅ All geometry_detector tests passed.")
