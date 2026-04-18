"""
VesselCenterlineExtractor - 3D centerline extraction for vascular masks.

Refactored from PAN-VIQ (https://github.com/IMIT-PMCL/PDAC):
  - 3D skeletonization via skimage.morphology.skeletonize_3d  (scikit-image >= 0.24.0)
    OR per-slice zhang skeletonization as fallback (scikit-image any version)
  - Graph construction via networkx (26-neighborhood adjacency)
  - Shortest-path ordering between farthest graph endpoints
"""

from __future__ import annotations

import numpy as np
import networkx as nx
from scipy.ndimage import label as ndimage_label
from typing import List, Tuple

try:
    from skimage.morphology import skeletonize_3d
    _HAS_SKELETONIZE_3D = True
except ImportError:
    _HAS_SKELETONIZE_3D = False

from skimage.morphology import skeletonize


def _skeletonize_slice_by_slice(mask_3d: np.ndarray) -> np.ndarray:
    """
    Per-slice 2D zhang skeletonization recomposed into a 3D volume.
    Fallback when skeletonize_3d is unavailable (scikit-image >= 0.25.0).

    Parameters
    ----------
    mask_3d : np.ndarray
        3D binary array (z, y, x).

    Returns
    -------
    np.ndarray
        3D binary skeleton array.
    """
    skels = []
    for z in range(mask_3d.shape[0]):
        sl = mask_3d[z]
        if sl.sum() > 10:  # only process non-empty slices
            try:
                sk = skeletonize(sl.astype(bool), method="zhang")
            except Exception:
                sk = np.zeros_like(sl)
            skels.append(sk)
        else:
            skels.append(np.zeros_like(sl))
    return np.stack(skels)


class VesselCenterlineExtractor:
    """
    Extract a smooth, ordered 3D centerline from a binary vessel mask.

    Parameters
    ----------
    keep_largest_component : bool
        If True, keep only the largest connected component before skeletonizing.
    skeleton_backend : str
        Either "3d" (prefer skeletonize_3d, scikit-image < 0.25) or
        "slice_by_slice" (per-slice zhang, scikit-image any version).
        Default: "3d" if available, else "slice_by_slice".
    """

    def __init__(self, keep_largest_component: bool = True, skeleton_backend: str = "auto"):
        self.keep_largest_component = keep_largest_component
        if skeleton_backend == "auto":
            self._backend = "3d" if _HAS_SKELETONIZE_3D else "slice_by_slice"
        else:
            self._backend = skeleton_backend

    def extract(self, binary_mask: np.ndarray) -> List[Tuple[int, int, int]]:
        """
        Extract ordered centerline points from a 3D vessel mask.

        Parameters
        ----------
        binary_mask : np.ndarray
            3D binary array (z, y, x).

        Returns
        -------
        List[Tuple[int, int, int]]
            Ordered list of (z, y, x) voxel coordinates along the vessel centerline.
            Returns empty list if the mask is empty.

        Notes
        -----
        Pipeline:
          1. Binarize input (threshold = 0.5)
          2. Optionally keep only the largest connected component
          3. Skeletonize:
             - "3d" backend: skeletonize_3d (scikit-image < 0.25)
             - "slice_by_slice" backend: per-slice zhang + stack
          4. Build 26-neighborhood networkx graph
          5. Find graph diameter endpoints (farthest pair)
          6. Shortest path = ordered centerline
        """
        mask = (binary_mask > 0.5).astype(np.uint8)

        if mask.sum() == 0:
            return []

        if self.keep_largest_component:
            mask = self._keep_lcc(mask)

        # Skeletonization with version-aware backend
        if self._backend == "3d":
            skeleton = skeletonize_3d(mask).astype(np.uint8)
        else:
            skeleton = _skeletonize_slice_by_slice(mask).astype(np.uint8)

        if skeleton.sum() == 0:
            return []

        graph = self._skeleton_to_graph(skeleton)
        if len(graph) == 0:
            return []

        try:
            path_nodes = self._ordered_path(graph)
        except ValueError:
            return []

        coords = [(int(p[0]), int(p[1]), int(p[2])) for p in path_nodes]
        return coords

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _keep_lcc(mask: np.ndarray) -> np.ndarray:
        """Keep only the largest connected component."""
        labeled, n = ndimage_label(mask, structure=np.ones((3, 3, 3), dtype=np.uint8))
        if n <= 1:
            return mask
        counts = np.bincount(labeled.ravel())
        counts[0] = 0
        keep = np.argmax(counts)
        return (labeled == keep).astype(np.uint8)

    @staticmethod
    def _skeleton_to_graph(skel: np.ndarray) -> nx.Graph:
        """Build a 26-neighborhood graph from a skeleton volume."""
        coords = np.argwhere(skel > 0)
        idx = {tuple(c): i for i, c in enumerate(coords)}
        G = nx.Graph()
        for c in coords:
            c_tuple = tuple(c)
            i = idx[c_tuple]
            G.add_node(i, xyz=c_tuple)
            z, y, x = c_tuple
            for dz in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dx == dy == dz == 0:
                            continue
                        n = (z + dz, y + dy, x + dx)
                        j = idx.get(n)
                        if j is not None:
                            w = np.sqrt(dx * dx + dy * dy + dz * dz)
                            G.add_edge(i, j, weight=w)
        return G

    @staticmethod
    def _farthest_pair(G: nx.Graph) -> Tuple[int, int]:
        """Return the two farthest nodes in the graph (graph diameter endpoints)."""
        start = next(iter(G.nodes))
        d1 = nx.single_source_dijkstra_path_length(G, start, weight="weight")
        a = max(d1, key=d1.get)
        d2 = nx.single_source_dijkstra_path_length(G, a, weight="weight")
        b = max(d2, key=d2.get)
        return a, b

    def _ordered_path(self, G: nx.Graph) -> np.ndarray:
        """
        Compute the ordered centerline path as a (N, 3) array in voxel coordinates.
        """
        a, b = self._farthest_pair(G)
        path_node_ids = nx.shortest_path(G, a, b, weight="weight")
        coords = np.array([G.nodes[n]["xyz"] for n in path_node_ids], dtype=float)
        return coords
