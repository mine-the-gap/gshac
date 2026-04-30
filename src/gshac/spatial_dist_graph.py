"""
spatial_dist_graph.py

Computes a sparse symmetric distance matrix for hierarchical clustering by
exploiting geographic locality: only pairs of features within a maximum
distance ``h_max`` are computed and stored. Pairs beyond ``h_max`` are
structurally absent and are treated as infinity by any hierarchical
clustering algorithm, which is correct when cutting the dendrogram at any
height ``h <= h_max``.

The algorithm:
  1. Build a spatial index (KD-tree for projected, Ball-tree for geographic)
  2. For each feature i, query neighbours within ``h_max`` -> candidate pairs
     ``(i, j)`` with ``j > i`` (upper triangle only)
  3. Compute exact pairwise distances for candidate pairs
  4. Store as scipy sparse symmetric CSR matrix
  5. Find connected components (``scipy.sparse.csgraph``)

Public API
----------
spatial_dist_graph(coords, h_max, metric, crs=None)
    Full sparse distance graph (distances stored as edge weights).

geographic_connectivity(coords, h_max, metric, crs=None)
    Binary connectivity matrix for use with
    ``sklearn.cluster.AgglomerativeClustering(connectivity=...)``.

Dependencies
------------
* numpy, scipy, scikit-learn (always)
* pyproj (optional, for CRS introspection)
"""

from __future__ import annotations

from typing import Any, Literal, Optional, Tuple, Union

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

try:
    from ._gshac import haversine_edges as _c_haversine
    _GSHAC_C = True
except ImportError:
    _GSHAC_C = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EARTH_RADIUS_M = 6_371_000.0
"""Mean Earth radius (metres) used for the haversine sphere model.

Matches the constant baked into the C extension (``_gshac.haversine_edges``).
"""


_MetricStr = Literal["euclidean", "haversine"]
_VALID_METRICS = ("euclidean", "haversine")


# ---------------------------------------------------------------------------
# CRS / input handling
# ---------------------------------------------------------------------------

def _coerce_crs(crs: Any) -> Optional[Any]:
    """Coerce a user-supplied CRS spec into a ``pyproj.CRS`` instance.

    ``None`` passes through as ``None`` (missing CRS). Strings, integer EPSG
    codes, and existing ``pyproj.CRS`` objects are all accepted.
    """
    if crs is None:
        return None
    try:
        from pyproj import CRS
    except ImportError as e:  # pragma: no cover - exercised only when pyproj missing
        raise ImportError(
            "pyproj is required to specify a CRS. "
            "Install via `pip install gshac[geo]`."
        ) from e
    if isinstance(crs, CRS):
        return crs
    return CRS.from_user_input(crs)


def _crs_kind(crs: Any) -> str:
    """Return ``'geographic'``, ``'projected'``, or ``'missing'``.

    Unknown CRS shapes (neither geographic nor projected per pyproj, e.g.
    geocentric) raise ``ValueError`` — gshac point clustering is not defined
    for those.
    """
    if crs is None:
        return "missing"
    if crs.is_geographic:
        return "geographic"
    if crs.is_projected:
        return "projected"
    raise ValueError(
        f"Unsupported CRS type: {crs.name!r} is neither geographic nor "
        "projected. gshac supports only geographic (lon/lat) or projected "
        "(planar) coordinate systems."
    )


def _resolve_inputs(
    coords: Any,
    crs: Any,
) -> Tuple[np.ndarray, Optional[Any]]:
    """Normalise ``coords`` into ``(ndarray, pyproj.CRS|None)``.

    For now only raw ndarray inputs are accepted; geopandas ingestion is
    added in a follow-up commit. Validates shape and accepts a
    pyproj-acceptable ``crs=`` spec.
    """
    arr = np.asarray(coords, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(
            f"coords must be shape (n, 2); got shape {arr.shape!r}."
        )
    return arr, _coerce_crs(crs)


def _resolve_metric(metric: str, crs_kind: str) -> str:
    """Validate ``(metric, CRS kind)`` and return the resolved metric.

    Currently a pure validator: it does not auto-dispatch. A follow-up commit
    adds ``"auto"`` and the geographic <-> geodesic mapping. Centralising the
    validation matrix in this function lets later commits extend the dispatch
    without touching the call site in ``spatial_dist_graph``.
    """
    if metric not in _VALID_METRICS:
        raise ValueError(
            f"Unknown metric: {metric!r}. Use one of {_VALID_METRICS!r}."
        )
    return metric


# ---------------------------------------------------------------------------
# Distance kernels
# ---------------------------------------------------------------------------

def _euclidean_pairs(
    coords: np.ndarray,
    h_max: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Candidate pairs and exact distances for planar Euclidean clustering."""
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=h_max, output_type="ndarray")
    if len(pairs) == 0:
        return np.empty((0, 2), dtype=np.int64), np.empty((0,), dtype=np.float64)

    diffs = coords[pairs[:, 0]] - coords[pairs[:, 1]]
    dists = np.linalg.norm(diffs, axis=1)
    return pairs.astype(np.int64, copy=False), dists


def _haversine_pairs(
    coords: np.ndarray,
    h_max: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Candidate pairs (upper triangle) and exact haversine distances.

    ``coords`` is ``(lon, lat)`` in degrees; ``h_max`` is metres on a sphere
    of radius ``EARTH_RADIUS_M``.
    """
    from sklearn.neighbors import BallTree

    n = len(coords)
    coords_rad = np.radians(coords[:, [1, 0]])  # swap lon/lat -> lat/lon
    h_max_rad = h_max / EARTH_RADIUS_M

    tree = BallTree(coords_rad, metric="haversine")
    indices = tree.query_radius(coords_rad, r=h_max_rad)

    counts = np.array([len(js) for js in indices], dtype=np.intp)
    if counts.sum() == 0:
        return np.empty((0, 2), dtype=np.int64), np.empty((0,), dtype=np.float64)

    i_rep = np.repeat(np.arange(n, dtype=np.int64), counts)
    j_flat = np.concatenate(list(indices)).astype(np.int64)
    mask = j_flat > i_rep
    pairs = np.column_stack([i_rep[mask], j_flat[mask]])

    if len(pairs) == 0:
        return pairs, np.empty((0,), dtype=np.float64)

    lon1 = np.radians(coords[pairs[:, 0], 0])
    lat1 = np.radians(coords[pairs[:, 0], 1])
    lon2 = np.radians(coords[pairs[:, 1], 0])
    lat2 = np.radians(coords[pairs[:, 1], 1])
    if _GSHAC_C:
        dists = _c_haversine(lon1, lat1, lon2, lat2, EARTH_RADIUS_M)
    else:
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        dists = 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))
    return pairs, dists


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def spatial_dist_graph(
    coords: np.ndarray,
    h_max: float,
    metric: _MetricStr = "euclidean",
    crs: Any = None,
) -> dict:
    """Build a sparse geographic distance graph.

    Parameters
    ----------
    coords : ndarray, shape (n, 2)
        Feature coordinates. For ``metric="euclidean"``: ``(x, y)`` in metres
        (projected CRS). For ``metric="haversine"``: ``(lon, lat)`` in
        degrees.
    h_max : float
        Maximum distance threshold in metres.
    metric : {"euclidean", "haversine"}, default "euclidean"
        Distance metric. ``"haversine"`` is the spherical great-circle
        distance with mean Earth radius ``EARTH_RADIUS_M``.
    crs : pyproj.CRS or pyproj-acceptable spec (str, int EPSG code), optional
        Coordinate reference system metadata. Currently used for validation
        only; full CRS-aware dispatch is added in a follow-up commit.

    Returns
    -------
    dict with keys ``matrix``, ``components``, ``n_components``, ``n_edges``,
    ``density``. See module docstring for details.
    """
    arr, resolved_crs = _resolve_inputs(coords, crs)
    n = len(arr)
    if n < 2:
        raise ValueError("coords must have at least 2 features")

    kind = _crs_kind(resolved_crs)
    resolved_metric = _resolve_metric(metric, kind)

    # --- candidate pairs + exact distances ---------------------------------
    if resolved_metric == "euclidean":
        pairs, dists = _euclidean_pairs(arr, h_max)
    elif resolved_metric == "haversine":
        pairs, dists = _haversine_pairs(arr, h_max)
    else:  # pragma: no cover
        raise AssertionError(f"unreachable: metric={resolved_metric!r}")

    # --- empty result -------------------------------------------------------
    if len(pairs) == 0:
        mat = csr_matrix((n, n), dtype=np.float64)
        n_comp, labels = connected_components(mat, directed=False)
        return dict(
            matrix=mat,
            components=labels,
            n_components=n_comp,
            n_edges=0,
            density=0.0,
        )

    # Replace co-located points (dist == 0) with 1 m.
    dists = np.where(dists == 0.0, 1.0, dists)

    i_idx = pairs[:, 0]
    j_idx = pairs[:, 1]
    data = np.concatenate([dists, dists])
    row = np.concatenate([i_idx, j_idx])
    col = np.concatenate([j_idx, i_idx])
    mat = csr_matrix((data, (row, col)), shape=(n, n), dtype=np.float64)

    n_comp, labels = connected_components(mat, directed=False)
    return dict(
        matrix=mat,
        components=labels,
        n_components=n_comp,
        n_edges=len(pairs),
        density=len(pairs) / (n * (n - 1) / 2),
    )


def geographic_connectivity(
    coords: np.ndarray,
    h_max: float,
    metric: _MetricStr = "euclidean",
    crs: Any = None,
) -> "csr_matrix":
    """Binary connectivity matrix for use with sklearn AgglomerativeClustering.

    Returns a sparse symmetric CSR matrix where entry ``(i, j) == 1`` iff
    ``dist(i, j) <= h_max``. Pass directly to::

        from sklearn.cluster import AgglomerativeClustering
        connectivity = geographic_connectivity(coords, h_max)
        model = AgglomerativeClustering(
            distance_threshold=h_cut,
            n_clusters=None,
            connectivity=connectivity,
        )
        model.fit(coords)

    Parameters
    ----------
    coords, h_max, metric, crs
        Same semantics as ``spatial_dist_graph``.

    Returns
    -------
    scipy.sparse.csr_matrix of shape (n, n), dtype float64
        Binary (0/1) symmetric connectivity matrix.
    """
    graph = spatial_dist_graph(coords, h_max, metric=metric, crs=crs)
    mat = graph["matrix"].copy()
    mat.data[:] = 1.0
    return mat
