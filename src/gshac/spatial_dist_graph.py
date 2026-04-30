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

import warnings
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

# Multiplier applied to ``h_max`` when prefiltering geodesic candidates with a
# haversine ball-tree. Spherical haversine on a 6_371_000 m sphere can
# under-estimate the true WGS-84 ellipsoidal distance by at most ~0.27%
# (worst case is high-latitude E-W chords). 1.003 (=0.3%) is a safe upper
# bound that admits no false negatives. See ``docs/design/distance_metrics.md``
# section "Two-stage prefilter" and the test
# ``tests/test_crs_dispatch.py::test_haversine_prefilter_safety``.
_HAVERSINE_GEODESIC_SAFETY = 1.003


_MetricStr = Literal["auto", "euclidean", "haversine", "geodesic"]
_VALID_METRICS = ("auto", "euclidean", "haversine", "geodesic")

# One-time warning state for missing-CRS auto-dispatch. Reset is intentionally
# not exposed: tests that need to re-trigger the warning manipulate this
# module-level flag directly via monkeypatch.
_MISSING_CRS_WARNED = False


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
    """Resolve ``metric="auto"`` to a concrete metric and validate the pair.

    The validation matrix mirrors R's ``sf::st_distance``: distances on a
    geographic CRS must be spherical or ellipsoidal; distances on a
    projected CRS must be planar; mixing the two is a user error.

    Resolution rules:

    * ``"auto"`` — dispatch on CRS kind. Geographic -> ``"geodesic"``,
      projected -> ``"euclidean"``, missing -> ``"euclidean"`` with a
      one-time ``UserWarning`` advising the caller to set the CRS.
    * ``"geodesic"`` / ``"haversine"`` — require a geographic CRS or
      missing CRS. Reject projected CRSs because lon/lat trig on projected
      coordinates is silently wrong.
    * ``"euclidean"`` — require a projected or missing CRS. Reject
      geographic CRSs because degree differences are not metres; sf would
      not even offer this combination.
    """
    if metric not in _VALID_METRICS:
        raise ValueError(
            f"Unknown metric: {metric!r}. Use one of {_VALID_METRICS!r}."
        )

    if metric == "auto":
        if crs_kind == "geographic":
            return "geodesic"
        if crs_kind == "projected":
            return "euclidean"
        # Missing CRS: warn once per process and assume planar.
        global _MISSING_CRS_WARNED
        if not _MISSING_CRS_WARNED:
            warnings.warn(
                "CRS not set; assuming planar coordinates. Pass crs= or use "
                "a GeoDataFrame to silence this warning.",
                UserWarning,
                stacklevel=3,
            )
            _MISSING_CRS_WARNED = True
        return "euclidean"

    if metric in ("haversine", "geodesic"):
        if crs_kind == "projected":
            raise ValueError(
                f"metric={metric!r} requires geographic (lon/lat) coordinates "
                "but a projected CRS was provided. Either pass "
                'metric="euclidean" or reproject your data to a geographic '
                "CRS (e.g. EPSG:4326)."
            )
        # Missing-CRS + spherical/ellipsoidal: allow without warning. The
        # caller has explicitly chosen the metric, so they have asserted
        # that the coordinates are lon/lat.
        return metric

    if metric == "euclidean":
        if crs_kind == "geographic":
            raise ValueError(
                'metric="euclidean" is invalid for a geographic (lon/lat) '
                'CRS: degree differences are not metres. Use metric="auto", '
                '"geodesic", or "haversine" instead, or reproject your data '
                "to a projected CRS first."
            )
        return "euclidean"

    raise AssertionError(f"unreachable: metric={metric!r}")  # pragma: no cover


def _check_metre_units(crs: Any) -> None:
    """Warn if a projected CRS does not use linear-metre axis units.

    ``h_max`` is documented as metres; if the CRS is projected in feet or
    degrees, the result will be wrong. We surface a ``UserWarning`` rather
    than erroring because some users legitimately work in input units (e.g.
    cartesian simulation coords with ``crs=None``); the warning gives them
    a nudge to either reproject or accept the unit mismatch consciously.
    """
    if crs is None or not crs.is_projected:
        return
    try:
        unit = crs.axis_info[0].unit_name
    except (AttributeError, IndexError):
        return
    if unit and unit not in ("metre", "meter", "metres", "meters"):
        warnings.warn(
            f"Projected CRS {crs.name!r} uses axis unit {unit!r}, not metres; "
            "h_max is interpreted in input units. To compute distances in "
            "metres, reproject to a CRS with metre axes (e.g. UTM).",
            UserWarning,
            stacklevel=3,
        )


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
    h_max_rad_factor: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Candidate pairs (upper triangle) and exact haversine distances.

    Parameters
    ----------
    coords : ndarray (n, 2)
        ``(lon, lat)`` in degrees.
    h_max : float
        Distance threshold in metres on a sphere of radius ``EARTH_RADIUS_M``.
    h_max_rad_factor : float
        Multiplier applied to the ball-tree query radius. Used by the
        geodesic path to enlarge the prefilter; a value of 1.0 gives the
        standard haversine path with byte-identical results to prior
        releases.
    """
    from sklearn.neighbors import BallTree

    n = len(coords)
    coords_rad = np.radians(coords[:, [1, 0]])  # swap lon/lat -> lat/lon
    h_max_rad = (h_max * h_max_rad_factor) / EARTH_RADIUS_M

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


def _geodesic_pairs(
    coords: np.ndarray,
    h_max: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Candidate pairs and exact WGS-84 ellipsoidal distances.

    Two-stage scheme: a haversine ball-tree on a sphere of radius
    ``EARTH_RADIUS_M`` provides the prefilter (with the radius enlarged by
    ``_HAVERSINE_GEODESIC_SAFETY``), and ``pyproj.Geod.inv`` computes the
    exact ellipsoidal distance for the surviving candidates. Pairs whose
    true geodesic distance exceeds ``h_max`` are dropped.

    The motivation: building a separate spatial index on the ellipsoid would
    require either projecting to ECEF and using cKDTree (which gets the
    chord wrong) or a custom bvh. The haversine ball-tree is already in the
    sklearn dependency we use for the haversine path, and the worst-case
    haversine-vs-geodesic discrepancy is small enough that a 0.3% inflation
    of the prefilter radius admits no false negatives.

    See ``docs/design/distance_metrics.md`` section "Two-stage prefilter"
    for the full rationale.
    """
    try:
        from pyproj import Geod
    except ImportError as e:
        raise ImportError(
            'metric="geodesic" requires pyproj. '
            "Install via `pip install gshac[geo]`."
        ) from e

    pairs, _ = _haversine_pairs(
        coords, h_max, h_max_rad_factor=_HAVERSINE_GEODESIC_SAFETY,
    )
    if len(pairs) == 0:
        return pairs, np.empty((0,), dtype=np.float64)

    geod = Geod(ellps="WGS84")
    lon1 = coords[pairs[:, 0], 0]
    lat1 = coords[pairs[:, 0], 1]
    lon2 = coords[pairs[:, 1], 0]
    lat2 = coords[pairs[:, 1], 1]
    # Geod.inv returns (forward azimuth, back azimuth, distance_m); we only
    # need the distance.
    _, _, dists = geod.inv(lon1, lat1, lon2, lat2)
    dists = np.asarray(dists, dtype=np.float64)

    # Drop pairs whose exact ellipsoidal distance exceeds h_max. The
    # haversine prefilter is conservative (admits some pairs slightly beyond
    # h_max because of the safety factor), so this final filter is required.
    keep = dists <= h_max
    return pairs[keep], dists[keep]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def spatial_dist_graph(
    coords: np.ndarray,
    h_max: float,
    metric: _MetricStr = "auto",
    crs: Any = None,
) -> dict:
    """Build a sparse geographic distance graph.

    Parameters
    ----------
    coords : ndarray, shape (n, 2)
        Feature coordinates. For a geographic CRS the ordering is
        ``(longitude, latitude)`` in degrees; for a projected CRS, ``(x, y)``
        in the CRS's linear units (metres assumed for ``h_max`` to be
        meaningful — see Notes).
    h_max : float
        Maximum distance threshold. Metres for ``"haversine"``, ``"geodesic"``,
        and ``"euclidean"`` on a metre-unit projected CRS. For ``"euclidean"``
        with a missing CRS, ``h_max`` is in input units.
    metric : {"auto", "euclidean", "haversine", "geodesic"}, default "auto"
        Distance metric.

        * ``"auto"`` — dispatch on the detected CRS: geographic ->
          ``"geodesic"``, projected -> ``"euclidean"``, missing ->
          ``"euclidean"`` with a one-time ``UserWarning``. Mirrors
          ``sf::st_distance`` (R) but defaults to ellipsoid rather than
          sphere for geographic CRSs.
        * ``"geodesic"`` — exact WGS-84 ellipsoidal distance via
          ``pyproj.Geod.inv``; uses a haversine ball-tree prefilter with a
          0.3% safety factor (see ``_HAVERSINE_GEODESIC_SAFETY``).
          Geographic / missing CRS only.
        * ``"haversine"`` — spherical great-circle distance with mean Earth
          radius ``EARTH_RADIUS_M``. Geographic / missing CRS only. Used by
          the paper's benchmarks; preserved as the fast path.
        * ``"euclidean"`` — planar L2 distance. Projected / missing CRS only.

    crs : pyproj.CRS or pyproj-acceptable spec (str, int EPSG code), optional
        Coordinate reference system metadata. Together with ``metric``
        determines the dispatch and validation matrix.

    Returns
    -------
    dict with keys ``matrix``, ``components``, ``n_components``, ``n_edges``,
    ``density``. See module docstring for details.

    Raises
    ------
    ValueError
        If ``coords`` has fewer than 2 points, the metric is unknown, or the
        ``(metric, CRS)`` combination is invalid (see the validation matrix
        in ``docs/design/distance_metrics.md``).

    Notes
    -----
    Backward compatibility: calling ``spatial_dist_graph(arr, h_max,
    metric="haversine")`` or ``metric="euclidean"`` on a raw ndarray with no
    ``crs=`` is byte-identical to the pre-CRS-aware behaviour. The paper's
    benchmark numbers depend on this.
    """
    arr, resolved_crs = _resolve_inputs(coords, crs)
    n = len(arr)
    if n < 2:
        raise ValueError("coords must have at least 2 features")

    kind = _crs_kind(resolved_crs)
    resolved_metric = _resolve_metric(metric, kind)

    # Optional unit check for projected CRSs — a non-metre unit silently
    # changes what h_max means, which is a footgun.
    if resolved_metric == "euclidean":
        _check_metre_units(resolved_crs)

    # --- candidate pairs + exact distances ---------------------------------
    if resolved_metric == "euclidean":
        pairs, dists = _euclidean_pairs(arr, h_max)
    elif resolved_metric == "haversine":
        pairs, dists = _haversine_pairs(arr, h_max)
    elif resolved_metric == "geodesic":
        pairs, dists = _geodesic_pairs(arr, h_max)
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
    metric: _MetricStr = "auto",
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
