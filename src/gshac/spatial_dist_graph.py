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
spatial_dist_graph(coords, h_max, metric="auto", crs=None)
    Full sparse distance graph (distances stored as edge weights).

geographic_connectivity(coords, h_max, metric="auto", crs=None)
    Binary connectivity matrix for use with
    ``sklearn.cluster.AgglomerativeClustering(connectivity=...)``.

CRS-aware dispatch
------------------
The ``metric`` argument is conditional on the coordinate reference
system, mirroring the design of R's ``sf::st_distance``. With
``metric="auto"`` (the default), gshac dispatches as follows:

* geographic CRS -> ``"geodesic"`` (ellipsoidal, via ``pyproj.Geod.inv``)
* projected CRS  -> ``"euclidean"``
* missing CRS    -> ``"euclidean"`` with a one-time ``UserWarning``

Unlike sf, the default for geographic CRS is the ellipsoid, not the
sphere; haversine remains explicitly available as a fast path. The
geodesic path uses the haversine ball-tree as a prefilter and verifies
each surviving candidate with ``pyproj.Geod.inv``.

Dependencies
------------
* numpy, scipy, scikit-learn (always)
* pyproj (only when ``metric`` resolves to ``"geodesic"`` or ``crs=`` is
  set; install via ``pip install gshac[geo]``)
* geopandas (only when passing a ``GeoSeries`` / ``GeoDataFrame``;
  install via ``pip install gshac[geo]``)
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
# haversine ball-tree. The worst-case ratio geodesic/haversine on a
# 6_371_000 m sphere reaches ~1.00449 at high latitudes for short east-west
# chords (empirical scan of (lat, dlat, dlon) on WGS-84); 1.005 covers this
# with a small margin for numerical wobble in BallTree's haversine kernel.
# Pinned by ``tests/test_crs_dispatch.py::test_haversine_prefilter_safety``.
_HAVERSINE_GEODESIC_SAFETY = 1.005


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
    """Normalise ``coords`` and ``crs`` into ``(ndarray, pyproj.CRS|None)``.

    Accepts a raw ``(n, 2)`` ndarray, a ``geopandas.GeoSeries`` of points,
    or a ``geopandas.GeoDataFrame``. If ``coords`` carries its own CRS,
    the user-supplied ``crs=`` must either be ``None`` or match exactly —
    a mismatch indicates a copy-paste bug, not a useful override.
    """
    geo_coords, geo_crs = _try_extract_from_geopandas(coords)
    if geo_coords is not None:
        if crs is not None:
            user_crs = _coerce_crs(crs)
            if geo_crs is None:
                # Caller passed a CRS for a geopandas object that has none;
                # treat that as informational and adopt it.
                geo_crs = user_crs
            elif user_crs != geo_crs:
                raise ValueError(
                    "crs= argument does not match the CRS of the geopandas "
                    f"input: argument={user_crs!r} vs input.crs={geo_crs!r}. "
                    "Pass crs=None or omit to inherit the input's CRS."
                )
        return geo_coords, geo_crs

    arr = np.asarray(coords, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(
            f"coords must be shape (n, 2); got shape {arr.shape!r}."
        )
    return arr, _coerce_crs(crs)


def _try_extract_from_geopandas(
    coords: Any,
) -> Tuple[Optional[np.ndarray], Optional[Any]]:
    """If ``coords`` is a geopandas object, return ``(arr, crs)``.

    Otherwise return ``(None, None)``. ``geopandas`` is imported lazily so
    that gshac remains importable without the ``[geo]`` extras.

    Non-Point geometries raise ``NotImplementedError`` with a message
    pointing at ``representative_point()`` / ``centroid``. Point-only is
    intentional for v0.x: HAC on lines and polygons requires a Hausdorff
    or Frechet kernel that is out of scope for this release.
    """
    # Check by class name first to avoid importing geopandas for ndarray inputs.
    cls_name = type(coords).__name__
    if cls_name not in ("GeoSeries", "GeoDataFrame"):
        return None, None

    try:
        import geopandas as gpd
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "geopandas is required to pass GeoSeries / GeoDataFrame inputs. "
            "Install via `pip install gshac[geo]`."
        ) from e

    if isinstance(coords, gpd.GeoDataFrame):
        geom = coords.geometry
    elif isinstance(coords, gpd.GeoSeries):
        geom = coords
    else:  # pragma: no cover - defensive, matched by class-name check above
        return None, None

    geom_types = set(geom.geom_type.unique())
    if geom_types - {"Point"}:
        raise NotImplementedError(
            "Only Point geometries are supported in v0.x; got "
            f"{sorted(geom_types)!r}. For polygons/lines, use "
            "`geom.representative_point()` or `geom.centroid` to obtain a "
            "Point approximation. Hausdorff/Frechet distance for non-point "
            "geometries is future work."
        )

    # Extract (x, y) — this is (lon, lat) for geographic CRSs.
    xs = geom.x.to_numpy(dtype=np.float64)
    ys = geom.y.to_numpy(dtype=np.float64)
    arr = np.column_stack([xs, ys])
    return arr, geom.crs


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
    chord wrong) or a custom bvh. The haversine ball-tree is already used
    by the haversine path, and the worst-case haversine-vs-geodesic
    discrepancy is small enough that a ~0.5% inflation of the prefilter
    radius admits no false negatives.
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
    coords: Union[np.ndarray, "Any"],
    h_max: float,
    metric: _MetricStr = "auto",
    crs: Any = None,
) -> dict:
    """Build a sparse geographic distance graph.

    Parameters
    ----------
    coords : ndarray of shape (n, 2), or geopandas.GeoSeries, or geopandas.GeoDataFrame
        Feature coordinates. For a ndarray with a geographic CRS the
        ordering is ``(longitude, latitude)`` in degrees; for a projected
        CRS, ``(x, y)`` in the CRS's linear units (metres assumed for
        ``h_max`` to be meaningful — see Notes). For ``GeoSeries`` /
        ``GeoDataFrame`` inputs, only Point geometries are supported in
        v0.x; non-Point inputs raise ``NotImplementedError`` with a
        suggestion to use ``representative_point()`` or ``centroid``.
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
          0.5% safety factor (see ``_HAVERSINE_GEODESIC_SAFETY``).
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
        ``(metric, CRS)`` combination is invalid (e.g. ``"euclidean"`` on a
        geographic CRS, or ``"haversine"``/``"geodesic"`` on a projected CRS).
    NotImplementedError
        If a non-Point geometry is passed via ``GeoSeries`` /
        ``GeoDataFrame``.
    ImportError
        If ``metric`` resolves to ``"geodesic"`` but ``pyproj`` is not
        installed, or a geopandas input is passed but ``geopandas`` is not
        installed.

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
    coords: Union[np.ndarray, "Any"],
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
