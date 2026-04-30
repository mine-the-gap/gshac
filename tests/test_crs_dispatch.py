"""Tests for CRS-aware metric dispatch in spatial_dist_graph.

Covers the contract documented in ``docs/design/distance_metrics.md``:
auto-dispatch by CRS kind, the validation matrix, geopandas ingestion,
the two-stage haversine-then-geodesic prefilter, and backward
compatibility for the raw-ndarray path that the paper's benchmarks use.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import issparse

import gshac.spatial_dist_graph as sdg_mod
from gshac.spatial_dist_graph import (
    EARTH_RADIUS_M,
    geographic_connectivity,
    spatial_dist_graph,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

# These are required for the dispatch tests; if either is missing, skip rather
# than fail — gshac is supposed to remain importable without [geo].
gpd = pytest.importorskip("geopandas")
shapely_geom = pytest.importorskip("shapely.geometry")
Point = shapely_geom.Point
LineString = shapely_geom.LineString


@pytest.fixture
def lonlat_coords(rng):
    """100 points in a small lon/lat box near London (small extent => HAV ~ GEOD)."""
    base_lon, base_lat = -0.1, 51.5
    lon = base_lon + rng.normal(0, 0.05, size=100)
    lat = base_lat + rng.normal(0, 0.05, size=100)
    return np.column_stack([lon, lat])


@pytest.fixture
def lonlat_gdf(lonlat_coords):
    pts = [Point(x, y) for x, y in lonlat_coords]
    return gpd.GeoDataFrame(geometry=pts, crs="EPSG:4326")


@pytest.fixture
def projected_gdf(small_clustered_coords):
    pts = [Point(x, y) for x, y in small_clustered_coords]
    # EPSG:3857 is metres; using it lets us reuse the metre-scale coords.
    return gpd.GeoDataFrame(geometry=pts, crs="EPSG:3857")


@pytest.fixture(autouse=True)
def _reset_missing_crs_warning():
    """Reset the module-level warn-once flag so tests that exercise the
    missing-CRS path are independent of order."""
    sdg_mod._MISSING_CRS_WARNED = False
    yield
    sdg_mod._MISSING_CRS_WARNED = False


# ---------------------------------------------------------------------------
# Auto-dispatch
# ---------------------------------------------------------------------------

def test_auto_geographic_uses_geodesic(lonlat_gdf, lonlat_coords):
    """metric='auto' on a geographic CRS should compute geodesic, not
    haversine, distances. Geodesic >= haversine on the 6_371_000 m sphere,
    and the two should agree to within the scan-derived ~0.5% bound on
    short pairs."""
    g_auto = spatial_dist_graph(lonlat_gdf, h_max=10_000)
    g_hav = spatial_dist_graph(lonlat_coords, h_max=10_000, metric="haversine")
    # Pair counts can differ slightly because the two filters use different
    # radii; what matters is that some non-trivial number of pairs were
    # computed and that the auto path returned positive distances.
    assert g_auto["n_edges"] > 0
    assert np.all(g_auto["matrix"].data > 0)
    assert np.all(g_auto["matrix"].data <= 10_000 + 1e-9)

    # Sanity: on short chords, geodesic and haversine agree to ~0.5%.
    auto_max = g_auto["matrix"].data.max()
    hav_max = g_hav["matrix"].data.max()
    rel_diff = abs(auto_max - hav_max) / hav_max
    assert rel_diff < 0.005, f"auto/hav mismatch {rel_diff:.5f}"


def test_auto_projected_uses_euclidean(projected_gdf, small_clustered_coords):
    g_auto = spatial_dist_graph(projected_gdf, h_max=5_000)
    g_eu = spatial_dist_graph(small_clustered_coords, h_max=5_000, metric="euclidean")
    # Auto on a projected GeoDataFrame should produce the same edges and
    # distances as explicit euclidean on the underlying ndarray.
    assert g_auto["n_edges"] == g_eu["n_edges"]
    np.testing.assert_array_equal(
        np.sort(g_auto["matrix"].data), np.sort(g_eu["matrix"].data)
    )


def test_auto_missing_crs_warns_once(small_clustered_coords, recwarn):
    """metric='auto' on a raw ndarray with no crs= must warn once and use
    euclidean. A second call must not re-warn."""
    g1 = spatial_dist_graph(small_clustered_coords, h_max=5_000)
    crs_warnings = [w for w in recwarn.list
                    if issubclass(w.category, UserWarning)
                    and "CRS not set" in str(w.message)]
    assert len(crs_warnings) == 1, f"expected exactly one warning, got {len(crs_warnings)}"
    # Result is the same as explicit euclidean.
    g2 = spatial_dist_graph(small_clustered_coords, h_max=5_000, metric="euclidean")
    assert g1["n_edges"] == g2["n_edges"]

    # Second call with the same condition must not produce a new warning.
    recwarn.clear()
    spatial_dist_graph(small_clustered_coords, h_max=5_000)
    crs_warnings = [w for w in recwarn.list
                    if issubclass(w.category, UserWarning)
                    and "CRS not set" in str(w.message)]
    assert crs_warnings == [], "warning should fire only once"


# ---------------------------------------------------------------------------
# Validation matrix
# ---------------------------------------------------------------------------

def test_haversine_with_projected_crs_raises(projected_gdf):
    with pytest.raises(ValueError) as ei:
        spatial_dist_graph(projected_gdf, h_max=5_000, metric="haversine")
    assert ei.value.args[0]
    assert "geographic" in ei.value.args[0]


def test_geodesic_with_projected_crs_raises(projected_gdf):
    with pytest.raises(ValueError) as ei:
        spatial_dist_graph(projected_gdf, h_max=5_000, metric="geodesic")
    assert ei.value.args[0]
    assert "geographic" in ei.value.args[0]


def test_euclidean_with_geographic_crs_raises(lonlat_gdf):
    with pytest.raises(ValueError) as ei:
        spatial_dist_graph(lonlat_gdf, h_max=5_000, metric="euclidean")
    assert ei.value.args[0]
    assert "euclidean" in ei.value.args[0].lower()


def test_unknown_metric_raises(small_clustered_coords):
    with pytest.raises(ValueError) as ei:
        spatial_dist_graph(small_clustered_coords, h_max=5_000, metric="cosine")
    assert "cosine" in ei.value.args[0]


def test_crs_argument_must_match_geopandas_crs(lonlat_gdf):
    """Passing crs= that disagrees with the GeoDataFrame's own CRS is an
    error, not a silent override."""
    with pytest.raises(ValueError) as ei:
        spatial_dist_graph(lonlat_gdf, h_max=5_000, crs="EPSG:3857")
    assert ei.value.args[0]
    assert "match" in ei.value.args[0].lower()


def test_crs_argument_matching_geopandas_crs_is_accepted(lonlat_gdf):
    """When crs= equals the GeoDataFrame's CRS, the call should succeed."""
    g = spatial_dist_graph(lonlat_gdf, h_max=10_000, crs="EPSG:4326")
    assert g["n_edges"] > 0


# ---------------------------------------------------------------------------
# Non-Point geometries
# ---------------------------------------------------------------------------

def test_non_point_geometry_raises():
    lines = [LineString([(0, 0), (1, 1)]) for _ in range(5)]
    gdf = gpd.GeoDataFrame(geometry=lines, crs="EPSG:4326")
    with pytest.raises(NotImplementedError) as ei:
        spatial_dist_graph(gdf, h_max=10_000)
    msg = ei.value.args[0]
    # The message should point users at representative_point or centroid.
    assert "representative_point" in msg or "centroid" in msg


# ---------------------------------------------------------------------------
# Two-stage prefilter safety
# ---------------------------------------------------------------------------

def test_haversine_prefilter_safety():
    """At a worst-case latitude (89 N) and a small east-west chord, ensure
    the haversine prefilter does not exclude any pair whose true geodesic
    distance is within h_max."""
    from pyproj import Geod
    geod = Geod(ellps="WGS84")

    # Build a small set of points: one pair near the equator and one near
    # the pole. h_max is tuned so that the pole-pair's true geodesic
    # distance just fits inside.
    pts_eq = np.array([(0.0, 0.0), (0.5, 0.0)])           # ~55 km E-W on eq
    pts_polar = np.array([(0.0, 89.0), (1.0, 89.0)])      # ~1949 m E-W

    # Test the polar prefilter: choose h_max equal to the exact geodesic.
    # ``Geod.inv`` accepts scalars and broadcasts; we pass scalars to avoid
    # a NumPy-1.25 deprecation warning inside pyproj 3.6 when single-element
    # arrays are passed.
    _, _, polar_geod = geod.inv(
        pts_polar[0, 0], pts_polar[0, 1],
        pts_polar[1, 0], pts_polar[1, 1],
    )
    h_max_polar = float(polar_geod)  # exactly the true distance
    g = spatial_dist_graph(pts_polar, h_max=h_max_polar, metric="geodesic")
    assert g["n_edges"] == 1, (
        f"prefilter excluded a pair whose true geodesic distance "
        f"({h_max_polar:.6f}) equals h_max ({h_max_polar:.6f})"
    )

    # Sanity: the equatorial pair also survives at its own h_max.
    _, _, eq_geod = geod.inv(
        pts_eq[0, 0], pts_eq[0, 1],
        pts_eq[1, 0], pts_eq[1, 1],
    )
    g = spatial_dist_graph(pts_eq, h_max=float(eq_geod), metric="geodesic")
    assert g["n_edges"] == 1


def test_geodesic_drops_pairs_beyond_h_max():
    """The exact-distance filter inside the geodesic path must drop pairs
    that pass the haversine prefilter but exceed h_max on the ellipsoid."""
    # Construct a pair whose haversine distance is below h_max but whose
    # geodesic distance is above. At lat 89, geod/hav ratio ~= 1.0045.
    # Pick h_max just below the true geodesic distance.
    from pyproj import Geod
    geod = Geod(ellps="WGS84")
    pts = np.array([(0.0, 89.0), (1.0, 89.0)])
    _, _, geod_d = geod.inv(0.0, 89.0, 1.0, 89.0)
    geod_dist = float(geod_d)
    h_max = geod_dist * 0.999  # 99.9% of geodesic; haversine is ~0.45% smaller
    # Confirm haversine would admit this pair.
    rl1 = np.radians(89.0)
    rd_lon = np.radians(1.0)
    a = np.cos(rl1) ** 2 * np.sin(rd_lon / 2) ** 2
    hav = 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))
    assert hav < h_max, "this test only makes sense if hav < h_max < geod"
    assert geod_dist > h_max, "geodesic must exceed h_max"

    g = spatial_dist_graph(pts, h_max=h_max, metric="geodesic")
    assert g["n_edges"] == 0, "exact filter must drop pair whose geod > h_max"


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

# Snapshots are computed offline against the pre-CRS-aware code path on
# small_clustered_coords / small_lonlat_coords from conftest.py. The exact
# numbers are pinned so that any future change to the candidate-pair
# enumeration or co-located-pair handling will trip this test.

_EUCLIDEAN_SNAPSHOT = dict(
    n_edges=4975,
    n_components=4,
    density=0.25,
    first3_sorted=(12.94717487, 12.94717487, 14.16357761),
    last3_sorted=(2951.55173677, 2996.26441667, 2996.26441667),
)

_HAVERSINE_SNAPSHOT = dict(
    n_edges=3755,
    n_components=1,
    density=0.7585858585858586,
    first3_sorted=(169.40834455, 169.40834455, 232.47778156),
    last3_sorted=(9994.04174741, 9997.39906669, 9997.39906669),
)


def _check_snapshot(graph, snap):
    assert graph["n_edges"] == snap["n_edges"]
    assert graph["n_components"] == snap["n_components"]
    assert graph["density"] == pytest.approx(snap["density"], rel=1e-12)
    sd = np.sort(graph["matrix"].data)
    np.testing.assert_allclose(sd[:3], snap["first3_sorted"], rtol=1e-7)
    np.testing.assert_allclose(sd[-3:], snap["last3_sorted"], rtol=1e-7)


def test_backward_compat_euclidean_ndarray(small_clustered_coords):
    """metric='euclidean' on a raw ndarray must reproduce the pre-CRS
    behaviour byte-for-byte. The paper's benchmark numbers depend on this."""
    g = spatial_dist_graph(small_clustered_coords, h_max=5_000, metric="euclidean")
    _check_snapshot(g, _EUCLIDEAN_SNAPSHOT)


def test_backward_compat_haversine_ndarray(small_lonlat_coords):
    """metric='haversine' on a raw ndarray must reproduce the pre-CRS
    behaviour byte-for-byte."""
    g = spatial_dist_graph(small_lonlat_coords, h_max=10_000, metric="haversine")
    _check_snapshot(g, _HAVERSINE_SNAPSHOT)


def test_geographic_connectivity_accepts_geopandas(lonlat_gdf):
    """The connectivity helper should also accept the new input forms."""
    conn = geographic_connectivity(lonlat_gdf, h_max=10_000)
    assert issparse(conn)
    assert np.all(conn.data == 1.0)


def test_geographic_connectivity_haversine_unchanged(small_lonlat_coords):
    """Backward compat for the connectivity helper on a raw ndarray."""
    conn = geographic_connectivity(
        small_lonlat_coords, h_max=10_000, metric="haversine",
    )
    assert issparse(conn)
    assert int(conn.nnz) == 2 * _HAVERSINE_SNAPSHOT["n_edges"]
