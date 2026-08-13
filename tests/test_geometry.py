"""Tests for non-point geometry support (polygons, lines, mixed).

Validates the GEOS boundary-to-boundary distance path against brute-force
shapely distance matrices: edge-set completeness of the STRtree ``dwithin``
enumeration, exact partition agreement with dense HAC on the true distance
matrix for single/complete/average linkage, the distance-0 convention for
touching features, the Ward refusal, and the CRS validation matrix.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score

from gshac.spatial_dist_graph import spatial_dist_graph
from gshac.sparse_hclust import sparse_hclust

gpd = pytest.importorskip("geopandas")
shapely = pytest.importorskip("shapely")
from shapely.geometry import LineString, Point, Polygon  # noqa: E402

PROJ_CRS = "EPSG:3857"  # metre-unit projected CRS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _square(x, y, side=1_000.0):
    return Polygon([(x, y), (x + side, y), (x + side, y + side), (x, y + side)])


def _polygon_clusters():
    """Three clusters of squares, 40 km apart; within-cluster gaps < 3 km.

    Cluster geometry is chosen so that each cluster is one connected
    component at h_max = 5 km but contains at least one intra-component
    pair farther apart than h_max (chain layout), exercising the exact
    dense recomputation for complete/average linkage.
    """
    polys = []
    for cx, cy in [(0.0, 0.0), (40_000.0, 0.0), (0.0, 40_000.0)]:
        # chain of 4 squares spaced 3 km apart: consecutive gaps 2 km
        # (edge-to-edge), endpoints ~8 km apart edge-to-edge (> h_max).
        for k in range(4):
            polys.append(_square(cx + k * 3_000.0, cy))
    return gpd.GeoSeries(polys, crs=PROJ_CRS)


def _brute_force_matrix(geoms):
    arr = geoms.to_numpy()
    return np.asarray(shapely.distance(arr[:, None], arr[None, :]), dtype=np.float64)


def _dense_labels(dmat, h_cut, method):
    Z = linkage(squareform(dmat, checks=False), method=method)
    return fcluster(Z, t=h_cut, criterion="distance")


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def test_polygon_graph_edges_match_brute_force():
    geoms = _polygon_clusters()
    h_max = 5_000.0
    g = spatial_dist_graph(geoms, h_max)

    dmat = _brute_force_matrix(geoms)
    iu = np.triu_indices(len(geoms), k=1)
    expected = {(int(i), int(j)) for i, j in zip(*iu) if dmat[i, j] <= h_max}

    coo = g["matrix"].tocoo()
    got = {(int(i), int(j)) for i, j in zip(coo.row, coo.col) if i < j}
    assert got == expected

    # Stored weights are the true GEOS distances (none are 0 here).
    for i, j in got:
        assert g["matrix"][i, j] == pytest.approx(dmat[i, j], rel=1e-12)

    assert g["feature_type"] == "geometry"
    assert g["geometries"] is not None
    assert g["n_components"] == 3


def test_touching_polygons_distance_zero_convention():
    # Two squares sharing an edge (distance 0) plus one far square.
    geoms = gpd.GeoSeries(
        [_square(0, 0), _square(1_000, 0), _square(50_000, 0)], crs=PROJ_CRS
    )
    g = spatial_dist_graph(geoms, h_max=2_000.0)
    # Touching pair is connected; stored as 1 unit (CSR cannot hold
    # explicit zeros), far square is a singleton component.
    assert g["matrix"][0, 1] == 1.0
    assert g["n_components"] == 2
    res = sparse_hclust(g, [2_000.0], method="single")
    labels = res["labels"][2_000.0]
    assert labels[0] == labels[1]
    assert labels[0] != labels[2]


def test_geometry_requires_projected_crs():
    geoms = gpd.GeoSeries([_square(0, 0), _square(1, 1, side=1.0)], crs="EPSG:4326")
    with pytest.raises(ValueError, match="projected CRS"):
        spatial_dist_graph(geoms, h_max=1_000.0)


@pytest.mark.parametrize("metric", ["haversine", "geodesic"])
def test_geometry_rejects_spherical_metrics(metric):
    geoms = gpd.GeoSeries([_square(0, 0), _square(3_000, 0)], crs=PROJ_CRS)
    with pytest.raises(ValueError, match="not supported for non-point"):
        spatial_dist_graph(geoms, h_max=1_000.0, metric=metric)


def test_lines_and_mixed_geometries():
    geoms = gpd.GeoSeries(
        [
            LineString([(0, 0), (2_000, 0)]),
            Point(2_500, 0),                    # 500 m from line end
            _square(4_000, 0),                  # 1.5 km from the point
            LineString([(60_000, 0), (62_000, 0)]),
        ],
        crs=PROJ_CRS,
    )
    g = spatial_dist_graph(geoms, h_max=2_000.0)
    assert g["matrix"][0, 1] == pytest.approx(500.0)
    assert g["matrix"][1, 2] == pytest.approx(1_500.0)
    assert g["n_components"] == 2  # chain of three + far line


# ---------------------------------------------------------------------------
# Exactness vs dense HAC on the brute-force matrix
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["single", "complete", "average"])
def test_polygon_partition_matches_dense(method):
    geoms = _polygon_clusters()
    h_max = 5_000.0
    g = spatial_dist_graph(geoms, h_max)
    dmat = _brute_force_matrix(geoms)

    for h_cut in [1_500.0, 2_500.0, 5_000.0]:
        res = sparse_hclust(g, [h_cut], method=method)
        expected = _dense_labels(dmat, h_cut, method)
        assert adjusted_rand_score(res["labels"][h_cut], expected) == 1.0, (
            f"{method} @ {h_cut}"
        )


def test_polygon_dense_path_exercises_beyond_hmax_pairs():
    """The chain clusters contain intra-component pairs > h_max, so the
    exact GEOS sub-matrix recomputation (not the sentinel) must run —
    no approximation warning may be raised for average linkage."""
    geoms = _polygon_clusters()
    g = spatial_dist_graph(geoms, h_max=5_000.0)
    dmat = _brute_force_matrix(geoms)
    comp = g["components"]
    over = [
        (i, j)
        for i in range(len(geoms))
        for j in range(i + 1, len(geoms))
        if comp[i] == comp[j] and dmat[i, j] > 5_000.0
    ]
    assert over, "fixture must contain intra-component pairs beyond h_max"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sparse_hclust(g, [4_000.0], method="average")


def test_ward_refused_for_geometries():
    geoms = _polygon_clusters()
    g = spatial_dist_graph(geoms, h_max=5_000.0)
    with pytest.raises(ValueError, match="Ward linkage requires point features"):
        sparse_hclust(g, [2_000.0], method="ward")


def test_all_point_geoseries_still_uses_point_path():
    """All-Point geopandas input keeps the coordinate fast path (no
    geometry payload in the graph), preserving pre-1.1 behaviour."""
    pts = gpd.GeoSeries([Point(0, 0), Point(3_000, 0), Point(50_000, 0)],
                        crs=PROJ_CRS)
    g = spatial_dist_graph(pts, h_max=5_000.0)
    assert g["feature_type"] == "point"
    assert g["geometries"] is None
    assert g["matrix"][0, 1] == pytest.approx(3_000.0)
