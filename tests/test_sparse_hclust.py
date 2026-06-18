"""Tests for sparse_hclust, dense_hclust, stitch_linkage, and the sklearn API."""

import warnings

import numpy as np
import pytest
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import adjusted_rand_score

from gshac.spatial_dist_graph import spatial_dist_graph
from gshac.sparse_hclust import (
    sparse_hclust,
    dense_hclust,  # not part of public API; tested here as benchmark baseline
    stitch_linkage,
    SparseAgglomerativeClustering,
)


# ---------------------------------------------------------------------------
# sparse_hclust — basic
# ---------------------------------------------------------------------------

def test_labels_shape(small_clustered_coords):
    graph = spatial_dist_graph(small_clustered_coords, h_max=5_000)
    result = sparse_hclust(graph, h_cuts=[2_000, 5_000])
    assert set(result["labels"].keys()) == {2_000.0, 5_000.0}
    for labels in result["labels"].values():
        assert labels.shape == (200,)


def test_labels_are_positive_integers(small_clustered_coords):
    graph = spatial_dist_graph(small_clustered_coords, h_max=5_000)
    result = sparse_hclust(graph, h_cuts=[2_000])
    labels = result["labels"][2_000.0]
    assert labels.dtype == np.int64
    assert np.all(labels >= 1)


def test_cluster_count_monotone(small_clustered_coords):
    """Fewer clusters at larger cut heights."""
    graph = spatial_dist_graph(small_clustered_coords, h_max=20_000)
    result = sparse_hclust(graph, h_cuts=[1_000, 5_000, 15_000])
    n1 = len(np.unique(result["labels"][1_000.0]))
    n5 = len(np.unique(result["labels"][5_000.0]))
    n15 = len(np.unique(result["labels"][15_000.0]))
    assert n1 >= n5 >= n15


def test_custom_ids(small_clustered_coords):
    n = len(small_clustered_coords)
    ids = [f"pt_{i}" for i in range(n)]
    graph = spatial_dist_graph(small_clustered_coords, h_max=5_000)
    result = sparse_hclust(graph, h_cuts=[2_000], ids=ids)
    assert result["ids"] == ids


def test_all_singletons_no_edges():
    coords = np.array([[0.0, 0.0], [1_000.0, 0.0], [0.0, 1_000.0]])
    graph = spatial_dist_graph(coords, h_max=1.0)
    result = sparse_hclust(graph, h_cuts=[0.5])
    labels = result["labels"][0.5]
    assert len(np.unique(labels)) == 3


def test_results_field_keys(small_clustered_coords):
    graph = spatial_dist_graph(small_clustered_coords, h_max=5_000)
    result = sparse_hclust(graph, h_cuts=[2_000])
    assert "ids" in result
    assert "components" in result
    assert "labels" in result
    assert "linkage_trees" not in result  # not requested


# ---------------------------------------------------------------------------
# C extension vs Python fallback produce identical results
# ---------------------------------------------------------------------------

def test_c_and_python_linkage_agree(monkeypatch):
    """C union-find linkage and Python fallback must produce the same Z matrix."""
    import sys
    import gshac.sparse_hclust

    rng = np.random.default_rng(0)
    coords = rng.uniform(0, 10_000, size=(40, 2))
    graph = spatial_dist_graph(coords, h_max=20_000)

    monkeypatch.setattr(sys.modules["gshac.sparse_hclust"], "_GSHAC_C", True)
    result_c = sparse_hclust(graph, h_cuts=[3_000, 8_000], method="single")

    monkeypatch.setattr(sys.modules["gshac.sparse_hclust"], "_GSHAC_C", False)
    result_py = sparse_hclust(graph, h_cuts=[3_000, 8_000], method="single")

    for h in [3_000.0, 8_000.0]:
        ari = adjusted_rand_score(result_c["labels"][h], result_py["labels"][h])
        assert ari == 1.0, f"C and Python paths disagree at h={h} (ARI={ari:.4f})"


def test_c_and_python_fcluster_batch_agree(monkeypatch):
    """C fcluster_batch and scipy fcluster fallback must produce the same labels."""
    import sys
    import gshac.sparse_hclust

    rng = np.random.default_rng(5)
    coords = rng.uniform(0, 10_000, size=(50, 2))
    graph = spatial_dist_graph(coords, h_max=20_000)
    h_cuts = [2_000, 5_000, 9_000]

    monkeypatch.setattr(sys.modules["gshac.sparse_hclust"], "_GSHAC_C", True)
    result_c = sparse_hclust(graph, h_cuts=h_cuts, method="complete")

    monkeypatch.setattr(sys.modules["gshac.sparse_hclust"], "_GSHAC_C", False)
    result_py = sparse_hclust(graph, h_cuts=h_cuts, method="complete")

    for h in [float(x) for x in h_cuts]:
        ari = adjusted_rand_score(result_c["labels"][h], result_py["labels"][h])
        assert ari == 1.0, f"C and Python fcluster paths disagree at h={h} (ARI={ari:.4f})"


# ---------------------------------------------------------------------------
# Exactness: sparse vs dense produce the same cluster counts
# ---------------------------------------------------------------------------

def test_sparse_matches_dense(backend):
    rng = np.random.default_rng(7)
    coords = rng.uniform(0, 50_000, size=(300, 2))
    h_max = 10_000
    h_cuts = [2_000, 5_000, 8_000]

    graph = spatial_dist_graph(coords, h_max=h_max)
    sp = sparse_hclust(graph, h_cuts=h_cuts)
    dn = dense_hclust(coords, h_cuts=h_cuts)

    for h in h_cuts:
        n_sp = len(np.unique(sp["labels"][float(h)]))
        n_dn = len(np.unique(dn["labels"][float(h)]))
        assert n_sp == n_dn, f"Mismatch at h={h}: sparse={n_sp}, dense={n_dn}"


def test_sparse_matches_dense_exact_labels(backend):
    """Not just cluster counts — verify identical cluster memberships."""
    rng = np.random.default_rng(99)
    coords = rng.uniform(0, 30_000, size=(150, 2))
    h_max = 15_000
    h_cuts = [3_000, 7_000, 12_000]

    graph = spatial_dist_graph(coords, h_max=h_max)
    sp = sparse_hclust(graph, h_cuts=h_cuts)
    dn = dense_hclust(coords, h_cuts=h_cuts)

    for h in h_cuts:
        sp_labels = sp["labels"][float(h)]
        dn_labels = dn["labels"][float(h)]
        # Labels may differ in numbering, but the partition must be identical.
        # Two points share a label in sparse iff they share a label in dense.
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                same_sp = sp_labels[i] == sp_labels[j]
                same_dn = dn_labels[i] == dn_labels[j]
                assert same_sp == same_dn, (
                    f"h={h}, pts ({i},{j}): sparse_same={same_sp}, dense_same={same_dn}"
                )


@pytest.mark.parametrize("method", ["single", "complete", "average", "ward"])
def test_sparse_matches_dense_all_linkages(backend, method):
    """With coords=, every linkage (incl. average) matches the dense baseline.

    The threshold graph omits > h_max intra-component pairs; average linkage
    averages those distances into its merge heights, so it is exact only when
    the full per-component sub-matrix is recomputed from coordinates (the
    coords= path). Regression guard for that path.
    """
    rng = np.random.default_rng(11)
    coords = rng.uniform(0, 50_000, size=(300, 2))
    h_max = 20_000
    h_cuts = [5_000, 10_000, 15_000]

    graph = spatial_dist_graph(coords, h_max=h_max)
    sp = sparse_hclust(graph, h_cuts=h_cuts, method=method, coords=coords)
    dn = dense_hclust(coords, h_cuts=h_cuts, method=method)

    for h in h_cuts:
        n_sp = len(np.unique(sp["labels"][float(h)]))
        n_dn = len(np.unique(dn["labels"][float(h)]))
        assert n_sp == n_dn, f"{method} mismatch at h={h}: sparse={n_sp}, dense={n_dn}"


def test_haversine_average_matches_dense(backend):
    """Exact average linkage on a haversine graph via the coords= path."""
    rng = np.random.default_rng(5)
    coords = np.column_stack([rng.uniform(-1.0, 1.0, 250),
                              rng.uniform(50.0, 51.0, 250)])
    h_cuts = [10_000, 20_000]
    graph = spatial_dist_graph(coords, h_max=30_000, metric="haversine")
    sp = sparse_hclust(graph, h_cuts=h_cuts, method="average", coords=coords)
    dn = dense_hclust(coords, h_cuts=h_cuts, method="average", metric="haversine")
    for h in h_cuts:
        assert len(np.unique(sp["labels"][float(h)])) == \
               len(np.unique(dn["labels"][float(h)]))


def test_average_without_coords_warns_and_validates_shape():
    """Average linkage without coords= warns (approximate); bad coords raise."""
    rng = np.random.default_rng(11)
    coords = rng.uniform(0, 50_000, size=(300, 2))
    graph = spatial_dist_graph(coords, h_max=20_000)
    with pytest.warns(UserWarning, match="average linkage without coords"):
        sparse_hclust(graph, h_cuts=[10_000], method="average")
    with pytest.raises(ValueError, match=r"coords must be shape"):
        sparse_hclust(graph, h_cuts=[10_000], method="average",
                      coords=coords[:, :1])


def _chain_with_long_intra_pairs():
    """A single connected component whose graph omits intra-component pairs.

    Points are spaced so that consecutive points are within ``h_max`` (chain
    connectivity, one component) but the endpoints are farther apart than
    ``h_max``. The threshold graph therefore drops those long intra-component
    pairs, so the no-coords dense reconstruction must fill them with a
    sentinel (``zero_mask.any()`` is True). Returns ``(coords, graph, h_max)``.
    """
    # 6 collinear points, 4 km apart; h_max=5 km links neighbours only.
    coords = np.array([[float(i) * 4_000.0, 0.0] for i in range(6)])
    h_max = 5_000.0
    graph = spatial_dist_graph(coords, h_max=h_max)
    assert graph["n_components"] == 1, "must be a single connected component"
    return coords, graph, h_max


@pytest.mark.parametrize("method", ["complete", "ward"])
def test_no_coords_fallback_non_average_sentinel(method):
    """No-coords dense fallback for non-average linkage hits the sentinel fill.

    Exercises the ``method != "average"`` branch of the absent-pair sentinel
    fill (no UserWarning is raised). The chained component has intra-component
    pairs beyond ``h_max`` that the threshold graph omits, so ``zero_mask`` is
    non-empty and the sentinel-fill path runs.
    """
    coords, graph, _ = _chain_with_long_intra_pairs()
    # Must NOT warn for non-average linkage.
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # turn any warning into an error
        result = sparse_hclust(graph, h_cuts=[4_500], method=method)
    labels = result["labels"][4_500.0]
    assert labels.shape == (6,)
    # The two chain endpoints (> h_max apart) must never share a cluster at a
    # cut height <= h_max: the sentinel keeps them separated.
    assert labels[0] != labels[-1]


def test_geodesic_full_distance_matrix_matches_pyproj():
    """The geodesic _full_distance_matrix equals an independent pyproj.Geod ref."""
    pyproj = pytest.importorskip("pyproj")
    from gshac.sparse_hclust import _full_distance_matrix

    rng = np.random.default_rng(3)
    coords = np.column_stack([rng.uniform(-1.0, 1.0, 40),
                              rng.uniform(50.0, 51.0, 40)])
    geod = pyproj.Geod(ellps="WGS84")
    n = len(coords)
    ref = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            _, _, d = geod.inv(coords[i, 0], coords[i, 1],
                               coords[j, 0], coords[j, 1])
            ref[i, j] = ref[j, i] = d

    fdm = _full_distance_matrix(coords, "geodesic")
    assert fdm.shape == (n, n)
    np.testing.assert_allclose(np.diag(fdm), 0.0, atol=1e-9)
    np.testing.assert_allclose(fdm, ref, rtol=1e-9, atol=1e-6)


@pytest.mark.parametrize("method", ["complete", "average", "ward"])
def test_geodesic_coords_path_matches_dense_reference(backend, method):
    """Exact non-single linkage on a geodesic graph via the coords= path.

    dense_hclust has no geodesic backend, so the reference is a dense scipy
    linkage built from a full geodesic distance matrix computed independently
    with pyproj.Geod. For complete/average/ward, cross-component pairs are all
    > h_max, so the per-component coords= result must match the dense reference.
    """
    pyproj = pytest.importorskip("pyproj")
    from scipy.cluster.hierarchy import linkage as _scipy_linkage

    rng = np.random.default_rng(8)
    coords = np.column_stack([rng.uniform(-1.0, 1.0, 90),
                              rng.uniform(50.0, 51.0, 90)])
    h_max = 40_000
    h_cuts = [10_000, 20_000]

    geod = pyproj.Geod(ellps="WGS84")
    n = len(coords)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            _, _, d = geod.inv(coords[i, 0], coords[i, 1],
                               coords[j, 0], coords[j, 1])
            D[i, j] = D[j, i] = d

    graph = spatial_dist_graph(coords, h_max=h_max, metric="geodesic")
    sp = sparse_hclust(graph, h_cuts=h_cuts, method=method, coords=coords)

    Zref = _scipy_linkage(squareform(D, checks=False), method=method)
    for h in h_cuts:
        n_sp = len(np.unique(sp["labels"][float(h)]))
        n_ref = len(np.unique(fcluster(Zref, t=h, criterion="distance")))
        assert n_sp == n_ref, f"{method} geodesic mismatch at h={h}: {n_sp} vs {n_ref}"


def test_full_distance_matrix_unknown_metric_raises():
    """_full_distance_matrix rejects an unknown metric with a clear ValueError."""
    from gshac.sparse_hclust import _full_distance_matrix

    coords = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
    with pytest.raises(ValueError, match=r"Unknown metric for exact sub-matrix"):
        _full_distance_matrix(coords, "manhattan")


def test_unknown_graph_metric_propagates_to_full_distance_matrix():
    """A graph carrying an unknown metric raises via the coords= dense path.

    The metric stored on the graph dict is what sparse_hclust passes to
    _full_distance_matrix; an unsupported value surfaces as a ValueError from
    the non-single (coords=) dense path, not a silent wrong answer.
    """
    coords, graph, _ = _chain_with_long_intra_pairs()
    graph = dict(graph)            # shallow copy; don't mutate the fixture graph
    graph["metric"] = "manhattan"  # unsupported by _full_distance_matrix
    with pytest.raises(ValueError, match=r"Unknown metric for exact sub-matrix"):
        sparse_hclust(graph, h_cuts=[4_500], method="complete", coords=coords)


def test_euclidean_full_distance_matrix_matches_cdist():
    """The euclidean _full_distance_matrix branch equals scipy cdist."""
    from scipy.spatial.distance import cdist
    from gshac.sparse_hclust import _full_distance_matrix

    rng = np.random.default_rng(1)
    coords = rng.uniform(0, 10_000, size=(25, 2))
    fdm = _full_distance_matrix(coords, "euclidean")
    np.testing.assert_allclose(fdm, cdist(coords, coords, metric="euclidean"))


# ---------------------------------------------------------------------------
# return_linkage and stitch_linkage
# ---------------------------------------------------------------------------

def test_return_linkage_flag(small_clustered_coords):
    graph = spatial_dist_graph(small_clustered_coords, h_max=5_000)
    result = sparse_hclust(graph, h_cuts=[2_000], return_linkage=True)
    assert "linkage_trees" in result
    assert isinstance(result["linkage_trees"], list)
    for Z, idx in result["linkage_trees"]:
        assert Z.ndim == 2 and Z.shape[1] == 4
        assert Z.shape[0] == len(idx) - 1
        assert idx.ndim == 1


def test_stitch_linkage_shape(small_clustered_coords):
    graph = spatial_dist_graph(small_clustered_coords, h_max=5_000)
    result = sparse_hclust(graph, h_cuts=[2_000], return_linkage=True)
    Z = stitch_linkage(result)
    n = len(small_clustered_coords)
    assert Z.shape == (n - 1, 4)


def test_stitch_linkage_valid_structure(small_clustered_coords):
    """Stitched Z should be a valid scipy linkage matrix."""
    graph = spatial_dist_graph(small_clustered_coords, h_max=5_000)
    result = sparse_hclust(graph, h_cuts=[2_000], return_linkage=True)
    Z = stitch_linkage(result)
    n = len(small_clustered_coords)

    # All leaf references (< n) and internal references (>= n) must be valid.
    all_refs = Z[:, :2].ravel()
    assert np.all(all_refs >= 0)
    assert np.all(all_refs < 2 * n - 1)

    # Last row count should equal n.
    assert Z[-1, 3] == n

    # Distances should be non-decreasing within each component's block,
    # and the inf merges come last.
    finite_mask = np.isfinite(Z[:, 2])
    inf_mask = ~finite_mask
    if inf_mask.any():
        # All inf rows should come after all finite rows.
        first_inf = np.argmax(inf_mask)
        assert np.all(finite_mask[:first_inf])
        assert np.all(inf_mask[first_inf:])


def test_stitch_linkage_fcluster_matches_sparse():
    """Cutting the stitched Z at a given height should produce the same
    cluster count as sparse_hclust."""
    rng = np.random.default_rng(42)
    coords = rng.uniform(0, 50_000, size=(200, 2))
    h_max = 10_000
    h_cuts = [3_000, 7_000]

    graph = spatial_dist_graph(coords, h_max=h_max)
    result = sparse_hclust(graph, h_cuts=h_cuts, return_linkage=True)
    Z = stitch_linkage(result)

    for h in h_cuts:
        from_sparse = len(np.unique(result["labels"][float(h)]))
        from_Z = len(np.unique(fcluster(Z, t=h, criterion="distance")))
        assert from_sparse == from_Z, (
            f"h={h}: sparse_hclust gives {from_sparse} clusters, "
            f"fcluster(Z) gives {from_Z}"
        )


def test_stitch_linkage_all_singletons():
    """When all points are singletons, stitched Z has only inf merges."""
    coords = np.array([[0.0, 0.0], [1e6, 0.0], [0.0, 1e6]])
    graph = spatial_dist_graph(coords, h_max=1.0)
    result = sparse_hclust(graph, h_cuts=[0.5], return_linkage=True)

    assert len(result["linkage_trees"]) == 0
    Z = stitch_linkage(result)
    assert Z.shape == (2, 4)
    assert np.all(np.isinf(Z[:, 2]))


def test_stitch_linkage_single_component():
    """When all points are in one component, stitched Z has no inf merges."""
    rng = np.random.default_rng(0)
    coords = rng.uniform(0, 100, size=(20, 2))
    graph = spatial_dist_graph(coords, h_max=200)
    assert graph["n_components"] == 1

    result = sparse_hclust(graph, h_cuts=[50], return_linkage=True)
    Z = stitch_linkage(result)
    assert Z.shape == (19, 4)
    assert np.all(np.isfinite(Z[:, 2]))


def test_dense_hclust_return_linkage():
    rng = np.random.default_rng(0)
    coords = rng.uniform(0, 10_000, size=(50, 2))
    result = dense_hclust(coords, h_cuts=[3_000], return_linkage=True)
    assert "linkage_trees" in result
    assert len(result["linkage_trees"]) == 1
    Z, idx = result["linkage_trees"][0]
    assert Z.shape == (49, 4)
    assert len(idx) == 50


# ---------------------------------------------------------------------------
# dense_hclust
# ---------------------------------------------------------------------------

def test_dense_hclust_labels_shape():
    rng = np.random.default_rng(0)
    coords = rng.uniform(0, 10_000, size=(50, 2))
    result = dense_hclust(coords, h_cuts=[3_000, 8_000])
    for h in [3_000.0, 8_000.0]:
        assert result["labels"][h].shape == (50,)


# ---------------------------------------------------------------------------
# SparseAgglomerativeClustering (sklearn API)
# ---------------------------------------------------------------------------

def test_sklearn_api_fit(small_clustered_coords):
    model = SparseAgglomerativeClustering(h_max=5_000, distance_threshold=3_000)
    model.fit(small_clustered_coords)
    assert hasattr(model, "labels_")
    assert model.labels_.shape == (200,)
    assert model.n_clusters_ >= 1
    assert model.n_leaves_ == 200
    assert model.n_features_in_ == 2


def test_sklearn_api_fit_predict(small_clustered_coords):
    model = SparseAgglomerativeClustering(h_max=5_000, distance_threshold=3_000)
    labels = model.fit_predict(small_clustered_coords)
    assert labels.shape == (200,)


def test_sklearn_api_exposes_linkage(small_clustered_coords):
    model = SparseAgglomerativeClustering(h_max=5_000, distance_threshold=3_000)
    model.fit(small_clustered_coords)
    n = len(small_clustered_coords)

    assert hasattr(model, "linkage_matrix_")
    assert model.linkage_matrix_.shape == (n - 1, 4)

    assert hasattr(model, "children_")
    assert model.children_.shape == (n - 1, 2)

    assert hasattr(model, "distances_")
    assert model.distances_.shape == (n - 1,)


# ---------------------------------------------------------------------------
# Linkage method correctness: sparse_hclust vs scipy for each HAC method
#
# Strategy: use a small dataset where all points fall within h_max, so the
# sparse graph is fully connected and sparse_hclust must produce results
# identical to scipy.cluster.hierarchy.linkage + fcluster.
# Partition equivalence is checked with adjusted_rand_score (ARI = 1.0),
# which is invariant to label permutations — matching the guarantee that
# "groups are identical, labels may differ".
# ---------------------------------------------------------------------------

# Small well-separated clusters so every cut height produces a non-trivial
# partition (not all singletons, not one big cluster).
_METHODS_COORDS = np.array([
    [0, 0], [1, 0], [0, 1], [1, 1],      # cluster A
    [10, 0], [11, 0], [10, 1], [11, 1],  # cluster B
    [5, 8], [6, 8], [5, 9], [6, 9],      # cluster C
], dtype=float)

# h_max large enough to connect all points; h_cut chosen to recover 3 clusters
_METHODS_H_MAX = 30.0
_METHODS_H_CUT = 5.0

HAC_METHODS = ["single", "complete", "average", "weighted", "ward"]


@pytest.mark.parametrize("method", HAC_METHODS)
def test_method_partition_matches_scipy(method, backend):
    """sparse_hclust with each linkage method produces the same partition as scipy."""
    coords = _METHODS_COORDS
    h_max = _METHODS_H_MAX
    h_cut = _METHODS_H_CUT

    graph = spatial_dist_graph(coords, h_max=h_max, metric="euclidean")
    assert graph["n_components"] == 1, "all points must be in one component"

    result = sparse_hclust(graph, h_cuts=[h_cut], method=method)
    gshac_labels = result["labels"][float(h_cut)]

    dists = pdist(coords)
    Z = linkage(dists, method=method)
    scipy_labels = fcluster(Z, t=h_cut, criterion="distance")

    ari = adjusted_rand_score(scipy_labels, gshac_labels)
    assert ari == 1.0, (
        f"method={method!r}: partition differs from scipy (ARI={ari:.4f})\n"
        f"  gshac : {gshac_labels}\n"
        f"  scipy : {scipy_labels}"
    )


@pytest.mark.parametrize("method", HAC_METHODS)
def test_method_cluster_count_matches_scipy(method, backend):
    """Cluster count from sparse_hclust matches scipy at multiple cut heights."""
    rng = np.random.default_rng(42)
    coords = rng.uniform(0, 20, size=(40, 2))
    h_max = 100.0  # fully connected
    h_cuts = [2.0, 5.0, 10.0]

    graph = spatial_dist_graph(coords, h_max=h_max, metric="euclidean")
    result = sparse_hclust(graph, h_cuts=h_cuts, method=method)

    dists = pdist(coords)
    Z = linkage(dists, method=method)

    for h in h_cuts:
        gshac_k = len(np.unique(result["labels"][float(h)]))
        scipy_k = len(np.unique(fcluster(Z, t=h, criterion="distance")))
        assert gshac_k == scipy_k, (
            f"method={method!r}, h={h}: gshac={gshac_k} clusters, scipy={scipy_k}"
        )


@pytest.mark.parametrize("method", HAC_METHODS)
def test_method_exact_partition_random(method, backend):
    """Verify identical partition (not just count) for a random dataset."""
    rng = np.random.default_rng(7)
    coords = rng.uniform(0, 50, size=(30, 2))
    h_max = 200.0  # fully connected
    h_cuts = [5.0, 15.0, 30.0]

    graph = spatial_dist_graph(coords, h_max=h_max, metric="euclidean")
    result = sparse_hclust(graph, h_cuts=h_cuts, method=method)

    dists = pdist(coords)
    Z = linkage(dists, method=method)

    for h in h_cuts:
        gshac_labels = result["labels"][float(h)]
        scipy_labels = fcluster(Z, t=h, criterion="distance")
        ari = adjusted_rand_score(scipy_labels, gshac_labels)
        assert ari == 1.0, (
            f"method={method!r}, h={h}: partition differs from scipy (ARI={ari:.4f})"
        )
