"""
sparse_hclust.py

Hierarchical clustering using a sparse geographic distance graph.

For each connected component of the graph, runs HAC and cuts at the requested
height thresholds. Because inter-component distances are all > h_max, features
in different components never merge at any cut height h <= h_max.

Single linkage uses the component MST (no dense sub-matrix). Complete / average /
Ward use a per-component dense sub-matrix: when ``coords`` are passed to
``sparse_hclust`` the sub-matrix is recomputed exactly from coordinates; without
``coords`` it is reconstructed from the threshold graph with a sentinel for
absent (> h_max) pairs (exact for complete, approximate for average — pass
``coords`` for exact average).

Exactness guarantee for a cut at any ``h <= h_max``: GUARANTEED for single,
complete, and average. These linkages have inter-cluster dissimilarity
``D(A, B) >= min_{a in A, b in B} dist(a, b)``, so clusters in different
components (every cross pair > h_max) never merge below h_max, and the
per-component result equals the dense result. **Ward is NOT guaranteed**: its
variance/centroid-based merge height is not bounded below by the minimum
inter-point distance and can fall below h_max for clusters in different
components (e.g. one component spatially surrounding another), so the
per-component result may differ from a full dense Ward. Ward is exact in practice
for spatially compact clusters.

Also provides a dense baseline that computes the full O(n^2) distance matrix.

Dependencies: numpy, scipy
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import squareform, cdist

try:
    import fastcluster as _fc
    _linkage = _fc.linkage
except ImportError:
    from scipy.cluster.hierarchy import linkage as _linkage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree as _mst, connected_components as _cc
from typing import Optional, Sequence

try:
    from sklearn.base import BaseEstimator, ClusterMixin
    from sklearn.utils.validation import validate_data
    _SKLEARN = True
except ImportError:  # pragma: no cover
    _SKLEARN = False

try:
    from ._gshac import linkage_from_mst as _c_linkage_from_mst
    from ._gshac import fcluster_batch as _c_fcluster_batch
    _GSHAC_C = True
except ImportError:
    _GSHAC_C = False


_EARTH_RADIUS_M = 6_371_000.0  # matches spatial_dist_graph / dense_hclust haversine


def _full_distance_matrix(coords_sub: np.ndarray, metric: str) -> np.ndarray:
    """Full ``(m, m)`` pairwise distance matrix for one component's coordinates.

    Uses the same metric and constants as ``spatial_dist_graph`` /
    ``dense_hclust``. Required for *exact* complete / average / Ward linkage:
    the threshold graph omits intra-component pairs farther than ``h_max``
    apart, and those distances affect the merge heights of these linkages —
    critically for average linkage, which averages them into the merge height.
    """
    if metric == "euclidean":
        return cdist(coords_sub, coords_sub, metric="euclidean")
    if metric == "haversine":
        rad = np.radians(coords_sub[:, [1, 0]])          # (lat, lon) radians
        lat = rad[:, 0][:, None]
        lon = rad[:, 1][:, None]
        dlat = lat - lat.T
        dlon = lon - lon.T
        a = (np.sin(dlat / 2.0) ** 2
             + np.cos(lat) * np.cos(lat.T) * np.sin(dlon / 2.0) ** 2)
        return 2.0 * _EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    if metric == "geodesic":
        try:
            from pyproj import Geod
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                'metric="geodesic" requires pyproj for the exact non-single '
                "linkage sub-matrix. Install via `pip install gshac[geo]`."
            ) from e
        geod = Geod(ellps="WGS84")
        m = len(coords_sub)
        ii, jj = np.meshgrid(np.arange(m), np.arange(m), indexing="ij")
        lon = coords_sub[:, 0]
        lat = coords_sub[:, 1]
        _, _, d = geod.inv(lon[ii].ravel(), lat[ii].ravel(),
                           lon[jj].ravel(), lat[jj].ravel())
        D = np.asarray(d, dtype=np.float64).reshape(m, m)
        np.fill_diagonal(D, 0.0)
        return D
    raise ValueError(
        f"Unknown metric for exact sub-matrix: {metric!r}. "
        "Use 'euclidean', 'haversine', or 'geodesic'."
    )


def _build_Z_from_mst(sub_csr: csr_matrix, size: int) -> np.ndarray:
    """
    Build a scipy linkage matrix Z for single linkage using the MST of a
    sparse distance sub-matrix.  O(m log m) time, O(n + m) memory — avoids
    the O(n²) dense sub-matrix entirely.

    Uses optimised C union-find when the _gshac extension is available.

    Parameters
    ----------
    sub_csr : csr_matrix
        Sparse symmetric distance sub-matrix for one connected component.
    size : int
        Number of points in the component (``sub_csr.shape[0]``).

    Returns
    -------
    Z : ndarray of shape ``(size - 1, 4)``
        Scipy-format linkage matrix for the component.
    """
    mst     = _mst(sub_csr)           # upper-triangular CSR of MST edges
    rows, cols = mst.nonzero()
    weights    = np.asarray(mst[rows, cols]).ravel()
    order      = np.argsort(weights)
    rows       = rows[order]
    cols       = cols[order]
    weights    = weights[order]

    if _GSHAC_C:
        return _c_linkage_from_mst(
            rows.astype(np.int64),
            cols.astype(np.int64),
            weights.astype(np.float64),
            size,
        )

    # Python fallback: union-find with scipy cluster-label tracking
    parent    = np.arange(size, dtype=np.intp)
    sz        = np.ones(size,  dtype=np.intp)
    sci_label = np.arange(size, dtype=np.float64)
    next_lbl  = float(size)

    Z = np.empty((size - 1, 4), dtype=np.float64)

    for k in range(size - 1):
        i, j, w = int(rows[k]), int(cols[k]), float(weights[k])

        ri = i
        while parent[ri] != ri:
            parent[ri] = parent[parent[ri]]
            ri = parent[ri]
        rj = j
        while parent[rj] != rj:
            parent[rj] = parent[parent[rj]]
            rj = parent[rj]

        Z[k, 0] = sci_label[ri]
        Z[k, 1] = sci_label[rj]
        Z[k, 2] = w
        Z[k, 3] = float(sz[ri] + sz[rj])

        if sz[ri] < sz[rj]:
            ri, rj = rj, ri
        parent[rj]    = ri
        sz[ri]       += sz[rj]
        sci_label[ri] = next_lbl
        next_lbl     += 1.0

    return Z


def sparse_hclust(
    graph: dict,
    h_cuts: Sequence[float],
    method: str = "single",
    ids: Optional[Sequence] = None,
    return_linkage: bool = False,
    coords: Optional[np.ndarray] = None,
) -> dict:
    """
    Hierarchical clustering via sparse geographic distance graph.

    Parameters
    ----------
    graph : dict
        Output of spatial_dist_graph(). Must have keys 'matrix' and 'components'.
    h_cuts : sequence of float
        Cut heights in the same units as the distances in graph['matrix'] (metres).
    method : str
        Linkage method: 'single', 'complete', 'average', 'ward', etc.
        Passed directly to scipy.cluster.hierarchy.linkage.
    ids : sequence, optional
        Feature identifiers of length n. If None, integer indices 0..n-1 are used.
    return_linkage : bool, default False
        If True, include per-component linkage matrices in the result under the
        key ``'linkage_trees'``.  Each entry is a ``(Z, indices)`` tuple where
        *Z* is a scipy-format ``(n_k - 1, 4)`` linkage matrix for one connected
        component and *indices* is an ndarray of the global point indices that
        belong to that component.  Singletons are omitted (they have no merges).

        Use :func:`stitch_linkage` to combine these into a single ``(n - 1, 4)``
        matrix compatible with :func:`scipy.cluster.hierarchy.dendrogram`.
    coords : ndarray of shape (n, 2), optional
        The coordinates the graph was built from, in the same order. Required
        for **exact average linkage** (and makes complete/Ward exact in
        general): the per-component sub-matrix is recomputed from these
        coordinates using the graph's metric, recovering the intra-component
        distances beyond ``h_max`` that the threshold graph omits. Ignored for
        single linkage. If omitted, non-single linkage falls back to a
        threshold-graph + sentinel reconstruction — exact for single, complete,
        and Ward, but approximate for average (a ``UserWarning`` is raised).

    Returns
    -------
    dict with keys:
        ids        : list of feature IDs (length n)
        components : ndarray of int, component labels
        labels     : dict mapping each h in h_cuts to an ndarray of int
                     cluster labels (length n, globally unique across components)
        linkage_trees : list of (Z, indices) tuples (only when return_linkage=True)

    Raises
    ------
    ValueError
        If ``graph`` is missing required keys, or ``ids`` length does not
        match the number of points in the graph.
    """
    mat        = graph["matrix"]
    components = graph["components"]
    n          = mat.shape[0]
    metric     = graph.get("metric", "euclidean")

    coords_arr = None
    if coords is not None:
        coords_arr = np.asarray(coords, dtype=np.float64)
        if coords_arr.shape != (n, 2):
            raise ValueError(
                f"coords must be shape (n, 2) matching the graph's {n} points; "
                f"got shape {coords_arr.shape!r}."
            )

    if ids is None:
        ids = list(range(n))
    else:
        ids = list(ids)

    h_cuts = sorted(float(h) for h in h_cuts)
    labels = {h: np.zeros(n, dtype=np.int64) for h in h_cuts}

    linkage_trees = [] if return_linkage else None

    # Pre-sort indices by component label (O(n log n)) to avoid
    # O(n_comp × n) scans — critical for large n with many components.
    global_id = 0

    sort_order   = np.argsort(components, kind="stable")
    sorted_comps = components[sort_order]
    _, g_starts  = np.unique(sorted_comps, return_index=True)
    g_ends       = np.append(g_starts[1:], n)
    g_sizes      = g_ends - g_starts

    # Batch-assign all singleton components in one vectorised step.
    singleton_mask = g_sizes == 1
    if singleton_mask.any():
        singleton_rows = sort_order[g_starts[singleton_mask]]
        n_singletons   = singleton_mask.sum()
        ids_assigned   = np.arange(global_id + 1,
                                   global_id + n_singletons + 1,
                                   dtype=np.int64)
        for h in h_cuts:
            labels[h][singleton_rows] = ids_assigned
        global_id += n_singletons

    for start, end, size in zip(g_starts, g_ends, g_sizes):
        if size == 1:
            continue
        idx = sort_order[start:end]

        if method == "single":
            # MST fast path: O(m_k log m_k) time, O(n_k + m_k) memory.
            # Avoids the O(c_k^2) dense sub-matrix entirely.
            if size == 2:
                # Trivial: one edge, one merge
                d = mat[idx[0], idx[1]]
                Z = np.array([[0.0, 1.0, d, 2.0]], dtype=np.float64)
            else:
                sub_csr = mat[np.ix_(idx, idx)].tocsr()
                Z = _build_Z_from_mst(sub_csr, size)
        else:
            # Dense path for complete / average / Ward linkage.
            if coords_arr is not None:
                # Exact: recompute the full per-component distance matrix from
                # coordinates with the graph's metric. The threshold graph omits
                # > h_max intra-component pairs, whose true distances drive
                # complete/average/Ward merge heights (critically for average).
                sub = _full_distance_matrix(coords_arr[idx], metric)
            else:
                # Fallback (no coords): reconstruct from the threshold graph and
                # fill absent (> h_max) pairs with a sentinel. Exact for complete
                # (max-dominated) and, in practice, Ward; approximate for average.
                sub = mat[np.ix_(idx, idx)].toarray()
                np.fill_diagonal(sub, 0.0)
                off_diag  = ~np.eye(size, dtype=bool)
                zero_mask = off_diag & (sub == 0.0)
                if zero_mask.any():
                    if method == "average":
                        warnings.warn(
                            "average linkage without coords= is approximate: "
                            "intra-component pairs beyond h_max are filled with a "
                            "sentinel that distorts averaged merge heights. Pass "
                            "coords= to sparse_hclust for exact average linkage.",
                            UserWarning, stacklevel=2,
                        )
                    max_edge = sub[off_diag & (sub > 0.0)].max()
                    sub[zero_mask] = max_edge * 2

            Z = _linkage(squareform(sub, checks=False), method=method)

        if return_linkage:
            linkage_trees.append((Z.copy(), idx.copy()))

        if _GSHAC_C:
            h_arr = np.array(h_cuts, dtype=np.float64)
            batch_labels = _c_fcluster_batch(Z, h_arr, size)
            max_local = 0
            for ci, h in enumerate(h_cuts):
                local_labels = batch_labels[ci]
                labels[h][idx] = global_id + local_labels
                mx = local_labels.max()
                if mx > max_local:
                    max_local = mx
            global_id += max_local
        else:
            max_local = 0
            for h in h_cuts:
                local_labels = fcluster(Z, t=h, criterion="distance")
                labels[h][idx] = global_id + local_labels
                max_local = max(max_local, local_labels.max())
            global_id += max_local

    out = dict(ids=ids, components=components, labels=labels)
    if return_linkage:
        out["linkage_trees"] = linkage_trees
    return out


def stitch_linkage(result: dict) -> np.ndarray:
    """
    Combine per-component linkage trees into a single ``(n - 1, 4)`` matrix.

    The returned matrix is fully compatible with
    :func:`scipy.cluster.hierarchy.dendrogram` and
    :func:`scipy.cluster.hierarchy.fcluster`.

    Inter-component merges are appended at distance ``inf`` — these appear as
    flat horizontal lines at the top of a dendrogram and are semantically
    correct: components separated by more than ``h_max`` never merge at any
    finite cut height.

    Parameters
    ----------
    result : dict
        Output of ``sparse_hclust(..., return_linkage=True)``.
        Must contain ``'linkage_trees'``, ``'components'``, and ``'ids'``.

    Returns
    -------
    Z : ndarray of shape ``(n - 1, 4)``
        Scipy-format linkage matrix covering all *n* points.

    Raises
    ------
    KeyError
        If ``result`` does not contain ``'linkage_trees'``, ``'components'``,
        or ``'ids'`` — i.e. ``sparse_hclust`` was called without
        ``return_linkage=True``.
    """
    linkage_trees = result["linkage_trees"]
    n = len(result["ids"])

    if n <= 1:
        return np.empty((0, 4), dtype=np.float64)

    next_node = n           # next available internal-node index
    roots = []              # (node_id, subtree_size) per component/singleton
    Z_rows = []

    # Track which points belong to non-singleton components.
    covered = set()
    for _, idx in linkage_trees:
        covered.update(idx.tolist())

    # Singletons are their own roots (leaf node, size 1).
    for i in range(n):
        if i not in covered:
            roots.append((i, 1))

    # Relabel each component's Z from local to global indices.
    for Z_local, idx in linkage_trees:
        size = len(idx)

        # Mapping: local leaf j → idx[j], local internal (size+k) → next_node+k
        local_to_global = np.empty(2 * size - 1, dtype=np.float64)
        for j in range(size):
            local_to_global[j] = float(idx[j])
        for k in range(size - 1):
            local_to_global[size + k] = float(next_node + k)

        for k in range(size - 1):
            left  = local_to_global[int(Z_local[k, 0])]
            right = local_to_global[int(Z_local[k, 1])]
            Z_rows.append([left, right, Z_local[k, 2], Z_local[k, 3]])

        # Root of this component is the last internal node created.
        roots.append((next_node + size - 2, size))
        next_node += size - 1

    # Merge all roots at inf to form a single hierarchy.
    if len(roots) >= 2:
        cur_node, cur_size = roots[0]
        for i in range(1, len(roots)):
            other_node, other_size = roots[i]
            new_size = cur_size + other_size
            Z_rows.append([float(cur_node), float(other_node),
                           np.inf, float(new_size)])
            cur_node = next_node
            cur_size = new_size
            next_node += 1

    Z = np.array(Z_rows, dtype=np.float64) if Z_rows else np.empty((0, 4), dtype=np.float64)
    return Z


# =============================================================================
# Dense baseline
# =============================================================================

def dense_hclust(
    coords: np.ndarray,
    h_cuts: Sequence[float],
    method: str = "single",
    ids: Optional[Sequence] = None,
    metric: str = "euclidean",
    return_linkage: bool = False,
) -> dict:
    """
    Dense hierarchical clustering baseline (O(n^2) time and memory).

    Parameters
    ----------
    coords : ndarray, shape (n, 2)
        Feature coordinates (metres for euclidean, degrees for haversine).
    h_cuts : sequence of float
        Cut heights (metres).
    method : str
        Linkage method.
    ids : sequence, optional
        Feature identifiers.
    metric : str
        'euclidean' or 'haversine'. For haversine, distances computed via
        the same formula as in spatial_dist_graph.
    return_linkage : bool, default False
        If True, include the linkage matrix under ``'linkage_trees'``.

    Returns
    -------
    dict with keys:
        ids        : list of feature IDs (length n)
        components : ndarray of int, shape (n,), all zeros (one component)
        labels     : dict mapping each h in h_cuts to an ndarray of int
                     cluster labels (length n)
        linkage_trees : list with a single ``(Z, indices)`` tuple
                        (only when ``return_linkage=True``)

    Raises
    ------
    ValueError
        If ``metric`` is not ``'euclidean'`` or ``'haversine'``.
    """
    n = len(coords)
    if ids is None:
        ids = list(range(n))

    h_cuts = sorted(float(h) for h in h_cuts)

    if metric == "euclidean":
        condensed = cdist(coords, coords, metric="euclidean")
        condensed = squareform(condensed, checks=False)

    elif metric == "haversine":
        EARTH_RADIUS_M = 6_371_000.0
        coords_rad = np.radians(coords[:, [1, 0]])  # lat, lon in radians
        # pairwise haversine
        full = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            dlat = coords_rad[i+1:, 0] - coords_rad[i, 0]
            dlon = coords_rad[i+1:, 1] - coords_rad[i, 1]
            a = (np.sin(dlat / 2) ** 2
                 + np.cos(coords_rad[i, 0]) * np.cos(coords_rad[i+1:, 0])
                 * np.sin(dlon / 2) ** 2)
            d = 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))
            full[i, i+1:] = d
            full[i+1:, i] = d
        condensed = squareform(full, checks=False)

    else:
        raise ValueError(f"Unknown metric: {metric!r}")

    Z = _linkage(condensed, method=method)
    labels = {}
    for h in h_cuts:
        labels[h] = fcluster(Z, t=h, criterion="distance").astype(np.int64)

    out = dict(ids=ids, components=np.zeros(n, dtype=np.int64), labels=labels)
    if return_linkage:
        out["linkage_trees"] = [(Z, np.arange(n))]
    return out


# =============================================================================
# scikit-learn compatible estimator
# =============================================================================

if _SKLEARN:
    class SparseAgglomerativeClustering(BaseEstimator, ClusterMixin):
        """
        Hierarchical clustering of spatial data via a sparse geographic distance
        graph.  Produces identical results to AgglomerativeClustering for any cut
        height h <= h_max, with O(n·k) memory instead of O(n²).

        Designed to be a drop-in replacement for::

            from sklearn.cluster import AgglomerativeClustering
            AgglomerativeClustering(
                distance_threshold=h_cut,
                n_clusters=None,
                metric="euclidean",
                linkage="single",
            )

        with the addition of ``h_max`` to enable the sparse optimisation.

        Parameters
        ----------
        h_max : float
            Maximum linkage distance in metres.  Pairs beyond this distance are
            never merged, so any cut height h <= h_max is exact.
        distance_threshold : float, optional
            Cut height for producing flat cluster labels (metres).
            Defaults to h_max.  Mirrors sklearn's ``distance_threshold``.
        linkage : str, default 'single'
            Linkage criterion: 'single', 'complete', 'average', or 'ward'.
            Mirrors sklearn's ``linkage`` parameter.
        metric : str, default 'euclidean'
            'euclidean' for projected coordinates (metres) or 'haversine' for
            geographic coordinates (lon, lat in degrees).

        Attributes
        ----------
        labels_ : ndarray of shape (n_samples,)
            Cluster label for each sample.
        n_clusters_ : int
            Number of clusters found.
        n_connected_components_ : int
            Number of connected components in the sparse graph.
        children_ : ndarray of shape (n_merges, 2)
            The children of each non-leaf node.  ``children_[i]`` holds the
            pair of node indices merged at step *i*.  Compatible with
            :func:`scipy.cluster.hierarchy.dendrogram` via ``linkage_matrix_``.
        distances_ : ndarray of shape (n_merges,)
            Distance at which each merge in ``children_`` occurred.
        linkage_matrix_ : ndarray of shape (n_samples - 1, 4)
            Full scipy-format linkage matrix (see :func:`stitch_linkage`).
            Pass directly to :func:`scipy.cluster.hierarchy.dendrogram`.
        n_leaves_ : int
            Number of samples.
        n_features_in_ : int
            Number of features seen during ``fit`` (always 2).

        Notes
        -----
        This estimator is intended as a future contribution to
        ``sklearn.cluster``.  The API mirrors ``AgglomerativeClustering``
        intentionally so that existing code only needs to swap the class name
        and add ``h_max``.
        """

        def __init__(
            self,
            h_max: float,
            distance_threshold: Optional[float] = None,
            linkage: str = "single",
            metric: str = "euclidean",
        ):
            self.h_max = h_max
            self.distance_threshold = distance_threshold
            self.linkage = linkage
            self.metric = metric

        def fit(self, X: np.ndarray, y: None = None) -> "SparseAgglomerativeClustering":
            """
            Fit the clustering model.

            Parameters
            ----------
            X : array-like of shape (n_samples, 2)
                Coordinates.  (x, y) in metres for metric='euclidean';
                (lon, lat) in degrees for metric='haversine'.
            y : ignored
                Not used; present for sklearn API compatibility.

            Returns
            -------
            self : SparseAgglomerativeClustering
                Fitted estimator.

            Raises
            ------
            ValueError
                If ``X`` does not have exactly 2 columns.
            """
            from gshac.spatial_dist_graph import spatial_dist_graph

            X = validate_data(self, X)
            h_cut = (
                self.distance_threshold
                if self.distance_threshold is not None
                else self.h_max
            )

            graph = spatial_dist_graph(X, self.h_max, metric=self.metric)
            result = sparse_hclust(graph, [h_cut], method=self.linkage,
                                   return_linkage=True)

            self.labels_ = result["labels"][h_cut]
            self.n_clusters_ = int(np.unique(self.labels_).size)
            self.n_connected_components_ = int(graph["n_components"])
            self.n_leaves_ = X.shape[0]
            self.n_features_in_ = X.shape[1]

            Z = stitch_linkage(result)
            self.linkage_matrix_ = Z
            self.children_ = Z[:, :2].astype(int)
            self.distances_ = Z[:, 2]

            return self

        def fit_predict(self, X: np.ndarray, y: None = None) -> np.ndarray:
            """
            Fit the model and return cluster labels.

            Parameters
            ----------
            X : array-like of shape (n_samples, 2)
                Coordinates.  (x, y) in metres for metric='euclidean';
                (lon, lat) in degrees for metric='haversine'.
            y : ignored
                Not used; present for sklearn API compatibility.

            Returns
            -------
            labels : ndarray of shape (n_samples,)
                Cluster label for each sample.
            """
            return self.fit(X, y).labels_
