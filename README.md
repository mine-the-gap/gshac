# GSHAC

[![Tests](https://github.com/mine-the-gap/gshac/actions/workflows/tests.yml/badge.svg)](https://github.com/mine-the-gap/gshac/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org)
[![License: GPL v3](https://img.shields.io/badge/license-GPL--3.0-green.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![codecov](https://codecov.io/gh/mine-the-gap/gshac/graph/badge.svg)](https://codecov.io/gh/mine-the-gap/gshac)

**Geographically Sparse Hierarchical Agglomerative Clustering**

Exact hierarchical clustering for spatial data, using a sparse geographic distance graph instead of the O(n²) dense distance matrix. For any cut height h <= h_max, GSHAC produces results **identical** to standard HAC (fastcluster/scipy) while requiring only O(n.k) time and memory, where k is the mean neighborhood size within h_max.

An default C extension provides optimized union-find linkage, haversine distances, and batch dendrogram cutting. The library has a fallback pure Python implementation, but the C is recommended for large datasets.

## Installation

Requires Python >= 3.10 and a C compiler (for the optional extension).

```bash
git clone https://github.com/mine-the-gap/gshac.git
cd gshac

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install package + build C extension
pip install -e .
python3 setup.py build_ext --inplace
```

The library falls back to pure Python if the C extension is not built.

### Extras

```bash
pip install -e ".[plot]"   # + matplotlib for dendrogram visualisation
pip install -e ".[dev]"    # + pytest + all optional deps
```

## Quick start

### scikit-learn API

```python
from gshac import SparseAgglomerativeClustering
import numpy as np

coords = np.random.default_rng(0).uniform(0, 500_000, size=(5_000, 2))

model = SparseAgglomerativeClustering(
    h_max=10_000,              # sparse graph radius (metres)
    distance_threshold=5_000,  # cut height (metres)
    linkage="single",
)
model.fit(coords)
print(model.labels_)       # cluster labels
print(model.n_clusters_)   # number of clusters
```

### Low-level API

```python
from gshac import spatial_dist_graph, sparse_hclust

graph = spatial_dist_graph(coords, h_max=10_000)
result = sparse_hclust(graph, h_cuts=[5_000, 10_000], method="single")
# result["labels"][5000] -> cluster labels at 5 km cut
```

### Geographic data and CRS

`spatial_dist_graph` is CRS-aware. With `metric="auto"` (the default), the
metric is chosen from the CRS: geographic -> geodesic (WGS-84 via
`pyproj.Geod`), projected -> euclidean, missing CRS -> euclidean (with a
one-time `UserWarning`). Combinations like euclidean on a geographic CRS,
or haversine/geodesic on a projected CRS, are rejected up front. Three
input forms are accepted:

```python
import numpy as np, geopandas as gpd
from shapely.geometry import Point
from gshac import spatial_dist_graph

coords = np.random.default_rng(0).uniform(0, 500_000, size=(5_000, 2))
spatial_dist_graph(coords, h_max=10_000)                         # ndarray, missing CRS
spatial_dist_graph(coords, h_max=10_000, crs="EPSG:3857")        # ndarray + explicit CRS
gdf = gpd.GeoDataFrame(geometry=[Point(*c) for c in coords[:, :2]], crs="EPSG:4326")
spatial_dist_graph(gdf, h_max=10_000)                            # GeoDataFrame -> geodesic
```

See the docstring of `spatial_dist_graph` for the full metric/CRS
validation rules.

### Dendrograms

```python
import matplotlib.pyplot as plt
from gshac import sparse_hclust, spatial_dist_graph, plot_dendrogram

graph = spatial_dist_graph(coords, h_max=10_000)
result = sparse_hclust(graph, h_cuts=[5_000], return_linkage=True)
plot_dendrogram(result, color_threshold=5_000)
plt.savefig("dendrogram.pdf", bbox_inches="tight")
```

Or from a fitted estimator:

```python
plot_dendrogram(model)
plt.savefig("dendrogram.pdf", bbox_inches="tight")
```

Per-component dendrograms (avoids the infinity stitching):

```python
from gshac import plot_component_dendrograms
fig, axes = plot_component_dendrograms(result, top_k=4)
fig.savefig("components.pdf", bbox_inches="tight")
```

### Linkage matrix access

The `return_linkage=True` flag returns per-component scipy linkage matrices.
Use `stitch_linkage()` to combine them into a single `(n-1, 4)` matrix
compatible with `scipy.cluster.hierarchy.fcluster`:

```python
from gshac import stitch_linkage
from scipy.cluster.hierarchy import fcluster

Z = stitch_linkage(result)
labels = fcluster(Z, t=5_000, criterion="distance")
```

Note: the stitched Z contains `inf` distances for inter-component merges,
so pass it through `plot_dendrogram()` (which handles this) rather than
calling `scipy.cluster.hierarchy.dendrogram(Z)` directly.

The sklearn estimator also exposes `children_`, `distances_`, and
`linkage_matrix_` after fitting.

## Running tests

```bash
source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -v
```

## API reference

| Function / Class | Description |
|---|---|
| `spatial_dist_graph(coords, h_max, metric, crs)` | Build sparse distance graph (CSR matrix); CRS-aware metric dispatch |
| `sparse_hclust(graph, h_cuts, method, return_linkage)` | Component-wise HAC on the sparse graph |
| `stitch_linkage(result)` | Combine per-component Z matrices into one |
| `SparseAgglomerativeClustering` | sklearn-compatible estimator |
| `geographic_connectivity(coords, h_max, metric, crs)` | Binary connectivity matrix for sklearn |
| `plot_dendrogram(model_or_result, ...)` | Plot dendrogram from estimator or result |
| `plot_component_dendrograms(result, top_k)` | Plot dendrograms for largest components |

## Acknowledgements

GSHAC builds on and is designed to interoperate with the scientific Python ecosystem:

- **[scipy](https://scipy.org/)** (BSD-3-Clause) — the linkage matrix format produced by GSHAC follows the `scipy.cluster.hierarchy` convention. The internal `fcluster_batch` C routine reimplements the logic of `scipy.cluster.hierarchy.fcluster(Z, t, criterion='distance')` to process multiple thresholds in a single pass; it is an independent implementation written for this project, not derived from scipy source code.
- **[fastcluster](https://github.com/dmuellner/fastcluster)** (BSD-2-Clause) — used as an optional faster backend for per-component linkage computation.
- **[scikit-learn](https://scikit-learn.org/)** (BSD-3-Clause) — the `SparseAgglomerativeClustering` estimator follows the sklearn `BaseEstimator` / `ClusterMixin` API conventions.

## License

GPL-3.0. See [LICENSE](LICENSE).

## Funding

Funded by the European Union. This work was supported by the European Research Council (ERC) project MINE-THE-GAP [https://minethegap.eu](https://minethegap.eu) (grant agreement no. 101170578 [10.3030/101170578](https://doi.org/10.3030/101170578). Views and opinions expressed are however those of the author only and do not necessarily reflect those of the European Union or the European Research Council Executive Agency. Neither the European Union nor the granting authority can be held responsible for them.