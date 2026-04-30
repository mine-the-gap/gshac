# Distance metrics and CRS-aware dispatch

This document records the design of gshac's distance API: the input forms
accepted by `spatial_dist_graph` and `geographic_connectivity`, the
metric/CRS validation matrix, the geodesic implementation, and the
constraints that motivate each decision. It is intended for future
contributors extending gshac to non-point geometries or alternative
ellipsoids, and for users who want to understand the conditions under
which the package treats coordinates correctly. The companion test
module is `tests/test_crs_dispatch.py`.

## Background: distance is not a property of coordinates alone

A two-dimensional coordinate pair `(x, y)` is meaningless without a
coordinate reference system (CRS). The same numbers `(0.5, 0.0)` are
8.7 micrometres from the origin in EPSG:3857 web-mercator, 55.6 km in
EPSG:4326 lon/lat at the equator, and undefined in a geocentric CRS.
Any function that treats coordinates as if they were embedded in
Euclidean R^2, with no regard for the CRS that produced them, is
quietly assuming "planar" — and on geographic data that assumption is
silently wrong by 0.5% to 100% depending on latitude.

The pre-CRS-aware version of gshac accepted a raw `(n, 2)` ndarray plus
a `metric` string (`"euclidean"` or `"haversine"`) and trusted the user
to match them up. That works for an experienced user with a single
dataset, but it is a footgun in any workflow that mixes lon/lat from
one source with metres from another.

## Design pattern: CRS-conditional dispatch

The API mirrors the dispatch pattern of R's `sf::st_distance`. In sf,
the distance computation is conditional on the CRS family: geographic
data uses an Earth-model engine (s2 by default, or lwgeom if
`sf_use_s2(FALSE)`), and projected data uses GEOS planar geometry. The
user passes a high-level `which=` argument, but the set of valid `which`
values depends on the CRS — you cannot ask for the Euclidean distance
between two `sf` objects in EPSG:4326, and you cannot ask for a
great-circle distance between two `sf` objects in a projected CRS. sf
either selects the right engine automatically or, when the user asks
for an inappropriate combination, raises an error.

gshac adopts the same separation. The relevant analogy:

| Earth model | sf (R)                       | gshac (Python)                       |
|-------------|------------------------------|--------------------------------------|
| Sphere      | s2 (default for geographic)  | haversine (opt-in)                   |
| Ellipsoid   | lwgeom / `sf_use_s2(FALSE)`  | `pyproj.Geod.inv` (default for geographic) |
| Cartesian   | GEOS (default for projected) | numpy euclidean (default for projected) |

There is one deliberate departure from sf's defaults: gshac defaults to
the **ellipsoid** for geographic CRSs, not the sphere.

## Departure from sf: ellipsoid is the default

sf defaults to s2 (sphere) for performance on global-scale workloads;
the s2 implementation is a vectorised C++ library, and the haversine
trig itself is a few floating-point operations per pair. For
continental-scale tessellations of millions of features, the per-pair
cost of switching from sphere to ellipsoid is non-negligible.

gshac is in a different regime. The library targets desktop-scale
clustering of point sets in the 1k to 1M range, where the dominant
runtime is the BallTree neighbour query (which on a sphere is
fundamentally O(n log n) for h_max-bounded radii) and the memory cost
of storing the sparse graph. Per-pair distance computation, even with
`pyproj.Geod.inv`, is cheap by comparison: on the order of microseconds
per pair, well below the BallTree's per-pair cost in practice. The
accuracy improvement from haversine to geodesic is sub-millimetre on
short pairs (< 100 km) and ~30 m at 10 km worst case — small in
absolute terms, but free in this regime.

There is a second consideration. The clustering output of gshac
typically feeds into downstream maps, area summaries, or further
analytical pipelines where ellipsoidal correctness is a property the
caller will assume rather than verify. Spherical "great-circle"
distance is a mean-radius approximation; the absolute error is small
but it compounds through subsequent area, length, and adjacency
computations. Setting the default to ellipsoidal removes a silent
failure mode without imposing measurable cost in the regime where
gshac operates.

Haversine remains explicitly available as `metric="haversine"` for two
reasons: (i) it is what the paper's benchmarks are timed against, so we
must continue to produce byte-identical numbers when called that way;
and (ii) the C extension `_gshac.haversine_edges` is available as a
fast path for users who do not need ellipsoidal accuracy.

## Validation matrix

The `(metric, CRS-kind)` combination is validated up front:

|              | geographic CRS               | projected CRS              | missing CRS                    |
|--------------|-------------------------------|----------------------------|---------------------------------|
| `"auto"`     | -> `"geodesic"`              | -> `"euclidean"`           | -> `"euclidean"` (UserWarning) |
| `"geodesic"` | OK                            | **error**                  | OK (caller asserted lon/lat)   |
| `"haversine"`| OK                            | **error**                  | OK (caller asserted lon/lat)   |
| `"euclidean"`| **error**                     | OK                         | OK                             |
| anything else| **error** (unknown metric)    | **error**                  | **error**                       |

The reasoning for each cell:

* `"euclidean"` on a geographic CRS is rejected because degree
  differences are not metres. Even ignoring the longitudinal
  contraction with latitude (which would only affect E-W distances by
  `cos(phi)`), the unit mismatch makes any `h_max` value meaningless.
  sf takes the same position; the combination is simply not offered.

* `"haversine"` and `"geodesic"` on a projected CRS are rejected
  because the trig kernels assume `(lon, lat)` in degrees. Feeding
  them metres-coordinates produces values that are not just wrong but
  wildly so (typically `nan` from `arcsin` of an out-of-range argument).
  Failing fast is the only useful behaviour.

* `"auto"` + missing CRS picks `"euclidean"` because that is the
  pre-CRS-aware default and covers the existing benchmark and example
  code that passes raw arrays. The one-time `UserWarning` nudges the
  caller to either pass `crs=` or migrate to a `GeoDataFrame`. We deemed
  a hard error too aggressive: it would break the existing benchmark
  scripts and any user code that has been working correctly with
  metres-on-metres ndarray inputs.

* Unknown metric values raise a `ValueError` listing the valid options.
  The previous version accepted `"cosine"` through to a fall-through
  branch that raised; the new validator rejects it earlier and with a
  better message.

In addition to the metric/CRS combination, a soft warning is emitted
when the CRS is projected but its primary axis unit is not metre
(e.g. EPSG:2229 California State Plane in feet). `h_max` is documented
as metres; in a non-metre CRS it is silently in input units, and we
prefer a warning to a wrong answer.

## Two-stage prefilter for geodesic

The implementation follows the *filter-and-refine* pattern long
established in spatial databases: when the precise distance is
expensive (here, ellipsoidal geodesic via pyproj's iterative solver)
but a cheaper bounded metric admits an indexable search, one
over-fetches with the cheap metric and re-verifies each candidate
with the expensive one. PostGIS's `ST_DWithin` on the `geography`
type uses a spheroid-padded bounding box prefilter followed by exact
geodesic distance; Lucene's `geo_distance` query in Elasticsearch
uses a haversine prefilter followed by exact verification; R-tree
implementations in GEOS, shapely, and Oracle Spatial generalise the
pattern to bounding boxes followed by exact geometry distance. In
each case the index radius (or bounding box) is padded by the
worst-case ratio between the cheap and exact metrics over the input
domain, so that no true positive can be pruned.

A naive geodesic implementation would instead build a spatial index
directly on the ellipsoid (e.g. converting to Earth-Centred
Earth-Fixed (ECEF) Cartesian coordinates and using a KD-tree). This
is correct, but it requires either a custom bounding-volume structure
or an exotic chord-vs-geodesic correction inside the KD-tree's
distance predicate.

gshac instead reuses the existing haversine BallTree as a prefilter and
finishes with an exact ellipsoidal distance computation:

1. Convert `(lon, lat)` to `(lat_rad, lon_rad)` and build a
   `BallTree(metric="haversine")`.
2. Query the tree with `r = h_max * f / R`, where `R = 6_371_000` is the
   sphere radius and `f >= 1` is a safety factor. This collects every
   pair `(i, j)` whose haversine distance is at most `h_max * f`.
3. For each surviving pair, compute the exact WGS-84 ellipsoidal
   distance via `pyproj.Geod(ellps="WGS84").inv`. `Geod.inv` is
   vectorised over input arrays.
4. Drop pairs whose true ellipsoidal distance exceeds `h_max`.

The safety factor `f` must be at least the worst-case ratio of
ellipsoidal to spherical distance over the input domain. Otherwise a
pair whose true geodesic distance is below `h_max` but whose haversine
distance is above `h_max * f` would be excluded by the prefilter and
silently dropped. An empirical scan over `(latitude, dlat, dlon)`
shows the worst-case ratio reaches `~1.00449` near latitude 89, for
short east-west chords (the spot where the local ellipsoidal radius of
curvature in the prime-vertical direction differs most from the mean
spherical radius). The constant `_HAVERSINE_GEODESIC_SAFETY = 1.005`
covers this with a small margin for numerical wobble in BallTree's
haversine kernel; the corresponding test `test_haversine_prefilter_safety`
pins the worst-case behaviour at exactly that latitude.

The over-fetch from the inflated radius is negligible. At 0.5%, the
extra disc area scanned is `(1.005)^2 - 1 ~= 1%` of the original
radius's area, so we examine ~1% more candidate pairs than strictly
needed, then `Geod.inv` rejects them. This is much cheaper than
building a separate ellipsoidal index.

A future maintainer who wants to support a non-WGS84 ellipsoid (e.g.
GRS80 or a planetary body) should re-run the empirical scan to
reaffirm the safety factor; nothing in the code path is hard-coded to
WGS-84 except the `Geod(ellps="WGS84")` line and the constant 1.005.

## Input forms

`spatial_dist_graph` accepts three input forms:

* `(n, 2)` `numpy.ndarray` — the legacy form. Treated as raw
  coordinates; if `crs=` is `None`, the CRS is "missing" and dispatch
  falls back to the warn-once euclidean path. The paper's benchmarks
  use this form.
* `geopandas.GeoSeries` of Point geometries. The CRS is read from
  `gs.crs`; the user does not need to pass `crs=`.
* `geopandas.GeoDataFrame`. Same as GeoSeries, using the active
  geometry column.

For geopandas inputs, an explicit `crs=` argument is allowed only if it
matches the input's own CRS or if the input's CRS is `None`. A
mismatch raises a `ValueError`: silently overriding metadata that the
GeoDataFrame already carries would invite reprojection bugs.

`geopandas` is imported lazily so that gshac remains importable
without the `[geo]` extra. `pyproj` is similarly lazy: it is only
required when `metric` resolves to `"geodesic"` (which includes
`metric="auto"` on a geographic CRS) or when `crs=` is set explicitly.

## Limitations

* **Point geometries only.** Lines and polygons raise
  `NotImplementedError` with a message pointing the caller at
  `representative_point()` or `centroid` for an explicit Point
  approximation. Distance between non-point geometries is not a
  uniquely defined scalar (Hausdorff, Frechet, area-weighted, etc.) and
  HAC on lines or polygons therefore requires choosing a kernel that
  v0.x does not commit to. sf supports several via lwgeom and GEOS.
* **Projected CRSs must use metre axes** for `h_max` to be in metres.
  Other linear units (foot, link, etc.) trigger a soft `UserWarning`
  and are interpreted in input units. We do not auto-convert because
  an unintended unit conversion is at least as confusing as a wrong
  number.
* **WGS-84 only for the ellipsoid.** `_geodesic_pairs` constructs a
  `Geod(ellps="WGS84")`; alternative ellipsoids are not exposed. Most
  users on Earth do not need anything else, and an Earth-vs-Mars toggle
  belongs in a different layer.
* **Sphere radius is fixed** at `EARTH_RADIUS_M = 6_371_000`. The C
  extension's `_gshac.haversine_edges` bakes in the same constant; any
  future change to allow a configurable radius needs to update both
  the Python kernel and the C kernel together.
* **Missing CRS does not emit per-call warnings.** The first call with
  `metric="auto"` and no CRS warns; subsequent calls in the same
  process do not. A test that wants to re-trigger the warning resets
  the module-level `_MISSING_CRS_WARNED` flag via monkeypatch. We
  considered using `warnings.simplefilter("once")` but the
  module-level flag composes more cleanly with pytest's `recwarn`
  fixture and lets us reset it between tests.

## Future work

* **Hausdorff and Frechet distance for non-point geometries.** Adding
  these requires choosing a kernel and a candidate-pair structure
  appropriate for envelope-overlap rather than radius queries. sf's
  `st_distance(by_element = TRUE)` is the obvious template; the
  geodesic version would build on `Geod.inv` for closest-point
  computation.
* **Alternative ellipsoids and global toggle.** Exposing
  `Geod(ellps=...)` as a parameter is straightforward. The
  haversine/geodesic distinction could also be promoted to a global
  toggle for users who want bulk performance over accuracy; the
  resulting API would mirror `sf::sf_use_s2()` more closely.
* **Sphere-vs-ellipsoid policy injection.** A subclass of
  `SparseAgglomerativeClustering` that pins `metric="haversine"` for
  benchmark reproducibility could let downstream packages opt into
  bit-exact reproduction of the paper's numbers without leaking the
  default into casual usage.

## References

* sf documentation: <https://r-spatial.github.io/sf/articles/sf6.html>
  (sf and units; sf and projected geometries).
* `sf::st_distance` source: <https://github.com/r-spatial/sf/blob/main/R/geom-measures.R>.
* pyproj `Geod`: <https://pyproj4.github.io/pyproj/stable/api/geod.html>.
* WGS-84 ellipsoid parameters: NIMA TR8350.2.
