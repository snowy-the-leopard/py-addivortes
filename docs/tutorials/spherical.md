# Modelling spherical data

This tutorial shows how to use AddiVortes with **spherical data**, such as
measurements at geographic locations on the surface of a globe. Spherical data
requires a specialised distance measure because Euclidean distance does not
respect curvature or wrap-around at the poles and date line.

## 1. What is spherical data?

Spherical data are observations whose locations lie on the surface of a sphere.
A familiar example is geographic data specified by **latitude** and
**longitude**. Two properties distinguish this from planar data:

1. **Wrap-around**: Longitude values near −180° and +180° are close on the
   globe but far apart in Euclidean space. A good metric must recognise this.
2. **Curvature**: Distances should follow great-circle arcs rather than
   straight lines through the interior of the sphere.

AddiVortes uses **great-circle distance** when `metric="spherical"` is
specified (also accepted: `"S"`, `"sphere"`, or `1`). Points near the poles and
date line are then treated as neighbours when they are genuinely close on the
sphere.

## 2. Coordinate convention

Spherical coordinates must be in **radians**:

- **Latitude-type dimensions** (all spherical columns except the last in each
  group): values in [−π/2, π/2].
- **Longitude / azimuthal dimension** (the *last* column in each spherical
  group): values in [−π, π].

For a single globe you need two columns, latitude first and longitude last:

```python
import numpy as np

X = np.column_stack([latitude_rad, longitude_rad])
```

Pass `metric="spherical"` so great-circle distance is used when assigning
observations to Voronoi cells.

For mixed spherical and Euclidean covariates, pass one metric per column, for
example `metric=["spherical", "spherical", "euclidean"]`.

## 3. Generating synthetic spherical data

We simulate 300 locations on a globe. The response is a synthetic temperature
that is warmest at the equator with a slight east-west gradient.

```python
from addivortes import AddiVortesRegressor

rng = np.random.default_rng(42)
n = 300

lat = rng.uniform(-np.pi / 2, np.pi / 2, size=n)
lon = rng.uniform(-np.pi, np.pi, size=n)

y_true = 20 * np.cos(lat) + 5 * np.sin(lon)
y = y_true + rng.normal(scale=2, size=n)

X = np.column_stack([lat, lon])
```

Plot the spatial distribution of responses (degrees are used only for display):

```python
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

cmap = LinearSegmentedColormap.from_list("temp", ["blue", "white", "red"])
norm = plt.Normalize(y.min(), y.max())

fig, ax = plt.subplots(figsize=(7, 4))
sc = ax.scatter(
    lon * 180 / np.pi,
    lat * 180 / np.pi,
    c=y,
    cmap=cmap,
    norm=norm,
    s=18,
)
ax.set(
    xlabel="Longitude (degrees)",
    ylabel="Latitude (degrees)",
    title="Simulated response on the globe",
)
fig.colorbar(sc, ax=ax, label="Response")
plt.tight_layout()
plt.show()
```

## 4. Fitting the spherical model

Fit with `metric="spherical"` so both columns use great-circle distance.

```python
fit_sph = AddiVortesRegressor(
    n_tessellations=50,
    total_mcmc_iter=500,
    burn_in=100,
    metric="spherical",
    random_state=42,
    verbose=False,
)
fit_sph.fit(X, y)

print("In-sample RMSE (spherical metric):", round(fit_sph.in_sample_rmse_, 3))
```

For comparison, fit the same model with the default Euclidean metric. This
treats latitude and longitude as ordinary continuous variables and does not
handle wrap-around near ±π correctly.

```python
fit_euc = AddiVortesRegressor(
    n_tessellations=50,
    total_mcmc_iter=500,
    burn_in=100,
    metric="euclidean",
    random_state=42,
    verbose=False,
)
fit_euc.fit(X, y)

print("In-sample RMSE (Euclidean metric):", round(fit_euc.in_sample_rmse_, 3))
```

## 5. Out-of-sample evaluation

Generate a held-out test set and compare test RMSE for both models.

```python
test_rng = np.random.default_rng(101)
n_test = 200

lat_test = test_rng.uniform(-np.pi / 2, np.pi / 2, size=n_test)
lon_test = test_rng.uniform(-np.pi, np.pi, size=n_test)

y_test_true = 20 * np.cos(lat_test) + 5 * np.sin(lon_test)
y_test = y_test_true + test_rng.normal(scale=2, size=n_test)

X_test = np.column_stack([lat_test, lon_test])

preds_sph = fit_sph.predict(X_test)
preds_euc = fit_euc.predict(X_test)

rmse_sph = float(np.sqrt(np.mean((y_test - preds_sph) ** 2)))
rmse_euc = float(np.sqrt(np.mean((y_test - preds_euc) ** 2)))

print("Test RMSE (spherical metric):", round(rmse_sph, 3))
print("Test RMSE (Euclidean metric):", round(rmse_euc, 3))
```

## 6. Visualising predictions

```python
y_range = (min(y_test.min(), preds_sph.min(), preds_euc.min()),
           max(y_test.max(), preds_sph.max(), preds_euc.max()))

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(y_test, preds_sph, color="darkblue", s=25, label="Spherical metric")
ax.scatter(y_test, preds_euc, marker="x", color="darkred", s=25, label="Euclidean metric")
ax.plot(y_range, y_range, color="grey", linestyle="--", linewidth=2, label="y = x")
ax.set(
    xlim=y_range,
    ylim=y_range,
    xlabel="Observed values",
    ylabel="Predicted values",
    title="Spherical vs Euclidean metric: predicted vs observed",
)
ax.legend(loc="upper left", frameon=False)
plt.tight_layout()
plt.show()
```

## 7. Multiple spherical covariates

When covariates lie on **separate** spherical surfaces, pass `members` to
indicate which columns belong to which surface. If `members` is omitted, all
spherical columns are grouped into a single surface, which is incorrect when
more than one azimuthal coordinate is present.

Augment the data with a second spherical coordinate system:

```python
lat_prime1 = rng.uniform(-np.pi / 2, np.pi / 2, size=n)
lat_prime2 = rng.uniform(-np.pi / 2, np.pi / 2, size=n)
lon_prime = rng.uniform(-np.pi, np.pi, size=n)

y_prime_true = (
    20 * np.cos(lat)
    + 5 * np.sin(lon)
    - 5 * np.cos(lat_prime1) * np.sin(lat_prime2)
    + 10 * np.sin(lon_prime) ** 2 * np.sin(np.abs(lat_prime1 - lat_prime2))
)
y_prime = y_prime_true + rng.normal(scale=3, size=n)

X_prime = np.column_stack([lat, lon, lat_prime1, lat_prime2, lon_prime])
```

The first two columns belong to surface 1; the remaining three belong to
surface 2:

```python
fit_sph_multi = AddiVortesRegressor(
    n_tessellations=50,
    total_mcmc_iter=500,
    burn_in=100,
    metric="spherical",
    members=[1, 1, 2, 2, 2],
    random_state=42,
    verbose=False,
)
fit_sph_multi.fit(X_prime, y_prime)

fit_euc_multi = AddiVortesRegressor(
    n_tessellations=50,
    total_mcmc_iter=500,
    burn_in=100,
    metric="euclidean",
    random_state=42,
    verbose=False,
)
fit_euc_multi.fit(X_prime, y_prime)
```

Evaluate on a held-out test set:

```python
lat_prime_test1 = test_rng.uniform(-np.pi / 2, np.pi / 2, size=n_test)
lat_prime_test2 = test_rng.uniform(-np.pi / 2, np.pi / 2, size=n_test)
lon_prime_test = test_rng.uniform(-np.pi, np.pi, size=n_test)

y_prime_test_true = (
    20 * np.cos(lat_test)
    + 5 * np.sin(lon_test)
    - 5 * np.cos(lat_prime_test1) * np.sin(lat_prime_test2)
    + 10 * np.sin(lon_prime_test) ** 2 * np.sin(np.abs(lat_prime_test1 - lat_prime_test2))
)
y_prime_test = y_prime_test_true + test_rng.normal(scale=3, size=n_test)

X_prime_test = np.column_stack(
    [lat_test, lon_test, lat_prime_test1, lat_prime_test2, lon_prime_test]
)

preds_prime_sph = fit_sph_multi.predict(X_prime_test)
preds_prime_euc = fit_euc_multi.predict(X_prime_test)

rmse_prime_sph = float(np.sqrt(np.mean((y_prime_test - preds_prime_sph) ** 2)))
rmse_prime_euc = float(np.sqrt(np.mean((y_prime_test - preds_prime_euc) ** 2)))

print("Test RMSE (spherical metric):", round(rmse_prime_sph, 3))
print("Test RMSE (Euclidean metric):", round(rmse_prime_euc, 3))
```

## 8. Summary

- Set `metric="spherical"` to use great-circle distance for all columns, or
  pass a per-column list such as `["spherical", "spherical", "euclidean"]`.
- For multiple spherical coordinate systems, pass `members` with one integer per
  column, for example `members=[1, 1, 2, 2, 2]` for two surfaces with two and
  three columns respectively.
- Spherical coordinates must be in **radians**: latitude in [−π/2, π/2] and
  longitude in [−π, π].
- The **last** column in each membership group is the azimuthal (longitude)
  dimension; earlier columns in the group are polar (latitude) dimensions.
- Great-circle distance handles wrap-around, so points near longitude ±π are
  recognised as neighbours.
