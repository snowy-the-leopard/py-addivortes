# Bayesian regression and prediction

This tutorial demonstrates Bayesian regression and out-of-sample prediction
with AddiVortes. We generate synthetic data with a known piecewise structure,
fit a model, and evaluate predictive performance on new observations.

## 1. Generating a synthetic dataset

The response depends on a simple rule involving the first two predictors, plus
Gaussian noise. This gives us a known data-generating process for evaluation.

```python
import numpy as np
from addivortes import AddiVortesRegressor

rng = np.random.default_rng(42)

X = rng.random((500, 5))
X[:, 0] = -10 - X[:, 0] * 10
X[:, 1] = X[:, 1] * 100
X[:, 2] = -9 + X[:, 2] * 10
X[:, 3] = 8 + X[:, 3]
X[:, 4] = X[:, 4] * 10

y_underlying = np.where(-X[:, 1] > 10 * X[:, 0] + 100, 10.0, 0.0)
y = y_underlying + rng.normal(size=y_underlying.size)
```

To explore the covariate structure, colour points by the underlying group:

```python
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import pandas as pd

frame = pd.DataFrame(X, columns=[f"x{i + 1}" for i in range(5)])
colors = np.where(y_underlying == 10, "red", "blue")
scatter_matrix(frame, c=colors, figsize=(7, 6), alpha=0.5, diagonal="hist")
plt.suptitle("Structure of predictor variables", y=1.02)
plt.show()
```

## 2. Fitting the model

For a quick demonstration we use a modest number of tessellations and MCMC
iterations.

```python
model = AddiVortesRegressor(
    n_tessellations=50,
    total_mcmc_iter=500,
    burn_in=100,
    random_state=42,
    verbose=False,
)
model.fit(X, y)

print("In-sample RMSE:", round(model.in_sample_rmse_, 3))
```

The in-sample RMSE indicates fit on the training data. Predictive performance
on new data is the stronger test.

## 3. Out-of-sample prediction

Generate a test set with the same construction and predict posterior means and
90% credible intervals.

```python
test_rng = np.random.default_rng(101)

X_test = test_rng.random((200, 5))
X_test[:, 0] = -10 - X_test[:, 0] * 10
X_test[:, 1] = X_test[:, 1] * 100
X_test[:, 2] = -9 + X_test[:, 2] * 10
X_test[:, 3] = 8 + X_test[:, 3]
X_test[:, 4] = X_test[:, 4] * 10

y_test_underlying = np.where(-X_test[:, 1] > 10 * X_test[:, 0] + 100, 10.0, 0.0)
y_test = y_test_underlying + test_rng.normal(size=y_test_underlying.size)

preds = model.predict(X_test)
preds_q = model.predict(X_test, kind="quantile", quantiles=(0.05, 0.95))
```

## 4. Visualising prediction performance

Plot observed values against predicted means. Vertical segments show 90%
credible intervals. Dashed pink lines mark the true underlying means (0 and
10).

```python
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(y_test, preds, color="darkblue", s=25, zorder=3)
for i in range(len(y_test)):
    ax.vlines(y_test[i], preds_q[i, 0], preds_q[i, 1], color=(0, 0, 0.5, 0.5), linewidth=1.5)

y_min, y_max = float(np.min(y_test)), float(np.max(y_test))
ax.plot([y_min - 0.2, 3], [0, 0], color="pink", linewidth=3, linestyle="--")
ax.plot([7, y_max + 0.2], [10, 10], color="pink", linewidth=3, linestyle="--")

ax.set(
    xlabel="Observed values",
    ylabel="Predicted mean values",
    title="Out-of-sample prediction performance",
)
ax.legend(
    ["Predicted mean and 90% interval", "True underlying mean"],
    loc="lower right",
    frameon=False,
)
plt.tight_layout()
plt.show()
```

A well-specified model should produce predictions clustered around the true
mean values of 0 and 10.
