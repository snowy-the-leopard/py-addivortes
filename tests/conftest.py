from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for entry in tuple(sys.path):
    if Path(entry or ".").resolve() == REPO_ROOT:
        sys.path.remove(entry)

from addivortes import AddiVortesRegressor


def regression_data(seed=123, n_obs=18, n_features=3):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_obs, n_features))
    y = x[:, 0] - 0.5 * x[:, min(1, n_features - 1)] + rng.normal(scale=0.1, size=n_obs)
    return x, y


def fast_model(**overrides):
    params = {
        "n_tessellations": 4,
        "total_mcmc_iter": 16,
        "burn_in": 6,
        "thinning": 2,
        "random_state": 42,
    }
    params.update(overrides)
    return AddiVortesRegressor(**params)
