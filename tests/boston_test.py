import time

import numpy as np
import pandas as pd
from addivortes import AddiVortesRegressor


url = "https://raw.githubusercontent.com/anonymous2738/AddiVortesAlgorithm/DataSets/BostonHousing_Data.csv"

Boston = pd.read_csv(url)

# R: X_Boston <- as.matrix(Boston[, 2:14])
# Python uses zero-based indexing, so columns 2:14 in R are iloc[:, 1:14]
X_Boston = Boston.iloc[:, 1:14].to_numpy(dtype=float)

# R: Y_Boston <- as.numeric(as.matrix(Boston[, 15]))
# Column 15 in R is iloc[:, 14] in Python
Y_Boston = Boston.iloc[:, 14].to_numpy(dtype=float)

n = len(Y_Boston)

rng = np.random.default_rng(1025)

# R:
# TrainSet <- sort(sample.int(n, 5 * n / 6))
#
# R uses 1-based indices; Python uses 0-based indices.
train_size = int(5 * n / 6)
train_set = np.sort(rng.choice(n, size=train_size, replace=False))

all_indices = np.arange(n)
test_set = np.setdiff1d(all_indices, train_set)

start = time.perf_counter()

model = AddiVortesRegressor(
    n_tessellations=200,    # R: m = 200
    total_mcmc_iter=2000,   # R: totalMCMCIter = 2000
    burn_in=200,            # R: mcmcBurnIn = 200
    nu=6,
    q=0.85,
    k=3,
    omega=3,
    lambda_rate=25,
    initial_sigma="linear", # R: InitialSigma = "Linear"
    random_state=1025,
    verbose=False,
)

model.fit(X_Boston[train_set], Y_Boston[train_set])

preds = model.predict(X_Boston[test_set])

elapsed = time.perf_counter() - start

print(f"Elapsed seconds: {elapsed:.3f}")

print("In-sample RMSE:")
print(model.in_sample_rmse_)

test_rmse = np.sqrt(np.mean((Y_Boston[test_set] - preds) ** 2))

print("Test RMSE:")
print(test_rmse)