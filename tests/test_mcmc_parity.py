from pathlib import Path

import numpy as np

from addivortes import _core

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "mcmc_parity"


def _load_fixture():
    meta = np.load(FIXTURE_DIR / "meta.npy", allow_pickle=True).item()
    return {
        "x_scaled": np.load(FIXTURE_DIR / "x_scaled.npy"),
        "y_scaled": np.load(FIXTURE_DIR / "y_scaled.npy"),
        "metric": np.load(FIXTURE_DIR / "metric.npy"),
        "members": np.load(FIXTURE_DIR / "members.npy"),
        "proposal_sd": np.load(FIXTURE_DIR / "proposal_sd.npy"),
        "proposal_mu": np.load(FIXTURE_DIR / "proposal_mu.npy"),
        "binary_cols": np.load(FIXTURE_DIR / "binary_cols.npy"),
        "init_tess": list(np.load(FIXTURE_DIR / "init_tess.npy", allow_pickle=True)),
        "init_dim": list(np.load(FIXTURE_DIR / "init_dim.npy", allow_pickle=True)),
        "init_pred": list(np.load(FIXTURE_DIR / "init_pred.npy", allow_pickle=True)),
        **meta,
    }


def test_mcmc_parity_fixture_is_stable():
    if not (FIXTURE_DIR / "meta.npy").exists():
        import importlib.util
        import sys

        spec = importlib.util.spec_from_file_location(
            "generate_fixture",
            FIXTURE_DIR / "generate_fixture.py",
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["generate_fixture"] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        module.main()

    data = _load_fixture()
    result = _core.run_mcmc(
        data["x_scaled"],
        data["y_scaled"],
        data["metric"],
        data["members"],
        data["m"],
        data["total_iter"],
        data["burn_in"],
        data["thinning"],
        data["nu"],
        data["lambda_value"],
        data["sigma_squared_mu"],
        data["omega"],
        data["lambda_rate"],
        data["proposal_sd"],
        data["proposal_mu"],
        data["init_tess"],
        data["init_dim"],
        data["init_pred"],
        data["binary_cols"],
        data["cat_scaling"],
        data["seed"],
        False,
    )

    reference_path = FIXTURE_DIR / "reference_posterior_sigma.npy"
    if not reference_path.exists():
        np.save(reference_path, np.asarray(result["posterior_sigma"], dtype=float))
        np.save(FIXTURE_DIR / "reference_trace_log_likelihood.npy", result["trace_stats"]["log_likelihood"])
        pytest_skip = False
    else:
        pytest_skip = True

    posterior_sigma = np.asarray(result["posterior_sigma"], dtype=float)
    trace_log_lik = np.asarray(result["trace_stats"]["log_likelihood"], dtype=float)

    if pytest_skip:
        np.testing.assert_allclose(posterior_sigma, np.load(reference_path))
        np.testing.assert_allclose(trace_log_lik, np.load(FIXTURE_DIR / "reference_trace_log_likelihood.npy"))

    assert result["trace_stats"]["iteration"].shape[0] == data["total_iter"]
    assert np.sum(result["trace_stats"]["is_burn_in"]) == data["burn_in"]
