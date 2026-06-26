import numpy as np

from conftest import fast_model, regression_data


def _run_fixture_mcmc(**overrides):
    x, y = regression_data(seed=7, n_obs=12, n_features=3)
    model = fast_model(**overrides)
    model.fit(x, y)
    return model


def test_trace_stats_cover_all_iterations():
    model = _run_fixture_mcmc(total_mcmc_iter=20, burn_in=8, thinning=2)
    trace = model.trace_stats_

    assert trace is not None
    assert trace.iteration.shape == (20,)
    assert trace.is_burn_in.shape == (20,)
    assert np.array_equal(trace.iteration, np.arange(1, 21))
    assert np.sum(trace.is_burn_in) == 8
    assert np.all(~trace.is_burn_in[8:])
    assert trace.average_centres_per_tessellation.shape == (20,)
    assert trace.sd_centres_per_tessellation.shape == (20,)
    assert trace.average_dimensions_per_tessellation.shape == (20,)
    assert trace.log_likelihood.shape == (20,)
    assert np.all(np.isfinite(trace.log_likelihood))


def test_posterior_sample_count_respects_burn_in_and_thinning():
    model = _run_fixture_mcmc(total_mcmc_iter=20, burn_in=8, thinning=2)
    expected = (20 - 8) // 2
    assert len(model.posterior_.tessellations) == expected
    assert model.posterior_.sigma.shape == (expected,)
    assert model.posterior_.prediction_matrix.shape == (12, expected)


def test_mcmc_with_fixed_seed_is_reproducible():
    x, y = regression_data(seed=11, n_obs=10, n_features=2)
    kwargs = dict(n_tessellations=2, total_mcmc_iter=12, burn_in=4, thinning=1, random_state=99)

    first = fast_model(**kwargs)
    second = fast_model(**kwargs)
    first.fit(x, y)
    second.fit(x, y)

    np.testing.assert_allclose(first.posterior_.sigma, second.posterior_.sigma)
    np.testing.assert_allclose(
        first.trace_stats_.log_likelihood,
        second.trace_stats_.log_likelihood,
    )
