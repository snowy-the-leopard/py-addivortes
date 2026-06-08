import sys
import types

import numpy as np
import pytest

from addivortes import plot

from conftest import fast_model, regression_data


class FakeFigure:
    def __init__(self):
        self.tight_layout_called = False

    def tight_layout(self):
        self.tight_layout_called = True


class FakeAxis:
    transAxes = object()

    def __init__(self, figure):
        self.figure = figure
        self.calls = []
        self.visible = True

    def scatter(self, *args, **kwargs):
        self.calls.append(("scatter", args, kwargs))

    def plot(self, *args, **kwargs):
        self.calls.append(("plot", args, kwargs))

    def axhline(self, *args, **kwargs):
        self.calls.append(("axhline", args, kwargs))

    def vlines(self, *args, **kwargs):
        self.calls.append(("vlines", args, kwargs))

    def text(self, *args, **kwargs):
        self.calls.append(("text", args, kwargs))

    def legend(self, *args, **kwargs):
        self.calls.append(("legend", args, kwargs))

    def set(self, *args, **kwargs):
        self.calls.append(("set", args, kwargs))

    def set_xlim(self, *args, **kwargs):
        self.calls.append(("set_xlim", args, kwargs))

    def set_ylim(self, *args, **kwargs):
        self.calls.append(("set_ylim", args, kwargs))

    def set_visible(self, visible):
        self.visible = visible


class FakePyplot(types.ModuleType):
    def __init__(self):
        super().__init__("matplotlib.pyplot")
        self.figures = []
        self.show_called = False

    def subplots(self, rows, cols, squeeze=False):
        assert squeeze is False
        figure = FakeFigure()
        axes = np.array([[FakeAxis(figure) for _ in range(cols)] for _ in range(rows)], dtype=object)
        self.figures.append(figure)
        return figure, axes

    def show(self):
        self.show_called = True


@pytest.fixture
def fake_pyplot(monkeypatch):
    matplotlib = types.ModuleType("matplotlib")
    pyplot = FakePyplot()
    monkeypatch.setitem(sys.modules, "matplotlib", matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", pyplot)
    return pyplot


def test_plot_draws_requested_diagnostics(fake_pyplot):
    x, y = regression_data(seed=2024, n_obs=18, n_features=3)
    model = fast_model(n_tessellations=4, total_mcmc_iter=16, burn_in=6, random_state=2024).fit(x, y)
    sigma_trace = np.linspace(0.1, 0.5, len(model.posterior_.tessellations))

    axes = model.plot(x, y, sigma_trace=sigma_trace, which=(1, 2, 3, 4), show=True)

    assert len(axes) == 4
    assert fake_pyplot.show_called
    assert axes[0].figure.tight_layout_called
    assert any(call[0] == "scatter" for call in axes[0].calls)
    assert any(call[0] == "axhline" for call in axes[0].calls)
    assert any(call[0] == "plot" for call in axes[1].calls)
    assert any(call[0] == "plot" for call in axes[2].calls)
    assert any(call[0] == "vlines" for call in axes[3].calls)
    assert any(call[0] == "legend" for call in axes[3].calls)


def test_top_level_plot_uses_supplied_axes(fake_pyplot):
    x, y = regression_data(seed=2025, n_obs=18, n_features=3)
    model = fast_model(n_tessellations=4, total_mcmc_iter=16, burn_in=6, random_state=2025).fit(x, y)
    figure = FakeFigure()
    supplied_axes = [FakeAxis(figure), FakeAxis(figure)]

    axes = plot(model, x, y, which=(2, 3), axes=supplied_axes)

    assert axes == supplied_axes
    assert not fake_pyplot.figures
    assert any(call[0] == "plot" for call in axes[0].calls)
    assert any(call[0] == "plot" for call in axes[1].calls)


def test_plot_validation_errors(fake_pyplot):
    x, y = regression_data(seed=2026, n_obs=18, n_features=3)

    with pytest.raises(RuntimeError, match="not fitted"):
        fast_model().plot(x, y)

    model = fast_model(n_tessellations=4, total_mcmc_iter=16, burn_in=6, random_state=2026).fit(x, y)

    with pytest.raises(ValueError, match="same number of observations"):
        model.plot(x, y[:-1])

    with pytest.raises(ValueError, match="which"):
        model.plot(x, y, which=(99,))

    with pytest.raises(ValueError, match="sigma_trace"):
        model.plot(x, y, sigma_trace=np.array([[1.0]]), which=(2,))
