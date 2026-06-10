"""Python interface for AddiVortes regression models."""

from importlib.metadata import PackageNotFoundError, version

from .model import AddiVortes, AddiVortesRegressor, plot

try:
    __version__ = version("addivortes")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = ["AddiVortes", "AddiVortesRegressor", "__version__", "plot"]
