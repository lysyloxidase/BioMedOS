"""Forecast biomedical topic trajectories."""

from __future__ import annotations


class ForecastModel:
    """Forecast future literature or discovery trends."""

    def forecast(self, series: list[float], *, horizon: int = 12) -> list[float]:
        """Forecast future observations.

        Args:
            series: Historical observations.
            horizon: Number of future time steps.

        Returns:
            Forecast values.
        """

        # TODO: Implement forecasting over literature and discovery trend series.
        raise NotImplementedError("ForecastModel.forecast is not implemented yet.")
