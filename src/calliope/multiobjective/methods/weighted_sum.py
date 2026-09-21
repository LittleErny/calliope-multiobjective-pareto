"""Normalised weighted-sum Pareto sweep."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import pandas as pd
import xarray as xr

from calliope.multiobjective.methods._common import (
    ParetoMethod,
    _anchor_solutions,
    _objective_values,
    _report_progress,
    _safe_range,
    _solve_weighted_objective,
    _validate_model,
)
from calliope.multiobjective.result import ParetoResult

if TYPE_CHECKING:
    from calliope.multiobjective.study import ParetoStudy


@dataclass(frozen=True)
class WeightedSumSweep(ParetoMethod):
    """Sweep normalised weights over two Calliope cost classes."""

    name: ClassVar[str] = "weighted_sum"
    points: int = 11
    normalization_tolerance: float = 1e-12
    endpoint_tie_breaker: float = 1e-6
    show_progress: bool = False

    def __post_init__(self) -> None:
        """Validate sweep settings."""
        if self.points < 2:
            raise ValueError("WeightedSumSweep.points must be at least 2.")
        if not 0 < self.endpoint_tie_breaker < 1:
            raise ValueError("endpoint_tie_breaker must be between 0 and 1.")

    def run(self, study: ParetoStudy) -> ParetoResult:
        """Build once, update objective weights, and solve for every weight."""
        model = study._new_built_model()
        objectives = (study.objective_1, study.objective_2)
        costs = _validate_model(model, objectives)
        _report_progress(self.show_progress, "Weighted sum", None, self.points)
        anchors = _anchor_solutions(
            model,
            costs,
            objectives,
            study,
            self.normalization_tolerance,
            self.endpoint_tie_breaker,
        )
        ranges = (
            _safe_range(anchors.ranges[0], objectives[0], self.normalization_tolerance),
            _safe_range(anchors.ranges[1], objectives[1], self.normalization_tolerance),
        )

        rows: list[dict] = []
        solutions: list[xr.Dataset] = []
        for point_id, weight_1 in enumerate(np.linspace(0, 1, self.points)):
            weight_2 = 1 - weight_1
            if np.isclose(weight_1, 0):
                result = anchors.endpoint_2
            elif np.isclose(weight_1, 1):
                result = anchors.endpoint_1
            else:
                result = _solve_weighted_objective(
                    model,
                    costs,
                    objectives,
                    weight_1 / ranges[0],
                    weight_2 / ranges[1],
                    study,
                )

            objective_1, objective_2 = _objective_values(result, objectives)
            rows.append(
                {
                    "point_id": point_id,
                    "weight_1": float(weight_1),
                    "weight_2": float(weight_2),
                    "objective_1": objective_1,
                    "objective_2": objective_2,
                }
            )
            solutions.append(result)
            _report_progress(
                self.show_progress, "Weighted sum", point_id + 1, self.points
            )

        return ParetoResult(
            method=self.name,
            objective_1=objectives[0],
            objective_2=objectives[1],
            points=pd.DataFrame(rows),
            solutions=tuple(solutions),
        )
