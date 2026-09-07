"""Augmented epsilon-constraint Pareto sweep."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import pandas as pd
import xarray as xr

from calliope.multiobjective.methods._common import (
    _UNMET_DEMAND_PENALTY,
    ParetoMethod,
    _anchor_solutions,
    _cost_expression,
    _objective_values,
    _safe_range,
    _validate_model,
)
from calliope.multiobjective.result import ParetoResult

if TYPE_CHECKING:
    from calliope.model import Model
    from calliope.multiobjective.study import Objective, ParetoStudy


@dataclass(frozen=True)
class AugmentedEpsilonConstraint(ParetoMethod):
    """Sweep an augmented bound on objective 2 while minimising objective 1."""

    name: ClassVar[str] = "augmented_epsilon_constraint"
    points: int = 11
    augmentation: float = 1e-6
    normalization_tolerance: float = 1e-12
    endpoint_tie_breaker: float = 1e-6

    _EPSILON_VARIABLE: ClassVar[str] = "pareto_epsilon_normalized"
    _SLACK_VARIABLE: ClassVar[str] = "pareto_epsilon_slack_normalized"
    _CONSTRAINT: ClassVar[str] = "pareto_epsilon_constraint"
    _OBJECTIVE: ClassVar[str] = "pareto_augmented_epsilon_objective"

    def __post_init__(self) -> None:
        """Validate sweep settings."""
        if self.points < 2:
            raise ValueError("AugmentedEpsilonConstraint.points must be at least 2.")
        if not 0 < self.augmentation < 1:
            raise ValueError("augmentation must be between 0 and 1.")
        if not 0 < self.endpoint_tie_breaker < 1:
            raise ValueError("endpoint_tie_breaker must be between 0 and 1.")

    def run(self, study: ParetoStudy) -> ParetoResult:
        """Generate efficient points over an automatic objective-2 epsilon grid."""
        model = study._new_built_model()
        objectives = (study.objective_1, study.objective_2)
        costs = _validate_model(model, objectives)
        anchors = _anchor_solutions(
            model,
            costs,
            objectives,
            study,
            self.normalization_tolerance,
            self.endpoint_tie_breaker,
        )

        if anchors.ranges[1] <= self.normalization_tolerance:
            raise ValueError(
                f"Objective `{objectives[1].cost_class}` has no measurable payoff "
                "range for an epsilon sweep."
            )

        epsilon_min = anchors.values_2[1]
        epsilon_max = anchors.values_1[1]
        epsilon_grid = np.linspace(epsilon_min, epsilon_max, self.points)
        if self.points > 2:
            primary_range = _safe_range(
                anchors.ranges[0], objectives[0], self.normalization_tolerance
            )
            self._add_backend_components(
                model, objectives, epsilon_min, primary_range, anchors.ranges[1]
            )

        rows: list[dict] = []
        solutions: list[xr.Dataset] = []
        for point_id, epsilon in enumerate(epsilon_grid):
            if point_id == 0:
                result = anchors.endpoint_2
            elif point_id == self.points - 1:
                result = anchors.endpoint_1
            else:
                model.backend.update_variable_bounds(
                    self._EPSILON_VARIABLE,
                    min=float(epsilon) / anchors.ranges[1],
                    max=float(epsilon) / anchors.ranges[1],
                )
                model.solve(force=True, **study.solve_options)
                result = model.results.copy(deep=True)

            objective_1, objective_2 = _objective_values(result, objectives)
            rows.append(
                {
                    "point_id": point_id,
                    "epsilon": float(epsilon),
                    "slack": max(0.0, float(epsilon) - objective_2),
                    "objective_1": objective_1,
                    "objective_2": objective_2,
                }
            )
            solutions.append(result)

        return ParetoResult(
            method=self.name,
            objective_1=objectives[0],
            objective_2=objectives[1],
            points=pd.DataFrame(rows),
            solutions=tuple(solutions),
        )

    def _add_backend_components(
        self,
        model: Model,
        objectives: tuple[Objective, Objective],
        epsilon: float,
        primary_range: float,
        constrained_range: float,
    ) -> None:
        """Inject the scalar AUGMECON formulation into a built backend."""
        primary = _cost_expression(objectives[0])
        constrained = _cost_expression(objectives[1])
        normalized_epsilon = float(epsilon) / constrained_range
        augmentation_coefficient = self.augmentation * primary_range

        model.backend.add_variable(
            self._EPSILON_VARIABLE,
            {
                "bounds": {"min": normalized_epsilon, "max": normalized_epsilon},
                "default": normalized_epsilon,
            },
        )
        model.backend.add_variable(
            self._SLACK_VARIABLE, {"bounds": {"min": 0, "max": np.inf}, "default": 0}
        )
        model.backend.add_constraint(
            self._CONSTRAINT,
            {
                "equations": [
                    {
                        "expression": (
                            f"{constrained} / {constrained_range!r} + "
                            f"{self._SLACK_VARIABLE} "
                            f"== {self._EPSILON_VARIABLE}"
                        )
                    }
                ]
            },
        )
        model.backend.add_objective(
            self._OBJECTIVE,
            {
                "equations": [
                    {
                        "expression": (
                            f"{primary} - {augmentation_coefficient!r} * "
                            f"{self._SLACK_VARIABLE} + {_UNMET_DEMAND_PENALTY}"
                        )
                    }
                ],
                "sense": "minimise",
            },
        )
        model.backend.set_objective(self._OBJECTIVE)
