"""Augmented weighted Tchebycheff Pareto sweep."""

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
    _objective_weights,
    _safe_range,
    _validate_model,
)
from calliope.multiobjective.result import ParetoResult

if TYPE_CHECKING:
    from calliope.model import Model
    from calliope.multiobjective.study import Objective, ParetoStudy


@dataclass(frozen=True)
class AugmentedTchebycheffSweep(ParetoMethod):
    """Sweep an augmented weighted Tchebycheff scalarisation."""

    name: ClassVar[str] = "augmented_tchebycheff"
    points: int = 11
    augmentation: float = 1e-6
    normalization_tolerance: float = 1e-12
    endpoint_tie_breaker: float = 1e-6

    _AUXILIARY_VARIABLE: ClassVar[str] = "pareto_tchebycheff_z"
    _CONSTRAINT_1: ClassVar[str] = "pareto_tchebycheff_constraint_1"
    _CONSTRAINT_2: ClassVar[str] = "pareto_tchebycheff_constraint_2"
    _OBJECTIVE: ClassVar[str] = "pareto_augmented_tchebycheff_objective"

    def __post_init__(self) -> None:
        """Validate sweep settings."""
        if self.points < 2:
            raise ValueError("AugmentedTchebycheffSweep.points must be at least 2.")
        if not 0 < self.augmentation < 1:
            raise ValueError("augmentation must be between 0 and 1.")
        if not 0 < self.endpoint_tie_breaker < 1:
            raise ValueError("endpoint_tie_breaker must be between 0 and 1.")

    def run(self, study: ParetoStudy) -> ParetoResult:
        """Generate efficient points over a two-objective weight grid."""
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

        ranges = (
            _safe_range(
                abs(anchors.values_2[0] - anchors.ideal_values[0]),
                objectives[0],
                self.normalization_tolerance,
            ),
            _safe_range(
                abs(anchors.values_1[1] - anchors.ideal_values[1]),
                objectives[1],
                self.normalization_tolerance,
            ),
        )
        if self.points > 2:
            self._add_backend_components(
                model, objectives, anchors.ideal_values, ranges
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
                weights = _objective_weights(
                    costs, objectives, float(weight_1), float(weight_2)
                )
                model.backend.update_input("objective_cost_weights", weights)
                model.solve(force=True, **study.solve_options)
                result = model.results.copy(deep=True)

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
        ideal_values: tuple[float, float],
        ranges: tuple[float, float],
    ) -> None:
        """Inject the augmented weighted Tchebycheff formulation."""
        normalized = tuple(
            (f"(({_cost_expression(objective)}) - {ideal!r}) / {objective_range!r}")
            for objective, ideal, objective_range in zip(
                objectives, ideal_values, ranges
            )
        )

        model.backend.add_variable(
            self._AUXILIARY_VARIABLE,
            {"bounds": {"min": 0, "max": np.inf}, "default": 0},
        )
        for name, objective, expression in zip(
            (self._CONSTRAINT_1, self._CONSTRAINT_2), objectives, normalized
        ):
            model.backend.add_constraint(
                name,
                {
                    "equations": [
                        {
                            "expression": (
                                "objective_cost_weights"
                                f"[costs={objective.cost_class}] * "
                                f"{expression} <= {self._AUXILIARY_VARIABLE}"
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
                            f"{self._AUXILIARY_VARIABLE} + "
                            f"{self.augmentation!r} * "
                            f"({normalized[0]} + {normalized[1]}) + "
                            f"{_UNMET_DEMAND_PENALTY}"
                        )
                    }
                ],
                "sense": "minimise",
            },
        )
        model.backend.set_objective(self._OBJECTIVE)
