"""Pareto-front generation methods.

Weighted-sum, augmented epsilon-constraint, and augmented Tchebycheff sweeps
are implemented without putting method branches into :class:`calliope.Model`.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import pandas as pd
import xarray as xr

from calliope.multiobjective.result import ParetoResult

if TYPE_CHECKING:
    from calliope.model import Model
    from calliope.multiobjective.study import Objective, ParetoStudy


class ParetoMethod(ABC):
    """Interface implemented by every Pareto-front generation strategy."""

    name: ClassVar[str]

    @abstractmethod
    def run(self, study: ParetoStudy) -> ParetoResult:
        """Generate points for ``study``."""


@dataclass(frozen=True)
class _AnchorSolutions:
    """Two refined single-objective endpoints and their payoff ranges."""

    endpoint_1: xr.Dataset
    endpoint_2: xr.Dataset
    ideal_values: tuple[float, float]
    values_1: tuple[float, float]
    values_2: tuple[float, float]
    ranges: tuple[float, float]


_UNMET_DEMAND_PENALTY = (
    "sum(sum(unmet_demand - unused_supply, over=[carriers, nodes]) "
    "* timestep_weights, over=timesteps) * bigM"
)


def _validate_model(model: Model, objectives: tuple[Objective, Objective]) -> list:
    if "costs" not in model.inputs.coords:
        raise ValueError("Pareto methods require a Calliope `costs` dimension.")
    if "objective_cost_weights" not in model.backend.inputs:
        raise ValueError("Pareto methods require the `objective_cost_weights` input.")

    costs = list(model.inputs.coords["costs"].values)
    missing = [obj.cost_class for obj in objectives if obj.cost_class not in costs]
    if missing:
        raise ValueError(f"Unknown objective cost class(es): {missing}")
    return costs


def _objective_values(
    result: xr.Dataset, objectives: tuple[Objective, Objective]
) -> tuple[float, float]:
    return tuple(
        float(result["cost"].sel(costs=objective.cost_class).fillna(0).sum().item())
        for objective in objectives
    )


def _objective_weights(
    costs: list,
    objectives: tuple[Objective, Objective],
    weight_1: float,
    weight_2: float,
) -> xr.DataArray:
    """Create a complete Calliope cost-class weight array."""
    return xr.DataArray(
        [
            weight_1
            if cost == objectives[0].cost_class
            else weight_2
            if cost == objectives[1].cost_class
            else 0.0
            for cost in costs
        ],
        coords={"costs": costs},
        dims="costs",
    )


def _cost_expression(objective: Objective) -> str:
    """Return the scalar total-cost expression for one cost class."""
    return f"sum(cost[costs={objective.cost_class}], over=[nodes, techs])"


def _safe_range(value: float, objective: Objective, tolerance: float) -> float:
    if value > tolerance:
        return value
    warnings.warn(
        f"Objective `{objective.cost_class}` has no measurable payoff range; "
        "using 1.0 for normalisation.",
        stacklevel=3,
    )
    return 1.0


def _solve_weighted_objective(
    model: Model,
    costs: list,
    objectives: tuple[Objective, Objective],
    backend_weight_1: float,
    backend_weight_2: float,
    study: ParetoStudy,
) -> xr.Dataset:
    """Solve one weighted objective while keeping its coefficients well scaled."""
    scale = max(abs(backend_weight_1), abs(backend_weight_2))
    backend_weight_1 /= scale
    backend_weight_2 /= scale
    weights = _objective_weights(
        costs, objectives, backend_weight_1, backend_weight_2
    )
    model.backend.update_input("objective_cost_weights", weights)
    model.backend.set_objective("min_cost_optimisation")
    model.solve(force=model.is_solved, **study.solve_options)
    return model.results.copy(deep=True)


def _anchor_solutions(
    model: Model,
    costs: list,
    objectives: tuple[Objective, Objective],
    study: ParetoStudy,
    normalization_tolerance: float,
    endpoint_tie_breaker: float,
) -> _AnchorSolutions:
    """Solve and refine both single-objective endpoints."""
    raw_endpoint_1 = _solve_weighted_objective(
        model, costs, objectives, 1.0, 0.0, study
    )
    raw_endpoint_2 = _solve_weighted_objective(
        model, costs, objectives, 0.0, 1.0, study
    )
    raw_values_1 = _objective_values(raw_endpoint_1, objectives)
    raw_values_2 = _objective_values(raw_endpoint_2, objectives)
    provisional_ranges = (
        _safe_range(
            abs(raw_values_2[0] - raw_values_1[0]),
            objectives[0],
            normalization_tolerance,
        ),
        _safe_range(
            abs(raw_values_1[1] - raw_values_2[1]),
            objectives[1],
            normalization_tolerance,
        ),
    )

    endpoint_1 = _solve_weighted_objective(
        model,
        costs,
        objectives,
        1 / provisional_ranges[0],
        endpoint_tie_breaker / provisional_ranges[1],
        study,
    )
    endpoint_2 = _solve_weighted_objective(
        model,
        costs,
        objectives,
        endpoint_tie_breaker / provisional_ranges[0],
        1 / provisional_ranges[1],
        study,
    )
    values_1 = _objective_values(endpoint_1, objectives)
    values_2 = _objective_values(endpoint_2, objectives)
    return _AnchorSolutions(
        endpoint_1=endpoint_1,
        endpoint_2=endpoint_2,
        ideal_values=(raw_values_1[0], raw_values_2[1]),
        values_1=values_1,
        values_2=values_2,
        ranges=(
            abs(values_2[0] - values_1[0]),
            abs(values_1[1] - values_2[1]),
        ),
    )


@dataclass(frozen=True)
class WeightedSumSweep(ParetoMethod):
    """Sweep normalised weights over two Calliope cost classes."""

    name: ClassVar[str] = "weighted_sum"
    points: int = 11
    normalization_tolerance: float = 1e-12
    endpoint_tie_breaker: float = 1e-6

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
                anchors.ranges[0],
                objectives[0],
                self.normalization_tolerance,
            ),
            _safe_range(
                anchors.ranges[1],
                objectives[1],
                self.normalization_tolerance,
            ),
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

        return ParetoResult(
            method=self.name,
            objective_1=objectives[0],
            objective_2=objectives[1],
            points=pd.DataFrame(rows),
            solutions=tuple(solutions),
        )


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
            raise ValueError(
                "AugmentedEpsilonConstraint.points must be at least 2."
            )
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
                anchors.ranges[0],
                objectives[0],
                self.normalization_tolerance,
            )
            self._add_backend_components(
                model,
                objectives,
                epsilon_min,
                primary_range,
                anchors.ranges[1],
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
            self._SLACK_VARIABLE,
            {"bounds": {"min": 0, "max": np.inf}, "default": 0},
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
            raise ValueError(
                "AugmentedTchebycheffSweep.points must be at least 2."
            )
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
            (
                f"(({_cost_expression(objective)}) - {ideal!r}) "
                f"/ {objective_range!r}"
            )
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
