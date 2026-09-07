"""Shared machinery for bi-objective Pareto-front generation."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

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
    weights = _objective_weights(costs, objectives, backend_weight_1, backend_weight_2)
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
        ranges=(abs(values_2[0] - values_1[0]), abs(values_1[1] - values_2[1])),
    )
