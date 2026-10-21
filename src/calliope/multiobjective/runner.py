"""Run Pareto scalarisations as a native Calliope solve mode."""

from __future__ import annotations

from typing import TYPE_CHECKING

import xarray as xr

from calliope import exceptions
from calliope.multiobjective.methods import (
    AugmentedEpsilonConstraint,
    AugmentedTchebycheffSweep,
    WeightedSumSweep,
)
from calliope.multiobjective.study import Objective, ParetoRun

if TYPE_CHECKING:
    from calliope.model import Model
    from calliope.schemas.config_schema import Solve


def solve_multiobjective(model: Model, solver_config: Solve) -> xr.Dataset:
    """Solve a configured Pareto sweep on an already built Calliope model."""
    config = solver_config.multiobjective
    if not config.objective_1 or not config.objective_2:
        raise exceptions.ModelError(
            "Multiobjective mode requires `solve.multiobjective.objective_1` "
            "and `solve.multiobjective.objective_2` cost classes."
        )
    if config.objective_1 == config.objective_2:
        raise exceptions.ModelError(
            "Multiobjective mode requires two different cost classes."
        )
    if "costs" not in model.inputs.coords:
        raise exceptions.ModelError("Multiobjective mode requires a `costs` dimension.")
    costs = set(model.inputs.costs.values.tolist())
    missing = {config.objective_1, config.objective_2} - costs
    if missing:
        raise exceptions.ModelError(
            f"Unknown multiobjective cost class(es): {sorted(missing)}."
        )

    method_class = {
        "weighted_sum": WeightedSumSweep,
        "augmented_epsilon_constraint": AugmentedEpsilonConstraint,
        "augmented_tchebycheff": AugmentedTchebycheffSweep,
    }[config.method]
    options = {
        "points": config.points,
        "normalization_tolerance": config.normalization_tolerance,
        "endpoint_tie_breaker": config.endpoint_tie_breaker,
        "show_progress": config.show_progress,
    }
    if config.method != "weighted_sum":
        options["augmentation"] = config.augmentation

    solve_count = 0

    def solve_one() -> xr.Dataset:
        nonlocal solve_count
        solve_count += 1
        result = model.backend._solve(solver_config, warmstart=False)
        condition = result.attrs.get("termination_condition")
        if condition not in {"optimal", "feasible"} or "cost" not in result:
            raise exceptions.ModelError(
                f"Multiobjective scalar solve {solve_count} did not return a "
                f"usable solution (termination condition: {condition})."
            )
        return result.copy(deep=True)

    run = ParetoRun(
        model, Objective(config.objective_1), Objective(config.objective_2), solve_one
    )
    result = method_class(**options).run(run)
    return result.to_dataset()
