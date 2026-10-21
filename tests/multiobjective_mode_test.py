"""Integration checks for Calliope's native multiobjective solve mode."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import calliope
from calliope.exceptions import ModelError
from calliope.multiobjective import ParetoResult

MODEL_PATH = (
    Path(__file__).parents[1]
    / "src/calliope/multiobjective/examples/national_scale/model.yaml"
)


def _built_model(configured=True):
    model = calliope.read_yaml(
        MODEL_PATH,
        scenario=(
            "minimize_emissions_costs,multiobjective"
            if configured
            else "minimize_emissions_costs"
        ),
        **({} if configured else {"mode": "multiobjective"}),
        subset={"timesteps": ["2005-01-01 00:00:00", "2005-01-01 01:00:00"]},
    )
    assert model.runtime.math_priority == ["base", "multiobjective"]
    model.build()
    assert model.backend.inputs.objective_cost_weights.dims == ("costs",)
    return model


def test_native_mode_expands_scalar_cost_weights():
    model = calliope.read_yaml(
        MODEL_PATH,
        scenario="minimize_emissions_costs,multiobjective",
        override_dict={"data_definitions": {"objective_cost_weights": 1}},
        subset={"timesteps": ["2005-01-01 00:00:00", "2005-01-01 01:00:00"]},
    )
    model.build()
    assert model.backend.inputs.objective_cost_weights.dims == ("costs",)


@pytest.mark.parametrize(
    "method", ["weighted_sum", "augmented_epsilon_constraint", "augmented_tchebycheff"]
)
def test_native_mode_solves_and_retains_complete_results(method):
    model = _built_model()
    if method == "weighted_sum":
        model.solve()
    else:
        model.solve(multiobjective={"method": method})

    assert model.runtime.termination_condition == "optimal"
    assert model.results.pareto_point.values.tolist() == [0, 1, 2]
    assert model.results.attrs["pareto_method"] == method
    assert "flow_cap" in model.results
    assert "cost" in model.results
    np.testing.assert_allclose(
        model.results.pareto_objective_1,
        model.results.cost.sel(costs="monetary").fillna(0).sum(("nodes", "techs")),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        model.results.pareto_objective_2,
        model.results.cost.sel(costs="emissions").fillna(0).sum(("nodes", "techs")),
        rtol=1e-6,
    )
    assert model.results.pareto_objective_1[0] > model.results.pareto_objective_1[-1]
    assert model.results.pareto_objective_2[0] < model.results.pareto_objective_2[-1]


def test_native_mode_roundtrip_and_force_rerun(tmp_path):
    model = _built_model()
    model.solve(multiobjective={"method": "augmented_tchebycheff"})
    path = tmp_path / "pareto.nc"
    model.to_netcdf(path)
    restored = calliope.read_netcdf(path)
    xr.testing.assert_allclose(model.results, restored.results)
    pareto = ParetoResult.from_dataset(restored.results)
    assert pareto.points.point_id.tolist() == [0, 1, 2]
    xr.testing.assert_allclose(
        pareto.solution(1).cost, restored.results.sel(pareto_point=1, drop=True).cost
    )

    model.solve(force=True, multiobjective={"method": "weighted_sum", "points": 4})
    assert model.results.sizes["pareto_point"] == 4
    assert model.results.attrs["pareto_method"] == "weighted_sum"
    assert model.runtime.termination_condition == "optimal"


def test_native_mode_requires_two_known_cost_classes():
    model = _built_model(configured=False)
    with pytest.raises(ModelError, match="requires.*objective_1"):
        model.solve()
    with pytest.raises(ModelError, match="two different"):
        model.solve(
            multiobjective={"objective_1": "monetary", "objective_2": "monetary"}
        )
    with pytest.raises(ModelError, match="Unknown multiobjective cost class"):
        model.solve(multiobjective={"objective_2": "missing"})
