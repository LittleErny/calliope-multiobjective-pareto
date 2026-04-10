import pytest
import xarray as xr

from calliope.exceptions import ModelError

from .common.util import build_test_model as build_model


class TestParetoMode:
    def test_pareto_mode_solves_multiple_epsilons(self):
        # Build from a minimal two-tech trade-off model
        m = build_model(model_file="model_pareto.yaml")
        m.build()
        m.solve()

        assert "pareto" in m.results.dims
        eps = list(m.results.coords["pareto"].values)
        assert eps == sorted(eps)
        assert len(eps) == 3

        # Check that the secondary cost constraint is respected at each pareto point
        # (The pareto math adds a postprocessed `pareto_secondary_cost_total` scalar per point.)
        secondary_total = m.results["pareto_secondary_cost_total"].values
        for e, tot in zip(eps, secondary_total):
            assert float(tot) <= float(e) + 1e-6

        # Check that tightening epsilon cannot decrease the primary cost
        primary_total = m.results["pareto_primary_cost_total"]
        # As eps increases, mprimray_total should be non-increasing
        # (relaxing the emissions cap can only help the primary objective)
        assert (primary_total.diff("pareto") <= 1e-6).all()

    def test_pareto_requires_epsilons(self):
        m = build_model(model_file="model_pareto.yaml")
        m.config = m.config.update({"solve.pareto.epsilons": []})
        m.build()
        with pytest.raises(ModelError, match="requires a non-empty list of epsilons"):
            m.solve(force=True)

    def test_pareto_errors_on_missing_secondary_cost(self):
        m = build_model(model_file="model_pareto.yaml")
        m.config = m.config.update({"solve.pareto.secondary_cost": "co2"})
        m.build()
        with pytest.raises(ModelError, match="Secondary cost 'co2' not found"):
            m.solve(force=True)
