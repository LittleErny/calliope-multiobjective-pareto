import pytest

from calliope.exceptions import ModelError
from .common.util import build_test_model as build_model


class TestParetoMode:
    def test_pareto_mode_solves_multiple_epsilons(self):
        # Build from the pareto toy model
        m = build_model(model_file="model_pareto.yaml")

        # Force using a solver available on this machine
        # (avoid relying on whatever is written in the YAML, e.g. cbc)
        m.config = m.config.update({"solve.solver": "glpk"})

        # Make epsilons "safe" to avoid infeasibility due to scale/unit differences.
        # This guarantees the epsilon constraint is feasible and we can test
        # that the pareto mode runs and respects the constraint.
        m.config = m.config.update({"solve.pareto.epsilons": [1e6, 1e8, 1e10]})

        m.build()
        m.solve()

        assert "pareto" in m.results.dims
        eps = list(m.results.coords["pareto"].values)
        assert eps == sorted(eps)
        assert len(eps) == 3

        # Check that the secondary cost constraint is respected at each pareto point
        secondary_total = m.results["pareto_secondary_cost_total"].values
        for e, tot in zip(eps, secondary_total):
            assert float(tot) <= float(e) + 1e-6

        # Check that relaxing epsilon cannot increase the primary cost
        monetary = (
            m.results["cost"].sel(costs="monetary").fillna(0).sum(("nodes", "techs"))
        )
        # As eps increases, monetary should be non-increasing (or equal)
        assert (monetary.diff("pareto") <= 1e-6).all()

    def test_pareto_requires_epsilons(self):
        m = build_model(model_file="model_pareto.yaml")
        m.config = m.config.update({"solve.solver": "glpk"})
        m.config = m.config.update({"solve.pareto.epsilons": []})
        m.build()
        with pytest.raises(ModelError, match="requires a non-empty list of epsilons"):
            m.solve(force=True)

    def test_pareto_errors_on_missing_secondary_cost(self):
        m = build_model(model_file="model_pareto.yaml")
        m.config = m.config.update({"solve.solver": "glpk"})
        m.config = m.config.update({"solve.pareto.secondary_cost": "co2"})
        m.build()
        with pytest.raises(ModelError, match="Secondary cost 'co2' not found"):
            m.solve(force=True)
