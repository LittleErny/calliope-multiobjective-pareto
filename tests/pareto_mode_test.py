import pytest

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
        assert eps == [20, 110, 200]

        # Check that the secondary cost constraint is respected at each pareto point
        # (The pareto math adds a postprocessed `pareto_secondary_cost_total` scalar per point.)
        secondary_total = m.results["pareto_secondary_cost_total"].values
        slack = m.results["pareto_slack"].values
        for e, tot in zip(eps, secondary_total):
            assert float(tot) <= float(e) + 1e-6
        assert secondary_total + slack == pytest.approx(eps)

        assert m.results["pareto_primary_cost_total"].values == pytest.approx(
            [60, 40, 20]
        )
        assert m.results["pareto_secondary_cost_total"].values == pytest.approx(
            [20, 110, 200]
        )

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

    def test_pareto_math_is_not_loaded_in_base_mode(self):
        m = build_model(
            model_file="model_pareto.yaml",
            override_dict={"config.init.mode": "base"},
        )
        m.build()

        assert "pareto_epsilon" not in m.math.build.parameters
        assert "pareto_secondary_cost_weights" not in m.math.build.parameters

    def test_pareto_secondary_metric_can_be_overridden_by_math(self):
        m = build_model(
            model_file="model_pareto.yaml",
            math_dict={
                "global_expressions": {
                    "pareto_secondary_metric": {
                        "description": "Doubled emissions metric.",
                        "default": 0,
                        "unit": "cost",
                        "equations": [
                            {
                                "expression": "2 * sum(sum(cost, over=[nodes, techs]) * pareto_secondary_cost_weights, over=costs)"
                            }
                        ],
                    }
                }
            },
            override_dict={"config.solve.pareto.epsilons": [40, 220, 400]},
        )
        m.build()
        m.solve()

        assert m.results["pareto_primary_cost_total"].values == pytest.approx(
            [60, 40, 20]
        )

    def test_pareto_generates_epsilons_from_payoff_table(self):
        m = build_model(
            model_file="model_pareto.yaml",
            override_dict={
                "config.solve.pareto.epsilons": [],
                "config.solve.pareto.n_points": 3,
            },
        )
        m.build()
        m.solve()

        assert list(m.results.coords["pareto"].values) == [20, 110, 200]
