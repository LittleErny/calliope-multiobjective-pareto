import numpy as np
import pytest
import xarray as xr

from calliope.multiobjective import (
    AugmentedEpsilonConstraint,
    AugmentedTchebycheffSweep,
    Objective,
    ParetoStudy,
    WeightedSumSweep,
)


class _FakeBackend:
    def __init__(self, model):
        self.model = model
        self.objective = "min_cost_optimisation"
        self.inputs = xr.Dataset(
            {"objective_cost_weights": ("costs", [1.0, 0.0])},
            coords={"costs": ["money", "emissions"]},
        )

    def update_input(self, name, values):
        self.inputs[name] = values

    def set_objective(self, name):
        self.objective = name

    def add_parameter(self, name, values, definition):
        self.inputs[name] = values

    def add_variable(self, name, definition):
        if name == "pareto_epsilon_normalized":
            self.inputs[name] = definition["bounds"]["min"]

    def update_variable_bounds(self, name, *, min=None, max=None):
        assert min == max
        self.inputs[name] = min

    def add_constraint(self, name, definition):
        pass

    def add_objective(self, name, definition):
        pass


class _FakeModel:
    feasible_points = ((1.0, 10.0), (3.0, 5.0), (10.0, 1.0))

    def __init__(self):
        self.inputs = xr.Dataset(coords={"costs": ["money", "emissions"]})
        self.backend = _FakeBackend(self)
        self.is_built = True
        self.is_solved = False
        self.results = xr.Dataset()

    def solve(self, force=False, **kwargs):
        if self.backend.objective == "pareto_augmented_epsilon_objective":
            objective_2_values = [point[1] for point in self.feasible_points]
            objective_2_range = max(objective_2_values) - min(objective_2_values)
            epsilon = (
                self.backend.inputs["pareto_epsilon_normalized"].item()
                * objective_2_range
            )
            feasible = [point for point in self.feasible_points if point[1] <= epsilon]
            point = min(feasible, key=lambda candidate: (candidate[0], candidate[1]))
        else:
            weights = self.backend.inputs["objective_cost_weights"].values
            point = min(self.feasible_points, key=lambda p: np.dot(weights, p))
        self.results = xr.Dataset(
            {"cost": (("nodes", "techs", "costs"), [[[point[0], point[1]]]])},
            coords={
                "nodes": ["node"],
                "techs": ["tech"],
                "costs": ["money", "emissions"],
            },
        )
        self.is_solved = True


class _UnsupportedFakeModel(_FakeModel):
    feasible_points = ((1.0, 10.0), (4.0, 8.0), (10.0, 1.0))


@pytest.fixture
def study():
    return ParetoStudy(
        _FakeModel,
        Objective("money", label="System cost", unit="EUR"),
        Objective("emissions", label="CO2", unit="kg"),
    )


def test_weighted_sum_sweep(study):
    result = study.run(WeightedSumSweep(points=3))

    assert result.method == "weighted_sum"
    assert result.points["weight_1"].tolist() == [0.0, 0.5, 1.0]
    assert result.points[["objective_1", "objective_2"]].values.tolist() == [
        [10.0, 1.0],
        [3.0, 5.0],
        [1.0, 10.0],
    ]
    assert len(result.solutions) == 3
    assert result.solution(1)["cost"].sel(costs="money").item() == 3.0


def test_plot_returns_figure(study):
    pytest.importorskip("plotly")
    result = study.run(WeightedSumSweep(points=3))

    figure = result.plot()

    assert len(figure.data[0].x) == 3


def test_plot_groups_coincident_points_and_lists_weights(study):
    pytest.importorskip("plotly")
    result = study.run(WeightedSumSweep(points=5))

    figure = result.plot()

    assert len(figure.data[0].x) == 3
    assert figure.data[0].customdata[:, 2].tolist() == [2, 1, 2]
    assert list(figure.data[0].marker.size) == [13, 10, 13]
    assert "weight_1=0.7500" in figure.data[0].customdata[0, 1]
    assert "weight_2=0.2500" in figure.data[0].customdata[0, 1]
    assert "weight_1=1.0000" in figure.data[0].customdata[0, 1]
    assert "weight_2=0.0000" in figure.data[0].customdata[0, 1]
    assert figure.layout.title.text.endswith("5 points (3 unique)")


def test_augmented_epsilon_constraint(study):
    result = study.run(AugmentedEpsilonConstraint(points=3))

    assert result.method == "augmented_epsilon_constraint"
    assert result.points["epsilon"].tolist() == [1.0, 5.5, 10.0]
    assert result.points[["objective_1", "objective_2"]].values.tolist() == [
        [10.0, 1.0],
        [3.0, 5.0],
        [1.0, 10.0],
    ]
    assert result.points["slack"].tolist() == [0.0, 0.5, 0.0]


def test_augmented_epsilon_finds_unsupported_point():
    unsupported_study = ParetoStudy(
        _UnsupportedFakeModel,
        Objective("money"),
        Objective("emissions"),
    )

    weighted = unsupported_study.run(WeightedSumSweep(points=101))
    epsilon = unsupported_study.run(AugmentedEpsilonConstraint(points=10))

    weighted_points = set(
        map(tuple, weighted.points[["objective_1", "objective_2"]].values)
    )
    epsilon_points = set(
        map(tuple, epsilon.points[["objective_1", "objective_2"]].values)
    )
    assert (4.0, 8.0) not in weighted_points
    assert (4.0, 8.0) in epsilon_points


def test_epsilon_plot_shows_method_parameters(study):
    pytest.importorskip("plotly")
    result = study.run(AugmentedEpsilonConstraint(points=3))

    figure = result.plot()

    assert "epsilon=5.5" in figure.data[0].customdata[1, 1]
    assert "slack=0.5" in figure.data[0].customdata[1, 1]


def test_future_method_placeholder(study):
    with pytest.raises(NotImplementedError):
        study.run(AugmentedTchebycheffSweep())


def test_study_requires_distinct_objectives():
    with pytest.raises(ValueError, match="two different objectives"):
        ParetoStudy(_FakeModel, Objective("money"), Objective("money"))
