"""Pareto study results and a minimal plotting helper."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Any

import pandas as pd
import xarray as xr

from calliope.multiobjective.study import Objective


@dataclass(frozen=True)
class ParetoResult:
    """A compact point table alongside the full result of every solve."""

    method: str
    objective_1: Objective
    objective_2: Objective
    points: pd.DataFrame
    solutions: tuple[xr.Dataset, ...]

    def solution(self, point_id: int) -> xr.Dataset:
        """Return the full Calliope result associated with a plotted point."""
        matches = [
            position
            for position, candidate in enumerate(self.points["point_id"])
            if candidate == point_id
        ]
        if not matches:
            raise KeyError(f"Unknown Pareto point_id: {point_id}")
        position = matches[0]
        return self.solutions[position]

    def to_dataset(self) -> xr.Dataset:
        """Combine complete solutions and sweep metadata as Calliope results."""
        point_ids = pd.Index(self.points["point_id"].to_numpy(), name="pareto_point")
        results = xr.concat(self.solutions, dim=point_ids, combine_attrs="drop")
        for column in self.points.columns:
            if column == "point_id":
                continue
            results[f"pareto_{column}"] = xr.DataArray(
                self.points[column].to_numpy(),
                coords={"pareto_point": point_ids},
                dims="pareto_point",
            )
        conditions = [
            solution.attrs.get("termination_condition", "")
            for solution in self.solutions
        ]
        if all(condition in {"optimal", "feasible"} for condition in conditions):
            results.attrs["termination_condition"] = (
                "feasible" if "feasible" in conditions else "optimal"
            )
        results.attrs.update(
            pareto_method=self.method,
            pareto_objective_1=self.objective_1.cost_class,
            pareto_objective_2=self.objective_2.cost_class,
        )
        return results

    @classmethod
    def from_dataset(cls, results: xr.Dataset) -> ParetoResult:
        """Restore the plotting interface from native mode results."""
        required = {"pareto_method", "pareto_objective_1", "pareto_objective_2"}
        if "pareto_point" not in results.dims or not required <= results.attrs.keys():
            raise ValueError("Dataset does not contain Calliope Pareto mode results.")
        point_columns = {
            name.removeprefix("pareto_"): results[name].values
            for name in results.data_vars
            if name.startswith("pareto_") and results[name].dims == ("pareto_point",)
        }
        points = pd.DataFrame(
            {"point_id": results.pareto_point.values, **point_columns}
        )
        metadata_names = [f"pareto_{name}" for name in point_columns]
        full_results = results.drop_vars(metadata_names)
        solutions = tuple(
            full_results.sel(pareto_point=point_id, drop=True)
            for point_id in results.pareto_point.values
        )
        return cls(
            method=results.attrs["pareto_method"],
            objective_1=Objective(results.attrs["pareto_objective_1"]),
            objective_2=Objective(results.attrs["pareto_objective_2"]),
            points=points,
            solutions=solutions,
        )

    def plot(self, **plot_kwargs: Any) -> Any:
        """Plot the generated points with Plotly and return the figure.

        Plotly is imported lazily so Pareto generation itself adds no
        plotting dependency to Calliope.
        """
        try:
            import plotly.express as px
        except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
            raise ModuleNotFoundError(
                "ParetoResult.plot() requires the optional `plotly` package."
            ) from exc

        points = self.points.copy()
        parameter_columns = [
            column
            for column in points.columns
            if column not in {"point_id", "objective_1", "objective_2"}
        ]
        points["method_parameters"] = points.apply(
            lambda row: "<br>".join(
                f"{column}={self._format_parameter(column, row[column])}"
                for column in parameter_columns
            ),
            axis=1,
        )
        plot_points = (
            points.groupby(["objective_1", "objective_2"], as_index=False)
            .agg(
                point_ids=("point_id", lambda values: ", ".join(map(str, values))),
                parameters=("method_parameters", "<br>".join),
                coincident_points=("point_id", "size"),
            )
            .sort_values("objective_1")
        )
        defaults = {
            "x": "objective_1",
            "y": "objective_2",
            "markers": True,
            "custom_data": ["point_ids", "parameters", "coincident_points"],
            "labels": {
                "objective_1": self._axis_label(self.objective_1),
                "objective_2": self._axis_label(self.objective_2),
            },
            "title": (
                f"Pareto front: {self.method} — {len(points)} points "
                f"({len(plot_points)} unique)"
            ),
        }
        defaults.update(plot_kwargs)
        figure = px.line(plot_points, **defaults)
        figure.update_traces(
            marker={
                "size": [
                    10 + 3 * min(count - 1, 4)
                    for count in plot_points["coincident_points"]
                ]
            },
            hovertemplate=(
                f"{self._axis_label(self.objective_1)}: %{{x:,.4g}}<br>"
                f"{self._axis_label(self.objective_2)}: %{{y:,.4g}}<br>"
                "Point IDs: %{customdata[0]}<br>"
                "Parameters:<br>%{customdata[1]}<br>"
                "Coincident points: %{customdata[2]}"
                "<extra></extra>"
            ),
        )
        return figure

    @staticmethod
    def _axis_label(objective: Objective) -> str:
        label = objective.display_name
        return f"{label} [{objective.unit}]" if objective.unit else label

    @staticmethod
    def _format_parameter(name: str, value: Any) -> str:
        if name.startswith("weight_"):
            return f"{float(value):.4f}"
        if isinstance(value, Real):
            return f"{value:.6g}"
        return str(value)
