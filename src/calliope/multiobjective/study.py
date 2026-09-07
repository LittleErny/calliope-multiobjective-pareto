"""Public entry point for a bi-objective Pareto study."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from calliope.model import Model
    from calliope.multiobjective.methods import ParetoMethod
    from calliope.multiobjective.result import ParetoResult


@dataclass(frozen=True)
class Objective:
    """A Calliope cost class used as one optimisation objective.

    The first implementation deliberately supports cost classes only. A later
    extension can add arbitrary scalar Calliope math expressions here without
    changing the method or result interfaces.
    """

    cost_class: str
    label: str | None = None
    unit: str | None = None

    @property
    def display_name(self) -> str:
        """Return the human-readable objective name."""
        return self.label or self.cost_class


@dataclass
class ParetoStudy:
    """Coordinate repeated solves of one fresh Calliope model.

    ``model_factory`` must return a new, unsolved model. Keeping model creation
    outside this class makes the workflow independent of YAML/dict loading and
    prevents separate Pareto methods from mutating the same backend.
    """

    model_factory: Callable[[], Model]
    objective_1: Objective
    objective_2: Objective
    build_options: dict[str, Any] = field(default_factory=dict)
    solve_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the small public contract."""
        if self.objective_1.cost_class == self.objective_2.cost_class:
            raise ValueError("A Pareto study requires two different objectives.")

    def run(self, method: ParetoMethod) -> ParetoResult:
        """Run one Pareto-front generation method."""
        return method.run(self)

    def _new_built_model(self) -> Model:
        """Create and, if necessary, build a clean model for one method."""
        model = self.model_factory()
        if model.is_solved:
            raise ValueError("model_factory must return an unsolved Calliope model.")
        if not model.is_built:
            model.build(**self.build_options)
        return model
