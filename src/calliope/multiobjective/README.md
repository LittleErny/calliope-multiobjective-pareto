# Bi-objective Pareto sweeps for Calliope

This module is a small solver-backed MVP for generating Pareto-front samples from
two Calliope cost classes. It keeps the standard Calliope model and solver workflow:
each multi-objective method scalarises the two objectives into one LP/MILP objective,
then solves the same built backend repeatedly with different scalarisation parameters.

The implementation is intentionally limited to:

- exactly two minimisation objectives;
- objectives represented by Calliope `costs` classes;
- uniform weight or epsilon grids;
- sequential solves in one process.

## Quick start

```python
from pathlib import Path

import calliope
from calliope.multiobjective import (
    AugmentedEpsilonConstraint,
    AugmentedTchebycheffSweep,
    Objective,
    ParetoStudy,
    WeightedSumSweep,
)

MODEL_PATH = Path("path/to/model.yaml")


def model_factory():
    # The factory must return a fresh, unsolved model on every call.
    return calliope.read_yaml(MODEL_PATH, scenario="minimize_emissions_costs")


study = ParetoStudy(
    model_factory=model_factory,
    objective_1=Objective("monetary", label="Total system cost", unit="EUR"),
    objective_2=Objective("emissions", label="CO2 emissions", unit="kg CO2"),
    solve_options={
        "solver": "cbc",
        "solver_options": {
            "primalTolerance": 1e-10,
            "dualTolerance": 1e-10,
        },
    },
)

weighted = study.run(WeightedSumSweep(points=20))
epsilon = study.run(AugmentedEpsilonConstraint(points=20, augmentation=1e-6))
tchebycheff = study.run(
    AugmentedTchebycheffSweep(points=20, augmentation=1e-6)
)

tchebycheff.points       # one row per requested point
tchebycheff.plot()       # interactive Plotly figure
tchebycheff.solution(10) # complete Calliope result for point_id=10
```

The model must contain both selected cost classes and the mutable
`objective_cost_weights` input. The example supplies this input with an
`override_dict` so both coordinates exist before the backend is built.

For a complete runnable comparison, see
[`examples/three_method_pareto_front.ipynb`](examples/three_method_pareto_front.ipynb).
It uses the included copy of Calliope's documented
[`national_scale`](examples/national_scale) model and overlays all three fronts in
one figure.

## Architecture

- `Objective` identifies a Calliope cost class and its display metadata.
- `ParetoStudy` owns the model factory, the two objectives, and build/solve options.
- A `ParetoMethod` builds a fresh model, calculates anchor solutions, and performs
  its parameter sweep.
- `ParetoResult` stores the point table and a deep copy of the complete Calliope
  result for every point.

Every method first solves both objectives independently. These anchor solutions
define the ideal values and payoff ranges used for normalisation. A small secondary
weight refines each endpoint, avoiding a weakly efficient endpoint when the primary
objective has alternative optima.

## Methods

Let the two minimisation objectives be $f_1(x)$ and $f_2(x)$. Their normalised
deviations from the ideal point are

$$
\hat f_i(x) = \frac{f_i(x) - f_i^{ideal}}{R_i},
$$

where $R_i$ is the objective's payoff range calculated from the anchor solutions.

### Normalised weighted-sum sweep

`WeightedSumSweep` evaluates a uniform grid with $w_1 \in [0, 1]$ and
$w_2 = 1-w_1$. Each interior point solves

$$
\min_{x \in X} w_1 \hat f_1(x) + w_2 \hat f_2(x).
$$

The two refined anchor solutions are reused as the endpoints. Different weights may
return the same solution, and a weighted sum cannot recover unsupported points on a
non-convex Pareto front.

Research background:
[`§2.2 Native Implementation of Weighted Sum` (PDF p. 3)](references/multiobjective_optimisation_expose.pdf#page=3).

### Augmented epsilon-constraint sweep

`AugmentedEpsilonConstraint` minimises objective 1 while bounding objective 2. For
each value on an automatically generated epsilon grid, it solves the equivalent
normalised formulation

$$
\begin{aligned}
\min_{x,s}\quad & f_1(x) - \rho R_1 s \\
\text{subject to}\quad & \frac{f_2(x)}{R_2} + s = \frac{\epsilon}{R_2}, \\
& s \ge 0, \quad x \in X.
\end{aligned}
$$

The small augmentation term rewards unused slack and prevents weakly efficient
solutions. Swap `objective_1` and `objective_2` in `ParetoStudy` to reverse which
objective is minimised and which is constrained.

Research background:
[`§3.1 ε-Constraint Method` (PDF p. 4)](references/multiobjective_optimisation_expose.pdf#page=4).

### Augmented weighted Tchebycheff sweep

`AugmentedTchebycheffSweep` introduces a non-negative auxiliary variable $z$. For
each weight pair it solves

$$
\begin{aligned}
\min_{x,z}\quad & z + \rho\left(\hat f_1(x)+\hat f_2(x)\right) \\
\text{subject to}\quad & w_1\hat f_1(x) \le z, \\
& w_2\hat f_2(x) \le z, \\
& z \ge 0, \quad x \in X.
\end{aligned}
$$

Minimising $z$ is the linear epigraph form of minimising the worst weighted
normalised deviation from the ideal point. The L1 augmentation distinguishes
solutions with the same maximum deviation. Unlike a weighted sum, this formulation
can recover unsupported Pareto points.

Research background:
[`§3.2 Chebyshev / Min–Max Method` (PDF p. 5)](references/multiobjective_optimisation_expose.pdf#page=5).

## Results and plotting

`ParetoResult.points` contains scalarisation parameters and the two objective values.
`ParetoResult.solutions` contains the corresponding full `xarray.Dataset` objects.
`result.solution(point_id)` retrieves one of those datasets.

`result.plot()` groups exactly coincident objective vectors, shows their point IDs and
method parameters on hover, and reports both requested and unique point counts in the
title. Plotly is imported lazily and is only required for plotting.

## Numerical settings

Normalisation makes the custom scalar objectives approximately order one. Both
primal feasibility and dual optimality tolerances therefore matter. The included CBC
example uses:

```python
"solver_options": {
    "primalTolerance": 1e-10,
    "dualTolerance": 1e-10,
}
```

The augmentation coefficient must remain small enough not to replace the primary
scalarisation criterion, but large enough to remain visible at the selected solver
tolerances. The example uses `1e-6`.

## Included research note

The formulation and motivation are documented in
[`Multi-Objective Optimisation Methods for Economic Dispatch Energy Models in Calliope`](references/multiobjective_optimisation_expose.pdf).
The most relevant sections are the three method-specific links above.
