# Plotting recipe

After solving in `pareto` mode, results are stored along a `pareto` dimension (the epsilon values).

Two KPIs are provided by the Pareto math:

- `pareto_primary_cost_total`: realised total primary cost (objective cost class).
- `pareto_secondary_cost_total`: realised total secondary cost (constrained cost class).

## Pandas dataframe

```python
df = (
    model.results[["pareto_primary_cost_total", "pareto_secondary_cost_total"]]
    .to_dataframe()
    .reset_index()
    .sort_values("pareto")
)

# Convenience aliases
df = df.rename(
    columns={
        "pareto": "epsilon",
        "pareto_primary_cost_total": "primary_total",
        "pareto_secondary_cost_total": "secondary_total",
    }
)

print(df.head())
```

## Plot the Pareto front (matplotlib)

```python
import matplotlib.pyplot as plt

plt.figure()
plt.plot(df["secondary_total"], df["primary_total"], marker="o")
plt.xlabel("Secondary total")
plt.ylabel("Primary total")
plt.title("Pareto front (epsilon-constraint)")
plt.show()
```
## Notes

- Epsilon values are stored in the `pareto` coordinate (and appear as `epsilon` in the dataframe above).
- With `stop_on_infeasible: true` (default), the run stops when a non-optimal/infeasible epsilon is encountered (same behaviour as `spores`).