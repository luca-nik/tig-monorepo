# c007_a002 — Remove Aliases and Scales

## What we did

We implemented a second ZK challenge algorithm (`c007_a002`) that extends the existing baseline (`c007_a001 / remove_aliases`) by also eliminating **scale constraints**.

The optimizer runs in two passes:

1. **Pass 1 — alias removal** (existing baseline): eliminates copy constraints of the form `out = src`
2. **Pass 2 — scale removal** (new): eliminates linear scaling constraints of the form `out = k * src`

Both passes work by **variable substitution followed by row deletion**: instead of keeping the constraint, we inline the definition of `out` everywhere else in the circuit, then drop the row. The result is a smaller circuit that computes the same function.

## The optimization

The circuit generator injects constraints in five categories. Three of them are deliberate optimization traps:

| Op | R1CS encoding | Removable? |
|----|--------------|-----------|
| `Alias` — `out = src` | `(out) * 1 = src` | Yes — `c007_a001` |
| `Scale` — `out = k * src` | `(k·src) * 1 = out` | Yes — `c007_a002` |
| `Pow5` — `out = src^5` | 3 rows | Partially |
| `Add` / `Mul` | 1 row each | No |

### Why scale removal works

A scale row encodes `out = k * src` as a single R1CS constraint. Since `out` is just a scaled copy of `src`, it carries no new information — any downstream constraint that references `out` with coefficient `v` can instead reference `src` with coefficient `v * k`. Once all references are updated, the scale row itself is redundant and can be dropped.

This is the same idea as alias removal, with one extra step: multiply the substituted coefficient by `k` at each substitution site. Curve25519 scalar arithmetic handles the field multiplication correctly.

### Chain resolution

Both passes resolve substitution chains before applying them. For example, if `a = 3*b` and `b = 7*c`, the chain resolves to `a = 21*c` in a single pass, avoiding stale intermediate variables and ensuring the compacted circuit has no dangling references.

### Topological order is preserved

The circuit is generated in topological order (deep dependencies first, outputs last). Both alias and scale removal only **delete** rows — they never reorder or introduce new ones. The forward witness solver therefore succeeds on the output without any reordering step.

## Results (delta = 1, ~1000 constraints)

| Algorithm | Constraints remaining | ε (reduction) |
|-----------|----------------------|--------------|
| Baseline C0 | 1000 | — |
| `c007_a001` remove_aliases | 869 | 0.131 |
| `c007_a002` remove_aliases_and_scales | 738 | **0.262** |

Scale removal adds roughly **13% on top of alias removal**, doubling the epsilon of the baseline.

## What's next

The third injected pattern — `Pow5` (`out = src^5`, 3 constraints) — is not yet handled. Eliminating one intermediate in the `sq → qd → out` chain would give a further ~10% reduction. This requires more care because the two intermediate variables (`sq`, `qd`) may be shared with other parts of the circuit.
