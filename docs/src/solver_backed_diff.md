# Solver-Backed Differentiation

## Overview

Some convex optimization solvers cache matrix factorizations (e.g., LU, Cholesky)
during the solve that can be reused for the backward and forward passes. The
`SolverBackedDiff` module lets solver authors expose this capability so that
DiffOpt delegates differentiation to the solver rather than reconstructing the
KKT system from scratch.

When a solver implements the `SolverBackedDiff` interface, DiffOpt detects it
automatically — no user-side configuration is needed:

```julia
using JuMP
import DiffOpt

# MySolver implements the SolverBackedDiff interface
model = DiffOpt.diff_optimizer(MySolver.Optimizer)

x = MOI.add_variables(model, n)
# ... add constraints, objective ...
MOI.optimize!(model)

# Differentiate as usual — DiffOpt uses the solver's native backward pass
MOI.set(model, DiffOpt.ReverseVariablePrimal(), x[1], 1.0)
DiffOpt.reverse_differentiate!(model)
dobj = MOI.get(model, DiffOpt.ReverseObjectiveFunction())
```

!!! note
    If a `ModelConstructor` is explicitly set by the user (e.g., to force
    `QuadraticProgram.Model`), it takes precedence and the native solver
    differentiation is not used.

## Solver Interface

A solver opts in to native differentiation by implementing standard MOI
attribute methods using [`DiffOpt.BackwardDifferentiate`](@ref) and
[`DiffOpt.ForwardDifferentiate`](@ref). All MOI indices passed to these
methods are in the **solver's own index space** (DiffOpt handles the mapping
between wrapper and solver indices internally).

### Required for reverse mode

```julia
MOI.supports(::MySolver, ::DiffOpt.BackwardDifferentiate) = true
```

Declare that the solver supports native backward differentiation.

```julia
MOI.set(::MySolver, ::DiffOpt.BackwardDifferentiate, (dx, dy))
```

Perform the backward pass using the solver's cached factorizations. `dx` is a
`Dict{MOI.VariableIndex, Float64}` mapping variable indices to ``\partial l /
\partial x`` values, and `dy` is a `Dict{MOI.ConstraintIndex, Float64}` mapping
constraint indices to ``\partial l / \partial \lambda`` values.

```julia
MOI.get(::MySolver, ::DiffOpt.ReverseObjectiveFunction) -> MOI.ScalarAffineFunction{Float64}
```

Return the objective sensitivity after the backward pass. The returned
function encodes sensitivities with respect to the linear objective coefficients.

```julia
MOI.get(::MySolver, ::DiffOpt.ReverseConstraintFunction, ci) -> MOI.ScalarAffineFunction{Float64}
```

Return constraint `ci` sensitivity after the backward pass. The affine
terms encode sensitivities with respect to constraint coefficients (the matrix
rows), and the constant encodes the negated sensitivity with respect to the
constraint right-hand side.

### Optional for forward mode

```julia
MOI.supports(::MySolver, ::DiffOpt.ForwardDifferentiate) = true
```

Declare that the solver supports native forward differentiation.

```julia
MOI.set(::MySolver, ::DiffOpt.ForwardDifferentiate, (dobj, dcons))
```

Perform the forward pass. `dobj` is either `nothing` or a
`MOI.ScalarAffineFunction{Float64}` representing the objective perturbation.
`dcons` is a `Dict{MOI.ConstraintIndex, MOI.ScalarAffineFunction{Float64}}`
representing constraint perturbations.

```julia
MOI.get(::MySolver, ::DiffOpt.ForwardVariablePrimal, vi) -> Float64
```

Return the primal tangent for variable `vi` after the forward pass.

```julia
MOI.get(::MySolver, ::DiffOpt.ForwardConstraintDual, ci) -> Float64
```

Return the dual tangent for constraint `ci` after the forward pass.

## Example: Equality-Constrained QP Solver

Consider a simple QP solver for problems of the form:

```math
\begin{align*}
& \min_{x} & \frac{1}{2} x^T Q x + c^T x \\
& \text{s.t.} & A x = b
\end{align*}
```

The KKT system is:

```math
\begin{bmatrix} Q & A^T \\ A & 0 \end{bmatrix}
\begin{bmatrix} x \\ \nu \end{bmatrix} =
\begin{bmatrix} -c \\ b \end{bmatrix}
```

During the solve, we factorize the KKT matrix once. The key insight is that the
**same factorization** can be reused for the backward pass:

```julia
# During solve:
s.kkt_factor = lu(K)
sol = s.kkt_factor \ rhs

# During backward pass — reuse the factorization:
function MOI.set(s::MySolver, ::DiffOpt.BackwardDifferentiate, seeds)
    dx, dy = seeds
    # Build the adjoint RHS from the seeds dx, dy
    rhs = zeros(n + m)
    for (vi, val) in dx
        rhs[vi.value] = val
    end
    for (ci, val) in dy
        rhs[n + ci.value] = val
    end

    # Reuse the cached factorization
    adj = s.kkt_factor \ rhs
    adj_x = adj[1:n]
    adj_ν = adj[n+1:end]

    # Compute parameter sensitivities via implicit function theorem
    s.dc = -adj_x
    s.db = adj_ν
    s.dA = -(s.ν * adj_x' + adj_ν * s.x')
end
```

A complete working example with an equality-constrained QP solver can be found
in the test file `test/solver_backed_diff.jl`.

## How It Works

When `DiffOpt.reverse_differentiate!` or `DiffOpt.forward_differentiate!` is
called on a `diff_optimizer` model:

1. DiffOpt walks through the MOI wrapper layers (CachingOptimizer, bridges,
   POI) to find the innermost solver using `_unwrap_solver`.
2. If the solver supports `DiffOpt.BackwardDifferentiate` and no explicit
   `ModelConstructor` has been set, DiffOpt creates a `SolverBackedDiff.Model`
   that wraps the solver.
3. The `SolverBackedDiff.Model` translates between DiffOpt's index space and
   the solver's index space, then delegates the actual differentiation via
   standard `MOI.set` and `MOI.get` calls on the solver.
4. Results are translated back to the caller's index space automatically.

## API Reference

```@docs
DiffOpt.BackwardDifferentiate
DiffOpt.ForwardDifferentiate
DiffOpt.SolverBackedDiff.Model
DiffOpt.SolverBackedDiff.set_index_mapping!
```
