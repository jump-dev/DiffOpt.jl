# Manual

## Supported objectives & constraints - scheme 1

For `QuadraticProgram` backend, the package supports following `Function-in-Set` constraints: 

|  MOI Function | MOI Set |
|:-------|:---------------|
|    `VariableIndex`    |    `GreaterThan`    |
|    `VariableIndex`    |    `LessThan`    |
|    `VariableIndex`    |    `EqualTo`    |
|    `ScalarAffineFunction`    |    `GreaterThan`    |
|    `ScalarAffineFunction`    |    `LessThan`    |
|    `ScalarAffineFunction`    |    `EqualTo`    |

and the following objective types: 

|  MOI Function |
|:-------:|
|   `VariableIndex`   |
|   `ScalarAffineFunction`   |
| `ScalarQuadraticFunction`  | 


## Supported objectives & constraints - `ConicProgram` backend

For the `ConicProgram` backend, the package supports following `Function-in-Set` constraints: 

|  MOI Function | MOI Set |
|:-------|:---------------|
|    `VectorOfVariables`    |    `Nonnegatives`    |
|    `VectorOfVariables`    |    `Nonpositives`    |
|    `VectorOfVariables`    |    `Zeros`    |
|    `VectorOfVariables`    |    `SecondOrderCone`    |
|    `VectorOfVariables`    |    `PositiveSemidefiniteConeTriangle`    |
|    `VectorAffineFunction`    |    `Nonnegatives`    |
|    `VectorAffineFunction`    |    `Nonpositives`    |
|    `VectorAffineFunction`    |    `Zeros`    |
|    `VectorAffineFunction`    |    `SecondOrderCone`    |
|    `VectorAffineFunction`    |    `PositiveSemidefiniteConeTriangle`    |

and the following objective types: 

|  MOI Function |
|:-------:|
|   `VariableIndex`   |
|   `ScalarAffineFunction`   |

Other conic sets such as `RotatedSecondOrderCone` and `PositiveSemidefiniteConeSquare` are supported through bridges.

## Supported objectives & constraints - `NonlinearProgram` backend

For the `NonlinearProgram` backend, the package supports following `Function-in-Set` constraints:

|  MOI Function | MOI Set |
|:-------|:---------------|
|    `VariableIndex`    |    `GreaterThan`    |
|    `VariableIndex`    |    `LessThan`    |
|    `VariableIndex`    |    `EqualTo`    |
|    `ScalarAffineFunction`    |    `GreaterThan`    |
|    `ScalarAffineFunction`    |    `LessThan`    |
|    `ScalarAffineFunction`    |    `EqualTo`    |
|    `ScalarQuadraticFunction`    |    `GreaterThan`    |
|    `ScalarQuadraticFunction`    |    `LessThan`    |
|    `ScalarQuadraticFunction`    |    `EqualTo`    |
|    `ScalarNonlinearFunction`    |    `GreaterThan`    |
|    `ScalarNonlinearFunction`    |    `LessThan`    |
|    `ScalarNonlinearFunction`    |    `EqualTo`    |

and the following objective types: 

|  MOI Function |
|:-------:|
|   `VariableIndex`   |
|   `ScalarAffineFunction`   |
| `ScalarQuadraticFunction`  | 
| `ScalarNonlinearFunction`  |

### `VectorNonlinearOracle` bridge (`NonlinearProgram`)

`DiffOpt.NonLinearProgram` supports `MOI.VectorOfVariables` in
`MOI.VectorNonlinearOracle` through an internal bridge that rewrites the vector
oracle into scalar nonlinear constraints.

At a high level, for a vector oracle $f:\mathbb{R}^n \to \mathbb{R}^m$ with
bounds $l \le f(x) \le u$:

- one scalar nonlinear operator is registered per output row $f_i(x)$
- each row is converted into one or two scalar constraints based on bounds:
  - finite and equal $l[i] == u[i]$: `EqualTo(l[i])`
  - finite lower only: `GreaterThan(l[i])`
  - finite upper only: `LessThan(u[i])`
  - both finite and different: one `GreaterThan` and one `LessThan`
  - infinite bounds are skipped

Callback signature requirements follow MOI:

- univariate (`input_dimension == 1`):
  - `f(x)::Real`
  - `∇f(x)::Real`
  - `∇²f(x)::Real`
- multivariate (`input_dimension > 1`):
  - `f(x...)::Real`
  - `∇f(g, x...)` fills `g`
  - `∇²f(H, x...)` fills the lower-triangular part of `H`

Warm-start mapping for bridged constraints:

- `ConstraintPrimalStart` expects an input vector `x` (not `f(x)`), evaluates
  the oracle at `x`, and writes starts for each generated scalar constraint.
- `ConstraintDualStart` accepts either:
  - length = output dimension `m`: treated as direct row duals
  - length = input dimension `n`: interpreted as `J' * λ` and converted to row
    duals `λ` via a least-squares solve

Current limitation:

- for dual starts on rows with both finite bounds (`l[i] < u[i]`), only the
  lower-bound side is propagated explicitly (the upper-side split is not
  modeled in this bridge-level helper).

Minimal JuMP example:

```julia
using JuMP, DiffOpt, Ipopt
import MathOptInterface as MOI

model = DiffOpt.nonlinear_diff_model(Ipopt.Optimizer)
@variable(model, x[1:2])
@objective(model, Min, x[1]^2 + x[2]^2)

function eval_f(ret, z)
    ret[1] = z[1]^2 + z[2]^2
    return
end
function eval_jacobian(ret, z)
    ret[1] = 2z[1]
    ret[2] = 2z[2]
    return
end
function eval_hessian_lagrangian(ret, z, μ)
    ret[1] = 2μ[1]  # (1,1)
    ret[2] = 2μ[1]  # (2,2)
    return
end

set = MOI.VectorNonlinearOracle(;
    dimension = 2,
    l = [-Inf],
    u = [1.0],
    eval_f,
    jacobian_structure = [(1, 1), (1, 2)],
    eval_jacobian,
    hessian_lagrangian_structure = [(1, 1), (2, 2)],
    eval_hessian_lagrangian,
)

@constraint(model, [x[1], x[2]] in set)
optimize!(model)
```

## Creating a differentiable MOI optimizer

You can create a differentiable optimizer over an existing MOI solver by using the `diff_optimizer` utility. 
```@docs
diff_optimizer
```

## Projections on cone sets

DiffOpt requires taking projections and finding projection gradients of vectors while computing the jacobians. For this purpose, we use [MathOptSetDistances.jl](https://github.com/matbesancon/MathOptSetDistances.jl), which is a dedicated package for computing set distances, projections and projection gradients.


## Conic problem formulation

!!! note
    As of now, when defining a conic or convex quadratic problem, the package is using `SCS` geometric form for affine expressions in cones.

Consider a convex conic optimization problem in its primal (P) and dual (D) forms:
```math
\begin{split}
\begin{array} {llcc}
\textbf{Primal Problem} & & \textbf{Dual Problem} & \\
\mbox{minimize} & c^T x  \quad \quad & \mbox{minimize} & b^T y  \\
\mbox{subject to} & A x + s = b  \quad \quad & \mbox{subject to} & A^T y + c = 0 \\
& s \in \mathcal{K} &  & y \in \mathcal{K}^*
\end{array}
\end{split}
```

where
- ``x \in R^n`` is the primal variable, ``y \in R^m`` is the dual variable, and ``s \in R^m`` is the primal slack
variable
- ``\mathcal{K} \subseteq R^m`` is a closed convex cone and ``\mathcal{K}^* \subseteq R^m`` is the corresponding dual cone
variable
- ``A \in R^{m \times n}``, ``b \in R^m``, ``c \in R^n`` are problem data

In the light of above, DiffOpt differentiates program variables ``x``, ``s``, ``y``  w.r.t pertubations/sensivities in problem data i.e. ``dA``, ``db``, ``dc``. This is achieved via *implicit differentiation* and *matrix differential calculus*.

> Note that the primal (P) and dual (D) are self-duals of each other. Similarly, for the constraints we support, ``\mathcal{K}`` is same in format as ``\mathcal{K}^*``.


### Reference articles

- [_Differentiating Through a Cone Program_](https://arxiv.org/abs/1904.09043) - Akshay Agrawal, Shane Barratt, Stephen Boyd, Enzo Busseti, Walaa M. Moursi, 2019
- A fast and differentiable QP solver for PyTorch. Crafted by Brandon Amos and J. Zico Kolter.
- OptNet: Differentiable Optimization as a Layer in Neural Networks

### Backward Pass vector
One possible point of confusion in finding Jacobians is the role of the backward pass vector - above eqn (7), *OptNet: Differentiable Optimization as a Layer in Neural Networks*. While differentiating convex programs, it is often the case that we don't want to find the actual derivatives, rather we might be interested in computing the product of Jacobians with a *backward pass vector*, often used in backpropagation in machine learning/automatic differentiation. This is what happens in `DiffOpt` backends.
