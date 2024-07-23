# Manual

!!! note
    As of now, this package only works for optimization models that can be written either in convex conic form or convex quadratic form.


## Supported objectives & constraints - scheme 1

For `QPTH`/`OPTNET` style backend, the package supports following `Function-in-Set` constraints: 

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


## Supported objectives & constraints - scheme 2

For `DiffCP`/`CVXPY` style backend, the package supports following `Function-in-Set` constraints: 

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


## Creating a differentiable optimizer

You can create a differentiable optimizer over an existing MOI solver by using the `diff_optimizer` utility. 
```@docs
diff_optimizer
```

## Adding new sets and constraints

The DiffOpt `Optimizer` behaves similarly to other MOI Optimizers
and implements the `MOI.AbstractOptimizer` API.

## Projections on cone sets

DiffOpt requires taking projections and finding projection gradients of vectors while computing the jacobians. For this purpose, we use [MathOptSetDistances.jl](https://github.com/matbesancon/MathOptSetDistances.jl), which is a dedicated package for computing set distances, projections and projection gradients.


## Conic problem formulation

!!! note
    As of now, the package is using `SCS` geometric form for affine expressions in cones.

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
One possible point of confusion in finding Jacobians is the role of the backward pass vector - above eqn (7), *OptNet: Differentiable Optimization as a Layer in Neural Networks*. While differentiating convex programs, it is often the case that we don't want to find the actual derivatives, rather we might be interested in computing the product of Jacobians with a *backward pass vector*, often used in backprop in machine learning/automatic differentiation. This is what happens in scheme 1 of `DiffOpt` backend.

But, for the conic system (scheme 2), we provide perturbations in conic data (`dA`, `db`, `dc`) to compute pertubations (`dx`, `dy`, `dz`) in input variables. Unlike the quadratic case, these perturbations are actual derivatives, not the product with a backward pass vector. This is an important distinction between the two schemes of differential optimization.
