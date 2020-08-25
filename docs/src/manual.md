# Manual

!!! note
    As of now, this package only works for optimization models that can be written either in convex conic form or convex quadratic form.

!!! note
    As of now, the package is using `SCS` geometric form for affine expressions in cones.


## Supported objectives & constraints - scheme 1

For `QPTH`/`OPTNET` backend (using `backward!` method), the package supports following `Function-in-Set` constraints: 

|  MOI Function | MOI Set |
|:-------|:---------------|
|    `SingleVariable`    |    `GreaterThan`    |
|    `SingleVariable`    |    `LessThan`    |
|    `SingleVariable`    |    `EqualTo`    |
|    `ScalarAffineFunction`    |    `GreaterThan`    |
|    `ScalarAffineFunction`    |    `LessThan`    |
|    `ScalarAffineFunction`    |    `EqualTo`    |

and the following objective types: 

|  MOI Function |
|:-------:|
|   `SingleVariable`   |
|   `ScalarAffineFunction`   |
| `ScalarQuadraticFunction`  | 


## Supported objectives & constraints - scheme 2

For `DiffCP`/`CVXPY` backend (using `backward_conic!` method), the package supports following `Function-in-Set` constraints: 

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
|   `SingleVariable`   |
|   `ScalarAffineFunction`   |


## Creating a differentiable optimizer

You can create a differentiable optimizer over an existing MOI solver by using the `diff_optimizer` utility. 
```@docs
diff_optimizer
```

## Adding new sets and constraints

Usage interface DiffOpt models is same as other MOI Optimizers. So the same `add_variable`, `add_constraint` utilities can be used.


## Projections on cone sets

DiffOpt requires taking projections and finding projection gradients of vectors while computing the jacobians. For this purpose, we use [MathOptSetDistances.jl](https://github.com/matbesancon/MathOptSetDistances.jl), which is a dedicated package for computing set distances, projections and projection gradients.
