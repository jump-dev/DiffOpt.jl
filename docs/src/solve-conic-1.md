# Solving conic with PSD and SOC constraints

Consider an example program 

```math
\begin{split}
\begin{array} {llcc}
\mbox{minimize}  &
\left\langle
\left[
\begin{array} {ccc}
        2 & 1 & 0  \\
        1 & 2 & 1  \\
        0 & 1 & 2
   \end{array}
   \right],
        X \right\rangle
        + x_0   &  &   \\
          \mbox{subject to}  &
          \left\langle
          \left[
          \begin{array} {ccc}
          1 & 0 & 0  \\
          0 & 1 & 0  \\
          0 & 0 & 1
          \end{array}
          \right],
          X \right\rangle
          + x_0 & =  & 1,   \\
            &
            \left\langle
            \left[
            \begin{array}{ccc}
            1 & 1 & 1  \\
            1 & 1 & 1  \\
            1 & 1 & 1
            \end{array}
            \right],
            X \right\rangle + x_1 + x_2
            & = & 1/2,  \\
            & (x_0, x_1, x_2) \in \mathbb{Q}^3 \text{ or } x_0 \geq \sqrt{{x_1}^2 + {x_2}^2} \\
            & X \succeq 0, X \in \mathbb{S}^3_{+}
\end{array}
\end{split}
```
where
```math
\mathbb{S}^n_{+} =
\left\lbrace
X \in \mathbb{S}^n: z^T X z \geq 0, \quad \forall z \in \mathbb{R}^n
\right\rbrace,
```

> Refered from Mosek examples: https://docs.mosek.com/9.2/toolbox/tutorial-sdo-shared.html#example-sdo1


## Equivalent DiffCP program to differentiate
```python
import numpy as np
import cvxpy as cp
from scipy import sparse
import diffcp

A = sparse.csc_matrix((11+1,7+1), dtype=np.float64)
A[2 , 1]  =  1.0
A[3 , 1]  =  -1.0
A[9 , 1]  =  -0.45
A[10, 1]  =  0.45
A[11, 1]  =  -0.45
A[2 , 2]  =  1.0
A[4 , 2]  =  -1.0
A[9 , 2]  =  -0.8
A[10, 2]  =  0.318198
A[11, 2]  =  -0.1
A[2 , 3]  =  1.0
A[5 , 3]  =  -1.0
A[9 , 3]  =  -0.9
A[2 , 4]  =  1.0
A[6 , 4]  =  -1.0
A[9 , 4]  =  -0.225
A[2 , 5]  =  1.0
A[7 , 5]  =  -1.0
A[9 , 5]  =  -0.1125
A[10, 5]  =  0.1125
A[11, 5]  =  -0.1125
A[2 , 6]  =  1.0
A[8 , 6]  =  -1.0
A[11, 6]  =  -0.225
A[9 , 7]  =  1.0
A[11, 7]  =  1.0

A = A[1:, 1:]

# equivalent to: https://github.com/jump-dev/MathOptInterface.jl/blob/master/src/Test/contconic.jl#L2575

cone_dict = {
    diffcp.POS: 7,
    diffcp.PSD: [2],
    diffcp.ZERO: 1
}

b = np.array([0.0, 10.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.0, 0.0, 0.0])
c = np.array([-0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -1.0])

x, y, s, D, DT = diffcp.solve_and_derivative(A, b, c, cone_dict)
print(x) # MOI.VariablePrimal
print(s) # MOI.ConstraintPrimal
print(y) # MOI.ConstraintDual


dx, dy, ds = D(sparse.csc_matrix(np.ones((11,7))), np.ones(11), np.ones(7))
print(dx)
print(ds)
print(dy)
```

## Equivalent DiffOpt program
```julia
using SCS
using DiffOpt
using MathOptInterface

const MOI = MathOptInterface;


model = diff_optimizer(SCS.Optimizer)
MOI.set(model, MathOptInterface.Silent(), true)

δ = √(1 + (3*√2+2)*√(-116*√2+166) / 14) / 2
ε = √((1 - 2*(√2-1)*δ^2) / (2-√2))
y2 = 1 - ε*δ
y1 = 1 - √2*y2
obj = y1 + y2/2
k = -2*δ/ε
x2 = ((3-2obj)*(2+k^2)-4) / (4*(2+k^2)-4*√2)
α = √(3-2obj-4x2)/2
β = k*α

X = MOI.add_variables(model, 6)
x = MOI.add_variables(model, 3)

vov = MOI.VectorOfVariables(X)

cX = MOI.add_constraint(
    model, 
    MOI.VectorAffineFunction{Float64}(vov), MOI.PositiveSemidefiniteConeTriangle(3)
)

cx = MOI.add_constraint(
    model, 
    MOI.VectorAffineFunction{Float64}(MOI.VectorOfVariables(x)), MOI.SecondOrderCone(3)
)

c1 = MOI.add_constraint(
    model, 
    MOI.VectorAffineFunction(
        MOI.VectorAffineTerm.(1:1,
            MOI.ScalarAffineTerm.([1., 1., 1., 1.], [X[1], X[3], X[end], x[1]])), 
        [-1.0]
    ), 
    MOI.Zeros(1)
)

c2 = MOI.add_constraint(
    model, 
    MOI.VectorAffineFunction(
        MOI.VectorAffineTerm.(1:1,
            MOI.ScalarAffineTerm.([1., 2, 1, 2, 2, 1, 1, 1], [X; x[2]; x[3]])), 
        [-0.5]
    ), 
    MOI.Zeros(1)
)

objXidx = [1:3; 5:6]
objXcoefs = 2*ones(5)
MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
    MOI.ScalarAffineFunction(
        MOI.ScalarAffineTerm.([objXcoefs; 1.0], [X[objXidx]; x[1]]), 0.0))
MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)

sol = MOI.optimize!(model)

# fetch solution
x_sol = MOI.get(model, MOI.VariablePrimal(), vcat(X, x))
s_sol = MOI.get(model, MOI.ConstraintPrimal(), [cX, cx, c1, c2])
y_sol = MOI.get(model, MOI.ConstraintDual(), [cX, cx, c1, c2])

println("x -> ", round.(x_sol; digits=3))
println("s -> ", round.(s_sol; digits=3))
println("y -> ", round.(y_sol; digits=3))

# perturbations in all the parameters
fx = MOI.SingleVariable.(x)
MOI.set(model,
    DiffOpt.ForwardInConstraint(), c1, MOIU.vectorize(ones(1, 9) * fx + ones(1)))
MOI.set(model,
    DiffOpt.ForwardInConstraint(), c2, MOIU.vectorize(ones(6, 9) * fx + ones(6)))
MOI.set(model,
    DiffOpt.ForwardInConstraint(), c3, MOIU.vectorize(ones(3, 9) * fx + ones(3)))
MOI.set(model,
    DiffOpt.ForwardInConstraint(), c4, MOIU.vectorize(ones(1, 9) * fx + ones(1)))

# differentiate and get the gradients
DiffOpt.forward(model)

dx = MOI.get.(model,
    DiffOpt.ForwardOut{MOI.VariablePrimal}(), vcat(X, x))

println("dx -> ", round.(dx; digits=3))
# println("ds -> ", round.(ds; digits=3))
# println("dy -> ", round.(dy; digits=3))
```
