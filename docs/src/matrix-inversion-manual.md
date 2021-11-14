# Differentiating a QP wrt a single variable

Consider the quadratic program

```math
\begin{split}
\begin{array} {ll}
\mbox{minimize} & \frac{1}{2} x^T Q x + q^T x \\
\mbox{subject to} & G x \leq h, x \in \mathcal{R}^2, h \in \mathcal{R} \\
\end{array}
\end{split}
```

where `Q`, `q`, `G` are fixed and `h` is the single parameter.

In this example, we'll try to differentiate the QP wrt `h`, by finding its jacobian by hand (using Eqn (6) of [QPTH article](https://arxiv.org/pdf/1703.00443.pdf)) and compare the results:
- In python, using CVXPYLayers - https://github.com/cvxgrp/cvxpylayers#tensorflow-2
- In Julia, using LinearAlgebra, Dualization.jl and MOI

Assuming 
```
Q = [[4, 1], [1, 2]]
q = [1, 1]
G = [1, 1]
```
and begining with a starting value of `h=-1`

few values just for reference

| variable | optimal value | note |
|----|------|-----|
| x* | [-0.25; -0.75] | Primal optimal | 
| 𝜆∗ | -0.75 | Dual optimal | 


## Finding Jacobian using matrix inversion
Lets formulate Eqn (6) of [QPTH article](https://arxiv.org/pdf/1703.00443.pdf) for our QP. If we assume `h` as the only parameter and `Q`,`q`,`G` as fixed problem data - also note that our QP doesn't involves `Ax=b` constraint - then Eqn (6) reduces to 
```math
\begin{gather}
 \begin{bmatrix} 
     Q & g^T \\
     \lambda^* g & g z^* - h
 \end{bmatrix}
 \begin{bmatrix} 
     dz \\
     d \lambda
 \end{bmatrix}
 =
  \begin{bmatrix}
   0 \\
   \lambda^* dh
   \end{bmatrix}
\end{gather}
```

Now to find the jacobians $$ \frac{\partial z}{\partial h}, \frac{\partial \lambda}{\partial h}$$
we substitute `dh = I = [1]` and plug in values of `Q`,`q`,`G` to get
```math
\begin{gather}
 \begin{bmatrix} 
     4 & 1 & 1 \\
     1 & 2 & 1 \\
     -0.75 & -0.75 & 0
 \end{bmatrix}
 \begin{bmatrix} 
     \frac{\partial z_1}{\partial h} \\
     \frac{\partial z_2}{\partial h} \\
     \frac{\partial \lambda}{\partial h}
 \end{bmatrix}
 =
  \begin{bmatrix}
   0 \\
   0 \\
   -0.75
   \end{bmatrix}
\end{gather}
```

Upon solving using matrix inversion, the jacobian is
```math
\frac{\partial z_1}{\partial h} = 0.25, \frac{\partial z_2}{\partial h} = 0.75, \frac{\partial \lambda}{\partial h} = -1.75 
```


## Finding Jacobian in CVXPYLayers

```python
import cvxpy as cp
import tensorflow as tf
from cvxpylayers.tensorflow import CvxpyLayer

n, m = 2, 1
x = cp.Variable(n)
Q = np.array([[4, 1], [1, 2]])
q = np.array([1, 1])
G = np.array([1, 1])
h = cp.Parameter(m)
constraints = [G@x <= h]
objective = cp.Minimize(0.5*cp.quad_form(x, Q) + q.T @ x)
problem = cp.Problem(objective, constraints)
assert problem.is_dpp()

cvxpylayer = CvxpyLayer(problem, parameters=[h], variables=[x])
h_tf = tf.Variable([-1.0])  # set a starting value

with tf.GradientTape() as tape:
  # solve the problem, setting the values of h to h_tf
  solution, = cvxpylayer(h_tf)

  summed_solution = tf.math.reduce_sum(solution)
  
# note - solution is [-0.25, -0.75]
#        summed_solution is (-0.25) + (-0.75)

# cvxpylayers allows gradient of the summed solution only, with respect to h
gradh = tape.gradient(summed_solution, [h_tf])
```


## Finding Jacobian using MOI, Dualization.jl, LinearAlgebra.jl


```julia
using Random
using MathOptInterface
using Dualization
using OSQP
using LinearAlgebra

const MOI = MathOptInterface
const MOIU = MathOptInterface.Utilities;

n = 2 # variable dimension
m = 1; # no of inequality constraints

Q = [4. 1.;1. 2.]
q = [1.; 1.]
G = [1. 1.;]
h = [-1.;]   # initial values set


# create the optimizer
model = MOI.instantiate(OSQP.Optimizer, with_bridge_type=Float64)
x = MOI.add_variables(model, n);

# define objective
quad_terms = MOI.ScalarQuadraticTerm{Float64}[]
for i in 1:n
    for j in i:n # indexes (i,j), (j,i) will be mirrored. specify only one kind
        push!(quad_terms, MOI.ScalarQuadraticTerm(Q[i,j],x[i],x[j]))
    end
end

objective_function = MOI.ScalarQuadraticFunction(quad_terms, MOI.ScalarAffineTerm.(q, x), 0.0)
MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(), objective_function)
MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)

# add constraint
MOI.add_constraint(
    model,
    MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(G[1,:], x), 0.),
    MOI.LessThan(h[1])
)

# solve
MOI.optimize!(model)

# sanity-check
@assert MOI.get(model, MOI.TerminationStatus()) in [MOI.LOCALLY_SOLVED, MOI.OPTIMAL]

x̄ = MOI.get(model, MOI.VariablePrimal(), x)

# obtaining λ*

joint_object    = dualize(model)
dual_model_like = joint_object.dual_model # this is MOI.ModelLike, not an MOI.AbstractOptimizer; can't call optimizer on it
primal_dual_map = joint_object.primal_dual_map;

# copy the dual model objective, constraints, and variables to an optimizer
dual_model = MOI.instantiate(OSQP.Optimizer, with_bridge_type=Float64)
MOI.copy_to(dual_model, dual_model_like)

# solve dual
MOI.optimize!(dual_model);

map = primal_dual_map.primal_con_dual_var

for con_index in keys(map)
    λ = MOI.get(dual_model, MOI.VariablePrimal(), map[con_index][1])
    println(λ)
end

LHS = [4 1 1; 1 2 1; 1 1 0]  # of Eqn (6)
RHS = [0; 0; 1]  # of Eqn (6)

pp \ qq  # the jacobian
```


    3-element Array{Float64,1}:
      0.25
      0.75
     -1.75
