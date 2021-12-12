# # Differentiating a QP wrt a single variable

#md # [![](https://img.shields.io/badge/show-github-579ACA.svg)](@__REPO_ROOT_URL__/docs/src/examples/matrix-inversion-manual.jl)
#md # [![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](@__NBVIEWER_ROOT_URL__/generated/matrix-inversion-manual.ipynb)


# Consider the quadratic program

# ```math
# \begin{split}
# \begin{array} {ll}
# \mbox{minimize} & \frac{1}{2} x^T Q x + q^T x \\
# \mbox{subject to} & G x \leq h, x \in \mathcal{R}^2 \\
# \end{array}
# \end{split}
# ```

# where `Q`, `q`, `G` are fixed and `h` is the single parameter.

# In this example, we'll try to differentiate the QP wrt `h`, by finding its
# jacobian by hand (using Eqn (6) of [QPTH article](https://arxiv.org/pdf/1703.00443.pdf))
# and compare the results:
# - Manual compuation
# - Using JuMP and DiffOpt

# Assuming 
# ```
# Q = [[4, 1], [1, 2]]
# q = [1, 1]
# G = [1, 1]
# ```
# and begining with a starting value of `h=-1`

# few values just for reference

# | variable | optimal value | note |
# |----|------|-----|
# | x* | [-0.25; -0.75] | Primal optimal | 
# | 𝜆∗ | -0.75 | Dual optimal | 


# ## Finding Jacobian using matrix inversion
# Lets formulate Eqn (6) of [QPTH article](https://arxiv.org/pdf/1703.00443.pdf) for our QP. If we assume `h` as the only parameter and `Q`,`q`,`G` as fixed problem data - also note that our QP doesn't involves `Ax=b` constraint - then Eqn (6) reduces to 
# ```math
# \begin{gather}
#  \begin{bmatrix} 
#      Q & G^T \\
#      \lambda^* G & G x^* - h
#  \end{bmatrix}
#  \begin{bmatrix} 
#      dx \\
#      d \lambda
#  \end{bmatrix}
#  =
#   \begin{bmatrix}
#    0 \\
#    \lambda^* dh
#    \end{bmatrix}
# \end{gather}
# ```

# Now to find the jacobians $$ \frac{\partial x}{\partial h}, \frac{\partial \lambda}{\partial h}$$
# we substitute `dh = I = [1]` and plug in values of `Q`,`q`,`G` to get
# ```math
# \begin{gather}
#  \begin{bmatrix} 
#      4 & 1 & 1 \\
#      1 & 2 & 1 \\
#      -0.75 & -0.75 & 0
#  \end{bmatrix}
#  \begin{bmatrix} 
#      \frac{\partial x_1}{\partial h} \\
#      \frac{\partial x_2}{\partial h} \\
#      \frac{\partial \lambda}{\partial h}
#  \end{bmatrix}
#  =
#   \begin{bmatrix}
#    0 \\
#    0 \\
#    -0.75
#    \end{bmatrix}
# \end{gather}
# ```

# Upon solving using matrix inversion, the jacobian is
# ```math
# \frac{\partial x_1}{\partial h} = 0.25, \frac{\partial x_2}{\partial h} = 0.75, \frac{\partial \lambda}{\partial h} = -1.75 
# ```

# ## Finding Jacobian using JuMP and DiffOpt

using JuMP
import DiffOpt
import Ipopt

n = 2 # variable dimension
m = 1; # no of inequality constraints

Q = [4. 1.;1. 2.]
q = [1.; 1.]
G = [1. 1.;]
h = [-1.;]   # initial values set

# Initialize empty model

model = Model(() -> DiffOpt.diff_optimizer(Ipopt.Optimizer))
set_silent(model)

# Add the variables

@variable(model, x[1:2])

# Add the constraints.

@constraint(
    model,
    cons[j in 1:1],
    sum(G[j, i] * x[i] for i in 1:2)  <= h[j]
);

@objective(
    model,
    Min,
    1/2 * sum(Q[j, i] * x[i] *x[j] for i in 1:2, j in 1:2) +
    sum(q[i] * x[i] for i in 1:2)
)

# Solve problem

optimize!(model)

# primal solution

value.(x)

# dual solution

dual.(cons)

# set sentivitity

MOI.set(
    model,
    DiffOpt.ForwardInConstraint(),
    cons[1],
    0.0 * index(x[1]) - 1.0,  # the tangent of the ConstraintFunction, i.e., ∂(Gx - h)/∂h = -1
)

# Note that `0.0 * index(x[1])` is used to make its type `typeof(0.0 * index(x[1]) - 1.0) <: MOI.AbstractScalarFunction`.
# To indicate different direction to get directional derivative, users should replace `0.0 * index(x[1]) - 1.0` as the form of `dG*x - dh`, where `dG` and `dh` correspond to the elements of direction vectors along `G` and `h` axes, respectively.

# Compute derivatives

DiffOpt.forward(model)

# Query derivative

dx = MOI.get.(
    model,
    DiffOpt.ForwardOutVariablePrimal(),
    x,
)  # ∂x/∂h

using Test                                  #src
@test dx ≈ [0.25 ,0.75] atol=1e-4 rtol=1e-4 #src
