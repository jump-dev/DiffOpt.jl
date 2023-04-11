# Copyright (c) 2020: Akshay Sharma and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

using Test
import DiffOpt
import MathOptInterface as MOI
import Ipopt

Q = [
    4.0 1.0
    1.0 2.0
]
q = [1.0, 1.0]
G = [1.0 1.0]
h = [-1.0]

model = DiffOpt.diff_optimizer(Ipopt.Optimizer)
MOI.set(model, MOI.Silent(), true)
x = MOI.add_variables(model, 2)

# define objective
quad_terms = MOI.ScalarQuadraticTerm{Float64}[]
for i in 1:2
    for j in i:2 # indexes (i,j), (j,i) will be mirrored. specify only one kind
        push!(quad_terms, MOI.ScalarQuadraticTerm(Q[i, j], x[i], x[j]))
    end
end
objective_function =
    MOI.ScalarQuadraticFunction(quad_terms, MOI.ScalarAffineTerm.(q, x), 0.0)
MOI.set(
    model,
    MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(),
    objective_function,
)
MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)

# add constraint
c = MOI.add_constraint(
    model,
    MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(G[1, :], x), 0.0),
    MOI.LessThan(h[1]),
)
MOI.optimize!(model)

@test MOI.get(model, MOI.VariablePrimal(), x) ≈ [-0.25; -0.75] atol = ATOL rtol =
    RTOL

@test model.diff === nothing
MOI.set.(model, DiffOpt.ReverseVariablePrimal(), x, ones(2))
DiffOpt.reverse_differentiate!(model)

grad_wrt_h =
    MOI.constant(MOI.get(model, DiffOpt.ReverseConstraintFunction(), c))
@test grad_wrt_h ≈ -1.0 atol = 2ATOL rtol = RTOL
@test model.diff !== nothing

# adding two variables invalidates the cache
y = MOI.add_variables(model, 2)
for yi in y
    MOI.delete(model, yi)
end
@test model.diff === nothing
MOI.optimize!(model)

MOI.set.(model, DiffOpt.ReverseVariablePrimal(), x, ones(2))
DiffOpt.reverse_differentiate!(model)

grad_wrt_h =
    MOI.constant(MOI.get(model, DiffOpt.ReverseConstraintFunction(), c))

@test grad_wrt_h ≈ -1.0 atol = 1e-3
@test model.diff !== nothing
