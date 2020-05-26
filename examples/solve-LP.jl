using Random
using GLPK
using MathOptInterface
using Dualization

const MOI = MathOptInterface;

D = 10  # variable dimension
N = 20; # no of inequality constraints

s = rand(N)
s = 2*s.-1
λ = max.(-s, 0)
s = max.(s, 0)
x̂ = rand(D)
A = rand(N, D)
b = A*x̂ .+ s
c = -A'*λ;

# can feed dual problem to optimizer like this:
# model = MOI.instantiate(dual_optimizer(GLPK.Optimizer), with_bridge_type=Float64)

model = GLPK.Optimizer()
x = MOI.add_variables(model, D)

# define objective
objective_function = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(c, x), 0.0)
MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), objective_function)
MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)

# will be useful later
constraint_indices = []

# set constraints
for i in 1:N
    push!(constraint_indices, MOI.add_constraint(model,MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(A[i,:], x), 0.),MOI.LessThan(b[i])))
end

for i in 1:D
    push!(constraint_indices, MOI.add_constraint(model,MOI.SingleVariable(x[i]),MOI.GreaterThan(0.)))
end

MOI.optimize!(model)

@assert MOI.get(model, MOI.TerminationStatus()) in [MOI.LOCALLY_SOLVED, MOI.OPTIMAL]

x̄ = MOI.get(model, MOI.VariablePrimal(), x);  # solution

@assert abs(c'x̄ - c'x̂) <= 1e-8   # sanity check

joint_object    = dualize(model)
dual_model_like = joint_object.dual_model # this is MOI.ModelLike, not an MOI.AbstractOptimizer; can't call optimizer on it
primal_dual_map = joint_object.primal_dual_map;

# copy the dual model objective, constraints, and variables to an optimizer
dual_model = GLPK.Optimizer()
MOI.copy_to(dual_model, dual_model_like)

# solve dual
MOI.optimize!(dual_model);

# NOTE: You can obtain components of the dual model individually by -
# dual_objective = dual_model_like.objective  # b'y
# dual_variable_indices = [primal_dual_map.primal_con_dual_var[x][1] for x in constraint_indices]
# dual_constraint_indices = [primal_dual_map.primal_var_dual_con[i] for i in x];

# ŷ = MOI.get(dm, MOI.VariablePrimal(), dual_variable_indices)

# check if strong duality holds
@assert abs(MOI.get(model, MOI.ObjectiveValue()) - MOI.get(dual_model, MOI.ObjectiveValue())) <= 1e-8


