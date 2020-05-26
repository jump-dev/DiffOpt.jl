using Random
using Ipopt
using MathOptInterface
using Dualization
using OSQP

const MOI = MathOptInterface;

n = 20 # variable dimension
m = 15 # no of inequality constraints
p = 10; # no of equality constraints

x̂ = rand(n)
Q = rand(n, n)
Q = Q'*Q # ensure PSD
q = rand(n)
G = rand(m, n)
h = G*x̂ + rand(m)
A = rand(p, n)
b = A*x̂;

model = MOI.instantiate(OSQP.Optimizer, with_bridge_type=Float64)
x = MOI.add_variables(model, n);

# define objective

quad_terms = MOI.ScalarQuadraticTerm{Float64}[]
for i in 1:n
    for j in i:n # indexes (i,j), (j,i) will be mirrored. specify only one kind
        push!(quad_terms, MOI.ScalarQuadraticTerm(Q[i,j],x[i],x[j]))
    end
end

objective_function = MOI.ScalarQuadraticFunction(MOI.ScalarAffineTerm.(q, x),quad_terms,0.0)
MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(), objective_function)
MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)

# add constraints
for i in 1:m
    MOI.add_constraint(model,MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(G[i,:], x), -h[i]),MOI.LessThan(0.))
end

for i in 1:p
    MOI.add_constraint(model,MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(A[i,:], x), -b[i]),MOI.EqualTo(0.))
end

MOI.optimize!(model)

@assert MOI.get(model, MOI.TerminationStatus()) in [MOI.LOCALLY_SOLVED, MOI.OPTIMAL]

x̄ = MOI.get(model, MOI.VariablePrimal(), x);

# objective value (predicted vs actual) sanity check
@assert 0.5*x̄'*Q*x̄ + q'*x̄  <= 0.5*x̂'*Q*x̂ + q'*x̂   

joint_object    = dualize(model)
dual_model_like = joint_object.dual_model # this is MOI.ModelLike, not an MOI.AbstractOptimizer; can't call optimizer on it
primal_dual_map = joint_object.primal_dual_map;

# copy the dual model objective, constraints, and variables to an optimizer
dual_model = MOI.instantiate(OSQP.Optimizer, with_bridge_type=Float64)
MOI.copy_to(dual_model, dual_model_like)

# solve dual
MOI.optimize!(dual_model);

# check if strong duality holds
@assert abs(MOI.get(model, MOI.ObjectiveValue()) - MOI.get(dual_model, MOI.ObjectiveValue())) <= 1e-2


