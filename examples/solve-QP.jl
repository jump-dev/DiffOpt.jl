using Random
using Test

using MathOptInterface
const MOI = MathOptInterface
const MOIU = MathOptInterface.Utilities;

using Ipopt

n = 20 # variable dimension
m = 15 # no of inequality constraints

x̂ = rand(n)
Q = rand(n, n)
Q = Q' * Q # ensure PSD
q = rand(n)
G = rand(m, n)
h = G * x̂ + rand(m);

model = MOI.instantiate(Ipopt.Optimizer, with_bridge_type=Float64)
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

# maintain constrain to index map - will be useful later
constraint_indices = [
    MOI.add_constraint(
        model,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(G[i,:], x), 0.),MOI.LessThan(h[i]),
    ) for i in 1:m
]


MOI.optimize!(model)

@assert MOI.get(model, MOI.TerminationStatus()) in (MOI.LOCALLY_SOLVED, MOI.OPTIMAL)

x̄ = MOI.get(model, MOI.VariablePrimal(), x)

# objective value (predicted vs actual) sanity check
@test 0.5*x̄'*Q*x̄ + q'*x̄  <= 0.5*x̂'*Q*x̂ + q'*x̂   

# NOTE: can't use Ipopt
# Ipopt.Optimizer doesn't supports accessing MOI.ObjectiveFunctionType

#
# Verifying KKT Conditions
#

# complimentary slackness  + dual feasibility
for i in 1:size(constraint_indices)[1]
    con_index = constraint_indices[i]
    μ_i = MOI.get(model, MOI.ConstraintDual(), con_index)
    
    # μ[i] * (G * x - h)[i] = 0
    @test abs(μ_i * (G[i,:]' * x̄ - h[i])) < 3e-2

    # μ[i] <= 0
    @test μ_i <= 1e-2
end


# checking stationarity
for j in 1:n
    G_mu_sum = 0

    for i in 1:size(constraint_indices)[1]
        con_index = constraint_indices[i]
        μ_i = MOI.get(model, MOI.ConstraintDual(), con_index)

        G_mu_sum += μ_i * G[i,j]
    end

    @test abs(G_mu_sum - (Q * x̄ + q)[j]) < 1e-2
end
