using JuMP
using Clp
using DiffOpt
using Test

# This script creates the JuMP problem for a small unit commitment instance

ATOL=1e-4
RTOL=1e-4

## Problem data
unit_codes = [1, 2] # Generator identifiers
load_names = ["Load1", "Load2"] # Load identifiers
n_periods = 4 # Number of time periods
Pmin = Dict(1 => fill(0.5, n_periods), 2 => fill(0.5, n_periods)) # Minimum power output (pu)
Pmax = Dict(1 => fill(3.0, n_periods), 2 => fill(3.0, n_periods)) # Maximum power output (pu)
RR = Dict(1 => 0.25, 2 => 0.25) # Ramp rates (pu/min)
P0 = Dict(1 => 0.0, 2 => 0.0) # Initial power output (pu)
D = Dict("Load1" => [1.0, 1.2, 1.4, 1.6], "Load2" => [1.0, 1.2, 1.4, 1.6]) # Demand
Cp = Dict(1 => 1000.0, 2 => 1500.0) # Generation cost coefficient ($/pu)
Cnl = Dict(1 => 500.0, 2 => 1000.0) # No-load cost ($)

model = Model(() -> diff_optimizer(Clp.Optimizer))

## Variables
@variable(model, 0 <= u[g in unit_codes, t in 1:n_periods] <= 1) # Commitment
@variable(model, p[g in unit_codes, t in 1:n_periods] >= 0) # Power output

## Constraints

# Energy balance
@constraint(
    model,
    energy_balance_cons[t in 1:n_periods],
    sum(p[g, t] for g in unit_codes) == sum(D[l][t] for l in load_names)
)

# Generation limits
@constraint(model, [g in unit_codes, t in 1:n_periods], Pmin[g][t] * u[g, t] <= p[g, t])
@constraint(model, [g in unit_codes, t in 1:n_periods], p[g, t] <= Pmax[g][t] * u[g, t])

# Ramp rates
@constraint(model, [g in unit_codes, t in 2:n_periods], p[g, t] - p[g, t - 1] <= 60 * RR[g])
@constraint(model, [g in unit_codes], p[g, 1] - P0[g] <= 60 * RR[g])
@constraint(model, [g in unit_codes, t in 2:n_periods], p[g, t - 1] - p[g, t] <= 60 * RR[g])
@constraint(model, [g in unit_codes], P0[g] - p[g, 1] <= 60 * RR[g])

# Objective
@objective(
    model,
    Min,
    sum((Cp[g] * p[g, t]) + (Cnl[g] * u[g, t]) for g in unit_codes, t in 1:n_periods),
)

optimize!(model)

diff_opt = backend(model).optimizer.model
v = MOI.get(model, MOI.ListOfVariableIndices())
nv = length(v)

MOI.set.(diff_opt, DiffOpt.BackwardInVariablePrimal(), v, ones(nv))

DiffOpt.backward(diff_opt)

# sensitivity wrt linear objective
for (i,iv) in enumerate(v)
    grad = MOI.get(diff_opt, DiffOpt.BackwardOut{DiffOpt.LinearObjective}(), iv)
    @test grad ≈ 0.0  atol=ATOL rtol=RTOL
end

# sensitivity wrt RHS of constraints
for (idx, econs) in enumerate(energy_balance_cons)
    grad_energy_balance = MOI.get(model, DiffOpt.BackwardOut{DiffOpt.ConstraintConstant}(), econs)
    @test !≈(grad_energy_balance, 0)
end
