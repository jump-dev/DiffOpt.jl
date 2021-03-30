using JuMP
using Clp
using DiffOpt
using Test

# This script creates the JuMP problem for a small unit commitment instance

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
    [t in 1:n_periods],
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
nv = MOI.get(diff_opt, MOI.NumberOfVariables())
(dh, db, dq) = backward_quad(diff_opt, ["h", "b", "q"], ones(nv))

@test !all(dh .≈ 0)
@test !all(db .≈ 0)
@test all(dq .≈ 0)
