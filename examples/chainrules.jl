using JuMP
using Clp
using DiffOpt
using Test
using ChainRulesCore

# This script creates the JuMP model for a small unit commitment instance
# represented in a solution map function taking parameters as arguments and returning
# the optimal solution as output.

# The derivatives of this solution map can then be expressed in ChainRules semantics
# and implemented using DiffOpt

ATOL=1e-4
RTOL=1e-4

"""
Solution map of the problem using parameters:
- `load1_demand, load2_demand` for the demand of two nodes
- `gen_costs` is the vector of generator costs
- `noload_costs` is the vector of fixed activation costs of the generators,
and returning the optimal output power `p`.
"""
function unit_commitment(load1_demand, load2_demand, gen_costs, noload_costs; model = Model(() -> diff_optimizer(Clp.Optimizer)))
    ## Problem data
    unit_codes = [1, 2] # Generator identifiers
    load_names = ["Load1", "Load2"] # Load identifiers
    n_periods = 4 # Number of time periods
    Pmin = Dict(1 => fill(0.5, n_periods), 2 => fill(0.5, n_periods)) # Minimum power output (pu)
    Pmax = Dict(1 => fill(3.0, n_periods), 2 => fill(3.0, n_periods)) # Maximum power output (pu)
    RR = Dict(1 => 0.25, 2 => 0.25) # Ramp rates (pu/min)
    P0 = Dict(1 => 0.0, 2 => 0.0) # Initial power output (pu)
    D = Dict("Load1" => load1_demand, "Load2" => load2_demand) # Demand (pu)
    Cp = Dict(1 => gen_costs[1], 2 => gen_costs[2]) # Generation cost coefficient ($/pu)
    Cnl = Dict(1 => noload_costs[1], 2 => noload_costs[2]) # No-load cost ($)

    ## Variables
    # Note: u represents the activation of generation units.
    # Would be binary in the typical UC problem, relaxed here to u ∈ [0,1]
    # for a linear relaxation.
    @variable(model, 0 <= u[g in unit_codes, t in 1:n_periods] <= 1) # Commitment
    @variable(model, p[g in unit_codes, t in 1:n_periods] >= 0) # Power output (pu)

    ## Constraints

    # Energy balance
    @constraint(
        model,
        energy_balance_cons[t in 1:n_periods],
        sum(p[g, t] for g in unit_codes) == sum(D[l][t] for l in load_names),
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
    @assert termination_status(model) == MOI.OPTIMAL
    # converting to dense matrix
    return JuMP.value.(p.data)
end

@show unit_commitment([1.0, 1.2, 1.4, 1.6], [1.0, 1.2, 1.4, 1.6], [1000.0, 1500.0], [500.0, 1000.0])

# Forward differentiation rule for the solution map of the unit commitment problem
# taking in input perturbations on the input parameters and returning perturbations propagated to the result
function ChainRulesCore.frule((_, Δload1_demand, Δload2_demand, Δgen_costs, Δnoload_costs), ::typeof(unit_commitment), load1_demand, load2_demand, gen_costs, noload_costs)
    # creating the UC model with a DiffOpt optimizer wrapper around Clp
    model = Model(() -> diff_optimizer(Clp.Optimizer))
    # building and solving the main model
    pv = unit_commitment(load1_demand, load2_demand, gen_costs, noload_costs, model=model)
    energy_balance_cons = model[:energy_balance_cons]

    # Setting some perturbation of the right-hand side of the energy balance constraints
    # the RHS is equal to the sum of load demands at each period.
    # the corresponding perturbation are set accordingly as the set of perturbations of the two loads
    MOI.set.(
        model,
        DiffOpt.ForwardInConstraint(), energy_balance_cons,
        AffExpr[d1 + d2 for (d1, d2) in zip(Δload1_demand, Δload2_demand)],
    )


    p = model[:p]
    u = model[:u]

    # setting the perturbation of the linear objective
    Δobj = sum(Δgen_costs ⋅ p[:,t] + Δnoload_costs ⋅ u[:,t] for t in size(p, 2))
    MOI.set(model, DiffOpt.ForwardInObjective(), Δobj)
    # FIXME Workaround for https://github.com/jump-dev/JuMP.jl/issues/2797
    optimize!(model)
    DiffOpt.forward(JuMP.backend(model))
    # querying the corresponding perturbation of the decision
    Δp = MOI.get.(model, DiffOpt.ForwardOutVariablePrimal(), p)
    return (pv, Δp.data)
end


load1_demand = [1.0, 1.2, 1.4, 1.6]
load2_demand = [1.0, 1.2, 1.4, 1.6]
gen_costs = [1000.0, 1500.0]
noload_costs = [500.0, 1000.0]

Δload1_demand = 0 * load1_demand .+ 0.1
Δload2_demand = 0 * load2_demand .+ 0.2
Δgen_costs = 0 * gen_costs .+ 0.1
Δnoload_costs = 0 * noload_costs .+ 0.4
@show (pv, Δpv) = ChainRulesCore.frule((nothing, Δload1_demand, Δload2_demand, Δgen_costs, Δnoload_costs), unit_commitment, load1_demand, load2_demand, gen_costs, noload_costs)

# Reverse-mode differentiation of the solution map
# The computed pullback takes a seed for the optimal solution `̄p` and returns
# derivatives wrt each input parameter.
function ChainRulesCore.rrule(::typeof(unit_commitment), load1_demand, load2_demand, gen_costs, noload_costs; model = Model(() -> diff_optimizer(Clp.Optimizer)))
    # solve the forward UC problem
    pv = unit_commitment(load1_demand, load2_demand, gen_costs, noload_costs, model=model)
    function pullback_unit_commitment(pb)
        p = model[:p]
        u = model[:u]
        energy_balance_cons = model[:energy_balance_cons]

        MOI.set.(model, DiffOpt.BackwardInVariablePrimal(), p, pb)
        DiffOpt.backward(JuMP.backend(model))

        obj = MOI.get(model, DiffOpt.BackwardOutObjective())

        # computing derivative wrt linear objective costs
        dgen_costs = similar(gen_costs)
        dgen_costs[1] = sum(JuMP.coefficient.(obj, p[1,:]))
        dgen_costs[2] = sum(JuMP.coefficient.(obj, p[2,:]))

        dnoload_costs = similar(noload_costs)
        dnoload_costs[1] = sum(JuMP.coefficient.(obj, u[1,:]))
        dnoload_costs[2] = sum(JuMP.coefficient.(obj, u[2,:]))

        # computing derivative wrt constraint constant
        dload1_demand = JuMP.constant.(MOI.get.(model, DiffOpt.BackwardOutConstraint(), energy_balance_cons))
        dload2_demand = copy(dload1_demand)
        return (dload1_demand, dload2_demand, dgen_costs, dnoload_costs)
    end
    return (pv, pullback_unit_commitment)
end

(pv, pullback_unit_commitment) = ChainRulesCore.rrule(unit_commitment, load1_demand, load2_demand, gen_costs, noload_costs; model = Model(() -> diff_optimizer(Clp.Optimizer)))
@show pullback_unit_commitment(ones(size(pv)))
