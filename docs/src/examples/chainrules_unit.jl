# # ChainRules integration demo: Relaxed Unit Commitment

#md # [![](https://img.shields.io/badge/show-github-579ACA.svg)](@__REPO_ROOT_URL__/docs/src/examples/chainrules_unit.jl)


# In this example, we will demonstrate the integration of DiffOpt with
# [ChainRulesCore.jl](https://juliadiff.org/ChainRulesCore.jl/stable/),
# the library allowing the definition of derivatives for functions
# that can then be used by automatic differentiation systems.


using JuMP
import DiffOpt
import Plots
import LinearAlgebra: ⋅
import HiGHS
import ChainRulesCore

# ## Unit commitment problem

# We will consider a unit commitment problem, finding the cost-minimizing activation
# of generation units in a power network over multiple time periods.
# The considered constraints include:
# - Demand satisfaction of several loads
# - Ramping constraints
# - Generation limits.

# The decisions are:
# - ``u_{it} \in \{0,1\}``: activation of the ``i``-th unit at time ``t``
# - ``p_{it}``: power output of the ``i``-th unit at time ``t``.

# DiffOpt handles convex optimization problems only, we therefore
# relax the domain of the ``u_{it}`` variables to ``\left[0,1\right]``.

# ## Primal UC problem

# ChainRules defines the differentiation of functions.
# The actual function that is differentiated in the context of DiffOpt is the
# solution map taking in input the problem parameters and returning the solution.

function unit_commitment(
        load1_demand, load2_demand, gen_costs, noload_costs;
        model = Model(HiGHS.Optimizer), silent=false)
    MOI.set(model, MOI.Silent(), silent)

    ## Problem data
    units = [1, 2] # Generator identifiers
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
    ## Note: u represents the activation of generation units.
    ## Would be binary in the typical UC problem, relaxed here to u ∈ [0,1]
    ## for a linear relaxation.
    @variable(model, 0 <= u[g in units, t in 1:n_periods] <= 1) # Commitment
    @variable(model, p[g in units, t in 1:n_periods] >= 0) # Power output (pu)

    ## Constraints

    ## Energy balance
    @constraint(
        model,
        energy_balance_cons[t in 1:n_periods],
        sum(p[g, t] for g in units) == sum(D[l][t] for l in load_names),
    )

    ## Generation limits
    @constraint(model, [g in units, t in 1:n_periods], Pmin[g][t] * u[g, t] <= p[g, t])
    @constraint(model, [g in units, t in 1:n_periods], p[g, t] <= Pmax[g][t] * u[g, t])

    ## Ramp rates
    @constraint(model, [g in units, t in 2:n_periods], p[g, t] - p[g, t - 1] <= 60 * RR[g])
    @constraint(model, [g in units], p[g, 1] - P0[g] <= 60 * RR[g])
    @constraint(model, [g in units, t in 2:n_periods], p[g, t - 1] - p[g, t] <= 60 * RR[g])
    @constraint(model, [g in units], P0[g] - p[g, 1] <= 60 * RR[g])

    ## Objective
    @objective(
        model,
        Min,
        sum((Cp[g] * p[g, t]) + (Cnl[g] * u[g, t]) for g in units, t in 1:n_periods),
    )

    optimize!(model)
    ## asserting finite optimal value
    @assert termination_status(model) == MOI.OPTIMAL
    ## converting to dense matrix
    return JuMP.value.(p.data)
end

m = Model(HiGHS.Optimizer)
@show unit_commitment(
    [1.0, 1.2, 1.4, 1.6], [1.0, 1.2, 1.4, 1.6],
    [1000.0, 1500.0], [500.0, 1000.0],
    model=m, silent=true
)

# ## Perturbation of a single input parameter

# Let us vary the demand at the second time frame on both loads:


demand_values = 0.05:0.05:3.0
pvalues = map(demand_values) do di
    unit_commitment(
        [1.0, di, 1.4, 1.6], [1.0, di, 1.4, 1.6],
        [1000.0, 1500.0], [500.0, 1000.0],
        silent=true,
    )
end
pflat = [getindex.(pvalues, i) for i in eachindex(pvalues[1])];


# The influence of this variation of the demand is piecewise linear on the
# generation at different time frames:

Plots.scatter(demand_values, pflat, xaxis = ("Demand"), yaxis = ("Generation"))
Plots.title!("Different time frames and generators")
Plots.xlims!(0.0, 3.5)

# ## Forward Differentiation

# Forward differentiation rule for the solution map of the unit commitment problem.
# It takes as arguments:
# 1. the perturbations on the input parameters
# 2. the differentiated function
# 3. the primal values of the input parameters,

# and returns a tuple `(primal_output, perturbations)`, the main primal result
# and the perturbation propagated to this result:


function ChainRulesCore.frule(
        (_, Δload1_demand, Δload2_demand, Δgen_costs, Δnoload_costs),
        ::typeof(unit_commitment),
        load1_demand, load2_demand, gen_costs, noload_costs;
        optimizer=HiGHS.Optimizer,
        )
    ## creating the UC model with a DiffOpt optimizer wrapper around HiGHS
    model = Model(() -> DiffOpt.diff_optimizer(optimizer))
    ## building and solving the main model
    pv = unit_commitment(
        load1_demand, load2_demand, gen_costs, noload_costs, model=model)
    energy_balance_cons = model[:energy_balance_cons]

    ## Setting some perturbation of the energy balance constraints
    ## Perturbations are set as MOI functions
    Δenergy_balance = [
        convert(MOI.ScalarAffineFunction{Float64}, d1 + d2)
        for (d1, d2) in zip(Δload1_demand, Δload2_demand)
    ]
    MOI.set.(
        model,
        DiffOpt.ForwardConstraintFunction(), energy_balance_cons,
        Δenergy_balance,
    )

    p = model[:p]
    u = model[:u]

    ## setting the perturbation of the linear objective
    Δobj = sum(Δgen_costs ⋅ p[:,t] + Δnoload_costs ⋅ u[:,t] for t in size(p, 2))
    MOI.set(model, DiffOpt.ForwardObjective(), Δobj)
    DiffOpt.forward_differentiate!(JuMP.backend(model))
    ## querying the corresponding perturbation of the decision
    Δp = MOI.get.(model, DiffOpt.ForwardVariablePrimal(), p)
    return (pv, Δp.data)
end

# We can now compute the perturbation of the output powers `Δpv`
# for a perturbation of the first load demand at time 2:


load1_demand = [1.0, 1.0, 1.4, 1.6]
load2_demand = [1.0, 1.0, 1.4, 1.6]
gen_costs = [1000.0, 1500.0]
noload_costs = [500.0, 1000.0];

# all input perturbations are 0
# except first load at time 2
Δload1_demand = 0 * load1_demand
Δload1_demand[2] = 1.0
Δload2_demand = 0 * load2_demand
Δgen_costs = 0 * gen_costs
Δnoload_costs = 0 * noload_costs
(pv, Δpv) = ChainRulesCore.frule(
    (nothing, Δload1_demand, Δload2_demand, Δgen_costs, Δnoload_costs),
    unit_commitment,
    load1_demand, load2_demand, gen_costs, noload_costs,
)

Δpv


# The result matches what we observe in the previous figure:
# the generation of the first generator at the second time frame (third element on the plot).

# # Reverse-mode differentiation of the solution map

# The `rrule` returns the primal and a pullback.
# The pullback takes a seed for the optimal solution `̄p` and returns
# derivatives with respect to each input parameter of the function.

function ChainRulesCore.rrule(
        ::typeof(unit_commitment),
        load1_demand, load2_demand, gen_costs, noload_costs;
        optimizer=HiGHS.Optimizer,
        silent=false)
    model = Model(() -> DiffOpt.diff_optimizer(optimizer))
    ## solve the forward UC problem
    pv = unit_commitment(
        load1_demand, load2_demand, gen_costs, noload_costs,
        model=model, silent=silent)
    function pullback_unit_commitment(pb)
        p = model[:p]
        u = model[:u]
        energy_balance_cons = model[:energy_balance_cons]

        MOI.set.(model, DiffOpt.ReverseVariablePrimal(), p, pb)
        DiffOpt.reverse_differentiate!(JuMP.backend(model))

        obj = MOI.get(model, DiffOpt.ReverseObjective())

        ## computing derivative wrt linear objective costs
        dgen_costs = similar(gen_costs)
        dgen_costs[1] = sum(JuMP.coefficient.(obj, p[1,:]))
        dgen_costs[2] = sum(JuMP.coefficient.(obj, p[2,:]))

        dnoload_costs = similar(noload_costs)
        dnoload_costs[1] = sum(JuMP.coefficient.(obj, u[1,:]))
        dnoload_costs[2] = sum(JuMP.coefficient.(obj, u[2,:]))

        ## computing derivative wrt constraint constant
        dload1_demand = JuMP.constant.(
            MOI.get.(model, DiffOpt.ReverseConstraintFunction(), energy_balance_cons))
        dload2_demand = copy(dload1_demand)
        return (dload1_demand, dload2_demand, dgen_costs, dnoload_costs)
    end
    return (pv, pullback_unit_commitment)
end

# We can set a seed of one on the power of the first generator at the second time frame and zero for all other
# parts of the solution:

(pv, pullback_unit_commitment) = ChainRulesCore.rrule(
    unit_commitment,
    load1_demand, load2_demand, gen_costs, noload_costs,
    optimizer=HiGHS.Optimizer,
    silent=true,
)
dpv = 0 * pv
dpv[1,2] = 1
dargs = pullback_unit_commitment(dpv)
(dload1_demand, dload2_demand, dgen_costs, dnoload_costs) = dargs;

# The sensitivities with respect to the load demands are:
dload1_demand

# and:
dload2_demand

# The sensitivity of the generation is propagated to the sensitivity of both
# loads at the second time frame.

# This example integrating ChainRules was designed with support
# from [Invenia Technical Computing](https://www.invenia.ca/).
