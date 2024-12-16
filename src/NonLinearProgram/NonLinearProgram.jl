module NonLinearProgram

import DiffOpt
import JuMP
import MathOptInterface as MOI
using SparseArrays

include("nlp_utilities.jl")

Base.@kwdef struct Cache
    primal_vars::Vector{JuMP.VariableRef}         # Sorted primal variables
    dual_mapping::Dict{MOI.ConstraintIndex, Int}  # Unified mapping for constraints and bounds
    params::Vector{JuMP.VariableRef}              # VariableRefs for parameters
    index_duals::Vector{Int}                       # Indices for dual variables
    evaluator                                     # Cached evaluator for derivative computation
    cons                                          # Cached constraints from evaluator
end

Base.@kwdef struct ForwCache
    primal_Δs::Matrix{Float64}         # Sensitivity for primal variables (excluding slacks)
    dual_Δs::Matrix{Float64}           # Sensitivity for constraints and bounds (indexed by ConstraintIndex)
end

Base.@kwdef struct ReverseCache
    primal_Δs_T::Matrix{Float64}
    dual_Δs_T::Matrix{Float64}
end

mutable struct Form <: MOI.ModelLike
    model::MOI.Nonlinear.Model
    num_variables::Int
    num_constraints::Int
    sense::MOI.OptimizationSense
    list_of_constraint::MOI.Utilities.DoubleDicts.IndexDoubleDict
end

Form() = Form(MOI.Nonlinear.Model(), 0, 0, MOI.MIN_SENSE, MOI.Utilities.DoubleDicts.IndexDoubleDict())

function MOI.is_valid(model::Form, ref::MOI.VariableIndex) 
    return ref.value <= model.num_variables
end

function MOI.is_valid(model::Form, ref::MOI.ConstraintIndex)
    return ref.value <= model.num_constraints
end

function MOI.add_variable(form::Form)
    form.num_variables += 1
    return MOI.VariableIndex(form.num_variables)
end

function MOI.add_variables(form::Form, n)
    idxs = Vector{MOI.VariableIndex}(undef, n)
    for i in 1:n
        idxs[i] = MOI.add_variable(form)
    end
    return idxs
end

function MOI.supports(
    form::Form,
    attribute,
    val,
)
    return MOI.supports(form.model, attribute, val)
end

function MOI.supports_constraint(
    ::Form,
    ::Type{F},
    ::Type{S},
) where {
    F<:Union{MOI.ScalarNonlinearFunction,
        MOI.ScalarQuadraticFunction{Float64},
        MOI.ScalarAffineFunction{Float64}
    },
    S<:Union{
        MOI.GreaterThan{Float64},
        MOI.LessThan{Float64},
        MOI.Interval{Float64},
        MOI.EqualTo{Float64},
    }
}
    return true
end

function MOI.add_constraint(
    form::Form,
    func::F,
    set::S
) where {
    F<:Union{MOI.ScalarNonlinearFunction,
        MOI.ScalarQuadraticFunction{Float64},
        MOI.ScalarAffineFunction{Float64}
    },
    S<:Union{
        MOI.GreaterThan{Float64},
        MOI.LessThan{Float64},
        MOI.Interval{Float64},
        MOI.EqualTo{Float64},
    }
}
    form.num_constraints += 1
    MOI.Nonlinear.add_constraint(form.model, func, set)
    idx = MOI.ConstraintIndex{F, S}(form.num_constraints)
    form.list_of_constraint[idx] = idx
    return idx
end

function MOI.get(form::Form, ::MOI.ListOfConstraintTypesPresent)
    return collect(MOI.Utilities.DoubleDicts.outer_keys(form.list_of_constraint))
end

function MOI.get(form::Form, ::MOI.NumberOfConstraints{F,S}) where {F,S}
    return length(form.list_of_constraint[F,S])
end

function MOI.get(form::Form, ::MOI.ConstraintPrimalStart)
    return 
end

function MOI.supports(::Form, ::MOI.ObjectiveSense)
    return true
end

# function MOI.supports(::Form, ::MOI.ObjectiveFunction{MOI.ScalarNonlinearFunction})
#     return true
# end

function MOI.supports(::Form, ::MOI.ObjectiveFunction)
    return true
end

function MOI.set(
    form::Form,
    ::MOI.ObjectiveSense,
    sense::MOI.OptimizationSense
)
    form.sense = sense
    return
end

function MOI.get(
    form::Form,
    ::MOI.ObjectiveSense,
)
    return form.sense
end

function MOI.set(
    form::Form,
    ::MOI.ObjectiveFunction,
    func #::MOI.ScalarNonlinearFunction
)
    MOI.Nonlinear.set_objective(form.model, func)
    return
end

"""
    DiffOpt.NonLinearProgram.Model <: DiffOpt.AbstractModel

Model to differentiate nonlinear programs.

Supports forward and reverse differentiation, caching sensitivity data
for primal variables, constraints, and bounds, excluding slack variables.
"""
mutable struct Model <: DiffOpt.AbstractModel
    model::Form
    cache::Union{Nothing, Cache} # Cache for evaluator and mappings
    forw_grad_cache::Union{Nothing, ForwCache} # Cache for forward sensitivity results
    back_grad_cache::Union{Nothing, ReverseCache} # Cache for reverse sensitivity results
    diff_time::Float64
    x::Vector{Float64}
    y::Vector{Float64}
    s::Vector{Float64}
end

function Model()
    return Model(
        Form(),
        nothing,
        nothing,
        nothing,
        NaN,
        [],
        [],
        [],
    )
end

function MOI.set(
    model::Model,
    ::MOI.ConstraintPrimalStart,
    ci::MOI.ConstraintIndex,
    value,
)
    MOI.throw_if_not_valid(model, ci)
    return DiffOpt._enlarge_set(
        model.s,
        ci.value,
        value,
    )
end

function MOI.set(
    model::Model,
    ::MOI.ConstraintDualStart,
    ci::MOI.ConstraintIndex,
    value,
)
    MOI.throw_if_not_valid(model, ci)
    return DiffOpt._enlarge_set(
        model.y,
        ci.value,
        value,
    )
end

function MOI.set(
    model::Model,
    ::MOI.VariablePrimalStart,
    vi::MOI.VariableIndex,
    value,
)
    MOI.throw_if_not_valid(model, vi)
    return DiffOpt._enlarge_set(
        model.x,
        vi.value,
        value,
    )
end

function MOI.is_empty(model::Model)
    return model.cache === nothing
end

function MOI.empty!(model::Model)
    model.cache = nothing
    model.forw_grad_cache = nothing
    model.back_grad_cache = nothing
    model.diff_time = NaN
    return
end

function _cache_evaluator!(model::Model)
    # Retrieve and sort primal variables by index
    params = params=sort(all_params(model.model), by=x -> x.index.value)
    primal_vars = sort(all_primal_vars(model.model), by=x -> x.index.value)
    num_primal = length(primal_vars)

    # Create evaluator and constraints
    evaluator, cons = create_evaluator(model.model; x=[primal_vars; params])
    num_constraints = length(cons)
    # Analyze constraints and bounds
    leq_locations, geq_locations = find_inequealities(cons)
    num_leq = length(leq_locations)
    num_geq = length(geq_locations)
    has_up = findall(x -> has_upper_bound(x), primal_vars)
    has_low = findall(x -> has_lower_bound(x), primal_vars)
    num_low = length(has_low)
    num_up = length(has_up)

    # Create unified dual mapping
    # TODO: This assumes that these are all possible constraints available. We should change to either a dict or a sparse array
    # TODO: Check that the variable equal to works - Perhaps use bridge to change from equal to <= and >=
    dual_mapping = Vector{Int}(undef, num_constraints + num_low + num_up)
    for (i, ci) in enumerate(cons)
        dual_mapping[ci.index.value] = i
    end

    # Add bounds to dual mapping
    for (i, var_idx) in enumerate(has_low)
        lb = MOI.LowerBoundRef(primal_vars[var_idx])
        dual_mapping[lb.index] = num_constraints + i
    end
    offset += num_low
    for (i, var_idx) in enumerate(has_up)
        ub = MOI.UpperBoundRef(primal_vars[var_idx])
        dual_mapping[ub.index] = offset + i
    end

    num_slacks = num_leq + num_geq
    num_w = num_primal + num_slacks
    index_duals = [num_w+1:num_w+num_constraints; num_w+num_constraints+1:num_w+num_constraints+num_low; num_w+num_constraints+num_low+num_geq+1:num_w+num_constraints+num_low+num_geq+num_up]

    model.cache = Cache(
        primal_vars=primal_vars,
        dual_mapping=dual_mapping,
        params=params,
        index_duals=index_duals,
        evaluator=evaluator,
        cons=cons,
    )
    return model.cache
end

function DiffOpt.forward_differentiate!(model::Model; params=nothing)
    model.diff_time = @elapsed begin
        cache = _cache_evaluator!(model; params=params)

        Δp = [MOI.get(model, DiffOpt.ForwardParameter(), p.index) for p in cache.params]

        # Compute Jacobian
        Δs = compute_sensitivity(cache.evaluator, cache.cons; primal_vars=cache.primal_vars, params=cache.params)

        # Extract primal and dual sensitivities
        primal_Δs = Δs[1:cache.num_primal, :] * Δp # Exclude slacks
        dual_Δs = Δs[cache.index_duals, :]  * Δp # Includes constraints and bounds

        model.forw_grad_cache = ForwCache(
            primal_Δs=primal_Δs,
            dual_Δs=dual_Δs,
        )
    end
    return nothing
end

function DiffOpt.reverse_differentiate!(model::Model; params=nothing)
    model.diff_time = @elapsed begin
        cache = _cache_evaluator!(model; params=params)

        # Compute Jacobian
        Δs = compute_sensitivity(cache.evaluator, cache.cons; primal_vars=cache.primal_vars, params=cache.params)

        Δx = [MOI.get(model, DiffOpt.ReverseVariablePrimal(), x.index) for x in cache.primal_vars]
        Δλ = [MOI.get(model, DiffOpt.ReverseConstraintDual(), c.index) for c in cache.cons]
        Δvdown = [MOI.get(model, DiffOpt.ReverseVariableDual(), MOI.LowerBoundRef(x)) for x in cache.primal_vars if has_lower_bound(x)]
        Δvup = [MOI.get(model, DiffOpt.ReverseVariableDual(), MOI.UpperBoundRef(x)) for x in cache.primal_vars if has_upper_bound(x)]
        # Extract primal and dual sensitivities
        # TODO: multiply everyone together before indexing
        dual_Δ = zeros(size(Δs, 2))
        dual_Δ[1:cache.num_primal] = Δx
        dual_Δ[cache.index_duals] = [Δλ; Δvdown; Δvup]
        Δp = Δs' * dual_Δ

        model.back_grad_cache = ReverseCache(
            Δp=Δp,
        )
    end
    return nothing
end

function MOI.get(
    model::Model,
    ::DiffOpt.ForwardVariablePrimal,
    vi::MOI.VariableIndex,
)
    if model.forw_grad_cache === nothing
        error("Forward differentiation has not been performed yet.")
    end
    idx = vi.value # Direct mapping via sorted primal variables
    return model.forw_grad_cache.primal_Δs[idx, :]
end

function MOI.get(
    model::Model,
    ::DiffOpt.ForwardConstraintDual,
    ci::MOI.ConstraintIndex,
)
    if model.forw_grad_cache === nothing
        error("Forward differentiation has not been performed yet.")
    end
    try 
        idx = model.cache.dual_mapping[ci.value]
    catch
        error("ConstraintIndex not found in dual mapping.")
    end
    return model.forw_grad_cache.dual_Δs[idx, :]
end

# TODO: get for the reverse mode
function MOI.get(
    model::Model,
    ::DiffOpt.ReverseParameter,
    pi::MOI.VariableIndex,
)

return NaN
end

end # module NonLinearProgram
