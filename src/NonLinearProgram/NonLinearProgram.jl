# Copyright (c) 2025: Andrew Rosemberg and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module NonLinearProgram

import DiffOpt
import JuMP
import MathOptInterface as MOI
using SparseArrays
using LinearAlgebra

Base.@kwdef struct Cache
    primal_vars::Vector{MOI.VariableIndex}         # Sorted primal variables
    dual_mapping::Vector{Int}  # Unified mapping for constraints and bounds
    params::Vector{MOI.VariableIndex}              # VariableRefs for parameters
    index_duals::Vector{Int}                       # Indices for dual variables
    leq_locations::Vector{Int}                     # Locations of <= constraints
    geq_locations::Vector{Int}                     # Locations of >= constraints
    has_up::Vector{Int}                            # Variables with upper bounds
    has_low::Vector{Int}                           # Variables with lower bounds
    evaluator::MOI.Nonlinear.Evaluator            # Evaluator for the NLP
    cons::Vector{MOI.Nonlinear.ConstraintIndex} # Constraints index for the NLP
end

Base.@kwdef struct ForwCache
    primal_Δs::Dict{MOI.VariableIndex,Float64}  # Sensitivity for primal variables (indexed by VariableIndex)
    dual_Δs::Vector{Float64}           # Sensitivity for constraints and bounds (indexed by ConstraintIndex)
end

Base.@kwdef struct ReverseCache
    Δp::Vector{Float64}  # Sensitivity for parameters
end

# Define the form of the NLP
mutable struct Form <: MOI.ModelLike
    model::MOI.Nonlinear.Model
    num_variables::Int
    num_constraints::Int
    sense::MOI.OptimizationSense
    list_of_constraint::MOI.Utilities.DoubleDicts.IndexDoubleDict
    var2param::Dict{MOI.VariableIndex,MOI.Nonlinear.ParameterIndex}
    var2ci::Dict{MOI.VariableIndex,MOI.ConstraintIndex}
    upper_bounds::Dict{Int,Float64}
    lower_bounds::Dict{Int,Float64}
    constraint_upper_bounds::Dict{Int,MOI.ConstraintIndex}
    constraint_lower_bounds::Dict{Int,MOI.ConstraintIndex}
    constraints_2_nlp_index::Dict{
        MOI.ConstraintIndex,
        MOI.Nonlinear.ConstraintIndex,
    }
    nlp_index_2_constraint::Dict{
        MOI.Nonlinear.ConstraintIndex,
        MOI.ConstraintIndex,
    }
    leq_values::Dict{MOI.ConstraintIndex,Float64}
    geq_values::Dict{MOI.ConstraintIndex,Float64}
end

function Form()
    return Form(
        MOI.Nonlinear.Model(),
        0,
        0,
        MOI.MIN_SENSE,
        MOI.Utilities.DoubleDicts.IndexDoubleDict(),
        Dict{MOI.VariableIndex,MOI.Nonlinear.ParameterIndex}(),
        Dict{MOI.VariableIndex,MOI.ConstraintIndex}(),
        Dict{Int,Float64}(),
        Dict{Int,Float64}(),
        Dict{Int,MOI.ConstraintIndex}(),
        Dict{Int,MOI.ConstraintIndex}(),
        Dict{MOI.ConstraintIndex,MOI.Nonlinear.ConstraintIndex}(),
        Dict{MOI.Nonlinear.ConstraintIndex,MOI.ConstraintIndex}(),
        Dict{MOI.ConstraintIndex,Float64}(),
        Dict{MOI.ConstraintIndex,Float64}(),
    )
end

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

function MOI.supports(form::Form, attribute, val)
    return MOI.supports(form.model, attribute, val)
end

function MOI.supports_constraint(
    ::Form,
    ::Type{F},
    ::Type{S},
) where {
    F<:Union{
        MOI.ScalarNonlinearFunction,
        MOI.ScalarQuadraticFunction{Float64},
        MOI.ScalarAffineFunction{Float64},
        MOI.VariableIndex,
    },
    S<:Union{
        MOI.GreaterThan{Float64},
        MOI.LessThan{Float64},
        # MOI.Interval{Float64},
        MOI.EqualTo{Float64},
        MOI.Parameter{Float64},
    },
}
    return true
end

function _add_leq_geq(form::Form, idx::MOI.ConstraintIndex, set::MOI.GreaterThan)
    form.geq_values[idx] = set.lower
    return
end

function _add_leq_geq(form::Form, idx::MOI.ConstraintIndex, set::MOI.LessThan)
    form.leq_values[idx] = set.upper
    return
end

function _add_leq_geq(::Form, ::MOI.ConstraintIndex, ::MOI.EqualTo)
    return
end

function MOI.add_constraint(
    form::Form,
    func::F,
    set::S,
) where {
    F<:Union{
        MOI.ScalarNonlinearFunction,
        MOI.ScalarQuadraticFunction{Float64},
        MOI.ScalarAffineFunction{Float64},
    },
    S<:Union{
        MOI.GreaterThan{Float64},
        MOI.LessThan{Float64},
        # MOI.Interval{Float64},
        MOI.EqualTo{Float64},
    },
}
    form.num_constraints += 1
    idx_nlp = MOI.Nonlinear.add_constraint(form.model, func, set)
    idx = MOI.ConstraintIndex{F,S}(form.num_constraints)
    _add_leq_geq(form, idx, set)
    form.list_of_constraint[idx] = idx
    form.constraints_2_nlp_index[idx] = idx_nlp
    form.nlp_index_2_constraint[idx_nlp] = idx
    return idx
end

function MOI.add_constraint(
    form::Form,
    func::F,
    set::S,
) where {F<:MOI.VariableIndex,S<:MOI.EqualTo}
    form.num_constraints += 1
    idx_nlp = MOI.Nonlinear.add_constraint(form.model, func, set)
    idx = MOI.ConstraintIndex{F,S}(form.num_constraints)
    _add_leq_geq(form, idx, set)
    form.list_of_constraint[idx] = idx
    form.constraints_2_nlp_index[idx] = idx_nlp
    form.nlp_index_2_constraint[idx_nlp] = idx
    return idx
end

function MOI.add_constraint(
    form::Form,
    idx::F,
    set::S,
) where {F<:MOI.VariableIndex,S<:MOI.Parameter{Float64}}
    form.num_constraints += 1
    p = MOI.Nonlinear.add_parameter(form.model, set.value)
    form.var2param[idx] = p
    idx_ci = MOI.ConstraintIndex{F,S}(form.num_constraints)
    form.var2ci[idx] = idx_ci
    return idx_ci
end

function MOI.add_constraint(
    form::Form,
    var_idx::F,
    set::S,
) where {F<:MOI.VariableIndex,S<:MOI.GreaterThan}
    form.num_constraints += 1
    form.lower_bounds[var_idx.value] = set.lower
    idx = MOI.ConstraintIndex{F,S}(form.num_constraints)
    form.list_of_constraint[idx] = idx
    form.constraint_lower_bounds[var_idx.value] = idx
    return idx
end

function MOI.add_constraint(
    form::Form,
    var_idx::F,
    set::S,
) where {F<:MOI.VariableIndex,S<:MOI.LessThan}
    form.num_constraints += 1
    form.upper_bounds[var_idx.value] = set.upper
    idx = MOI.ConstraintIndex{F,S}(form.num_constraints)
    form.list_of_constraint[idx] = idx
    form.constraint_upper_bounds[var_idx.value] = idx
    return idx
end

function MOI.get(form::Form, ::MOI.ListOfConstraintTypesPresent)
    return collect(
        MOI.Utilities.DoubleDicts.outer_keys(form.list_of_constraint),
    )
end

function MOI.get(form::Form, ::MOI.NumberOfConstraints{F,S}) where {F,S}
    return length(form.list_of_constraint[F, S])
end

function MOI.get(::Form, ::MOI.ConstraintPrimalStart)
    return
end

function MOI.supports(::Form, ::MOI.ObjectiveSense)
    return true
end

function MOI.supports(::Form, ::MOI.ObjectiveFunction)
    return true
end

function MOI.set(form::Form, ::MOI.ObjectiveSense, sense::MOI.OptimizationSense)
    form.sense = sense
    return
end

function MOI.get(form::Form, ::MOI.ObjectiveSense)
    return form.sense
end

function MOI.set(
    form::Form,
    ::MOI.ObjectiveFunction,
    func, #::MOI.ScalarNonlinearFunction
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
    cache::Union{Nothing,Cache} # Cache for evaluator and mappings
    forw_grad_cache::Union{Nothing,ForwCache} # Cache for forward sensitivity results
    back_grad_cache::Union{Nothing,ReverseCache} # Cache for reverse sensitivity results
    diff_time::Float64
    input_cache::DiffOpt.InputCache
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
        DiffOpt.InputCache(),
        [],
        [],
        [],
    )
end

objective_sense(form::Form) = form.sense
objective_sense(model::Model) = objective_sense(model.model)

function MOI.set(
    model::Model,
    ::MOI.ConstraintPrimalStart,
    ci::MOI.ConstraintIndex,
    value,
)
    MOI.throw_if_not_valid(model, ci)
    return DiffOpt._enlarge_set(model.s, ci.value, value)
end

function MOI.supports(
    ::Model,
    ::MOI.ConstraintDualStart,
    ::Type{MOI.ConstraintIndex{MOI.VariableIndex,S}},
) where {
    S<:Union{
        MOI.GreaterThan{Float64},
        MOI.LessThan{Float64},
        MOI.EqualTo{Float64},
        MOI.Interval{Float64},
    },
}
    return true
end

function MOI.set(
    model::Model,
    ::MOI.ConstraintDualStart,
    ci::MOI.ConstraintIndex,
    value,
)
    MOI.throw_if_not_valid(model, ci)
    return DiffOpt._enlarge_set(model.y, ci.value, value)
end

function MOI.set(
    model::Model,
    ::MOI.VariablePrimalStart,
    vi::MOI.VariableIndex,
    value,
)
    MOI.throw_if_not_valid(model, vi)
    return DiffOpt._enlarge_set(model.x, vi.value, value)
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

include("nlp_utilities.jl")

_all_variables(form::Form) = MOI.VariableIndex.(1:form.num_variables)
_all_variables(model::Model) = _all_variables(model.model)
_all_params(form::Form) = collect(keys(form.var2param))
_all_params(model::Model) = _all_params(model.model)
_all_primal_vars(form::Form) = setdiff(_all_variables(form), _all_params(form))
_all_primal_vars(model::Model) = _all_primal_vars(model.model)

_get_num_constraints(form::Form) = length(form.constraints_2_nlp_index)
_get_num_constraints(model::Model) = _get_num_constraints(model.model)
_get_num_primal_vars(form::Form) = length(_all_primal_vars(form))
_get_num_primal_vars(model::Model) = _get_num_primal_vars(model.model)
_get_num_params(form::Form) = length(_all_params(form))
_get_num_params(model::Model) = _get_num_params(model.model)

function _cache_evaluator!(model::Model)
    form = model.model
    # Retrieve and sort primal variables by NLP index
    params = sort(_all_params(form); by = x -> x.value)
    primal_vars = sort(_all_primal_vars(form); by = x -> x.value)
    num_primal = length(primal_vars)

    # Create evaluator and constraints
    evaluator = _create_evaluator(form)
    num_constraints = _get_num_constraints(form)
    # Analyze constraints and bounds
    leq_locations, geq_locations = _find_inequalities(form)
    num_leq = length(leq_locations)
    num_geq = length(geq_locations)
    has_up = findall(i -> haskey(form.upper_bounds, i.value), primal_vars)
    has_low = findall(i -> haskey(form.lower_bounds, i.value), primal_vars)
    num_low = length(has_low)
    num_up = length(has_up)

    # Create unified dual mapping from constraint index to NLP index
    dual_mapping = Vector{Int}(undef, form.num_constraints)
    for (ci, cni) in form.constraints_2_nlp_index
        dual_mapping[ci.value] = cni.value
    end

    # Add bounds to dual mapping
    offset = num_constraints
    for (i, var_idx) in enumerate(primal_vars[has_low])
        # offset + i
        dual_mapping[form.constraint_lower_bounds[var_idx.value].value] =
            offset + i
    end
    offset += num_low
    for (i, var_idx) in enumerate(primal_vars[has_up])
        # offset + i
        dual_mapping[form.constraint_upper_bounds[var_idx.value].value] =
            offset + i
    end

    num_slacks = num_leq + num_geq
    num_w = num_primal + num_slacks
    # Create index for dual variables
    index_duals = [
        num_w+1:num_w+num_constraints
        num_w+num_constraints+1:num_w+num_constraints+num_low
        num_w+num_constraints+num_low+num_geq+1:num_w+num_constraints+num_low+num_geq+num_up
    ]
    cons = sort(collect(keys(form.nlp_index_2_constraint)); by = x -> x.value)

    model.cache = Cache(;
        primal_vars = primal_vars,
        dual_mapping = dual_mapping,
        params = params,
        index_duals = index_duals,
        leq_locations = leq_locations,
        geq_locations = geq_locations,
        has_up = has_up,
        has_low = has_low,
        evaluator = evaluator,
        cons = cons,
    )
    return model.cache
end

function DiffOpt.forward_differentiate!(
    model::Model;
    tol = 1e-6,
)
    model.diff_time = @elapsed begin
        cache = _cache_evaluator!(model)
        form = model.model
        # Fetch parameter sensitivities
        Δp = zeros(length(cache.params))
        for (i, var_idx) in enumerate(cache.params)
            ky = form.var2ci[var_idx]
            if haskey(model.input_cache.dp, ky) # only for set sensitivities
                Δp[i] = model.input_cache.dp[ky]
            end
        end

        # Compute Jacobian
        Δs = _compute_sensitivity(
            model;
            tol = tol,
        )

        # Extract primal and dual sensitivities
        primal_Δs = Δs[1:length(model.cache.primal_vars), :] * Δp # Exclude slacks
        dual_Δs = Δs[cache.index_duals, :] * Δp # Includes constraints and bounds

        model.forw_grad_cache = ForwCache(;
            primal_Δs = Dict(model.cache.primal_vars .=> primal_Δs),
            dual_Δs = dual_Δs,
        )
    end
    return nothing
end

function DiffOpt.reverse_differentiate!(
    model::Model;
    tol = 1e-6,
)
    model.diff_time = @elapsed begin
        cache = _cache_evaluator!(model)
        form = model.model

        # Compute Jacobian
        Δs = _compute_sensitivity(
            model;
            tol = tol,
        )
        num_primal = length(cache.primal_vars)
        # Fetch primal sensitivities
        Δx = zeros(num_primal)
        for (i, var_idx) in enumerate(cache.primal_vars)
            if haskey(model.input_cache.dx, var_idx)
                Δx[i] = model.input_cache.dx[var_idx]
            end
        end
        # Fetch dual sensitivities
        num_constraints = length(cache.cons)
        num_up = length(cache.has_up)
        num_low = length(cache.has_low)
        Δdual = zeros(num_constraints + num_up + num_low)
        for (i, ci) in enumerate(cache.cons)
            idx = form.nlp_index_2_constraint[ci]
            if haskey(model.input_cache.dy, idx)
                Δdual[i] = model.input_cache.dy[idx]
            end
        end
        for (i, var_idx) in enumerate(cache.primal_vars[cache.has_low])
            idx = form.constraint_lower_bounds[var_idx.value].value
            if haskey(model.input_cache.dy, idx)
                Δdual[num_constraints+i] = model.input_cache.dy[idx]
            end
        end
        for (i, var_idx) in enumerate(cache.primal_vars[cache.has_up])
            idx = form.constraint_upper_bounds[var_idx.value].value
            if haskey(model.input_cache.dy, idx)
                Δdual[num_constraints+num_low+i] = model.input_cache.dy[idx]
            end
        end
        # Extract Parameter sensitivities
        Δw = zeros(size(Δs, 1))
        Δw[1:num_primal] = Δx
        Δw[cache.index_duals] = Δdual
        Δp = Δs' * Δw

        # Order by ConstraintIndex
        varorder =
            sort(collect(keys(form.var2ci)); by = x -> form.var2ci[x].value)
        Δp = [Δp[form.var2param[var_idx].value] for var_idx in varorder]

        model.back_grad_cache = ReverseCache(; Δp = Δp)
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
    return model.forw_grad_cache.primal_Δs[vi]
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
        return model.forw_grad_cache.dual_Δs[idx]
    catch
        error("ConstraintIndex not found in dual mapping.")
    end
end

function MOI.get(
    model::Model,
    ::DiffOpt.ReverseConstraintSet,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,MOI.Parameter{T}},
) where {T}
    return MOI.Parameter{T}(model.back_grad_cache.Δp[ci.value])
end

end # module NonLinearProgram
