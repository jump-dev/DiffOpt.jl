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
    primal_Δs::Dict{MOI.VariableIndex, Float64}  # Sensitivity for primal variables (indexed by VariableIndex)
    dual_Δs::Vector{Float64}           # Sensitivity for constraints and bounds (indexed by ConstraintIndex)
end

Base.@kwdef struct ReverseCache
    Δp::Vector{Float64}  # Sensitivity for parameters
end

mutable struct Form <: MOI.ModelLike
    model::MOI.Nonlinear.Model
    num_variables::Int
    num_constraints::Int
    sense::MOI.OptimizationSense
    list_of_constraint::MOI.Utilities.DoubleDicts.IndexDoubleDict
    var2param::Dict{MOI.VariableIndex, MOI.Nonlinear.ParameterIndex}
    upper_bounds::Dict{Int, Float64}
    lower_bounds::Dict{Int, Float64}
    constraint_upper_bounds::Dict{Int, MOI.ConstraintIndex}
    constraint_lower_bounds::Dict{Int, MOI.ConstraintIndex}
    constraints_2_nlp_index::Dict{MOI.ConstraintIndex, MOI.Nonlinear.ConstraintIndex}
    nlp_index_2_constraint::Dict{MOI.Nonlinear.ConstraintIndex, MOI.ConstraintIndex}
end

Form() = Form(
    MOI.Nonlinear.Model(), 0, 0, MOI.MIN_SENSE, 
    MOI.Utilities.DoubleDicts.IndexDoubleDict(), 
    Dict{MOI.VariableIndex, MOI.Nonlinear.ParameterIndex}(),
    Dict{Int, Float64}(), Dict{Int, Float64}(),
    Dict{Int, MOI.ConstraintIndex}(), Dict{Int, MOI.ConstraintIndex}(),
    Dict{MOI.ConstraintIndex, MOI.Nonlinear.ConstraintIndex}(),
    Dict{MOI.Nonlinear.ConstraintIndex, MOI.ConstraintIndex}()
)

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
        MOI.ScalarAffineFunction{Float64},
        MOI.VariableIndex
    },
    S<:Union{
        MOI.GreaterThan{Float64},
        MOI.LessThan{Float64},
        MOI.Interval{Float64},
        MOI.EqualTo{Float64},
        MOI.Parameter{Float64}
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
        MOI.ScalarAffineFunction{Float64},
    },
    S<:Union{
        MOI.GreaterThan{Float64},
        MOI.LessThan{Float64},
        MOI.Interval{Float64},
        MOI.EqualTo{Float64},
    }
}
    form.num_constraints += 1
    idx_nlp = MOI.Nonlinear.add_constraint(form.model, func, set)
    idx = MOI.ConstraintIndex{F, S}(form.num_constraints)
    form.list_of_constraint[idx] = idx
    form.constraints_2_nlp_index[idx] = idx_nlp
    form.nlp_index_2_constraint[idx_nlp] = idx
    return idx
end

function MOI.add_constraint(
    form::Form,
    idx::F,
    set::S
) where {F<:MOI.VariableIndex, S<:MOI.Parameter{Float64}}
    form.num_constraints += 1
    p = MOI.Nonlinear.add_parameter(form.model, set.value)
    form.var2param[idx] = p
    idx = MOI.ConstraintIndex{F, S}(form.num_constraints)
    return idx
end

function MOI.add_constraint(
    form::Form,
    var_idx::F,
    set::S
) where {F<:MOI.VariableIndex, S<:MOI.GreaterThan}
    form.num_constraints += 1
    form.lower_bounds[var_idx.value] = set.lower
    idx = MOI.ConstraintIndex{F, S}(form.num_constraints)
    form.constraint_lower_bounds[var_idx.value] = idx
    return idx
end

function MOI.add_constraint(
    form::Form,
    var_idx::F,
    set::S
) where {F<:MOI.VariableIndex, S<:MOI.LessThan}
    form.num_constraints += 1
    form.upper_bounds[var_idx.value] = set.upper
    idx = MOI.ConstraintIndex{F, S}(form.num_constraints)
    form.constraint_upper_bounds[var_idx.value] = idx
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

include("nlp_utilities.jl")

all_variables(form::Form) = MOI.VariableIndex.(1:form.num_variables)
all_variables(model::Model) = all_variables(model.model)
all_params(form::Form) = collect(keys(form.var2param))
all_params(model::Model) = all_params(model.model)
all_primal_vars(form::Form) = setdiff(all_variables(form), all_params(form))
all_primal_vars(model::Model) = all_primal_vars(model.model)

get_num_constraints(form::Form) = length(form.list_of_constraint)
get_num_constraints(model::Model) = get_num_constraints(model.model)
get_num_primal_vars(form::Form) = length(all_primal_vars(form))
get_num_primal_vars(model::Model) = get_num_primal_vars(model.model)
get_num_params(form::Form) = length(all_params(form))
get_num_params(model::Model) = get_num_params(model.model)

function _cache_evaluator!(model::Model)
    form = model.model
    # Retrieve and sort primal variables by index
    params = params=sort(all_params(form), by=x -> x.value)
    primal_vars = sort(all_primal_vars(form), by=x -> x.value)
    num_primal = length(primal_vars)

    # Create evaluator and constraints
    evaluator = create_evaluator(form)
    num_constraints = get_num_constraints(form)
    # Analyze constraints and bounds
    leq_locations, geq_locations = find_inequealities(form)
    num_leq = length(leq_locations)
    num_geq = length(geq_locations)
    has_up = findall(i-> haskey(form.upper_bounds, i), primal_vars)
    has_low = findall(i-> haskey(form.lower_bounds, i), primal_vars)
    num_low = length(has_low)
    num_up = length(has_up)

    # Create unified dual mapping
    # TODO: This assumes that these are all possible constraints available. We should change to either a dict or a sparse array
    # TODO: Check that the variable equal to works - Perhaps use bridge to change from equal to <= and >=
    dual_mapping = Vector{Int}(undef, form.num_constraints)
    for (ci, cni) in form.constraints_2_nlp_index
        dual_mapping[ci.value] = cni.value
    end

    # Add bounds to dual mapping
    offset = num_constraints
    for (i, var_idx) in enumerate(has_low)
        # offset + i
        dual_mapping[form.constraint_lower_bounds[var_idx].value] = offset + i
    end
    offset += num_low
    for (i, var_idx) in enumerate(has_up)
        # offset + i
        dual_mapping[form.constraint_upper_bounds[var_idx].value] = offset + i
    end

    num_slacks = num_leq + num_geq
    num_w = num_primal + num_slacks
    index_duals = [num_w+1:num_w+num_constraints; num_w+num_constraints+1:num_w+num_constraints+num_low; num_w+num_constraints+num_low+num_geq+1:num_w+num_constraints+num_low+num_geq+num_up]
    cons = sort(collect(keys(form.nlp_index_2_constraint)), by=x->x.value)

    model.cache = Cache(
        primal_vars=primal_vars,
        dual_mapping=dual_mapping,
        params=params,
        index_duals=index_duals,
        leq_locations=leq_locations,
        geq_locations=geq_locations,
        has_up=has_up,
        has_low=has_low,
        evaluator=evaluator,
        cons=cons,
    )
    return model.cache
end

function DiffOpt.forward_differentiate!(model::Model)
    model.diff_time = @elapsed begin
        cache = _cache_evaluator!(model)
        Δp = [model.input_cache.dp[i] for i in cache.params]

        # Compute Jacobian
        Δs = compute_sensitivity(model)

        # Extract primal and dual sensitivities
        primal_Δs = Δs[1:length(model.cache.primal_vars), :] * Δp # Exclude slacks
        dual_Δs = Δs[cache.index_duals, :]  * Δp # Includes constraints and bounds

        model.forw_grad_cache = ForwCache(
            primal_Δs=Dict(model.cache.primal_vars .=> primal_Δs),
            dual_Δs=dual_Δs,
        )
    end
    return nothing
end

function DiffOpt.reverse_differentiate!(model::Model)
    model.diff_time = @elapsed begin
        cache = _cache_evaluator!(model)
        form = model.model

        # Compute Jacobian
        Δs = compute_sensitivity(model)
        num_primal = length(cache.primal_vars)
        Δx = zeros(num_primal)
        # [model.input_cache.dx[i] for i in cache.primal_vars]
        for (i, var_idx) in enumerate(cache.primal_vars)
            if haskey(model.input_cache.dx, var_idx)
                Δx[i] = model.input_cache.dx[var_idx]
            end
        end
        # ReverseConstraintDual
        num_constraints = length(cache.cons)
        num_up = length(cache.has_up)
        num_low = length(cache.has_low)
        Δdual = zeros(num_constraints + num_up + num_low)
        # Δdual[1:num_constraints] = [model.input_cache.dy[form.nlp_index_2_constraint[nlp_ci]] for nlp_ci in cache.cons]
        # Δdual[num_constraints+1:num_constraints+num_low] = [model.input_cache.dy[form.constraint_lower_bounds[form.primal_vars[i]].value] for i in cache.has_low]
        # Δdual[num_constraints+num_low+1:end] = [model.input_cache.dy[form.constraint_upper_bounds[form.primal_vars[i]].value] for i in cache.has_up]
        for (i, ci) in enumerate(cache.cons)
            idx = form.nlp_index_2_constraint[ci]
            if haskey(model.input_cache.dy, idx)
                Δdual[i] = model.input_cache.dy[idx]
            end
        end
        for (i, var_idx) in enumerate(cache.has_low)
            idx = form.constraint_lower_bounds[var_idx].value
            if haskey(model.input_cache.dy, idx)
                Δdual[num_constraints + i] = model.input_cache.dy[idx]
            end
        end
        for (i, var_idx) in enumerate(cache.has_up)
            idx = form.constraint_upper_bounds[var_idx].value
            if haskey(model.input_cache.dy, idx)
                Δdual[num_constraints + num_low + i] = model.input_cache.dy[idx]
            end
        end
        # Extract primal and dual sensitivities
        # TODO: multiply everyone together before indexing
        Δw = zeros(size(Δs, 1))
        Δw[1:num_primal] = Δx
        Δw[cache.index_duals] = Δdual
        Δp = Δs' * Δw

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
    form = model.model
    return model.back_grad_cache.Δp[form.var2param[pi].value]
end

end # module NonLinearProgram
