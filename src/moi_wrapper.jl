"""
    diff_optimizer(optimizer_constructor)::Optimizer

Creates a `DiffOpt.Optimizer`, which is an MOI layer with an internal optimizer
and other utility methods. Results (primal, dual and slack values) are obtained
by querying the internal optimizer instantiated using the
`optimizer_constructor`. These values are required for find jacobians with respect to problem data.

One define a differentiable model by using any solver of choice. Example:

```julia
julia> import DiffOpt, HiGHS

julia> model = DiffOpt.diff_optimizer(HiGHS.Optimizer)
julia> x = model.add_variable(model)
julia> model.add_constraint(model, ...)
```
"""
function diff_optimizer(optimizer_constructor)::Optimizer
    optimizer = MOI.instantiate(optimizer_constructor, with_bridge_type=Float64)
    # When we do `MOI.copy_to(diff, optimizer)` we need to efficiently `MOI.get`
    # the model information from `optimizer`. However, 1) `optimizer` may not
    # implement some getters or it may be inefficient and 2) the getters may be
    # unimplemented or inefficient through some bridges.
    # For this reason we add a cache layer, the same cache JuMP adds.
    caching_opt = MOIU.CachingOptimizer(
        MOIU.UniversalFallback(MOIU.Model{Float64}()),
        optimizer,
    )
    return Optimizer(caching_opt)
end

mutable struct Optimizer{OT <: MOI.ModelLike} <: MOI.AbstractOptimizer
    optimizer::OT

    model_constructors::Vector{Any}
    model_constructor::Any

    diff::Any

    index_map::Union{Nothing,MOI.Utilities.IndexMap}

    # sensitivity input cache using MOI like sparse format
    input_cache::InputCache

    function Optimizer(optimizer::OT) where {OT <: MOI.ModelLike}
        output = new{OT}(
            optimizer,
            Any[],
            nothing,
            nothing,
            nothing,
            InputCache(),
        )
        add_all_model_constructors(output)
        return output
    end
end

"""
   add_model_constructor(optimizer::Optimizer, model_constructor)

Add the constructor of [`AbstractModel`](@ref) for `optimizer` to choose
from when trying to differentiate.
"""
function add_model_constructor(optimizer::Optimizer, model_constructor)
    push!(optimizer.model_constructors, model_constructor)
    return
end

function MOI.add_variable(model::Optimizer)
    model.diff = nothing
    vi = MOI.add_variable(model.optimizer)
    return vi
end

function MOI.add_variables(model::Optimizer, N::Int)
    model.diff = nothing
    return VI[MOI.add_variable(model) for i in 1:N]
end

function MOI.add_constraint(model::Optimizer, f::MOI.AbstractFunction, s::MOI.AbstractSet)
    model.diff = nothing
    return MOI.add_constraint(model.optimizer, f, s)
end

function MOI.add_constraints(model::Optimizer, f::AbstractVector{F}, s::AbstractVector{S}) where {F<:MOI.AbstractFunction, S<: MOI.AbstractSet}
    model.diff = nothing
    return CI{F, S}[MOI.add_constraint(model, f[i], s[i]) for i in eachindex(f)]
end

function MOI.set(model::Optimizer, attr::MOI.ObjectiveFunction{F}, f::F) where {F<:MOI.AbstractFunction}
    model.diff = nothing
    MOI.set(model.optimizer, attr, f)
end

MOI.supports(model::Optimizer, attr::MOI.ObjectiveSense) = MOI.supports(model.optimizer, attr)

function MOI.set(model::Optimizer, attr::MOI.ObjectiveSense, sense::MOI.OptimizationSense)
    model.diff = nothing
    return MOI.set(model.optimizer, attr, sense)
end

function MOI.get(model::Optimizer, attr::MOI.AbstractModelAttribute)
    return MOI.get(model.optimizer, attr)
end

function MOI.get(model::Optimizer, attr::MOI.ListOfConstraintIndices)
    return MOI.get(model.optimizer, attr)
end

function MOI.get(model::Optimizer, attr::MOI.ConstraintSet, ci::CI)
    return MOI.get(model.optimizer, attr, ci)
end

function MOI.set(
    model::Optimizer, attr::MOI.ConstraintSet, ci::CI{F, S}, s::S
) where {F,S}
    model.diff = nothing
    return MOI.set(model.optimizer, attr, ci, s)
end

function MOI.get(
    model::Optimizer, attr::MOI.ConstraintFunction, ci::CI{F, S}
) where {F,S}
    return MOI.get(model.optimizer, attr, ci)
end

function MOI.set(
    model::Optimizer, attr::MOI.ConstraintFunction, ci::CI{F, S}, f::F
) where {F,S}
    model.diff = nothing
    return MOI.set(model.optimizer, attr, ci, f)
end

# `MOI.supports` methods

function MOI.supports(model::Optimizer, attr::MOI.AbstractModelAttribute)
    return MOI.supports(model.optimizer, attr)
end

function MOI.supports(model::Optimizer, attr::MOI.ObjectiveFunction)
    return MOI.supports(model.optimizer, attr)
end

function MOI.supports_constraint(
    model::Optimizer, ::Type{F}, ::Type{S},
) where {F<:MOI.AbstractFunction,S<:MOI.AbstractSet}
    MOI.supports_constraint(model.optimizer, F, S)
end
function MOI.supports(
    model::Optimizer, attr::MOI.ConstraintName, ::Type{CI{F, S}}
) where {F,S}
    return MOI.supports(model.optimizer, attr, CI{F,S})
end

MOI.get(model::Optimizer, attr::MOI.SolveTimeSec) = MOI.get(model.optimizer, attr)

function MOI.empty!(model::Optimizer)
    MOI.empty!(model.optimizer)
    model.diff = nothing
    model.index_map = nothing
    empty!(model.input_cache)
    return
end

function MOI.is_empty(model::Optimizer)
    return MOI.is_empty(model.optimizer) &&
           model.diff === nothing
end

function MOI.supports_incremental_interface(model::Optimizer)
    if !MOI.supports_incremental_interface(model.optimizer)
        error(
            "DiffOpt requires a solver that " *
            "`MOI.supports_incremental_interface`, which is not the case for " *
            "$(model.optimizer)")
    end
    return true
end

function MOI.copy_to(model::Optimizer, src::MOI.ModelLike)
    model.diff = nothing
    return MOIU.default_copy_to(model.optimizer, src)
end

function MOI.get(model::Optimizer, ::MOI.TerminationStatus)
    return MOI.get(model.optimizer, MOI.TerminationStatus())
end

function MOI.set(model::Optimizer, ::MOI.VariablePrimalStart,
                 vi::VI, value::Float64)
    model.diff = nothing
    MOI.set(model.optimizer, MOI.VariablePrimalStart(), vi, value)
end

function MOI.supports(model::Optimizer, attr::MOI.AbstractVariableAttribute,
                      ::Type{MOI.VariableIndex})
    return MOI.supports(model.optimizer, attr, MOI.VariableIndex)
end

function MOI.set(model::Optimizer, attr::MOI.AbstractVariableAttribute, v::MOI.VariableIndex, value)
    MOI.set(model.optimizer, attr, v, value)
end

function MOI.get(model::Optimizer, attr::MOI.AbstractVariableAttribute, vi::MOI.VariableIndex)
    return MOI.get(model.optimizer, attr, vi)
end

function MOI.delete(model::Optimizer, ci::CI{F,S}) where {F,S}
    model.diff = nothing
    MOI.delete(model.optimizer, ci)
end

function MOI.get(model::Optimizer, attr::MOI.ConstraintPrimal, ci::CI{F,S}) where {F,S}
    return MOI.get(model.optimizer, attr, ci)
end

function MOI.is_valid(model::Optimizer, v::VI)
    return MOI.is_valid(model.optimizer, v::VI)
end

MOI.is_valid(model::Optimizer, con::CI) = MOI.is_valid(model.optimizer, con)

function MOI.get(model::Optimizer, attr::MOI.ConstraintDual, ci::CI{F,S}) where {F,S}
    return MOI.get(model.optimizer, attr, ci)
end

function MOI.get(model::Optimizer, ::MOI.ConstraintBasisStatus, ci::CI{F,S}) where {F,S}
    return MOI.get(model.optimizer, MOI.ConstraintBasisStatus(), ci)
end

# helper methods to check if a constraint contains a Variable
function _constraint_contains(model::Optimizer, v::VI, ci::CI{MOI.VariableIndex})
    return v == MOI.get(model, MOI.ConstraintFunction(), ci)
end

function _constraint_contains(model::Optimizer, v::VI, ci::CI{MOI.ScalarAffineFunction{Float64}})
    func = MOI.get(model, MOI.ConstraintFunction(), ci)
    return any(term -> v == term.variable, func.terms)
end

function _constraint_contains(model::Optimizer, v::VI, ci::CI{MOI.VectorOfVariables})
    func = MOI.get(model, MOI.ConstraintFunction(), ci)
    return v in func.variables
end

function _constraint_contains(model::Optimizer, v::VI, ci::CI{MOI.VectorAffineFunction{Float64}})
    func = MOI.get(model, MOI.ConstraintFunction(), ci)
    return any(term -> v == term.scalar_term.variable, func.terms)
end


function MOI.delete(model::Optimizer, v::VI)
    model.diff = nothing
    MOI.delete(model.optimizer, v)
end

# for array deletion
function MOI.delete(model::Optimizer, indices::Vector{VI})
    model.diff = nothing
    for i in indices
        MOI.delete(model, i)
    end
end

function MOI.modify(
    model::Optimizer,
    ::MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}},
    chg::MOI.AbstractFunctionModification
)
    model.diff = nothing
    MOI.modify(
        model.optimizer,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        chg
    )
end

function MOI.modify(model::Optimizer, ci::CI, chg::MOI.AbstractFunctionModification)
    model.diff = nothing
    MOI.modify(model.optimizer, ci, chg)
end

function MOI.get(model::Optimizer, ::Type{MOI.VariableIndex}, name::String)
    return MOI.get(model.optimizer, MOI.VariableIndex, name)
end

function MOI.set(model::Optimizer, ::MOI.ConstraintName, con::CI, name::String)
    MOI.set(model.optimizer, MOI.ConstraintName(), con, name)
end

function MOI.get(model::Optimizer, ::Type{MOI.ConstraintIndex}, name::String)
    return MOI.get(model.optimizer, MOI.ConstraintIndex, name)
end

function MOI.get(model::Optimizer, ::MOI.ConstraintName, con::CI)
    return MOI.get(model.optimizer, MOI.ConstraintName(), con)
end

function MOI.get(model::Optimizer, ::MOI.ConstraintName, ::Type{CI{F, S}}) where {F,S}
    return MOI.get(model.optimizer, MOI.ConstraintName(), CI{F,S})
end

function MOI.set(model::Optimizer, ::MOI.Name, name::String)
    MOI.set(model.optimizer, MOI.Name(), name)
end

function MOI.get(model::Optimizer, ::Type{CI{F, S}}, name::String) where {F,S}
    return MOI.get(model.optimizer, CI{F,S}, name)
end

function MOI.supports(model::Optimizer, attr::MOI.TimeLimitSec)
    MOI.supports(model.optimizer, attr)
end
function MOI.set(model::Optimizer, ::MOI.TimeLimitSec, value::Union{Real, Nothing})
    MOI.set(model.optimizer, MOI.TimeLimitSec(), value)
end
function MOI.get(model::Optimizer, ::MOI.TimeLimitSec)
    return MOI.get(model.optimizer, MOI.TimeLimitSec())
end

MOI.supports(model::Optimizer, ::MOI.Silent) = MOI.supports(model.optimizer, MOI.Silent())

function MOI.set(model::Optimizer, ::MOI.Silent, value)
    MOI.set(model.optimizer, MOI.Silent(), value)
end

function MOI.get(model::Optimizer, ::MOI.Silent)
    return MOI.get(model.optimizer, MOI.Silent())
end

function MOI.get(model::Optimizer, attr::A) where {A <: Union{MOI.SolverName, MOI.SolverVersion}}
    return MOI.get(model.optimizer, attr)
end

function MOI.optimize!(model::Optimizer)
    model.diff = nothing
    MOI.optimize!(model.optimizer)
    return
end

"""
    ModelConstructor <: MOI.AbstractOptimizerAttribute

Determines which subtype of [`DiffOpt.AbstractModel`](@ref) to use for
differentiation. When set to `nothing`, the first one out of
`model.model_constructors` that support the problem is used.
"""
struct ModelConstructor <: MOI.AbstractOptimizerAttribute end

MOI.supports(::Optimizer, ::ModelConstructor) = true
MOI.get(model::Optimizer, ::ModelConstructor) = model.model_constructor
function MOI.set(model::Optimizer, ::ModelConstructor, model_constructor)
    model.diff = nothing
    model.model_constructor = model_constructor
end

function reverse_differentiate!(model::Optimizer)
    st = MOI.get(model.optimizer, MOI.TerminationStatus())
    if !in(st,  (MOI.LOCALLY_SOLVED, MOI.OPTIMAL))
        error("Trying to compute the reverse differentiation on a model with termination status $(st)")
    end
    diff = _diff(model)
    for (vi, value) in model.input_cache.dx
        MOI.set(diff, ReverseVariablePrimal(), model.index_map[vi], value)
    end
    return reverse_differentiate!(diff)
end

function _copy_forward_in_constraint(diff, index_map, con_map, constraints)
    for (index, value) in constraints
        MOI.set(diff, ForwardConstraintFunction(), con_map[index], MOI.Utilities.map_indices(index_map, value))
    end
end

function forward_differentiate!(model::Optimizer)
    st = MOI.get(model.optimizer, MOI.TerminationStatus())
    if !in(st,  (MOI.LOCALLY_SOLVED, MOI.OPTIMAL))
        error("Trying to compute the forward differentiation on a model with termination status $(st)")
    end
    diff = _diff(model)
    if model.input_cache.objective !== nothing
        MOI.set(diff, ForwardObjectiveFunction(), MOI.Utilities.map_indices(model.index_map, model.input_cache.objective))
    end
    for (F, S) in keys(model.input_cache.scalar_constraints.dict)
        _copy_forward_in_constraint(diff, model.index_map, model.index_map.con_map[F, S], model.input_cache.scalar_constraints[F, S])
    end
    for (F, S) in keys(model.input_cache.vector_constraints.dict)
        _copy_forward_in_constraint(diff, model.index_map, model.index_map.con_map[F, S], model.input_cache.vector_constraints[F, S])
    end
    forward_differentiate!(diff)
end

function _copy_constraint_start(dest, src, index_map::MOIU.DoubleDicts.IndexDoubleDictInner{F, S}, dest_attr, src_attr) where {F, S}
    for ci in MOI.get(src, MOI.ListOfConstraintIndices{F, S}())
        value = MOI.get(src, src_attr, ci)
        MOI.set(dest, dest_attr, index_map[ci], value)
    end
end

function _instantiate_with_bridges(model_constructor)
    model = MOI.Bridges.LazyBridgeOptimizer(
        MOI.instantiate(model_constructor)
    )
    # We don't add any variable bridge here because:
    # 1) If `ZerosBridge` is used, `MOI.Bridges.unbridged_function` does not work.
    #    This is in fact expected: since `ZerosBridge` drops the variable, we dont
    #    compute the derivative of the value of this variable as a function of its fixed value.
    #    This could be easily determined as the same as the derivative of the value but
    #    since the variable was also dropped from other constraints, we would ignore its impact on the other constraints.
    # 2) For affine variable bridges, `bridged_function` and `unbridged_function` don't treat the function as a derivative hence they will add constants
    MOI.Bridges.Constraint.add_all_bridges(model, Float64)
    MOI.Bridges.Objective.add_all_bridges(model, Float64)
    return model
end

function _diff(model::Optimizer)
    if model.diff === nothing
        _check_termination_status(model)
        model_constructor = MOI.get(model, ModelConstructor())
        if isnothing(model_constructor)
            model.diff = nothing
            for constructor in model.model_constructors
                model.diff = _instantiate_with_bridges(constructor)
                try
                    model.index_map = MOI.copy_to(model.diff, model.optimizer)
                catch err
                    if err isa MOI.UnsupportedConstraint || err isa MOI.UnsupportedAttribute
                        model.diff = nothing
                    else
                        rethrow(err)
                    end
                end
                if !isnothing(model.diff)
                    break
                end
            end
            if isnothing(model.diff)
                error("No differentiation model supports the problem. If you believe it should be supported, say by `DiffOpt.QuadraticProgram.Model`, use `MOI.set(model, DiffOpt.ModelConstructor, DiffOpt.QuadraticProgram.Model)` and try again to see an error indicating why it is not supported.")
            end
        else
            model.diff = _instantiate_with_bridges(model_constructor)
            model.index_map = MOI.copy_to(model.diff, model.optimizer)
        end
        vis_src = MOI.get(model.optimizer, MOI.ListOfVariableIndices())
        MOI.set(model.diff, MOI.VariablePrimalStart(), getindex.(Ref(model.index_map), vis_src), MOI.get(model.optimizer, MOI.VariablePrimal(), vis_src))
        for (F, S) in MOI.get(model.optimizer, MOI.ListOfConstraintTypesPresent())
            _copy_constraint_start(model.diff, model.optimizer, model.index_map.con_map[F, S], MOI.ConstraintPrimalStart(), MOI.ConstraintPrimal())
            _copy_constraint_start(model.diff, model.optimizer, model.index_map.con_map[F, S], MOI.ConstraintDualStart(), MOI.ConstraintDual())
        end
    end
    return model.diff
end

function _check_termination_status(model::Optimizer)
    if !in(
        MOI.get(model, MOI.TerminationStatus()), (MOI.LOCALLY_SOLVED, MOI.OPTIMAL)
        )
        error("problem status: ", MOI.get(model.optimizer, MOI.TerminationStatus()))
    end
end

# DiffOpt attributes redirected to `diff`

function _checked_diff(model::Optimizer, attr::MOI.AnyAttribute, call)
    if model.diff === nothing
        error("Cannot get attribute `$attr`. First call `DiffOpt.$call`.")
    end
    return model.diff
end

function MOI.get(model::Optimizer, attr::ReverseObjectiveFunction)
    return IndexMappedFunction(
        MOI.get(_checked_diff(model, attr, :reverse_differentiate!), attr),
        model.index_map,
    )
end
MOI.supports(::Optimizer, ::ForwardObjectiveFunction) = true
function MOI.get(model::Optimizer, ::ForwardObjectiveFunction)
    return model.input_cache.objective
end
function MOI.set(model::Optimizer, ::ForwardObjectiveFunction, objective)
    model.input_cache.objective = objective
    return
end

function MOI.get(model::Optimizer, attr::ForwardVariablePrimal, vi::MOI.VariableIndex)
    return MOI.get(_checked_diff(model, attr, :forward_differentiate!), attr, model.index_map[vi])
end
MOI.supports(::Optimizer, ::ReverseVariablePrimal, ::Type{VI}) = true
function MOI.get(model::Optimizer, ::ReverseVariablePrimal, vi::VI)
    return get(model.input_cache.dx, vi, 0.0)
end
function MOI.set(model::Optimizer, ::ReverseVariablePrimal, vi::VI, val)
    model.input_cache.dx[vi] = val
    return
end

function MOI.get(model::Optimizer, attr::ReverseConstraintFunction, ci::MOI.ConstraintIndex)
    return IndexMappedFunction(
        MOI.get(_checked_diff(model, attr, :reverse_differentiate!), attr, model.index_map[ci]),
        model.index_map,
    )
end
function MOI.get(model::Optimizer,
    ::ForwardConstraintFunction, ci::CI{MOI.ScalarAffineFunction{T},S}
) where {T,S}
    return get(model.input_cache.scalar_constraints, ci, zero(MOI.ScalarAffineFunction{T}))
end
function MOI.supports(
    ::Optimizer,
    ::ForwardConstraintFunction,
    ::Type{CI{F,S}}
) where {F<:Union{MOI.ScalarAffineFunction,MOI.VectorAffineFunction},S}
    return true
end
function MOI.get(model::Optimizer,
    ::ForwardConstraintFunction, ci::CI{MOI.VectorAffineFunction{T},S}
) where {T,S}
    func = get(model.input_cache.vector_constraints, ci, nothing)
    if func === nothing
        set = MOI.get(model, MOI.ConstraintSet(), ci)
        dim = MOI.dimension(set)
        return MOI.Utilities.zero_with_output_dimension(MOI.VectorAffineFunction{T}, dim)
    else
        return func
    end
end
function MOI.set(model::Optimizer,
    ::ForwardConstraintFunction,
    ci::CI{MOI.ScalarAffineFunction{T},S},
    func::MOI.ScalarAffineFunction{T},
) where {T,S}
    model.input_cache.scalar_constraints[ci] = func
    return
end
function MOI.set(model::Optimizer,
    ::ForwardConstraintFunction,
    ci::CI{MOI.VectorAffineFunction{T},S},
    func::MOI.VectorAffineFunction{T},
) where {T,S}
    model.input_cache.vector_constraints[ci] = func
    return
end
MOI.get(model::Optimizer, attr::DifferentiateTimeSec) = MOI.get(model.diff, attr)
