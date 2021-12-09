"""
    diff_optimizer(optimizer_constructor)::Optimizer

Creates a `DiffOpt.Optimizer`, which is an MOI layer with an internal optimizer
and other utility methods. Results (primal, dual and slack values) are obtained
by querying the internal optimizer instantiated using the
`optimizer_constructor`. These values are required for find jacobians with respect to problem data.

One define a differentiable model by using any solver of choice. Example:

```julia
julia> import DiffOpt, GLPK

julia> model = DiffOpt.diff_optimizer(GLPK.Optimizer)
julia> model.add_variable(x)
julia> model.add_constraint(...)

julia> _backward_quad(model)  # for convex quadratic models

julia> _backward_quad(model)  # for convex conic models
```
"""
function diff_optimizer(optimizer_constructor)::Optimizer
    return Optimizer(MOI.instantiate(optimizer_constructor, with_bridge_type=Float64))
end

mutable struct Optimizer{OT <: MOI.ModelLike} <: MOI.AbstractOptimizer
    optimizer::OT

    program_class::ProgramClassCode

    diff::Union{Nothing,QPDiff,ConicDiff}

    index_map::Union{Nothing,MOI.Utilities.IndexMap}

    # sensitivity input cache using MOI like sparse format
    input_cache::DiffInputCache

    function Optimizer(optimizer::OT) where {OT <: MOI.ModelLike}
        new{OT}(
            optimizer,
            AUTOMATIC,
            nothing,
            nothing,
            DiffInputCache(),
        )
    end
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

function MOI.add_constraint(model::Optimizer, f::SUPPORTED_SCALAR_FUNCTIONS, s::SUPPORTED_SCALAR_SETS)
    model.diff = nothing
    return MOI.add_constraint(model.optimizer, f, s)
end

function MOI.add_constraint(model::Optimizer, vf::SUPPORTED_VECTOR_FUNCTIONS, s::SUPPORTED_VECTOR_SETS)
    model.diff = nothing
    return MOI.add_constraint(model.optimizer, vf, s)
end

function MOI.add_constraints(model::Optimizer, f::AbstractVector{F}, s::AbstractVector{S}) where {F<:SUPPORTED_SCALAR_FUNCTIONS, S <: SUPPORTED_SCALAR_SETS}
    model.diff = nothing
    return CI{F, S}[MOI.add_constraint(model, f[i], s[i]) for i in eachindex(f)]
end

function MOI.set(model::Optimizer, attr::MOI.ObjectiveFunction{<: SUPPORTED_OBJECTIVES}, f::SUPPORTED_OBJECTIVES)
    model.diff = nothing
    MOI.set(model.optimizer, attr, f)
end

MOI.supports(::Optimizer, ::MOI.ObjectiveSense) = true

function MOI.set(model::Optimizer, attr::MOI.ObjectiveSense, sense::MOI.OptimizationSense)
    model.diff = nothing
    return MOI.set(model.optimizer, attr, sense)
end

function MOI.get(model::Optimizer, attr::MOI.AbstractModelAttribute)
    return MOI.get(model.optimizer, attr)
end

function MOI.get(model::Optimizer, attr::MOI.ListOfConstraintIndices{F, S}) where {F<:SUPPORTED_SCALAR_FUNCTIONS, S<:SUPPORTED_SCALAR_SETS}
    return MOI.get(model.optimizer, attr)
end

function MOI.get(model::Optimizer, attr::MOI.ConstraintSet, ci::CI{F, S}) where {F<:SUPPORTED_SCALAR_FUNCTIONS, S<:SUPPORTED_SCALAR_SETS}
    return MOI.get(model.optimizer, attr, ci)
end

function MOI.set(model::Optimizer, attr::MOI.ConstraintSet, ci::CI{F, S}, s::S) where {F<:SUPPORTED_SCALAR_FUNCTIONS, S<:SUPPORTED_SCALAR_SETS}
    model.diff = nothing
    return MOI.set(model.optimizer, attr, ci, s)
end

function MOI.get(model::Optimizer, attr::MOI.ConstraintFunction, ci::CI{F, S}) where {F<:SUPPORTED_SCALAR_FUNCTIONS, S<:SUPPORTED_SCALAR_SETS}
    return MOI.get(model.optimizer, attr, ci)
end

function MOI.set(model::Optimizer, attr::MOI.ConstraintFunction, ci::CI{F, S}, f::F) where {F<:SUPPORTED_SCALAR_FUNCTIONS, S<:SUPPORTED_SCALAR_SETS}
    model.diff = nothing
    return MOI.set(model.optimizer, attr, ci, f)
end

function MOI.get(model::Optimizer, attr::MOI.ListOfConstraintIndices{F, S}) where {F<:SUPPORTED_VECTOR_FUNCTIONS, S<:SUPPORTED_VECTOR_SETS}
    return MOI.get(model.optimizer, attr)
end

function MOI.get(model::Optimizer, attr::MOI.ConstraintSet, ci::CI{F, S}) where {F<:SUPPORTED_VECTOR_FUNCTIONS, S<:SUPPORTED_VECTOR_SETS}
    return MOI.get(model.optimizer, attr, ci)
end

function MOI.set(model::Optimizer, attr::MOI.ConstraintSet, ci::CI{F, S}, s::S) where {F<:SUPPORTED_VECTOR_FUNCTIONS, S<:SUPPORTED_VECTOR_SETS}
    model.diff = nothing
    return MOI.set(model.optimizer, attr, ci, s)
end

function MOI.get(model::Optimizer, attr::MOI.ConstraintFunction, ci::CI{F, S}) where {F<:SUPPORTED_VECTOR_FUNCTIONS, S<:SUPPORTED_VECTOR_SETS}
    return MOI.get(model.optimizer, attr, ci)
end

function MOI.set(model::Optimizer, attr::MOI.ConstraintFunction, ci::CI{F, S}, f::F) where {F<:SUPPORTED_VECTOR_FUNCTIONS, S<:SUPPORTED_VECTOR_SETS}
    model.diff = nothing
    return MOI.set(model.optimizer, attr, ci, f)
end

# `MOI.supports` methods

function MOI.supports(::Optimizer,
    ::MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}})
    return true
end

function MOI.supports(::Optimizer,
    ::MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}})
    return true
end

function MOI.supports(::Optimizer, ::MOI.AbstractModelAttribute)
    return true
end

function MOI.supports(::Optimizer, ::MOI.ObjectiveFunction)
    return false
end

function MOI.supports(::Optimizer, ::MOI.ObjectiveFunction{<: SUPPORTED_OBJECTIVES})
    return true
end

MOI.supports_constraint(::Optimizer, ::Type{<: SUPPORTED_SCALAR_FUNCTIONS}, ::Type{<: SUPPORTED_SCALAR_SETS}) = true
MOI.supports_constraint(::Optimizer, ::Type{<: SUPPORTED_VECTOR_FUNCTIONS}, ::Type{<: SUPPORTED_VECTOR_SETS}) = true
function MOI.supports(model::Optimizer, attr::MOI.ConstraintName, ::Type{CI{F, S}}) where {F<:SUPPORTED_SCALAR_FUNCTIONS, S<:SUPPORTED_SCALAR_SETS}
    return MOI.supports(model.optimizer, attr, CI{F,S})
end
function MOI.supports(model::Optimizer, attr::MOI.ConstraintName, ::Type{CI{F, S}}) where {F<:SUPPORTED_VECTOR_FUNCTIONS, S<:SUPPORTED_VECTOR_SETS}
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

# now supports name too
MOI.supports_incremental_interface(::Optimizer) = true

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

function MOI.delete(model::Optimizer, ci::CI{F,S}) where {F <: SUPPORTED_SCALAR_FUNCTIONS, S <: SUPPORTED_SCALAR_SETS}
    model.diff = nothing
    MOI.delete(model.optimizer, ci)
end

function MOI.delete(model::Optimizer, ci::CI{F,S}) where {F <: SUPPORTED_VECTOR_FUNCTIONS, S <: SUPPORTED_VECTOR_SETS}
    model.diff = nothing
    MOI.delete(model.optimizer, ci)
end

function MOI.get(model::Optimizer, attr::MOI.ConstraintPrimal, ci::CI{F,S}) where {F <: SUPPORTED_SCALAR_FUNCTIONS, S <: SUPPORTED_SCALAR_SETS}
    return MOI.get(model.optimizer, attr, ci)
end

function MOI.get(model::Optimizer, attr::MOI.ConstraintPrimal, ci::CI{F,S}) where {F <: SUPPORTED_VECTOR_FUNCTIONS, S <: SUPPORTED_VECTOR_SETS}
    return MOI.get(model.optimizer, attr, ci)
end

function MOI.is_valid(model::Optimizer, v::VI)
    return MOI.is_valid(model.optimizer, v::VI)
end

MOI.is_valid(model::Optimizer, con::CI) = MOI.is_valid(model.optimizer, con)

function MOI.get(model::Optimizer, attr::MOI.ConstraintDual, ci::CI{F,S}) where {F <: SUPPORTED_SCALAR_FUNCTIONS, S <: SUPPORTED_SCALAR_SETS}
    return MOI.get(model.optimizer, attr, ci)
end

function MOI.get(model::Optimizer, attr::MOI.ConstraintDual, ci::CI{F,S}) where {F <: SUPPORTED_VECTOR_FUNCTIONS, S <: SUPPORTED_VECTOR_SETS}
    return MOI.get(model.optimizer, attr, ci)
end

function MOI.get(model::Optimizer, ::MOI.ConstraintBasisStatus, ci::CI{F,S}) where {F <: SUPPORTED_SCALAR_FUNCTIONS, S <: SUPPORTED_SCALAR_SETS}
    return MOI.get(model.optimizer, MOI.ConstraintBasisStatus(), ci)
end

# helper methods to check if a constraint contains a Variable
function _constraint_contains(model::Optimizer, v::VI, ci::CI{MOI.VariableIndex, S}) where {S <: SUPPORTED_SCALAR_SETS}
    return v == MOI.get(model, MOI.ConstraintFunction(), ci)
end

function _constraint_contains(model::Optimizer, v::VI, ci::CI{MOI.ScalarAffineFunction{Float64}, S}) where {S <: SUPPORTED_SCALAR_SETS}
    func = MOI.get(model, MOI.ConstraintFunction(), ci)
    return any(term -> v == term.variable, func.terms)
end

function _constraint_contains(model::Optimizer, v::VI, ci::CI{MOI.VectorOfVariables, S}) where {S <: SUPPORTED_VECTOR_SETS}
    func = MOI.get(model, MOI.ConstraintFunction(), ci)
    return v in func.variables
end

function _constraint_contains(model::Optimizer, v::VI, ci::CI{MOI.VectorAffineFunction{Float64}, S}) where {S <: SUPPORTED_VECTOR_SETS}
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

function MOI.modify(model::Optimizer, ci::CI{F, S}, chg::MOI.AbstractFunctionModification) where {F<:SUPPORTED_SCALAR_FUNCTIONS, S <: SUPPORTED_SCALAR_SETS}
    model.diff = nothing
    MOI.modify(model.optimizer, ci, chg)
end

function MOI.modify(model::Optimizer, ci::CI{F, S}, chg::MOI.AbstractFunctionModification) where {F<:MOI.VectorAffineFunction{Float64}, S <: SUPPORTED_VECTOR_SETS}
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

function MOI.get(model::Optimizer, ::MOI.ConstraintName, ::Type{CI{F, S}}) where {F<:SUPPORTED_SCALAR_FUNCTIONS, S<:SUPPORTED_SCALAR_SETS}
    return MOI.get(model.optimizer, MOI.ConstraintName(), CI{F,S})
end

function MOI.get(model::Optimizer, ::MOI.ConstraintName, ::Type{CI{VF, VS}}) where {VF<:SUPPORTED_VECTOR_FUNCTIONS, VS<:SUPPORTED_VECTOR_SETS}
    return MOI.get(model.optimizer, MOI.ConstraintName(), CI{VF,VS})
end

function MOI.set(model::Optimizer, ::MOI.Name, name::String)
    MOI.set(model.optimizer, MOI.Name(), name)
end

function MOI.get(model::Optimizer, ::Type{CI{F, S}}, name::String) where {F<:SUPPORTED_SCALAR_FUNCTIONS, S<:SUPPORTED_SCALAR_SETS}
    return MOI.get(model.optimizer, CI{F,S}, name)
end

function MOI.get(model::Optimizer, ::Type{CI{VF, VS}}, name::String) where {VF<:SUPPORTED_VECTOR_FUNCTIONS, VS<:SUPPORTED_VECTOR_SETS}
    return MOI.get(model.optimizer, CI{VF,VS}, name)
end

MOI.supports(::Optimizer, ::MOI.TimeLimitSec) = true
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

function MOI.get(model::Optimizer, ::MOI.SolverName)
    return MOI.get(model.optimizer, MOI.SolverName())
end

function MOI.optimize!(model::Optimizer)
    model.diff = nothing
    MOI.optimize!(model.optimizer)

    # do not fail. interferes with MOI.Tests.linear12test
    if !in(MOI.get(model.optimizer, MOI.TerminationStatus()),  (MOI.LOCALLY_SOLVED, MOI.OPTIMAL))
        @warn "problem status: $(MOI.get(model.optimizer, MOI.TerminationStatus()))"
        return
    end

    return
end

"""
    ProgramClass <: MOI.AbstractOptimizerAttribute

Determines which program class to used from [`ProgramClassCode`](@ref). The
default is `AUTOMATIC`.

One important advantage of setting the class explicitly is that it will allow
necessary bridges to be used. If the class is `AUTOMATIC` then
`DiffOpt.Optimizer` will report that it supports both objective and constraints
of the QP and CP classes. For instance, it will reports that is supports both
quadratic objective and conic constraints. However, at the differentiation
stage, we won't be able to differentiate since QP does not support conic
constraints and CP does not support quadratic objective. On the other hand, if
the `ProgramClass` is set to `CONIC` then `DiffOpt.Optimizer` will report that
it does not support quadratic objective hence it will be bridged to second-order
cone constraints and we will be able to use CP to differentiate.
"""
struct ProgramClass <: MOI.AbstractOptimizerAttribute end

MOI.supports(::Optimizer, ::ProgramClass) = true
MOI.get(model::Optimizer, ::ProgramClass) = model.program_class
function MOI.set(model::Optimizer, ::ProgramClass, class::ProgramClassCode)
    model.program_class = class
end


"""
    ProgramClassUsed <: MOI.AbstractOptimizerAttribute

Program class actually used, same as [`ProgramClass`](@ref) except that it does
not return `AUTOMATIC` but the class automatically chosen instead. This attribute
is read-only, it cannot be set, set [`ProgramClass`](@ref) instead.
"""
struct ProgramClassUsed <: MOI.AbstractOptimizerAttribute end

function MOI.get(model::Optimizer, ::ProgramClassUsed)
    if model.program_class == AUTOMATIC
        if _qp_supported(model.optimizer)
            return QUADRATIC
        else
            return CONIC
        end
    else
        return model.program_class
    end
end



"""
    backward(model::Optimizer)

Wrapper method for the backward pass.
This method will consider as input a currently solved problem and differentials
with respect to the solution set with the [`BackwardInVariablePrimal`](@ref) attribute.
The output problem data differentials can be queried with the
attributes [`BackwardOutObjective`](@ref) and [`BackwardOutConstraint`](@ref).
"""
function backward(model::Optimizer)
    diff = _diff(model)
    for (vi, value) in model.input_cache.dx
        MOI.set(diff, BackwardInVariablePrimal(), model.index_map[vi], value)
    end
    backward(diff)
end

function _copy_forward_in_constraint(diff, index_map, con_map, constraints)
    for (index, value) in constraints
        MOI.set(diff, ForwardInConstraint(), con_map[index], MOI.Utilities.map_indices(index_map, value))
    end
end

"""
    forward(model::Optimizer)

Wrapper method for the forward pass.
This method will consider as input a currently solved problem and
differentials with respect to problem data set with
the [`ForwardInObjective`](@ref) and  [`ForwardInConstraint`](@ref) attributes.
The output solution differentials can be queried with the attribute
[`ForwardOutVariablePrimal`](@ref).
"""
function forward(model::Optimizer)
    diff = _diff(model)
    if model.input_cache.objective !== nothing
        MOI.set(diff, ForwardInObjective(), MOI.Utilities.map_indices(model.index_map, model.input_cache.objective))
    end
    for (F, S) in keys(model.input_cache.scalar_constraints.dict)
        _copy_forward_in_constraint(diff, model.index_map, model.index_map.con_map[F, S], model.input_cache.scalar_constraints[F, S])
    end
    for (F, S) in keys(model.input_cache.vector_constraints.dict)
        _copy_forward_in_constraint(diff, model.index_map, model.index_map.con_map[F, S], model.input_cache.vector_constraints[F, S])
    end
    forward(diff)
end

function _diff(model::Optimizer)
    if model.diff === nothing
        if MOI.get(model, ProgramClassUsed()) == QUADRATIC
            model.diff = QPDiff()
        else
            _check_termination_status(model)
            model.diff = ConicDiff()
        end
        model.index_map = MOI.copy_to(model.diff, model.optimizer)
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
        error("Cannot get attribute `attr`. First call `DiffOpt.$call`.")
    end
    return model.diff
end

function MOI.get(model::Optimizer, attr::BackwardOutObjective)
    return IndexMappedFunction(
        MOI.get(_checked_diff(model, attr, :backward), attr),
        model.index_map,
    )
end
function MOI.get(model::Optimizer, ::ForwardInObjective)
    return model.input_cache.objective
end
function MOI.set(model::Optimizer, ::ForwardInObjective, objective)
    model.input_cache.objective = objective
    return
end

function MOI.get(model::Optimizer, attr::ForwardOutVariablePrimal, vi::MOI.VariableIndex)
    return MOI.get(_checked_diff(model, attr, :forward), attr, model.index_map[vi])
end
function MOI.get(model::Optimizer, ::BackwardInVariablePrimal, vi::VI)
    return get(model.input_cache.dx, vi, 0.0)
end
function MOI.set(model::Optimizer, ::BackwardInVariablePrimal, vi::VI, val)
    model.input_cache.dx[vi] = val
    return
end

function MOI.get(model::Optimizer, attr::BackwardOutConstraint, ci::MOI.ConstraintIndex)
    return IndexMappedFunction(
        MOI.get(_checked_diff(model, attr, :backward), attr, model.index_map[ci]),
        model.index_map,
    )
end
function MOI.get(model::Optimizer,
    ::ForwardInConstraint, ci::CI{MOI.ScalarAffineFunction{T},S}
) where {T,S}
    return get(model.input_cache.scalar_constraints, ci, zero(MOI.ScalarAffineFunction{T}))
end
function MOI.get(model::Optimizer,
    ::ForwardInConstraint, ci::CI{MOI.VectorAffineFunction{T},S}
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
    ::ForwardInConstraint,
    ci::CI{MOI.ScalarAffineFunction{T},S},
    func::MOI.ScalarAffineFunction{T},
) where {T,S}
    model.input_cache.scalar_constraints[ci] = func
    return
end
function MOI.set(model::Optimizer,
    ::ForwardInConstraint,
    ci::CI{MOI.VectorAffineFunction{T},S},
    func::MOI.VectorAffineFunction{T},
) where {T,S}
    model.input_cache.vector_constraints[ci] = func
    return
end
