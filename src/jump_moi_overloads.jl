# Copyright (c) 2020: Akshay Sharma and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

# FIXME
# Some function in this file are overloads to skip JuMP dirty state.
# Workaround for https://github.com/jump-dev/JuMP.jl/issues/2797
# This workaround is necessary because once some attributes are set the JuMP
# model changes to a dirty state, then getting some attributes is blocked.
# However, getting and setting forward and backward sensitivities is
# done after the model is optimized, so we add function to bypass the
# dirty state.

function MOI.set(
    model::JuMP.Model,
    attr::ForwardObjectiveFunction,
    func::JuMP.AbstractJuMPScalar,
)
    JuMP.check_belongs_to_model(func, model)
    return MOI.set(model, attr, JuMP.moi_function(func))
end

function MOI.set(
    model::JuMP.Model,
    attr::MFactorization,
    factorization::Function,
)
    return MOI.set(JuMP.backend(model), attr, factorization)
end

function MOI.set(
    model::JuMP.Model,
    attr::ForwardObjectiveFunction,
    func::Number,
)
    return MOI.set(model, attr, JuMP.AffExpr(func))
end

function MOI.set(
    model::JuMP.Model,
    attr::ForwardConstraintFunction,
    con_ref::JuMP.ConstraintRef,
    func::JuMP.AbstractJuMPScalar,
)
    JuMP.check_belongs_to_model(func, model)
    return MOI.set(model, attr, con_ref, JuMP.moi_function(func))
end

function MOI.set(
    model::JuMP.Model,
    attr::ForwardConstraintFunction,
    con_ref::JuMP.ConstraintRef,
    func::Number,
)
    return MOI.set(model, attr, con_ref, JuMP.AffExpr(func))
end

function MOI.get(
    model::JuMP.Model,
    attr::ForwardConstraintDual,
    con_ref::JuMP.ConstraintRef,
)
    JuMP.check_belongs_to_model(con_ref, model)
    moi_func = MOI.get(JuMP.backend(model), attr, JuMP.index(con_ref))
    return JuMP.jump_function(model, moi_func)
end

function MOI.get(model::JuMP.Model, attr::ReverseObjectiveFunction)
    func = MOI.get(JuMP.backend(model), attr)
    return JuMP.jump_function(model, func)
end

function MOI.get(
    model::JuMP.Model,
    attr::ReverseConstraintFunction,
    con_ref::JuMP.ConstraintRef,
)
    JuMP.check_belongs_to_model(con_ref, model)
    moi_func = MOI.get(JuMP.backend(model), attr, JuMP.index(con_ref))
    return JuMP.jump_function(model, moi_func)
end

# see FIXME comment in the top of the file
function _moi_get_result(model::MOI.ModelLike, args...)
    if MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMIZE_NOT_CALLED
        throw(OptimizeNotCalled())
    end
    return MOI.get(model, args...)
end

function _moi_get_result(model::MOI.Utilities.CachingOptimizer, args...)
    if MOI.Utilities.state(model) == MOI.Utilities.NO_OPTIMIZER
        throw(NoOptimizer())
    elseif MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMIZE_NOT_CALLED
        throw(OptimizeNotCalled())
    end
    return MOI.get(model, args...)
end

function MOI.get(
    model::JuMP.Model,
    attr::ForwardVariablePrimal,
    var_ref::JuMP.VariableRef,
)
    JuMP.check_belongs_to_model(var_ref, model)
    return _moi_get_result(JuMP.backend(model), attr, JuMP.index(var_ref))
end

function MOI.get(
    model::JuMP.Model,
    attr::ReverseConstraintSet,
    var_ref::JuMP.ConstraintRef,
)
    JuMP.check_belongs_to_model(var_ref, model)
    return _moi_get_result(JuMP.backend(model), attr, JuMP.index(var_ref))
end

function MOI.set(
    model::JuMP.Model,
    attr::ForwardConstraintSet,
    con_ref::JuMP.ConstraintRef,
    set::MOI.AbstractScalarSet,
)
    JuMP.check_belongs_to_model(con_ref, model)
    return MOI.set(JuMP.backend(model), attr, JuMP.index(con_ref), set)
end

function MOI.set(
    model::JuMP.Model,
    attr::ForwardConstraintSet,
    con_ref::JuMP.ConstraintRef,
    set::JuMP.AbstractScalarSet,
)
    JuMP.check_belongs_to_model(con_ref, model)
    return MOI.set(model, attr, con_ref, JuMP.moi_set(set))
end

"""
    abstract type AbstractLazyScalarFunction <: MOI.AbstractScalarFunction end

Subtype of `MOI.AbstractScalarFunction` that is not a standard MOI scalar
function but can be converted to one using [`standard_form`](@ref).

The function can also be inspected lazily using `JuMP.coefficient` or
[`quad_sym_half`](@ref).
"""
abstract type AbstractLazyScalarFunction <: MOI.AbstractScalarFunction end

"""
    standard_form(func::AbstractLazyScalarFunction)

Converts `func` to a standard MOI scalar function.

    standard_form(func::MOItoJuMP)

Converts `func` to a standard JuMP scalar function.
"""
function standard_form end

"""
    quad_sym_half(func, vi1::MOI.VariableIndex, vi2::MOI.VariableIndex)

Return `Q[i,j] = Q[j,i]` where the quadratic terms of `func` is represented
by `x' Q x / 2` for a symmetric matrix `Q` where `x[i] = vi1` and `x[j] = vi2`.
Note that while this is equal to `JuMP.coefficient(func, vi1, vi2)` if `vi1 != vi2`,
in the case `vi1 == vi2`, it is rather equal to
`2JuMP.coefficient(func, vi1, vi2)`.
"""
function quad_sym_half end

function standard_form(
    func::Union{
        MOI.VariableIndex,
        MOI.ScalarAffineFunction,
        MOI.ScalarQuadraticFunction,
    },
)
    return func
end

function Base.isapprox(
    func1::AbstractLazyScalarFunction,
    func2::MOI.AbstractScalarFunction;
    kws...,
)
    return isapprox(standard_form(func1), standard_form(func2); kws...)
end

function quad_sym_half(
    func::MatrixScalarQuadraticFunction,
    vi1::MOI.VariableIndex,
    vi2::MOI.VariableIndex,
)
    return func.terms[vi1.value, vi2.value]
end

function JuMP.coefficient(
    func::MatrixScalarQuadraticFunction,
    vi1::MOI.VariableIndex,
    vi2::MOI.VariableIndex,
)
    coef = quad_sym_half(func, vi1, vi2)
    if vi1 == vi2
        return coef / 2
    else
        return coef
    end
end

function Base.convert(
    ::Type{MOI.ScalarQuadraticFunction{T}},
    func::MatrixScalarQuadraticFunction,
) where {T}
    n = length(func.affine.terms)
    aff = convert(MOI.ScalarAffineFunction{T}, func.affine)
    quad = MOI.ScalarQuadraticTerm{T}[
        MOI.ScalarQuadraticTerm{T}(
            quad_sym_half(func, MOI.VariableIndex(i), MOI.VariableIndex(j)),
            MOI.VariableIndex(i),
            MOI.VariableIndex(j),
        ) for j in 1:n for i in 1:j if !iszero(
            quad_sym_half(func, MOI.VariableIndex(i), MOI.VariableIndex(j)),
        )
    ]
    return MOI.ScalarQuadraticFunction{T}(quad, aff.terms, aff.constant)
end

function standard_form(func::MatrixScalarQuadraticFunction{T}) where {T}
    return convert(MOI.ScalarQuadraticFunction{T}, func)
end

"""
    IndexMappedFunction{F<:MOI.AbstractFunction} <: AbstractLazyScalarFunction

Lazily represents the function `MOI.Utilities.map_indices(index_map, DiffOpt.standard_form(func))`.
"""
struct IndexMappedFunction{F<:MOI.AbstractFunction} <:
       AbstractLazyScalarFunction
    func::F
    index_map::MOI.Utilities.IndexMap
end

MOI.constant(func::IndexMappedFunction) = MOI.constant(func.func)

function JuMP.coefficient(func::IndexMappedFunction, vi::MOI.VariableIndex)
    return JuMP.coefficient(func.func, func.index_map[vi])
end

function quad_sym_half(
    func::IndexMappedFunction,
    vi1::MOI.VariableIndex,
    vi2::MOI.VariableIndex,
)
    return quad_sym_half(func.func, func.index_map[vi1], func.index_map[vi2])
end

function JuMP.coefficient(
    func::IndexMappedFunction,
    vi1::MOI.VariableIndex,
    vi2::MOI.VariableIndex,
)
    return JuMP.coefficient(func.func, func.index_map[vi1], func.index_map[vi2])
end

function standard_form(func::IndexMappedFunction)
    return MOI.Utilities.map_indices(func.index_map, standard_form(func.func))
end

function MOI.Utilities.isapprox_zero(func::IndexMappedFunction, tol)
    return MOI.Utilities.isapprox_zero(func.func, tol)
end

function MOI.Utilities.map_indices(
    index_map::MOI.Utilities.IndexMap,
    func::AbstractLazyScalarFunction,
)
    return IndexMappedFunction(func, index_map)
end

"""
    MOItoJuMP{F<:MOI.AbstractScalarFunction} <: JuMP.AbstractJuMPScalar

Lazily represents the function `JuMP.jump_function(model, DiffOpt.standard_form(func))`.
"""
struct MOItoJuMP{F<:MOI.AbstractScalarFunction} <: JuMP.AbstractJuMPScalar
    model::JuMP.Model
    func::F
end

Base.broadcastable(func::MOItoJuMP) = Ref(func)

JuMP.constant(func::MOItoJuMP) = MOI.constant(func.func)

function JuMP.coefficient(func::MOItoJuMP, var_ref::JuMP.VariableRef)
    check_belongs_to_model(var_ref, func.model)
    return JuMP.coefficient(func.func, JuMP.index(var_ref))
end

function quad_sym_half(
    func::MOItoJuMP,
    var1_ref::JuMP.VariableRef,
    var2_ref::JuMP.VariableRef,
)
    check_belongs_to_model.([var1_ref, var2_ref], Ref(func.model))
    return quad_sym_half(func.func, JuMP.index(var1_ref), JuMP.index(var2_ref))
end

function JuMP.coefficient(
    func::MOItoJuMP,
    var1_ref::JuMP.VariableRef,
    var2_ref::JuMP.VariableRef,
)
    check_belongs_to_model.([var1_ref, var2_ref], Ref(func.model))
    return JuMP.coefficient(
        func.func,
        JuMP.index(var1_ref),
        JuMP.index(var2_ref),
    )
end

function Base.convert(
    ::Type{JuMP.GenericAffExpr{T,JuMP.VariableRef}},
    func::MOItoJuMP,
) where {T}
    return JuMP.GenericAffExpr{T,JuMP.VariableRef}(
        func.model,
        convert(MOI.ScalarAffineFunction{T}, func.func),
    )
end

function Base.convert(
    ::Type{JuMP.GenericQuadExpr{T,JuMP.VariableRef}},
    func::MOItoJuMP,
) where {T}
    return JuMP.GenericQuadExpr{T,JuMP.VariableRef}(
        func.model,
        convert(MOI.ScalarQuadraticFunction{T}, func.func),
    )
end

JuMP.moi_function(func::MOItoJuMP) = func.func

function JuMP.jump_function(model::JuMP.Model, func::AbstractLazyScalarFunction)
    return MOItoJuMP(model, func)
end

function standard_form(func::MOItoJuMP)
    return JuMP.jump_function(func.model, standard_form(func.func))
end

function JuMP.function_string(mode, func::MOItoJuMP)
    return JuMP.function_string(mode, standard_form(func))
end

# JuMP

function reverse_differentiate!(model::JuMP.Model; kwargs...)
    return reverse_differentiate!(JuMP.backend(model); kwargs...)
end

function forward_differentiate!(model::JuMP.Model; kwargs...)
    return forward_differentiate!(JuMP.backend(model); kwargs...)
end

function empty_input_sensitivities!(model::JuMP.Model)
    empty_input_sensitivities!(JuMP.backend(model))
    return
end

# MOI.Utilities

function reverse_differentiate!(
    model::MOI.Utilities.CachingOptimizer;
    kwargs...,
)
    return reverse_differentiate!(model.optimizer; kwargs...)
end

function forward_differentiate!(
    model::MOI.Utilities.CachingOptimizer;
    kwargs...,
)
    return forward_differentiate!(model.optimizer; kwargs...)
end

function empty_input_sensitivities!(model::MOI.Utilities.CachingOptimizer)
    empty_input_sensitivities!(model.optimizer)
    return
end

# MOIB

function reverse_differentiate!(
    model::MOI.Bridges.AbstractBridgeOptimizer;
    kwargs...,
)
    return reverse_differentiate!(model.model; kwargs...)
end

function forward_differentiate!(
    model::MOI.Bridges.AbstractBridgeOptimizer;
    kwargs...,
)
    return forward_differentiate!(model.model; kwargs...)
end

function empty_input_sensitivities!(model::MOI.Bridges.AbstractBridgeOptimizer)
    empty_input_sensitivities!(model.model)
    return
end
