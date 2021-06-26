function MOI.set(model::JuMP.Model, attr::ForwardInObjective, func::JuMP.AbstractJuMPScalar)
    JuMP.check_belongs_to_model(func, model)
    return MOI.set(model, attr, JuMP.moi_function(func))
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

standard_form(func::Union{MOI.SingleVariable,MOI.ScalarAffineFunction,MOI.ScalarQuadraticFunction}) = func
function Base.isapprox(func1::AbstractLazyScalarFunction, func2::MOI.AbstractScalarFunction; kws...)
    return isapprox(standard_form(func1), standard_form(func2); kws...)
end

# In the future, we could replace by https://github.com/jump-dev/MathOptInterface.jl/pull/1238
"""
    struct VectorScalarAffineFunction{T, VT} <: MOI.AbstractScalarFunction
        terms::VT
        constant::T
    end

Represents the function `x ⋅ terms + constant`
as an `MOI.AbstractScalarFunction` where `x[i] = MOI.VariableIndex(i)`.
Use [`standard_form`](@ref) to convert it to a `MOI.ScalarAffineFunction{T}`.
"""
struct VectorScalarAffineFunction{T, VT} <: MOI.AbstractScalarFunction
    terms::VT
    constant::T
end
function JuMP.coefficient(func::VectorScalarAffineFunction, vi::MOI.VariableIndex)
    return func.terms[vi.value]
end
function Base.convert(::Type{MOI.ScalarAffineFunction{T}}, func::VectorScalarAffineFunction) where {T}
    n = length(func.terms)
    return MOI.ScalarAffineFunction{T}(
        # TODO we should do better if the vector is a `SparseVector`, I think
        #      I have some code working for both vector types in Polyhedra.jl
        MOI.ScalarAffineTerm{T}[
            MOI.ScalarAffineTerm{T}(func.terms[i], VI(i))
            for i in eachindex(func.terms) if !iszero(func.terms[i])
        ],
        func.constant,
    )
end
function standard_form(func::VectorScalarAffineFunction{T}) where {T}
    return convert(MOI.ScalarAffineFunction{T}, func)
end

"""
    struct MatrixScalarQuadraticFunction{T, VT, MT} <: MOI.AbstractScalarFunction
        affine::VectorScalarAffineFunction{T,VT}
        terms::MT
    end

Represents the function `x' * terms * x / 2 + affine` as an
`MOI.AbstractScalarFunction` where `x[i] = MOI.VariableIndex(i)`.
Use [`standard_form`](@ref) to convert it to a `MOI.ScalarQuadraticFunction{T}`.
"""
struct MatrixScalarQuadraticFunction{T, VT, MT} <: MOI.AbstractScalarFunction
    affine::VectorScalarAffineFunction{T,VT}
    terms::MT
end
function JuMP.coefficient(func::MatrixScalarQuadraticFunction, vi::MOI.VariableIndex)
    return JuMP.coefficient(func.affine, vi)
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
function Base.convert(::Type{MOI.ScalarQuadraticFunction{T}}, func::MatrixScalarQuadraticFunction) where {T}
    n = length(func.affine.terms)
    aff = convert(MOI.ScalarAffineFunction{T}, func.affine)
    quad = MOI.ScalarQuadraticTerm{T}[
        MOI.ScalarQuadraticTerm{T}(quad_sym_half(func, VI(i), VI(j)), VI(i), VI(j))
        for j in 1:n for i in 1:j if !iszero(quad_sym_half(func, VI(i), VI(j)))
    ]
    return MOI.ScalarQuadraticFunction{T}(aff.terms, quad, aff.constant)
end
function standard_form(func::MatrixScalarQuadraticFunction{T}) where {T}
    return convert(MOI.ScalarQuadraticFunction{T}, func)
end

# Only used for testing at the moment so performance is not critical so
# converting to standard form is ok
function MOIU.isapprox_zero(func::Union{VectorScalarAffineFunction,MatrixScalarQuadraticFunction}, tol)
    return MOIU.isapprox_zero(standard_form(func), tol)
end

"""
    struct IndexMappedFunction{F<:MOI.AbstractScalarFunction} <: AbstractLazyScalarFunction
        func::F
        index_map::MOIU.IndexMap
    end

Lazily represents the function `MOI.Utilities.map_indices(index_map, DiffOpt.standard_form(func))`.
"""
struct IndexMappedFunction{F<:MOI.AbstractScalarFunction} <: AbstractLazyScalarFunction
    func::F
    index_map::MOIU.IndexMap
end
function JuMP.coefficient(func::IndexMappedFunction, vi::MOI.VariableIndex)
    return JuMP.coefficient(func.func, func.index_map[vi])
end
function quad_sym_half(func::IndexMappedFunction, vi1::MOI.VariableIndex, vi2::MOI.VariableIndex)
    return quad_sym_half(func.func, func.index_map[vi1], func.index_map[vi2])
end
function JuMP.coefficient(func::IndexMappedFunction, vi1::MOI.VariableIndex, vi2::MOI.VariableIndex)
    return JuMP.coefficient(func.func, func.index_map[vi1], func.index_map[vi2])
end
function standard_form(func::IndexMappedFunction)
    return MOIU.map_indices(func.index_map, standard_form(func.func))
end
MOIU.isapprox_zero(func::IndexMappedFunction, tol) = MOIU.isapprox_zero(func.func, tol)

function MOIU.map_indices(index_map::MOIU.IndexMap, func::AbstractLazyScalarFunction)
    return IndexMappedFunction(func, index_map)
end

"""
    struct MOItoJuMP{F<:MOI.AbstractScalarFunction} <: JuMP.AbstractJuMPScalar
        model::JuMP.Model
        func::F
    end

Lazily represents the function `JuMP.jump_function(model, DiffOpt.standard_form(func))`.
"""
struct MOItoJuMP{F<:MOI.AbstractScalarFunction} <: JuMP.AbstractJuMPScalar
    model::JuMP.Model
    func::F
end
Base.broadcastable(func::MOItoJuMP) = Ref(func)
function JuMP.coefficient(func::MOItoJuMP, var_ref::JuMP.VariableRef)
    check_belongs_to_model(var_ref, func.model)
    return JuMP.coefficient(func.func, JuMP.index(var_ref))
end
function quad_sym_half(func::MOItoJuMP, var1_ref::JuMP.VariableRef, var2_ref::JuMP.VariableRef)
    check_belongs_to_model(var1_ref, func.model)
    return quad_sym_half(func.func, JuMP.index(vi1), JuMP.index(var2_ref))
end
function JuMP.coefficient(func::MOItoJuMP, var1_ref::JuMP.VariableRef, var2_ref::JuMP.VariableRef)
    check_belongs_to_model(var2_ref, func.model)
    return JuMP.coefficient(func.func, JuMP.index(vi1), JuMP.index(var2_ref))
end
function Base.convert(::Type{JuMP.GenericAffExpr{T,JuMP.VariableRef}}, func::MOItoJuMP) where {T}
    return JuMP.GenericAffExpr{T,JuMP.VariableRef}(func.model, convert(MOI.ScalarAffineFunction{T}, func.func))
end
function Base.convert(::Type{JuMP.GenericQuadExpr{T,JuMP.VariableRef}}, func::MOItoJuMP) where {T}
    return JuMP.GenericQuadExpr{T,JuMP.VariableRef}(func.model, convert(MOI.ScalarQuadraticFunction{T}, func.func))
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

function MOI.get(model::JuMP.Model, attr::BackwardOutObjective)
    func = MOI.get(JuMP.backend(model), attr)
    return JuMP.jump_function(model, func)
end


# extend caching optimizer
function MOI.set(
    m::MOI.Utilities.CachingOptimizer,
    attr::AbstractDiffAttribute,
    index::MOI.Index,
    value,
)
    if m.state == MOI.Utilities.ATTACHED_OPTIMIZER
        optimizer_index = m.model_to_optimizer_map[index]
        optimizer_value = MOI.Utilities.map_indices(m.model_to_optimizer_map, value)
        # if m.mode == AUTOMATIC
        #     try
        #         MOI.set(m.optimizer, attr,
        #             optimizer_index, optimizer_value)
        #     catch err
        #         if err isa MOI.NotAllowedError
        #             reset_optimizer(m)
        #         else
        #             rethrow(err)
        #         end
        #     end
        # else
            MOI.set(m.optimizer, attr,
                optimizer_index, optimizer_value)
        # end
    else
        error("Cannot set AbstractDiffAttribute $(attr) is the state is different from ATTACHED_OPTIMIZER")
    end
    # return MOI.set(m.model_cache, attr, index, value)
end
function MOI.set(
    m::MOI.Utilities.CachingOptimizer,
    attr::AbstractDiffAttribute,
    index::MOI.Index,
    index2::MOI.Index,
    value,
)
    if m.state == MOI.Utilities.ATTACHED_OPTIMIZER
        optimizer_index = m.model_to_optimizer_map[index]
        optimizer_index2 = m.model_to_optimizer_map[index2]
        optimizer_value = MOI.Utilities.map_indices(m.model_to_optimizer_map, value)
        # if m.mode == AUTOMATIC
        #     try
        #         MOI.set(m.optimizer, attr,
        #             optimizer_index, optimizer_index2, optimizer_value)
        #     catch err
        #         if err isa MOI.NotAllowedError
        #             MOI.Utilities.reset_optimizer(m)
        #         else
        #             rethrow(err)
        #         end
        #     end
        # else
            MOI.set(m.optimizer, attr,
                optimizer_index, optimizer_index2, optimizer_value)
        # end
    else
        error("Cannot set AbstractDiffAttribute $(attr) is the state is different from ATTACHED_OPTIMIZER")
    end
    # return MOI.set(m.model_cache, attr, index, index2, value)
end

function MOI.get(
    model::MOI.Utilities.CachingOptimizer,
    attr::AbstractDiffAttribute,
    index::MOI.Index,
)
    if MOI.Utilities.state(model) == MOI.Utilities.NO_OPTIMIZER
        error(
            "Cannot query $(attr) from caching optimizer because no " *
            "optimizer is attached.",
        )
    end
    return MOI.Utilities.map_indices(
        model.optimizer_to_model_map,
        MOI.get(model.optimizer, attr, model.model_to_optimizer_map[index]),
    )
end
function MOI.get(
    model::MOI.Utilities.CachingOptimizer,
    attr::AbstractDiffAttribute,
    index::MOI.Index,
    index2::MOI.Index,
)
    if MOI.Utilities.state(model) == MOI.Utilities.NO_OPTIMIZER
        error(
            "Cannot query $(attr) from caching optimizer because no " *
            "optimizer is attached.",
        )
    end
    return MOI.Utilities.map_indices(
        model.optimizer_to_model_map,
        MOI.get(model.optimizer, attr,
            model.model_to_optimizer_map[index],
            model.model_to_optimizer_map[index2]),
    )
end


function MOI.get(
    model::JuMP.Model,
    attr::AbstractDiffAttribute,
    v::Union{JuMP.VariableRef,JuMP.ConstraintRef},
)
    JuMP.check_belongs_to_model(v, model)
    return MOI.get(JuMP.backend(model), attr, JuMP.index(v))
end
function MOI.get(
    model::JuMP.Model,
    attr::AbstractDiffAttribute,
    i1::JuMP.VariableRef,
    i2::Union{JuMP.VariableRef,JuMP.ConstraintRef},
)
    JuMP.check_belongs_to_model(i1, model)
    JuMP.check_belongs_to_model(i2, model)
    return MOI.get(JuMP.backend(model), attr, JuMP.index(i1), JuMP.index(i2))
end

function MOI.set(
    model::JuMP.Model,
    attr::AbstractDiffAttribute,
    v::Union{JuMP.VariableRef,JuMP.ConstraintRef},
    value,
)
    JuMP.check_belongs_to_model(v, model)
    return MOI.set(JuMP.backend(model), attr, JuMP.index(v), value)
end
function MOI.set(
    model::JuMP.Model,
    attr::AbstractDiffAttribute,
    i1::JuMP.VariableRef,
    i2::Union{JuMP.VariableRef,JuMP.ConstraintRef},
    value,
)
    JuMP.check_belongs_to_model(i1, model)
    JuMP.check_belongs_to_model(i2, model)
    return MOI.set(JuMP.backend(model), attr, JuMP.index(i1), JuMP.index(i2), value)
end

# JuMP
backward(model::JuMP.Model) = backward(JuMP.backend(model))
forward(model::JuMP.Model) = forward(JuMP.backend(model))

# MOIU
backward(model::MOI.Utilities.CachingOptimizer) = backward(model.optimizer)
forward(model::MOI.Utilities.CachingOptimizer) = forward(model.optimizer)

# MOIB
backward(model::MOI.Bridges.AbstractBridgeOptimizer) = backward(model.model)
forward(model::MOI.Bridges.AbstractBridgeOptimizer) = forward(model.model)

# bridges
# TODO: bridging is non-trivial
# there might be transformations that we are ignoring
# we should at least check for bridge and block if they are used
function MOI.get(
    b::MOI.Bridges.AbstractBridgeOptimizer,
    attr::AbstractDiffAttribute,
    index::MOI.Index,
)
    # if is_bridged(b, index)
    #     value = call_in_context(
    #         b,
    #         index,
    #         bridge -> MOI.get(b, attr, bridge, _index(b, index)...),
    #     )
    # else
    value = MOI.get(b.model, attr, index)
    # end
    # return unbridged_function(b, value)
end
function MOI.get(
    b::MOI.Bridges.AbstractBridgeOptimizer,
    attr::AbstractDiffAttribute,
    index::MOI.Index,
    index2::MOI.Index,
)
    value = MOI.get(b.model, attr, index, index2)
end
function MOI.set(
    b::MOI.Bridges.AbstractBridgeOptimizer,
    attr::AbstractDiffAttribute,
    index::MOI.Index,
    value,
)
    MOI.set(b.model, attr, index, value)
end
function MOI.set(
    b::MOI.Bridges.AbstractBridgeOptimizer,
    attr::AbstractDiffAttribute,
    index::MOI.Index,
    index2::MOI.Index,
    value,
)
    MOI.set(b.model, attr, index, index2, value)
end
