function MOI.set(model::JuMP.Model, attr::ForwardInObjective, func::JuMP.AbstractJuMPScalar)
    JuMP.check_belongs_to_model(func, model)
    return MOI.set(model, attr, JuMP.moi_function(func))
end

function MOI.set(model::JuMP.Model, attr::ForwardInConstraint, con_ref::JuMP.ConstraintRef, func::JuMP.AbstractJuMPScalar)
    JuMP.check_belongs_to_model(func, model)
    return MOI.set(model, attr, con_ref, JuMP.moi_function(func))
end

function MOI.get(model::JuMP.Model, attr::BackwardOutObjective)
    func = MOI.get(JuMP.backend(model), attr)
    return JuMP.jump_function(model, func)
end

function MOI.get(model::JuMP.Model, attr::BackwardOutConstraint, con_ref::JuMP.ConstraintRef)
    check_belongs_to_model(con_ref, model)
    moi_func = MOI.get(JuMP.backend(model), attr, JuMP.index(con_ref))
    return JuMP.jump_function(model, moi_func)
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
    VectorScalarAffineFunction{T, VT} <: MOI.AbstractScalarFunction

Represents the function `x ⋅ terms + constant`
as an `MOI.AbstractScalarFunction` where `x[i] = MOI.VariableIndex(i)`.
Use [`standard_form`](@ref) to convert it to a `MOI.ScalarAffineFunction{T}`.
"""
struct VectorScalarAffineFunction{T, VT} <: MOI.AbstractScalarFunction
    terms::VT
    constant::T
end
MOI.constant(func::VectorScalarAffineFunction) = func.constant
function JuMP.coefficient(func::VectorScalarAffineFunction, vi::MOI.VariableIndex)
    return func.terms[vi.value]
end
function Base.convert(::Type{MOI.ScalarAffineFunction{T}}, func::VectorScalarAffineFunction) where {T}
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
MOI.constant(func::MatrixScalarQuadraticFunction) = MOI.constant(func.affine)
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

"""
    MatrixVectorAffineFunction{T, VT} <: MOI.AbstractVectorFunction

Represents the function `terms * x + constant`
as an `MOI.AbstractVectorFunction` where `x[i] = MOI.VariableIndex(i)`.
Use [`standard_form`](@ref) to convert it to a `MOI.VectorAffineFunction{T}`.
"""
struct MatrixVectorAffineFunction{AT, VT} <: MOI.AbstractVectorFunction
    terms::AT
    constants::VT
end
MOI.constant(func::MatrixVectorAffineFunction) = func.constants
function Base.convert(::Type{MOI.VectorAffineFunction{T}}, func::MatrixVectorAffineFunction) where {T}
    return MOI.VectorAffineFunction{T}(
        MOI.VectorAffineTerm{T}[
            # TODO we should do better if the matrix is a `SparseMatrixCSC`
            MOI.VectorAffineTerm(i, MOI.ScalarAffineTerm{T}(func.terms[i, j], VI(j)))
            for i in 1:size(func.terms, 1) for j in 1:size(func.terms, 2) if !iszero(func.terms[i, j])
        ],
        func.constants,
    )
end
function standard_form(func::MatrixVectorAffineFunction{T}) where {T}
    return convert(MOI.VectorAffineFunction{T}, func)
end

# Only used for testing at the moment so performance is not critical so
# converting to standard form is ok
function MOIU.isapprox_zero(func::Union{VectorScalarAffineFunction,MatrixScalarQuadraticFunction}, tol)
    return MOIU.isapprox_zero(standard_form(func), tol)
end

"""
    IndexMappedFunction{F<:MOI.AbstractFunction} <: AbstractLazyScalarFunction

Lazily represents the function `MOI.Utilities.map_indices(index_map, DiffOpt.standard_form(func))`.
"""
struct IndexMappedFunction{F<:MOI.AbstractFunction} <: AbstractLazyScalarFunction
    func::F
    index_map::MOIU.IndexMap
end
MOI.constant(func::IndexMappedFunction) = MOI.constant(func.func)
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

# JuMP
backward(model::JuMP.Model) = backward(JuMP.backend(model))
forward(model::JuMP.Model) = forward(JuMP.backend(model))

# MOIU
backward(model::MOI.Utilities.CachingOptimizer) = backward(model.optimizer)
forward(model::MOI.Utilities.CachingOptimizer) = forward(model.optimizer)

# MOIB
backward(model::MOI.Bridges.AbstractBridgeOptimizer) = backward(model.model)
forward(model::MOI.Bridges.AbstractBridgeOptimizer) = forward(model.model)
