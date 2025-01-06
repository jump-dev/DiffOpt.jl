# Copyright (c) 2020: Akshay Sharma and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

# Representation of MOI functions using SparseArrays
# Might be able to replace in the future by a function in MOI, see
# https://github.com/jump-dev/MathOptInterface.jl/pull/1238
function sparse_array_representation(
    terms::Vector{MOI.ScalarAffineTerm{T}},
    num_variables,
    index_map,
) where {T}
    n = length(terms)
    indices = Vector{Int64}(undef, n)
    coefficients = Vector{T}(undef, n)
    for i in eachindex(terms)
        term = terms[i]
        indices[i] = index_map[term.variable].value
        coefficients[i] = term.coefficient
    end
    return SparseArrays.sparsevec(indices, coefficients, num_variables)
end

function sparse_array_representation(
    terms::Vector{MOI.VectorAffineTerm{T}},
    num_rows,
    num_variables,
    index_map,
) where {T}
    n = length(terms)
    I = Vector{Int64}(undef, n)
    J = Vector{Int64}(undef, n)
    K = Vector{T}(undef, n)
    for i in eachindex(terms)
        term = terms[i]
        I[i] = term.output_index
        J[i] = index_map[term.scalar_term.variable].value
        K[i] = term.scalar_term.coefficient
    end
    return SparseArrays.sparse(I, J, K, num_rows, num_variables)
end

function sparse_array_representation(
    terms::Vector{MOI.ScalarQuadraticTerm{T}},
    num_variables,
    index_map,
) where {T}
    n = length(terms)
    I = Vector{Int64}(undef, n)
    J = Vector{Int64}(undef, n)
    V = Vector{T}(undef, n)
    for k in eachindex(terms)
        term = terms[k]
        i = index_map[term.variable_1].value
        j = index_map[term.variable_2].value
        I[k] = i
        J[k] = j
        V[k] = term.coefficient
        if i != j
            push!(I, j)
            push!(J, i)
            push!(V, term.coefficient)
        end
    end
    return SparseArrays.sparse(I, J, V, num_variables, num_variables)
end

struct SparseScalarAffineFunction{T}
    terms::SparseArrays.SparseVector{T,Int64}
    constant::T
end

function sparse_array_representation(
    func::MOI.ScalarAffineFunction,
    num_variables,
    index_map,
)
    return SparseScalarAffineFunction(
        sparse_array_representation(func.terms, num_variables, index_map),
        func.constant,
    )
end

struct SparseScalarQuadraticFunction{T}
    quadratic_terms::SparseArrays.SparseMatrixCSC{T,Int64}
    affine_terms::SparseArrays.SparseVector{T,Int64}
    constant::T
end

function sparse_array_representation(
    func::MOI.ScalarQuadraticFunction,
    num_variables,
    index_map,
)
    return SparseScalarQuadraticFunction(
        sparse_array_representation(
            func.quadratic_terms,
            num_variables,
            index_map,
        ),
        sparse_array_representation(
            func.affine_terms,
            num_variables,
            index_map,
        ),
        func.constant,
    )
end

_convert(::Type{F}, ::Nothing) where {F} = zero(F)

_convert(::Type{F}, obj) where {F} = convert(F, obj)

struct IdentityMap end

Base.getindex(::IdentityMap, index) = index

function sparse_array_representation(func::MOI.AbstractFunction, num_variables)
    return sparse_array_representation(func, num_variables, IdentityMap())
end

"""
    struct SparseVectorAffineFunction{T} <: MOI.AbstractVectorFunction
        terms::SparseArrays.SparseMatrixCSC{T,Int}
        constants::Vector{T}
    end

The vector-valued affine function ``A x + b``, where:

* ``A`` is the sparse matrix given by `terms`
* ``b`` is the vector `constants`
"""
struct SparseVectorAffineFunction{T} <: MOI.AbstractVectorFunction
    terms::SparseArrays.SparseMatrixCSC{T,Int}
    constants::Vector{T}
end

function sparse_array_representation(
    func::MOI.VectorAffineFunction,
    num_variables,
    index_map,
)
    return SparseVectorAffineFunction(
        sparse_array_representation(
            func.terms,
            MOI.output_dimension(func),
            num_variables,
            index_map,
        ),
        func.constants,
    )
end

# In the future, we could replace by https://github.com/jump-dev/MathOptInterface.jl/pull/1238
"""
    VectorScalarAffineFunction{T, VT} <: MOI.AbstractScalarFunction

Represents the function `x ⋅ terms + constant`
as an `MOI.AbstractScalarFunction` where `x[i] = MOI.VariableIndex(i)`.
Use [`standard_form`](@ref) to convert it to a `MOI.ScalarAffineFunction{T}`.
"""
struct VectorScalarAffineFunction{T,VT} <: MOI.AbstractScalarFunction
    terms::VT
    constant::T
end
MOI.constant(func::VectorScalarAffineFunction) = func.constant
function JuMP.coefficient(
    func::VectorScalarAffineFunction,
    vi::MOI.VariableIndex,
)
    return func.terms[vi.value]
end
function Base.convert(
    ::Type{VectorScalarAffineFunction{T,VT}},
    v::SparseScalarAffineFunction,
) where {T,VT}
    return VectorScalarAffineFunction{T,VT}(v.terms, v.constant)
end
function Base.convert(
    ::Type{MOI.ScalarAffineFunction{T}},
    func::VectorScalarAffineFunction,
) where {T}
    return MOI.ScalarAffineFunction{T}(
        # TODO we should do better if the vector is a `SparseVector`, I think
        #      I have some code working for both vector types in Polyhedra.jl
        MOI.ScalarAffineTerm{T}[
            MOI.ScalarAffineTerm{T}(func.terms[i], MOI.VariableIndex(i)) for
            i in eachindex(func.terms) if !iszero(func.terms[i])
        ],
        func.constant,
    )
end
function standard_form(func::VectorScalarAffineFunction{T}) where {T}
    return convert(MOI.ScalarAffineFunction{T}, func)
end

function MOI.Utilities.operate(
    ::typeof(-),
    ::Type{T},
    func::VectorScalarAffineFunction{T},
) where {T}
    return VectorScalarAffineFunction(
        LazyArrays.ApplyArray(-, func.terms),
        -func.constant,
    )
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
struct MatrixScalarQuadraticFunction{T,VT,MT} <: MOI.AbstractScalarFunction
    affine::VectorScalarAffineFunction{T,VT}
    terms::MT
end
MOI.constant(func::MatrixScalarQuadraticFunction) = MOI.constant(func.affine)
function JuMP.coefficient(
    func::MatrixScalarQuadraticFunction,
    vi::MOI.VariableIndex,
)
    return JuMP.coefficient(func.affine, vi)
end

"""
    MatrixVectorAffineFunction{T, VT} <: MOI.AbstractVectorFunction

Represents the function `terms * x + constant`
as an `MOI.AbstractVectorFunction` where `x[i] = MOI.VariableIndex(i)`.
Use [`standard_form`](@ref) to convert it to a `MOI.VectorAffineFunction{T}`.
"""
struct MatrixVectorAffineFunction{AT,VT} <: MOI.AbstractVectorFunction
    terms::AT
    constants::VT
end

MOI.constant(func::MatrixVectorAffineFunction) = func.constants
function Base.convert(
    ::Type{MOI.VectorAffineFunction{T}},
    func::MatrixVectorAffineFunction,
) where {T}
    return MOI.VectorAffineFunction{T}(
        MOI.VectorAffineTerm{T}[
            # TODO we should do better if the matrix is a `SparseMatrixCSC`
            MOI.VectorAffineTerm(
                i,
                MOI.ScalarAffineTerm{T}(func.terms[i, j], MOI.VariableIndex(j)),
            ) for i in 1:size(func.terms, 1) for
            j in 1:size(func.terms, 2) if !iszero(func.terms[i, j])
        ],
        func.constants,
    )
end
function standard_form(func::MatrixVectorAffineFunction{T}) where {T}
    return convert(MOI.VectorAffineFunction{T}, func)
end

# Only used for testing at the moment so performance is not critical so
# converting to standard form is ok
function MOIU.isapprox_zero(
    func::Union{VectorScalarAffineFunction,MatrixScalarQuadraticFunction},
    tol,
)
    return MOIU.isapprox_zero(standard_form(func), tol)
end

function MOI.Utilities.scalar_type(::Type{<:MatrixVectorAffineFunction})
    return VectorScalarAffineFunction
end

function MOI.Utilities.scalar_type(::Type{<:SparseVectorAffineFunction})
    return SparseScalarAffineFunction
end


function Base.getindex(
    it::MOI.Utilities.ScalarFunctionIterator{F},
    output_index::Integer,
) where {F<:Union{MatrixVectorAffineFunction,SparseVectorAffineFunction}}
    return MOI.Utilities.scalar_type(F)(
        it.f.terms[output_index, :],
        it.f.constants[output_index],
    )
end

function _index_map_to_oneto!(index_map, v::MOI.VariableIndex)
    if !haskey(index_map, v)
        n = length(index_map.var_map)
        index_map[v] = MOI.VariableIndex(n + 1)
    end
    return
end

function _index_map_to_oneto!(index_map, term::MOI.ScalarAffineTerm)
    _index_map_to_oneto!(index_map, term.variable)
    return
end

function _index_map_to_oneto!(index_map, term::MOI.ScalarQuadraticTerm)
    _index_map_to_oneto!(index_map, term.variable_1)
    _index_map_to_oneto!(index_map, term.variable_2)
    return
end

function _index_map_to_oneto!(index_map, term::MOI.VectorAffineTerm)
    _index_map_to_oneto!(index_map, term.scalar_term)
    return
end

function _index_map_to_oneto!(index_map, terms)
    for term in terms
        _index_map_to_oneto!(index_map, term)
    end
    return
end

function index_map_to_oneto!(index_map, func::MOI.VectorAffineFunction)
    _index_map_to_oneto!(index_map, func.terms)
    return
end

function index_map_to_oneto!(index_map, func::MOI.ScalarQuadraticFunction)
    _index_map_to_oneto!(index_map, func.quadratic_terms)
    _index_map_to_oneto!(index_map, func.affine_terms)
    return
end

function index_map_to_oneto(func::MOI.AbstractFunction)
    index_map = MOI.Utilities.IndexMap()
    index_map_to_oneto!(index_map, func)
    return index_map
end
