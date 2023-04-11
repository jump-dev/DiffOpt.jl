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
    return sparsevec(indices, coefficients, num_variables)
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
    return sparse(I, J, K, num_rows, num_variables)
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
    return sparse(I, J, V, num_variables, num_variables)
end
struct SparseScalarAffineFunction{T}
    terms::SparseVector{T,Int64}
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
    quadratic_terms::SparseMatrixCSC{T,Int64}
    affine_terms::SparseVector{T,Int64}
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
        terms::SparseMatrixCSC{T,Int}
        constants::Vector{T}
    end

The vector-valued affine function ``A x + b``, where:

* ``A`` is the sparse matrix given by `terms`
* ``b`` is the vector `constants`
"""
struct SparseVectorAffineFunction{T} <: MOI.AbstractVectorFunction
    terms::SparseMatrixCSC{T,Int}
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

function _index_map_to_oneto!(index_map, v::MOI.VariableIndex)
    if !haskey(index_map, v)
        n = length(index_map.var_map)
        index_map[v] = MOI.VariableIndex(n + 1)
    end
    return
end

function _index_map_to_oneto!(index_map, term::MOI.ScalarAffineTerm)
    return _index_map_to_oneto!(index_map, term.variable)
end

function _index_map_to_oneto!(index_map, term::MOI.ScalarQuadraticTerm)
    _index_map_to_oneto!(index_map, term.variable_1)
    _index_map_to_oneto!(index_map, term.variable_2)
    return
end

function _index_map_to_oneto!(index_map, term::MOI.VectorAffineTerm)
    return _index_map_to_oneto!(index_map, term.scalar_term)
end

function _index_map_to_oneto!(index_map, terms)
    for term in terms
        _index_map_to_oneto!(index_map, term)
    end
end

function index_map_to_oneto!(index_map, func::MOI.VectorAffineFunction)
    return _index_map_to_oneto!(index_map, func.terms)
end

function index_map_to_oneto!(index_map, func::MOI.ScalarQuadraticFunction)
    _index_map_to_oneto!(index_map, func.quadratic_terms)
    return _index_map_to_oneto!(index_map, func.affine_terms)
end

function index_map_to_oneto(func::MOI.AbstractFunction)
    index_map = MOI.Utilities.IndexMap()
    index_map_to_oneto!(index_map, func)
    return index_map
end
