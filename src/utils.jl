# Representation of MOI functions using SparseArrays
# Might be able to replace in the future by a function in MOI, see
# https://github.com/jump-dev/MathOptInterface.jl/pull/1238
function sparse_array_representation(terms::Vector{MOI.ScalarAffineTerm{T}}, num_variables, index_map) where {T}
    n = length(terms)
    indices = Vector{Int64}(undef, n)
    coefficients = Vector{T}(undef, n)
    for i in eachindex(terms)
        term = terms[i]
        indices[i] = index_map[term.variable_index].value
        coefficients[i] = term.coefficient
    end
    return sparsevec(indices, coefficients, num_variables)
end
function sparse_array_representation(terms::Vector{MOI.ScalarQuadraticTerm{T}}, num_variables, index_map) where {T}
    n = length(terms)
    I = Vector{Int64}(undef, n)
    J = Vector{Int64}(undef, n)
    V = Vector{T}(undef, n)
    for k in eachindex(terms)
        term = terms[k]
        i = index_map[term.variable_index_1].value
        j = index_map[term.variable_index_2].value
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
function sparse_array_representation(func::MOI.ScalarAffineFunction, num_variables, index_map)
    return SparseScalarAffineFunction(
        sparse_array_representation(func.terms, num_variables, index_map),
        func.constant,
    )
end
struct SparseScalarQuadraticFunction{T}
    affine_terms::SparseVector{T,Int64}
    quadratic_terms::SparseMatrixCSC{T,Int64}
    constant::T
end
function sparse_array_representation(func::MOI.ScalarQuadraticFunction, num_variables, index_map)
    return SparseScalarQuadraticFunction(
        sparse_array_representation(func.affine_terms, num_variables, index_map),
        sparse_array_representation(func.quadratic_terms, num_variables, index_map),
        func.constant,
    )
end
_convert(::Type{F}, ::Nothing) where {F} = zero(F)
_convert(::Type{F}, obj) where {F} = convert(F, obj)
