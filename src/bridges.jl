# Copyright (c) 2020: Akshay Sharma and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

function MOI.set(
    model::MOI.ModelLike,
    attr::ForwardConstraintFunction,
    bridge::MOI.Bridges.Constraint.VectorizeBridge{T},
    value,
) where {T}
    return MOI.set(
        model,
        attr,
        bridge.vector_constraint,
        MOI.Utilities.operate(vcat, T, value),
    )
end

function MOI.get(
    model::MOI.ModelLike,
    attr::DiffOpt.ReverseConstraintFunction,
    bridge::MOI.Bridges.Constraint.AbstractFunctionConversionBridge,
)
    return MOI.get(model, attr, bridge.constraint)
end

function MOI.set(
    model::MOI.ModelLike,
    attr::DiffOpt.ForwardConstraintFunction,
    bridge::MOI.Bridges.Constraint.SetMapBridge,
    func,
)
    mapped_func = MOI.Bridges.map_function(typeof(bridge), func)
    return MOI.set(model, attr, bridge.constraint, mapped_func)
end

function MOI.get(
    model::MOI.ModelLike,
    attr::DiffOpt.ReverseConstraintFunction,
    bridge::MOI.Bridges.Constraint.SetMapBridge,
)
    func = MOI.get(model, attr, bridge.constraint)
    return MOI.Bridges.adjoint_map_function(typeof(bridge), func)
end

function MOI.set(
    model::MOI.ModelLike,
    attr::DiffOpt.ForwardConstraintFunction,
    bridge::MOI.Bridges.Constraint.QuadtoSOCBridge{T},
    diff::MOI.ScalarQuadraticFunction,
) where {T}
    func = MOI.get(model, MOI.ConstraintFunction(), bridge.soc)
    index_map = index_map_to_oneto(func)
    index_map_to_oneto!(index_map, diff)
    n = length(index_map.var_map)
    func_array = sparse_array_representation(func, n, index_map)
    # x'Qx/2 + a'x + β is bridged into
    # (1, -a'x - β, U*x) in RSOC
    # where Q = U' * U
    # `func_array.terms` is `[0; -a'; U]`
    U = func_array.terms[3:end, :]
    diff_array = sparse_array_representation(diff, n, index_map)
    dU = Matrix(diff_array.quadratic_terms)
    dU = dU_from_dQ!(Matrix(diff_array.quadratic_terms), U)
    x = Vector{MOI.VariableIndex}(undef, n)
    for v in keys(index_map.var_map)
        x[index_map[v].value] = v
    end
    diff_aff = MOI.VectorAffineFunction{T}(
        MOI.VectorAffineTerm{T}[],
        zeros(T, size(dU, 1) + 2),
    )
    for t in diff.affine_terms
        push!(
            diff_aff.terms,
            MOI.VectorAffineTerm(
                2,
                MOI.ScalarAffineTerm(-t.coefficient, t.variable),
            ),
        )
    end
    diff_aff.constants[2] = -diff.constant
    for i in axes(dU, 1)
        for j in axes(dU, 2)
            if !iszero(dU[i, j])
                scalar = MOI.ScalarAffineTerm(dU[i, j], x[j])
                push!(diff_aff.terms, MOI.VectorAffineTerm(2 + i, scalar))
            end
        end
    end
    MOI.set(model, attr, bridge.soc, diff_aff)
    return
end

"""
    dU_from_dQ!(dQ, U)

Return the solution `dU` of the matrix equation
`dQ = dU' * U + U' * dU`
where `dQ` and `U` are the two argument of the function.

This function overwrites the first argument `dQ` to store the solution.
The matrix `U` is not however modified.

The matrix `dQ` is assumed to be symmetric and the matrix `U` is assumed to be
upper triangular.

We can exploit the structure of `U` here:

* If the factorization was obtained from SVD, `U` would be orthogonal
* If the factorization was obtained from Cholesky, `U` would be upper triangular.

The MOI bridge uses Cholesky in order to exploit sparsity so we are in the
second case.

We look for an upper triangular `dU` as well.

We can find each column of `dU` by solving a triangular linear system once the
previous column have been found.
Indeed, let `dj` be the `j`th column of `dU`
`dU' * U = vcat(dj'U for j in axes(U, 2))`
Therefore,
`dQ[j, 1:j]` = di'U[:, 1:j] + U[:, j]'dU[:, 1:j]`
So
`dQ[j, 1:(j-1)] - U[:, j]' * dU[:, 1:(j-1)] = dj'U[:, 1:(j-1)]`
and
`dQ[j, j] / 2 = dj'U[:, 1:(j-1)]`
"""
function dU_from_dQ!(dQ, U)
    n = LinearAlgebra.checksquare(dQ)
    for j in 1:n
        for i in axes(dQ, 1)
            if i < j
                # `dQ[:, i]` was modified to correspond to `dU[:, i]`
                # in the iteration `j := i` of the outer loop
                dd = view(U, 1:i, j)'view(dQ, 1:i, i)
                dQ[i, j] -= dd
            elseif i == j
                dQ[i, j] /= 2
            else
                dQ[i, j] = 0
            end
        end
        Ut = LinearAlgebra.UpperTriangular(view(U, 1:j, 1:j))'
        LinearAlgebra.ldiv!(Ut, view(dQ, 1:j, j))
    end
    return dQ
end
