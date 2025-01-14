# Copyright (c) 2020: Akshay Sharma and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

function MOI.get(
    model::MOI.ModelLike,
    ::ObjectiveFunctionAttribute{ReverseObjectiveFunction},
    bridge::MOI.Bridges.Objective.SlackBridge,
)
    return MOI.get(model, ReverseConstraintFunction(), bridge.constraint)
end

function MOI.set(
    model::MOI.ModelLike,
    ::ObjectiveFunctionAttribute{ForwardObjectiveFunction},
    bridge::MOI.Bridges.Objective.SlackBridge,
    value,
)
    return MOI.set(model, ForwardConstraintFunction(), bridge.constraint, value)
end

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
    attr::ReverseConstraintFunction,
    bridge::MOI.Bridges.Constraint.VectorizeBridge,
)
    return MOI.Utilities.eachscalar(
        MOI.get(model, attr, bridge.vector_constraint),
    )[1]
end

function MOI.set(
    model::MOI.ModelLike,
    attr::ForwardConstraintFunction,
    bridge::MOI.Bridges.Constraint.ScalarizeBridge,
    value,
)
    MOI.set.(model, attr, bridge.scalar_constraints, value)
    return
end

function MOI.get(
    model::MOI.ModelLike,
    attr::ReverseConstraintFunction,
    bridge::MOI.Bridges.Constraint.ScalarizeBridge,
)
    return _vectorize(MOI.get.(model, attr, bridge.scalar_constraints))
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

function _variable_to_index_map(bridge)
    return Dict{MOI.VariableIndex,MOI.VariableIndex}(
        v => MOI.VariableIndex(i) for
        (i, v) in enumerate(bridge.index_to_variable_map)
    )
end

function _U(func::MOI.VectorAffineFunction, n, variable_to_index_map)
    # x'Qx/2 + a'x + β is bridged into
    # (1, -a'x - β, U*x) in RSOC
    # where Q = U' * U
    # The linear part of `func` is `[0; -a'; U]`
    Ux = MOI.Utilities.eachscalar(func)[3:end]
    func_array = sparse_array_representation(Ux, n, variable_to_index_map)
    return func_array.terms
end

function MOI.get(
    model::MOI.ModelLike,
    attr::DiffOpt.ReverseConstraintFunction,
    bridge::MOI.Bridges.Constraint.SetMapBridge,
)
    func = MOI.get(model, attr, bridge.constraint)
    return MOI.Bridges.adjoint_map_function(typeof(bridge), func)
end

function MOI.get(
    model::MOI.ModelLike,
    attr::DiffOpt.ReverseConstraintFunction,
    bridge::MOI.Bridges.Constraint.QuadtoSOCBridge{T},
) where {T}
    variable_to_index_map = _variable_to_index_map(bridge)
    n = length(bridge.index_to_variable_map)
    U = _U(
        MOI.get(model, MOI.ConstraintFunction(), bridge.soc),
        n,
        variable_to_index_map,
    )
    Δfunc =
        convert(MOI.VectorAffineFunction{T}, MOI.get(model, attr, bridge.soc))
    filter!(Δfunc.terms) do t
        return haskey(variable_to_index_map, t.scalar_term.variable)
    end
    aff = sparse_array_representation(
        -MOI.Utilities.eachscalar(Δfunc)[2],
        n,
        variable_to_index_map,
    )
    ΔU = _U(Δfunc, n, variable_to_index_map)
    ΔQ = ΔQ_from_ΔU!(ΔU, U)
    func = MatrixScalarQuadraticFunction(
        convert(VectorScalarAffineFunction{T,Vector{T}}, aff),
        ΔQ,
    )
    index_map = MOI.Utilities.IndexMap()
    for (i, vi) in enumerate(bridge.index_to_variable_map)
        index_map[MOI.VariableIndex(i)] = vi
    end
    Δ = standard_form(IndexMappedFunction(func, index_map))
    if !bridge.less_than
        Δ = -Δ
    end
    return Δ
end

function _quad_to_soc_diff(
    ::MOI.ModelLike,
    bridge::MOI.Bridges.Constraint.QuadtoSOCBridge{T},
    diff::MOI.ScalarAffineFunction{T},
) where {T}
    n = length(bridge.index_to_variable_map)
    diff_soc =
        MOI.VectorAffineFunction{T}(MOI.VectorAffineTerm{T}[], zeros(T, n + 2))
    for t in diff.terms
        push!(
            diff_soc.terms,
            MOI.VectorAffineTerm(
                2,
                MOI.ScalarAffineTerm(-t.coefficient, t.variable),
            ),
        )
    end
    diff_soc.constants[2] = -diff.constant
    return diff_soc
end

function _quad_to_soc_diff(
    model::MOI.ModelLike,
    bridge::MOI.Bridges.Constraint.QuadtoSOCBridge{T},
    diff::MOI.ScalarQuadraticFunction{T},
) where {T}
    variable_to_index_map = _variable_to_index_map(bridge)
    n = length(bridge.index_to_variable_map)
    U = _U(
        MOI.get(model, MOI.ConstraintFunction(), bridge.soc),
        n,
        variable_to_index_map,
    )
    # We assume `diff.quadratic_terms` only contains quadratic terms with
    # variables already in a quadratic term of the initial constraint
    # Otherwise, the `U` matrix will not be square so it's a TODO
    # We remove the affine terms here as they might have other variables not
    # in `variable_to_index_map`
    diff_quad = MOI.ScalarQuadraticFunction(
        diff.quadratic_terms,
        MOI.ScalarAffineTerm{T}[],
        zero(T),
    )
    diff_aff = MOI.ScalarAffineFunction(diff.affine_terms, diff.constant)
    diff_array =
        sparse_array_representation(diff_quad, n, variable_to_index_map)
    dU = dU_from_dQ!(Matrix(diff_array.quadratic_terms), U)
    diff_soc = _quad_to_soc_diff(model, bridge, diff_aff)
    for i in axes(dU, 1)
        for j in axes(dU, 2)
            if !iszero(dU[i, j])
                scalar = MOI.ScalarAffineTerm(
                    dU[i, j],
                    bridge.index_to_variable_map[j],
                )
                push!(diff_soc.terms, MOI.VectorAffineTerm(2 + i, scalar))
            end
        end
    end
    return diff_soc
end

function MOI.set(
    model::MOI.ModelLike,
    attr::DiffOpt.ForwardConstraintFunction,
    bridge::MOI.Bridges.Constraint.QuadtoSOCBridge{T},
    diff::MOI.AbstractScalarFunction,
) where {T}
    if !bridge.less_than
        diff = -diff
    end
    diff_soc = _quad_to_soc_diff(model, bridge, diff)
    MOI.set(model, attr, bridge.soc, diff_soc)
    return
end

"""
    ΔQ_from_ΔU!(ΔU, U)

Return the symmetric solution `ΔQ` of the matrix equation
`triu(ΔU) = 2triu(U * ΔQ)`
where `ΔU` and `U` are the two argument of the function.

This function overwrites the first argument `ΔU` to store the solution.
The matrix `U` is not however modified.

The matrix `U` is assumed to be upper triangular.

We can exploit the structure of `U` here:

* If the factorization was obtained from SVD, `U` would be orthogonal
* If the factorization was obtained from Cholesky, `U` would be upper triangular.

The MOI bridge uses Cholesky in order to exploit sparsity so we are in the
second case.

We can find each column of `ΔQ` by solving a triangular linear system.
"""
function ΔQ_from_ΔU!(ΔU, U)
    n = LinearAlgebra.checksquare(ΔU)
    LinearAlgebra.rdiv!(ΔU, 2)
    for j in n:-1:1
        # FIXME MA does not support mutating subarray
        #MA.operate!(MA.sub_mul, view(ΔU, 1:j, j), view(U, 1:j, (j + 1):n), view(ΔU, j, (j+1):n))
        for row in 1:j
            for col in (j+1):n
                ΔU[row, j] -= U[row, col] * ΔU[j, col]
            end
        end
        _U = LinearAlgebra.UpperTriangular(view(U, 1:j, 1:j))
        LinearAlgebra.ldiv!(_U, view(ΔU, 1:j, j))
    end
    for j in 1:n
        for i in 1:(j-1)
            if i != j
                ΔU[j, i] = ΔU[i, j]
            end
        end
    end
    return ΔU
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
`dQ[j, 1:j]` = dj'U[:, 1:j] + U[:, j]'dU[:, 1:j]`
So
`dQ[j, 1:(j-1)] - U[:, j]' * dU[:, 1:(j-1)] = dj'U[:, 1:(j-1)]`
and
`dQ[j, j] / 2 = dj'U[:, j]`
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
