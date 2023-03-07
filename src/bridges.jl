function MOI.set(
    model::MOI.ModelLike,
    ::ObjectiveFunctionAttribute{ForwardObjectiveFunction},
    bridge::MOI.Bridges.Objective.SlackBridge,
    value,
)
    MOI.set(model, ForwardConstraintFunction(), bridge.constraint, value)
end

function MOI.set(
    model::MOI.ModelLike,
    attr::ForwardConstraintFunction,
    bridge::MOI.Bridges.Constraint.VectorizeBridge{T},
    value,
) where {T}
    MOI.set(model, attr, bridge.vector_constraint, MOI.Utilities.operate(vcat, T, value))
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
    MOI.set(model, attr, bridge.constraint, mapped_func)
end
function MOI.get(
    model::MOI.ModelLike,
    attr::DiffOpt.ReverseConstraintFunction,
    bridge::MOI.Bridges.Constraint.SetMapBridge,
)
    func = MOI.get(model, attr, bridge.constraint)
    return MOI.Bridges.adjoint_map_function(typeof(bridge), func)
end

function _quad_to_soc_diff(
    ::MOI.ModelLike,
    bridge::MOI.Bridges.Constraint.QuadtoSOCBridge{T},
    diff::MOI.ScalarAffineFunction{T},
) where {T}
    n = length(bridge.index_to_variable_map)
    diff_soc = MOI.VectorAffineFunction{T}(
        MOI.VectorAffineTerm{T}[],
        zeros(T, n + 2),
    )
    for t in diff.terms
        push!(diff_soc.terms, MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(-t.coefficient, t.variable)))
    end
    diff_soc.constants[2] = -diff.constant
    return diff_soc
end

function _quad_to_soc_diff(
    model::MOI.ModelLike,
    bridge::MOI.Bridges.Constraint.QuadtoSOCBridge{T},
    diff::MOI.ScalarQuadraticFunction{T},
) where {T}
    func = MOI.get(model, MOI.ConstraintFunction(), bridge.soc)
    variable_to_index_map = Dict{MOI.VariableIndex,MOI.VariableIndex}(
        v => MOI.VariableIndex(i) for (i, v) in enumerate(bridge.index_to_variable_map)
    )
    n = length(bridge.index_to_variable_map)
    Ux = MOI.Utilities.eachscalar(func)[3:end]
    func_array = sparse_array_representation(Ux, n, variable_to_index_map)
    # x'Qx/2 + a'x + β is bridged into
    # (1, -a'x - β, U*x) in RSOC
    # where Q = U' * U
    # `func_array.terms` is `[0; -a'; U]`
    U = func_array.terms
    # We assume `diff.quadratic_terms` only contains quadratic terms with
    # variables already in a quadratic term of the initial constraint
    # Otherwise, the `U` matrix will not be square so it's a TODO
    # We remove the affine terms here as they might have other variables not
    # in `variable_to_index_map`
    diff_quad = MOI.ScalarQuadraticFunction(diff.quadratic_terms, MOI.ScalarAffineTerm{T}[], zero(T))
    diff_aff = MOI.ScalarAffineFunction(diff.affine_terms, diff.constant)
    diff_array = sparse_array_representation(diff_quad, n, variable_to_index_map)
    dU = Matrix(diff_array.quadratic_terms)
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
    diff_soc = _quad_to_soc_diff(model, bridge, diff)
    MOI.set(model, attr, bridge.soc, diff_soc)
    return
end

"""
    dU_from_dQ!(dQ, U)

Return the solution `dU` of the matrix equation
`dQ = dU' * U + U' * dU`
where `dQ` and `U` are the two argument of the function.

This function overwrites the first argument `dQ` to store the solution.
The matrix `U` is not however modified.

The matrix `dQ` is assumed to be symmetric and the matrix `U` is assumed to be upper triangular.

We can exploit the structure of `U` here:

* If the factorization was obtained from SVD, `U` would be orthogonal
* If the factorization was obtained from Cholesky, `U` would be upper triangular.

The MOI bridge uses Cholesky in order to exploit sparsity so we are in the second case.
We look for an upper triangular `dU` as well.
We can find each column of `dU` by solving a triangular linear system once the previous
column have been found.
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
