const PROBLEM_DATA = Tuple{
    SparseArrays.SparseMatrixCSC{Float64,Int64}, Vector{Float64}, # Q, q
    SparseArrays.SparseMatrixCSC{Float64,Int64}, Vector{Float64}, # G, h
    SparseArrays.SparseMatrixCSC{Float64,Int64}, Vector{Float64}, # A, b
    Int, Vector{VI}, # nz, var_list
    Int, Vector{MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.LessThan{Float64}}}, # nineq_le, le_con_idx
    Int, Vector{MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.GreaterThan{Float64}}}, # nineq_ge, ge_con_idx
    Int, Vector{MOI.ConstraintIndex{MOI.VariableIndex, MOI.LessThan{Float64}}}, # nineq_sv_le, le_con_sv_idx
    Int, Vector{MOI.ConstraintIndex{MOI.VariableIndex, MOI.GreaterThan{Float64}}}, # nineq_sv_ge, ge_con_sv_idx
    Int, Vector{MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.EqualTo{Float64}}}, # neq, eq_con_idx
    Int, Vector{MOI.ConstraintIndex{MOI.VariableIndex, MOI.EqualTo{Float64}}}, # neq_sv, eq_con_sv_idx
}

# FIXME temporary hack
MOI.is_valid(::PROBLEM_DATA, ::MOI.VariableIndex) = true

mutable struct QPDiff <: DiffModel
    model::Union{Nothing,PROBLEM_DATA} # TODO use `MatrixOfConstraints`

    # storage for problem data in matrix form
    # includes maps from matrix indices to problem data held in `optimizer`
    # also includes KKT matrices
    # also includes the solution
    gradient_cache::Union{Nothing,QPCache}

    # caches for sensitivity output
    # result from solving KKT/residualmap linear systems
    # this allows keeping the same `gradient_cache`
    # if only sensitivy input changes
    forw_grad_cache::Union{Nothing,QPForwBackCache}
    back_grad_cache::Union{Nothing,QPForwBackCache}

    # sensitivity input cache using MOI like sparse format
    input_cache::DiffInputCache

    x::Vector{Float64} # Primal
    λ::Vector{Float64} # Dual of inequalities
    ν::Vector{Float64} # Dual of equalities
end
function QPDiff()
    return QPDiff(nothing, nothing, nothing, nothing, DiffInputCache(), Float64[], Float64[], Float64[])
end

function MOI.empty!(model::QPDiff)
    model.model = nothing
    model.gradient_cache = nothing
    model.forw_grad_cache = nothing
    model.back_grad_cache = nothing
    empty!(model.input_cache)
    empty!(model.x)
    empty!(model.λ)
    empty!(model.ν)
    return
end

function MOI.copy_to(dest::QPDiff, src::MOI.ModelLike)
    MOI.empty!(dest)
    dest.model, index_map = get_problem_data(src)

    vis_src = MOI.get(src, MOI.ListOfVariableIndices())
    MOI.set(dest, MOI.VariablePrimalStart(), getindex.(Ref(index_map), vis_src), MOI.get(src, MOI.VariablePrimal(), vis_src))

    (
        Q, q, G, h, A, b,
        nz, var_list,
        nineq_le, le_con_idx,
        nineq_ge, ge_con_idx,
        nineq_sv_le, le_con_sv_idx,
        nineq_sv_ge, ge_con_sv_idx,
        neq, eq_con_idx,
        neq_sv, eq_con_sv_idx,
    ) = dest.model

    # separate λ, ν

    λ = -MOI.get.(src, MOI.ConstraintDual(), le_con_idx)
    append!(
        λ,
        MOI.get.(src, MOI.ConstraintDual(), ge_con_idx),
    )
    append!(
        λ,
        -MOI.get.(src, MOI.ConstraintDual(), le_con_sv_idx),
    )
    append!(
        λ,
        MOI.get.(src, MOI.ConstraintDual(), ge_con_sv_idx),
    )
    dest.λ = convert(Vector{Float64}, λ)
    # We want to stay consistent with the variable `ν` defined in (3) of
    # Left hand side of eq. (6) in https://arxiv.org/pdf/1703.00443.pdf
    # However, in eq. (6), they put it in the lagrangian as
    # `+ ν ⋅ (Az - b)`
    # while in MOI, we put it as
    # `- ν ⋅ (Az - b)`
    # so the we should reverse the sign if we want to use the same equations
    # as in the paper.
    ν = -MOI.get.(src, MOI.ConstraintDual(), eq_con_idx)
    append!(
        ν,
        -MOI.get.(src, MOI.ConstraintDual(), eq_con_sv_idx),
    )
    dest.ν = convert(Vector{Float64}, ν)

    return index_map
end

function _gradient_cache(model::QPDiff)
    if model.gradient_cache !== nothing
        return model.gradient_cache
    end

    (
        Q, q, G, h, A, b,
        nz, var_list,
        nineq_le, le_con_idx,
        nineq_ge, ge_con_idx,
        nineq_sv_le, le_con_sv_idx,
        nineq_sv_ge, ge_con_sv_idx,
        neq, eq_con_idx,
        neq_sv, eq_con_sv_idx,
    ) = model.model

    z = model.x
    λ = model.λ
    ν = model.ν

    LHS = create_LHS_matrix(model.x, λ, Q, G, h, A)

    model.gradient_cache = QPCache(
        λ,
        ν,
        z,
        LHS,
    )

    return model.gradient_cache
end

# TODO: create test functions for the methods

# """
#     Left hand side of eqn(6) in https://arxiv.org/pdf/1703.00443.pdf
# """
# function create_LHS_matrix(z, λ, Q, G, h, A=nothing)
#     if A == nothing || size(A)[1] == 0
#         return [Q                G';
#                 Diagonal(λ) * G    Diagonal(G * z - h)]
#     else
#         @assert size(A)[2] == size(G)[2]
#         p, n = size(A)
#         m    = size(G)[1]
#         return [Q                  G'                    A';
#                 Diagonal(λ) * G    Diagonal(G * z - h)   zeros(m, p);
#                 A                  zeros(p, m)           zeros(p, p)]
#     end
# end


"""
    create_LHS_matrix(z, λ, Q, G, h, A=nothing)

Inverse matrix specified on RHS of eqn(7) in https://arxiv.org/pdf/1703.00443.pdf

Helper method while calling `_backward_quad`
"""
function create_LHS_matrix(z, λ, Q, G, h, A=nothing)::AbstractMatrix{Float64}
    if (A === nothing || size(A)[1] == 0) && (G === nothing || size(G)[1] == 0)
        return Q
    elseif A === nothing || size(A)[1] == 0
        return [Q         G' * Diagonal(λ);
                G         Diagonal(G * z - h)]
    elseif G === nothing || size(G)[1] == 0
        p, n = size(A)
        return [Q         A';
                A         spzeros(p, p)]
    else
        p, n  = size(A)
        m, n2 = size(G)
        if n != n2
            throw(DimensionError("Sizes of $A and $G do not match"))
        end
        return [Q         G' * Diagonal(λ)       A';
                G         Diagonal(G * z - h)    spzeros(m, p);
                A         spzeros(p, m)          spzeros(p, p)]
    end
end
# TODO: this is the transpose, check back for usage

# """
#     Right hand side of eqn(6) in https://arxiv.org/pdf/1703.00443.pdf
# """
# function create_RHS_matrix(z, dQ, dq, λ, dG, dh, ν=nothing, dA=nothing, db=nothing)
#     if dA == nothing || size(dA)[1] == 0
#         return -[dQ * z + dq + dG' * λ      ;
#                  Diagonal(λ) * (dG * z - dh)]
#     else
#         return -[dQ * z + dq + dG' * λ + dA' * ν;
#                  Diagonal(λ) * (dG * z - dh)    ;
#                  dA * z - db                    ]
#     end
# end

const _QP_SET_TYPES = Union{
    MOI.GreaterThan{Float64},
    MOI.LessThan{Float64},
    MOI.EqualTo{Float64},
    # MOI.Interval{Float64},
}

const _QP_FUNCTION_TYPES = Union{
    MOI.VariableIndex,
    MOI.ScalarAffineFunction{Float64},
}

_qp_supported(::Type{F}, ::Type{S}) where {F <: _QP_FUNCTION_TYPES, S <: _QP_SET_TYPES} = true
_qp_supported(::Type{F}, ::Type{S}) where {F, S} = false
function _qp_supported(model::MOI.AbstractOptimizer)
    return all(FS -> _qp_supported(FS...), MOI.get(model, MOI.ListOfConstraintTypesPresent()))
end

"""
    get_problem_data(model::MOI.AbstractOptimizer)

Return problem parameters as matrices along with other program info such as number of constraints, variables, etc
"""
function get_problem_data(model::MOI.AbstractOptimizer)
    for (F, S) in MOI.get(model, MOI.ListOfConstraintTypesPresent())
        if !_qp_supported(F, S)
            throw(MOI.UnsupportedConstraint{F,S}("DiffOpt does not support this constraint type for its Quadratic Programming differentiation. Maybe try the Conic Programming differentiation ? For this, do `MOI.set(model, DiffOpt.ProgramClass(), DiffOpt.CONIC)`."))
        end
    end
    var_list = MOI.get(model, MOI.ListOfVariableIndices())
    nz = length(var_list)

    index_map = MOIU.IndexMap()
    for (i,vi) in enumerate(var_list)
        index_map[vi] = VI(i)
    end

    # handle inequality constraints
    le_con_idx = MOI.get(
                        model,
                        MOI.ListOfConstraintIndices{
                            MOI.ScalarAffineFunction{Float64},
                            MOI.LessThan{Float64},
                        }())
    ge_con_idx = MOI.get(
                        model,
                        MOI.ListOfConstraintIndices{
                            MOI.ScalarAffineFunction{Float64},
                            MOI.GreaterThan{Float64},
                        }())
    nineq_le = length(le_con_idx)
    nineq_ge = length(ge_con_idx)
    le_con_sv_idx = MOI.get(
                        model,
                        MOI.ListOfConstraintIndices{
                            MOI.VariableIndex,
                            MOI.LessThan{Float64},
                        }())
    ge_con_sv_idx = MOI.get(
                        model,
                        MOI.ListOfConstraintIndices{
                            MOI.VariableIndex,
                            MOI.GreaterThan{Float64},
                        }())
    nineq_sv_le = length(le_con_sv_idx)
    nineq_sv_ge = length(ge_con_sv_idx)

    G = spzeros(nineq_le + nineq_ge + nineq_sv_le + nineq_sv_ge, nz)
    h = spzeros(nineq_le + nineq_ge + nineq_sv_le + nineq_sv_ge)

    ineq_cont = 0
    eq_cont = 0

    for i in 1:nineq_le
        con = le_con_idx[i]

        func = MOI.get(model, MOI.ConstraintFunction(), con)
        set = MOI.get(model, MOI.ConstraintSet(), con)

        for (j, var_idx) in enumerate(var_list)
            for term in func.terms
                if term.variable == var_idx
                    G[i,j] = MOI.coefficient(term)
                end
            end
        end
        h[i] = set.upper - func.constant

        ineq_cont += 1
        index_map[con] =
            CI{MOI.ScalarAffineFunction{Float64}, MOI.LessThan{Float64}}(ineq_cont)
    end
    for i in 1:nineq_ge
        # note: ax >= b needs to be converted in Gx <= h form
        con = ge_con_idx[i]

        func = MOI.get(model, MOI.ConstraintFunction(), con)
        set = MOI.get(model, MOI.ConstraintSet(), con)

        for (j, var_idx) in enumerate(var_list)
            for term in func.terms
                if term.variable == var_idx
                    G[i+nineq_le,j] = -MOI.coefficient(term)
                end
            end
        end
        h[i+nineq_le] = func.constant - set.lower
        ineq_cont += 1
        index_map[con] =
            CI{MOI.ScalarAffineFunction{Float64}, MOI.GreaterThan{Float64}}(ineq_cont)
    end
    for i in eachindex(le_con_sv_idx)
        con = le_con_sv_idx[i]
        func = MOI.get(model, MOI.ConstraintFunction(), con)
        set = MOI.get(model, MOI.ConstraintSet(), con)
        vidx = findfirst(v -> v == func, var_list)
        G[i+nineq_le+nineq_ge,vidx] = 1
        h[i+nineq_le+nineq_ge] = MOI.constant(set)
        ineq_cont += 1
        index_map[con] =
            CI{MOI.VariableIndex, MOI.LessThan{Float64}}(ineq_cont)
    end
    for i in eachindex(ge_con_sv_idx)
        # note: x >= b needs to be converted in Gx <= h form
        con = ge_con_sv_idx[i]
        func = MOI.get(model, MOI.ConstraintFunction(), con)
        set = MOI.get(model, MOI.ConstraintSet(), con)
        vidx = findfirst(isequal(func), var_list)
        G[i+nineq_le+nineq_ge+nineq_sv_le,vidx] = -1
        h[i+nineq_le+nineq_ge+nineq_sv_le] = -MOI.constant(set)
        ineq_cont += 1
        index_map[con] =
            CI{MOI.VariableIndex, MOI.GreaterThan{Float64}}(ineq_cont)
    end

    # handle equality constraints
    eq_con_idx = MOI.get(
                        model,
                        MOI.ListOfConstraintIndices{
                            MOI.ScalarAffineFunction{Float64},
                            MOI.EqualTo{Float64}
                        }())
    neq = length(eq_con_idx)

    eq_con_sv_idx = MOI.get(
        model,
        MOI.ListOfConstraintIndices{
            MOI.VariableIndex,
            MOI.EqualTo{Float64}
        }())
    neq_sv = length(eq_con_sv_idx)

    A = spzeros(neq + neq_sv, nz)
    b = spzeros(neq + neq_sv)

    for i in 1:neq
        con = eq_con_idx[i]

        func = MOI.get(model, MOI.ConstraintFunction(), con)
        set = MOI.get(model, MOI.ConstraintSet(), con)

        for x in func.terms
            # never nothing, variable is present
            vidx = findfirst(isequal(x.variable), var_list)
            A[i, vidx] = x.coefficient
        end
        b[i] = set.value - func.constant

        eq_cont += 1
        index_map[con] =
            CI{MOI.ScalarAffineFunction{Float64}, MOI.EqualTo{Float64}}(eq_cont)
    end
    for i in 1:neq_sv
        con = eq_con_sv_idx[i]
        func = MOI.get(model, MOI.ConstraintFunction(), con)
        set = MOI.get(model, MOI.ConstraintSet(), con)
        vidx = findfirst(isequal(func), var_list)
        A[i+neq,vidx] = 1
        b[i+neq] = set.value
        eq_cont += 1
        index_map[con] =
            CI{MOI.VariableIndex, MOI.EqualTo{Float64}}(eq_cont)
    end


    # handle objective
    # works both for any objective function convertible to a ScalarQuadraticFunction.
    # So in particular VariableIndex, ScalarAffineFunction and ScalarQuadraticFunction should work.
    objective_function = MOI.get(model, MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}())
    sparse_array_obj = sparse_array_representation(objective_function, nz, index_map)

    return (
        sparse_array_obj.quadratic_terms, sparse_array_obj.affine_terms,
        G, h, A, b,
        nz, var_list,
        nineq_le, le_con_idx,
        nineq_ge, ge_con_idx,
        nineq_sv_le, le_con_sv_idx,
        nineq_sv_ge, ge_con_sv_idx,
        neq, eq_con_idx,
        neq_sv, eq_con_sv_idx,
    ), index_map
end

function MOI.get(model::QPDiff, ::ForwardOutVariablePrimal, vi::VI)
    return model.forw_grad_cache.dz[vi.value]
end

_get_db(model::QPDiff, ci) = _get_db(model.back_grad_cache, model.gradient_cache, ci)
_neg_if_gt(x, ::Type{<:MOI.LessThan}) = x
_neg_if_gt(x, ::Type{<:MOI.GreaterThan}) = -x
function _get_db(b_cache::QPForwBackCache, g_cache::QPCache, ci::CI{F,S}
) where {F,S}
    i = ci.value
    # dh = -Diagonal(λ) * dλ
    dλ = b_cache.dλ
    λ = g_cache.inequality_duals
    return _neg_if_gt(λ[i] * dλ[i], S)
end
function _get_db(b_cache::QPForwBackCache, g_cache::QPCache, ci::CI{F,S}
) where {F,S<:MOI.EqualTo}
    i = ci.value
    dν = b_cache.dν
    return dν[i]
end

"""
    backward(model::QPDiff)

Method to differentiate optimal solution `z` and return
product of jacobian matrices (`dz / dQ`, `dz / dq`, etc) with
the backward pass vector `dl / dz`

The method computes the product of
1. jacobian of problem solution `z*` with respect to
    problem parameters set with the [`BackwardInVariablePrimal`](@ref)
2. a backward pass vector `dl / dz`, where `l` can be a loss function

Note that this method *does not returns* the actual jacobians.

For more info refer eqn(7) and eqn(8) of https://arxiv.org/pdf/1703.00443.pdf
"""
function backward(model::QPDiff)
    gradient_cache = _gradient_cache(model)
    (
        Q, q, G, h, A, b, nz, var_list,
        nineq_le, le_con_idx,
        nineq_ge, ge_con_idx,
        nineq_sv_le, le_con_sv_idx,
        nineq_sv_ge, ge_con_sv_idx,
        neq, eq_con_idx,
        neq_sv, eq_con_sv_idx,
    ) = model.model
    z = gradient_cache.var_primals
    λ = gradient_cache.inequality_duals
    ν = gradient_cache.equality_duals
    LHS = gradient_cache.lhs

    dl_dz = zeros(length(z))
    for (vi, value) in model.input_cache.dx
        dl_dz[vi.value] = value
    end

    nineq_total = nineq_le + nineq_ge + nineq_sv_le + nineq_sv_ge
    RHS = [dl_dz; zeros(neq + neq_sv + nineq_total)]

    partial_grads = if norm(Q) ≈ 0
        -lsqr(LHS, RHS)
    else
        -LHS \ RHS
    end

    dz = partial_grads[1:nz]
    dλ = partial_grads[nz+1:nz+nineq_total]
    dν = partial_grads[nz+nineq_total+1:nz+nineq_total+neq+neq_sv]

    model.back_grad_cache = QPForwBackCache(dz, dλ, dν)
    return nothing
    # dQ = 0.5 * (dz * z' + z * dz')
    # dq = dz
    # dG = Diagonal(λ) * (dλ * z' + λ * dz') # was: Diagonal(λ) * dλ * z' - λ * dz')
    # dh = -Diagonal(λ) * dλ
    # dA = dν * z'+ ν * dz' # was: dν * z' - ν * dz'
    # db = -dν
    # todo, check MOI signs for dA and dG
end

_linsolve(A, b) = A \ b
# See https://github.com/JuliaLang/julia/issues/32668
_linsolve(A, b::SparseVector) = A \ Vector(b)

# Just a hack that will be removed once we use `MOIU.MatrixOfConstraints`
struct _QPSets end
MOI.Utilities.rows(::_QPSets, ci::MOI.ConstraintIndex) = ci.value

"""
    forward(model::QPDiff)
"""
function forward(model::QPDiff)
    gradient_cache = _gradient_cache(model)
    (
        Q, q, G, h, A, b, nz, var_list,
        nineq_le, le_con_idx,
        nineq_ge, ge_con_idx,
        nineq_sv_le, le_con_sv_idx,
        nineq_sv_ge, ge_con_sv_idx,
        neq, eq_con_idx,
        neq_sv, eq_con_sv_idx,
    ) = model.model
    z = gradient_cache.var_primals
    λ = gradient_cache.inequality_duals
    ν = gradient_cache.equality_duals
    LHS = gradient_cache.lhs

    objective_function = _convert(MOI.ScalarQuadraticFunction{Float64}, model.input_cache.objective)
    sparse_array_obj = sparse_array_representation(objective_function, LinearAlgebra.checksquare(Q))
    dQ = sparse_array_obj.quadratic_terms
    dq = sparse_array_obj.affine_terms

    # The user sets the constraint function in the sense `func`-in-`set` while
    # `db` and `dh` corresponds to the tangents of the set constants. Therefore,
    # we should multiply the constant by `-1`. For `GreaterThan`, we needed to
    # multiply by `-1` to transform it to `LessThan` so it cancels out.
    db = zeros(length(b))
    _fill(isequal(MOI.EqualTo{Float64}), (::Type{MOI.EqualTo{Float64}}) -> true, gradient_cache, model.input_cache, _QPSets(), db)
    dh = zeros(length(h))
    _fill(!isequal(MOI.EqualTo{Float64}), !isequal(MOI.GreaterThan{Float64}), gradient_cache, model.input_cache, _QPSets(), dh)

    nz = nnz(A)
    (lines, cols) = size(A)
    dAi = zeros(Int, 0)
    dAj = zeros(Int, 0)
    dAv = zeros(Float64, 0)
    sizehint!(dAi, nz)
    sizehint!(dAj, nz)
    sizehint!(dAv, nz)
    _fill(isequal(MOI.EqualTo{Float64}), isequal(MOI.GreaterThan{Float64}), gradient_cache, model.input_cache, _QPSets(), dAi, dAj, dAv)
    dA = sparse(dAi, dAj, dAv, lines, cols)

    nz = nnz(G)
    (lines, cols) = size(G)
    dGi = zeros(Int, 0)
    dGj = zeros(Int, 0)
    dGv = zeros(Float64, 0)
    sizehint!(dGi, nz)
    sizehint!(dGj, nz)
    sizehint!(dGv, nz)
    _fill(!isequal(MOI.EqualTo{Float64}), isequal(MOI.GreaterThan{Float64}), gradient_cache, model.input_cache, _QPSets(), dGi, dGj, dGv)
    dG = sparse(dGi, dGj, dGv, lines, cols)


    RHS = [
        dQ * z + dq + dG' * λ + dA' * ν
        λ .* (dG * z) - λ .* dh
        dA * z - db
    ]

    partial_grads = if norm(Q) ≈ 0
        -lsqr(LHS', RHS)
    else
        -_linsolve(LHS', RHS)
    end


    nv = length(z)
    nineq_total = nineq_le + nineq_ge + nineq_sv_le + nineq_sv_ge
    dz = partial_grads[1:nv]
    dλ = partial_grads[nv+1:nv+nineq_total]
    dν = partial_grads[nv+nineq_total+1:nv+nineq_total+neq+neq_sv]

    model.forw_grad_cache = QPForwBackCache(dz, dλ, dν)
    return nothing
end



_get_dA(model::QPDiff, ci) = _get_dA(model.back_grad_cache, model.gradient_cache, ci)
# quadratic matrix indexes are split by type either == or (<=/>=)
function _get_dA(b_cache::QPForwBackCache, g_cache::QPCache, ci::CI{F,S}
) where {F, S<:MOI.EqualTo}
    i = ci.value
    z = g_cache.var_primals
    dz = b_cache.dz
    ν = g_cache.equality_duals
    dν = b_cache.dν
    return lazy_combination(+, dν[i], z, ν[i], dz)
end
function _get_dA(b_cache::QPForwBackCache, g_cache::QPCache, ci::CI{F,S}
) where {F,S}
    i = ci.value
    z = g_cache.var_primals
    dz = b_cache.dz
    λ = g_cache.inequality_duals
    dλ = b_cache.dλ
    l = _neg_if_gt(λ[i], S)
    return lazy_combination(+, l * dλ[i], z, l, dz)
end
