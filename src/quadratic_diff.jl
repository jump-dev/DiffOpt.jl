MOI.Utilities.@product_of_sets(
    Equalities,
    MOI.EqualTo{T},
)

MOI.Utilities.@product_of_sets(
    Inequalities,
    MOI.LessThan{T},
)

MOI.Utilities.@struct_of_constraints_by_set_types(
    EqualitiesOrInequalities,
    MOI.EqualTo{T},
    MOI.LessThan{T},
)

const QPForm{T} = MOI.Utilities.GenericModel{
    T,
    MOI.Utilities.ObjectiveContainer{T},
    MOI.Utilities.VariablesContainer{T},
    EqualitiesOrInequalities{T}{
        MOI.Utilities.MatrixOfConstraints{
            T,
            MOI.Utilities.MutableSparseMatrixCSC{
                T,
                Int,
                MOI.Utilities.OneBasedIndexing,
            },
            Vector{T},
            MOI.Utilities.Hyperrectangle{Float64},
        },
        MOI.Utilities.MatrixOfConstraints{
            MOI.Utilities.MutableSparseMatrixCSC{
                T,
                Int,
                MOI.Utilities.OneBasedIndexing,
            },
            Vector{T},
            MOI.Utilities.Hyperrectangle{Float64},
        },
    },
}

mutable struct QPDiff <: DiffModel
    # storage for problem data in matrix form
    model::QPForm{Float64}
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
    return QPDiff(QPForm{Float64}(), nothing, nothing, nothing, nothing, DiffInputCache(), Float64[], Float64[], Float64[])
end

function MOI.empty!(model::QPDiff)
    MOI.empty!(model.model)
    model.gradient_cache = nothing
    model.forw_grad_cache = nothing
    model.back_grad_cache = nothing
    empty!(model.input_cache)
    empty!(model.x)
    empty!(model.λ)
    empty!(model.ν)
    return
end

const EQ = MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},MOI.EqualTo{Float64}}
const LE = MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},MOI.LessThan{Float64}}

function MOI.set(model::QPDiff, ::MOI.ConstraintPrimalStart, ci::MOI.ConstraintIndex, value)
    MOI.throw_if_not_valid(model, ci)
    return # Ignored
end

# We want to stay consistent with the variable `ν` defined in (3) of
# Left hand side of eq. (6) in https://arxiv.org/pdf/1703.00443.pdf
# However, in eq. (6), they put it in the lagrangian as
# `+ ν ⋅ (Az - b)`
# while in MOI, we put it as
# `- ν ⋅ (Az - b)`
# so the we should reverse the sign if we want to use the same equations
# as in the paper.
function MOI.set(model::QPDiff, ::MOI.ConstraintDualStart, ci::EQ, value)
    MOI.throw_if_not_valid(model, ci)
    _enlarge_set(model.ν, MOI.Utilities.rows(model.model.equalto.constraints, ci), -value)
end

function MOI.set(model::QPDiff, ::MOI.ConstraintDualStart, ci::LE, value)
    MOI.throw_if_not_valid(model, ci)
    _enlarge_set(model.λ, MOI.Utilities.rows(model.model.lessthan.constraints, ci), value)
end

function _gradient_cache(model::QPDiff)
    if model.gradient_cache !== nothing
        return model.gradient_cache
    end


    A = convert(SparseMatrixCSC{Float64, Int}, model.model.constraints.equalto.coefficients)
    b = model.model.constraints.equalto.constants.upper
    G = convert(SparseMatrixCSC{Float64, Int}, model.model.constraints.lessthan.coefficients)
    h = model.model.constraints.lessthan.constants.upper

    nz = size(A, 2)
    objective_function = MOI.get(model.model, MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}())
    sparse_array_obj = sparse_array_representation(objective_function, nz)
    Q = sparse_array_obj.quadratic_terms
    q = sparse_array_obj.affine_terms

    LHS = create_LHS_matrix(model.x, model.λ, Q, G, h, A)

    model.gradient_cache = QPCache(LHS)

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

function MOI.get(model::QPDiff, ::ForwardOutVariablePrimal, vi::VI)
    return model.forw_grad_cache.dz[vi.value]
end

function _get_db(model::QPDiff, ci::LE)
    i = ci.value
    # dh = -Diagonal(λ) * dλ
    return model.λ[i] * model.back_grad_cache.dλ[i]
end
function _get_db(model::QPDiff, ci::EQ)
    return model.back_grad_cache.dν[ci.value]
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
    LHS = gradient_cache.lhs

    neq = length(model.ν)
    nineq = length(model.λ)

    dl_dz = zeros(length(model.z))
    for (vi, value) in model.input_cache.dx
        dl_dz[vi.value] = value
    end

    RHS = [dl_dz; zeros(nineq + neq)]

    partial_grads = if norm(Q) ≈ 0
        -lsqr(LHS, RHS)
    else
        -LHS \ RHS
    end

    dz = partial_grads[1:nz]
    dλ = partial_grads[nz+1:nz+nineq]
    dν = partial_grads[nz+neq+1:end]

    model.back_grad_cache = QPForwBackCache(dz, dλ, dν)
    return
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
    LHS = gradient_cache.lhs

    objective_function = _convert(MOI.ScalarQuadraticFunction{Float64}, model.input_cache.objective)
    sparse_array_obj = sparse_array_representation(objective_function, LinearAlgebra.checksquare(Q))
    dQ = sparse_array_obj.quadratic_terms
    dq = sparse_array_obj.affine_terms

    # The user sets the constraint function in the sense `func`-in-`set` while
    # `db` and `dh` corresponds to the tangents of the set constants. Therefore,
    # we should multiply the constant by `-1`. For `GreaterThan`, we needed to
    # multiply by `-1` to transform it to `LessThan` so it cancels out.
    db = zero(model.ν)
    _fill(isequal(MOI.EqualTo{Float64}), (::Type{MOI.EqualTo{Float64}}) -> true, gradient_cache, model.input_cache, _QPSets(), db)
    dh = zero(model.λ)
    _fill(!isequal(MOI.EqualTo{Float64}), !isequal(MOI.GreaterThan{Float64}), gradient_cache, model.input_cache, _QPSets(), dh)

    nv = length(z)

    dAi = zeros(Int, 0)
    dAj = zeros(Int, 0)
    dAv = zeros(Float64, 0)
    _fill(isequal(MOI.EqualTo{Float64}), isequal(MOI.GreaterThan{Float64}), gradient_cache, model.input_cache, _QPSets(), dAi, dAj, dAv)
    dA = sparse(dAi, dAj, dAv, length(model.ν), nv)

    dGi = zeros(Int, 0)
    dGj = zeros(Int, 0)
    dGv = zeros(Float64, 0)
    _fill(!isequal(MOI.EqualTo{Float64}), isequal(MOI.GreaterThan{Float64}), gradient_cache, model.input_cache, _QPSets(), dGi, dGj, dGv)
    dG = sparse(dGi, dGj, dGv, length(model.λ), nv)


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


    dz = partial_grads[1:nv]
    dλ = partial_grads[nv+1:nv+length(λ)]
    dν = partial_grads[nv+length(λ)+1:end]

    model.forw_grad_cache = QPForwBackCache(dz, dλ, dν)
    return
end


# quadratic matrix indexes are split by type either == or <=
function _get_dA(model::QPDiff, ci::EQ)
    i = ci.value
    dz = model.back_grad_cache.dz
    dν = model.back_grad_cache.dν
    return lazy_combination(+, dν[i], model.z, model.ν[i], dz)
end
function _get_dA(model::QPDiff, ci::LE)
    i = ci.value
    dz = model.back_grad_cache.dz
    dλ = model.back_grad_cache.dλ
    l = model.λ[i]
    return lazy_combination(+, l * dλ[i], model.z, l, dz)
end
