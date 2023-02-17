module ConicProgram

using LinearAlgebra, SparseArrays

using MathOptInterface
const MOI = MathOptInterface

import BlockDiagonals
import IterativeSolvers

import DiffOpt

Base.@kwdef struct Cache
    M::SparseMatrixCSC{Float64, Int}
    vp::Vector{Float64}
    Dπv::BlockDiagonals.BlockDiagonal{Float64, Matrix{Float64}}
    A::SparseMatrixCSC{Float64, Int}
    b::Vector{Float64}
    c::Vector{Float64}
end

Base.@kwdef struct ForwCache
    du::Vector{Float64}
    dv::Vector{Float64}
    dw::Vector{Float64}
end
Base.@kwdef struct ReverseCache
    g::Vector{Float64}
    πz::Vector{Float64}
end

# Geometric conic standard form
const Form{T} = MOI.Utilities.GenericModel{
    T,
    MOI.Utilities.ObjectiveContainer{T},
    MOI.Utilities.FreeVariables,
    MOI.Utilities.MatrixOfConstraints{
        T,
        MOI.Utilities.MutableSparseMatrixCSC{
            T,
            Int,
            # We use `OneBasedIndexing` as it is the same indexing as used
            # by `SparseMatrixCSC` so we can do an allocation-free conversion to
            # `SparseMatrixCSC`.
            MOI.Utilities.OneBasedIndexing,
        },
        Vector{T},
        DiffOpt.ProductOfSets{T},
    },
}

"""
    Diffopt.ConicProgram.Model <: DiffOpt.AbstractModel

Model to differentiate conic programs.

The forward differentiation computes the product of the derivative (Jacobian) at the
conic program parameters `A`, `b`, `c` to the perturbations `dA`, `db`, `dc`.

The reverse differentiation computes the product of the transpose of the derivative (Jacobian) at the
conic program parameters `A`, `b`, `c` to the perturbations `dx`, `dy`, `ds`.

For theoretical background, refer Section 3 of Differentiating Through a Cone Program, https://arxiv.org/abs/1904.09043
"""
mutable struct Model <: DiffOpt.AbstractModel
    # storage for problem data in matrix form
    model::Form{Float64}
    # includes maps from matrix indices to problem data held in `optimizer`
    # also includes KKT matrices
    # also includes the solution
    gradient_cache::Union{Nothing,Cache}

    # caches for sensitivity output
    # result from solving KKT/residualmap linear systems
    # this allows keeping the same `gradient_cache`
    # if only sensitivy input changes
    forw_grad_cache::Union{Nothing,ForwCache}
    back_grad_cache::Union{Nothing,ReverseCache}

    # sensitivity input cache using MOI like sparse format
    input_cache::DiffOpt.InputCache

    x::Vector{Float64} # Primal
    s::Vector{Float64} # Slack
    y::Vector{Float64} # Dual
    diff_time::Float64
end
function Model()
    return Model(Form{Float64}(), nothing, nothing, nothing, DiffOpt.InputCache(), Float64[], Float64[], Float64[])
end

function MOI.is_empty(model::Model)
    return MOI.is_empty(model.model)
end

function MOI.empty!(model::Model)
    MOI.empty!(model.model)
    model.gradient_cache = nothing
    model.forw_grad_cache = nothing
    model.back_grad_cache = nothing
    empty!(model.input_cache)
    empty!(model.x)
    empty!(model.s)
    empty!(model.y)
    model.diff_time = NaN
    return
end

MOI.get(model::Model, ::DifferentiateTimeSec) = model.diff_time

function MOI.supports_constraint(model::Model, F::Type{MOI.VectorAffineFunction{Float64}}, ::Type{S}) where {S<:MOI.AbstractVectorSet}
    if DiffOpt.add_set_types(model.model.constraints.sets, S)
        push!(model.model.constraints.caches, Tuple{F,S}[])
        push!(model.model.constraints.are_indices_mapped, BitSet())
    end
    return MOI.supports_constraint(model.model, F, S)
end

function MOI.set(model::Model, ::MOI.ConstraintPrimalStart, ci::MOI.ConstraintIndex, value)
    MOI.throw_if_not_valid(model, ci)
    DiffOpt._enlarge_set(model.s, MOI.Utilities.rows(model.model.constraints, ci), value)
end

function MOI.set(model::Model, ::MOI.ConstraintDualStart, ci::MOI.ConstraintIndex, value)
    MOI.throw_if_not_valid(model, ci)
    DiffOpt._enlarge_set(model.y, MOI.Utilities.rows(model.model.constraints, ci), value)
end

function _gradient_cache(model::Model)
    if model.gradient_cache !== nothing
        return model.gradient_cache
    end

    # For theoretical background, refer Section 3 of Differentiating Through a Cone Program, https://arxiv.org/abs/1904.09043

    A = -convert(SparseMatrixCSC{Float64, Int}, model.model.constraints.coefficients)
    b = model.model.constraints.constants

    if MOI.get(model, MOI.ObjectiveSense()) == MOI.FEASIBILITY_SENSE
        c = spzeros(size(A, 2))
    else
        obj = MOI.get(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}())
        c = DiffOpt.sparse_array_representation(obj, size(A, 2)).terms
        if MOI.get(model, MOI.ObjectiveSense()) == MOI.MAX_SENSE
            c = -c
        end
    end

    # programs in tests were cross-checked against `diffcp`, which follows SCS format
    # hence, some arrays saved during `MOI.optimize!` are not same across all optimizers
    # specifically there's an extra preprocessing step for `PositiveSemidefiniteConeTriangle` constraint for SCS/Mosek

    # pre-compute quantities for the derivative
    m = A.m
    n = A.n
    N = m + n + 1
    # NOTE: w = 1.0 systematically since we asserted the primal-dual pair is optimal
    (u, v, w) = (model.x, model.y - model.s, 1.0)


    # find gradient of projections on dual of the cones
    Dπv = DiffOpt.Dπ(v, model.model, model.model.constraints.sets)

    # Q = [
    #      0   A'   c;
    #     -A   0    b;
    #     -c' -b'   0;
    # ]
    # M = (Q- I) * B + I
    # with B =
    # [
    #  I    .   .    # Πx = x because x is a solution and hence satistfies the constraints
    #  .  Dπv   .
    #  .    .   1    # w >= 0, but in the solution x = 1
    # ]
    # see: https://stanford.edu/~boyd/papers/pdf/cone_prog_refine.pdf
    # for the definition of Π and why we get I and 1 for x and w respectectively
    # K is defined in (5), Π in sect 2, and projection sin sect 3

    M = [
        spzeros(n,n)     (A' * Dπv)    c
        -A               -Dπv + I      b
        -c'              -b' * Dπv     0.0
    ]
    # find projections on dual of the cones
    vp = DiffOpt.π(v, model.model, model.model.constraints.sets)

    model.gradient_cache = Cache(
        M = M,
        vp = vp,
        Dπv = Dπv,
        A = A,
        b = b,
        c = c,
    )

    return model.gradient_cache
end

function DiffOpt.forward_differentiate!(model::Model)
    model.diff_time = @elapsed begin
        gradient_cache = _gradient_cache(model)
        M = gradient_cache.M
        vp = gradient_cache.vp
        Dπv = gradient_cache.Dπv
        x = model.x
        y = model.y
        s = model.s
        A = gradient_cache.A
        b = gradient_cache.b
        c = gradient_cache.c

        objective_function = DiffOpt._convert(MOI.ScalarAffineFunction{Float64}, model.input_cache.objective)
        sparse_array_obj = DiffOpt.sparse_array_representation(objective_function, length(c))
        dc = sparse_array_obj.terms

        db = zeros(length(b))
        DiffOpt._fill(S -> false, gradient_cache, model.input_cache, model.model.constraints.sets, db)
        (lines, cols) = size(A)
        nz = nnz(A)
        dAi = zeros(Int, 0)
        dAj = zeros(Int, 0)
        dAv = zeros(Float64, 0)
        sizehint!(dAi, nz)
        sizehint!(dAj, nz)
        sizehint!(dAv, nz)
        DiffOpt._fill(S -> false, gradient_cache, model.input_cache, model.model.constraints.sets, dAi, dAj, dAv)
        dA = sparse(dAi, dAj, dAv, lines, cols)

        m = size(A, 1)
        n = size(A, 2)
        N = m + n + 1
        # NOTE: w = 1 systematically since we asserted the primal-dual pair is optimal
        (u, v, w) = (x, y - s, 1.0)

        # g = dQ * Π(z/|w|) = dQ * [u, vp, 1.0]
        RHS = [dA' * vp + dc; -dA * u + db; -dc ⋅ u - db ⋅ vp]

        dz = if norm(RHS) <= 1e-400 # TODO: parametrize or remove
            RHS .= 0 # because M is square
        else
            IterativeSolvers.lsqr(M, RHS)
        end

        du, dv, dw = dz[1:n], dz[n+1:n+m], dz[n+m+1]
        model.forw_grad_cache = ForwCache(du, dv, [dw])
    end
    return nothing
    # dx = du - x * dw
    # dy = Dπv * dv - y * dw
    # ds = Dπv * dv - dv - s * dw
    # return -dx, -dy, -ds
end

function DiffOpt.reverse_differentiate!(model::Model)
    model.diff_time = @elapsed begin
        gradient_cache = _gradient_cache(model)
        M = gradient_cache.M
        vp = gradient_cache.vp
        Dπv = gradient_cache.Dπv
        x = model.x
        y = model.y
        s = model.s
        A = gradient_cache.A
        b = gradient_cache.b
        c = gradient_cache.c

        dx = zeros(length(c))
        for (vi, value) in model.input_cache.dx
            dx[vi.value] = value
        end
        dy = zeros(length(b))
        ds = zeros(length(b))

        m = size(A, 1)
        n = size(A, 2)
        N = m + n + 1
        # NOTE: w = 1 systematically since we asserted the primal-dual pair is optimal
        (u, v, w) = (x, y - s, 1.0)

        # dz = D \phi (z)^T (dx,dy,dz)
        dz = [
            dx
            Dπv' * (dy + ds) - ds
            - x' * dx - y' * dy - s' * ds
        ]

        g = if norm(dz) <= 1e-4 # TODO: parametrize or remove
            dz .= 0 # because M is square
        else
            IterativeSolvers.lsqr(M, dz)
        end

        πz = [
            u
            vp
            1.0
        ]

        # TODO: very important
        # contrast with:
        # http://reports-archive.adm.cs.cmu.edu/anon/2019/CMU-CS-19-109.pdf
        # pg 97, cap 7.4.2

        model.back_grad_cache = ReverseCache(g, πz)
    end
    return nothing
    # dQ = - g * πz'
    # dA = - dQ[1:n, n+1:n+m]' + dQ[n+1:n+m, 1:n]
    # db = - dQ[n+1:n+m, end] + dQ[end, n+1:n+m]'
    # dc = - dQ[1:n, end] + dQ[end, 1:n]'
    # return dA, db, dc
end

function MOI.get(model::Model, ::DiffOpt.ReverseObjectiveFunction)
    g = model.back_grad_cache.g
    πz = model.back_grad_cache.πz
    dc = DiffOpt.lazy_combination(-, πz, g, length(g))
    return DiffOpt.VectorScalarAffineFunction(dc, 0.0)
end

function MOI.get(model::Model, ::DiffOpt.ForwardVariablePrimal, vi::MOI.VariableIndex)
    i = vi.value
    du = model.forw_grad_cache.du
    dw = model.forw_grad_cache.dw
    return -(du[i] - model.x[i] * dw[])
end
function DiffOpt._get_db(model::Model, ci::MOI.ConstraintIndex{F,S}
) where {F<:MOI.AbstractVectorFunction,S}
    i = MOI.Utilities.rows(model.model.constraints, ci) # vector
    # i = ci.value
    n = length(model.x) # columns in A
    # db = - dQ[n+1:n+m, end] + dQ[end, n+1:n+m]'
    g = model.back_grad_cache.g
    πz = model.back_grad_cache.πz
    return DiffOpt.lazy_combination(-, πz, g, length(g), n .+ i)
end
function DiffOpt._get_dA(model::Model, ci::MOI.ConstraintIndex{<:MOI.AbstractVectorFunction})
    i = MOI.Utilities.rows(model.model.constraints, ci) # vector
    # i = ci.value
    n = length(model.x) # columns in A
    m = length(model.y) # lines in A
    # dA = - dQ[1:n, n+1:n+m]' + dQ[n+1:n+m, 1:n]
    g = model.back_grad_cache.g
    πz = model.back_grad_cache.πz
    return DiffOpt.lazy_combination(-, g, πz, i, n .+ (1:n))
end

end
