# Copyright (c) 2020: Akshay Sharma and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module BasisLinearProgram

import DiffOpt
import LinearAlgebra
import MathOptInterface as MOI
import ParametricOptInterface as POI
import SparseArrays

# ============================================================================
# Extension points: _supports_basis_solve + MOI-level _basis_solve/_basis_transpose_solve
# ============================================================================

"""
    _supports_basis_solve(optimizer) -> Bool

Returns whether `optimizer` supports native basis operations
(`_basis_solve`/`_basis_transpose_solve`). Defaults to `false`.
Solver extensions override this for their optimizer type.
"""
_supports_basis_solve(::Any) = false

"""
    _basis_solve(optimizer, db::Dict{MOI.ConstraintIndex, Float64}) -> Dict{MOI.VariableIndex, Float64}

Solve B * dx = db at the MOI level. Input `db` maps constraint indices to RHS
perturbations. Returns a dict mapping basic variable indices to their changes.

Each optimizer layer translates indices and forwards to the next layer.
The base case (e.g., HiGHS.Optimizer) does the actual computation.
"""
function _basis_solve end

"""
    _basis_transpose_solve(optimizer, dx::Dict{MOI.VariableIndex, Float64}) -> Dict{MOI.ConstraintIndex, Float64}

Solve B' * y = dx at the MOI level. Input `dx` maps variable indices to
adjoint values. Returns a dict mapping constraint indices to adjoint values.

Each optimizer layer translates indices and forwards to the next layer.
The base case (e.g., HiGHS.Optimizer) does the actual computation.
"""
function _basis_transpose_solve end

# ---------- Layer forwarding: CachingOptimizer ----------

function _supports_basis_solve(co::MOI.Utilities.CachingOptimizer)
    return _supports_basis_solve(co.optimizer)
end

# ---------- Layer forwarding: LazyBridgeOptimizer ----------

function _supports_basis_solve(opt::MOI.Bridges.LazyBridgeOptimizer)
    return _supports_basis_solve(opt.model)
end

# ---------- Layer forwarding: POI.Optimizer ----------

const POI = DiffOpt.POI

function _supports_basis_solve(opt::POI.Optimizer)
    return _supports_basis_solve(opt.optimizer)
end

# ============================================================================
# DirectModel — sentinel type for ModelConstructor
# ============================================================================

"""
    DirectModel

Sentinel type used as `ModelConstructor` value to select the
_SensitivityCache-based differentiation path for LPs with solver-native
basis operations. No longer stores state — all sensitivity I/O is handled
by `DiffOpt._SensitivityCache.Optimizer`.
"""
struct DirectModel end

# ============================================================================
# GeneralModel — standard copy path, queries basis via MOI API
# ============================================================================

"""
    GeneralModel <: DiffOpt.AbstractModel

Differentiation model for LPs using MOI-based basis queries and LU factorization.
Subtypes `DiffOpt.AbstractModel` — uses standard `copy_to` / `_copy_dual` path.
Works with any simplex solver that supports `MOI.ConstraintBasisStatus`.
"""
mutable struct GeneralModel <: DiffOpt.AbstractModel
    model::MOI.Utilities.Model{Float64}

    # Basis status (copied from optimizer at init via _copy_basis)
    var_basis_status::Dict{MOI.VariableIndex,MOI.BasisStatusCode}
    con_basis_status::Dict{MOI.ConstraintIndex,MOI.BasisStatusCode}

    # Problem data (built lazily at first differentiate)
    A::Union{Nothing,SparseArrays.SparseMatrixCSC{Float64,Int}}
    vi_list::Vector{MOI.VariableIndex}
    ci_list::Vector{MOI.ConstraintIndex}
    ci_set_type::Vector{Type}
    vi_to_col::Dict{MOI.VariableIndex,Int}
    ci_to_row::Dict{MOI.ConstraintIndex,Int}

    # Basis (computed at differentiate time)
    B_lu::Union{Nothing,LinearAlgebra.LU}
    basic_structural::Vector{Int}  # which columns of A are basic structural vars
    slack_basic::Vector{Int}       # which rows have basic slacks

    # Sensitivity I/O
    input_cache::DiffOpt.InputCache
    forw_dx::Union{Nothing,Dict{MOI.VariableIndex,Float64}}
    back_db::Union{Nothing,Dict{MOI.ConstraintIndex,Float64}}

    # Solution vectors (populated by _copy_dual)
    x::Vector{Float64}
    λ::Vector{Float64}   # dual of LessThan
    ν::Vector{Float64}   # dual of EqualTo

    diff_time::Float64
end

function GeneralModel()
    return GeneralModel(
        MOI.Utilities.Model{Float64}(),
        Dict{MOI.VariableIndex,MOI.BasisStatusCode}(),
        Dict{MOI.ConstraintIndex,MOI.BasisStatusCode}(),
        nothing,
        MOI.VariableIndex[],
        MOI.ConstraintIndex[],
        Type[],
        Dict{MOI.VariableIndex,Int}(),
        Dict{MOI.ConstraintIndex,Int}(),
        nothing,
        Int[],
        Int[],
        DiffOpt.InputCache(),
        nothing,
        nothing,
        Float64[],
        Float64[],
        Float64[],
        NaN,
    )
end

function MOI.is_empty(model::GeneralModel)
    return MOI.is_empty(model.model)
end

function MOI.empty!(model::GeneralModel)
    MOI.empty!(model.model)
    empty!(model.var_basis_status)
    empty!(model.con_basis_status)
    model.A = nothing
    empty!(model.vi_list)
    empty!(model.ci_list)
    empty!(model.ci_set_type)
    empty!(model.vi_to_col)
    empty!(model.ci_to_row)
    model.B_lu = nothing
    empty!(model.basic_structural)
    empty!(model.slack_basic)
    empty!(model.input_cache)
    model.forw_dx = nothing
    model.back_db = nothing
    empty!(model.x)
    empty!(model.λ)
    empty!(model.ν)
    model.diff_time = NaN
    return
end

MOI.get(model::GeneralModel, ::DiffOpt.DifferentiateTimeSec) = model.diff_time

# Reject quadratic objectives (LP only)
function MOI.supports(
    ::GeneralModel,
    ::MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}},
)
    return false
end

# Force POI wrapping when parameters are present (GeneralModel doesn't handle
# parametric differentiation natively — needs POI for ForwardConstraintSet translation)
function MOI.supports_add_constrained_variable(
    ::GeneralModel,
    ::Type{MOI.Parameter{T}},
) where {T}
    return false
end

# ---------- GeneralModel: Init data attributes ----------

# Basis status: store (set via _copy_basis)
function MOI.set(
    model::GeneralModel,
    ::DiffOpt._InputConstraintBasisStatus,
    ci::MOI.ConstraintIndex,
    status::MOI.BasisStatusCode,
)
    model.con_basis_status[ci] = status
    return
end

function MOI.set(
    model::GeneralModel,
    ::DiffOpt._InputVariableBasisStatus,
    vi::MOI.VariableIndex,
    status::MOI.BasisStatusCode,
)
    model.var_basis_status[vi] = status
    return
end

function MOI.supports(
    ::GeneralModel,
    ::DiffOpt._InputConstraintBasisStatus,
    ::Type{MOI.ConstraintIndex{F,S}},
) where {F,S}
    return true
end

function MOI.supports(
    ::GeneralModel,
    ::DiffOpt._InputVariableBasisStatus,
    ::Type{MOI.VariableIndex},
)
    return true
end

# ---------- GeneralModel: _copy_dual support ----------

function MOI.set(
    model::GeneralModel,
    ::MOI.ConstraintPrimalStart,
    ci::MOI.ConstraintIndex,
    value,
)
    MOI.throw_if_not_valid(model, ci)
    return  # Ignored
end

const EQ_CI =
    MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},MOI.EqualTo{Float64}}
const LE_CI =
    MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},MOI.LessThan{Float64}}
const GE_CI = MOI.ConstraintIndex{
    MOI.ScalarAffineFunction{Float64},
    MOI.GreaterThan{Float64},
}

function MOI.set(
    model::GeneralModel,
    ::MOI.ConstraintDualStart,
    ci::EQ_CI,
    value,
)
    MOI.throw_if_not_valid(model, ci)
    return DiffOpt._enlarge_set(model.ν, ci.value, -value)
end

function MOI.set(
    model::GeneralModel,
    ::MOI.ConstraintDualStart,
    ci::LE_CI,
    value,
)
    MOI.throw_if_not_valid(model, ci)
    return DiffOpt._enlarge_set(model.λ, ci.value, -value)
end

function MOI.set(
    model::GeneralModel,
    ::MOI.ConstraintDualStart,
    ci::GE_CI,
    value,
)
    MOI.throw_if_not_valid(model, ci)
    return  # Not used for basis sensitivity
end

function MOI.set(
    model::GeneralModel,
    ::MOI.ConstraintDualStart,
    ci::MOI.ConstraintIndex{MOI.VariableIndex},
    value,
)
    MOI.throw_if_not_valid(model, ci)
    return  # Not used for basis sensitivity
end

# ---------- GeneralModel: Sensitivity I/O ----------

# Override ForwardConstraintFunction to skip Parameter check
function MOI.set(
    model::GeneralModel,
    ::DiffOpt.ForwardConstraintFunction,
    ci::MOI.ConstraintIndex{MOI.ScalarAffineFunction{T},S},
    func::MOI.ScalarAffineFunction{T},
) where {T,S}
    model.input_cache.scalar_constraints[ci] = func
    return
end

# Unsupported attributes
function MOI.set(::GeneralModel, ::DiffOpt.ReverseObjectiveSensitivity, ::Any)
    return throw(
        MOI.UnsupportedAttribute(DiffOpt.ReverseObjectiveSensitivity()),
    )
end

function MOI.get(::GeneralModel, ::DiffOpt.ForwardObjectiveSensitivity)
    return throw(
        MOI.UnsupportedAttribute(DiffOpt.ForwardObjectiveSensitivity()),
    )
end

# Output getters
function MOI.get(
    model::GeneralModel,
    ::DiffOpt.ForwardVariablePrimal,
    vi::MOI.VariableIndex,
)
    model.forw_dx === nothing && return 0.0
    return get(model.forw_dx, vi, 0.0)
end

function MOI.get(model::GeneralModel, ::DiffOpt.ReverseObjectiveFunction)
    return DiffOpt.VectorScalarAffineFunction(zeros(length(model.x)), 0.0)
end

function DiffOpt._get_db(model::GeneralModel, ci::MOI.ConstraintIndex)
    model.back_db === nothing && return 0.0
    return get(model.back_db, ci, 0.0)
end

function DiffOpt._get_dA(model::GeneralModel, ::MOI.ConstraintIndex)
    return DiffOpt.VectorScalarAffineFunction(zeros(length(model.x)), 0.0)
end

# ============================================================================
# GeneralModel: A matrix extraction and basis identification
# ============================================================================

"""
    _build_A!(model::GeneralModel)

Lazily extract the constraint matrix A from `model.model`.
Populates `vi_list`, `ci_list`, `ci_set_type`, `vi_to_col`, `ci_to_row`, and `A`.
"""
function _build_A!(model::GeneralModel)
    model.A !== nothing && return
    inner = model.model

    # Collect variables
    vis = MOI.get(inner, MOI.ListOfVariableIndices())
    model.vi_list = vis
    n = length(vis)
    empty!(model.vi_to_col)
    for (j, vi) in enumerate(vis)
        model.vi_to_col[vi] = j
    end

    # Collect SAF constraints and build sparse A
    empty!(model.ci_list)
    empty!(model.ci_set_type)
    empty!(model.ci_to_row)
    I_vec = Int[]
    J_vec = Int[]
    V_vec = Float64[]
    row = 0
    for (F, S) in MOI.get(inner, MOI.ListOfConstraintTypesPresent())
        F <: MOI.ScalarAffineFunction || continue
        (
            S <: Union{
                MOI.EqualTo{Float64},
                MOI.LessThan{Float64},
                MOI.GreaterThan{Float64},
            }
        ) || continue
        for ci in MOI.get(inner, MOI.ListOfConstraintIndices{F,S}())
            row += 1
            push!(model.ci_list, ci)
            push!(model.ci_set_type, S)
            model.ci_to_row[ci] = row
            func = MOI.get(inner, MOI.ConstraintFunction(), ci)
            for term in func.terms
                j = model.vi_to_col[term.variable]
                push!(I_vec, row)
                push!(J_vec, j)
                push!(V_vec, term.coefficient)
            end
        end
    end
    m = row
    model.A = SparseArrays.sparse(I_vec, J_vec, V_vec, m, n)
    return
end

"""
    _validate_basis!(model::GeneralModel)

Validate that basis status has been set, classify variables into basic
structural and basic slack, and check that the total count matches the
number of constraints. Call after `_build_A!`.

Throws an error if:
- No basis status is available (neither variable nor constraint)
- The total number of basic variables doesn't match the number of constraints
- An EqualTo constraint has a basic slack
"""
function _validate_basis!(model::GeneralModel)
    if isempty(model.var_basis_status) && isempty(model.con_basis_status)
        error(
            "GeneralModel: basis status not set. " *
            "Ensure _copy_basis was called after copy_to.",
        )
    end

    m = length(model.ci_list)

    # Determine basic structural variables
    empty!(model.basic_structural)
    for (j, vi) in enumerate(model.vi_list)
        status = get(model.var_basis_status, vi, MOI.NONBASIC)
        if status == MOI.BASIC
            push!(model.basic_structural, j)
        end
    end

    # Determine basic slacks (SAF constraints where slack is basic)
    empty!(model.slack_basic)
    for (i, ci) in enumerate(model.ci_list)
        status = get(model.con_basis_status, ci, MOI.NONBASIC)
        if status == MOI.BASIC
            push!(model.slack_basic, i)
        end
    end

    n_basic = length(model.basic_structural) + length(model.slack_basic)
    if n_basic != m
        error(
            "Basis size mismatch: expected $m basic variables, " *
            "got $(length(model.basic_structural)) structural + " *
            "$(length(model.slack_basic)) slack = $n_basic",
        )
    end
    return
end

"""
    _slack_coefficient(S::Type) -> Float64

Return the slack variable coefficient for constraint set type `S`.
LessThan uses `+1` (a'x + s = b), GreaterThan uses `-1` (a'x - s = b).
EqualTo constraints have no slack and throw an error.
"""
function _slack_coefficient(S::Type)
    if S <: MOI.LessThan
        return 1.0
    elseif S <: MOI.GreaterThan
        return -1.0
    else
        error("EqualTo constraint should not have a basic slack")
    end
end

"""
    _ensure_basis!(model::GeneralModel)

Lazily identify the basis from optimizer_ref and build/factorize B.
"""
function _ensure_basis!(model::GeneralModel)
    model.B_lu !== nothing && return
    _build_A!(model)
    _validate_basis!(model)

    # Build basis matrix B = [A[:, basic_structural] | slack_columns]
    m = length(model.ci_list)
    B_structural = model.A[:, model.basic_structural]
    if !isempty(model.slack_basic)
        n_slack = length(model.slack_basic)
        slack_rows = Int[]
        slack_cols = Int[]
        slack_vals = Float64[]
        for (k, row_idx) in enumerate(model.slack_basic)
            push!(slack_rows, row_idx)
            push!(slack_cols, k)
            push!(slack_vals, _slack_coefficient(model.ci_set_type[row_idx]))
        end
        B_slack =
            SparseArrays.sparse(slack_rows, slack_cols, slack_vals, m, n_slack)
        B = hcat(B_structural, B_slack)
    else
        B = B_structural
    end

    model.B_lu = LinearAlgebra.lu(Matrix(B))
    return
end

include("forward.jl")
include("reverse.jl")

end # module
