# Copyright (c) 2020: Akshay Sharma and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module _SensitivityCache

import DiffOpt
import MathOptInterface as MOI

const BLP = DiffOpt.BasisLinearProgram
const MOIDD = MOI.Utilities.DoubleDicts

# ============================================================================
# Optimizer struct
# ============================================================================

"""
    Optimizer{OT} <: MOI.AbstractOptimizer

A thin wrapper around a solver (e.g., HiGHS.Optimizer) that caches DiffOpt
sensitivity attributes and implements `forward_differentiate!`/`reverse_differentiate!`
by calling `_basis_solve`/`_basis_transpose_solve` on the inner solver.

The user explicitly wraps their solver:
```julia
optimizer = DiffOpt._SensitivityCache.Optimizer(HiGHS.Optimizer())
```
"""
mutable struct Optimizer{OT} <: MOI.AbstractOptimizer
    optimizer::OT

    # Forward input: ForwardConstraintFunction (set via MOI chain)
    scalar_constraints::MOIDD.DoubleDict{MOI.ScalarAffineFunction{Float64}}

    # Reverse input: ReverseVariablePrimal (set via MOI chain)
    dx::Dict{MOI.VariableIndex,Float64}

    # Forward output
    forw_dx::Union{Nothing,Dict{MOI.VariableIndex,Float64}}

    # Reverse output
    back_db::Union{Nothing,Dict{MOI.ConstraintIndex,Float64}}

    diff_time::Float64
end

function Optimizer(optimizer)
    return Optimizer(
        optimizer,
        MOIDD.DoubleDict{MOI.ScalarAffineFunction{Float64}}(),
        Dict{MOI.VariableIndex,Float64}(),
        nothing,
        nothing,
        NaN,
    )
end

# ============================================================================
# MOI passthrough: core operations
# ============================================================================

MOI.optimize!(s::Optimizer) = MOI.optimize!(s.optimizer)

function MOI.copy_to(s::Optimizer, src::MOI.ModelLike)
    return MOI.copy_to(s.optimizer, src)
end

function MOI.empty!(s::Optimizer)
    MOI.empty!(s.optimizer)
    empty!(s.scalar_constraints)
    empty!(s.dx)
    s.forw_dx = nothing
    s.back_db = nothing
    s.diff_time = NaN
    return
end

MOI.is_empty(s::Optimizer) = MOI.is_empty(s.optimizer)

# ============================================================================
# MOI passthrough: model attributes
# ============================================================================

function MOI.get(s::Optimizer, attr::MOI.AbstractModelAttribute)
    return MOI.get(s.optimizer, attr)
end

function MOI.set(s::Optimizer, attr::MOI.AbstractModelAttribute, value)
    return MOI.set(s.optimizer, attr, value)
end

function MOI.supports(s::Optimizer, attr::MOI.AbstractModelAttribute)
    return MOI.supports(s.optimizer, attr)
end

# ============================================================================
# MOI passthrough: optimizer attributes
# ============================================================================

function MOI.get(s::Optimizer, attr::MOI.AbstractOptimizerAttribute)
    return MOI.get(s.optimizer, attr)
end

function MOI.set(s::Optimizer, attr::MOI.AbstractOptimizerAttribute, value)
    return MOI.set(s.optimizer, attr, value)
end

function MOI.supports(s::Optimizer, attr::MOI.AbstractOptimizerAttribute)
    return MOI.supports(s.optimizer, attr)
end

# ============================================================================
# MOI passthrough: variable attributes
# ============================================================================

function MOI.get(
    s::Optimizer,
    attr::MOI.AbstractVariableAttribute,
    vi::MOI.VariableIndex,
)
    return MOI.get(s.optimizer, attr, vi)
end

function MOI.set(
    s::Optimizer,
    attr::MOI.AbstractVariableAttribute,
    vi::MOI.VariableIndex,
    value,
)
    return MOI.set(s.optimizer, attr, vi, value)
end

function MOI.supports(
    s::Optimizer,
    attr::MOI.AbstractVariableAttribute,
    ::Type{MOI.VariableIndex},
)
    return MOI.supports(s.optimizer, attr, MOI.VariableIndex)
end

# ============================================================================
# MOI passthrough: constraint attributes
# ============================================================================

function MOI.get(
    s::Optimizer,
    attr::MOI.AbstractConstraintAttribute,
    ci::MOI.ConstraintIndex,
)
    return MOI.get(s.optimizer, attr, ci)
end

function MOI.set(
    s::Optimizer,
    attr::MOI.AbstractConstraintAttribute,
    ci::MOI.ConstraintIndex,
    value,
)
    return MOI.set(s.optimizer, attr, ci, value)
end

function MOI.supports(
    s::Optimizer,
    attr::MOI.AbstractConstraintAttribute,
    ::Type{MOI.ConstraintIndex{F,S}},
) where {F,S}
    return MOI.supports(s.optimizer, attr, MOI.ConstraintIndex{F,S})
end

# ============================================================================
# MOI passthrough: other
# ============================================================================

function MOI.supports_constraint(
    s::Optimizer,
    ::Type{F},
    ::Type{S},
) where {F<:MOI.AbstractFunction,S<:MOI.AbstractSet}
    return MOI.supports_constraint(s.optimizer, F, S)
end

function MOI.get(
    s::Optimizer,
    attr::MOI.ListOfConstraintIndices{F,S},
) where {F,S}
    return MOI.get(s.optimizer, attr)
end

function MOI.get(s::Optimizer, ::MOI.ListOfVariableIndices)
    return MOI.get(s.optimizer, MOI.ListOfVariableIndices())
end

function MOI.get(s::Optimizer, ::MOI.ListOfConstraintTypesPresent)
    return MOI.get(s.optimizer, MOI.ListOfConstraintTypesPresent())
end

function MOI.is_valid(s::Optimizer, vi::MOI.VariableIndex)
    return MOI.is_valid(s.optimizer, vi)
end

function MOI.is_valid(s::Optimizer, ci::MOI.ConstraintIndex)
    return MOI.is_valid(s.optimizer, ci)
end

function MOI.get(s::Optimizer, ::MOI.NumberOfVariables)
    return MOI.get(s.optimizer, MOI.NumberOfVariables())
end

# ============================================================================
# Absorb DiffOpt model-level attributes (not relevant for basis differentiation)
# ============================================================================

function MOI.set(
    ::Optimizer,
    ::DiffOpt.NonLinearKKTJacobianFactorization,
    ::Any,
)
    return nothing
end

function MOI.set(::Optimizer, ::DiffOpt.AllowObjectiveAndSolutionInput, ::Any)
    return nothing
end

# ============================================================================
# BasisLinearProgram extension points
# ============================================================================

BLP._supports_basis_solve(s::Optimizer) = BLP._supports_basis_solve(s.optimizer)

# ============================================================================
# DiffOpt attribute setters (input)
# ============================================================================

function MOI.set(
    s::Optimizer,
    ::DiffOpt.ForwardConstraintFunction,
    ci::MOI.ConstraintIndex{MOI.ScalarAffineFunction{T},S},
    func::MOI.ScalarAffineFunction{T},
) where {T,S}
    s.scalar_constraints[ci] = func
    return
end

function MOI.set(
    s::Optimizer,
    ::DiffOpt.ReverseVariablePrimal,
    vi::MOI.VariableIndex,
    value,
)
    s.dx[vi] = value
    return
end

# ============================================================================
# DiffOpt attribute getters (output)
# ============================================================================

function MOI.get(
    s::Optimizer,
    ::DiffOpt.ForwardVariablePrimal,
    vi::MOI.VariableIndex,
)
    s.forw_dx === nothing && return 0.0
    return get(s.forw_dx, vi, 0.0)
end

MOI.get(s::Optimizer, ::DiffOpt.DifferentiateTimeSec) = s.diff_time

function MOI.get(::Optimizer, ::DiffOpt.ReverseObjectiveFunction)
    return MOI.ScalarAffineFunction(MOI.ScalarAffineTerm{Float64}[], 0.0)
end

function MOI.get(::Optimizer, ::DiffOpt.ForwardObjectiveSensitivity)
    return throw(
        MOI.UnsupportedAttribute(DiffOpt.ForwardObjectiveSensitivity()),
    )
end

function DiffOpt._get_db(s::Optimizer, ci::MOI.ConstraintIndex)
    s.back_db === nothing && return 0.0
    return get(s.back_db, ci, 0.0)
end

function MOI.get(
    s::Optimizer,
    ::DiffOpt.ReverseConstraintFunction,
    ci::MOI.ConstraintIndex,
)
    db = DiffOpt._get_db(s, ci)
    return MOI.ScalarAffineFunction(MOI.ScalarAffineTerm{Float64}[], db)
end

# ============================================================================
# forward_differentiate! / reverse_differentiate!
# ============================================================================

function DiffOpt.forward_differentiate!(s::Optimizer)
    s.diff_time = @elapsed begin
        # Build db from cached ForwardConstraintFunction entries
        db = Dict{MOI.ConstraintIndex,Float64}()
        for (F, S) in MOIDD.nonempty_outer_keys(s.scalar_constraints)
            for (ci, func) in s.scalar_constraints[F, S]
                if !isempty(func.terms)
                    error(
                        "BasisLinearProgram: constraint coefficient " *
                        "perturbation (dA) is not supported.",
                    )
                end
                db[ci] = -MOI.constant(func)
            end
        end

        if !isempty(db)
            s.forw_dx = BLP._basis_solve(s.optimizer, db)
        else
            s.forw_dx = Dict{MOI.VariableIndex,Float64}()
        end
    end
    return
end

function DiffOpt.reverse_differentiate!(s::Optimizer)
    s.diff_time = @elapsed begin
        if !isempty(s.dx)
            raw_db = BLP._basis_transpose_solve(s.optimizer, s.dx)
            s.back_db = Dict{MOI.ConstraintIndex,Float64}()
            for (ci, val) in raw_db
                s.back_db[ci] = -val  # sign: dL/d(func_constant) = -y
            end
        else
            s.back_db = Dict{MOI.ConstraintIndex,Float64}()
        end
    end
    return
end

# ============================================================================
# empty_input_sensitivities!
# ============================================================================

function DiffOpt.empty_input_sensitivities!(s::Optimizer)
    empty!(s.scalar_constraints)
    empty!(s.dx)
    s.forw_dx = nothing
    s.back_db = nothing
    return
end

end # module
