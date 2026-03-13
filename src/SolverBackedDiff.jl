# Copyright (c) 2020: Akshay Sharma and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

"""
    SolverBackedDiff

Module for integrating solvers that provide their own (native) differentiation.

Many convex optimization solvers cache matrix factorizations during the solve
that can be reused for the backward pass. This module provides a
`DiffOpt.AbstractModel` subtype that delegates the differentiation to the
solver rather than reconstructing the KKT system from scratch.

## Solver Interface

A solver opts in to native differentiation by implementing:

    DiffOpt.SolverBackedDiff.supports_native_differentiation(::MySolver) = true

Then it must implement the methods below. All MOI indices are in the
**solver's own index space**.

### Required for reverse mode

    DiffOpt.SolverBackedDiff.reverse_differentiate!(solver, dx, dy)

Perform the backward pass using cached factorizations. `dx` maps variable
indices to ∂l/∂x values; `dy` maps constraint indices to ∂l/∂λ values.

    DiffOpt.SolverBackedDiff.reverse_objective(solver)

Return objective sensitivity as `MOI.ScalarAffineFunction{Float64}`.

    DiffOpt.SolverBackedDiff.reverse_constraint(solver, ci)

Return constraint `ci` sensitivity as `MOI.ScalarAffineFunction{Float64}`.

### Optional for forward mode

    DiffOpt.SolverBackedDiff.forward_differentiate!(solver, dobj, dcons)
    DiffOpt.SolverBackedDiff.forward_primal(solver, vi) -> Float64
    DiffOpt.SolverBackedDiff.forward_dual(solver, ci) -> Float64

## Usage

```julia
# The solver just needs to implement the interface above.
# DiffOpt will detect it automatically — no user-side configuration needed.
model = DiffOpt.diff_optimizer(MySolver)
x = MOI.add_variables(model, n)
# ... add constraints, objective ...
MOI.optimize!(model)
MOI.set(model, DiffOpt.ReverseVariablePrimal(), x[1], 1.0)
DiffOpt.reverse_differentiate!(model)
dobj = MOI.get(model, DiffOpt.ReverseObjectiveFunction())
```
"""
module SolverBackedDiff

import DiffOpt
import MathOptInterface as MOI

# ─────────────────────────────────────────────────────────────────────────────
# Trait: does this solver support native differentiation?
# ─────────────────────────────────────────────────────────────────────────────

"""
    supports_native_differentiation(solver) -> Bool

Return `true` if `solver` implements the `SolverBackedDiff` interface.
Solvers opt in by defining a method:

    DiffOpt.SolverBackedDiff.supports_native_differentiation(::MySolver) = true
"""
supports_native_differentiation(::Any) = false

# ─────────────────────────────────────────────────────────────────────────────
# Solver interface (the solver must implement these)
# ─────────────────────────────────────────────────────────────────────────────

"""
    reverse_differentiate!(solver, dx, dy)

Perform the backward pass using the solver's cached factorizations.
"""
function reverse_differentiate! end

"""
    reverse_objective(solver) -> MOI.ScalarAffineFunction{Float64}

Return objective sensitivity after `reverse_differentiate!`.
"""
function reverse_objective end

"""
    reverse_constraint(solver, ci) -> MOI.ScalarAffineFunction{Float64}

Return constraint `ci` sensitivity after `reverse_differentiate!`.
"""
function reverse_constraint end

"""
    forward_differentiate!(solver, dobj, dcons)

Perform the forward pass. Optional; required only for forward mode.
"""
function forward_differentiate! end

"""
    forward_primal(solver, vi) -> Float64

Return primal tangent for variable `vi` after `forward_differentiate!`.
"""
function forward_primal end

"""
    forward_dual(solver, ci) -> Float64

Return dual tangent for constraint `ci` after `forward_differentiate!`.
"""
function forward_dual end

# ─────────────────────────────────────────────────────────────────────────────
# Unwrap the solver from DiffOpt/MOI wrapper layers
# ─────────────────────────────────────────────────────────────────────────────

"""
    _unwrap_solver(model) -> solver_or_nothing

Walk through CachingOptimizer, bridge, and POI layers to find the innermost
solver. Returns `nothing` if no natively-differentiable solver is found.
"""
function _unwrap_solver(model::MOI.Utilities.CachingOptimizer)
    return _unwrap_solver(model.optimizer)
end

function _unwrap_solver(model::MOI.Bridges.AbstractBridgeOptimizer)
    return _unwrap_solver(model.model)
end

function _unwrap_solver(model)
    # Check if this model has an `optimizer` field (e.g. POI.Optimizer)
    if hasfield(typeof(model), :optimizer)
        return _unwrap_solver(getfield(model, :optimizer))
    end
    # Leaf: check if it supports native differentiation
    if supports_native_differentiation(model)
        return model
    end
    return nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# SolverBackedDiff.Model — the DiffOpt AbstractModel implementation
# ─────────────────────────────────────────────────────────────────────────────

"""
    Model{S} <: DiffOpt.AbstractModel

A differentiation model that delegates to a solver's native backward/forward
pass instead of reconstructing the KKT system.
"""
mutable struct Model{S} <: DiffOpt.AbstractModel
    # Inner MOI model — receives the problem copy so that _diff()/copy_to works.
    model::MOI.Utilities.UniversalFallback{MOI.Utilities.Model{Float64}}

    # Direct reference to the solver.
    solver::S

    # Bidirectional index maps between our index space and the solver's.
    var_to_solver::Dict{MOI.VariableIndex,MOI.VariableIndex}
    var_from_solver::Dict{MOI.VariableIndex,MOI.VariableIndex}
    con_to_solver::Dict{MOI.ConstraintIndex,MOI.ConstraintIndex}
    con_from_solver::Dict{MOI.ConstraintIndex,MOI.ConstraintIndex}

    # Standard fields required by DiffOpt.AbstractModel
    input_cache::DiffOpt.InputCache
    x::Vector{Float64}
    diff_time::Float64
end

function Model(solver::S) where {S}
    return Model{S}(
        MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
        solver,
        Dict{MOI.VariableIndex,MOI.VariableIndex}(),
        Dict{MOI.VariableIndex,MOI.VariableIndex}(),
        Dict{MOI.ConstraintIndex,MOI.ConstraintIndex}(),
        Dict{MOI.ConstraintIndex,MOI.ConstraintIndex}(),
        DiffOpt.InputCache(),
        Float64[],
        NaN,
    )
end

# ── MOI required methods ─────────────────────────────────────────────────────

function MOI.is_empty(model::Model)
    return MOI.is_empty(model.model)
end

function MOI.empty!(model::Model)
    MOI.empty!(model.model)
    empty!(model.var_to_solver)
    empty!(model.var_from_solver)
    empty!(model.con_to_solver)
    empty!(model.con_from_solver)
    empty!(model.input_cache)
    empty!(model.x)
    model.diff_time = NaN
    return
end

MOI.get(model::Model, ::DiffOpt.DifferentiateTimeSec) = model.diff_time

# ── Index mapping ─────────────────────────────────────────────────────────────
# By default, identity mapping: SBM index VI(k) <-> solver index VI(k).
# This works when the solver uses sequential 1-based indexing and no bridges
# transform variable indices.

"""
    set_index_mapping!(model, var_map, con_map)

Set custom bidirectional index mappings between SBM and solver indices.
"""
function set_index_mapping!(
    model::Model,
    var_map::Dict{MOI.VariableIndex,MOI.VariableIndex},
    con_map::Dict{MOI.ConstraintIndex,MOI.ConstraintIndex},
)
    model.var_to_solver = var_map
    model.var_from_solver = Dict(v => k for (k, v) in var_map)
    model.con_to_solver = con_map
    model.con_from_solver = Dict(v => k for (k, v) in con_map)
    return
end

function _ensure_index_maps!(model::Model)
    if !isempty(model.var_to_solver)
        return
    end
    for vi in MOI.get(model.model, MOI.ListOfVariableIndices())
        model.var_to_solver[vi] = vi
        model.var_from_solver[vi] = vi
    end
    for (F, S) in MOI.get(model.model, MOI.ListOfConstraintTypesPresent())
        for ci in MOI.get(model.model, MOI.ListOfConstraintIndices{F,S}())
            model.con_to_solver[ci] = ci
            model.con_from_solver[ci] = ci
        end
    end
    return
end

# ── ConstraintDualStart / ConstraintPrimalStart ──────────────────────────────
# _copy_dual sets these; we ignore them since the solver already has the solution.

function MOI.set(model::Model, ::MOI.ConstraintDualStart, ci::MOI.ConstraintIndex, value)
    MOI.throw_if_not_valid(model, ci)
    return
end

function MOI.set(model::Model, ::MOI.ConstraintPrimalStart, ci::MOI.ConstraintIndex, value)
    MOI.throw_if_not_valid(model, ci)
    return
end

# ── Reverse differentiation ──────────────────────────────────────────────────

function DiffOpt.reverse_differentiate!(model::Model)
    _ensure_index_maps!(model)
    model.diff_time = @elapsed begin
        solver_dx = Dict{MOI.VariableIndex,Float64}()
        for (vi, val) in model.input_cache.dx
            solver_dx[model.var_to_solver[vi]] = val
        end
        solver_dy = Dict{MOI.ConstraintIndex,Float64}()
        for (ci, val) in model.input_cache.dy
            solver_dy[model.con_to_solver[ci]] = val
        end
        reverse_differentiate!(model.solver, solver_dx, solver_dy)
    end
    return
end

# ── Reverse output attributes ────────────────────────────────────────────────

function _to_vector_affine(model::Model, func::MOI.ScalarAffineFunction{Float64})
    n = length(model.x)
    if n == 0
        n = MOI.get(model.model, MOI.NumberOfVariables())
    end
    coeffs = zeros(n)
    for t in func.terms
        our_vi = model.var_from_solver[t.variable]
        coeffs[our_vi.value] += t.coefficient
    end
    return DiffOpt.VectorScalarAffineFunction(coeffs, func.constant)
end

function MOI.get(model::Model, ::DiffOpt.ReverseObjectiveFunction)
    _ensure_index_maps!(model)
    return _to_vector_affine(model, reverse_objective(model.solver))
end

function MOI.get(model::Model, ::DiffOpt.ReverseConstraintFunction, ci::MOI.ConstraintIndex)
    _ensure_index_maps!(model)
    solver_ci = model.con_to_solver[ci]
    return _to_vector_affine(model, reverse_constraint(model.solver, solver_ci))
end

# ── Forward input: override to bypass the Parameter check in AbstractModel ────

function MOI.set(
    model::Model,
    ::DiffOpt.ForwardConstraintFunction,
    ci::MOI.ConstraintIndex{MOI.ScalarAffineFunction{T},S},
    func::MOI.ScalarAffineFunction{T},
) where {T,S}
    model.input_cache.scalar_constraints[ci] = func
    return
end

# ── Forward differentiation ──────────────────────────────────────────────────

function DiffOpt.forward_differentiate!(model::Model)
    _ensure_index_maps!(model)
    model.diff_time = @elapsed begin
        solver_dobj = nothing
        if model.input_cache.objective !== nothing
            solver_dobj = _remap_affine_to_solver_space(model, model.input_cache.objective)
        end
        solver_dcons = Dict{MOI.ConstraintIndex,MOI.ScalarAffineFunction{Float64}}()
        for (ci, func) in model.input_cache.scalar_constraints
            solver_dcons[model.con_to_solver[ci]] =
                _remap_affine_to_solver_space(model, func)
        end
        forward_differentiate!(model.solver, solver_dobj, solver_dcons)
    end
    return
end

function _remap_affine_to_solver_space(model::Model, func::MOI.ScalarAffineFunction{Float64})
    new_terms = [MOI.ScalarAffineTerm(t.coefficient, model.var_to_solver[t.variable])
                 for t in func.terms]
    return MOI.ScalarAffineFunction(new_terms, func.constant)
end

# ── Forward output attributes ────────────────────────────────────────────────

function MOI.get(model::Model, ::DiffOpt.ForwardVariablePrimal, vi::MOI.VariableIndex)
    return forward_primal(model.solver, model.var_to_solver[vi])
end

function MOI.get(model::Model, ::DiffOpt.ForwardConstraintDual, ci::MOI.ConstraintIndex)
    return forward_dual(model.solver, model.con_to_solver[ci])
end

end # module SolverBackedDiff
