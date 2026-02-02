# Copyright (c) 2025: Andrew Rosemberg and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

# This file is included inside `module NonLinearProgram`.

import MathOptInterface as MOI

const _B = MOI.Bridges

# A simple counter for unique operator symbols.
const _vno_op_counter = Ref(0)

# --------------------------------------------------------------------------
# Helpers: unwrap various MOI optimizer wrappers to reach the `Form` object,
# so we can call `MOI.Nonlinear.register_operator` on the underlying
# `MOI.Nonlinear.Model`.
# --------------------------------------------------------------------------
function _unwrap_to_form(m)
    if m isa Form
        return m
    elseif m isa Model
        return m.model
    end
    # Common wrappers in MOI: LazyBridgeOptimizer has a `model` field,
    # CachingOptimizer has an `optimizer` field. Use reflection to peel.
    T = typeof(m)
    if hasfield(T, :model)
        return _unwrap_to_form(getfield(m, :model))
    elseif hasfield(T, :optimizer)
        return _unwrap_to_form(getfield(m, :optimizer))
    end
    return error(
        "VectorNonlinearOracle bridge could not unwrap optimizer to NonLinearProgram.Form; got $(typeof(m))",
    )
end

_nl_model(m) = _unwrap_to_form(m).model  # this is the MOI.Nonlinear.Model

function _vno_op_symbol(base_id::Int, row::Int)
    return Symbol(:_diffopt_vno_, base_id, :_, row)
end

# --------------------------------------------------------------------------
# Register a scalar operator for one row of the vector oracle:
#   op_i(x...) = (oracle_f(x))[i]
# with gradient and Hessian (lower triangle) provided through the oracle.
#
# MOI.Nonlinear.register_operator docs:
# https://jump.dev/MathOptInterface.jl/stable/submodules/Nonlinear/reference/
#
# Note: For univariate functions (n=1), MOI expects:
#   f(x) -> Float64
#   ∇f(x) -> Float64  (the derivative, not a vector)
#   ∇²f(x) -> Float64 (the second derivative, not a matrix)
#
# For multivariate functions (n>1), MOI expects:
#   f(x...) -> Float64
#   ∇f(g::Vector, x...) -> nothing (fills g with gradient)
#   ∇²f(H::Matrix, x...) -> nothing (fills H with lower-triangular Hessian)
# --------------------------------------------------------------------------
function _register_vno_row_operator!(
    nlm::MOI.Nonlinear.Model,
    op::Symbol,
    s::MOI.VectorNonlinearOracle{T},
    row::Int,
) where {T<:Real}
    m = s.output_dimension
    n = s.input_dimension
    @assert length(s.l) == m
    @assert length(s.u) == m
    @assert 1 <= row <= m

    if n == 1
        # Univariate case: MOI expects scalar return values for derivatives
        _register_univariate_vno_row_operator!(nlm, op, s, row)
    else
        # Multivariate case: MOI expects in-place gradient/Hessian filling
        _register_multivariate_vno_row_operator!(nlm, op, s, row)
    end
    return
end

# Univariate version (n=1)
function _register_univariate_vno_row_operator!(
    nlm::MOI.Nonlinear.Model,
    op::Symbol,
    s::MOI.VectorNonlinearOracle{T},
    row::Int,
) where {T<:Real}
    m = s.output_dimension

    f = function (x::T)
        ret = Vector{T}(undef, m)
        s.eval_f(ret, [x])
        return ret[row]
    end

    # For univariate, gradient returns a scalar (the derivative df/dx)
    ∇f = function (x::T)
        vals = Vector{T}(undef, length(s.jacobian_structure))
        s.eval_jacobian(vals, [x])
        grad = zero(T)
        @inbounds for (k, (r, c)) in enumerate(s.jacobian_structure)
            if r == row && c == 1
                grad += vals[k]
            end
        end
        return grad
    end

    # For univariate, Hessian returns a scalar (the second derivative d²f/dx²)
    ∇²f = function (x::T)
        μ = zeros(T, m)
        μ[row] = one(T)
        vals = Vector{T}(undef, length(s.hessian_lagrangian_structure))
        s.eval_hessian_lagrangian(vals, [x], μ)
        hess = zero(T)
        @inbounds for (k, (r, c)) in enumerate(s.hessian_lagrangian_structure)
            if r == 1 && c == 1
                hess += vals[k]
            end
        end
        return hess
    end

    MOI.Nonlinear.register_operator(nlm, op, 1, f, ∇f, ∇²f)
    return
end

# Multivariate version (n>1)
function _register_multivariate_vno_row_operator!(
    nlm::MOI.Nonlinear.Model,
    op::Symbol,
    s::MOI.VectorNonlinearOracle{T},
    row::Int,
) where {T<:Real}
    m = s.output_dimension
    n = s.input_dimension

    f = function (x::T...)
        xv = collect(x)
        ret = Vector{T}(undef, m)
        s.eval_f(ret, xv)
        return ret[row]
    end

    ∇f = function (g::AbstractVector{T}, x::T...)
        fill!(g, zero(T))
        xv = collect(x)
        vals = Vector{T}(undef, length(s.jacobian_structure))
        s.eval_jacobian(vals, xv)
        @inbounds for (k, (r, c)) in enumerate(s.jacobian_structure)
            if r == row
                g[c] += vals[k]
            end
        end
        return
    end

    ∇²f = function (H::AbstractMatrix{T}, x::T...)
        # Note: H may be a lower-triangular view, so we can't use fill! 
        # or access upper triangular elements
        # Zero out only the lower triangular part
        for i in 1:n
            for j in 1:i
                H[i, j] = zero(T)
            end
        end
        # Hessian of op_row is Hessian of (μ' f) with μ = e_row
        xv = collect(x)
        μ = zeros(T, m)
        μ[row] = one(T)
        vals = Vector{T}(undef, length(s.hessian_lagrangian_structure))
        s.eval_hessian_lagrangian(vals, xv, μ)
        @inbounds for (k, (r, c)) in enumerate(s.hessian_lagrangian_structure)
            # MOI expects lower-triangular fill only
            if r >= c
                H[r, c] += vals[k]
            else
                # Transpose to lower triangular
                H[c, r] += vals[k]
            end
        end
        return
    end

    MOI.Nonlinear.register_operator(nlm, op, n, f, ∇f, ∇²f)
    return
end

# --------------------------------------------------------------------------
# The actual bridge type
# --------------------------------------------------------------------------
struct VNOToScalarNLBridge{T<:Real} <: _B.Constraint.AbstractBridge
    f::MOI.VectorOfVariables
    s::MOI.VectorNonlinearOracle{T}
    leq::Vector{
        MOI.ConstraintIndex{MOI.ScalarNonlinearFunction,MOI.LessThan{T}},
    }
    geq::Vector{
        MOI.ConstraintIndex{MOI.ScalarNonlinearFunction,MOI.GreaterThan{T}},
    }
    eq::Vector{MOI.ConstraintIndex{MOI.ScalarNonlinearFunction,MOI.EqualTo{T}}}
end

function MOI.supports_constraint(
    ::Type{VNOToScalarNLBridge{T}},
    ::Type{MOI.VectorOfVariables},
    ::Type{MOI.VectorNonlinearOracle{T}},
) where {T}
    return true
end

function _B.Constraint.concrete_bridge_type(
    ::Type{VNOToScalarNLBridge{T}},
    ::Type{MOI.VectorOfVariables},
    ::Type{MOI.VectorNonlinearOracle{T}},
) where {T}
    return VNOToScalarNLBridge{T}
end

function _B.added_constrained_variable_types(
    ::Type{VNOToScalarNLBridge{T}},
) where {T}
    return Tuple{Type}[]
end

function _B.added_constraint_types(::Type{VNOToScalarNLBridge{T}}) where {T}
    return Tuple{Type,Type}[
        (MOI.ScalarNonlinearFunction, MOI.LessThan{T}),
        (MOI.ScalarNonlinearFunction, MOI.GreaterThan{T}),
        (MOI.ScalarNonlinearFunction, MOI.EqualTo{T}),
    ]
end

function _B.Constraint.bridge_constraint(
    ::Type{VNOToScalarNLBridge{T}},
    model::MOI.ModelLike,
    f::MOI.VectorOfVariables,
    s::MOI.VectorNonlinearOracle{T},
) where {T<:Real}
    vars = f.variables
    @assert length(vars) == s.input_dimension
    m = s.output_dimension
    @assert length(s.l) == m
    @assert length(s.u) == m

    nlm = _nl_model(model)

    leq = MOI.ConstraintIndex{MOI.ScalarNonlinearFunction,MOI.LessThan{T}}[]
    geq = MOI.ConstraintIndex{MOI.ScalarNonlinearFunction,MOI.GreaterThan{T}}[]
    eq = MOI.ConstraintIndex{MOI.ScalarNonlinearFunction,MOI.EqualTo{T}}[]

    base_id = (_vno_op_counter[] += 1)

    for i in 1:m
        op = _vno_op_symbol(base_id, i)
        _register_vno_row_operator!(nlm, op, s, i)

        sf = MOI.ScalarNonlinearFunction(op, Any[vars...])

        li = s.l[i]
        ui = s.u[i]

        if isfinite(li) && isfinite(ui) && li == ui
            push!(eq, MOI.add_constraint(model, sf, MOI.EqualTo{T}(li)))
        else
            if isfinite(li)
                push!(
                    geq,
                    MOI.add_constraint(model, sf, MOI.GreaterThan{T}(li)),
                )
            end
            if isfinite(ui)
                push!(leq, MOI.add_constraint(model, sf, MOI.LessThan{T}(ui)))
            end
        end
    end

    return VNOToScalarNLBridge{T}(f, s, leq, geq, eq)
end

# Bridge transparency (optional but nice)
function MOI.get(
    ::MOI.ModelLike,
    ::MOI.ConstraintFunction,
    b::VNOToScalarNLBridge,
)
    return b.f
end
function MOI.get(::MOI.ModelLike, ::MOI.ConstraintSet, b::VNOToScalarNLBridge)
    return b.s
end

function MOI.delete(model::MOI.ModelLike, b::VNOToScalarNLBridge)
    for ci in b.leq
        MOI.delete(model, ci)
    end
    for ci in b.geq
        MOI.delete(model, ci)
    end
    for ci in b.eq
        MOI.delete(model, ci)
    end
    return
end

# Support for ConstraintPrimalStart (used for warm-starting/copying solution)
function MOI.supports(
    ::MOI.ModelLike,
    ::MOI.ConstraintPrimalStart,
    ::Type{VNOToScalarNLBridge{T}},
) where {T}
    return true
end

function MOI.set(
    model::MOI.ModelLike,
    ::MOI.ConstraintPrimalStart,
    b::VNOToScalarNLBridge{T},
    value::AbstractVector,
) where {T}
    # The value passed here is the ConstraintPrimal from VectorOfVariables in VNO,
    # which is the values of the variables x (not f(x)).
    # We need to evaluate f(x) and set the constraint primal for each scalar constraint.
    m = b.s.output_dimension
    n = b.s.input_dimension
    @assert length(value) == n  # value is x, not f(x)

    # Evaluate f(x)
    f_x = Vector{T}(undef, m)
    b.s.eval_f(f_x, value)

    # Now set the constraint primal for each bridged constraint
    # The constraints are created in order of output dimensions
    leq_idx = 1
    geq_idx = 1
    eq_idx = 1

    for i in 1:m
        li = b.s.l[i]
        ui = b.s.u[i]

        if isfinite(li) && isfinite(ui) && li == ui
            # Equality constraint
            if eq_idx <= length(b.eq)
                MOI.set(
                    model,
                    MOI.ConstraintPrimalStart(),
                    b.eq[eq_idx],
                    f_x[i],
                )
                eq_idx += 1
            end
        else
            if isfinite(li) && geq_idx <= length(b.geq)
                MOI.set(
                    model,
                    MOI.ConstraintPrimalStart(),
                    b.geq[geq_idx],
                    f_x[i],
                )
                geq_idx += 1
            end
            if isfinite(ui) && leq_idx <= length(b.leq)
                MOI.set(
                    model,
                    MOI.ConstraintPrimalStart(),
                    b.leq[leq_idx],
                    f_x[i],
                )
                leq_idx += 1
            end
        end
    end
    return
end

# Support for ConstraintDualStart (used for copying dual values)
function MOI.supports(
    ::MOI.ModelLike,
    ::MOI.ConstraintDualStart,
    ::Type{VNOToScalarNLBridge{T}},
) where {T}
    return true
end

function MOI.set(
    model::MOI.ModelLike,
    ::MOI.ConstraintDualStart,
    b::VNOToScalarNLBridge{T},
    value::AbstractVector,
) where {T}
    # For VectorOfVariables in VectorNonlinearOracle, the ConstraintDual from solvers 
    # like Ipopt has length = input_dimension (variables), not output_dimension.
    # This is because it represents the dual contribution per variable: J' * λ
    # where J is the Jacobian and λ is the vector of constraint duals.
    #
    # For our bridged scalar constraints, we need the dual per output (λ).
    # We can recover λ from the Jacobian: λ = (J * J')^{-1} * J * dual_per_var

    m = b.s.output_dimension
    n = b.s.input_dimension

    if length(value) == m
        # Direct mapping: value[i] is dual for output i
        _set_duals_from_output_duals!(model, b, value)
    elseif length(value) == n
        # The dual is per-input-variable (J' * λ), need to recover λ
        _set_duals_from_input_duals!(model, b, value)
    else
        # Unexpected size, skip
        return
    end
    return
end

function _set_duals_from_input_duals!(
    model::MOI.ModelLike,
    b::VNOToScalarNLBridge{T},
    dual_per_var::AbstractVector,
) where {T}
    m = b.s.output_dimension
    n = b.s.input_dimension

    # Get the current primal values to compute Jacobian
    x = zeros(T, n)
    for (i, vi) in enumerate(b.f.variables)
        x[i] = MOI.get(model, MOI.VariablePrimalStart(), vi)
    end

    # Compute the Jacobian at current point
    vals = Vector{T}(undef, length(b.s.jacobian_structure))
    b.s.eval_jacobian(vals, x)

    # Build the sparse Jacobian matrix J (m x n)
    J = zeros(T, m, n)
    for (k, (r, c)) in enumerate(b.s.jacobian_structure)
        J[r, c] = vals[k]
    end

    # Compute λ from dual_per_var = J' * λ
    # λ = (J * J')^{-1} * J * dual_per_var
    # For numerical stability, use least squares: λ = J' \ dual_per_var
    # which solves min ||J' * λ - dual_per_var||
    λ = J' \ dual_per_var

    # Now set the scalar constraint duals
    _set_duals_from_output_duals!(model, b, λ)
    return
end

function _set_duals_from_output_duals!(
    model::MOI.ModelLike,
    b::VNOToScalarNLBridge{T},
    value::AbstractVector,
) where {T}
    m = b.s.output_dimension

    leq_idx = 1
    geq_idx = 1
    eq_idx = 1

    for i in 1:m
        li = b.s.l[i]
        ui = b.s.u[i]

        if isfinite(li) && isfinite(ui) && li == ui
            # Equality constraint
            if eq_idx <= length(b.eq)
                MOI.set(
                    model,
                    MOI.ConstraintDualStart(),
                    b.eq[eq_idx],
                    value[i],
                )
                eq_idx += 1
            end
        else
            if isfinite(li) && geq_idx <= length(b.geq)
                MOI.set(
                    model,
                    MOI.ConstraintDualStart(),
                    b.geq[geq_idx],
                    value[i],
                )
                geq_idx += 1
            end
            if isfinite(ui) && leq_idx <= length(b.leq)
                # Note: For an interval constraint, the dual might need to be split
                # For now, we only set if there's no corresponding geq (i.e., only upper bound)
                if !isfinite(li)
                    MOI.set(
                        model,
                        MOI.ConstraintDualStart(),
                        b.leq[leq_idx],
                        value[i],
                    )
                end
                leq_idx += 1
            end
        end
    end
    return
end
