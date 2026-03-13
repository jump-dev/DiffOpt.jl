# Example: Using SolverBackedDiff with a solver that has native differentiation
#
# This example shows how a solver author would integrate their natively
# differentiable solver with DiffOpt.jl.
#
# We use an equality-constrained QP where the KKT system is a single linear
# system factorized once during the solve and reused exactly for the backward
# pass.
#
#   min  (1/2) x'Qx + c'x
#   s.t. Ax = b
#
# KKT system:
#   [Q  A'] [x] = [-c]
#   [A  0 ] [ν]   [ b]

using LinearAlgebra
import MathOptInterface as MOI

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Define the solver with standard MOI interface
# ─────────────────────────────────────────────────────────────────────────────

mutable struct EqQPSolver <: MOI.AbstractOptimizer
    Q::Matrix{Float64}
    c::Vector{Float64}
    A::Matrix{Float64}
    b::Vector{Float64}
    x::Vector{Float64}
    ν::Vector{Float64}
    kkt_factor::Any  # ★ cached factorization ★
    n_vars::Int
    n_cons::Int
    var_indices::Vector{MOI.VariableIndex}
    con_indices::Vector{MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},MOI.EqualTo{Float64}}}
    status::MOI.TerminationStatusCode
    dc::Vector{Float64}
    dQ::Matrix{Float64}
    dA::Matrix{Float64}
    db::Vector{Float64}

    function EqQPSolver()
        return new(
            zeros(0, 0), Float64[], zeros(0, 0), Float64[],
            Float64[], Float64[], nothing,
            0, 0, MOI.VariableIndex[],
            MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},MOI.EqualTo{Float64}}[],
            MOI.OPTIMIZE_NOT_CALLED,
            Float64[], zeros(0, 0), zeros(0, 0), Float64[],
        )
    end
end

const EQ_CI = MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},MOI.EqualTo{Float64}}

# ── Standard MOI boilerplate ─────────────────────────────────────────────────

function MOI.is_empty(s::EqQPSolver)
    return s.n_vars == 0 && s.n_cons == 0
end

function MOI.empty!(s::EqQPSolver)
    s.Q = zeros(0, 0); s.c = Float64[]; s.A = zeros(0, 0); s.b = Float64[]
    s.x = Float64[]; s.ν = Float64[]; s.kkt_factor = nothing
    s.n_vars = 0; s.n_cons = 0
    empty!(s.var_indices); empty!(s.con_indices)
    s.status = MOI.OPTIMIZE_NOT_CALLED
    s.dc = Float64[]; s.dQ = zeros(0, 0); s.dA = zeros(0, 0); s.db = Float64[]
    return
end

function MOI.add_variable(s::EqQPSolver)
    s.n_vars += 1
    vi = MOI.VariableIndex(s.n_vars)
    push!(s.var_indices, vi)
    n = s.n_vars
    Q_new = zeros(n, n)
    Q_new[1:n-1, 1:n-1] .= s.Q
    s.Q = Q_new
    push!(s.c, 0.0)
    s.A = s.n_cons > 0 ? hcat(s.A, zeros(s.n_cons)) : zeros(0, n)
    return vi
end

MOI.supports_constraint(::EqQPSolver, ::Type{MOI.ScalarAffineFunction{Float64}}, ::Type{MOI.EqualTo{Float64}}) = true

function MOI.add_constraint(s::EqQPSolver, func::MOI.ScalarAffineFunction{Float64}, set::MOI.EqualTo{Float64})
    s.n_cons += 1
    new_row = zeros(1, s.n_vars)
    for term in func.terms
        new_row[1, term.variable.value] += term.coefficient
    end
    s.A = vcat(s.A, new_row)
    push!(s.b, set.value - func.constant)
    ci = EQ_CI(s.n_cons)
    push!(s.con_indices, ci)
    return ci
end

MOI.supports(::EqQPSolver, ::MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}) = true
MOI.supports(::EqQPSolver, ::MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}) = true

function MOI.set(s::EqQPSolver, ::MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}, func::MOI.ScalarQuadraticFunction{Float64})
    fill!(s.Q, 0.0); fill!(s.c, 0.0)
    for t in func.quadratic_terms
        i, j = t.variable_1.value, t.variable_2.value
        s.Q[i, j] += t.coefficient
        if i != j
            s.Q[j, i] += t.coefficient
        end
    end
    for t in func.affine_terms
        s.c[t.variable.value] += t.coefficient
    end
    return
end

function MOI.set(s::EqQPSolver, ::MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}, func::MOI.ScalarAffineFunction{Float64})
    fill!(s.Q, 0.0); fill!(s.c, 0.0)
    for t in func.terms
        s.c[t.variable.value] += t.coefficient
    end
    return
end

MOI.supports(::EqQPSolver, ::MOI.ObjectiveSense) = true
MOI.set(::EqQPSolver, ::MOI.ObjectiveSense, ::MOI.OptimizationSense) = nothing
MOI.get(::EqQPSolver, ::MOI.ObjectiveSense) = MOI.MIN_SENSE
MOI.get(s::EqQPSolver, ::MOI.TerminationStatus) = s.status
MOI.get(s::EqQPSolver, ::MOI.PrimalStatus) = s.status == MOI.OPTIMAL ? MOI.FEASIBLE_POINT : MOI.NO_SOLUTION
MOI.get(s::EqQPSolver, ::MOI.DualStatus) = s.status == MOI.OPTIMAL ? MOI.FEASIBLE_POINT : MOI.NO_SOLUTION
MOI.get(s::EqQPSolver, ::MOI.ResultCount) = s.status == MOI.OPTIMAL ? 1 : 0
MOI.get(s::EqQPSolver, ::MOI.RawStatusString) = string(s.status)
MOI.get(s::EqQPSolver, ::MOI.ObjectiveValue) = 0.5 * dot(s.x, s.Q * s.x) + dot(s.c, s.x)
MOI.get(::EqQPSolver, ::MOI.SolverName) = "EqQPSolver"
MOI.get(s::EqQPSolver, ::MOI.VariablePrimal, vi::MOI.VariableIndex) = s.x[vi.value]
MOI.get(s::EqQPSolver, ::MOI.ConstraintDual, ci::EQ_CI) = s.ν[ci.value]
MOI.get(s::EqQPSolver, ::MOI.ConstraintPrimal, ci::EQ_CI) = dot(s.A[ci.value, :], s.x)
MOI.get(s::EqQPSolver, ::MOI.ListOfVariableIndices) = copy(s.var_indices)
MOI.get(s::EqQPSolver, ::MOI.ListOfConstraintIndices{MOI.ScalarAffineFunction{Float64},MOI.EqualTo{Float64}}) = copy(s.con_indices)
MOI.get(s::EqQPSolver, ::MOI.NumberOfVariables) = s.n_vars
MOI.is_valid(s::EqQPSolver, vi::MOI.VariableIndex) = 1 <= vi.value <= s.n_vars
MOI.is_valid(s::EqQPSolver, ci::EQ_CI) = 1 <= ci.value <= s.n_cons

function MOI.get(s::EqQPSolver, ::MOI.ListOfConstraintTypesPresent)
    return s.n_cons > 0 ? [(MOI.ScalarAffineFunction{Float64}, MOI.EqualTo{Float64})] : Tuple{Type,Type}[]
end

function MOI.get(s::EqQPSolver, ::MOI.ConstraintFunction, ci::EQ_CI)
    row = ci.value
    terms = [MOI.ScalarAffineTerm(s.A[row, j], MOI.VariableIndex(j))
             for j in 1:s.n_vars if !iszero(s.A[row, j])]
    return MOI.ScalarAffineFunction(terms, 0.0)
end

MOI.get(s::EqQPSolver, ::MOI.ConstraintSet, ci::EQ_CI) = MOI.EqualTo(s.b[ci.value])
MOI.get(::EqQPSolver, ::MOI.ObjectiveFunctionType) = MOI.ScalarAffineFunction{Float64}

function MOI.get(s::EqQPSolver, ::MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}})
    terms = [MOI.ScalarAffineTerm(s.c[j], MOI.VariableIndex(j))
             for j in 1:s.n_vars if !iszero(s.c[j])]
    return MOI.ScalarAffineFunction(terms, 0.0)
end

function MOI.copy_to(dest::EqQPSolver, src::MOI.ModelLike)
    MOI.empty!(dest)
    index_map = MOI.Utilities.IndexMap()
    for vi in MOI.get(src, MOI.ListOfVariableIndices())
        index_map[vi] = MOI.add_variable(dest)
    end
    obj_type = MOI.get(src, MOI.ObjectiveFunctionType())
    obj_func = MOI.Utilities.map_indices(index_map, MOI.get(src, MOI.ObjectiveFunction{obj_type}()))
    if obj_func isa MOI.ScalarQuadraticFunction{Float64}
        MOI.set(dest, MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(), obj_func)
    elseif obj_func isa MOI.ScalarAffineFunction{Float64}
        MOI.set(dest, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), obj_func)
    end
    MOI.set(dest, MOI.ObjectiveSense(), MOI.get(src, MOI.ObjectiveSense()))
    for (F, S) in MOI.get(src, MOI.ListOfConstraintTypesPresent())
        for ci in MOI.get(src, MOI.ListOfConstraintIndices{F,S}())
            func = MOI.Utilities.map_indices(index_map, MOI.get(src, MOI.ConstraintFunction(), ci))
            set = MOI.get(src, MOI.ConstraintSet(), ci)
            index_map[ci] = MOI.add_constraint(dest, func, set)
        end
    end
    return index_map
end

# ── The solve: factorize KKT once, cache it ─────────────────────────────────

function MOI.optimize!(s::EqQPSolver)
    n, m = s.n_vars, s.n_cons
    At = transpose(s.A)
    K = vcat(hcat(s.Q, At), hcat(s.A, zeros(m, m)))
    rhs = vcat(-s.c, s.b)

    # ★ Factorize once — reused for backward pass ★
    s.kkt_factor = lu(K)
    sol = s.kkt_factor \ rhs

    s.x = sol[1:n]
    s.ν = sol[n+1:end]
    s.status = MOI.OPTIMAL
    return
end

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Implement the SolverBackedDiff interface
#
# This is ALL a solver author needs to add on top of their existing solver.
# ─────────────────────────────────────────────────────────────────────────────

import DiffOpt

# Opt in to native differentiation:
DiffOpt.SolverBackedDiff.supports_native_differentiation(::EqQPSolver) = true

function DiffOpt.SolverBackedDiff.reverse_differentiate!(
    s::EqQPSolver,
    dx::Dict{MOI.VariableIndex,Float64},
    dy::Dict{MOI.ConstraintIndex,Float64},
)
    n, m = s.n_vars, s.n_cons

    rhs = zeros(n + m)
    for (vi, val) in dx
        rhs[vi.value] = val
    end
    for (ci, val) in dy
        rhs[n + ci.value] = val
    end

    # ★ Reuse the cached factorization! ★
    adj = s.kkt_factor \ rhs
    dx_adj = adj[1:n]
    dν_adj = adj[n+1:end]

    # Implicit function theorem on KKT:
    #   F₁ = Qx + A'ν + c = 0,   F₂ = Ax - b = 0
    #   dl/dp = -adj' * ∂F/∂p
    s.dc = -dx_adj
    s.db = dν_adj
    s.dA = -(s.ν * dx_adj' + dν_adj * s.x')
    s.dQ = -0.5 * (dx_adj * s.x' + s.x * dx_adj')
    return
end

function DiffOpt.SolverBackedDiff.reverse_objective(s::EqQPSolver)
    terms = [MOI.ScalarAffineTerm(s.dc[j], MOI.VariableIndex(j))
             for j in 1:s.n_vars]
    return MOI.ScalarAffineFunction(terms, 0.0)
end

function DiffOpt.SolverBackedDiff.reverse_constraint(s::EqQPSolver, ci::EQ_CI)
    row = ci.value
    terms = [MOI.ScalarAffineTerm(s.dA[row, j], MOI.VariableIndex(j))
             for j in 1:s.n_vars]
    return MOI.ScalarAffineFunction(terms, -s.db[row])
end

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Use it — no special configuration needed!
# ─────────────────────────────────────────────────────────────────────────────

function run_example()
    # Just pass the solver to diff_optimizer. DiffOpt detects native
    # differentiation support automatically.
    model = DiffOpt.diff_optimizer(EqQPSolver)

    # Problem:
    #   min  (1/2)(x₁² + x₂²) + 3x₁ + 2x₂
    #   s.t. x₁ + x₂ = 1
    #
    # Solution: x₁ = 0, x₂ = 1, ν = -3
    x1 = MOI.add_variable(model)
    x2 = MOI.add_variable(model)

    c1 = MOI.add_constraint(model, 1.0 * x1 + 1.0 * x2, MOI.EqualTo(1.0))

    obj = MOI.ScalarQuadraticFunction(
        [MOI.ScalarQuadraticTerm(1.0, x1, x1),
         MOI.ScalarQuadraticTerm(1.0, x2, x2)],
        [MOI.ScalarAffineTerm(3.0, x1),
         MOI.ScalarAffineTerm(2.0, x2)],
        0.0,
    )
    MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(), obj)
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    MOI.optimize!(model)

    println("Optimal x₁ = ", MOI.get(model, MOI.VariablePrimal(), x1))
    println("Optimal x₂ = ", MOI.get(model, MOI.VariablePrimal(), x2))
    println("Objective  = ", MOI.get(model, MOI.ObjectiveValue()))

    # Reverse differentiate — uses the solver's cached KKT factorization
    MOI.set(model, DiffOpt.ReverseVariablePrimal(), x1, 1.0)
    DiffOpt.reverse_differentiate!(model)

    dobj = MOI.get(model, DiffOpt.ReverseObjectiveFunction())
    println("\nReverse mode sensitivities (∂l/∂x₁ = 1):")
    println("  ∂l/∂c₁ = ", JuMP.coefficient(dobj, x1))
    println("  ∂l/∂c₂ = ", JuMP.coefficient(dobj, x2))

    dcon1 = MOI.get(model, DiffOpt.ReverseConstraintFunction(), c1)
    println("  ∂l/∂A₁₁ = ", JuMP.coefficient(dcon1, x1))
    println("  ∂l/∂A₁₂ = ", JuMP.coefficient(dcon1, x2))
    println("  ∂l/∂b₁  = ", -MOI.constant(dcon1))

    return model
end

if abspath(PROGRAM_FILE) == @__FILE__
    using JuMP
    run_example()
end
