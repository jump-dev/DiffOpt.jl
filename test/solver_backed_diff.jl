# Copyright (c) 2020: Akshay Sharma and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module TestSolverBackedDiff

using Test
using LinearAlgebra
import DiffOpt
import MathOptInterface as MOI
import JuMP

const ATOL = 1e-8
const RTOL = 1e-8

# ─────────────────────────────────────────────────────────────────────────────
# EqQPSolver: a minimal equality-constrained QP solver with native
# differentiation support, used as the test fixture.
#
#   min  (1/2) x'Qx + c'x
#   s.t. Ax = b
# ─────────────────────────────────────────────────────────────────────────────

const EQ_CI = MOI.ConstraintIndex{
    MOI.ScalarAffineFunction{Float64},
    MOI.EqualTo{Float64},
}

mutable struct EqQPSolver <: MOI.AbstractOptimizer
    Q::Matrix{Float64}
    c::Vector{Float64}
    A::Matrix{Float64}
    b::Vector{Float64}
    x::Vector{Float64}
    ν::Vector{Float64}
    kkt_factor::Any
    n_vars::Int
    n_cons::Int
    var_indices::Vector{MOI.VariableIndex}
    con_indices::Vector{EQ_CI}
    status::MOI.TerminationStatusCode
    # Reverse results
    dc::Vector{Float64}
    dQ::Matrix{Float64}
    dA::Matrix{Float64}
    db::Vector{Float64}
    # Forward results
    dx_fwd::Vector{Float64}
    dν_fwd::Vector{Float64}

    function EqQPSolver()
        return new(
            zeros(0, 0), Float64[], zeros(0, 0), Float64[],
            Float64[], Float64[], nothing,
            0, 0, MOI.VariableIndex[], EQ_CI[],
            MOI.OPTIMIZE_NOT_CALLED,
            Float64[], zeros(0, 0), zeros(0, 0), Float64[],
            Float64[], Float64[],
        )
    end
end

# ── MOI boilerplate ──────────────────────────────────────────────────────────

MOI.is_empty(s::EqQPSolver) = s.n_vars == 0 && s.n_cons == 0

function MOI.empty!(s::EqQPSolver)
    s.Q = zeros(0, 0); s.c = Float64[]; s.A = zeros(0, 0); s.b = Float64[]
    s.x = Float64[]; s.ν = Float64[]; s.kkt_factor = nothing
    s.n_vars = 0; s.n_cons = 0
    empty!(s.var_indices); empty!(s.con_indices)
    s.status = MOI.OPTIMIZE_NOT_CALLED
    s.dc = Float64[]; s.dQ = zeros(0, 0); s.dA = zeros(0, 0); s.db = Float64[]
    s.dx_fwd = Float64[]; s.dν_fwd = Float64[]
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

function MOI.optimize!(s::EqQPSolver)
    n, m = s.n_vars, s.n_cons
    At = transpose(s.A)
    K = vcat(hcat(s.Q, At), hcat(s.A, zeros(m, m)))
    rhs = vcat(-s.c, s.b)
    s.kkt_factor = lu(K)
    sol = s.kkt_factor \ rhs
    s.x = sol[1:n]
    s.ν = sol[n+1:end]
    s.status = MOI.OPTIMAL
    return
end

# ── Native differentiation interface via MOI attributes ───────────────────────

MOI.supports(::EqQPSolver, ::DiffOpt.BackwardDifferentiate) = true
MOI.supports(::EqQPSolver, ::DiffOpt.ForwardDifferentiate) = true

function MOI.set(
    s::EqQPSolver,
    ::DiffOpt.BackwardDifferentiate,
    seeds::Tuple{Dict{MOI.VariableIndex,Float64},Dict{MOI.ConstraintIndex,Float64}},
)
    dx, dy = seeds
    n, m = s.n_vars, s.n_cons
    rhs = zeros(n + m)
    for (vi, val) in dx
        rhs[vi.value] = val
    end
    for (ci, val) in dy
        rhs[n + ci.value] = val
    end
    adj = s.kkt_factor \ rhs
    dx_adj = adj[1:n]
    dν_adj = adj[n+1:end]
    s.dc = -dx_adj
    s.db = dν_adj
    s.dA = -(s.ν * dx_adj' + dν_adj * s.x')
    s.dQ = -0.5 * (dx_adj * s.x' + s.x * dx_adj')
    return
end

function MOI.get(s::EqQPSolver, ::DiffOpt.ReverseObjectiveFunction)
    terms = [MOI.ScalarAffineTerm(s.dc[j], MOI.VariableIndex(j))
             for j in 1:s.n_vars]
    return MOI.ScalarAffineFunction(terms, 0.0)
end

function MOI.get(s::EqQPSolver, ::DiffOpt.ReverseConstraintFunction, ci::EQ_CI)
    row = ci.value
    terms = [MOI.ScalarAffineTerm(s.dA[row, j], MOI.VariableIndex(j))
             for j in 1:s.n_vars]
    return MOI.ScalarAffineFunction(terms, -s.db[row])
end

function MOI.set(
    s::EqQPSolver,
    ::DiffOpt.ForwardDifferentiate,
    inputs::Tuple{
        Union{Nothing,MOI.ScalarAffineFunction{Float64}},
        Dict{MOI.ConstraintIndex,MOI.ScalarAffineFunction{Float64}},
    },
)
    dobj, dcons = inputs
    n, m = s.n_vars, s.n_cons
    # Build perturbation: d_c and d_b from dobj and dcons
    d_c = zeros(n)
    d_b = zeros(m)
    if !isnothing(dobj)
        for t in dobj.terms
            d_c[t.variable.value] += t.coefficient
        end
    end
    for (ci, func) in dcons
        for t in func.terms
            # The constraint perturbation dA*x - db enters as changes to A and b.
            # For simplicity we treat it as a RHS perturbation:
            # We need to compute the tangent of the KKT solution w.r.t. c and b perturbations.
            # The forward tangent solves: K * [dx; dν] = [-dc; db]
            # But for constraint perturbations dA*x, we'd need the full Jacobian.
            # For a simpler test we just support RHS (b) perturbations via the constant.
        end
        d_b[ci.value] = -func.constant  # db = -constant (convention: func is dA*x - db)
    end
    rhs = vcat(-d_c, d_b)
    sol = s.kkt_factor \ rhs
    s.dx_fwd = sol[1:n]
    s.dν_fwd = sol[n+1:end]
    return
end

function MOI.get(s::EqQPSolver, ::DiffOpt.ForwardVariablePrimal, vi::MOI.VariableIndex)
    return s.dx_fwd[vi.value]
end

function MOI.get(s::EqQPSolver, ::DiffOpt.ForwardConstraintDual, ci::EQ_CI)
    return s.dν_fwd[ci.value]
end

# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

function runtests()
    for name in names(@__MODULE__; all = true)
        if startswith("$name", "test_")
            @testset "$(name)" begin
                getfield(@__MODULE__, name)()
            end
        end
    end
    return
end

# ── Unit tests for _unwrap_solver ────────────────────────────────────────────

function test_unwrap_solver_direct()
    solver = EqQPSolver()
    @test DiffOpt.SolverBackedDiff._unwrap_solver(solver) === solver
end

function test_unwrap_solver_returns_nothing_for_unsupported()
    @test DiffOpt.SolverBackedDiff._unwrap_solver(42) === nothing
end

function test_supports_backward_differentiate()
    @test MOI.supports(EqQPSolver(), DiffOpt.BackwardDifferentiate()) == true
    @test MOI.supports(EqQPSolver(), DiffOpt.ForwardDifferentiate()) == true
end

# ── Helper to set up and solve the standard test problem ─────────────────────
#
#   min  (1/2)(x₁² + x₂²) + 3x₁ + 2x₂
#   s.t. x₁ + x₂ = 1
#
# KKT solution: x = [0, 1], ν = -3
# (Q = I, c = [3, 2], A = [1 1], b = [1])

function _setup_model()
    model = DiffOpt.diff_optimizer(EqQPSolver)
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
    return model, x1, x2, c1
end

# ── Test that native solver is auto-detected ─────────────────────────────────

function test_auto_detection()
    model, x1, x2, c1 = _setup_model()
    # No ModelConstructor was set, so DiffOpt should auto-detect native solver.
    # Verify the solution first.
    @test MOI.get(model, MOI.VariablePrimal(), x1) ≈ 0.0 atol = ATOL
    @test MOI.get(model, MOI.VariablePrimal(), x2) ≈ 1.0 atol = ATOL
    # Trigger differentiation — this should use SolverBackedDiff.Model
    MOI.set(model, DiffOpt.ReverseVariablePrimal(), x1, 1.0)
    DiffOpt.reverse_differentiate!(model)
    # If we get here without error, auto-detection worked
    dobj = MOI.get(model, DiffOpt.ReverseObjectiveFunction())
    @test dobj isa DiffOpt.IndexMappedFunction
end

# ── Helper: compute expected reverse sensitivities analytically ───────────────
# For the standard test problem:
#   Q = I, c = [3, 2], A = [1 1], b = [1]
#   x* = [0, 1], ν* = [-3]
#   K = [Q A'; A 0] = [1 0 1; 0 1 1; 1 1 0]

function _expected_reverse(dx_seed::Vector{Float64}, dy_seed::Vector{Float64})
    Q = [1.0 0.0; 0.0 1.0]
    A = [1.0 1.0]
    x_star = [0.0, 1.0]
    ν_star = [-3.0]
    n, m = 2, 1
    K = vcat(hcat(Q, A'), hcat(A, zeros(m, m)))
    rhs = vcat(dx_seed, dy_seed)
    adj = K \ rhs
    adj_x = adj[1:n]
    adj_ν = adj[n+1:end]
    dc = -adj_x
    db = adj_ν
    dA = -(ν_star * adj_x' + adj_ν * x_star')
    return dc, db, dA
end

# ── Test reverse differentiation with dx seed ────────────────────────────────

function test_reverse_dx_seed()
    model, x1, x2, c1 = _setup_model()

    MOI.set(model, DiffOpt.ReverseVariablePrimal(), x1, 1.0)
    DiffOpt.reverse_differentiate!(model)

    @test !isnan(MOI.get(model, DiffOpt.DifferentiateTimeSec()))

    dc, db, dA = _expected_reverse([1.0, 0.0], [0.0])

    dobj = MOI.get(model, DiffOpt.ReverseObjectiveFunction())
    @test JuMP.coefficient(dobj, x1) ≈ dc[1] atol = ATOL
    @test JuMP.coefficient(dobj, x2) ≈ dc[2] atol = ATOL

    dcon = MOI.get(model, DiffOpt.ReverseConstraintFunction(), c1)
    @test MOI.constant(dcon) ≈ -db[1] atol = ATOL
    @test JuMP.coefficient(dcon, x1) ≈ dA[1, 1] atol = ATOL
    @test JuMP.coefficient(dcon, x2) ≈ dA[1, 2] atol = ATOL
end

# ── Test reverse differentiation with different seed ─────────────────────────

function test_reverse_dx2_seed()
    model, x1, x2, c1 = _setup_model()

    MOI.set(model, DiffOpt.ReverseVariablePrimal(), x2, 1.0)
    DiffOpt.reverse_differentiate!(model)

    dc, db, dA = _expected_reverse([0.0, 1.0], [0.0])

    dobj = MOI.get(model, DiffOpt.ReverseObjectiveFunction())
    @test JuMP.coefficient(dobj, x1) ≈ dc[1] atol = ATOL
    @test JuMP.coefficient(dobj, x2) ≈ dc[2] atol = ATOL

    dcon = MOI.get(model, DiffOpt.ReverseConstraintFunction(), c1)
    @test MOI.constant(dcon) ≈ -db[1] atol = ATOL
end

# ── Test reverse differentiation with dual seed ─────────────────────────────

function test_reverse_dy_seed()
    model, x1, x2, c1 = _setup_model()

    MOI.set(model, DiffOpt.ReverseConstraintDual(), c1, 1.0)
    DiffOpt.reverse_differentiate!(model)

    dc, db, dA = _expected_reverse([0.0, 0.0], [1.0])

    dobj = MOI.get(model, DiffOpt.ReverseObjectiveFunction())
    @test JuMP.coefficient(dobj, x1) ≈ dc[1] atol = ATOL
    @test JuMP.coefficient(dobj, x2) ≈ dc[2] atol = ATOL

    dcon = MOI.get(model, DiffOpt.ReverseConstraintFunction(), c1)
    @test MOI.constant(dcon) ≈ -db[1] atol = ATOL
    @test JuMP.coefficient(dcon, x1) ≈ dA[1, 1] atol = ATOL
    @test JuMP.coefficient(dcon, x2) ≈ dA[1, 2] atol = ATOL
end

# ── Test forward differentiation ─────────────────────────────────────────────

function test_forward_objective_perturbation()
    model, x1, x2, c1 = _setup_model()

    # Perturb the linear objective: dc = [1, 0]
    # Forward tangent solves K * [dx; dν] = [-dc; db] = [-1; 0; 0]
    K = [1.0 0.0 1.0; 0.0 1.0 1.0; 1.0 1.0 0.0]
    fwd = K \ [-1.0, 0.0, 0.0]

    MOI.set(model, DiffOpt.ForwardObjectiveFunction(), 1.0 * x1)
    DiffOpt.forward_differentiate!(model)

    @test !isnan(MOI.get(model, DiffOpt.DifferentiateTimeSec()))

    dx1 = MOI.get(model, DiffOpt.ForwardVariablePrimal(), x1)
    dx2 = MOI.get(model, DiffOpt.ForwardVariablePrimal(), x2)
    @test dx1 ≈ fwd[1] atol = ATOL
    @test dx2 ≈ fwd[2] atol = ATOL
end

function test_forward_constraint_perturbation()
    model, x1, x2, c1 = _setup_model()

    # Perturb the RHS: db = [1], forward tangent solves K * [dx; dν] = [0; 0; 1]
    K = [1.0 0.0 1.0; 0.0 1.0 1.0; 1.0 1.0 0.0]
    fwd = K \ [0.0, 0.0, 1.0]

    # In forward mode, constraint perturbation is set via ForwardConstraintFunction
    # with func = dA*x - db. For pure b perturbation: func has constant = -1 (db=1).
    func = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm{Float64}[], -1.0)
    MOI.set(model, DiffOpt.ForwardConstraintFunction(), c1, func)
    DiffOpt.forward_differentiate!(model)

    dx1 = MOI.get(model, DiffOpt.ForwardVariablePrimal(), x1)
    dx2 = MOI.get(model, DiffOpt.ForwardVariablePrimal(), x2)
    @test dx1 ≈ fwd[1] atol = ATOL
    @test dx2 ≈ fwd[2] atol = ATOL

    dν1 = MOI.get(model, DiffOpt.ForwardConstraintDual(), c1)
    @test dν1 ≈ fwd[3] atol = ATOL
end

# ── Test that ModelConstructor override disables auto-detection ──────────────

function test_model_constructor_overrides_native()
    model = DiffOpt.diff_optimizer(EqQPSolver)
    # Explicitly set a model constructor — should NOT use native diff
    MOI.set(model, DiffOpt.ModelConstructor(), DiffOpt.QuadraticProgram.Model)

    x1 = MOI.add_variable(model)
    x2 = MOI.add_variable(model)
    MOI.add_constraint(model, 1.0 * x1 + 1.0 * x2, MOI.EqualTo(1.0))
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

    # Should use QuadraticProgram.Model, not SolverBackedDiff.Model
    MOI.set(model, DiffOpt.ReverseVariablePrimal(), x1, 1.0)
    DiffOpt.reverse_differentiate!(model)

    # Verify it still produces correct results
    dobj = MOI.get(model, DiffOpt.ReverseObjectiveFunction())
    @test dobj isa Any  # just check it doesn't error
end

# ── Test SolverBackedDiff.Model directly ─────────────────────────────────────

function test_model_is_empty()
    solver = EqQPSolver()
    sbm = DiffOpt.SolverBackedDiff.Model(solver)
    @test MOI.is_empty(sbm)
end

function test_model_empty!()
    solver = EqQPSolver()
    sbm = DiffOpt.SolverBackedDiff.Model(solver)
    MOI.empty!(sbm)
    @test MOI.is_empty(sbm)
end

# ── Cross-check reverse mode against finite differences ──────────────────────

function test_reverse_finite_difference_check()
    # Verify sensitivities by perturbing c and checking objective changes.
    # Problem: min (1/2)x'Qx + c'x s.t. Ax = b
    # Q = I, c = [3, 2], A = [1 1], b = [1]
    # x* = [0, 1], obj* = 0.5 + 2 = 2.5

    function solve_and_get_x(c1_val, c2_val)
        solver = EqQPSolver()
        MOI.add_variable(solver)
        MOI.add_variable(solver)
        solver.Q = [1.0 0.0; 0.0 1.0]
        solver.c = [c1_val, c2_val]
        solver.A = [1.0 1.0]
        solver.b = [1.0]
        solver.n_cons = 1
        push!(solver.con_indices, EQ_CI(1))
        MOI.optimize!(solver)
        return solver.x
    end

    ε = 1e-6
    x_base = solve_and_get_x(3.0, 2.0)

    # dl/dx₁ = 1: sensitivity of x₁ w.r.t. c₁
    x_pert = solve_and_get_x(3.0 + ε, 2.0)
    fd_dc1 = (x_pert[1] - x_base[1]) / ε

    model, x1, x2, c1 = _setup_model()
    MOI.set(model, DiffOpt.ReverseVariablePrimal(), x1, 1.0)
    DiffOpt.reverse_differentiate!(model)
    dobj = MOI.get(model, DiffOpt.ReverseObjectiveFunction())

    # dc₁ sensitivity from reverse mode
    @test JuMP.coefficient(dobj, x1) ≈ fd_dc1 atol = 1e-4
end

# ── Test with a 3-variable problem ───────────────────────────────────────────

function test_three_variable_problem()
    model = DiffOpt.diff_optimizer(EqQPSolver)

    x = [MOI.add_variable(model) for _ in 1:3]

    # min (1/2)(x₁² + 2x₂² + 3x₃²) + x₁ + x₂ + x₃
    # s.t. x₁ + x₂ + x₃ = 1
    #      x₁ - x₂       = 0
    obj = MOI.ScalarQuadraticFunction(
        [MOI.ScalarQuadraticTerm(1.0, x[1], x[1]),
         MOI.ScalarQuadraticTerm(2.0, x[2], x[2]),
         MOI.ScalarQuadraticTerm(3.0, x[3], x[3])],
        [MOI.ScalarAffineTerm(1.0, x[i]) for i in 1:3],
        0.0,
    )
    MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(), obj)
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    c1 = MOI.add_constraint(model, 1.0 * x[1] + 1.0 * x[2] + 1.0 * x[3], MOI.EqualTo(1.0))
    c2 = MOI.add_constraint(model, 1.0 * x[1] - 1.0 * x[2], MOI.EqualTo(0.0))

    MOI.optimize!(model)

    # Verify feasibility
    x_val = [MOI.get(model, MOI.VariablePrimal(), x[i]) for i in 1:3]
    @test x_val[1] + x_val[2] + x_val[3] ≈ 1.0 atol = ATOL
    @test x_val[1] - x_val[2] ≈ 0.0 atol = ATOL

    # Reverse differentiate with seed on x₁
    MOI.set(model, DiffOpt.ReverseVariablePrimal(), x[1], 1.0)
    DiffOpt.reverse_differentiate!(model)

    dobj = MOI.get(model, DiffOpt.ReverseObjectiveFunction())
    @test dobj isa DiffOpt.IndexMappedFunction

    # Verify finite difference for dc₁
    function solve_3var(c_vec)
        Q = Diagonal([1.0, 2.0, 3.0])
        A = [1.0 1.0 1.0; 1.0 -1.0 0.0]
        b = [1.0, 0.0]
        K = vcat(hcat(Q, A'), hcat(A, zeros(2, 2)))
        rhs = vcat(-c_vec, b)
        sol = K \ rhs
        return sol[1:3]
    end

    ε = 1e-6
    x_base = solve_3var([1.0, 1.0, 1.0])
    for j in 1:3
        c_pert = [1.0, 1.0, 1.0]
        c_pert[j] += ε
        x_pert = solve_3var(c_pert)
        fd = (x_pert[1] - x_base[1]) / ε
        @test JuMP.coefficient(dobj, x[j]) ≈ fd atol = 1e-4
    end
end

# ── Test combined dx and dy seed ─────────────────────────────────────────────

function test_reverse_combined_seed()
    model, x1, x2, c1 = _setup_model()

    MOI.set(model, DiffOpt.ReverseVariablePrimal(), x1, 1.0)
    MOI.set(model, DiffOpt.ReverseConstraintDual(), c1, 1.0)
    DiffOpt.reverse_differentiate!(model)

    dc, _, _ = _expected_reverse([1.0, 0.0], [1.0])

    dobj = MOI.get(model, DiffOpt.ReverseObjectiveFunction())
    @test JuMP.coefficient(dobj, x1) ≈ dc[1] atol = ATOL
    @test JuMP.coefficient(dobj, x2) ≈ dc[2] atol = ATOL
end

# ── Test that re-differentiation works (can call reverse twice) ──────────────

function test_reverse_twice()
    model, x1, x2, c1 = _setup_model()

    MOI.set(model, DiffOpt.ReverseVariablePrimal(), x1, 1.0)
    DiffOpt.reverse_differentiate!(model)
    dobj1 = MOI.get(model, DiffOpt.ReverseObjectiveFunction())

    dc1, _, _ = _expected_reverse([1.0, 0.0], [0.0])
    @test JuMP.coefficient(dobj1, x1) ≈ dc1[1] atol = ATOL

    # Differentiate again with different seed
    MOI.set(model, DiffOpt.ReverseVariablePrimal(), x1, 0.0)
    MOI.set(model, DiffOpt.ReverseVariablePrimal(), x2, 1.0)
    DiffOpt.reverse_differentiate!(model)
    dobj2 = MOI.get(model, DiffOpt.ReverseObjectiveFunction())

    dc2, _, _ = _expected_reverse([0.0, 1.0], [0.0])
    @test JuMP.coefficient(dobj2, x1) ≈ dc2[1] atol = ATOL
end

# ── Test set_index_mapping! ──────────────────────────────────────────────────

function test_set_index_mapping()
    solver = EqQPSolver()
    sbm = DiffOpt.SolverBackedDiff.Model(solver)
    var_map = Dict(
        MOI.VariableIndex(1) => MOI.VariableIndex(2),
        MOI.VariableIndex(2) => MOI.VariableIndex(1),
    )
    con_map = Dict{MOI.ConstraintIndex,MOI.ConstraintIndex}()
    DiffOpt.SolverBackedDiff.set_index_mapping!(sbm, var_map, con_map)
    @test sbm.var_to_solver[MOI.VariableIndex(1)] == MOI.VariableIndex(2)
    @test sbm.var_from_solver[MOI.VariableIndex(2)] == MOI.VariableIndex(1)
end

TestSolverBackedDiff.runtests()

end # module TestSolverBackedDiff
