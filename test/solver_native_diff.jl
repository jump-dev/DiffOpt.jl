# Copyright (c) 2020: Akshay Sharma and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module TestSolverNativeDiff

using Test
using LinearAlgebra
import DiffOpt
import MathOptInterface as MOI
import JuMP

const ATOL = 1e-8
const RTOL = 1e-8

# ─────────────────────────────────────────────────────────────────────────────
# EqQPSolver: a minimal equality-constrained QP solver with native
# differentiation support via DiffOpt.BackwardDifferentiate and
# DiffOpt.ForwardDifferentiate.
#
#   min  (1/2) x'Qx + c'x
#   s.t. Ax = b
# ─────────────────────────────────────────────────────────────────────────────

const EQ_CI =
    MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},MOI.EqualTo{Float64}}

mutable struct EqQPSolver <: MOI.AbstractOptimizer
    Q::Matrix{Float64}
    c::Vector{Float64}
    A::Matrix{Float64}
    b::Vector{Float64}
    x::Vector{Float64}
    nu::Vector{Float64}
    kkt_factor::Any
    n_vars::Int
    n_cons::Int
    var_indices::Vector{MOI.VariableIndex}
    con_indices::Vector{EQ_CI}
    status::MOI.TerminationStatusCode
    # Reverse input seeds
    rev_dx::Dict{MOI.VariableIndex,Float64}
    rev_dy::Dict{EQ_CI,Float64}
    rev_dobj::Float64
    # Reverse results
    dc::Vector{Float64}
    dQ::Matrix{Float64}
    dA::Matrix{Float64}
    db::Vector{Float64}
    # Forward input perturbations
    fwd_objective::Union{Nothing,MOI.AbstractScalarFunction}
    fwd_constraints::Dict{EQ_CI,MOI.ScalarAffineFunction{Float64}}
    # Forward results
    dx_fwd::Vector{Float64}
    dnu_fwd::Vector{Float64}
    fwd_obj_sensitivity::Float64
    diff_time::Float64

    function EqQPSolver()
        return new(
            zeros(0, 0),
            Float64[],
            zeros(0, 0),
            Float64[],
            Float64[],
            Float64[],
            nothing,
            0,
            0,
            MOI.VariableIndex[],
            EQ_CI[],
            MOI.OPTIMIZE_NOT_CALLED,
            Dict{MOI.VariableIndex,Float64}(),
            Dict{EQ_CI,Float64}(),
            0.0,
            Float64[],
            zeros(0, 0),
            zeros(0, 0),
            Float64[],
            nothing,
            Dict{EQ_CI,MOI.ScalarAffineFunction{Float64}}(),
            Float64[],
            Float64[],
            0.0,
            0.0,
        )
    end
end

# ── MOI boilerplate ──────────────────────────────────────────────────────────

MOI.is_empty(s::EqQPSolver) = s.n_vars == 0 && s.n_cons == 0

function MOI.empty!(s::EqQPSolver)
    s.Q = zeros(0, 0)
    s.c = Float64[]
    s.A = zeros(0, 0)
    s.b = Float64[]
    s.x = Float64[]
    s.nu = Float64[]
    s.kkt_factor = nothing
    s.n_vars = 0
    s.n_cons = 0
    empty!(s.var_indices)
    empty!(s.con_indices)
    s.status = MOI.OPTIMIZE_NOT_CALLED
    empty!(s.rev_dx)
    empty!(s.rev_dy)
    s.rev_dobj = 0.0
    s.dc = Float64[]
    s.dQ = zeros(0, 0)
    s.dA = zeros(0, 0)
    s.db = Float64[]
    s.fwd_objective = nothing
    empty!(s.fwd_constraints)
    s.dx_fwd = Float64[]
    s.dnu_fwd = Float64[]
    s.fwd_obj_sensitivity = 0.0
    s.diff_time = 0.0
    return
end

function MOI.add_variable(s::EqQPSolver)
    s.n_vars += 1
    vi = MOI.VariableIndex(s.n_vars)
    push!(s.var_indices, vi)
    n = s.n_vars
    Q_new = zeros(n, n)
    Q_new[1:(n-1), 1:(n-1)] .= s.Q
    s.Q = Q_new
    push!(s.c, 0.0)
    s.A = s.n_cons > 0 ? hcat(s.A, zeros(s.n_cons)) : zeros(0, n)
    return vi
end

function MOI.supports_constraint(
    ::EqQPSolver,
    ::Type{MOI.ScalarAffineFunction{Float64}},
    ::Type{MOI.EqualTo{Float64}},
)
    return true
end

function MOI.add_constraint(
    s::EqQPSolver,
    func::MOI.ScalarAffineFunction{Float64},
    set::MOI.EqualTo{Float64},
)
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

function MOI.supports(
    ::EqQPSolver,
    ::MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}},
)
    return true
end

function MOI.supports(
    ::EqQPSolver,
    ::MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}},
)
    return true
end

function MOI.set(
    s::EqQPSolver,
    ::MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}},
    func::MOI.ScalarQuadraticFunction{Float64},
)
    fill!(s.Q, 0.0)
    fill!(s.c, 0.0)
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

function MOI.set(
    s::EqQPSolver,
    ::MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}},
    func::MOI.ScalarAffineFunction{Float64},
)
    fill!(s.Q, 0.0)
    fill!(s.c, 0.0)
    for t in func.terms
        s.c[t.variable.value] += t.coefficient
    end
    return
end

MOI.supports(::EqQPSolver, ::MOI.ObjectiveSense) = true

function MOI.set(::EqQPSolver, ::MOI.ObjectiveSense, ::MOI.OptimizationSense)
    return
end

MOI.get(::EqQPSolver, ::MOI.ObjectiveSense) = MOI.MIN_SENSE

function MOI.get(s::EqQPSolver, ::MOI.TerminationStatus)
    return s.status
end

function MOI.get(s::EqQPSolver, ::MOI.PrimalStatus)
    return s.status == MOI.OPTIMAL ? MOI.FEASIBLE_POINT : MOI.NO_SOLUTION
end

function MOI.get(s::EqQPSolver, ::MOI.DualStatus)
    return s.status == MOI.OPTIMAL ? MOI.FEASIBLE_POINT : MOI.NO_SOLUTION
end

function MOI.get(s::EqQPSolver, ::MOI.ResultCount)
    return s.status == MOI.OPTIMAL ? 1 : 0
end

MOI.get(s::EqQPSolver, ::MOI.RawStatusString) = string(s.status)

function MOI.get(s::EqQPSolver, ::MOI.ObjectiveValue)
    return 0.5 * dot(s.x, s.Q * s.x) + dot(s.c, s.x)
end

MOI.get(::EqQPSolver, ::MOI.SolverName) = "EqQPSolver"

function MOI.get(s::EqQPSolver, ::MOI.VariablePrimal, vi::MOI.VariableIndex)
    return s.x[vi.value]
end

function MOI.get(s::EqQPSolver, ::MOI.ConstraintDual, ci::EQ_CI)
    return s.nu[ci.value]
end

function MOI.get(s::EqQPSolver, ::MOI.ConstraintPrimal, ci::EQ_CI)
    return dot(s.A[ci.value, :], s.x)
end

MOI.get(s::EqQPSolver, ::MOI.ListOfVariableIndices) = copy(s.var_indices)

function MOI.get(
    s::EqQPSolver,
    ::MOI.ListOfConstraintIndices{
        MOI.ScalarAffineFunction{Float64},
        MOI.EqualTo{Float64},
    },
)
    return copy(s.con_indices)
end

MOI.get(s::EqQPSolver, ::MOI.NumberOfVariables) = s.n_vars

function MOI.is_valid(s::EqQPSolver, vi::MOI.VariableIndex)
    return 1 <= vi.value <= s.n_vars
end

function MOI.is_valid(s::EqQPSolver, ci::EQ_CI)
    return 1 <= ci.value <= s.n_cons
end

function MOI.get(s::EqQPSolver, ::MOI.ListOfConstraintTypesPresent)
    if s.n_cons > 0
        return [(MOI.ScalarAffineFunction{Float64}, MOI.EqualTo{Float64})]
    end
    return Tuple{Type,Type}[]
end

function MOI.get(s::EqQPSolver, ::MOI.ConstraintFunction, ci::EQ_CI)
    row = ci.value
    terms = [
        MOI.ScalarAffineTerm(s.A[row, j], MOI.VariableIndex(j)) for
        j in 1:s.n_vars if !iszero(s.A[row, j])
    ]
    return MOI.ScalarAffineFunction(terms, 0.0)
end

function MOI.get(s::EqQPSolver, ::MOI.ConstraintSet, ci::EQ_CI)
    return MOI.EqualTo(s.b[ci.value])
end

function MOI.get(::EqQPSolver, ::MOI.ObjectiveFunctionType)
    return MOI.ScalarAffineFunction{Float64}
end

function MOI.get(
    s::EqQPSolver,
    ::MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}},
)
    terms = [
        MOI.ScalarAffineTerm(s.c[j], MOI.VariableIndex(j)) for
        j in 1:s.n_vars if !iszero(s.c[j])
    ]
    return MOI.ScalarAffineFunction(terms, 0.0)
end

function MOI.copy_to(dest::EqQPSolver, src::MOI.ModelLike)
    MOI.empty!(dest)
    index_map = MOI.Utilities.IndexMap()
    for vi in MOI.get(src, MOI.ListOfVariableIndices())
        index_map[vi] = MOI.add_variable(dest)
    end
    obj_type = MOI.get(src, MOI.ObjectiveFunctionType())
    obj_func = MOI.Utilities.map_indices(
        index_map,
        MOI.get(src, MOI.ObjectiveFunction{obj_type}()),
    )
    if obj_func isa MOI.ScalarQuadraticFunction{Float64}
        MOI.set(
            dest,
            MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(),
            obj_func,
        )
    elseif obj_func isa MOI.ScalarAffineFunction{Float64}
        MOI.set(
            dest,
            MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
            obj_func,
        )
    end
    MOI.set(dest, MOI.ObjectiveSense(), MOI.get(src, MOI.ObjectiveSense()))
    for (F, S) in MOI.get(src, MOI.ListOfConstraintTypesPresent())
        for ci in MOI.get(src, MOI.ListOfConstraintIndices{F,S}())
            func = MOI.Utilities.map_indices(
                index_map,
                MOI.get(src, MOI.ConstraintFunction(), ci),
            )
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
    s.nu = sol[(n+1):end]
    s.status = MOI.OPTIMAL
    return
end

# ── Native differentiation via DiffOpt attributes ────────────────────────────

MOI.supports(::EqQPSolver, ::DiffOpt.BackwardDifferentiate) = true
MOI.supports(::EqQPSolver, ::DiffOpt.ForwardDifferentiate) = true

# Reverse input: seeds
function MOI.supports(
    ::EqQPSolver,
    ::DiffOpt.ReverseVariablePrimal,
    ::Type{MOI.VariableIndex},
)
    return true
end

function MOI.set(
    s::EqQPSolver,
    ::DiffOpt.ReverseVariablePrimal,
    vi::MOI.VariableIndex,
    value,
)
    s.rev_dx[vi] = value
    return
end

function MOI.set(
    s::EqQPSolver,
    ::DiffOpt.ReverseConstraintDual,
    ci::EQ_CI,
    value,
)
    s.rev_dy[ci] = value
    return
end

function MOI.set(s::EqQPSolver, ::DiffOpt.ReverseObjectiveSensitivity, value)
    s.rev_dobj = value
    return
end

# Trigger backward differentiation
function MOI.set(s::EqQPSolver, ::DiffOpt.BackwardDifferentiate, ::Nothing)
    s.diff_time = @elapsed _backward_differentiate!(s)
    return
end

function _backward_differentiate!(s::EqQPSolver)
    n, m = s.n_vars, s.n_cons
    rhs = zeros(n + m)
    for (vi, val) in s.rev_dx
        rhs[vi.value] = val
    end
    for (ci, val) in s.rev_dy
        rhs[n+ci.value] = val
    end
    # If dobj seed is nonzero, add the gradient of the objective w.r.t. x
    # dobj * (Qx + c) contributes to the rhs for the primal variables
    if !iszero(s.rev_dobj)
        rhs[1:n] .+= s.rev_dobj .* (s.Q * s.x .+ s.c)
    end
    adj = s.kkt_factor \ rhs
    dx_adj = adj[1:n]
    dnu_adj = adj[(n+1):end]
    s.dc = -dx_adj
    s.db = dnu_adj
    s.dA = -(s.nu * dx_adj' + dnu_adj * s.x')
    s.dQ = -0.5 * (dx_adj * s.x' + s.x * dx_adj')
    # Clear seeds
    empty!(s.rev_dx)
    empty!(s.rev_dy)
    s.rev_dobj = 0.0
    return
end

# Reverse output: results
function MOI.get(s::EqQPSolver, ::DiffOpt.ReverseObjectiveFunction)
    terms = [
        MOI.ScalarAffineTerm(s.dc[j], MOI.VariableIndex(j)) for j in 1:s.n_vars
    ]
    return MOI.ScalarAffineFunction(terms, 0.0)
end

function MOI.get(s::EqQPSolver, ::DiffOpt.ReverseConstraintFunction, ci::EQ_CI)
    row = ci.value
    terms = [
        MOI.ScalarAffineTerm(s.dA[row, j], MOI.VariableIndex(j)) for
        j in 1:s.n_vars
    ]
    return MOI.ScalarAffineFunction(terms, -s.db[row])
end

# Forward input: perturbations
function MOI.set(
    s::EqQPSolver,
    ::DiffOpt.ForwardObjectiveFunction,
    func::MOI.AbstractScalarFunction,
)
    s.fwd_objective = func
    return
end

function MOI.set(
    s::EqQPSolver,
    ::DiffOpt.ForwardConstraintFunction,
    ci::EQ_CI,
    func::MOI.ScalarAffineFunction{Float64},
)
    s.fwd_constraints[ci] = func
    return
end

# Trigger forward differentiation
function MOI.set(s::EqQPSolver, ::DiffOpt.ForwardDifferentiate, ::Nothing)
    s.diff_time = @elapsed _forward_differentiate!(s)
    return
end

function _forward_differentiate!(s::EqQPSolver)
    n, m = s.n_vars, s.n_cons
    d_c = zeros(n)
    d_b = zeros(m)
    if s.fwd_objective !== nothing
        for t in MOI.Utilities.canonical(s.fwd_objective).terms
            d_c[t.variable.value] += t.coefficient
        end
    end
    for (ci, func) in s.fwd_constraints
        d_b[ci.value] = -func.constant
    end
    rhs = vcat(-d_c, d_b)
    sol = s.kkt_factor \ rhs
    s.dx_fwd = sol[1:n]
    s.dnu_fwd = sol[(n+1):end]
    # Compute forward objective sensitivity: d(obj)/dt = (Qx + c)'dx + x'Q*dx/2 + d_c'x
    # Actually: obj = 1/2 x'Qx + c'x, so dobj/dt = (Qx+c)'dx_fwd + d_c'x
    s.fwd_obj_sensitivity = dot(s.Q * s.x .+ s.c, s.dx_fwd) + dot(d_c, s.x)
    # Clear inputs
    s.fwd_objective = nothing
    empty!(s.fwd_constraints)
    return
end

# Forward output: results
function MOI.get(
    s::EqQPSolver,
    ::DiffOpt.ForwardVariablePrimal,
    vi::MOI.VariableIndex,
)
    return s.dx_fwd[vi.value]
end

function MOI.get(s::EqQPSolver, ::DiffOpt.ForwardConstraintDual, ci::EQ_CI)
    return s.dnu_fwd[ci.value]
end

# ForwardObjectiveSensitivity
function MOI.get(s::EqQPSolver, ::DiffOpt.ForwardObjectiveSensitivity)
    return s.fwd_obj_sensitivity
end

# DifferentiateTimeSec
MOI.get(s::EqQPSolver, ::DiffOpt.DifferentiateTimeSec) = s.diff_time

# empty_input_sensitivities!
function DiffOpt.empty_input_sensitivities!(s::EqQPSolver)
    empty!(s.rev_dx)
    empty!(s.rev_dy)
    s.rev_dobj = 0.0
    s.fwd_objective = nothing
    empty!(s.fwd_constraints)
    return
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

# ── Helper: compute expected reverse sensitivities analytically ───────────────
# For the standard test problem:
#   Q = I, c = [3, 2], A = [1 1], b = [1]
#   x* = [0, 1], nu* = [-3]
#   K = [Q A'; A 0] = [1 0 1; 0 1 1; 1 1 0]

function _expected_reverse(dx_seed::Vector{Float64}, dy_seed::Vector{Float64})
    Q = [1.0 0.0; 0.0 1.0]
    A = [1.0 1.0]
    x_star = [0.0, 1.0]
    nu_star = [-3.0]
    n, m = 2, 1
    K = vcat(hcat(Q, A'), hcat(A, zeros(m, m)))
    rhs = vcat(dx_seed, dy_seed)
    adj = K \ rhs
    adj_x = adj[1:n]
    adj_nu = adj[(n+1):end]
    dc = -adj_x
    db = adj_nu
    dA = -(nu_star * adj_x' + adj_nu * x_star')
    return dc, db, dA
end

# ── Helper: set up and solve the standard test problem ────────────────────────
#
#   min  (1/2)(x1^2 + x2^2) + 3x1 + 2x2
#   s.t. x1 + x2 = 1
#
# KKT solution: x = [0, 1], nu = -3

function _setup_model()
    model = DiffOpt.diff_optimizer(EqQPSolver)
    x1 = MOI.add_variable(model)
    x2 = MOI.add_variable(model)
    c1 = MOI.add_constraint(model, 1.0 * x1 + 1.0 * x2, MOI.EqualTo(1.0))
    obj = MOI.ScalarQuadraticFunction(
        [
            MOI.ScalarQuadraticTerm(1.0, x1, x1),
            MOI.ScalarQuadraticTerm(1.0, x2, x2),
        ],
        [MOI.ScalarAffineTerm(3.0, x1), MOI.ScalarAffineTerm(2.0, x2)],
        0.0,
    )
    MOI.set(
        model,
        MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(),
        obj,
    )
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.optimize!(model)
    return model, x1, x2, c1
end

# ── Test that the solver supports native differentiation ─────────────────────

function test_supports_native_differentiation()
    @test MOI.supports(EqQPSolver(), DiffOpt.BackwardDifferentiate()) == true
    @test MOI.supports(EqQPSolver(), DiffOpt.ForwardDifferentiate()) == true
end

# ── Test that native solver is auto-detected ─────────────────────────────────

function test_auto_detection()
    model, x1, x2, c1 = _setup_model()
    # Verify the solution
    @test MOI.get(model, MOI.VariablePrimal(), x1) ≈ 0.0 atol = ATOL
    @test MOI.get(model, MOI.VariablePrimal(), x2) ≈ 1.0 atol = ATOL
    # Trigger differentiation — should use native solver path
    MOI.set(model, DiffOpt.ReverseVariablePrimal(), x1, 1.0)
    DiffOpt.reverse_differentiate!(model)
    # If we get here without error, auto-detection worked
    dobj = MOI.get(model, DiffOpt.ReverseObjectiveFunction())
    @test dobj isa DiffOpt.IndexMappedFunction
end

# ── Test reverse differentiation with dx seed ────────────────────────────────

function test_reverse_dx_seed()
    model, x1, x2, c1 = _setup_model()
    MOI.set(model, DiffOpt.ReverseVariablePrimal(), x1, 1.0)
    DiffOpt.reverse_differentiate!(model)

    @test MOI.get(model, DiffOpt.DifferentiateTimeSec()) >= 0.0

    dc, db, dA = _expected_reverse([1.0, 0.0], [0.0])

    dobj = MOI.get(model, DiffOpt.ReverseObjectiveFunction())
    @test JuMP.coefficient(dobj, x1) ≈ dc[1] atol = ATOL
    @test JuMP.coefficient(dobj, x2) ≈ dc[2] atol = ATOL

    dcon = MOI.get(model, DiffOpt.ReverseConstraintFunction(), c1)
    @test MOI.constant(dcon) ≈ -db[1] atol = ATOL
    @test JuMP.coefficient(dcon, x1) ≈ dA[1, 1] atol = ATOL
    @test JuMP.coefficient(dcon, x2) ≈ dA[1, 2] atol = ATOL
end

# ── Test reverse differentiation with x2 seed ───────────────────────────────

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

# ── Test reverse differentiation with dual seed ──────────────────────────────

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

# ── Test forward differentiation with objective perturbation ─────────────────

function test_forward_objective_perturbation()
    model, x1, x2, c1 = _setup_model()

    # Perturb the linear objective: dc = [1, 0]
    # Forward tangent solves K * [dx; dnu] = [-dc; db] = [-1; 0; 0]
    K = [1.0 0.0 1.0; 0.0 1.0 1.0; 1.0 1.0 0.0]
    fwd = K \ [-1.0, 0.0, 0.0]

    MOI.set(model, DiffOpt.ForwardObjectiveFunction(), 1.0 * x1)
    DiffOpt.forward_differentiate!(model)

    @test MOI.get(model, DiffOpt.DifferentiateTimeSec()) >= 0.0

    dx1 = MOI.get(model, DiffOpt.ForwardVariablePrimal(), x1)
    dx2 = MOI.get(model, DiffOpt.ForwardVariablePrimal(), x2)
    @test dx1 ≈ fwd[1] atol = ATOL
    @test dx2 ≈ fwd[2] atol = ATOL
end

# ── Test forward differentiation with constraint perturbation ────────────────

function test_forward_constraint_perturbation()
    model, x1, x2, c1 = _setup_model()

    # Perturb the RHS: db = [1], forward tangent solves K * [dx; dnu] = [0; 0; 1]
    K = [1.0 0.0 1.0; 0.0 1.0 1.0; 1.0 1.0 0.0]
    fwd = K \ [0.0, 0.0, 1.0]

    # Constraint perturbation: func constant = -1 means db = 1
    func = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm{Float64}[], -1.0)
    MOI.set(model, DiffOpt.ForwardConstraintFunction(), c1, func)
    DiffOpt.forward_differentiate!(model)

    dx1 = MOI.get(model, DiffOpt.ForwardVariablePrimal(), x1)
    dx2 = MOI.get(model, DiffOpt.ForwardVariablePrimal(), x2)
    @test dx1 ≈ fwd[1] atol = ATOL
    @test dx2 ≈ fwd[2] atol = ATOL

    dnu1 = MOI.get(model, DiffOpt.ForwardConstraintDual(), c1)
    @test dnu1 ≈ fwd[3] atol = ATOL
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

# ── Test re-differentiation (call reverse twice with different seeds) ────────

function test_reverse_twice()
    model, x1, x2, c1 = _setup_model()

    MOI.set(model, DiffOpt.ReverseVariablePrimal(), x1, 1.0)
    DiffOpt.reverse_differentiate!(model)
    dobj1 = MOI.get(model, DiffOpt.ReverseObjectiveFunction())

    dc1, _, _ = _expected_reverse([1.0, 0.0], [0.0])
    @test JuMP.coefficient(dobj1, x1) ≈ dc1[1] atol = ATOL

    # Differentiate again with different seed
    DiffOpt.empty_input_sensitivities!(model)
    MOI.set(model, DiffOpt.ReverseVariablePrimal(), x2, 1.0)
    DiffOpt.reverse_differentiate!(model)
    dobj2 = MOI.get(model, DiffOpt.ReverseObjectiveFunction())

    dc2, _, _ = _expected_reverse([0.0, 1.0], [0.0])
    @test JuMP.coefficient(dobj2, x1) ≈ dc2[1] atol = ATOL
end

# ── Cross-check reverse mode against finite differences ──────────────────────

function test_reverse_finite_difference_check()
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

    eps = 1e-6
    x_base = solve_and_get_x(3.0, 2.0)

    # dl/dx1 = 1: sensitivity of x1 w.r.t. c1
    x_pert = solve_and_get_x(3.0 + eps, 2.0)
    fd_dc1 = (x_pert[1] - x_base[1]) / eps

    model, x1, x2, c1 = _setup_model()
    MOI.set(model, DiffOpt.ReverseVariablePrimal(), x1, 1.0)
    DiffOpt.reverse_differentiate!(model)
    dobj = MOI.get(model, DiffOpt.ReverseObjectiveFunction())

    @test JuMP.coefficient(dobj, x1) ≈ fd_dc1 atol = 1e-4
end

# ── Test with a 3-variable problem ───────────────────────────────────────────

function test_three_variable_problem()
    model = DiffOpt.diff_optimizer(EqQPSolver)

    x = [MOI.add_variable(model) for _ in 1:3]

    # min (1/2)(x1^2 + 2*x2^2 + 3*x3^2) + x1 + x2 + x3
    # s.t. x1 + x2 + x3 = 1
    #      x1 - x2       = 0
    obj = MOI.ScalarQuadraticFunction(
        [
            MOI.ScalarQuadraticTerm(1.0, x[1], x[1]),
            MOI.ScalarQuadraticTerm(2.0, x[2], x[2]),
            MOI.ScalarQuadraticTerm(3.0, x[3], x[3]),
        ],
        [MOI.ScalarAffineTerm(1.0, x[i]) for i in 1:3],
        0.0,
    )
    MOI.set(
        model,
        MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(),
        obj,
    )
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    c1 = MOI.add_constraint(
        model,
        1.0 * x[1] + 1.0 * x[2] + 1.0 * x[3],
        MOI.EqualTo(1.0),
    )
    c2 = MOI.add_constraint(model, 1.0 * x[1] - 1.0 * x[2], MOI.EqualTo(0.0))

    MOI.optimize!(model)

    # Verify feasibility
    x_val = [MOI.get(model, MOI.VariablePrimal(), x[i]) for i in 1:3]
    @test x_val[1] + x_val[2] + x_val[3] ≈ 1.0 atol = ATOL
    @test x_val[1] - x_val[2] ≈ 0.0 atol = ATOL

    # Reverse differentiate with seed on x1
    MOI.set(model, DiffOpt.ReverseVariablePrimal(), x[1], 1.0)
    DiffOpt.reverse_differentiate!(model)

    dobj = MOI.get(model, DiffOpt.ReverseObjectiveFunction())
    @test dobj isa DiffOpt.IndexMappedFunction

    # Verify finite difference for dc_j
    function solve_3var(c_vec)
        Q = Diagonal([1.0, 2.0, 3.0])
        A = [1.0 1.0 1.0; 1.0 -1.0 0.0]
        b = [1.0, 0.0]
        K = vcat(hcat(Q, A'), hcat(A, zeros(2, 2)))
        rhs = vcat(-c_vec, b)
        sol = K \ rhs
        return sol[1:3]
    end

    eps = 1e-6
    x_base = solve_3var([1.0, 1.0, 1.0])
    for j in 1:3
        c_pert = [1.0, 1.0, 1.0]
        c_pert[j] += eps
        x_pert = solve_3var(c_pert)
        fd = (x_pert[1] - x_base[1]) / eps
        @test JuMP.coefficient(dobj, x[j]) ≈ fd atol = 1e-4
    end
end

# ── Test reverse with dobj seed (ReverseObjectiveSensitivity) ─────────────

function test_reverse_dobj_seed()
    model, x1, x2, c1 = _setup_model()

    # Set dobj = 1.0: sensitivity of the objective value w.r.t. problem data
    MOI.set(model, DiffOpt.ReverseObjectiveSensitivity(), 1.0)
    DiffOpt.reverse_differentiate!(model)

    # x* = [0, 1], nu* = [-3], Q = I, c = [3, 2]
    # grad_obj w.r.t. x = Q*x + c = [3, 3]
    # KKT solve with rhs = [3, 3, 0] gives the adjoint
    K = [1.0 0.0 1.0; 0.0 1.0 1.0; 1.0 1.0 0.0]
    rhs = [3.0, 3.0, 0.0]
    adj = K \ rhs
    dc_expected = -adj[1:2]

    dobj = MOI.get(model, DiffOpt.ReverseObjectiveFunction())
    @test JuMP.coefficient(dobj, x1) ≈ dc_expected[1] atol = ATOL
    @test JuMP.coefficient(dobj, x2) ≈ dc_expected[2] atol = ATOL
end

# ── Test forward objective sensitivity ────────────────────────────────────

function test_forward_objective_sensitivity()
    model, x1, x2, c1 = _setup_model()

    # Perturb dc = [1, 0]
    MOI.set(model, DiffOpt.ForwardObjectiveFunction(), 1.0 * x1)
    DiffOpt.forward_differentiate!(model)

    # x* = [0, 1], Q = I, c = [3, 2]
    # K * [dx; dnu] = [-dc; 0] = [-1; 0; 0]
    K = [1.0 0.0 1.0; 0.0 1.0 1.0; 1.0 1.0 0.0]
    fwd = K \ [-1.0, 0.0, 0.0]
    # dobj/dt = (Qx + c)'dx + dc'x = [3,3]'*dx + [1,0]'*[0,1]
    expected = dot([3.0, 3.0], fwd[1:2]) + dot([1.0, 0.0], [0.0, 1.0])

    @test MOI.get(model, DiffOpt.ForwardObjectiveSensitivity()) ≈ expected atol =
        ATOL
end

# ── Test forward twice (re-differentiation in forward mode) ──────────────

function test_forward_twice()
    model, x1, x2, c1 = _setup_model()

    K = [1.0 0.0 1.0; 0.0 1.0 1.0; 1.0 1.0 0.0]

    # First: perturb dc = [1, 0]
    MOI.set(model, DiffOpt.ForwardObjectiveFunction(), 1.0 * x1)
    DiffOpt.forward_differentiate!(model)

    fwd1 = K \ [-1.0, 0.0, 0.0]
    @test MOI.get(model, DiffOpt.ForwardVariablePrimal(), x1) ≈ fwd1[1] atol =
        ATOL

    # Second: perturb dc = [0, 1]
    DiffOpt.empty_input_sensitivities!(model)
    MOI.set(model, DiffOpt.ForwardObjectiveFunction(), 1.0 * x2)
    DiffOpt.forward_differentiate!(model)

    fwd2 = K \ [0.0, -1.0, 0.0]
    @test MOI.get(model, DiffOpt.ForwardVariablePrimal(), x1) ≈ fwd2[1] atol =
        ATOL
    @test MOI.get(model, DiffOpt.ForwardVariablePrimal(), x2) ≈ fwd2[2] atol =
        ATOL
end

# ── Test ReverseConstraintSet (not applicable without parameters, but test the getter path) ──

function test_reverse_constraint_set()
    model, x1, x2, c1 = _setup_model()

    MOI.set(model, DiffOpt.ReverseVariablePrimal(), x1, 1.0)
    DiffOpt.reverse_differentiate!(model)

    dc, db, dA = _expected_reverse([1.0, 0.0], [0.0])

    # ReverseConstraintFunction returns dA and db
    dcon = MOI.get(model, DiffOpt.ReverseConstraintFunction(), c1)
    @test MOI.constant(dcon) ≈ -db[1] atol = ATOL

    # standard_form should convert the lazy function
    sf = DiffOpt.standard_form(dcon)
    @test sf isa MOI.ScalarAffineFunction
end

# ── Test empty_input_sensitivities! ──────────────────────────────────────────

function test_empty_input_sensitivities()
    model, x1, x2, c1 = _setup_model()
    MOI.set(model, DiffOpt.ReverseVariablePrimal(), x1, 1.0)
    DiffOpt.empty_input_sensitivities!(model)
    # After clearing, differentiation with zero seeds should give zero results
    DiffOpt.reverse_differentiate!(model)
    dobj = MOI.get(model, DiffOpt.ReverseObjectiveFunction())
    sf = DiffOpt.standard_form(dobj)
    for term in sf.terms
        @test term.coefficient ≈ 0.0 atol = ATOL
    end
end

TestSolverNativeDiff.runtests()

end # module TestSolverNativeDiff
