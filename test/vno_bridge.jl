module TestVNOBridge

using DiffOpt
using JuMP
using Ipopt
using Test
using FiniteDiff
import DelimitedFiles
using SparseArrays
using LinearAlgebra
import MathOptInterface as MOI

function runtests()
    for name in names(@__MODULE__; all = true)
        if startswith("$name", "test_")
            @testset "$(name)" begin
                getfield(@__MODULE__, name)()
            end
        end
    end
    return
end # module

TestVNOBridge.runtests()

mutable struct _BridgeMockModel <: MOI.ModelLike
    deleted::Vector{MOI.ConstraintIndex}
    primal_start::Dict{MOI.ConstraintIndex,Float64}
    dual_start::Dict{MOI.ConstraintIndex,Float64}
    variable_start::Dict{MOI.VariableIndex,Float64}
end

function _BridgeMockModel()
    return _BridgeMockModel(
        MOI.ConstraintIndex[],
        Dict{MOI.ConstraintIndex,Float64}(),
        Dict{MOI.ConstraintIndex,Float64}(),
        Dict{MOI.VariableIndex,Float64}(),
    )
end

function MOI.delete(model::_BridgeMockModel, ci::MOI.ConstraintIndex)
    push!(model.deleted, ci)
    return
end

function MOI.set(
    model::_BridgeMockModel,
    ::MOI.ConstraintPrimalStart,
    ci::MOI.ConstraintIndex,
    value,
)
    model.primal_start[ci] = value
    return
end

function MOI.set(
    model::_BridgeMockModel,
    ::MOI.ConstraintDualStart,
    ci::MOI.ConstraintIndex,
    value,
)
    model.dual_start[ci] = value
    return
end

function MOI.get(
    model::_BridgeMockModel,
    ::MOI.VariablePrimalStart,
    vi::MOI.VariableIndex,
)
    return model.variable_start[vi]
end

function test_VectorNonlinearOracle_default_constructor_selection()
    # Ensure default `diff_optimizer` routes VNO constraints to NonLinearProgram.
    model = Model(() -> DiffOpt.diff_optimizer(Ipopt.Optimizer))
    set_silent(model)

    p_val = 0.3
    @variable(model, x)
    @variable(model, p in Parameter(p_val))
    @objective(model, Min, (x - p)^2)

    function eval_f(ret::AbstractVector, z::AbstractVector)
        ret[1] = z[1]^2
        return
    end
    jacobian_structure = [(1, 1)]
    function eval_jacobian(ret::AbstractVector, z::AbstractVector)
        ret[1] = 2.0 * z[1]
        return
    end
    hessian_lagrangian_structure = [(1, 1)]
    function eval_hessian_lagrangian(
        ret::AbstractVector,
        z::AbstractVector,
        μ::AbstractVector,
    )
        ret[1] = 2.0 * μ[1]
        return
    end

    set = MOI.VectorNonlinearOracle(;
        dimension = 1,
        l = [-Inf],
        u = [1.0],
        eval_f,
        jacobian_structure,
        eval_jacobian,
        hessian_lagrangian_structure,
        eval_hessian_lagrangian,
    )
    @constraint(model, [x] in set)

    optimize!(model)
    @test is_solved_and_feasible(model)
    @test value(x) ≈ p_val atol = 1e-7

    diff_backend = DiffOpt._diff(model.moi_backend.optimizer.model)
    if diff_backend isa MOI.Bridges.LazyBridgeOptimizer
        @test diff_backend.model isa DiffOpt.NonLinearProgram.Model
    else
        @test diff_backend isa DiffOpt.NonLinearProgram.Model
    end

    DiffOpt.empty_input_sensitivities!(model)
    DiffOpt.set_reverse_variable(model, x, 1.0)
    DiffOpt.reverse_differentiate!(model)
    @test DiffOpt.get_reverse_parameter(model, p) ≈ 1.0 atol = 1e-6
end

function test_VectorNonlinearOracle_univariate()
    # Univariate test: 1 input, 1 output
    # Constraint: x^2 <= 1
    model = DiffOpt.nonlinear_diff_model(Ipopt.Optimizer)
    set_silent(model)

    p_val = 0.3
    @variable(model, x)
    @variable(model, p in Parameter(p_val))

    # Minimize (x - p)^2; solution should be x = p as long as constraint is inactive
    @objective(model, Min, (x - p)^2)

    # Oracle for constraint: x^2 <= 1  (written as l <= f(x) <= u with l=-Inf, u=1)
    function eval_f(ret::AbstractVector, z::AbstractVector)
        ret[1] = z[1]^2
        return
    end
    jacobian_structure = [(1, 1)]
    function eval_jacobian(ret::AbstractVector, z::AbstractVector)
        ret[1] = 2.0 * z[1]
        return
    end
    hessian_lagrangian_structure = [(1, 1)]
    function eval_hessian_lagrangian(
        ret::AbstractVector,
        z::AbstractVector,
        μ::AbstractVector,
    )
        # Hessian of μ1 * z1^2 is 2*μ1
        ret[1] = 2.0 * μ[1]
        return
    end

    set = MOI.VectorNonlinearOracle(;
        dimension = 1,
        l = [-Inf],
        u = [1.0],
        eval_f,
        jacobian_structure,
        eval_jacobian,
        hessian_lagrangian_structure,
        eval_hessian_lagrangian,
    )

    @constraint(model, c, [x] in set)

    optimize!(model)
    @test is_solved_and_feasible(model)

    # pick p strictly inside (-1, 1) so constraint is inactive and x == p
    @test value(x) ≈ p_val atol = 1e-7

    # Reverse-mode: dx/dp = 1, so with seed 1.0 on x, sensitivity on p should be 1.0
    DiffOpt.empty_input_sensitivities!(model)
    DiffOpt.set_reverse_variable(model, x, 1.0)
    DiffOpt.reverse_differentiate!(model)
    @test DiffOpt.get_reverse_parameter(model, p) ≈ 1.0 atol = 1e-6
end

function test_VectorNonlinearOracle_multivariate()
    # Multivariate test: 2 inputs, 2 outputs
    # Constraint: [x1^2 + x2^2; x1 * x2] in [-Inf, 1] x [0, Inf] 
    # (i.e., x1^2+x2^2 <= 1, x1*x2 >= 0)
    model = DiffOpt.nonlinear_diff_model(Ipopt.Optimizer)
    set_silent(model)

    p_val = [0.2, 0.3]
    @variable(model, x[1:2])
    @variable(model, p[1:2] in Parameter.(p_val))

    # Minimize sum((x - p)^2); solution should be x = p if constraint is inactive
    @objective(model, Min, sum((x .- p) .^ 2))

    # f(x) = [x1^2 + x2^2; x1 * x2]
    function eval_f(ret::AbstractVector, z::AbstractVector)
        ret[1] = z[1]^2 + z[2]^2
        ret[2] = z[1] * z[2]
        return
    end

    # Jacobian structure: df1/dx1, df1/dx2, df2/dx1, df2/dx2
    # df1/dx1 = 2*x1, df1/dx2 = 2*x2
    # df2/dx1 = x2, df2/dx2 = x1
    jacobian_structure = [(1, 1), (1, 2), (2, 1), (2, 2)]

    function eval_jacobian(ret::AbstractVector, z::AbstractVector)
        ret[1] = 2.0 * z[1]  # df1/dx1
        ret[2] = 2.0 * z[2]  # df1/dx2
        ret[3] = z[2]        # df2/dx1
        ret[4] = z[1]        # df2/dx2
        return
    end

    # Hessian of Lagrangian = μ1 * H_f1 + μ2 * H_f2
    # H_f1 = [2 0; 0 2], H_f2 = [0 1; 1 0]
    # Hessian structure (lower triangular): (1,1), (2,1), (2,2)
    hessian_lagrangian_structure = [(1, 1), (2, 1), (2, 2)]

    function eval_hessian_lagrangian(
        ret::AbstractVector,
        z::AbstractVector,
        μ::AbstractVector,
    )
        # (1,1): μ1 * 2 + μ2 * 0 = 2*μ1
        ret[1] = 2.0 * μ[1]
        # (2,1): μ1 * 0 + μ2 * 1 = μ2
        ret[2] = μ[2]
        # (2,2): μ1 * 2 + μ2 * 0 = 2*μ1
        ret[3] = 2.0 * μ[1]
        return
    end

    set = MOI.VectorNonlinearOracle(;
        dimension = 2,
        l = [-Inf, 0.0],
        u = [1.0, Inf],
        eval_f,
        jacobian_structure,
        eval_jacobian,
        hessian_lagrangian_structure,
        eval_hessian_lagrangian,
    )

    @constraint(model, c, [x[1], x[2]] in set)

    optimize!(model)
    @test is_solved_and_feasible(model)

    # Constraints should be inactive at x = p = [0.2, 0.3]
    # (0.2^2 + 0.3^2 = 0.13 <= 1, 0.2*0.3 = 0.06 >= 0)
    @test value(x[1]) ≈ p_val[1] atol = 1e-6
    @test value(x[2]) ≈ p_val[2] atol = 1e-6

    # Reverse-mode: dx/dp = I (identity), so with seed [1.0, 1.0] on x, 
    # sensitivity on p should be [1.0, 1.0]
    DiffOpt.empty_input_sensitivities!(model)
    DiffOpt.set_reverse_variable(model, x[1], 1.0)
    DiffOpt.set_reverse_variable(model, x[2], 1.0)
    DiffOpt.reverse_differentiate!(model)
    dp1 = DiffOpt.get_reverse_parameter(model, p[1])
    dp2 = DiffOpt.get_reverse_parameter(model, p[2])
    @test dp1 ≈ 1.0 atol = 1e-5
    @test dp2 ≈ 1.0 atol = 1e-5
end

function test_VectorNonlinearOracle_active_constraint()
    # Test with an active constraint where sensitivities are affected
    # Multivariate: 2 inputs, 1 output with active constraint
    # Constraint: x1^2 + x2^2 <= 0.05  (active when p = [0.2, 0.3])
    model = DiffOpt.nonlinear_diff_model(Ipopt.Optimizer)
    set_silent(model)

    p_val = [0.2, 0.3]
    @variable(model, x[1:2])
    @variable(model, p[1:2] in Parameter.(p_val))

    # Minimize sum((x - p)^2)
    @objective(model, Min, sum((x .- p) .^ 2))

    # f(x) = x1^2 + x2^2
    function eval_f(ret::AbstractVector, z::AbstractVector)
        ret[1] = z[1]^2 + z[2]^2
        return
    end

    # Jacobian: df/dx1 = 2*x1, df/dx2 = 2*x2
    jacobian_structure = [(1, 1), (1, 2)]

    function eval_jacobian(ret::AbstractVector, z::AbstractVector)
        ret[1] = 2.0 * z[1]
        ret[2] = 2.0 * z[2]
        return
    end

    # Hessian: H = [2 0; 0 2]
    hessian_lagrangian_structure = [(1, 1), (2, 2)]

    function eval_hessian_lagrangian(
        ret::AbstractVector,
        z::AbstractVector,
        μ::AbstractVector,
    )
        ret[1] = 2.0 * μ[1]
        ret[2] = 2.0 * μ[1]
        return
    end

    set = MOI.VectorNonlinearOracle(;
        dimension = 2,
        l = [-Inf],
        u = [0.05],  # This will be active: 0.2^2 + 0.3^2 = 0.13 > 0.05
        eval_f,
        jacobian_structure,
        eval_jacobian,
        hessian_lagrangian_structure,
        eval_hessian_lagrangian,
    )

    @constraint(model, c, [x[1], x[2]] in set)

    optimize!(model)
    @test is_solved_and_feasible(model)

    # Constraint is active, so x should be on the boundary: x1^2 + x2^2 = 0.05
    @test value(x[1])^2 + value(x[2])^2 ≈ 0.05 atol = 1e-6

    # Test forward differentiation
    DiffOpt.empty_input_sensitivities!(model)
    DiffOpt.set_forward_parameter(model, p[1], 0.1)
    DiffOpt.set_forward_parameter(model, p[2], 0.0)
    DiffOpt.forward_differentiate!(model)

    # The sensitivity should be non-trivial since constraint is active
    dx1 = MOI.get(model, DiffOpt.ForwardVariablePrimal(), x[1])
    dx2 = MOI.get(model, DiffOpt.ForwardVariablePrimal(), x[2])
    # Just verify we get reasonable values (not NaN or Inf)
    @test isfinite(dx1)
    @test isfinite(dx2)
end

function test_VectorNonlinearOracle_equality_constraint()
    # Test with equality constraint
    # Constraint: x1^2 + x2^2 == 0.05
    model = DiffOpt.nonlinear_diff_model(Ipopt.Optimizer)
    set_silent(model)

    p_val = [0.2, 0.3]
    @variable(model, x[1:2])
    @variable(model, p[1:2] in Parameter.(p_val))

    # Minimize sum((x - p)^2)
    @objective(model, Min, sum((x .- p) .^ 2))

    # f(x) = x1^2 + x2^2
    function eval_f(ret::AbstractVector, z::AbstractVector)
        ret[1] = z[1]^2 + z[2]^2
        return
    end

    jacobian_structure = [(1, 1), (1, 2)]

    function eval_jacobian(ret::AbstractVector, z::AbstractVector)
        ret[1] = 2.0 * z[1]
        ret[2] = 2.0 * z[2]
        return
    end

    hessian_lagrangian_structure = [(1, 1), (2, 2)]

    function eval_hessian_lagrangian(
        ret::AbstractVector,
        z::AbstractVector,
        μ::AbstractVector,
    )
        ret[1] = 2.0 * μ[1]
        ret[2] = 2.0 * μ[1]
        return
    end

    # Equality constraint: f(x) == 0.05
    set = MOI.VectorNonlinearOracle(;
        dimension = 2,
        l = [0.05],
        u = [0.05],  # l == u means equality constraint
        eval_f,
        jacobian_structure,
        eval_jacobian,
        hessian_lagrangian_structure,
        eval_hessian_lagrangian,
    )

    @constraint(model, c, [x[1], x[2]] in set)

    optimize!(model)
    @test is_solved_and_feasible(model)

    # Solution should satisfy equality: x1^2 + x2^2 = 0.05
    @test value(x[1])^2 + value(x[2])^2 ≈ 0.05 atol = 1e-6

    # Test forward differentiation
    DiffOpt.empty_input_sensitivities!(model)
    DiffOpt.set_forward_parameter(model, p[1], 0.1)
    DiffOpt.set_forward_parameter(model, p[2], 0.0)
    DiffOpt.forward_differentiate!(model)

    dx1 = MOI.get(model, DiffOpt.ForwardVariablePrimal(), x[1])
    dx2 = MOI.get(model, DiffOpt.ForwardVariablePrimal(), x[2])
    @test isfinite(dx1)
    @test isfinite(dx2)
end

function test_VectorNonlinearOracle_geq_constraint()
    # Test with >= constraint (GreaterThan)
    # Constraint: x1^2 + x2^2 >= 0.5
    model = DiffOpt.nonlinear_diff_model(Ipopt.Optimizer)
    set_silent(model)

    # Start at p = [0.5, 0.5] so constraint is active (0.25 + 0.25 = 0.5)
    p_val = [0.5, 0.5]
    @variable(model, x[1:2])
    @variable(model, p[1:2] in Parameter.(p_val))

    # Minimize sum((x - p)^2)
    @objective(model, Min, sum((x .- p) .^ 2))

    function eval_f(ret::AbstractVector, z::AbstractVector)
        ret[1] = z[1]^2 + z[2]^2
        return
    end

    jacobian_structure = [(1, 1), (1, 2)]

    function eval_jacobian(ret::AbstractVector, z::AbstractVector)
        ret[1] = 2.0 * z[1]
        ret[2] = 2.0 * z[2]
        return
    end

    hessian_lagrangian_structure = [(1, 1), (2, 2)]

    function eval_hessian_lagrangian(
        ret::AbstractVector,
        z::AbstractVector,
        μ::AbstractVector,
    )
        ret[1] = 2.0 * μ[1]
        ret[2] = 2.0 * μ[1]
        return
    end

    # GreaterThan constraint: f(x) >= 0.5
    set = MOI.VectorNonlinearOracle(;
        dimension = 2,
        l = [0.5],
        u = [Inf],
        eval_f,
        jacobian_structure,
        eval_jacobian,
        hessian_lagrangian_structure,
        eval_hessian_lagrangian,
    )

    @constraint(model, c, [x[1], x[2]] in set)

    optimize!(model)
    @test is_solved_and_feasible(model)

    # Solution should satisfy: x1^2 + x2^2 >= 0.5
    @test value(x[1])^2 + value(x[2])^2 >= 0.5 - 1e-6

    # Test reverse differentiation
    DiffOpt.empty_input_sensitivities!(model)
    DiffOpt.set_reverse_variable(model, x[1], 1.0)
    DiffOpt.set_reverse_variable(model, x[2], 1.0)
    DiffOpt.reverse_differentiate!(model)

    dp1 = DiffOpt.get_reverse_parameter(model, p[1])
    dp2 = DiffOpt.get_reverse_parameter(model, p[2])
    @test isfinite(dp1)
    @test isfinite(dp2)
end

function test_VectorNonlinearOracle_hessian_transpose()
    # Test with Hessian structure that requires transposition (r < c)
    # This tests the else branch in the multivariate Hessian computation
    model = DiffOpt.nonlinear_diff_model(Ipopt.Optimizer)
    set_silent(model)

    p_val = [0.2, 0.3]
    @variable(model, x[1:2])
    @variable(model, p[1:2] in Parameter.(p_val))

    @objective(model, Min, sum((x .- p) .^ 2))

    # f(x) = x1 * x2 (cross term gives off-diagonal Hessian)
    function eval_f(ret::AbstractVector, z::AbstractVector)
        ret[1] = z[1] * z[2]
        return
    end

    jacobian_structure = [(1, 1), (1, 2)]

    function eval_jacobian(ret::AbstractVector, z::AbstractVector)
        ret[1] = z[2]  # df/dx1 = x2
        ret[2] = z[1]  # df/dx2 = x1
        return
    end

    # Hessian of x1*x2: d²f/dx1dx2 = 1
    # Provide as (1, 2) to test the transpose path
    hessian_lagrangian_structure = [(1, 2)]

    function eval_hessian_lagrangian(
        ret::AbstractVector,
        z::AbstractVector,
        μ::AbstractVector,
    )
        ret[1] = μ[1]  # d²L/dx1dx2 = μ[1]
        return
    end

    set = MOI.VectorNonlinearOracle(;
        dimension = 2,
        l = [-Inf],
        u = [1.0],
        eval_f,
        jacobian_structure,
        eval_jacobian,
        hessian_lagrangian_structure,
        eval_hessian_lagrangian,
    )

    @constraint(model, c, [x[1], x[2]] in set)

    optimize!(model)
    @test is_solved_and_feasible(model)

    # x = p = [0.2, 0.3], f(x) = 0.06 < 1, so constraint is inactive
    @test value(x[1]) ≈ p_val[1] atol = 1e-6
    @test value(x[2]) ≈ p_val[2] atol = 1e-6

    # Test forward differentiation
    DiffOpt.empty_input_sensitivities!(model)
    DiffOpt.set_forward_parameter(model, p[1], 1.0)
    DiffOpt.forward_differentiate!(model)

    dx1 = MOI.get(model, DiffOpt.ForwardVariablePrimal(), x[1])
    @test dx1 ≈ 1.0 atol = 1e-5
end

function test_VectorNonlinearOracle_bridge_utility_paths()
    NLP = DiffOpt.NonLinearProgram

    # Minimal one-row oracle.
    function eval_f(ret::AbstractVector, z::AbstractVector)
        ret[1] = z[1]
        return
    end
    jacobian_structure = [(1, 1)]
    function eval_jacobian(ret::AbstractVector, z::AbstractVector)
        ret[1] = 1.0
        return
    end
    hessian_lagrangian_structure = [(1, 1)]
    function eval_hessian_lagrangian(
        ret::AbstractVector,
        z::AbstractVector,
        μ::AbstractVector,
    )
        ret[1] = 0.0
        return
    end
    set = MOI.VectorNonlinearOracle(;
        dimension = 1,
        l = [-Inf],
        u = [1.0],
        eval_f,
        jacobian_structure,
        eval_jacobian,
        hessian_lagrangian_structure,
        eval_hessian_lagrangian,
    )
    f = MOI.VectorOfVariables([MOI.VariableIndex(1)])

    leq = MOI.ConstraintIndex{MOI.ScalarNonlinearFunction,MOI.LessThan{Float64}}
    geq = MOI.ConstraintIndex{
        MOI.ScalarNonlinearFunction,
        MOI.GreaterThan{Float64},
    }
    eq = MOI.ConstraintIndex{MOI.ScalarNonlinearFunction,MOI.EqualTo{Float64}}

    bridge =
        NLP.VNOToScalarNLBridge{Float64}(f, set, [leq(1)], [geq(2)], [eq(3)])

    @test MOI.get(NLP.Form(), MOI.ConstraintFunction(), bridge) == f
    @test MOI.get(NLP.Form(), MOI.ConstraintSet(), bridge) == set

    # Cover `_unwrap_to_form` branches for direct form, `optimizer` wrapper, and error.
    form = NLP.Form()
    @test NLP._unwrap_to_form(form) === form
    wrapper = (optimizer = form,)
    @test NLP._unwrap_to_form(wrapper) === form
    @test_throws ErrorException NLP._unwrap_to_form(nothing)

    mock = _BridgeMockModel()
    @test MOI.supports(
        mock,
        MOI.ConstraintPrimalStart(),
        NLP.VNOToScalarNLBridge{Float64},
    )
    @test MOI.supports(
        mock,
        MOI.ConstraintDualStart(),
        NLP.VNOToScalarNLBridge{Float64},
    )

    b_delete = NLP.VNOToScalarNLBridge{Float64}(
        f,
        set,
        [leq(10), leq(11)],
        [geq(20)],
        [eq(30)],
    )
    MOI.delete(mock, b_delete)
    @test mock.deleted == MOI.ConstraintIndex[leq(10), leq(11), geq(20), eq(30)]

    # Length mismatch should hit the early return path without setting any dual starts.
    MOI.set(mock, MOI.ConstraintDualStart(), bridge, [1.0, 2.0])
    @test isempty(mock.dual_start)
end

end
