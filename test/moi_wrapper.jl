# Copyright (c) 2020: Akshay Sharma and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module TestMOIWrapper

using Test
import DiffOpt
import HiGHS
import IterativeSolvers
import MathOptInterface as MOI

const ATOL = 2e-4
const RTOL = 2e-4

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

function test_moi_test_runtests()
    model = DiffOpt.diff_optimizer(HiGHS.Optimizer)
    # `Variable.ZerosBridge` makes dual needed by some tests fail.
    # MOI.Bridges.remove_bridge(
    #     model.optimizer.optimizer.optimizer,
    #     MOI.Bridges.Variable.ZerosBridge{Float64},
    # )
    MOI.set(model, MOI.Silent(), true)
    config =
        MOI.Test.Config(; atol = 1e-7, exclude = Any[MOI.compute_conflict!,])
    MOI.Test.runtests(
        model,
        config;
        exclude = Any[
            # removed because of the `ZerosBridge` issue:
            # https://github.com/jump-dev/MathOptInterface.jl/issues/2861
            # - zeros bridge does not support duals because it cumbersome
            # - many bridges do not support get ConstraintFunction because it is cumbersome
            # so there is no way out of this error for now.
            # at the same time this is a modeling corner case tha could be avoided
            # by the user.
            "test_conic_linear_VectorOfVariables_2",
            "test_nonlinear_expression_hs110",
            "test_nonlinear_expression_quartic",
        ],
    )
    return
end

function test_FEASIBILITY_SENSE_zeros_objective()
    model = DiffOpt.diff_optimizer(HiGHS.Optimizer)
    MOI.set(model, MOI.Silent(), true)
    x = MOI.add_variable(model)
    MOI.add_constraint(model, x, MOI.GreaterThan(1.0))
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(model, MOI.ObjectiveFunction{MOI.VariableIndex}(), x)
    MOI.optimize!(model)
    @test MOI.get(model, MOI.ObjectiveValue()) ≈ 1.0 atol = ATOL rtol = RTOL
    MOI.set(model, MOI.ObjectiveSense(), MOI.FEASIBILITY_SENSE)
    MOI.optimize!(model)
    @test MOI.get(model, MOI.ObjectiveValue()) == 0.0
    return
end

function test_forward_or_reverse_without_optimizing_throws()
    model = DiffOpt.diff_optimizer(HiGHS.Optimizer)
    MOI.set(model, MOI.Silent(), true)
    x = MOI.add_variable(model)
    MOI.add_constraint(model, x, MOI.GreaterThan(1.0))
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(model, MOI.ObjectiveFunction{MOI.VariableIndex}(), x)
    # do not optimize, just try to differentiate
    @test_throws ErrorException DiffOpt.forward_differentiate!(model)
    @test_throws ErrorException DiffOpt.reverse_differentiate!(model)
    # impossible constraint
    MOI.add_constraint(model, x, MOI.LessThan(0.5))
    MOI.optimize!(model)
    @test_throws ErrorException DiffOpt.forward_differentiate!(model)
    @test_throws ErrorException DiffOpt.reverse_differentiate!(model)
    return
end

struct TestSolver end

# always use IterativeSolvers
function DiffOpt.QuadraticProgram.solve_system(
    ::TestSolver,
    LHS,
    RHS,
    iterative::Bool,
)
    return IterativeSolvers.lsqr(LHS, RHS)
end

function test_setting_the_linear_solver_in_the_quadratic_solver()
    model = DiffOpt.QuadraticProgram.Model()
    @test MOI.supports(model, DiffOpt.QuadraticProgram.LinearAlgebraSolver())
    @test MOI.get(model, DiffOpt.QuadraticProgram.LinearAlgebraSolver()) ===
          nothing
    MOI.set(model, DiffOpt.QuadraticProgram.LinearAlgebraSolver(), TestSolver())
    @test MOI.get(model, DiffOpt.QuadraticProgram.LinearAlgebraSolver()) ==
          TestSolver()
    MOI.empty!(model)
    @test MOI.get(model, DiffOpt.QuadraticProgram.LinearAlgebraSolver()) ==
          TestSolver()
    return
end

function _test_dU_from_dQ(U, dU)
    dQ = dU'U + U'dU
    _dU = copy(dQ)
    __dU = copy(dQ)
    # Compiling
    DiffOpt.dU_from_dQ!(__dU, U)
    @test @allocated(DiffOpt.dU_from_dQ!(_dU, U)) == 0
    @test _dU ≈ dU
    return
end

function test_dU_from_dQ()
    _test_dU_from_dQ(2ones(1, 1), 3ones(1, 1))
    U = [1 2; 0 1]
    dU = [1 -1; 0 1]
    _test_dU_from_dQ(U, dU)
    U = [-3 5; 0 2.5]
    dU = [2 3.5; 0 -2]
    _test_dU_from_dQ(U, dU)
    U = [1.5 2 -1; 0 -1 3.5; 0 0 -2]
    dU = [2.5 -1 2; 0 5 -3; 0 0 3]
    _test_dU_from_dQ(U, dU)
    return
end

function test_eval_gradient_number()
    model = DiffOpt.diff_optimizer(HiGHS.Optimizer)
    grad = DiffOpt._eval_gradient(model, 5.0)
    @test isempty(grad)
    grad = DiffOpt._eval_gradient(model, 0.0)
    @test isempty(grad)
end

function test_eval_gradient_variable_index()
    model = DiffOpt.diff_optimizer(HiGHS.Optimizer)
    x = MOI.add_variable(model)
    grad = DiffOpt._eval_gradient(model, x)
    @test length(grad) == 1
    @test grad[x] == 1.0
end

function test_eval_gradient_scalar_affine_function()
    model = DiffOpt.diff_optimizer(HiGHS.Optimizer)
    MOI.set(model, MOI.Silent(), true)
    x = MOI.add_variable(model)
    y = MOI.add_variable(model)
    # f = 3x + 5y + 7
    f = MOI.ScalarAffineFunction(
        [MOI.ScalarAffineTerm(3.0, x), MOI.ScalarAffineTerm(5.0, y)],
        7.0,
    )
    grad = DiffOpt._eval_gradient(model, f)
    @test length(grad) == 2
    @test grad[x] == 3.0
    @test grad[y] == 5.0
end

function test_eval_gradient_scalar_affine_function_repeated_variable()
    model = DiffOpt.diff_optimizer(HiGHS.Optimizer)
    x = MOI.add_variable(model)
    # f = 3x + 2x = 5x (repeated variable in terms)
    f = MOI.ScalarAffineFunction(
        [MOI.ScalarAffineTerm(3.0, x), MOI.ScalarAffineTerm(2.0, x)],
        0.0,
    )
    grad = DiffOpt._eval_gradient(model, f)
    @test length(grad) == 1
    @test grad[x] == 5.0
end

function test_eval_gradient_quadratic_diagonal()
    model = DiffOpt.diff_optimizer(HiGHS.Optimizer)
    MOI.set(model, MOI.Silent(), true)
    x = MOI.add_variable(model)
    MOI.add_constraint(model, x, MOI.GreaterThan(0.0))
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    # f = 2x^2 (MOI stores as 0.5 * Q, so coefficient is 4 for 2x^2)
    # df/dx = 4x
    f = MOI.ScalarQuadraticFunction(
        [MOI.ScalarQuadraticTerm(4.0, x, x)],  # 0.5 * 4 * x^2 = 2x^2
        MOI.ScalarAffineTerm{Float64}[],
        0.0,
    )
    MOI.set(model, MOI.ObjectiveFunction{typeof(f)}(), f)
    MOI.optimize!(model)
    # At x=0, gradient should be 0
    grad = DiffOpt._eval_gradient(model, f)
    @test length(grad) == 1
    @test grad[x] ≈ 0.0 atol = ATOL

    # Now test with x = 3
    model2 = DiffOpt.diff_optimizer(HiGHS.Optimizer)
    MOI.set(model2, MOI.Silent(), true)
    x2 = MOI.add_variable(model2)
    MOI.add_constraint(model2, x2, MOI.EqualTo(3.0))
    MOI.set(model2, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    f2 = MOI.ScalarQuadraticFunction(
        [MOI.ScalarQuadraticTerm(4.0, x2, x2)],
        MOI.ScalarAffineTerm{Float64}[],
        0.0,
    )
    MOI.set(model2, MOI.ObjectiveFunction{typeof(f2)}(), f2)
    MOI.optimize!(model2)
    grad2 = DiffOpt._eval_gradient(model2, f2)
    # df/dx = 4 * 3 = 12
    @test grad2[x2] ≈ 12.0 atol = ATOL
end

function test_eval_gradient_quadratic_off_diagonal()
    model = DiffOpt.diff_optimizer(HiGHS.Optimizer)
    MOI.set(model, MOI.Silent(), true)
    x = MOI.add_variable(model)
    y = MOI.add_variable(model)
    MOI.add_constraint(model, x, MOI.EqualTo(2.0))
    MOI.add_constraint(model, y, MOI.EqualTo(5.0))
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    # Use convex objective: 3x^2 + 3y^2 + xy (Hessian [[6,1],[1,6]] is PD)
    # df/dx = 6x + y = 12 + 5 = 17
    # df/dy = x + 6y = 2 + 30 = 32
    v1, v2 = x.value <= y.value ? (x, y) : (y, x)
    f = MOI.ScalarQuadraticFunction(
        [
            MOI.ScalarQuadraticTerm(6.0, x, x),
            MOI.ScalarQuadraticTerm(1.0, v1, v2),
            MOI.ScalarQuadraticTerm(6.0, y, y),
        ],
        MOI.ScalarAffineTerm{Float64}[],
        0.0,
    )
    MOI.set(model, MOI.ObjectiveFunction{typeof(f)}(), f)
    MOI.optimize!(model)
    grad = DiffOpt._eval_gradient(model, f)
    @test length(grad) == 2
    @test grad[x] ≈ 17.0 atol = ATOL
    @test grad[y] ≈ 32.0 atol = ATOL
end

function test_eval_gradient_quadratic_mixed()
    model = DiffOpt.diff_optimizer(HiGHS.Optimizer)
    MOI.set(model, MOI.Silent(), true)
    x = MOI.add_variable(model)
    y = MOI.add_variable(model)
    MOI.add_constraint(model, x, MOI.EqualTo(2.0))
    MOI.add_constraint(model, y, MOI.EqualTo(3.0))
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    # f = x^2 + 2xy + 3y^2 + 5x + 7y
    # df/dx = 2x + 2y + 5 = 4 + 6 + 5 = 15
    # df/dy = 2x + 6y + 7 = 4 + 18 + 7 = 29
    v1, v2 = x.value <= y.value ? (x, y) : (y, x)
    f = MOI.ScalarQuadraticFunction(
        [
            MOI.ScalarQuadraticTerm(2.0, x, x),
            MOI.ScalarQuadraticTerm(2.0, v1, v2),
            MOI.ScalarQuadraticTerm(6.0, y, y),
        ],
        [MOI.ScalarAffineTerm(5.0, x), MOI.ScalarAffineTerm(7.0, y)],
        0.0,
    )
    MOI.set(model, MOI.ObjectiveFunction{typeof(f)}(), f)
    MOI.optimize!(model)
    grad = DiffOpt._eval_gradient(model, f)
    @test length(grad) == 2
    @test grad[x] ≈ 15.0 atol = ATOL
    @test grad[y] ≈ 29.0 atol = ATOL
end

end  # module

TestMOIWrapper.runtests()
