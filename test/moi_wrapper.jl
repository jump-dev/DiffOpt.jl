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
    config = MOI.Test.Config(; atol = 1e-7)
    MOI.Test.runtests(model, config;
        exclude = [
                "test_solve_conflict",
            ]
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

end  # module

TestMOIWrapper.runtests()
