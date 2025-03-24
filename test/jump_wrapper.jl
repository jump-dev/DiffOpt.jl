# Copyright (c) 2020: Akshay Sharma and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module TestJuMPWrapper

using Test
using JuMP
import DiffOpt
import HiGHS
import Ipopt
import SCS
import MathOptInterface as MOI

const ATOL = 1e-3
const RTOL = 1e-3

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

function test_jump_api()
    for (MODEL, SOLVER) in [
            (DiffOpt.diff_model, Ipopt.Optimizer),
            (DiffOpt.quadratic_diff_model, HiGHS.Optimizer),
            (DiffOpt.quadratic_diff_model, SCS.Optimizer),
            (DiffOpt.quadratic_diff_model, Ipopt.Optimizer),
            (DiffOpt.conic_diff_model, HiGHS.Optimizer),
            (DiffOpt.conic_diff_model, SCS.Optimizer), # conicmodel has a issue with sign
            (DiffOpt.conic_diff_model, Ipopt.Optimizer),
            # (DiffOpt.nonlinear_diff_model, HiGHS.Optimizer), #  SQF ctr not supported?
            # (DiffOpt.nonlinear_diff_model, SCS.Optimizer), # returns zero for sensitivity
            (DiffOpt.nonlinear_diff_model, Ipopt.Optimizer),
        ],
        ineq in [true, false],
        min in [true, false],
        flip in [true, false]

        @testset "$(MODEL) with: $(SOLVER), $(ineq ? "ineqs" : "eqs"), $(min ? "Min" : "Max"), $(flip ? "geq" : "leq")" begin
            model = MODEL(SOLVER)
            set_silent(model)

            p_val = 4.0
            pc_val = 2.0
            @variable(model, x)
            @variable(model, p in Parameter(p_val))
            @variable(model, pc in Parameter(pc_val))
            if ineq
                if !flip
                    cons = @constraint(model, pc * x >= 3 * p)
                else
                    cons = @constraint(model, pc * x <= 3 * p)
                end
            else
                @constraint(model, cons, pc * x == 3 * p)
            end
            sign = flip ? -1 : 1
            if min
                @objective(model, Min, 2x * sign)
            else
                @objective(model, Max, -2x * sign)
            end
            optimize!(model)
            @test value(x) ≈ 3 * p_val / pc_val atol = ATOL rtol = RTOL

            # the function is
            # x(p, pc) = 3p / pc
            # hence,
            # dx/dp = 3 / pc
            # dx/dpc = -3p / pc^2

            # First, try forward mode AD

            # differentiate w.r.t. p
            direction_p = 3.0
            DiffOpt.set_forward_parameter(model, p, direction_p)
            DiffOpt.forward_differentiate!(model)
            @test DiffOpt.get_forward_variable(model, x) ≈
                  direction_p * 3 / pc_val atol = ATOL rtol = RTOL

            # update p and pc
            p_val = 2.0
            pc_val = 6.0
            set_parameter_value(p, p_val)
            set_parameter_value(pc, pc_val)
            # re-optimize
            optimize!(model)
            # check solution
            @test value(x) ≈ 3 * p_val / pc_val atol = ATOL rtol = RTOL

            # stop differentiating with respect to p
            DiffOpt.empty_input_sensitivities!(model)
            # differentiate w.r.t. pc
            direction_pc = 10.0
            DiffOpt.set_forward_parameter(model, pc, direction_pc)
            DiffOpt.forward_differentiate!(model)
            @test DiffOpt.get_forward_variable(model, x) ≈
                  -direction_pc * 3 * p_val / pc_val^2 atol = ATOL rtol = RTOL

            # always a good practice to clear previously set sensitivities
            DiffOpt.empty_input_sensitivities!(model)
            # Now, reverse model AD
            direction_x = 10.0
            DiffOpt.set_reverse_variable(model, x, direction_x)
            DiffOpt.reverse_differentiate!(model)
            @test DiffOpt.get_reverse_parameter(model, p) ≈
                  direction_x * 3 / pc_val atol = ATOL rtol = RTOL
            @test DiffOpt.get_reverse_parameter(model, pc) ≈
                  -direction_x * 3 * p_val / pc_val^2 atol = ATOL rtol = RTOL
        end
    end
end

end # module

TestJuMPWrapper.runtests()
