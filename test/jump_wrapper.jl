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
import ParametricOptInterface as POI
import MathOptInterface as MOI
import ParametricOptInterface as POI

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

function test_obj()

        for (MODEL, SOLVER) in [
            (DiffOpt.diff_model, HiGHS.Optimizer),
            # (DiffOpt.diff_model, SCS.Optimizer),
            # (DiffOpt.diff_model, Ipopt.Optimizer),
            # (DiffOpt.quadratic_diff_model, HiGHS.Optimizer),
            # (DiffOpt.quadratic_diff_model, SCS.Optimizer),
            # (DiffOpt.quadratic_diff_model, Ipopt.Optimizer),
            # (DiffOpt.conic_diff_model, HiGHS.Optimizer),
            # (DiffOpt.conic_diff_model, SCS.Optimizer),
            # (DiffOpt.conic_diff_model, Ipopt.Optimizer),
            # (DiffOpt.nonlinear_diff_model, HiGHS.Optimizer),
            # (DiffOpt.nonlinear_diff_model, SCS.Optimizer),
            # (DiffOpt.nonlinear_diff_model, Ipopt.Optimizer),
        ],
        # ineq in [true, false],
        # _min in [true, false],
        # flip in [true, false],
        with_bridge_type in [Float64, nothing]

        if isnothing(with_bridge_type) && SOLVER === SCS.Optimizer
            continue
        end


        MODEL = DiffOpt.diff_model
        SOLVER = HiGHS.Optimizer
        with_bridge_type = Float64
        ineq = false
        _min = true
        flip = false

        @testset "$(MODEL) with: $(SOLVER), $(ineq ? "ineqs" : "eqs"), $(_min ? "Min" : "Max"), $(flip ? "geq" : "leq") bridge:$with_bridge_type" begin
            model = MODEL(SOLVER; with_bridge_type)
            set_silent(model)

            p_val = 4.0
            pc_val = 2.0
            @variable(model, x)
            @variable(model, p in Parameter(p_val))
            @variable(model, pc in Parameter(pc_val))
            # if ineq
            #     if !flip
            #         cons = @constraint(model, con, pc * x >= 3 * p)
            #     else
            #         cons = @constraint(model, con, pc * x <= 3 * p)
            #     end
            # else
                cons = @constraint(model, con, pc * x == 3 * p)
            # end
            # sign = flip ? -1 : 1
            # if _min
            #     @objective(model, Min, 2x * sign)
            # else
            #     @objective(model, Max, -2x * sign)
            # end

            for obj_coef in [-3, 2, 5]
                @objective(model, Min, obj_coef * x)

                optimize!(model)
                @test value(x) ≈ 3 * p_val / pc_val atol = ATOL rtol = RTOL

                DiffOpt.empty_input_sensitivities!(model)
                direction_obj = 2.0
                DiffOpt.set_reverse_objective(model, direction_obj)
                DiffOpt.reverse_differentiate!(model)
                @test DiffOpt.get_reverse_parameter(model, p) ≈ obj_coef * direction_obj * 3 / pc_val atol = ATOL rtol = RTOL
                @test DiffOpt.get_reverse_parameter(model, pc) ≈ -obj_coef * direction_obj * 3 * p_val / (pc_val^2) atol = ATOL rtol = RTOL

                DiffOpt.empty_input_sensitivities!(model)
                direction_p = 3.0
                DiffOpt.set_forward_parameter(model, p, direction_p)
                DiffOpt.forward_differentiate!(model)
                @test DiffOpt.get_forward_objective(model) ≈ obj_coef * direction_p * 3 / pc_val atol = ATOL rtol = RTOL

                            # stop differentiating with respect to p
                DiffOpt.empty_input_sensitivities!(model)
                # differentiate w.r.t. pc
                direction_pc = 10.0
                DiffOpt.set_forward_parameter(model, pc, direction_pc)
                DiffOpt.forward_differentiate!(model)
                @test DiffOpt.get_forward_objective(model) ≈
                    - obj_coef * direction_pc * 3 * p_val / pc_val^2 atol = ATOL rtol = RTOL

            end


        end
    end

    return
end

# TODO test quadratic obj

function test_jump_api()
    for (MODEL, SOLVER) in [
            (DiffOpt.diff_model, HiGHS.Optimizer),
            (DiffOpt.diff_model, SCS.Optimizer),
            (DiffOpt.diff_model, Ipopt.Optimizer),
            (DiffOpt.quadratic_diff_model, HiGHS.Optimizer),
            (DiffOpt.quadratic_diff_model, SCS.Optimizer),
            (DiffOpt.quadratic_diff_model, Ipopt.Optimizer),
            (DiffOpt.conic_diff_model, HiGHS.Optimizer),
            (DiffOpt.conic_diff_model, SCS.Optimizer),
            (DiffOpt.conic_diff_model, Ipopt.Optimizer),
            (DiffOpt.nonlinear_diff_model, HiGHS.Optimizer),
            (DiffOpt.nonlinear_diff_model, SCS.Optimizer),
            (DiffOpt.nonlinear_diff_model, Ipopt.Optimizer),
        ],
        ineq in [true, false],
        _min in [true, false],
        flip in [true, false],
        with_bridge_type in [Float64, nothing]

        if isnothing(with_bridge_type) && SOLVER === SCS.Optimizer
            continue
        end

        @testset "$(MODEL) with: $(SOLVER), $(ineq ? "ineqs" : "eqs"), $(_min ? "Min" : "Max"), $(flip ? "geq" : "leq") bridge:$with_bridge_type" begin
            model = MODEL(SOLVER; with_bridge_type)
            set_silent(model)

            p_val = 4.0
            pc_val = 2.0
            @variable(model, x)
            @variable(model, p in Parameter(p_val))
            @variable(model, pc in Parameter(pc_val))
            if ineq
                if !flip
                    cons = @constraint(model, con, pc * x >= 3 * p)
                else
                    cons = @constraint(model, con, pc * x <= 3 * p)
                end
            else
                cons = @constraint(model, con, pc * x == 3 * p)
            end
            sign = flip ? -1 : 1
            if _min
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
