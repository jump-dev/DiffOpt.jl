# Copyright (c) 2020: Akshay Sharma and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

# using Revise

module TestParameters

using Test
using JuMP
import DiffOpt
import MathOptInterface as MOI
import HiGHS
import SCS

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

function test_diff_rhs()
    model = Model(() -> DiffOpt.diff_optimizer(HiGHS.Optimizer, with_parametric_opt_interface = true))
    set_silent(model)
    @variable(model, x)
    @variable(model, p in Parameter(3.0))
    @constraint(model, cons, x >= 3 * p)
    @objective(model, Min, 2x)
    optimize!(model)
    @test value(x) ≈ 9
    # the function is
    # x(p) = 3p, hence x'(p) = 3
    # differentiate w.r.t. p
    # MOI.set(model, DiffOpt.ForwardConstraintSet(), ParameterRef(p), 1)
    DiffOpt.set_forward_parameter(model, p, 1)
    DiffOpt.forward_differentiate!(model)
    # @test MOI.get(model, DiffOpt.ForwardVariablePrimal(), x) ≈ 3
    @test DiffOpt.get_forward_variable(model, x) ≈ 3
    # again with different "direction"
    # MOI.set(model, DiffOpt.ForwardConstraintSet(), ParameterRef(p), 2)
    DiffOpt.set_forward_parameter(model, p, 2)
    DiffOpt.forward_differentiate!(model)
    # @test MOI.get(model, DiffOpt.ForwardVariablePrimal(), x) ≈ 6
    @test DiffOpt.get_forward_variable(model, x) ≈ 6
    #
    set_parameter_value(p, 2.0)
    optimize!(model)
    @test value(x) ≈ 6
    # differentiate w.r.t. p
    # MOI.set(model, DiffOpt.ForwardConstraintSet(), ParameterRef(p), 1)
    DiffOpt.set_forward_parameter(model, p, 1)
    DiffOpt.forward_differentiate!(model)
    # @test MOI.get(model, DiffOpt.ForwardVariablePrimal(), x) ≈ 3
    @test DiffOpt.get_forward_variable(model, x) ≈ 3
    # again with different "direction"
    # MOI.set(model, DiffOpt.ForwardConstraintSet(), ParameterRef(p), 2)
    DiffOpt.set_forward_parameter(model, p, 2)
    DiffOpt.forward_differentiate!(model)
    # @test MOI.get(model, DiffOpt.ForwardVariablePrimal(), x) ≈ 6
    @test DiffOpt.get_forward_variable(model, x) ≈ 6
    #
    # test reverse mode
    #
    # MOI.set(model, DiffOpt.ReverseVariablePrimal(), x, 1)
    DiffOpt.set_reverse_variable(model, x, 1)
    DiffOpt.reverse_differentiate!(model)
    # @test MOI.get(model, DiffOpt.ReverseConstraintSet(), ParameterRef(p)) ≈ 3
    @test DiffOpt.get_reverse_parameter(model, p) ≈ 3
    # again with different "direction"
    # MOI.set(model, DiffOpt.ReverseVariablePrimal(), x, 2)
    DiffOpt.set_reverse_variable(model, x, 2)
    DiffOpt.reverse_differentiate!(model)
    # @test MOI.get(model, DiffOpt.ReverseConstraintSet(), ParameterRef(p)) ≈ 6
    @test DiffOpt.get_reverse_parameter(model, p) ≈ 6
    return
end

function test_diff_vector_rhs()
    model = direct_model(DiffOpt.diff_optimizer(SCS.Optimizer, with_parametric_opt_interface = true))
    set_silent(model)
    @variable(model, x)
    @variable(model, p in Parameter(3.0))
    @constraint(model, cons, [x - 3 * p] in MOI.Zeros(1))

    # FIXME
    @constraint(model, fake_soc, [0, 0, 0] in SecondOrderCone())

    @objective(model, Min, 2x)
    optimize!(model)
    @test isapprox(value(x), 9, atol = 1e-3)
    # the function is
    # x(p) = 3p, hence x'(p) = 3
    # differentiate w.r.t. p
    for p_val in 0:3
        set_parameter_value(p, p_val)
        optimize!(model)
        @test isapprox(
            value(x),
            3 * p_val,
            atol = 1e-3,
        )
        for direction in 0:3
            MOI.set(model, DiffOpt.ForwardConstraintSet(), ParameterRef(p), direction)
            DiffOpt.forward_differentiate!(model)
            @test isapprox(
                MOI.get(model, DiffOpt.ForwardVariablePrimal(), x),
                direction * 3,
                atol = 1e-3,
            )
            # reverse mode
            MOI.set(model, DiffOpt.ReverseVariablePrimal(), x, direction)
            DiffOpt.reverse_differentiate!(model)
            @test isapprox(
                MOI.get(model, DiffOpt.ReverseConstraintSet(), ParameterRef(p)),
                direction * 3,
                atol = 1e-3,
            )
        end
    end
    return
end

function test_affine_changes()
    model = Model(() -> DiffOpt.diff_optimizer(HiGHS.Optimizer, with_parametric_opt_interface = true))
    direct_model
    set_silent(model)
    p_val = 3.0
    pc_val = 1.0
    @variable(model, x)
    @variable(model, p in Parameter(p_val))
    @variable(model, pc in Parameter(pc_val))
    @constraint(model, cons, pc * x >= 3 * p)
    @objective(model, Min, 2x)
    optimize!(model)
    @test value(x) ≈ 3 * p_val / pc_val
    # the function is
    # x(p, pc) = 3p / pc, hence dx/dp = 3 / pc, dx/dpc = -3p / pc^2
    # differentiate w.r.t. p
    for direction_p in 1:1#2
        MOI.set(model, DiffOpt.ForwardConstraintSet(), ParameterRef(p), direction_p)
        DiffOpt.forward_differentiate!(model)
        @test MOI.get(model, DiffOpt.ForwardVariablePrimal(), x) ≈
              direction_p * 3 / pc_val
    end
    # update p
    p_val = 2.0
    set_parameter_value(p, p_val)
    optimize!(model)
    @test value(x) ≈ 3 * p_val / pc_val
    # differentiate w.r.t. p
    for direction_p in 1:2
        MOI.set(model, DiffOpt.ForwardConstraintSet(), ParameterRef(p), direction_p)
        DiffOpt.forward_differentiate!(model)
        @test MOI.get(model, DiffOpt.ForwardVariablePrimal(), x) ≈
              direction_p * 3 / pc_val
    end
    # differentiate w.r.t. pc
    # stop differentiating with respect to p
    direction_p = 0.0
    MOI.set(model, DiffOpt.ForwardConstraintSet(), ParameterRef(p), direction_p)
    for direction_pc in 1:2
        MOI.set(model, DiffOpt.ForwardConstraintSet(), ParameterRef(pc), direction_pc)
        DiffOpt.forward_differentiate!(model)
        @test MOI.get(model, DiffOpt.ForwardVariablePrimal(), x) ≈
              -direction_pc * 3 * p_val / pc_val^2
    end
    # update pc
    pc_val = 2.0
    set_parameter_value(pc, pc_val)
    optimize!(model)
    @test value(x) ≈ 3 * p_val / pc_val
    for direction_pc in 1:1#2
        MOI.set(model, DiffOpt.ForwardConstraintSet(), ParameterRef(pc), direction_pc)
        DiffOpt.forward_differentiate!(model)
        @test MOI.get(model, DiffOpt.ForwardVariablePrimal(), x) ≈
              -direction_pc * 3 * p_val / pc_val^2
    end
    # test combines directions
    for direction_pc in 1:2, direction_p in 1:2
        MOI.set(model, DiffOpt.ForwardConstraintSet(), ParameterRef(p), direction_p)
        MOI.set(model, DiffOpt.ForwardConstraintSet(), ParameterRef(pc), direction_pc)
        DiffOpt.forward_differentiate!(model)
        @test MOI.get(model, DiffOpt.ForwardVariablePrimal(), x) ≈
              -direction_pc * 3 * p_val / pc_val^2 + direction_p * 3 / pc_val
    end
    return
end

function test_affine_changes_compact()
    model = Model(() -> DiffOpt.diff_optimizer(HiGHS.Optimizer, with_parametric_opt_interface = true))
    set_silent(model)
    p_val = 3.0
    pc_val = 1.0
    @variable(model, x)
    @variable(model, p in Parameter(p_val))
    @variable(model, pc in Parameter(pc_val))
    @constraint(model, cons, pc * x >= 3 * p)
    @objective(model, Min, 2x)
    # the function is
    # x(p, pc) = 3p / pc, hence dx/dp = 3 / pc, dx/dpc = -3p / pc^2
    for p_val in 1:3, pc_val in 1:3
        set_parameter_value(p, p_val)
        set_parameter_value(pc, pc_val)
        optimize!(model)
        @test value(x) ≈ 3 * p_val / pc_val
        for direction_pc in 0:2, direction_p in 0:2
            MOI.set(model, DiffOpt.ForwardConstraintSet(), ParameterRef(p), direction_p)
            MOI.set(model, DiffOpt.ForwardConstraintSet(), ParameterRef(pc), direction_pc)
            DiffOpt.forward_differentiate!(model)
            @test MOI.get(model, DiffOpt.ForwardVariablePrimal(), x) ≈
                  -direction_pc * 3 * p_val / pc_val^2 +
                  direction_p * 3 / pc_val
        end
        for direction_x in 0:2
            MOI.set(model, DiffOpt.ReverseVariablePrimal(), x, direction_x)
            DiffOpt.reverse_differentiate!(model)
            @test MOI.get(model, DiffOpt.ReverseConstraintSet(), ParameterRef(p)) ≈
                  direction_x * 3 / pc_val
            @test MOI.get(model, DiffOpt.ReverseConstraintSet(), ParameterRef(pc)) ≈
                  -direction_x * 3 * p_val / pc_val^2
        end
    end
    return
end

function test_quadratic_rhs_changes()
    model = Model(() -> DiffOpt.diff_optimizer(HiGHS.Optimizer, with_parametric_opt_interface = true))
    set_silent(model)
    p_val = 2.0
    q_val = 2.0
    r_val = 2.0
    s_val = 2.0
    t_val = 2.0
    @variable(model, x)
    @variable(model, p in Parameter(p_val))
    @variable(model, q in Parameter(q_val))
    @variable(model, r in Parameter(r_val))
    @variable(model, s in Parameter(s_val))
    @variable(model, t in Parameter(t_val))
    @constraint(model, cons, 11 * t * x >= 1 + 3 * p * q + 5 * r^2 + 7 * s)
    @objective(model, Min, 2x)
    # the function is
    # x(p, q, r, s, t) = (1 + 3pq + 5r^2 + 7s) / (11t)
    # hence
    # dx/dp = 3q / (11t)
    # dx/dq = 3p / (11t)
    # dx/dr = 10r / (11t)
    # dx/ds = 7 / (11t)
    # dx/dt = - (1 + 3pq + 5r^2 + 7s) / (11t^2)
    optimize!(model)
    @test value(x) ≈
          (1 + 3 * p_val * q_val + 5 * r_val^2 + 7 * s_val) / (11 * t_val)
    for p_val in 2:3, q_val in 2:3, r_val in 2:3, s_val in 2:3, t_val in 2:3
        set_parameter_value(p, p_val)
        set_parameter_value(q, q_val)
        set_parameter_value(r, r_val)
        set_parameter_value(s, s_val)
        set_parameter_value(t, t_val)
        optimize!(model)
        @test value(x) ≈
              (1 + 3 * p_val * q_val + 5 * r_val^2 + 7 * s_val) / (11 * t_val)
        for dir_p in 0:2, dir_q in 0:2, dir_r in 0:2, dir_s in 0:2, dir_t in 0:2
            MOI.set(model, DiffOpt.ForwardConstraintSet(), ParameterRef(p), dir_p)
            MOI.set(model, DiffOpt.ForwardConstraintSet(), ParameterRef(q), dir_q)
            MOI.set(model, DiffOpt.ForwardConstraintSet(), ParameterRef(r), dir_r)
            MOI.set(model, DiffOpt.ForwardConstraintSet(), ParameterRef(s), dir_s)
            MOI.set(model, DiffOpt.ForwardConstraintSet(), ParameterRef(t), dir_t)
            DiffOpt.forward_differentiate!(model)
            @test isapprox(
                MOI.get(model, DiffOpt.ForwardVariablePrimal(), x),
                dir_p * 3 * q_val / (11 * t_val) +
                dir_q * 3 * p_val / (11 * t_val) +
                dir_r * 10 * r_val / (11 * t_val) +
                dir_s * 7 / (11 * t_val) +
                dir_t * (
                    -(1 + 3 * p_val * q_val + 5 * r_val^2 + 7 * s_val) /
                    (11 * t_val^2)
                ),
                atol = 1e-10,
            )
        end
        for dir_x in 0:3
            MOI.set(model, DiffOpt.ReverseVariablePrimal(), x, dir_x)
            DiffOpt.reverse_differentiate!(model)
            @test isapprox(
                MOI.get(model, DiffOpt.ReverseConstraintSet(), ParameterRef(p)),
                dir_x * 3 * q_val / (11 * t_val),
                atol = 1e-10,
            )
            @test isapprox(
                MOI.get(model, DiffOpt.ReverseConstraintSet(), ParameterRef(q)),
                dir_x * 3 * p_val / (11 * t_val),
                atol = 1e-10,
            )
            @test isapprox(
                MOI.get(model, DiffOpt.ReverseConstraintSet(), ParameterRef(r)),
                dir_x * 10 * r_val / (11 * t_val),
                atol = 1e-10,
            )
            @test isapprox(
                MOI.get(model, DiffOpt.ReverseConstraintSet(), ParameterRef(s)),
                dir_x * 7 / (11 * t_val),
                atol = 1e-10,
            )
            @test isapprox(
                MOI.get(model, DiffOpt.ReverseConstraintSet(), ParameterRef(t)),
                dir_x * (
                    -(1 + 3 * p_val * q_val + 5 * r_val^2 + 7 * s_val) /
                    (11 * t_val^2)
                ),
                atol = 1e-10,
            )
        end
    end
    return
end

function test_affine_changes_compact_max()
    model = Model(() -> DiffOpt.diff_optimizer(HiGHS.Optimizer, with_parametric_opt_interface = true))
    set_silent(model)
    p_val = 3.0
    pc_val = 1.0
    @variable(model, x)
    @variable(model, p in Parameter(p_val))
    @variable(model, pc in Parameter(pc_val))
    @constraint(model, cons, pc * x >= 3 * p)
    @objective(model, Max, -2x)
    # the function is
    # x(p, pc) = 3p / pc, hence dx/dp = 3 / pc, dx/dpc = -3p / pc^2
    for p_val in 1:3, pc_val in 1:3
        set_parameter_value(p, p_val)
        set_parameter_value(pc, pc_val)
        optimize!(model)
        @test value(x) ≈ 3 * p_val / pc_val
        for direction_pc in 0:2, direction_p in 0:2
            MOI.set(model, DiffOpt.ForwardConstraintSet(), ParameterRef(p), direction_p)
            MOI.set(model, DiffOpt.ForwardConstraintSet(), ParameterRef(pc), direction_pc)
            DiffOpt.forward_differentiate!(model)
            @test MOI.get(model, DiffOpt.ForwardVariablePrimal(), x) ≈
                  -direction_pc * 3 * p_val / pc_val^2 +
                  direction_p * 3 / pc_val
        end
    end
    return
end

function test_diff_affine_objective()
    model = Model(() -> DiffOpt.diff_optimizer(HiGHS.Optimizer, with_parametric_opt_interface = true))
    set_silent(model)
    p_val = 3.0
    @variable(model, x)
    @variable(model, p in Parameter(p_val))
    @constraint(model, cons, x >= 3)
    @objective(model, Min, 2x + 3p)
    # x(p, pc) = 3, hence dx/dp = 0
    for p_val in 1:2
        set_parameter_value(p, p_val)
        optimize!(model)
        @test value(x) ≈ 3
        for direction_p in 0:2
            MOI.set(model, DiffOpt.ForwardConstraintSet(), ParameterRef(p), direction_p)
            DiffOpt.forward_differentiate!(model)
            @test MOI.get(model, DiffOpt.ForwardVariablePrimal(), x) ≈ 0.0
            # reverse mode
            MOI.set(model, DiffOpt.ReverseVariablePrimal(), x, direction_p)
            DiffOpt.reverse_differentiate!(model)
            @test MOI.get(model, DiffOpt.ReverseConstraintSet(), ParameterRef(p)) ≈ 0.0
        end
    end
    return
end

function test_diff_quadratic_objective()
    model = Model(() -> DiffOpt.diff_optimizer(HiGHS.Optimizer, with_parametric_opt_interface = true))
    set_silent(model)
    p_val = 3.0
    @variable(model, x)
    @variable(model, p in Parameter(p_val))
    @constraint(model, cons, x >= 3)
    @objective(model, Min, p * x)
    # x(p, pc) = 3, hence dx/dp = 0
    for p_val in 1:2
        set_parameter_value(p, p_val)
        optimize!(model)
        @test value(x) ≈ 3
        for direction_p in 0:2
            MOI.set(model, DiffOpt.ForwardConstraintSet(), ParameterRef(p), direction_p)
            DiffOpt.forward_differentiate!(model)
            @test MOI.get(model, DiffOpt.ForwardVariablePrimal(), x) ≈ 0.0
            # reverse mode
            MOI.set(model, DiffOpt.ReverseVariablePrimal(), x, direction_p)
            DiffOpt.reverse_differentiate!(model)
            @test MOI.get(model, DiffOpt.ReverseConstraintSet(), ParameterRef(p)) ≈ 0.0
        end
    end
    return
end

function test_quadratic_objective_qp()
    model = Model(() -> DiffOpt.diff_optimizer(HiGHS.Optimizer, with_parametric_opt_interface = true))
    set_silent(model)
    p_val = 3.0
    @variable(model, x)
    @variable(model, p in Parameter(p_val))
    @constraint(model, cons, x >= -10)
    @objective(model, Min, 3 * p * x + x * x + 5 * p + 7 * p^2)
    # 2x + 3p = 0, hence x = -3p/2
    # hence dx/dp = -3/2
    for p_val in 3:3
        set_parameter_value(p, p_val)
        optimize!(model)
        @test value(x) ≈ -3p_val / 2  atol = 1e-4
        for direction_p in 0:2
            MOI.set(model, DiffOpt.ForwardConstraintSet(), ParameterRef(p), direction_p)
            DiffOpt.forward_differentiate!(model)
            @test MOI.get(model, DiffOpt.ForwardVariablePrimal(), x) ≈
                  direction_p * (-3 / 2) atol = 1e-4
            # reverse mode
            MOI.set(model, DiffOpt.ReverseVariablePrimal(), x, direction_p)
            DiffOpt.reverse_differentiate!(model)
            @test MOI.get(model, DiffOpt.ReverseConstraintSet(), ParameterRef(p)) ≈
                  direction_p * (-3 / 2) atol = 1e-4
        end
    end
    return
end

function test_diff_errors()
    model = Model(() -> DiffOpt.diff_optimizer(HiGHS.Optimizer, with_parametric_opt_interface = true))
    set_silent(model)
    @variable(model, x)
    @variable(model, p in Parameter(3.0))
    @constraint(model, cons, x >= 3 * p)
    @objective(model, Min, 2x)
    optimize!(model)
    @test value(x) ≈ 9

    @test_throws ErrorException MOI.set(model, DiffOpt.ForwardConstraintSet(), ParameterRef(x), 1)
    @test_throws ErrorException MOI.set(
        model,
        DiffOpt.ReverseVariablePrimal(),
        p,
        1,
    )
    @test_throws ErrorException MOI.get(
        model,
        DiffOpt.ForwardVariablePrimal(),
        p,
    )
    @test_throws ErrorException MOI.get(model, DiffOpt.ReverseConstraintSet(), ParameterRef(x))

    @test_throws ErrorException MOI.set(model, DiffOpt.ForwardObjectiveFunction(), 3 * x)
    @test_throws ErrorException MOI.set(model, DiffOpt.ForwardConstraintFunction(), cons, 1 + 7 * x)
    @test_throws ErrorException MOI.get(model, DiffOpt.ReverseObjectiveFunction())
    @test_throws ErrorException MOI.get(model, DiffOpt.ReverseConstraintFunction(), cons)

    return
end

end # module

TestParameters.runtests()
