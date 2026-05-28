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
import LinearAlgebra
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

function test_obj_simple()
    for (MODEL, SOLVER) in [
            (DiffOpt.diff_model, HiGHS.Optimizer),
            (DiffOpt.diff_model, SCS.Optimizer),
            (DiffOpt.diff_model, Ipopt.Optimizer),
        ],
        sign in [+1, -1],
        sign_p in [-1, +1],
        sense in [:Min, :Max],
        with_bridge_type in [Float64, nothing]

        if isnothing(with_bridge_type) && SOLVER === SCS.Optimizer
            continue
        end

        @testset "$(MODEL) with: $(SOLVER), bridge:$with_bridge_type, sign:$sign, sense: $sense, sign_p: $sign_p" begin
            model = MODEL(SOLVER; with_bridge_type)
            set_silent(model)

            p_val = 4.0
            @variable(model, x)
            @variable(model, p in Parameter(p_val))
            @constraint(model, con, x == 3 * sign_p * p)
            @objective(model, Min, 2 * sign * x)
            if sense == :Max
                @objective(model, Max, 2 * sign * x)
            end
            optimize!(model)
            @test value(x) ≈ sign_p * 3 * p_val atol = ATOL rtol = RTOL

            DiffOpt.empty_input_sensitivities!(model)
            direction_obj = 2.0
            DiffOpt.set_reverse_objective(model, direction_obj)
            DiffOpt.reverse_differentiate!(model)
            @test DiffOpt.get_reverse_parameter(model, p) ≈
                  sign_p * sign * 6 * direction_obj atol = ATOL rtol = RTOL

            DiffOpt.empty_input_sensitivities!(model)
            direction_p = 3.0
            DiffOpt.set_forward_parameter(model, p, direction_p)
            DiffOpt.forward_differentiate!(model)
            @test DiffOpt.get_forward_objective(model) ≈
                  sign_p * sign * 6 * direction_p atol = ATOL rtol = RTOL
        end
    end

    return
end

function test_obj_simple_quad()
    # Note: conic_diff_model excluded - doesn't properly support quadratic objectives
    for (MODEL, SOLVER) in [
            (DiffOpt.diff_model, HiGHS.Optimizer),
            (DiffOpt.diff_model, SCS.Optimizer),
            (DiffOpt.diff_model, Ipopt.Optimizer),
            (DiffOpt.quadratic_diff_model, HiGHS.Optimizer),
            (DiffOpt.quadratic_diff_model, SCS.Optimizer),
            (DiffOpt.quadratic_diff_model, Ipopt.Optimizer),
            (DiffOpt.nonlinear_diff_model, HiGHS.Optimizer),
            (DiffOpt.nonlinear_diff_model, SCS.Optimizer),
            (DiffOpt.nonlinear_diff_model, Ipopt.Optimizer),
        ],
        sign in [+1, -1],
        sign_p in [-1, +1],
        sense in [:Min, :Max],
        with_bridge_type in [Float64, nothing]

        if isnothing(with_bridge_type) && SOLVER === SCS.Optimizer
            continue
        end
        # Skip invalid quadratic cases: convex (sign=1) needs Min, concave (sign=-1) needs Max
        if SOLVER != Ipopt.Optimizer &&
           ((sign == 1 && sense == :Max) || (sign == -1 && sense == :Min))
            continue
        end

        @testset "$(MODEL) with: $(SOLVER), bridge:$with_bridge_type, sign:$sign, sense: $sense, sign_p: $sign_p" begin
            model = MODEL(SOLVER; with_bridge_type)
            set_silent(model)

            p_val = 4.0
            @variable(model, x)
            @variable(model, p in Parameter(p_val))
            @constraint(model, con, x == 3 * sign_p * p)
            @objective(model, Min, sign * (2 * x^2 + 7x))
            if sense == :Max
                @objective(model, Max, sign * (2 * x^2 + 7x))
            end
            optimize!(model)
            @test value(x) ≈ sign_p * 3 * p_val atol = ATOL rtol = RTOL

            DiffOpt.empty_input_sensitivities!(model)
            direction_obj = 2.0
            DiffOpt.set_reverse_objective(model, direction_obj)
            DiffOpt.reverse_differentiate!(model)
            @test DiffOpt.get_reverse_parameter(model, p) ≈
                  sign_p * sign * 3 * (4 * value(x) + 7) * direction_obj atol =
                ATOL rtol = RTOL

            DiffOpt.empty_input_sensitivities!(model)
            direction_p = 3.0
            DiffOpt.set_forward_parameter(model, p, direction_p)
            DiffOpt.forward_differentiate!(model)
            @test DiffOpt.get_forward_objective(model) ≈
                  sign_p * sign * 3 * (4 * value(x) + 7) * direction_p atol =
                ATOL rtol = RTOL
        end
    end

    return
end

function _test_obj(MODEL, SOLVER; ineq, _min, flip, with_bridge_type)
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

    for obj_coef in [2, 5]
        sign = flip ? -1 : 1
        dir = _min ? 1 : -1
        if _min
            @objective(model, Min, dir * obj_coef * x * sign)
        else
            @objective(model, Max, dir * obj_coef * x * sign)
        end

        optimize!(model)
        @test value(x) ≈ 3 * p_val / pc_val atol = ATOL rtol = RTOL

        DiffOpt.empty_input_sensitivities!(model)
        direction_obj = 2.0
        DiffOpt.set_reverse_objective(model, direction_obj)
        DiffOpt.reverse_differentiate!(model)
        @test DiffOpt.get_reverse_parameter(model, p) ≈
              dir * sign * obj_coef * direction_obj * 3 / pc_val atol = ATOL rtol =
            RTOL
        @test DiffOpt.get_reverse_parameter(model, pc) ≈
              -dir * sign * obj_coef * direction_obj * 3 * p_val / (pc_val^2) atol =
            ATOL rtol = RTOL

        DiffOpt.empty_input_sensitivities!(model)
        direction_p = 3.0
        DiffOpt.set_forward_parameter(model, p, direction_p)
        DiffOpt.forward_differentiate!(model)
        @test DiffOpt.get_forward_objective(model) ≈
              dir * sign * obj_coef * direction_p * 3 / pc_val atol = ATOL rtol =
            RTOL

        # stop differentiating with respect to p
        DiffOpt.empty_input_sensitivities!(model)
        # differentiate w.r.t. pc
        direction_pc = 10.0
        DiffOpt.set_forward_parameter(model, pc, direction_pc)
        DiffOpt.forward_differentiate!(model)
        @test DiffOpt.get_forward_objective(model) ≈
              -dir * sign * obj_coef * direction_pc * 3 * p_val / pc_val^2 atol =
            ATOL rtol = RTOL
    end
end

function test_obj()
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
            _test_obj(MODEL, SOLVER; ineq, _min, flip, with_bridge_type)
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
            if SOLVER === Ipopt.Optimizer
                # avoiding MOI.TimeLimitSec to cover the RawOptimizerAttribute path
                # only tests that MOI doesn't throw an unsupported error (#335)
                set_optimizer_attribute(model, "max_wall_time", 600.0)
            end

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

    return
end

function test_forward_wrappers_non_parametric()
    # Tests set_forward_objective_function and set_forward_constraint_function
    # overloads on a non-parametric model.
    #
    # min x  s.t.  x + y == 1, x >= 0, y >= 0
    # Solution: x=0, y=1
    # Perturb constraint RHS: x + y + ϵ == 1
    # Sensitivity should be dy/dϵ = -1
    # since the slack goes to y and x is at its lower bound (active).
    model = JuMP.direct_model(DiffOpt.diff_optimizer(HiGHS.Optimizer))
    set_silent(model)
    @variable(model, x >= 0)
    @variable(model, y >= 0)
    @constraint(model, c1, x + y == 1)
    @objective(model, Min, 1.0 * x)
    optimize!(model)
    @test value(x) ≈ 0.0 atol = ATOL
    @test value(y) ≈ 1.0 atol = ATOL

    DiffOpt.set_forward_objective_function(model, 0.0)
    DiffOpt.set_forward_constraint_function(model, c1, 1.0)
    DiffOpt.forward_differentiate!(model)
    dy = DiffOpt.get_forward_variable(model, y)
    @test dy ≈ -1.0 atol = ATOL

    DiffOpt.empty_input_sensitivities!(model)
    DiffOpt.set_forward_constraint_function(model, c1, 0.0 * x + 1.0)
    DiffOpt.forward_differentiate!(model)
    dy2 = DiffOpt.get_forward_variable(model, y)
    @test dy2 ≈ dy atol = ATOL
    return
end

function test_forward_vector_constraint_wrappers()
    # Tests set_forward_constraint_function overloads for vector constraints
    # using conic_diff_model (direct_model + diff_optimizer bridges differently).
    model = DiffOpt.conic_diff_model(SCS.Optimizer)
    set_silent(model)
    @variable(model, x)
    @variable(model, y)
    @constraint(model, c_eq, x + y == 1)
    @constraint(model, c_nn, [1.0 * y, 1.0 * x] in MOI.Nonnegatives(2))
    @objective(model, Min, 1.0 * x)
    optimize!(model)
    @test value(x) ≈ 0.0 atol = ATOL
    @test value(y) ≈ 1.0 atol = ATOL

    DiffOpt.set_forward_constraint_function(model, c_nn, [0.0, 0.0])
    DiffOpt.set_forward_constraint_function(model, c_eq, 1.0)
    DiffOpt.forward_differentiate!(model)
    dy = DiffOpt.get_forward_variable(model, y)
    @test dy ≈ -1.0 atol = ATOL

    DiffOpt.empty_input_sensitivities!(model)
    DiffOpt.set_forward_constraint_function(model, c_nn, [0.0 * x, 0.0 * x])
    DiffOpt.set_forward_constraint_function(model, c_eq, 1.0)
    DiffOpt.forward_differentiate!(model)
    dy2 = DiffOpt.get_forward_variable(model, y)
    @test dy2 ≈ dy atol = ATOL
    return
end

function test_forward_psd_matrix_wrapper()
    # Tests set_forward_constraint_function with AbstractMatrix{<:AbstractJuMPScalar}
    # for PSD cone constraints. Uses a non-parametric model since
    # ForwardConstraintFunction is blocked on parametric models.
    model = Model(() -> DiffOpt.diff_optimizer(SCS.Optimizer))
    set_silent(model)
    @variable(model, x)
    @objective(model, Min, -x)
    @constraint(
        model,
        con,
        LinearAlgebra.Symmetric([1.0-x 0.0; 0.0 x]) in PSDCone(),
    )
    optimize!(model)
    @test value(x) ≈ 1.0 atol = ATOL

    perturbation = [1.0+0.0*x 0.0*x; 0.0*x 0.0*x]
    DiffOpt.set_forward_constraint_function(model, con, perturbation)
    DiffOpt.forward_differentiate!(model)
    @test DiffOpt.get_forward_variable(model, x) ≈ 1.0 atol = ATOL
    return
end

function test_set_get_attribute_smoke()
    # Smoke test for the `set_attribute` / `get_attribute` syntax with every
    # DiffOpt forward and reverse attribute. The wrapper functions are the
    # documented entry points; this covers the MOI attribute path.

    # Parametric NLP via Ipopt — supports every Forward/Reverse attribute,
    # including ForwardConstraintDual which is not implemented on the
    # auto-selected QuadraticProgram backend.
    model = DiffOpt.nonlinear_diff_model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x >= 0)
    @variable(model, y >= 0)
    @variable(model, p in Parameter(1.0))
    @constraint(model, c1, x + y == p)
    @objective(model, Min, 2x^2 + y^2 + x * y + x + y)
    optimize!(model)

    # Forward via attributes: ForwardParameterValue (set),
    # ForwardVariablePrimal (get), ForwardObjectiveSensitivity (get),
    # ForwardConstraintDual (get).
    DiffOpt.empty_input_sensitivities!(model)
    set_attribute(p, DiffOpt.ForwardParameterValue(), 1.0)
    DiffOpt.forward_differentiate!(model)
    @test get_attribute(x, DiffOpt.ForwardVariablePrimal()) ≈ 0.25 atol = ATOL rtol =
        RTOL
    @test get_attribute(y, DiffOpt.ForwardVariablePrimal()) ≈ 0.75 atol = ATOL rtol =
        RTOL
    @test get_attribute(c1, DiffOpt.ForwardConstraintDual()) ≈ 1.75 atol = ATOL rtol =
        RTOL
    @test get_attribute(model, DiffOpt.ForwardObjectiveSensitivity()) ≈ 2.75 atol =
        ATOL rtol = RTOL

    # Reverse via attributes: ReverseVariablePrimal (set),
    # ReverseParameterValue (get), ReverseConstraintSet (get on ParameterRef).
    DiffOpt.empty_input_sensitivities!(model)
    set_attribute(x, DiffOpt.ReverseVariablePrimal(), 1.0)
    DiffOpt.reverse_differentiate!(model)
    dp = get_attribute(p, DiffOpt.ReverseParameterValue())
    @test isfinite(dp)
    rcs = get_attribute(ParameterRef(p), DiffOpt.ReverseConstraintSet()).value
    @test rcs ≈ dp atol = ATOL

    # Reverse from the objective seed: ReverseObjectiveValue (set).
    DiffOpt.empty_input_sensitivities!(model)
    set_attribute(model, DiffOpt.ReverseObjectiveValue(), 1.0)
    DiffOpt.reverse_differentiate!(model)
    @test isfinite(get_attribute(p, DiffOpt.ReverseParameterValue()))

    # Non-parametric model — exercise ForwardObjectiveFunction (set),
    # ForwardConstraintFunction (set, scalar), ReverseObjectiveFunction (get),
    # ReverseConstraintFunction (get), ReverseConstraintDual (set) via the
    # attribute syntax.
    direct = JuMP.direct_model(DiffOpt.diff_optimizer(HiGHS.Optimizer))
    set_silent(direct)
    @variable(direct, x2 >= 0)
    @variable(direct, y2 >= 0)
    @constraint(direct, c2, x2 + y2 == 1)
    @objective(direct, Min, 1.0 * x2)
    optimize!(direct)

    set_attribute(direct, DiffOpt.ForwardObjectiveFunction(), 0.0)
    set_attribute(c2, DiffOpt.ForwardConstraintFunction(), 1.0)
    DiffOpt.forward_differentiate!(direct)
    @test get_attribute(y2, DiffOpt.ForwardVariablePrimal()) ≈ -1.0 atol = ATOL

    DiffOpt.empty_input_sensitivities!(direct)
    set_attribute(y2, DiffOpt.ReverseVariablePrimal(), 1.0)
    DiffOpt.reverse_differentiate!(direct)
    @test get_attribute(direct, DiffOpt.ReverseObjectiveFunction()) isa
          JuMP.AbstractJuMPScalar
    @test get_attribute(c2, DiffOpt.ReverseConstraintFunction()) isa
          JuMP.AbstractJuMPScalar

    DiffOpt.empty_input_sensitivities!(direct)
    set_attribute(c2, DiffOpt.ReverseConstraintDual(), 1.0)
    DiffOpt.reverse_differentiate!(direct)
    @test get_attribute(direct, DiffOpt.ReverseObjectiveFunction()) isa
          JuMP.AbstractJuMPScalar

    # Vector-form ForwardConstraintFunction via attributes on a conic model.
    cmodel = DiffOpt.conic_diff_model(SCS.Optimizer)
    set_silent(cmodel)
    @variable(cmodel, xv)
    @variable(cmodel, yv)
    @constraint(cmodel, c_eq, xv + yv == 1)
    @constraint(cmodel, c_nn, [1.0 * yv, 1.0 * xv] in MOI.Nonnegatives(2))
    @objective(cmodel, Min, 1.0 * xv)
    optimize!(cmodel)
    set_attribute(c_nn, DiffOpt.ForwardConstraintFunction(), [0.0, 0.0])
    set_attribute(c_eq, DiffOpt.ForwardConstraintFunction(), 1.0)
    DiffOpt.forward_differentiate!(cmodel)
    @test get_attribute(yv, DiffOpt.ForwardVariablePrimal()) ≈ -1.0 atol = ATOL

    # ForwardConstraintSet on a ParameterRef (the underlying mechanism behind
    # ForwardParameterValue).
    model2 = DiffOpt.diff_model(HiGHS.Optimizer)
    set_silent(model2)
    @variable(model2, xx)
    @variable(model2, pp in Parameter(1.0))
    @constraint(model2, xx == 2 * pp)
    @objective(model2, Min, xx)
    optimize!(model2)
    set_attribute(
        ParameterRef(pp),
        DiffOpt.ForwardConstraintSet(),
        MOI.Parameter(1.0),
    )
    DiffOpt.forward_differentiate!(model2)
    @test get_attribute(xx, DiffOpt.ForwardVariablePrimal()) ≈ 2.0 atol = ATOL
    return
end

end # module

TestJuMPWrapper.runtests()
