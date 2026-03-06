# Copyright (c) 2020: Akshay Sharma and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

"""
Shared tests for BasisLinearProgram that can be run with any solver.

Each `_test_*` function takes `both_models` — a tuple of model constructor
functions `(create_direct_model, create_general_model)`.

Use `run_common_tests(both_models)` to run all shared tests inside `@testset`s.
"""
module BasisLinearProgramTests

using Test
using JuMP
import DiffOpt
import MathOptInterface as MOI

const ATOL = 1e-6

# ============================================================================
# Forward differentiation
# ============================================================================

function _test_forward_simple_lp(both_models)
    # min -x - 2y, s.t. x + y <= b, x, y >= 0, b = 5
    # Optimal: x=0, y=5. Perturb b by 1 → dy = 1
    for create_fn in both_models
        model = create_fn()
        @variable(model, x >= 0)
        @variable(model, y >= 0)
        @variable(model, b in Parameter(5.0))
        @constraint(model, c1, x + y <= b)
        @objective(model, Min, -x - 2y)
        optimize!(model)

        @test termination_status(model) == MOI.OPTIMAL
        @test value(x) ≈ 0.0 atol = ATOL
        @test value(y) ≈ 5.0 atol = ATOL

        DiffOpt.set_forward_parameter(model, b, 1.0)
        DiffOpt.forward_differentiate!(model)

        dx = DiffOpt.get_forward_variable(model, x)
        dy = DiffOpt.get_forward_variable(model, y)
        @test dx + dy ≈ 1.0 atol = ATOL
    end
end

function _test_forward_equality_constraint(both_models)
    # min x + y, s.t. x + y == b, x, y >= 0, b = 5
    for create_fn in both_models
        model = create_fn()
        @variable(model, x >= 0)
        @variable(model, y >= 0)
        @variable(model, b in Parameter(5.0))
        @constraint(model, c_eq, x + y == b)
        @objective(model, Min, x + y)
        optimize!(model)

        @test termination_status(model) == MOI.OPTIMAL
        @test objective_value(model) ≈ 5.0 atol = ATOL

        DiffOpt.set_forward_parameter(model, b, 1.0)
        DiffOpt.forward_differentiate!(model)

        dx = DiffOpt.get_forward_variable(model, x)
        dy = DiffOpt.get_forward_variable(model, y)
        @test dx + dy ≈ 1.0 atol = ATOL
    end
end

function _test_forward_two_constraints(both_models)
    # min -x, s.t. x <= b1, x <= b2, x >= 0
    # b1=5 (binding), b2=10 (non-binding)
    for create_fn in both_models
        model = create_fn()
        @variable(model, x >= 0)
        @variable(model, b1 in Parameter(5.0))
        @variable(model, b2 in Parameter(10.0))
        @constraint(model, c1, x <= b1)
        @constraint(model, c2, x <= b2)
        @objective(model, Min, -x)
        optimize!(model)

        @test value(x) ≈ 5.0 atol = ATOL

        # Perturb binding b1
        DiffOpt.set_forward_parameter(model, b1, 1.0)
        DiffOpt.forward_differentiate!(model)
        @test DiffOpt.get_forward_variable(model, x) ≈ 1.0 atol = ATOL

        # Perturb non-binding b2
        DiffOpt.empty_input_sensitivities!(model)
        DiffOpt.set_forward_parameter(model, b2, 1.0)
        DiffOpt.forward_differentiate!(model)
        @test DiffOpt.get_forward_variable(model, x) ≈ 0.0 atol = ATOL
    end
end

function _test_forward_multiple_variables(both_models)
    # min -2x - 3y, s.t. x+y <= b1, x <= b2, y <= b3, x,y >= 0
    # b1=10, b2=7, b3=6. Optimal: x=4, y=6
    for create_fn in both_models
        model = create_fn()
        @variable(model, x >= 0)
        @variable(model, y >= 0)
        @variable(model, b1 in Parameter(10.0))
        @variable(model, b2 in Parameter(7.0))
        @variable(model, b3 in Parameter(6.0))
        @constraint(model, c1, x + y <= b1)
        @constraint(model, c2, x <= b2)
        @constraint(model, c3, y <= b3)
        @objective(model, Min, -2x - 3y)
        optimize!(model)

        @test value(x) ≈ 4.0 atol = ATOL
        @test value(y) ≈ 6.0 atol = ATOL

        # Perturb b1: x+y <= 10+δ → dx=1, dy=0
        DiffOpt.set_forward_parameter(model, b1, 1.0)
        DiffOpt.forward_differentiate!(model)

        @test DiffOpt.get_forward_variable(model, x) ≈ 1.0 atol = ATOL
        @test DiffOpt.get_forward_variable(model, y) ≈ 0.0 atol = ATOL
    end
end

function _test_forward_greaterthan(both_models)
    # min 3x+2y, s.t. x+y >= b1, 2x+y >= b2, x,y >= 0
    # b1=5, b2=8. Optimal: x=3, y=2 (both binding)
    # B = [1 1; 2 1], B⁻¹ = [-1 1; 2 -1]
    for create_fn in both_models
        model = create_fn()
        @variable(model, x >= 0)
        @variable(model, y >= 0)
        @variable(model, b1 in Parameter(5.0))
        @variable(model, b2 in Parameter(8.0))
        @constraint(model, c1, x + y >= b1)
        @constraint(model, c2, 2x + y >= b2)
        @objective(model, Min, 3x + 2y)
        optimize!(model)

        @test value(x) ≈ 3.0 atol = ATOL
        @test value(y) ≈ 2.0 atol = ATOL

        # Perturb b1: db=[1;0], dx = B⁻¹[1;0] = [-1;2]
        DiffOpt.set_forward_parameter(model, b1, 1.0)
        DiffOpt.forward_differentiate!(model)

        @test DiffOpt.get_forward_variable(model, x) ≈ -1.0 atol = ATOL
        @test DiffOpt.get_forward_variable(model, y) ≈ 2.0 atol = ATOL

        # Perturb b2: db=[0;1], dx = B⁻¹[0;1] = [1;-1]
        DiffOpt.empty_input_sensitivities!(model)
        DiffOpt.set_forward_parameter(model, b2, 1.0)
        DiffOpt.forward_differentiate!(model)

        @test DiffOpt.get_forward_variable(model, x) ≈ 1.0 atol = ATOL
        @test DiffOpt.get_forward_variable(model, y) ≈ -1.0 atol = ATOL
    end
end

function _test_forward_mixed_constraints(both_models)
    # min -x-y-z, s.t. x+y <= b_le, y+z >= b_ge, x+z == b_eq, x,y,z >= 0
    # b_le=8, b_ge=3, b_eq=5. Optimal: x=0, y=8, z=5
    for create_fn in both_models
        model = create_fn()
        @variable(model, x >= 0)
        @variable(model, y >= 0)
        @variable(model, z >= 0)
        @variable(model, b_le in Parameter(8.0))
        @variable(model, b_ge in Parameter(3.0))
        @variable(model, b_eq in Parameter(5.0))
        @constraint(model, c_le, x + y <= b_le)
        @constraint(model, c_ge, y + z >= b_ge)
        @constraint(model, c_eq, x + z == b_eq)
        @objective(model, Min, -x - y - z)
        optimize!(model)

        @test value(x) ≈ 0.0 atol = ATOL
        @test value(y) ≈ 8.0 atol = ATOL
        @test value(z) ≈ 5.0 atol = ATOL

        # Perturb b_le: x+y <= 8+δ
        DiffOpt.set_forward_parameter(model, b_le, 1.0)
        DiffOpt.forward_differentiate!(model)

        dx = DiffOpt.get_forward_variable(model, x)
        dy = DiffOpt.get_forward_variable(model, y)
        dz = DiffOpt.get_forward_variable(model, z)
        @test dx + dy ≈ 1.0 atol = ATOL
        @test dx + dz ≈ 0.0 atol = ATOL

        # Perturb non-binding b_ge → no change
        DiffOpt.empty_input_sensitivities!(model)
        DiffOpt.set_forward_parameter(model, b_ge, 1.0)
        DiffOpt.forward_differentiate!(model)

        @test DiffOpt.get_forward_variable(model, x) ≈ 0.0 atol = ATOL
        @test DiffOpt.get_forward_variable(model, y) ≈ 0.0 atol = ATOL
        @test DiffOpt.get_forward_variable(model, z) ≈ 0.0 atol = ATOL

        # Perturb b_eq: x+z == 5+δ
        DiffOpt.empty_input_sensitivities!(model)
        DiffOpt.set_forward_parameter(model, b_eq, 1.0)
        DiffOpt.forward_differentiate!(model)

        dx3 = DiffOpt.get_forward_variable(model, x)
        dz3 = DiffOpt.get_forward_variable(model, z)
        @test dx3 + dz3 ≈ 1.0 atol = ATOL
    end
end

# ============================================================================
# Reverse differentiation
# ============================================================================

function _test_reverse_simple_lp(both_models)
    # min 2x + 3y, s.t. x + y >= b, x, y >= 0, b = 4
    for create_fn in both_models
        model = create_fn()
        @variable(model, x >= 0)
        @variable(model, y >= 0)
        @variable(model, b in Parameter(4.0))
        @constraint(model, c1, x + y >= b)
        @objective(model, Min, 2x + 3y)
        optimize!(model)

        @test value(x) ≈ 4.0 atol = ATOL
        @test value(y) ≈ 0.0 atol = ATOL

        DiffOpt.set_reverse_variable(model, x, 1.0)
        DiffOpt.set_reverse_variable(model, y, 0.0)
        DiffOpt.reverse_differentiate!(model)

        db = DiffOpt.get_reverse_parameter(model, b)
        @test db ≈ 1.0 atol = ATOL
    end
end

function _test_reverse_nonbinding_constraint(both_models)
    # min -x, s.t. x <= b1, x <= b2, x >= 0
    # b1=5 (binding), b2=10 (non-binding)
    for create_fn in both_models
        model = create_fn()
        @variable(model, x >= 0)
        @variable(model, b1 in Parameter(5.0))
        @variable(model, b2 in Parameter(10.0))
        @constraint(model, c1, x <= b1)
        @constraint(model, c2, x <= b2)
        @objective(model, Min, -x)
        optimize!(model)

        DiffOpt.set_reverse_variable(model, x, 1.0)
        DiffOpt.reverse_differentiate!(model)

        db1 = DiffOpt.get_reverse_parameter(model, b1)
        @test db1 ≈ 1.0 atol = ATOL

        db2 = DiffOpt.get_reverse_parameter(model, b2)
        @test db2 ≈ 0.0 atol = ATOL
    end
end

# ============================================================================
# Forward/reverse consistency (adjoint identity)
# ============================================================================

function _test_forward_reverse_consistency(both_models)
    # Adjoint identity: ∑ dx_i * dL/dx_i == ∑ db_j * dL/db_j
    for create_fn in both_models
        model = create_fn()
        @variable(model, x >= 0)
        @variable(model, y >= 0)
        @variable(model, b1 in Parameter(10.0))
        @variable(model, b2 in Parameter(12.0))
        @constraint(model, c1, x + 2y <= b1)
        @constraint(model, c2, 3x + y <= b2)
        @objective(model, Min, -x - y)
        optimize!(model)

        # Forward
        db1, db2 = 0.5, 0.3
        DiffOpt.set_forward_parameter(model, b1, db1)
        DiffOpt.set_forward_parameter(model, b2, db2)
        DiffOpt.forward_differentiate!(model)
        dx_f = DiffOpt.get_forward_variable(model, x)
        dy_f = DiffOpt.get_forward_variable(model, y)

        # Reverse
        dl_dx, dl_dy = 1.7, -0.4
        DiffOpt.empty_input_sensitivities!(model)
        DiffOpt.set_reverse_variable(model, x, dl_dx)
        DiffOpt.set_reverse_variable(model, y, dl_dy)
        DiffOpt.reverse_differentiate!(model)

        dL_db1 = DiffOpt.get_reverse_parameter(model, b1)
        dL_db2 = DiffOpt.get_reverse_parameter(model, b2)

        forward_product = dx_f * dl_dx + dy_f * dl_dy
        reverse_product = db1 * dL_db1 + db2 * dL_db2
        @test forward_product ≈ reverse_product atol = ATOL
    end
end

function _test_forward_reverse_consistency_greaterthan(both_models)
    # Adjoint identity with GE constraints
    for create_fn in both_models
        model = create_fn()
        @variable(model, x >= 0)
        @variable(model, y >= 0)
        @variable(model, b1 in Parameter(4.0))
        @variable(model, b2 in Parameter(6.0))
        @constraint(model, c1, x + y >= b1)
        @constraint(model, c2, x + 2y >= b2)
        @objective(model, Min, 2x + y)
        optimize!(model)

        @test value(x) ≈ 0.0 atol = ATOL
        @test value(y) ≈ 4.0 atol = ATOL

        db1, db2 = 0.7, -0.3
        DiffOpt.set_forward_parameter(model, b1, db1)
        DiffOpt.set_forward_parameter(model, b2, db2)
        DiffOpt.forward_differentiate!(model)
        dx_f = DiffOpt.get_forward_variable(model, x)
        dy_f = DiffOpt.get_forward_variable(model, y)

        dl_dx, dl_dy = 1.3, -0.8
        DiffOpt.empty_input_sensitivities!(model)
        DiffOpt.set_reverse_variable(model, x, dl_dx)
        DiffOpt.set_reverse_variable(model, y, dl_dy)
        DiffOpt.reverse_differentiate!(model)

        dL_db1 = DiffOpt.get_reverse_parameter(model, b1)
        dL_db2 = DiffOpt.get_reverse_parameter(model, b2)

        forward_product = dx_f * dl_dx + dy_f * dl_dy
        reverse_product = db1 * dL_db1 + db2 * dL_db2
        @test forward_product ≈ reverse_product atol = ATOL
    end
end

function _test_forward_reverse_consistency_mixed(both_models)
    # Adjoint identity with mixed LE, GE, EQ
    for create_fn in both_models
        model = create_fn()
        @variable(model, x >= 0)
        @variable(model, y >= 0)
        @variable(model, z >= 0)
        @variable(model, b_le in Parameter(6.0))
        @variable(model, b_ge in Parameter(2.0))
        @variable(model, b_eq in Parameter(4.0))
        @constraint(model, c_le, x + y <= b_le)
        @constraint(model, c_ge, y + z >= b_ge)
        @constraint(model, c_eq, x + z == b_eq)
        @objective(model, Min, 3x + 2y + z)
        optimize!(model)

        @test value(x) ≈ 0.0 atol = ATOL
        @test value(y) ≈ 0.0 atol = ATOL
        @test value(z) ≈ 4.0 atol = ATOL

        db_le, db_ge, db_eq = 0.4, 0.2, -0.5
        DiffOpt.set_forward_parameter(model, b_le, db_le)
        DiffOpt.set_forward_parameter(model, b_ge, db_ge)
        DiffOpt.set_forward_parameter(model, b_eq, db_eq)
        DiffOpt.forward_differentiate!(model)
        dx_f = DiffOpt.get_forward_variable(model, x)
        dy_f = DiffOpt.get_forward_variable(model, y)
        dz_f = DiffOpt.get_forward_variable(model, z)

        dl_dx, dl_dy, dl_dz = 1.1, -0.5, 0.9
        DiffOpt.empty_input_sensitivities!(model)
        DiffOpt.set_reverse_variable(model, x, dl_dx)
        DiffOpt.set_reverse_variable(model, y, dl_dy)
        DiffOpt.set_reverse_variable(model, z, dl_dz)
        DiffOpt.reverse_differentiate!(model)

        dL_ble = DiffOpt.get_reverse_parameter(model, b_le)
        dL_bge = DiffOpt.get_reverse_parameter(model, b_ge)
        dL_beq = DiffOpt.get_reverse_parameter(model, b_eq)

        forward_product = dx_f * dl_dx + dy_f * dl_dy + dz_f * dl_dz
        reverse_product =
            db_le * dL_ble + db_ge * dL_bge + db_eq * dL_beq
        @test forward_product ≈ reverse_product atol = ATOL
    end
end

# ============================================================================
# dA error and objective sensitivity
# ============================================================================

function _test_forward_dA_perturbation_errors(both_models)
    # dA perturbation (coefficient changes) should error for BasisLP
    for create_fn in both_models
        model = create_fn()
        @variable(model, x >= 0)
        @constraint(model, c1, x <= 5.0)
        @objective(model, Min, -x)
        optimize!(model)
        MOI.set(
            model,
            DiffOpt.ForwardConstraintFunction(),
            c1,
            1.0 * x - 1.0,
        )
        @test_throws ErrorException DiffOpt.forward_differentiate!(model)
    end
end

function _test_forward_objective_sensitivity(both_models)
    # Forward objective sensitivity: dz*/dp via chain rule
    for create_fn in both_models
        model = create_fn()
        @variable(model, x >= 0)
        @variable(model, y >= 0)
        @variable(model, b in Parameter(5.0))
        @constraint(model, c, x + y <= b)
        @objective(model, Min, -x - 2y)
        optimize!(model)

        @test value(x) ≈ 0.0 atol = ATOL
        @test value(y) ≈ 5.0 atol = ATOL

        DiffOpt.set_forward_parameter(model, b, 1.0)
        DiffOpt.forward_differentiate!(model)

        # Objective = -x - 2y. With b↑1: dy=1, so dz = -2*1 = -2
        dobj = DiffOpt.get_forward_objective(model)
        @test dobj ≈ -2.0 atol = ATOL
    end
end

# ============================================================================
# Empty input handling
# ============================================================================

function _test_forward_empty_inputs(both_models)
    # Forward differentiate with no perturbations set
    for create_fn in both_models
        model = create_fn()
        @variable(model, x >= 0)
        @variable(model, b in Parameter(5.0))
        @constraint(model, c1, x <= b)
        @objective(model, Min, -x)
        optimize!(model)

        DiffOpt.forward_differentiate!(model)
        @test DiffOpt.get_forward_variable(model, x) ≈ 0.0 atol = ATOL
    end
end

function _test_reverse_empty_inputs(both_models)
    # Reverse differentiate with no variable seeds set
    for create_fn in both_models
        model = create_fn()
        @variable(model, x >= 0)
        @variable(model, b in Parameter(5.0))
        @constraint(model, c1, x <= b)
        @objective(model, Min, -x)
        optimize!(model)

        DiffOpt.reverse_differentiate!(model)
        @test DiffOpt.get_reverse_parameter(model, b) ≈ 0.0 atol = ATOL
    end
end

# ============================================================================
# Entry point: run all shared tests
# ============================================================================

function run_common_tests(both_models; excludes::Vector{String} = String[])
    for name in names(@__MODULE__; all = true)
        sname = string(name)
        if startswith(sname, "_test_")
            test_name = sname[2:end]  # strip leading underscore
            if test_name in excludes
                continue
            end
            @testset "$(test_name)" begin
                getfield(@__MODULE__, name)(both_models)
            end
        end
    end
    return
end

end # module
