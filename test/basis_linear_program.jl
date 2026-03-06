# Copyright (c) 2020: Akshay Sharma and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module TestBasisLinearProgram

include("BasisLinearProgramTests.jl")

using Test
using JuMP
import DiffOpt
import HiGHS
import MathOptInterface as MOI

const ATOL = 1e-6
const RTOL = 1e-6

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

"""
Helper to create a basis_diff_model with HiGHS in silent mode.
Auto-detects DirectModel since HiGHS supports _basis_solve/_basis_transpose_solve.
"""
function _create_model()
    model = DiffOpt.basis_diff_model(HiGHS.Optimizer)
    set_silent(model)
    return model
end

"""
Helper to create a GeneralModel-based diff model with HiGHS.
"""
function _create_general_model()
    inner = DiffOpt.diff_optimizer(HiGHS.Optimizer)
    MOI.set(
        inner,
        DiffOpt.ModelConstructor(),
        DiffOpt.BasisLinearProgram.GeneralModel,
    )
    model = direct_model(inner)
    set_silent(model)
    return model
end

const BOTH_MODELS = (_create_model, _create_general_model)

# ============================================================================
# Shared tests (forward, reverse, consistency, etc.)
# ============================================================================

function test_common()
    return BasisLinearProgramTests.run_common_tests(BOTH_MODELS)
end

# ============================================================================
# DirectModel-specific tests
# ============================================================================

function test_direct_auto_detection()
    model = _create_model()
    @variable(model, x >= 0)
    @variable(model, b in Parameter(5.0))
    @constraint(model, c1, x <= b)
    @objective(model, Min, -x)
    optimize!(model)
    DiffOpt.set_forward_parameter(model, b, 1.0)
    DiffOpt.forward_differentiate!(model)
    # Verify DirectModel path: diff is the optimizer chain (identity index_map)
    backend_model = backend(model)
    @test MOI.get(backend_model, DiffOpt.ModelConstructor()) ===
          DiffOpt.BasisLinearProgram.DirectModel
    # diff should be the optimizer chain root, not a separate model
    diff = backend_model.diff
    @test diff === backend_model.optimizer
end

function test_direct_differentiate_time_sec()
    model = _create_model()
    @variable(model, x >= 0)
    @variable(model, b in Parameter(5.0))
    @constraint(model, c1, x <= b)
    @objective(model, Min, -x)
    optimize!(model)
    DiffOpt.set_forward_parameter(model, b, 1.0)
    DiffOpt.forward_differentiate!(model)
    t = MOI.get(model, DiffOpt.DifferentiateTimeSec())
    @test t >= 0.0
    @test isfinite(t)
end

function test_forward_simple_lp_exact_values()
    # min -x-2y, s.t. x+y <= 5, x,y >= 0
    # Optimal: x=0, y=5 (y basic). Perturb b by 1 → dx=0, dy=1
    model = _create_model()
    @variable(model, x >= 0)
    @variable(model, y >= 0)
    @variable(model, b in Parameter(5.0))
    @constraint(model, c, x + y <= b)
    @objective(model, Min, -x - 2y)
    optimize!(model)

    DiffOpt.set_forward_parameter(model, b, 1.0)
    DiffOpt.forward_differentiate!(model)

    @test DiffOpt.get_forward_variable(model, x) ≈ 0.0 atol = ATOL
    @test DiffOpt.get_forward_variable(model, y) ≈ 1.0 atol = ATOL
end

function test_reverse_simple_lp_mixed_seeds()
    # min -x-2y, s.t. x+y <= 5, x,y >= 0
    # Optimal: x=0, y=5. Basic: y. B=[1], B⁻ᵀ=[1].
    # Seeds: dL/dx=1, dL/dy=0.5. Only basic y contributes: db = 0.5
    model = _create_model()
    @variable(model, x >= 0)
    @variable(model, y >= 0)
    @variable(model, b in Parameter(5.0))
    @constraint(model, c, x + y <= b)
    @objective(model, Min, -x - 2y)
    optimize!(model)

    DiffOpt.set_reverse_variable(model, x, 1.0)
    DiffOpt.set_reverse_variable(model, y, 0.5)
    DiffOpt.reverse_differentiate!(model)

    @test DiffOpt.get_reverse_parameter(model, b) ≈ 0.5 atol = ATOL
end

function test_allow_direct_false_forces_general_model()
    model = DiffOpt.basis_diff_model(HiGHS.Optimizer; _allow_direct = false)
    set_silent(model)
    backend_model = backend(model)
    @test MOI.get(backend_model, DiffOpt.ModelConstructor()) ===
          DiffOpt.BasisLinearProgram.GeneralModel

    # Should still produce correct results
    @variable(model, x >= 0)
    @variable(model, b in Parameter(5.0))
    @constraint(model, c1, x <= b)
    @objective(model, Min, -x)
    optimize!(model)

    DiffOpt.set_forward_parameter(model, b, 1.0)
    DiffOpt.forward_differentiate!(model)
    @test DiffOpt.get_forward_variable(model, x) ≈ 1.0 atol = ATOL
end

function test_supports_basis_status_attributes()
    # GeneralModel returns true
    gm = DiffOpt.BasisLinearProgram.GeneralModel()
    @test MOI.supports(
        gm,
        DiffOpt._InputConstraintBasisStatus(),
        MOI.ConstraintIndex{
            MOI.ScalarAffineFunction{Float64},
            MOI.LessThan{Float64},
        },
    )
    @test MOI.supports(
        gm,
        DiffOpt._InputVariableBasisStatus(),
        MOI.VariableIndex,
    )

    # Non-BasisLP models return false (via AbstractModel default)
    qm = DiffOpt.QuadraticProgram.Model()
    @test !MOI.supports(
        qm,
        DiffOpt._InputConstraintBasisStatus(),
        MOI.ConstraintIndex{
            MOI.ScalarAffineFunction{Float64},
            MOI.LessThan{Float64},
        },
    )
    @test !MOI.supports(
        qm,
        DiffOpt._InputVariableBasisStatus(),
        MOI.VariableIndex,
    )
end

function test_general_model_unsupported_attributes_direct()
    # GeneralModel throws on ReverseObjectiveSensitivity and
    # ForwardObjectiveSensitivity when accessed directly (not via JuMP API)
    gm = DiffOpt.BasisLinearProgram.GeneralModel()
    @test_throws MOI.UnsupportedAttribute MOI.set(
        gm,
        DiffOpt.ReverseObjectiveSensitivity(),
        1.0,
    )
    @test_throws MOI.UnsupportedAttribute MOI.get(
        gm,
        DiffOpt.ForwardObjectiveSensitivity(),
    )
end

function test_direct_model_objective_sensitivity_via_chain()
    # DirectModel doesn't support ForwardObjectiveSensitivity on the
    # SensitivityCache itself, but DiffOpt.Optimizer computes it at a
    # higher level via c' * dx. Verify it works end-to-end.
    model = _create_model()
    @variable(model, x >= 0)
    @variable(model, b in Parameter(5.0))
    @constraint(model, c1, x <= b)
    @objective(model, Min, -x)
    optimize!(model)

    DiffOpt.set_forward_parameter(model, b, 1.0)
    DiffOpt.forward_differentiate!(model)
    # dx=1, c=-1, so dobj = -1
    @test DiffOpt.get_forward_objective(model) ≈ -1.0 atol = ATOL
end

function test_copy_basis_skips_non_basis_models()
    # QuadraticProgram.Model defines MOI.supports → false for basis attributes,
    # so _copy_basis should skip the loop. Test that QP still works:
    model = DiffOpt.quadratic_diff_model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, x >= 0)
    @variable(model, b in Parameter(5.0))
    @constraint(model, c1, x <= b)
    @objective(model, Min, -x)
    optimize!(model)
    DiffOpt.set_forward_parameter(model, b, 1.0)
    DiffOpt.forward_differentiate!(model)
    @test DiffOpt.get_forward_variable(model, x) ≈ 1.0 atol = ATOL
end

function test_general_model_differentiate_time_sec()
    model = DiffOpt.basis_diff_model(HiGHS.Optimizer; _allow_direct = false)
    set_silent(model)
    @variable(model, x >= 0)
    @variable(model, b in Parameter(5.0))
    @constraint(model, c1, x <= b)
    @objective(model, Min, -x)
    optimize!(model)
    DiffOpt.set_forward_parameter(model, b, 1.0)
    DiffOpt.forward_differentiate!(model)
    t = MOI.get(model, DiffOpt.DifferentiateTimeSec())
    @test t >= 0.0
    @test isfinite(t)
end

function test_supports_basis_solve_forwarding()
    # Direct HiGHS
    @test DiffOpt.BasisLinearProgram._supports_basis_solve(HiGHS.Optimizer())

    # Through SensitivityCache
    sc = DiffOpt._SensitivityCache.Optimizer(HiGHS.Optimizer())
    @test DiffOpt.BasisLinearProgram._supports_basis_solve(sc)

    # Default fallback (non-HiGHS optimizer) — uses ::Any method
    @test !DiffOpt.BasisLinearProgram._supports_basis_solve(nothing)

    # Through CachingOptimizer layer
    inner_co = MOI.Utilities.CachingOptimizer(
        MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
        HiGHS.Optimizer(),
    )
    @test DiffOpt.BasisLinearProgram._supports_basis_solve(inner_co)

    # Through LazyBridgeOptimizer layer
    bridge_opt = MOI.Bridges.full_bridge_optimizer(HiGHS.Optimizer(), Float64)
    @test DiffOpt.BasisLinearProgram._supports_basis_solve(bridge_opt)
end

function test_supports_basis_solve_poi_forwarding()
    inner = HiGHS.Optimizer()
    poi_opt = DiffOpt.POI.Optimizer(inner)
    @test DiffOpt.BasisLinearProgram._supports_basis_solve(poi_opt)
end

function test_general_model_rejects_quadratic_objective()
    gm = DiffOpt.BasisLinearProgram.GeneralModel()
    @test !MOI.supports(
        gm,
        MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(),
    )
end

function test_basis_status_noop_on_non_basis_model()
    qm = DiffOpt.QuadraticProgram.Model()
    vi = MOI.VariableIndex(1)
    ci = MOI.ConstraintIndex{
        MOI.ScalarAffineFunction{Float64},
        MOI.LessThan{Float64},
    }(
        1,
    )
    # Should be no-ops (return nothing)
    @test MOI.set(qm, DiffOpt._InputVariableBasisStatus(), vi, MOI.BASIC) ===
          nothing
    @test MOI.set(qm, DiffOpt._InputConstraintBasisStatus(), ci, MOI.BASIC) ===
          nothing
end

function test_general_model_reverse_objective_function()
    gm = DiffOpt.BasisLinearProgram.GeneralModel()
    rof = MOI.get(gm, DiffOpt.ReverseObjectiveFunction())
    @test isa(rof, DiffOpt.VectorScalarAffineFunction)
    @test rof.constant ≈ 0.0 atol = ATOL
    @test isempty(rof.terms)
end

function test_reverse_with_parameter_in_dx()
    model = DiffOpt.basis_diff_model(HiGHS.Optimizer; _allow_direct = false)
    set_silent(model)
    @variable(model, x >= 0)
    @variable(model, b in Parameter(5.0))
    @constraint(model, c1, x <= b)
    @objective(model, Min, -x)
    optimize!(model)

    DiffOpt.set_reverse_variable(model, x, 1.0)
    DiffOpt.reverse_differentiate!(model)
    db = DiffOpt.get_reverse_parameter(model, b)
    @test db ≈ 1.0 atol = ATOL
end

function test_validate_basis_no_status_set()
    BLP = DiffOpt.BasisLinearProgram
    gm = BLP.GeneralModel()
    x = MOI.add_variable(gm)
    MOI.add_constraint(gm, 1.0 * x, MOI.LessThan(5.0))
    BLP._build_A!(gm)
    @test_throws(
        ErrorException(
            "GeneralModel: basis status not set. " *
            "Ensure _copy_basis was called after copy_to.",
        ),
        BLP._validate_basis!(gm),
    )
end

function test_validate_basis_size_mismatch()
    BLP = DiffOpt.BasisLinearProgram
    gm = BLP.GeneralModel()
    x = MOI.add_variable(gm)
    y = MOI.add_variable(gm)
    MOI.add_constraint(gm, 1.0 * x + 1.0 * y, MOI.LessThan(5.0))
    BLP._build_A!(gm)
    gm.var_basis_status[x] = MOI.BASIC
    gm.var_basis_status[y] = MOI.BASIC
    @test_throws(
        ErrorException(
            "Basis size mismatch: expected 1 basic variables, " *
            "got 2 structural + 0 slack = 2",
        ),
        BLP._validate_basis!(gm),
    )
end

struct _NotAnOptimizer end

function test_supports_basis_solve_any_fallback()
    @test !DiffOpt.BasisLinearProgram._supports_basis_solve(_NotAnOptimizer())
end

function test_slack_coefficient()
    BLP = DiffOpt.BasisLinearProgram
    @test BLP._slack_coefficient(MOI.LessThan{Float64}) == 1.0
    @test BLP._slack_coefficient(MOI.GreaterThan{Float64}) == -1.0
    @test_throws(
        ErrorException("EqualTo constraint should not have a basic slack"),
        BLP._slack_coefficient(MOI.EqualTo{Float64}),
    )
end

end # module

TestBasisLinearProgram.runtests()
