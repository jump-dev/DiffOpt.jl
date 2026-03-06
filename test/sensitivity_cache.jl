# Copyright (c) 2020: Akshay Sharma and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module TestSensitivityCache

using Test
using JuMP
import DiffOpt
import HiGHS
import MathOptInterface as MOI

const SC = DiffOpt._SensitivityCache
const ATOL = 1e-6

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

# ============================================================================
# Core lifecycle: constructor, empty!, is_empty
# ============================================================================

function test_constructor()
    opt = SC.Optimizer(HiGHS.Optimizer())
    @test MOI.is_empty(opt)
    @test isnan(opt.diff_time)
    @test opt.forw_dx === nothing
    @test opt.back_db === nothing
end

function test_empty_resets_state()
    opt = SC.Optimizer(HiGHS.Optimizer())
    # Populate some state
    vi = MOI.VariableIndex(1)
    opt.dx[vi] = 1.0
    opt.forw_dx = Dict{MOI.VariableIndex,Float64}(vi => 2.0)
    opt.back_db = Dict{MOI.ConstraintIndex,Float64}()
    opt.diff_time = 0.5

    MOI.empty!(opt)
    @test MOI.is_empty(opt)
    @test isempty(opt.dx)
    @test opt.forw_dx === nothing
    @test opt.back_db === nothing
    @test isnan(opt.diff_time)
end

# ============================================================================
# MOI passthrough: model attributes
# ============================================================================

function test_model_attribute_set()
    opt = SC.Optimizer(HiGHS.Optimizer())
    # MOI.Name is an AbstractModelAttribute
    MOI.set(opt, MOI.Name(), "test_model")
    @test MOI.get(opt, MOI.Name()) == "test_model"
    # ObjectiveSense is also an AbstractModelAttribute
    MOI.set(opt, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    @test MOI.get(opt, MOI.ObjectiveSense()) == MOI.MIN_SENSE
end

function test_model_attribute_supports()
    opt = SC.Optimizer(HiGHS.Optimizer())
    @test MOI.supports(opt, MOI.Name())
end

# ============================================================================
# MOI passthrough: optimizer attributes
# ============================================================================

function test_optimizer_attribute_get_set()
    opt = SC.Optimizer(HiGHS.Optimizer())
    MOI.set(opt, MOI.Silent(), true)
    @test MOI.get(opt, MOI.Silent()) == true
    MOI.set(opt, MOI.Silent(), false)
    @test MOI.get(opt, MOI.Silent()) == false
end

function test_optimizer_attribute_supports()
    opt = SC.Optimizer(HiGHS.Optimizer())
    @test MOI.supports(opt, MOI.Silent())
end

# ============================================================================
# MOI passthrough: variable attributes
# ============================================================================

function test_variable_attribute_passthrough()
    opt = SC.Optimizer(HiGHS.Optimizer())
    MOI.set(opt, MOI.Silent(), true)

    # Build a small model through the wrapper
    src = MOI.Utilities.Model{Float64}()
    x, _ = MOI.add_constrained_variable(src, MOI.GreaterThan(0.0))
    MOI.set(
        src,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        1.0 * x,
    )
    MOI.set(src, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.add_constraint(src, 1.0 * x, MOI.LessThan(5.0))
    MOI.copy_to(opt, src)
    MOI.optimize!(opt)

    # Variable attributes via passthrough
    vi = MOI.VariableIndex(1)
    @test MOI.is_valid(opt, vi)
    val = MOI.get(opt, MOI.VariablePrimal(), vi)
    @test val ≈ 0.0 atol = ATOL

    # supports passthrough
    @test MOI.supports(
        opt,
        MOI.VariablePrimalStart(),
        MOI.VariableIndex,
    )

    # set passthrough
    MOI.set(opt, MOI.VariablePrimalStart(), vi, 1.0)
end

# ============================================================================
# MOI passthrough: constraint attributes
# ============================================================================

function test_constraint_attribute_passthrough()
    opt = SC.Optimizer(HiGHS.Optimizer())
    MOI.set(opt, MOI.Silent(), true)

    src = MOI.Utilities.Model{Float64}()
    x, _ = MOI.add_constrained_variable(src, MOI.GreaterThan(0.0))
    MOI.set(
        src,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        1.0 * x,
    )
    MOI.set(src, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.add_constraint(src, 1.0 * x, MOI.LessThan(5.0))
    MOI.copy_to(opt, src)
    MOI.optimize!(opt)

    # Constraint attribute get
    le_ci = MOI.ConstraintIndex{
        MOI.ScalarAffineFunction{Float64},
        MOI.LessThan{Float64},
    }(1)
    @test MOI.is_valid(opt, le_ci)
    dual_val = MOI.get(opt, MOI.ConstraintDual(), le_ci)
    @test isa(dual_val, Float64)

    # Constraint attribute supports (use ConstraintName which HiGHS supports)
    @test MOI.supports(
        opt,
        MOI.ConstraintName(),
        MOI.ConstraintIndex{
            MOI.ScalarAffineFunction{Float64},
            MOI.LessThan{Float64},
        },
    )

    # Constraint attribute set
    MOI.set(opt, MOI.ConstraintName(), le_ci, "my_constraint")
    @test MOI.get(opt, MOI.ConstraintName(), le_ci) == "my_constraint"
end

# ============================================================================
# MOI passthrough: query methods
# ============================================================================

function test_list_queries()
    opt = SC.Optimizer(HiGHS.Optimizer())
    MOI.set(opt, MOI.Silent(), true)

    src = MOI.Utilities.Model{Float64}()
    x, _ = MOI.add_constrained_variable(src, MOI.GreaterThan(0.0))
    MOI.set(
        src,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        1.0 * x,
    )
    MOI.set(src, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.add_constraint(src, 1.0 * x, MOI.LessThan(5.0))
    MOI.copy_to(opt, src)

    # ListOfVariableIndices
    vis = MOI.get(opt, MOI.ListOfVariableIndices())
    @test length(vis) >= 1

    # ListOfConstraintTypesPresent
    types = MOI.get(opt, MOI.ListOfConstraintTypesPresent())
    @test length(types) >= 1

    # ListOfConstraintIndices
    le_cis = MOI.get(
        opt,
        MOI.ListOfConstraintIndices{
            MOI.ScalarAffineFunction{Float64},
            MOI.LessThan{Float64},
        }(),
    )
    @test length(le_cis) >= 1

    # NumberOfVariables
    @test MOI.get(opt, MOI.NumberOfVariables()) >= 1

    # is_valid for CI
    @test MOI.is_valid(opt, first(le_cis))

    # supports_constraint
    @test MOI.supports_constraint(
        opt,
        MOI.ScalarAffineFunction{Float64},
        MOI.LessThan{Float64},
    )
end

# ============================================================================
# Absorbed DiffOpt attributes (no-ops)
# ============================================================================

function test_absorbed_diffopt_attributes()
    opt = SC.Optimizer(HiGHS.Optimizer())
    # These should be no-ops
    @test MOI.set(opt, DiffOpt.NonLinearKKTJacobianFactorization(), :lu) ===
          nothing
    @test MOI.set(opt, DiffOpt.AllowObjectiveAndSolutionInput(), true) ===
          nothing
end

# ============================================================================
# BasisLinearProgram extension point forwarding
# ============================================================================

function test_supports_basis_solve_forwarding()
    opt = SC.Optimizer(HiGHS.Optimizer())
    @test DiffOpt.BasisLinearProgram._supports_basis_solve(opt)
end

# ============================================================================
# DiffOpt attribute setters (input)
# ============================================================================

function test_forward_constraint_function_setter()
    opt = SC.Optimizer(HiGHS.Optimizer())
    ci = MOI.ConstraintIndex{
        MOI.ScalarAffineFunction{Float64},
        MOI.LessThan{Float64},
    }(1)
    func = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm{Float64}[], -1.0)
    MOI.set(opt, DiffOpt.ForwardConstraintFunction(), ci, func)
    @test haskey(opt.scalar_constraints, ci)
end

function test_reverse_variable_primal_setter()
    opt = SC.Optimizer(HiGHS.Optimizer())
    vi = MOI.VariableIndex(1)
    MOI.set(opt, DiffOpt.ReverseVariablePrimal(), vi, 3.0)
    @test opt.dx[vi] == 3.0
end

# ============================================================================
# DiffOpt attribute getters (output)
# ============================================================================

function test_forward_variable_primal_getter_before_differentiate()
    opt = SC.Optimizer(HiGHS.Optimizer())
    vi = MOI.VariableIndex(1)
    # Before differentiate, forw_dx is nothing → returns 0.0
    @test MOI.get(opt, DiffOpt.ForwardVariablePrimal(), vi) == 0.0
end

function test_differentiate_time_sec()
    opt = SC.Optimizer(HiGHS.Optimizer())
    @test isnan(MOI.get(opt, DiffOpt.DifferentiateTimeSec()))
end

function test_reverse_objective_function()
    opt = SC.Optimizer(HiGHS.Optimizer())
    # Direct call on SC.Optimizer (no model built)
    rof = MOI.get(opt, DiffOpt.ReverseObjectiveFunction())
    @test isa(rof, MOI.ScalarAffineFunction{Float64})
    @test MOI.constant(rof) == 0.0
    @test isempty(rof.terms)

    # Also test through a model that actually solved + differentiated
    MOI.set(opt, MOI.Silent(), true)
    src = MOI.Utilities.Model{Float64}()
    x, _ = MOI.add_constrained_variable(src, MOI.GreaterThan(0.0))
    MOI.set(
        src,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        -1.0 * x,
    )
    MOI.set(src, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.add_constraint(src, 1.0 * x, MOI.LessThan(5.0))
    MOI.copy_to(opt, src)
    MOI.optimize!(opt)

    vi = MOI.VariableIndex(1)
    MOI.set(opt, DiffOpt.ReverseVariablePrimal(), vi, 1.0)
    DiffOpt.reverse_differentiate!(opt)

    # ReverseObjectiveFunction always returns zero for SensitivityCache
    rof2 = MOI.get(opt, DiffOpt.ReverseObjectiveFunction())
    @test isa(rof2, MOI.ScalarAffineFunction{Float64})
    @test MOI.constant(rof2) == 0.0
end

function test_forward_objective_sensitivity_throws()
    opt = SC.Optimizer(HiGHS.Optimizer())
    @test_throws MOI.UnsupportedAttribute MOI.get(
        opt,
        DiffOpt.ForwardObjectiveSensitivity(),
    )
end

function test_get_db_before_differentiate()
    opt = SC.Optimizer(HiGHS.Optimizer())
    ci = MOI.ConstraintIndex{
        MOI.ScalarAffineFunction{Float64},
        MOI.LessThan{Float64},
    }(1)
    # back_db is nothing → returns 0.0
    @test DiffOpt._get_db(opt, ci) == 0.0
end

function test_reverse_constraint_function()
    opt = SC.Optimizer(HiGHS.Optimizer())
    ci = MOI.ConstraintIndex{
        MOI.ScalarAffineFunction{Float64},
        MOI.LessThan{Float64},
    }(1)
    rcf = MOI.get(opt, DiffOpt.ReverseConstraintFunction(), ci)
    @test isa(rcf, MOI.ScalarAffineFunction{Float64})
    @test MOI.constant(rcf) == 0.0
end

# ============================================================================
# empty_input_sensitivities!
# ============================================================================

function test_empty_input_sensitivities()
    opt = SC.Optimizer(HiGHS.Optimizer())
    vi = MOI.VariableIndex(1)
    ci = MOI.ConstraintIndex{
        MOI.ScalarAffineFunction{Float64},
        MOI.LessThan{Float64},
    }(1)

    # Set some inputs
    MOI.set(opt, DiffOpt.ReverseVariablePrimal(), vi, 1.0)
    func = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm{Float64}[], -1.0)
    MOI.set(opt, DiffOpt.ForwardConstraintFunction(), ci, func)
    opt.forw_dx = Dict{MOI.VariableIndex,Float64}(vi => 2.0)
    opt.back_db = Dict{MOI.ConstraintIndex,Float64}()

    DiffOpt.empty_input_sensitivities!(opt)
    @test isempty(opt.dx)
    @test opt.forw_dx === nothing
    @test opt.back_db === nothing
end

# ============================================================================
# Forward/reverse differentiate error path (dA perturbation)
# ============================================================================

function test_passthrough_via_direct_dispatch()
    # Verify that SC.Optimizer passthrough methods are directly called
    # by constructing a model, solving, and querying attributes.
    opt = SC.Optimizer(HiGHS.Optimizer())
    MOI.set(opt, MOI.Silent(), true)

    src = MOI.Utilities.Model{Float64}()
    x, _ = MOI.add_constrained_variable(src, MOI.GreaterThan(0.0))
    MOI.set(
        src,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        1.0 * x,
    )
    MOI.set(src, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.add_constraint(src, 1.0 * x, MOI.LessThan(5.0))
    MOI.copy_to(opt, src)
    MOI.optimize!(opt)

    vi = MOI.VariableIndex(1)
    le_ci = MOI.ConstraintIndex{
        MOI.ScalarAffineFunction{Float64},
        MOI.LessThan{Float64},
    }(1)

    # Variable attribute passthrough: is_valid
    @test MOI.is_valid(opt, vi)
    # Constraint attribute passthrough: is_valid
    @test MOI.is_valid(opt, le_ci)

    # Variable attribute: supports + set (VariablePrimalStart)
    @test MOI.supports(
        opt,
        MOI.VariablePrimalStart(),
        MOI.VariableIndex,
    )
    MOI.set(opt, MOI.VariablePrimalStart(), vi, 2.0)

    # Constraint attribute: supports + set (ConstraintName)
    @test MOI.supports(
        opt,
        MOI.ConstraintName(),
        MOI.ConstraintIndex{
            MOI.ScalarAffineFunction{Float64},
            MOI.LessThan{Float64},
        },
    )
    MOI.set(opt, MOI.ConstraintName(), le_ci, "test_constraint")
    @test MOI.get(opt, MOI.ConstraintName(), le_ci) == "test_constraint"

    # ListOfConstraintIndices passthrough
    le_cis = MOI.get(
        opt,
        MOI.ListOfConstraintIndices{
            MOI.ScalarAffineFunction{Float64},
            MOI.LessThan{Float64},
        }(),
    )
    @test length(le_cis) >= 1
end

function test_forward_dA_error_on_sensitivity_cache()
    model = DiffOpt.basis_diff_model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, x >= 0)
    @constraint(model, c1, x <= 5.0)
    @objective(model, Min, -x)
    optimize!(model)

    # Coefficient perturbation → error
    MOI.set(
        model,
        DiffOpt.ForwardConstraintFunction(),
        c1,
        1.0 * x - 1.0,
    )
    @test_throws ErrorException DiffOpt.forward_differentiate!(model)
end

end # module

TestSensitivityCache.runtests()
