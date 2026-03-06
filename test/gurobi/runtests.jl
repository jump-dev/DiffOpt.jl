# Copyright (c) 2020: Akshay Sharma and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module TestGurobiExtension

include("../BasisLinearProgramTests.jl")

using Test
using JuMP
import DiffOpt
import Gurobi
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

function _create_model()
    model = DiffOpt.basis_diff_model(Gurobi.Optimizer)
    set_silent(model)
    return model
end

function _create_general_model()
    inner = DiffOpt.diff_optimizer(Gurobi.Optimizer)
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
    BasisLinearProgramTests.run_common_tests(BOTH_MODELS)
end

# ============================================================================
# Extension loading and auto-detection
# ============================================================================

function test_supports_basis_solve()
    @test DiffOpt.BasisLinearProgram._supports_basis_solve(Gurobi.Optimizer())
end

function test_direct_auto_detection()
    model = _create_model()
    @variable(model, x >= 0)
    @variable(model, b in Parameter(5.0))
    @constraint(model, c1, x <= b)
    @objective(model, Min, -x)
    optimize!(model)
    DiffOpt.set_forward_parameter(model, b, 1.0)
    DiffOpt.forward_differentiate!(model)
    bm = backend(model)
    @test MOI.get(bm, DiffOpt.ModelConstructor()) ===
          DiffOpt.BasisLinearProgram.DirectModel
end

function test_allow_direct_false_forces_general_model()
    for create_fn in (_create_general_model,)
        model = create_fn()
        @variable(model, x >= 0)
        @variable(model, b in Parameter(5.0))
        @constraint(model, c1, x <= b)
        @objective(model, Min, -x)
        optimize!(model)
        mc = MOI.get(backend(model), DiffOpt.ModelConstructor())
        @test mc === DiffOpt.BasisLinearProgram.GeneralModel
    end
end

end # module

TestGurobiExtension.runtests()
