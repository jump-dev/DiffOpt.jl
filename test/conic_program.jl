# Copyright (c) 2020: Akshay Sharma and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module TestConicProgram

using Test
import DiffOpt
import Ipopt
import LinearAlgebra
import MathOptInterface as MOI
import SCS
using JuMP

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

function _test_simple_socp(eq_vec::Bool)
    # referred from _soc2test, https://github.com/jump-dev/MathOptInterface.jl/blob/master/src/Test/contconic.jl#L1355
    # find reference diffcp python program here: https://github.com/AKS1996/jump-gsoc-2020/blob/master/diffcp_socp_1_py.ipynb
    # Problem SOC2
    # min  x
    # s.t. y ≥ 1/√2
    #      x² + y² ≤ 1
    # in conic form:
    # min  x
    # s.t.  -1/√2 + y ∈ R₊
    #        1 - t ∈ {0}
    #      (t,x,y) ∈ SOC₃

    model = JuMP.Model(() -> DiffOpt.diff_optimizer(SCS.Optimizer))
    set_silent(model)

    x = @variable(model)
    y = @variable(model)
    t = @variable(model)

    ceq = if eq_vec
        @constraint(model, [t] .== [1.0])
    else
        @constraint(model, t == 1.0)
    end
    cnon = @constraint(model, 1.0y >= 1 / √2)
    csoc = @constraint(model, [1.0t, 1.0x, 1.0y] in MOI.SecondOrderCone(3))

    @objective(model, Min, 1.0x)

    optimize!(model)

    # set foward sensitivities
    if eq_vec 
        MOI.set.(model, DiffOpt.ForwardConstraintFunction(), ceq, [1.0 * x])
    else
        MOI.set(model, DiffOpt.ForwardConstraintFunction(), ceq, 1.0 * x)
    end

    DiffOpt.forward_differentiate!(model)

    dx = -0.9999908
    @test MOI.get(model, DiffOpt.ForwardVariablePrimal(), x) ≈ dx atol=ATOL rtol=RTOL

    MOI.set(
        model,
        DiffOpt.ReverseVariablePrimal(),
        x,
        1.0
    )

    DiffOpt.reverse_differentiate!(model)

    if eq_vec
        @test all(isapprox.(JuMP.coefficient.(MOI.get.(model, DiffOpt.ReverseConstraintFunction(), ceq), x), dx, atol=ATOL, rtol=RTOL))
    else
        @test JuMP.coefficient(MOI.get(model, DiffOpt.ReverseConstraintFunction(), ceq), x) ≈ dx atol=ATOL rtol=RTOL
    end

    DiffOpt.empty_input_sensitivities!(model)

    MOI.set(model, DiffOpt.ForwardConstraintFunction(), cnon, 1.0 * y)

    DiffOpt.forward_differentiate!(model)

    dy = -0.707083
    @test MOI.get(model, DiffOpt.ForwardVariablePrimal(), y) ≈ dy atol=ATOL rtol=RTOL

    MOI.set(
        model,
        DiffOpt.ReverseVariablePrimal(),
        y,
        1.0
    )

    DiffOpt.reverse_differentiate!(model)

    @test JuMP.coefficient(MOI.get(model, DiffOpt.ReverseConstraintFunction(), cnon), y) ≈ dy atol=ATOL rtol=RTOL

    DiffOpt.empty_input_sensitivities!(model)
    
    MOI.set(
        model,
        DiffOpt.ForwardConstraintFunction(),
        csoc,
        MOI.Utilities.operate(vcat, Float64, 1.0 * t.index, 0.0, 0.0),
    )

    DiffOpt.forward_differentiate!(model)

    ds = 0.0
    @test MOI.get(model, DiffOpt.ForwardVariablePrimal(), t) ≈ ds atol=ATOL rtol=RTOL

    MOI.set(
        model,
        DiffOpt.ReverseVariablePrimal(),
        t,
        1.0
    )

    DiffOpt.reverse_differentiate!(model)

    # @test JuMP.coefficient(MOI.get(model, DiffOpt.ReverseConstraintFunction(), csoc).func.func.func, t.index) ≈ ds atol=ATOL rtol=RTOL
    
    return
end

test_differentiating_simple_SOCP_vector() = _test_simple_socp(true)

test_differentiating_simple_SOCP_scalar() = _test_simple_socp(false)

# refered from _psd0test, https://github.com/jump-dev/MathOptInterface.jl/blob/master/src/Test/contconic.jl#L3919
# find equivalent diffcp program here: https://github.com/AKS1996/jump-gsoc-2020/blob/master/diffcp_sdp_1_py.ipynb

# min X[1,1] + X[2,2]    max y
#     X[2,1] = 1         [0   y/2     [ 1  0
#                         y/2 0    <=   0  1]
#     X >= 0              y free
# Optimal solution:
#
#     ⎛ 1   1 ⎞
# X = ⎜       ⎟           y = 2
#     ⎝ 1   1 ⎠
function test_simple_psd()
    model = DiffOpt.diff_optimizer(SCS.Optimizer)
    MOI.set(model, MOI.Silent(), true)
    X = MOI.add_variables(model, 3)
    vov = MOI.VectorOfVariables(X)
    cX = MOI.add_constraint(
        model,
        MOI.VectorAffineFunction{Float64}(vov),
        MOI.PositiveSemidefiniteConeTriangle(2),
    )
    c = MOI.add_constraint(
        model,
        MOI.VectorAffineFunction(
            [MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, X[2]))],
            [-1.0],
        ),
        MOI.Zeros(1),
    )
    MOI.set(
        model,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        1.0 * X[1] + 1.0 * X[end],
    )
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.optimize!(model)
    x = MOI.get(model, MOI.VariablePrimal(), X)
    cone_types = unique([
        S for (F, S) in
        MOI.get(model.optimizer, MOI.ListOfConstraintTypesPresent())
    ])
    conic_form = DiffOpt.ConicProgram.Form{Float64}()
    cones = conic_form.constraints.sets
    DiffOpt.set_set_types(cones, cone_types)
    index_map = MOI.copy_to(conic_form, model)
    # s = DiffOpt.map_rows((ci, r) -> MOI.get(model.optimizer, MOI.ConstraintPrimal(), ci), model.optimizer, cones, index_map, DiffOpt.Flattened{Float64}())
    # y = DiffOpt.map_rows((ci, r) -> MOI.get(model.optimizer, MOI.ConstraintDual(), ci), model.optimizer, cones, index_map, DiffOpt.Flattened{Float64}())
    @test x ≈ ones(3) atol = ATOL rtol = RTOL
    # @test s ≈ [0.0; ones(3)] atol=ATOL rtol=RTOL
    # @test y ≈ [2.0, 1.0, -1.0, 1.0]  atol=ATOL rtol=RTOL
    # test1: changing the constant in `c`, i.e. changing value of X[2]
    MOI.set(
        model,
        DiffOpt.ForwardConstraintFunction(),
        c,
        MOI.VectorAffineFunction{Float64}(
            MOI.ScalarAffineTerm{Float64}[],
            [1.0],
        ),
    )
    DiffOpt.forward_differentiate!(model)
    dx = -ones(3)
    for (i, vi) in enumerate(X)
        @test dx[i] ≈ MOI.get(model, DiffOpt.ForwardVariablePrimal(), vi) atol =
            ATOL rtol = RTOL
    end
    # @test dx ≈ -ones(3) atol=ATOL rtol=RTOL  # will change the value of other 2 variables
    # @test ds[2:4] ≈ -ones(3)  atol=ATOL rtol=RTOL  # will affect PSD constraint too
    # test2: changing X[1], X[3] but keeping the objective (their sum) same
    MOI.set(
        model,
        DiffOpt.ForwardConstraintFunction(),
        c,
        MOI.Utilities.zero_with_output_dimension(
            MOI.VectorAffineFunction{Float64},
            1,
        ),
    )
    MOI.set(model, DiffOpt.ForwardObjectiveFunction(), -1.0X[1] + 1.0X[3])
    DiffOpt.forward_differentiate!(model)
    # @test dx ≈ [1.0, 0.0, -1.0] atol=ATOL rtol=RTOL  # note: no effect on X[2]
    dx = [1.0, 0.0, -1.0]
    for (i, vi) in enumerate(X)
        @test dx[i] ≈ MOI.get(model, DiffOpt.ForwardVariablePrimal(), vi) atol =
            ATOL rtol = RTOL
    end
    return
end

function test_differentiating_conic_with_PSD_and_SOC_constraints()
    # similar to _psd1test, https://github.com/jump-dev/MathOptInterface.jl/blob/master/src/Test/contconic.jl#L4054
    # find equivalent diffcp example here - https://github.com/AKS1996/jump-gsoc-2020/blob/master/diffcp_sdp_2_py.ipynb

    #     | 2 1 0 |
    # min | 1 2 1 | . X + x1
    #     | 0 1 2 |
    #
    #
    # s.t. | 1 0 0 |
    #      | 0 1 0 | . X + x1 = 1
    #      | 0 0 1 |
    #
    #      | 1 1 1 |
    #      | 1 1 1 | . X + x2 + x3 = 1/2
    #      | 1 1 1 |
    #
    #      (x1,x2,x3) in C^3_q
    #      X in C_psd
    #
    # The dual is
    # max y1 + y2/2
    #
    # s.t. | y1+y2    y2    y2 |
    #      |    y2 y1+y2    y2 | in C_psd
    #      |    y2    y2 y1+y2 |
    #
    #      (1-y1, -y2, -y2) in C^3_q
    model = DiffOpt.diff_optimizer(SCS.Optimizer)
    MOI.set(model, MOI.Silent(), true)
    δ = √(1 + (3 * √2 + 2) * √(-116 * √2 + 166) / 14) / 2
    ε = √((1 - 2 * (√2 - 1) * δ^2) / (2 - √2))
    y2 = 1 - ε * δ
    y1 = 1 - √2 * y2
    obj = y1 + y2 / 2
    k = -2 * δ / ε
    x2 = ((3 - 2obj) * (2 + k^2) - 4) / (4 * (2 + k^2) - 4 * √2)
    α = √(3 - 2obj - 4x2) / 2
    β = k * α
    X = MOI.add_variables(model, 6)
    x = MOI.add_variables(model, 3)
    vov = MOI.VectorOfVariables(X)
    cX = MOI.add_constraint(
        model,
        MOI.VectorAffineFunction{Float64}(vov),
        MOI.PositiveSemidefiniteConeTriangle(3),
    )
    cx = MOI.add_constraint(
        model,
        MOI.VectorAffineFunction{Float64}(MOI.VectorOfVariables(x)),
        MOI.SecondOrderCone(3),
    )
    c1 = MOI.add_constraint(
        model,
        MOI.VectorAffineFunction(
            MOI.VectorAffineTerm.(
                1:1,
                MOI.ScalarAffineTerm.(
                    [1.0, 1.0, 1.0, 1.0],
                    [X[1], X[3], X[end], x[1]],
                ),
            ),
            [-1.0],
        ),
        MOI.Zeros(1),
    )
    c2 = MOI.add_constraint(
        model,
        MOI.VectorAffineFunction(
            MOI.VectorAffineTerm.(
                1:1,
                MOI.ScalarAffineTerm.(
                    [1.0, 2, 1, 2, 2, 1, 1, 1],
                    [X; x[2]; x[3]],
                ),
            ),
            [-0.5],
        ),
        MOI.Zeros(1),
    )
    # this is a useless constraint - refer the tests below
    # even if we comment this, it won't affect the optimal values
    c_extra = MOI.add_constraint(
        model,
        MOI.VectorAffineFunction(
            MOI.VectorAffineTerm.(1:1, MOI.ScalarAffineTerm.(ones(3), x)),
            [100.0],
        ),
        MOI.Nonnegatives(1),
    )
    objXidx = [1:3; 5:6]
    objXcoefs = 2 * ones(5)
    MOI.set(
        model,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        MOI.ScalarAffineFunction(
            MOI.ScalarAffineTerm.([objXcoefs; 1.0], [X[objXidx]; x[1]]),
            0.0,
        ),
    )
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.optimize!(model)
    _x = MOI.get(model, MOI.VariablePrimal(), x)
    _X = MOI.get(model, MOI.VariablePrimal(), X)
    cone_types = unique([
        S for (F, S) in
        MOI.get(model.optimizer, MOI.ListOfConstraintTypesPresent())
    ])
    conic_form = DiffOpt.ConicProgram.Form{Float64}()
    cones = conic_form.constraints.sets
    DiffOpt.set_set_types(cones, cone_types)
    index_map = MOI.copy_to(conic_form, model)
    # s = DiffOpt.map_rows((ci, r) -> MOI.get(model.optimizer, MOI.ConstraintPrimal(), ci), model.optimizer, cones, index_map, DiffOpt.Flattened{Float64}())
    # y = DiffOpt.map_rows((ci, r) -> MOI.get(model.optimizer, MOI.ConstraintDual(), ci), model.optimizer, cones, index_map, DiffOpt.Flattened{Float64}())
    @test _X ≈ [
        0.21725121
        -0.25996907
        0.31108582
        0.21725009
        -0.25996907
        0.21725121
    ] atol = ATOL rtol = RTOL
    @test _x ≈ [0.2544097; 0.17989425; 0.17989425] atol = ATOL rtol = RTOL
    # @test x ≈ [ 0.21725121; -0.25996907;  0.31108582;  0.21725009; -0.25996907;  0.21725121;
    #             0.2544097;   0.17989425;  0.17989425] atol=ATOL rtol=RTOL
    # @test s ≈ [
    #     0.0, 0.0, 100.614,
    #     0.254408, 0.179894, 0.179894,
    #     0.217251, -0.25997, 0.31109, 0.217251, -0.25997, 0.217251,
    # ] atol=ATOL rtol=RTOL
    # TODO: it should be 100, not 100.614, its surely a residual error
    # Joaquim: it is not an error it is sum(_x)
    # there seemss to be an issue with this test (note the low precision)
    #
    # @test y ≈ [
    #     0.544758, 0.321905, 0.0,
    #     0.455242, -0.321905, -0.321905,
    #     1.13334, 0.678095, 1.13334, -0.321905, 0.678095, 1.13334,
    # ]  atol=ATOL rtol=RTOL
    # test c_extra
    MOI.set(
        model,
        DiffOpt.ForwardConstraintFunction(),
        c_extra,
        MOI.VectorAffineFunction{Float64}(
            MOI.ScalarAffineTerm{Float64}[],
            [1.0],
        ),
    )
    DiffOpt.forward_differentiate!(model)
    # a small change in the constant in c_extra should not affect any other variable or constraint other than c_extra itself
    for (i, vi) in enumerate(X)
        @test 0.0 ≈ MOI.get(model, DiffOpt.ForwardVariablePrimal(), vi) atol =
            1e-2 rtol = RTOL
    end
    for (i, vi) in enumerate(x)
        @test 0.0 ≈ MOI.get(model, DiffOpt.ForwardVariablePrimal(), vi) atol =
            1e-2 rtol = RTOL
    end
    # @test dx ≈ zeros(9) atol=1e-2
    # @test dy ≈ zeros(12) atol=0.012
    # @test [ds[1:2]; ds[4:end]] ≈ zeros(11) atol=1e-2
    # @test ds[3] ≈ 1.0 atol=1e-2   # except c_extra itself
    return
end

function test_differentiating_conic_with_PSD_and_POS_constraints()
    # refer psdt2test, https://github.com/jump-dev/MathOptInterface.jl/blob/master/src/Test/contconic.jl#L4306
    # find equivalent diffcp program here - https://github.com/AKS1996/jump-gsoc-2020/blob/master/diffcp_sdp_3_py.ipynb
    # model = DiffOpt.diff_optimizer(SCS.Optimizer)
    # MOI.set(model, MOI.Silent(), true)
    # x = MOI.add_variables(model, 7)
    # @test MOI.get(model, MOI.NumberOfVariables()) == 7
    # η = 10.0
    # c1 = MOI.add_constraint(
    #     model,
    #     MOI.VectorAffineFunction(
    #         MOI.VectorAffineTerm.(1, MOI.ScalarAffineTerm.(-1.0, x[1:6])),
    #         [η],
    #     ),
    #     MOI.Nonnegatives(1),
    # )
    # c2 = MOI.add_constraint(
    #     model,
    #     MOI.VectorAffineFunction(
    #         MOI.VectorAffineTerm.(1:6, MOI.ScalarAffineTerm.(1.0, x[1:6])),
    #         zeros(6),
    #     ),
    #     MOI.Nonnegatives(6),
    # )
    # α = 0.8
    # δ = 0.9
    # c3 = MOI.add_constraint(
    #     model,
    #     MOI.VectorAffineFunction(
    #         MOI.VectorAffineTerm.(
    #             [fill(1, 7); fill(2, 5); fill(3, 6)],
    #             MOI.ScalarAffineTerm.(
    #                 [
    #                     δ / 2,
    #                     α,
    #                     δ,
    #                     δ / 4,
    #                     δ / 8,
    #                     0.0,
    #                     -1.0,
    #                     -δ / (2 * √2),
    #                     -δ / 4,
    #                     0,
    #                     -δ / (8 * √2),
    #                     0.0,
    #                     δ / 2,
    #                     δ - α,
    #                     0,
    #                     δ / 8,
    #                     δ / 4,
    #                     -1.0,
    #                 ],
    #                 [x[1:7]; x[1:3]; x[5:6]; x[1:3]; x[5:7]],
    #             ),
    #         ),
    #         zeros(3),
    #     ),
    #     MOI.PositiveSemidefiniteConeTriangle(2),
    # )
    # c4 = MOI.add_constraint(
    #     model,
    #     MOI.VectorAffineFunction(
    #         MOI.VectorAffineTerm.(
    #             1,
    #             MOI.ScalarAffineTerm.(0.0, [x[1:3]; x[5:6]]),
    #         ),
    #         [0.0],
    #     ),
    #     MOI.Zeros(1),
    # )
    # MOI.set(
    #     model,
    #     MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
    #     MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, x[7])], 0.0),
    # )
    # MOI.set(model, MOI.ObjectiveSense(), MOI.MAX_SENSE)
    # MOI.optimize!(model)
    # MOI.get(model, MOI.ObjectiveValue())
    # # dc = ones(7)
    # MOI.get.(model, MOI.VariablePrimal(), x)

    # Make a JuMP model backed by DiffOpt.diff_optimizer(SCS.Optimizer)
    model = Model(() -> DiffOpt.diff_optimizer(SCS.Optimizer))
    set_silent(model)  # just to suppress solver output

    @variable(model, x[1:7])
    @test num_variables(model) == 7

    η = 10.0
    # c1: sum(-x[1:6]) + η >= 0
    c1 = @constraint(model, -sum(x[i] for i in 1:6) + η ≥ 0)

    # c2: x[i] >= 0  for each i in 1:6
    #     (this replicates MOI.Nonnegatives(6) on x[1:6])
    c2 = [
        @constraint(model, x[i] ≥ 0)
        for i in 1:6
    ]

    # c3: A 2×2 PSD constraint. In raw MOI form you had
    #     MOI.PositiveSemidefiniteConeTriangle(2),
    #     which means a vector dimension=3 => [a11, a12, a22].
    # We'll build that in JuMP with `[ [a11 a12]; [a12 a22] ] in PSDCone()`.
    α = 0.8
    δ = 0.9
    c3 = @constraint(
        model,
        LinearAlgebra.Symmetric(hcat(
            [((δ/2)*x[1] + α*x[2] + δ*x[3] + (δ/4)*x[4] + (δ/8)*x[5] - 1.0*x[7])   # a11
            ((-(δ/(2*√2)))*x[1] - (δ/4)*x[2] - (δ/(8*√2))*x[5])],
            [((-(δ/(2*√2)))*x[1] - (δ/4)*x[2] - (δ/(8*√2))*x[5])
            ((δ/2)*x[1] + (δ - α)*x[2] + 0.0*x[3] + (δ/8)*x[5] + (δ/4)*x[6] - x[7])])
        ) in PSDCone()
    )

    # c4: In the original MOI example it was some "useless" constraint on x[1:3] + x[5:6].
    #     For demonstration, we replicate a dimension=1 equality to zero:
    c4 = @constraint(model, 0.0 * x[1] + 0.0 * x[2] + 0.0 * x[3] + 0.0 * x[5] + 0.0 * x[6] == 0)

    # Make the objective: "maximize x[7]" exactly as in the original
    @objective(model, Max, x[7])

    # Solve
    optimize!(model)

    @test objective_value(model) ≈ 1.90192374 atol=ATOL rtol=RTOL

    MOI.set(
        model,
        DiffOpt.ForwardObjectiveFunction(),
       sum(x)
    )
    # db = ones(11)
    # dA = ones(11, 7)
    MOI.set(
        model,
        DiffOpt.ForwardConstraintFunction(),
        c1,
        sum(x) + 1,
    )
    MOI.set.(
        model,
        DiffOpt.ForwardConstraintFunction(),
        c2,
        sum(x) + 1,
    )
    MOI.set.(
        model,
        DiffOpt.ForwardConstraintFunction(),
        c3,
        sum(x) + 1,
    )
    MOI.set(
        model,
        DiffOpt.ForwardConstraintFunction(),
        c4,
        sum(x) + 1,
    )
    DiffOpt.forward_differentiate!(model) # ERROR HERE
    @test model.diff.model.x ≈
          [20 / 3.0, 0.0, 10 / 3.0, 0.0, 0.0, 0.0, 1.90192379] atol = ATOL rtol =
        RTOL
    @test model.diff.model.s ≈ [
        0.0,
        0.0,
        20 / 3.0,
        0.0,
        10 / 3.0,
        0.0,
        0.0,
        0.0,
        4.09807621,
        -2.12132,
        1.09807621,
    ] atol = ATOL rtol = RTOL
    @test model.diff.model.y ≈ [
        0.0,
        0.19019238,
        0.0,
        0.12597667,
        0.0,
        0.14264428,
        0.14264428,
        0.01274047,
        0.21132487,
        0.408248,
        0.78867513,
    ] atol = ATOL rtol = RTOL
    atol = 0.3
    rtol = 0.01
    # compare these with https://github.com/AKS1996/jump-gsoc-2020/blob/master/diffcp_sdp_3_py.ipynb
    # results are not exactly as: 1. there is some residual error   2. diffcp results are SCS specific, hence scaled
    dx = [-39.6066, 10.8953, -14.9189, 10.9054, 10.883, 10.9118, -21.7508]
    for (i, vi) in enumerate(x)
        @test dx[i] ≈ MOI.get(model, DiffOpt.ForwardVariablePrimal(), vi) atol =
            atol rtol = rtol
    end
    # @test dy ≈ [0.0, -3.56905, 0.0, -0.380035, 0.0, -0.41398, -0.385321, -0.00743119, -0.644986, -0.550542, -2.36765] atol=atol rtol=rtol
    # @test ds ≈ [0.0, 0.0, -50.4973, 0.0, -25.8066, 0.0, 0.0, 0.0, -7.96528, -1.62968, -2.18925] atol=atol rtol=rtol
    # TODO: future example, how to differentiate wrt a specific constraint/variable, refer QPLib article for more
    dA = zeros(11, 7)
    dA[3:8, 1:6] = Matrix{Float64}(LinearAlgebra.I, 6, 6)  # differentiating only wrt POS constraint c2
    db = zeros(11)
    dc = zeros(7)
    # db = zeros(11)
    # dA = zeros(11, 7)
    # dA[3:8, 1:6] = Matrix{Float64}(LinearAlgebra.I, 6, 6)  # differentiating only wrt POS constraint c2
    MOI.set(
        model,
        DiffOpt.ForwardConstraintFunction(),
        c1,
        MOI.Utilities.zero_with_output_dimension(
            MOI.VectorAffineFunction{Float64},
            1,
        ),
    )
    MOI.set(
        model,
        DiffOpt.ForwardConstraintFunction(),
        c2,
        MOI.Utilities.vectorize(ones(6) .* x[1:6]),
    )
    MOI.set(
        model,
        DiffOpt.ForwardConstraintFunction(),
        c3,
        MOI.Utilities.zero_with_output_dimension(
            MOI.VectorAffineFunction{Float64},
            3,
        ),
    )
    MOI.set(
        model,
        DiffOpt.ForwardConstraintFunction(),
        c4,
        MOI.Utilities.zero_with_output_dimension(
            MOI.VectorAffineFunction{Float64},
            1,
        ),
    )
    DiffOpt.forward_differentiate!(model)
    # for (i, vi) in enumerate(X)
    #     @test 0.0 ≈ MOI.get(model,
    #         DiffOpt.ForwardVariablePrimal(), vi) atol=1e-2 rtol=RTOL
    # end
    # TODO add a test here, probably on duals
    # # note that there's no change in the PSD slack values or dual optimas
    # @test dy ≈ [0.0, 0.0, 0.0, 0.125978, 0.0, 0.142644, 0.142641, 0.0127401, 0.0, 0.0, 0.0] atol=atol rtol=RTOL
    # @test ds ≈ [0.0, 0.0, -6.66672, 0.0, -3.33336, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] atol=atol rtol=RTOL
    return
end

function test_differentiating_a_simple_psd()
    # refer _psd3test, https://github.com/jump-dev/MathOptInterface.jl/blob/master/src/Test/contconic.jl#L4484
    # find equivalent diffcp program here - https://github.com/AKS1996/jump-gsoc-2020/blob/master/diffcp_sdp_0_py.ipynb

    # min x
    # s.t. [x 1 1]
    #      [1 x 1] ⪰ 0
    #      [1 1 x]
    model = DiffOpt.diff_optimizer(SCS.Optimizer)
    MOI.set(model, MOI.Silent(), true)
    x = MOI.add_variable(model)
    func = MOI.Utilities.operate(vcat, Float64, x, 1.0, x, 1.0, 1.0, x)
    # do not confuse this constraint with the matrix `c` in the conic form (of the matrices A, b, c)
    c = MOI.add_constraint(model, func, MOI.PositiveSemidefiniteConeTriangle(3))
    MOI.set(
        model,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(1.0, [x]), 0.0),
    )
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.optimize!(model)
    # SCS/Mosek specific
    # @test s' ≈ [1.         1.41421356 1.41421356 1.         1.41421356 1.        ] atol=ATOL rtol=RTOL
    # @test y' ≈ [ 0.33333333 -0.23570226 -0.23570226  0.33333333 -0.23570226  0.33333333]  atol=ATOL rtol=RTOL
    # dA = zeros(6, 1)
    # db = ones(6)
    MOI.set(
        model,
        DiffOpt.ForwardConstraintFunction(),
        c,
        MOI.VectorAffineFunction{Float64}(
            MOI.ScalarAffineTerm{Float64}[],
            ones(6),
        ),
    )
    # dc = zeros(1)
    DiffOpt.forward_differentiate!(model)
    @test model.diff.model.x ≈ [1.0] atol = 10ATOL rtol = 10RTOL
    @test model.diff.model.s ≈ ones(6) atol = ATOL rtol = RTOL
    @test model.diff.model.y ≈ [1 / 3, -1 / 6, 1 / 3, -1 / 6, -1 / 6, 1 / 3] atol =
        ATOL rtol = RTOL
    @test -0.5 ≈ MOI.get(model, DiffOpt.ForwardVariablePrimal(), x) atol = 1e-2 rtol =
        RTOL
    # @test dx ≈ [-0.5] atol=ATOL rtol=RTOL
    # @test dy ≈ zeros(6) atol=ATOL rtol=RTOL
    # @test ds ≈ [0.5, 1.0, 0.5, 1.0, 1.0, 0.5] atol=ATOL rtol=RTOL
    # test 2
    dA = zeros(6, 1)
    db = zeros(6)
    MOI.set(
        model,
        DiffOpt.ForwardConstraintFunction(),
        c,
        MOI.Utilities.zero_with_output_dimension(
            MOI.VectorAffineFunction{Float64},
            6,
        ),
    )
    MOI.set(model, DiffOpt.ForwardObjectiveFunction(), 1.0 * x)
    DiffOpt.forward_differentiate!(model)
    @test 0.0 ≈ MOI.get(model, DiffOpt.ForwardVariablePrimal(), x) atol = 1e-2 rtol =
        RTOL
    # @test dx ≈ zeros(1) atol=ATOL rtol=RTOL
    # @test dy ≈ [0.333333, -0.333333, 0.333333, -0.333333, -0.333333, 0.333333] atol=ATOL rtol=RTOL
    # @test ds ≈ zeros(6) atol=ATOL rtol=RTOL
    return
end

function test_verifying_cache_after_differentiating_a_qp()
    Q = [4.0 1.0; 1.0 2.0]
    q = [1.0, 1.0]
    G = [1.0 1.0]
    h = [-1.0]
    model = DiffOpt.diff_optimizer(SCS.Optimizer)
    MOI.set(model, MOI.Silent(), true)
    x = MOI.add_variables(model, 2)
    # define objective
    quad_terms = MOI.ScalarQuadraticTerm{Float64}[]
    for i in 1:2
        for j in i:2 # indexes (i,j), (j,i) will be mirrored. specify only one kind
            push!(quad_terms, MOI.ScalarQuadraticTerm(Q[i, j], x[i], x[j]))
        end
    end
    objective_function = MOI.ScalarQuadraticFunction(
        quad_terms,
        MOI.ScalarAffineTerm.(q, x),
        0.0,
    )
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(
        model,
        MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(),
        objective_function,
    )
    # add constraint
    c = MOI.add_constraint(
        model,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(G[1, :], x), 0.0),
        MOI.LessThan(h[1]),
    )
    MOI.optimize!(model)
    x_sol = MOI.get(model, MOI.VariablePrimal(), x)
    @test x_sol ≈ [-0.25, -0.75] atol = ATOL rtol = RTOL
    MOI.set.(model, DiffOpt.ReverseVariablePrimal(), x, ones(2))
    @test model.diff === nothing
    DiffOpt.reverse_differentiate!(model)
    grad_wrt_h =
        MOI.constant(MOI.get(model, DiffOpt.ReverseConstraintFunction(), c))
    @test grad_wrt_h ≈ -1.0 atol = 2ATOL rtol = RTOL
    # adding two variables invalidates the cache
    y = MOI.add_variables(model, 2)
    MOI.delete(model, y)
    @test model.diff === nothing
    MOI.optimize!(model)
    MOI.set.(model, DiffOpt.ReverseVariablePrimal(), x, ones(2))
    DiffOpt.reverse_differentiate!(model)
    grad_wrt_h =
        MOI.constant(MOI.get(model, DiffOpt.ReverseConstraintFunction(), c))
    @test grad_wrt_h ≈ -1.0 atol = 2ATOL rtol = RTOL
    # adding single variable invalidates the cache
    y0 = MOI.add_variable(model)
    @test model.diff === nothing
    MOI.add_constraint(
        model,
        MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, y0)], 0.0),
        MOI.EqualTo(42.0),
    )
    MOI.optimize!(model)
    @test model.diff === nothing
    MOI.set.(model, DiffOpt.ReverseVariablePrimal(), x, ones(2))
    DiffOpt.reverse_differentiate!(model)
    grad_wrt_h =
        MOI.constant(MOI.get(model, DiffOpt.ReverseConstraintFunction(), c))
    @test grad_wrt_h ≈ -1.0 atol = 5e-3 rtol = RTOL
    @test model.diff.model.gradient_cache isa DiffOpt.QuadraticProgram.Cache
    # adding constraint invalidates the cache
    c2 = MOI.add_constraint(
        model,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0, 1.0], x), 0.0),
        MOI.LessThan(0.0),
    )
    @test model.diff === nothing
    MOI.optimize!(model)
    MOI.set.(model, DiffOpt.ReverseVariablePrimal(), x, ones(2))
    DiffOpt.reverse_differentiate!(model)
    grad_wrt_h =
        MOI.constant(MOI.get(model, DiffOpt.ReverseConstraintFunction(), c))
    @test grad_wrt_h ≈ -1.0 atol = 5e-3 rtol = RTOL
    # second constraint inactive
    grad_wrt_h =
        MOI.constant(MOI.get(model, DiffOpt.ReverseConstraintFunction(), c2))
    @test grad_wrt_h ≈ 0.0 atol = 5e-3 rtol = RTOL
    @test model.diff.model.gradient_cache isa DiffOpt.QuadraticProgram.Cache
    return
end

function test_verifying_cache_on_a_psd()
    model = DiffOpt.diff_optimizer(SCS.Optimizer)
    MOI.set(model, MOI.Silent(), true)
    x = MOI.add_variable(model)
    func = MOI.Utilities.operate(vcat, Float64, x, 1.0, x, 1.0, 1.0, x)
    c = MOI.add_constraint(model, func, MOI.PositiveSemidefiniteConeTriangle(3))
    MOI.set(
        model,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(1.0, [x]), 0.0),
    )
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    @test model.diff === nothing
    MOI.optimize!(model)
    @test model.diff === nothing
    dA = zeros(6, 1)
    db = ones(6)
    MOI.set(
        model,
        DiffOpt.ForwardConstraintFunction(),
        c,
        MOI.VectorAffineFunction{Float64}(
            MOI.ScalarAffineTerm{Float64}[],
            ones(6),
        ),
    )
    dc = zeros(1)
    DiffOpt.forward_differentiate!(model)
    @test model.diff.model.x ≈ [1.0] atol = 10ATOL rtol = 10RTOL
    @test model.diff.model.s ≈ ones(6) atol = ATOL rtol = RTOL
    @test model.diff.model.y ≈ [1 / 3, -1 / 6, 1 / 3, -1 / 6, -1 / 6, 1 / 3] atol =
        ATOL rtol = RTOL
    @test -0.5 ≈ MOI.get(model, DiffOpt.ForwardVariablePrimal(), x) atol = 1e-2 rtol =
        RTOL
    @test model.diff.model.gradient_cache isa DiffOpt.ConicProgram.Cache
    DiffOpt.forward_differentiate!(model)
    @test -0.5 ≈ MOI.get(model, DiffOpt.ForwardVariablePrimal(), x) atol = 1e-2 rtol =
        RTOL
    # test 2
    MOI.set(
        model,
        DiffOpt.ForwardConstraintFunction(),
        c,
        MOI.VectorAffineFunction{Float64}(
            MOI.ScalarAffineTerm{Float64}[],
            zeros(6),
        ),
    )
    MOI.set(model, DiffOpt.ForwardObjectiveFunction(), 1.0 * x)
    DiffOpt.forward_differentiate!(model)
    @test 0.0 ≈ MOI.get(model, DiffOpt.ForwardVariablePrimal(), x) atol = 1e-2 rtol =
        RTOL
    return
end

# min X[1,1] + X[2,2]    max y
#     X[2,1] = 1         [0   y/2     [ 1  0
#                         y/2 0    <=   0  1]
#     X >= 0              y free
# Optimal solution:
#
#     ⎛ 1   1 ⎞
# X = ⎜       ⎟           y = 2
#     ⎝ 1   1 ⎠
function test_differentiating_simple_PSD_back()
    model = DiffOpt.diff_optimizer(SCS.Optimizer)
    MOI.set(model, MOI.Silent(), true)
    X = MOI.add_variables(model, 3)
    vov = MOI.VectorOfVariables(X)
    cX = MOI.add_constraint(
        model,
        MOI.VectorAffineFunction{Float64}(vov),
        MOI.PositiveSemidefiniteConeTriangle(2),
    )
    c = MOI.add_constraint(
        model,
        MOI.VectorAffineFunction(
            [MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, X[2]))],
            [-1.0],
        ),
        MOI.Zeros(1),
    )
    MOI.set(
        model,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        MOI.ScalarAffineFunction(
            MOI.ScalarAffineTerm.(1.0, [X[1], X[end]]),
            0.0,
        ),
    )
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.optimize!(model)
    x = MOI.get(model, MOI.VariablePrimal(), X)
    cone_types = unique([
        S for (F, S) in
        MOI.get(model.optimizer, MOI.ListOfConstraintTypesPresent())
    ])
    conic_form = DiffOpt.ConicProgram.Form{Float64}()
    cones = conic_form.constraints.sets
    DiffOpt.set_set_types(cones, cone_types)
    index_map = MOI.copy_to(conic_form, model)
    @test x ≈ ones(3) atol = ATOL rtol = RTOL
    MOI.set(model, DiffOpt.ReverseVariablePrimal(), X[1], 1.0)
    DiffOpt.reverse_differentiate!(model)
    db = MOI.constant(MOI.get(model, DiffOpt.ReverseConstraintFunction(), c))
    @test db ≈ [-1.0] atol = ATOL rtol = RTOL
    return
end

function test_singular_exception()
    Q = [4.0 1.0; 1.0 2.0]
    q = [1.0, 1.0]
    G = [1.0 1.0]
    h = [-1.0]
    model = DiffOpt.diff_optimizer(Ipopt.Optimizer)
    MOI.set(model, MOI.Silent(), true)
    x = MOI.add_variables(model, 2)
    quad_terms = MOI.ScalarQuadraticTerm{Float64}[]
    for i in 1:2
        for j in i:2
            push!(quad_terms, MOI.ScalarQuadraticTerm(Q[i, j], x[i], x[j]))
        end
    end
    objective_function = MOI.ScalarQuadraticFunction(
        quad_terms,
        MOI.ScalarAffineTerm.(q, x),
        0.0,
    )
    MOI.set(
        model,
        MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(),
        objective_function,
    )
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    c = MOI.add_constraint(
        model,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(G[1, :], x), 0.0),
        MOI.LessThan(h[1]),
    )
    MOI.optimize!(model)
    @test MOI.get(model, MOI.VariablePrimal(), x) ≈ [-0.25; -0.75] atol = ATOL rtol =
        RTOL
    @test model.diff === nothing
    MOI.set.(model, DiffOpt.ReverseVariablePrimal(), x, ones(2))
    DiffOpt.reverse_differentiate!(model)
    grad_wrt_h =
        MOI.constant(MOI.get(model, DiffOpt.ReverseConstraintFunction(), c))
    @test grad_wrt_h ≈ -1.0 atol = 2ATOL rtol = RTOL
    @test model.diff !== nothing
    # adding two variables invalidates the cache
    y = MOI.add_variables(model, 2)
    for yi in y
        MOI.delete(model, yi)
    end
    @test model.diff === nothing
    MOI.optimize!(model)
    MOI.set.(model, DiffOpt.ReverseVariablePrimal(), x, ones(2))
    DiffOpt.reverse_differentiate!(model)
    grad_wrt_h =
        MOI.constant(MOI.get(model, DiffOpt.ReverseConstraintFunction(), c))
    @test grad_wrt_h ≈ -1.0 atol = 1e-3
    @test model.diff !== nothing
    return
end

function test_quad_to_soc()
    model = DiffOpt.ConicProgram.Model()
    F = MOI.VectorAffineFunction{Float64}
    S = MOI.RotatedSecondOrderCone
    # Adds the set type to the list
    @test MOI.supports_constraint(model, F, S)
    bridged = MOI.Bridges.Constraint.QuadtoSOC{Float64}(model)
    x = MOI.add_variables(bridged, 3)
    # We pick orthogonal rows but this is not the `U` that cholesky is going to
    # take in the bridge.
    U = [1 -1 1; 1 2 1; 1 0 -1.0]
    a = [1, 2, -3.0]
    b = -5.0
    Q = U' * U
    f = x' * Q * x / 2.0 + a' * x
    c = MOI.add_constraint(bridged, f, MOI.LessThan(-b))
    dQ = [1 -1 0; -1 2 1; 0 1 3.0]
    da = [-1, 1.0, -2]
    db = 3.0
    df = x' * dQ * x / 2.0 + da' * x + db
    MOI.Utilities.final_touch(bridged, nothing)
    MOI.set(bridged, DiffOpt.ForwardConstraintFunction(), c, df)
    return
end

end  # module

TestConicProgram.runtests()
