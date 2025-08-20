# Copyright (c) 2020: Akshay Sharma and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module TestQuadraticProgram

using Test
import DelimitedFiles
import DiffOpt
import HiGHS
import Ipopt
import MathOptInterface as MOI
import SCS

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

include(joinpath(@__DIR__, "utils.jl"))

function test_forward_on_trivial_QP()
    # using example on https://osqp.org/docs/examples/setup-and-solve.html
    Q = [4.0 1.0; 1.0 2.0]
    q = [1.0, 1.0]
    G = [
        1.0 1.0
        1.0 0.0
        0.0 1.0
        -1.0 -1.0
        -1.0 0.0
        0.0 -1.0
    ]
    h = [1, 0.7, 0.7, -1, 0, 0]
    qp_test_with_solutions(
        Ipopt.Optimizer;
        Q = Q,
        q = q,
        G = G,
        h = h,
        dzb = ones(2),
        dQf = [1 -1; -1 1.0],
        dqf = [1, -1.0],
        dGf = ones(6, 2),
        dhf = ones(6),
        # Expected solutions
        z = [0.3, 0.7],
    )
    return
end

function test_differentiating_trivial_qp_1()
    Q = [
        4.0 1.0
        1.0 2.0
    ]
    q = [1.0, 1.0]
    G = [1.0 1.0]
    h = [-1.0]

    model = DiffOpt.diff_optimizer(Ipopt.Optimizer)
    MOI.set(model, MOI.Silent(), true)
    x = MOI.add_variables(model, 2)

    qp_test_with_solutions(
        Ipopt.Optimizer;
        Q = Q,
        q = q,
        G = G,
        h = h,
        dzb = ones(2),
        dQf = -ones(2, 2),
        dqf = ones(2),
        dGf = ones(1, 2),
        dhf = -ones(1),
        # Expected solutions
        z = [-0.25; -0.75],
        dhb = ones(1),
    )
    return
end

# @testset "Differentiating a non-convex QP" begin
#     Q = [0.0 0.0; 1.0 2.0]
#     q = [1.0; 1.0]
#     G = [1.0 1.0;]
#     h = [-1.0;]
#
#     model = DiffOpt.diff_optimizer(Ipopt.Optimizer)
#     x = MOI.add_variables(model, 2)
#
#     # define objective
#     quad_terms = MOI.ScalarQuadraticTerm{Float64}[]
#     for i in 1:2
#         for j in i:2 # indexes (i,j), (j,i) will be mirrored. specify only one kind
#             push!(
#                 quad_terms,
#                 MOI.ScalarQuadraticTerm(Q[i,j], x[i], x[j]),
#             )
#         end
#     end
#
#     objective_function = MOI.ScalarQuadraticFunction(
#                             quad_terms,
#                             MOI.ScalarAffineTerm.(q, x),
#                             0.0,
#                         )
#     MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(), objective_function)
#     MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
#
#     # add constraint
#     MOI.add_constraint(
#         model,
#         MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(G[1, :], x), 0.0),
#         MOI.LessThan(h[1]),
#     )
#
#     @test_throws ErrorException MOI.optimize!(model) # should break
# end

function test_differentiating_qp_with_inequality_and_equality_constraints()
    # refered from: https://www.mathworks.com/help/optim/ug/quadprog.html#d120e113424
    # Find equivalent qpth program here - https://github.com/AKS1996/jump-gsoc-2020/blob/master/DiffOpt_tests_4_py.ipynb
    Q = [
        1.0 -1.0 1.0
        -1.0 2.0 -2.0
        1.0 -2.0 4.0
    ]
    q = [2.0, -3.0, 1.0]
    G = [
        0.0 0.0 1.0
        0.0 1.0 0.0
        1.0 0.0 0.0
        0.0 0.0 -1.0
        0.0 -1.0 0.0
        -1.0 0.0 0.0
    ]
    h = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    A = [1.0 1.0 1.0;]
    b = [0.5]
    qp_test_with_solutions(
        Ipopt.Optimizer;
        Q = Q,
        q = q,
        G = G,
        h = h,
        A = A,
        b = b,
        dzb = ones(3),
        dQf = ones(3, 3),
        dqf = ones(3),
        dGf = ones(6, 3),
        dhf = ones(6),
        dAf = ones(1, 3),
        dbf = ones(1),
        # Expected solutions
        z = [0.0, 0.5, 0.0],
        dQb = zeros(3, 3),
        dqb = zeros(3),
        dGb = zeros(6, 3),
        dhb = zeros(6),
        dAb = [0.0 -0.5 0.0],
        dbb = [1.0],
    )
    return
end

# refered from https://github.com/jump-dev/MathOptInterface.jl/blob/master/src/Test/contquadratic.jl#L3
# Find equivalent CVXPYLayers and QPTH code here:
#               https://github.com/AKS1996/jump-gsoc-2020/blob/master/DiffOpt_tests_1_py.ipynb
function test_differentiating_moi_examples_1()
    # homogeneous quadratic objective
    # Min x^2 + xy + y^2 + yz + z^2
    # st  x + 2y + 3z >= 4 (c1)
    #     x +  y      >= 1 (c2)
    #     x, y, z \in R
    Q = [
        2.0 1.0 0.0
        1.0 2.0 1.0
        0.0 1.0 2.0
    ]
    q = zeros(3)
    G = [
        -1.0 -2.0 -3.0
        -1.0 -1.0 0.0
    ]
    h = [-4.0, -1.0]
    dQ = [
        -0.12244895 0.01530609 -0.11224488
        0.01530609 0.09183674 0.07653058
        -0.11224488 0.07653058 -0.06122449
    ]
    dq = [-0.2142857, 0.21428567, -0.07142857]
    dG = [
        0.05102692 0.30612244 0.25510856
        0.06120519 0.36734693 0.30610315
    ]
    dh = [-0.35714284; -0.4285714]
    qp_test_with_solutions(
        Ipopt.Optimizer;
        Q = Q,
        q = q,
        G = G,
        h = h,
        dzb = ones(3),
        dQf = dQ,
        dqf = dq,
        dGf = dG,
        dhf = dh,
        # Expected solutions
        dQb = dQ,
        dqb = dq,
        dGb = dG,
        dhb = dh,
    )
    return
end

# refered from https://github.com/jump-dev/MathOptInterface.jl/blob/master/src/Test/contquadratic.jl#L3
# Find equivalent CVXPYLayers and QPTH code here:
#               https://github.com/AKS1996/jump-gsoc-2020/blob/master/DiffOpt_tests_2_py.ipynb
function test_differentiating_moi_examples_2()
    # non-homogeneous quadratic objective
    #    minimize 2 x^2 + y^2 + xy + x + y
    #       s.t.  x, y >= 0
    #             x + y = 1
    Q = [
        4 1.0
        1 2
    ]
    q = [1, 1.0]
    G = [
        -1 0.0
        0 -1
    ]
    h = [0, 0.0]
    A = [1 1.0]
    b = [1.0]
    dQ = [
        -0.05 -0.05
        -0.05 0.15
    ]
    dq = [-0.2, 0.2]
    dG = zeros(2, 2)
    dh = zeros(2)
    dA = [0.375 -1.075]
    db = [0.7]
    qp_test_with_solutions(
        Ipopt.Optimizer;
        Q = Q,
        q = q,
        G = G,
        h = h,
        A = A,
        b = b,
        dzb = [1.3, 0.5],
        dQf = dQ,
        dqf = dq,
        dGf = dG,
        dhf = dh,
        dAf = dA,
        dbf = db,
        # Expected solutions
        dQb = dQ,
        dqb = dq,
        dGb = dG,
        dhb = dh,
        dAb = dA,
        dbb = db,
        z = [0.25, 0.75],
        dzf = [1.4875, -0.075],
        ∇zf = [-1.28125, 3.25625],
        ∇zb = [-0.2, 0.2],
        λ = zeros(2),
        dλf = zeros(2),
        dλb = zeros(2),
        ∇λb = [0.8, -0.8 / 3],
        ν = [-2.75],
        dνb = zeros(1),
        ∇νb = [-0.7],
    )
    return
end

function test_differentiating_non_trivial_convex_qp_moi()
    nz = 10
    nineq_le = 25
    neq = 10
    # read matrices from files
    names = ["P", "q", "G", "h", "A", "b"]
    matrices = []
    for name in names
        filename = joinpath(@__DIR__, "data", "$name.txt")
        push!(matrices, DelimitedFiles.readdlm(filename, ' ', Float64, '\n'))
    end
    Q, q, G, h, A, b = matrices
    q = vec(q)
    h = vec(h)
    b = vec(b)
    # read gradients from files
    names = ["dP", "dq", "dG", "dh", "dA", "db"]
    grads_actual = []
    for name in names
        filename = joinpath(@__DIR__, "data", "$name.txt")
        push!(
            grads_actual,
            DelimitedFiles.readdlm(filename, ' ', Float64, '\n'),
        )
    end
    dqb = vec(grads_actual[2])
    dhb = vec(grads_actual[4])
    dbb = vec(grads_actual[6])
    qp_test(
        Ipopt.Optimizer,
        DiffOpt.QuadraticProgram.Model,
        true,
        true,
        true;
        Q = Q,
        q = q,
        G = G,
        h = h,
        A = A,
        b = b,
        dzb = ones(nz),
        dQf = ones(nz, nz),
        dqf = ones(nz),
        dGf = ones(length(h), nz),
        dhf = ones(length(h)),
        dAf = ones(length(b), nz),
        dbf = ones(length(b)),
        # Expected solutions
        dqb = dqb,
        dhb = dhb,
        dbb = dbb,
        atol = 1e-3, # The values in `data` seems to have low accuracy
        rtol = 1e-3, # The values in `data` seems to have low accuracy
    )
    return
end

function test_ObjectiveSensitivity()
    model = DiffOpt.quadratic_diff_model(HiGHS.Optimizer)
    @variable(model, x)
    @variable(model, p in MOI.Parameter(1.0))
    @constraint(model, x >= p)
    @objective(model, Min, x)
    optimize!(model)
    direction_p = 2.0
    DiffOpt.set_forward_parameter(model, p, direction_p)

    DiffOpt.forward_differentiate!(model)

    # TODO: Change when implemented
    @test_throws ErrorException("Not implemented") MOI.get(model, DiffOpt.ForwardObjectiveSensitivity())

    # Clean up
    DiffOpt.empty_input_sensitivities!(model)

    # TODO: Change when implemented
    MOI.set(model, DiffOpt.ReverseObjectiveSensitivity(), 0.5)

    @test_throws ErrorException("Not implemented") DiffOpt.reverse_differentiate!(model)
end

end  # module

TestQuadraticProgram.runtests()
