using DiffOpt
using Test
using MathOptInterface
const MOI = MathOptInterface
const MOIU = MOI.Utilities

@testset "Testing forward on trivial QP" begin
    # using example on https://osqp.org/docs/examples/setup-and-solve.html
    Q = [
        4.0 1.0
        1.0 2.0
    ]
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

    model = diff_optimizer(Ipopt.Optimizer)
    MOI.set(model, MOI.Silent(), true)
    x = MOI.add_variables(model, 2)

    # define objective
    quad_terms = MOI.ScalarQuadraticTerm{Float64}[]
    for i in 1:2
        for j in i:2 # indexes (i,j), (j,i) will be mirrored. specify only one kind
            push!(
                quad_terms,
                MOI.ScalarQuadraticTerm(Q[i,j],x[i],x[j])
            )
        end
    end

    objective_function = MOI.ScalarQuadraticFunction(
                            MOI.ScalarAffineTerm.(q, x),
                            quad_terms,
                            0.0
                        )
    MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(), objective_function)
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    # add constraints
    for i in 1:6
        MOI.add_constraint(
            model,
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(G[i,:], x), 0.0),
            MOI.LessThan(h[i])
        )
    end

    MOI.optimize!(model)

    @test model.primal_optimal ≈ [0.3, 0.7] atol=ATOL rtol=RTOL
end



@testset "Differentiating trivial QP 1" begin
    Q = [
        4.0 1.0
        1.0 2.0
    ]
    q = [1.0, 1.0]
    G = [1.0 1.0]
    h = [-1.0]

    model = diff_optimizer(OSQP.Optimizer)
    MOI.set(model, MOI.Silent(), true)
    x = MOI.add_variables(model, 2)

    # define objective
    quad_terms = MOI.ScalarQuadraticTerm{Float64}[]
    for i in 1:2
        for j in i:2 # indexes (i,j), (j,i) will be mirrored. specify only one kind
            push!(
                quad_terms,
                MOI.ScalarQuadraticTerm(Q[i,j], x[i], x[j])
            )
        end
    end

    objective_function = MOI.ScalarQuadraticFunction(
                            MOI.ScalarAffineTerm.(q, x),
                            quad_terms,
                            0.0,
                        )
    MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(), objective_function)
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    # add constraint
    MOI.add_constraint(
        model,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(G[1, :], x), 0.0),
        MOI.LessThan(h[1])
    )

    MOI.optimize!(model)

    @test model.primal_optimal ≈ [-0.25; -0.75] atol=ATOL rtol=RTOL

    grad_wrt_h = backward(model, ["h"], ones(2))[1]

    @test grad_wrt_h ≈ [1.0] atol=2ATOL rtol=RTOL
end


# @testset "Differentiating a non-convex QP" begin
#     Q = [0.0 0.0; 1.0 2.0]
#     q = [1.0; 1.0]
#     G = [1.0 1.0;]
#     h = [-1.0;]

#     model = diff_optimizer(OSQP.Optimizer)
#     x = MOI.add_variables(model, 2)

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

#     objective_function = MOI.ScalarQuadraticFunction(
#                             MOI.ScalarAffineTerm.(q, x),
#                             quad_terms,
#                             0.0,
#                         )
#     MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(), objective_function)
#     MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)

#     # add constraint
#     MOI.add_constraint(
#         model,
#         MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(G[1, :], x), 0.0),
#         MOI.LessThan(h[1]),
#     )

#     @test_throws ErrorException MOI.optimize!(model) # should break
# end


@testset "Differentiating QP with inequality and equality constraints" begin
    # refered from: https://www.mathworks.com/help/optim/ug/quadprog.html#d120e113424
    # Find equivalent qpth program here - https://github.com/AKS1996/jump-gsoc-2020/blob/master/DiffOpt_tests_4_py.ipynb

    Q = [
         1.0 -1.0 1.0;
        -1.0  2.0 -2.0;
         1.0 -2.0 4.0
    ]
    q = [2.0, -3.0, 1.0]
    G = [0.0 0.0 1.0
         0.0 1.0 0.0
         1.0 0.0 0.0
         0.0 0.0 -1.0
         0.0 -1.0 0.0
         -1.0 0.0 0.0
    ]
    h = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0,]
    A = [1.0 1.0 1.0;]
    b = [0.5]

    model = diff_optimizer(Ipopt.Optimizer)
    MOI.set(model, MOI.Silent(), true)
    x = MOI.add_variables(model, 3)

    # define objective
    quad_terms = MOI.ScalarQuadraticTerm{Float64}[]
    for i in 1:3
        for j in i:3 # indexes (i,j), (j,i) will be mirrored. specify only one kind
            push!(
                quad_terms,
                MOI.ScalarQuadraticTerm(Q[i,j], x[i], x[j])
            )
        end
    end

    objective_function = MOI.ScalarQuadraticFunction(
                            MOI.ScalarAffineTerm.(q, x),
                            quad_terms,
                            0.0
                        )
    MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(), objective_function)
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    # add constraint
    for i in 1:6
        MOI.add_constraint(
            model,
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(G[i, :], x), 0.0),
            MOI.LessThan(h[i])
        )
    end

    for i in 1:1
        MOI.add_constraint(
            model,
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(A[i,:], x), 0.0),
            MOI.EqualTo(b[i])
        )
    end

    MOI.optimize!(model)

    z = model.primal_optimal

    @test z ≈ [0.0, 0.5, 0.0] atol=ATOL rtol=RTOL

    grads = backward(model, ["Q","q","G","h","A","b"], ones(3))

    dl_dQ = grads[1]
    dl_dq = grads[2]
    dl_dG = grads[3]
    dl_dh = grads[4]
    dl_dA = grads[5]
    dl_db = grads[6]

    @test dl_dQ ≈ zeros(3,3)  atol=ATOL rtol=RTOL

    @test dl_dq ≈ zeros(3,1) atol=ATOL rtol=RTOL

    @test dl_dG ≈ zeros(6,3) atol=ATOL rtol=RTOL

    @test dl_dh ≈ zeros(6,1) atol=ATOL rtol=RTOL

    @test dl_dA ≈ [0.0 -0.5 0.0] atol=ATOL rtol=RTOL

    @test dl_db ≈ [1.0] atol=ATOL rtol=RTOL
end



# refered from https://github.com/jump-dev/MathOptInterface.jl/blob/master/src/Test/contquadratic.jl#L3
# Find equivalent CVXPYLayers and QPTH code here:
#               https://github.com/AKS1996/jump-gsoc-2020/blob/master/DiffOpt_tests_1_py.ipynb
@testset "Differentiating MOI examples 1" begin
    # homogeneous quadratic objective
    # Min x^2 + xy + y^2 + yz + z^2
    # st  x + 2y + 3z >= 4 (c1)
    #     x +  y      >= 1 (c2)
    #     x, y, z \in R

    model = diff_optimizer(OSQP.Optimizer)
    MOI.set(model, MOI.Silent(), true)
    v = MOI.add_variables(model, 3)
    @test MOI.get(model, MOI.NumberOfVariables()) == 3

    c1 = MOI.add_constraint(
        model,
        MOI.ScalarAffineFunction(
            MOI.ScalarAffineTerm.([-1.0, -2.0, -3.0], v),
            0.0),
        MOI.LessThan(-4.0)
    )
    c2 = MOI.add_constraint(
        model,
        MOI.ScalarAffineFunction(
            MOI.ScalarAffineTerm.([-1.0, -1.0, 0.0], v),
            0.0),
        MOI.LessThan(-1.0)
    )

    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    @test MOI.get(model, MOI.ObjectiveSense()) == MOI.MIN_SENSE

    obj = MOI.ScalarQuadraticFunction(
        MOI.ScalarAffineTerm{Float64}[],
        MOI.ScalarQuadraticTerm.(
            [2.0, 1.0, 2.0, 1.0, 2.0],
            v[[1, 1, 2, 2, 3]],
            v[[1, 2, 2, 3, 3]]
        ),
        0.0,
    )
    MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(), obj)

    MOI.optimize!(model)

    z = model.primal_optimal

    @test z ≈ [4/7, 3/7, 6/7] atol=ATOL rtol=RTOL

    # obtain gradients
    grads = backward(model, ["Q","q","G","h"], ones(3))

    dl_dQ = grads[1]
    dl_dq = grads[2]
    dl_dG = grads[3]
    dl_dh = grads[4]

    @test dl_dQ ≈ [-0.12244895  0.01530609 -0.11224488;
                    0.01530609  0.09183674  0.07653058;
                   -0.11224488  0.07653058 -0.06122449]  atol=ATOL rtol=RTOL

    @test dl_dq ≈ [-0.2142857;  0.21428567; -0.07142857] atol=ATOL rtol=RTOL

    @test dl_dG ≈ [0.05102035   0.30612245  0.255102;
                   0.06122443   0.36734694  0.3061224] atol=ATOL rtol=RTOL

    @test dl_dh ≈ [-0.35714284; -0.4285714] atol=ATOL rtol=RTOL
end



# refered from https://github.com/jump-dev/MathOptInterface.jl/blob/master/src/Test/contquadratic.jl#L3
# Find equivalent CVXPYLayers and QPTH code here:
#               https://github.com/AKS1996/jump-gsoc-2020/blob/master/DiffOpt_tests_2_py.ipynb
@testset "Differentiating MOI examples 2 - non trivial backward pass vector" begin
    # non-homogeneous quadratic objective
    #    minimize 2 x^2 + y^2 + xy + x + y
    #       s.t.  x, y >= 0
    #             x + y = 1

    model = diff_optimizer(Ipopt.Optimizer)
    MOI.set(model, MOI.Silent(), true)

    x = MOI.add_variable(model)
    y = MOI.add_variable(model)

    c1 = MOI.add_constraint(
        model,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0,1.0], [x,y]), 0.0),
        MOI.EqualTo(1.0)
    )

    vc1 = MOI.add_constraint(
        model,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([-1.0,0.0], [x,y]), 0.0),
        MOI.LessThan(0.0)
    )
    @test vc1.value ≈ x.value atol=ATOL rtol=RTOL

    vc2 = MOI.add_constraint(
        model,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([0.0,-1.0], [x,y]), 0.0),
        MOI.LessThan(0.0)
    )
    @test vc2.value ≈ y.value atol=ATOL rtol=RTOL


    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    obj = MOI.ScalarQuadraticFunction(
        MOI.ScalarAffineTerm.([1.0, 1.0], [x, y]),
        MOI.ScalarQuadraticTerm.([4.0, 2.0, 1.0], [x, y, x], [x, y, y]),
        0.0
    )
    MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(), obj)

    MOI.optimize!(model)

    z = model.primal_optimal

    @test z ≈ [0.25, 0.75] atol=ATOL rtol=RTOL

    # obtain gradients
    dl_dz = [1.3, 0.5]   # choosing a non trivial backward pass vector
    grads = backward(model, ["Q", "q", "G", "h", "A", "b"], dl_dz)

    dl_dQ = grads[1]
    dl_dq = grads[2]
    dl_dG = grads[3]
    dl_dh = grads[4]
    dl_dA = grads[5]
    dl_db = grads[6]

    @test dl_dQ ≈ [-0.05   -0.05;
                   -0.05    0.15]  atol=ATOL rtol=RTOL

    @test dl_dq ≈ [-0.2; 0.2] atol=ATOL rtol=RTOL

    @test dl_dG ≈ [1e-8  1e-8; 1e-8 1e-8] atol=ATOL rtol=RTOL

    @test dl_dh ≈ [1e-8; 1e-8] atol=ATOL rtol=RTOL

    @test dl_dA ≈ [0.375 -1.075] atol=ATOL rtol=RTOL

    @test dl_db ≈ [0.7] atol=ATOL rtol=RTOL
end


@testset "Differentiating non trivial convex QP MOI" begin
    nz = 10
    nineq_le = 25
    neq = 10

    # read matrices from files
    names = ["P", "q", "G", "h", "A", "b"]
    matrices = []

    for name in names
        push!(matrices, readdlm(joinpath(dirname(dirname(pathof(DiffOpt))), "test", "data", name * ".txt"), ' ', Float64, '\n'))
    end

    Q, q, G, h, A, b = matrices
    q = vec(q)
    h = vec(h)
    b = vec(b)

    model = diff_optimizer(Ipopt.Optimizer)
    MOI.set(model, MOI.Silent(), true)

    x = MOI.add_variables(model, nz)

    # define objective
    quadratic_terms = MOI.ScalarQuadraticTerm{Float64}[]
    for i in 1:nz
        for j in i:nz # indexes (i,j), (j,i) will be mirrored. specify only one kind
            push!(quadratic_terms, MOI.ScalarQuadraticTerm(Q[i,j], x[i], x[j]))
        end
    end

    objective_function = MOI.ScalarQuadraticFunction(MOI.ScalarAffineTerm.(q, x), quadratic_terms, 0.0)
    MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(), objective_function)
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    # set constraints
    for i in 1:nineq_le
        MOI.add_constraint(
            model,
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(G[i,:], x), 0.0),MOI.LessThan(h[i])
        )
    end

    for i in 1:neq
        MOI.add_constraint(
            model,
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(A[i,:], x), 0.0), MOI.EqualTo(b[i]),
        )
    end

    MOI.optimize!(model)

    # obtain gradients
    grads = backward(model, ["Q", "q", "G", "h", "A", "b"], ones(nz))  # using dl_dz=[1,1,1,1,1,....]

    # read gradients from files
    names = ["dP", "dq", "dG", "dh", "dA", "db"]
    grads_actual = []

    for name in names
        push!(grads_actual, readdlm(Base.Filesystem.abspath(Base.Filesystem.joinpath("data",name*".txt")), ' ', Float64, '\n'))
    end

    grads_actual[2] = vec(grads_actual[2])
    grads_actual[4] = vec(grads_actual[4])
    grads_actual[6] = vec(grads_actual[6])

    # testing differences
    for i in 1:size(grads)[1]
        @test grads[i] ≈  grads_actual[i] atol=1e-2 rtol=1e-2
    end
end


@testset "Differentiating LP; checking gradients for non-active contraints" begin
    # Issue #40 from Gurobi.jl
    # min  x
    # s.t. x >= 0
    #      x >= 3

    model = diff_optimizer(Clp.Optimizer)
    MOI.set(model, MOI.Silent(), true)

    x = MOI.add_variables(model, 1)

    # define objective
    objective_function = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0], x), 0.0)
    MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), objective_function)
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    # set constraints
    MOI.add_constraint(
        model,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([-1.0], x), 0.),
        MOI.LessThan(0.0)
    )
    MOI.add_constraint(
        model,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([-1.0], x), 0.),
        MOI.LessThan(-3.0)
    )

    MOI.optimize!(model)

    # obtain gradients
    grads = backward(model, ["G", "h"], [1.0])

    @test grads[1] ≈ [0.0, 3.0] atol=ATOL rtol=RTOL
    @test grads[2] ≈ [0.0, -1.0] atol=ATOL rtol=RTOL
end


@testset "Differentiating a simple LP with GreaterThan constraint" begin
    # this is canonically same as above test
    # min  x
    # s.t. x >= 3
    model = diff_optimizer(Ipopt.Optimizer)
    MOI.set(model, MOI.Silent(), true)

    x = MOI.add_variables(model, 1)

    # define objective
    objective_function = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0], x), 0.0)
    MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), objective_function)
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    MOI.add_constraint(
        model,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0], x), 0.),
        MOI.GreaterThan(3.0),
    )

    MOI.optimize!(model)

    # obtain gradients
    grads = backward(model, ["G", "h"], [1.0])
    @test grads[1] ≈ [3.0] atol=ATOL rtol=RTOL
    @test grads[2] ≈ [-1.0] atol=ATOL rtol=RTOL   
end


@testset "Differentiating LP; checking gradients for non-active contraints" begin
    # refered from - https://en.wikipedia.org/wiki/Simplex_algorithm#Example

    # max 2x + 3y + 4z
    # s.t. 3x+2y+z <= 10
    #      2x+5y+3z <= 15
    #      x,y,z >= 0

    model = diff_optimizer(SCS.Optimizer)
    MOI.set(model, MOI.Silent(), true)
    v = MOI.add_variables(model, 3)

    # define objective
    objective_function = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([-2.0, -3.0, -4.0], v), 0.0)
    MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), objective_function)
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    # set constraints
    MOI.add_constraint(
        model,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([3.0, 2.0, 1.0], v), 0.),
        MOI.LessThan(10.0)
    )
    MOI.add_constraint(
        model,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([2.0, 5.0, 3.0], v), 0.),
        MOI.LessThan(15.0)
    )
    MOI.add_constraint(
        model,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([-1.0, 0.0, 0.0], v), 0.),
        MOI.LessThan(0.0)
    )
    MOI.add_constraint(
        model,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([0.0, -1.0, 0.0], v), 0.),
        MOI.LessThan(0.0)
    )
    MOI.add_constraint(
        model,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([0.0, 0.0, -1.0], v), 0.),
        MOI.LessThan(0.0)
    )

    MOI.optimize!(model)

    # obtain gradients
    grads = backward(model, ["Q", "q", "G", "h"], ones(3))  # using dl_dz=[1,1,1,1,1,....]

    @test grads[1] ≈ zeros(3,3) atol=ATOL rtol=RTOL
    @test grads[2] ≈ zeros(3) atol=ATOL rtol=RTOL
    @test grads[3] ≈ [0.0 0.0 0.0;
                    0.0 0.0 -5/3;
                    0.0 0.0 5/3;
                    0.0 0.0 -10/3;
                    0.0 0.0 0.0]   atol=ATOL rtol=RTOL
    @test grads[4] ≈ [0.0; 1/3; -1/3; 2/3; 0.0]   atol=ATOL rtol=RTOL
end

@testset "Differentiating LP with variable bounds" begin
    # max 2x + 3y + 4z
    # s.t. 3x+2y+z <= 10
    #      2x+5y+3z <= 15
    #      x ≤ 3
    #      0 ≤ y ≤ 2
    #      z ≥ 2
    #      x,y,z >= 0
    # variant of previous test with same solution

    model = diff_optimizer(GLPK.Optimizer)
    MOI.set(model, MOI.Silent(), true)
    v = MOI.add_variables(model, 3)

    # define objective
    objective_function = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([-2.0, -3.0, -4.0], v), 0.0)
    MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), objective_function)
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    # set constraints
    MOI.add_constraint(
        model,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([3.0, 2.0, 1.0], v), 0.),
        MOI.LessThan(10.0),
    )
    MOI.add_constraint(
        model,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([2.0, 5.0, 3.0], v), 0.),
        MOI.LessThan(15.0),
    )
    MOI.add_constraint(
        model,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([-1.0, 0.0, 0.0], v), 0.),
        MOI.LessThan(0.0),
    )
    MOI.add_constraint(
        model,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([0.0, -1.0, 0.0], v), 0.),
        MOI.LessThan(0.0),
    )
    MOI.add_constraint(
        model,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([0.0, 0.0, -1.0], v), 0.),
        MOI.LessThan(0.0),
    )
    #      x ≤ 3
    MOI.add_constraint(
        model,
        MOI.SingleVariable(v[1]),
        MOI.LessThan(3.0),
    )
    #      0 ≤ y ≤ 2
    MOI.add_constraint(
        model,
        MOI.SingleVariable(v[2]),
        MOI.LessThan(2.0),
    )
    #      z ≥ 2
    MOI.add_constraint(
        model,
        MOI.SingleVariable(v[3]),
        MOI.LessThan(6.0),
    )

    MOI.optimize!(model)

    # obtain gradients
    grads = backward(model, ["Q", "q", "G", "h"], ones(3))  # using dl_dz=[1,1,1]

    @test grads[1] ≈ zeros(3,3) atol=ATOL rtol=RTOL
    @test grads[2] ≈ zeros(3) atol=ATOL rtol=RTOL
    @test grads[3] ≈ [0.0 0.0 0.0;
                      0.0 0.0 -5/3;
                      0.0 0.0 5/3;
                      0.0 0.0 -10/3;
                      0.0 0.0 0.0
                      0.0 0.0 0.0
                      0.0 0.0 0.0
                      0.0 0.0 0.0                      
                      ]   atol=ATOL rtol=RTOL
    @test grads[4] ≈ [0.0, 1/3, -1/3, 2/3, 0.0, 0.0, 0.0, 0.0]   atol=ATOL rtol=RTOL

    model = diff_optimizer(GLPK.Optimizer)
    MOI.set(model, MOI.Silent(), true)
    v = MOI.add_variables(model, 3)

    # define objective
    objective_function = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([-2.0, -3.0, -4.0], v), 0.0)
    MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), objective_function)
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    # set constraints
    MOI.add_constraint(
        model,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([3.0, 2.0, 1.0], v), 0.0),
        MOI.LessThan(10.0),
    )
    MOI.add_constraint(
        model,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([2.0, 5.0, 3.0], v), 0.0),
        MOI.LessThan(15.0),
    )
    MOI.add_constraint(
        model,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([0.0, -1.0, 0.0], v), 0.0),
        MOI.LessThan(0.0),
    )
    MOI.add_constraint(
        model,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([0.0, 0.0, -1.0], v), 0.0),
        MOI.LessThan(0.0),
    )
    #      0 = x
    MOI.add_constraint(
        model,
        MOI.SingleVariable(v[1]),
        MOI.EqualTo(0.0),
    )

    MOI.optimize!(model)

    # obtain gradients
    grads = backward(model, ["Q", "q", "G", "h", "A", "b"], ones(3))  # using dl_dz=[1,1,1]

    @test grads[1] ≈ zeros(3,3) atol=ATOL rtol=RTOL
    @test grads[2] ≈ zeros(3) atol=ATOL rtol=RTOL
    @test grads[3] ≈ [0.0 0.0 0.0;
                      0.0 0.0 -5/3;
                      0.0 0.0 -10/3;
                      0.0 0.0 0.0
                      ]   atol=ATOL rtol=RTOL
    @test grads[4] ≈ [0.0, 1/3, 2/3, 0.0]   atol=ATOL rtol=RTOL
    @test grads[5] ≈ zeros(1, 3) .+ [0.0 0.0 -5/3]
    @test grads[6] ≈ [1/3]
end


@testset "Differentiating LP with SAF, SV with LE, GE constraints" begin
    """
        max  2x + 3y + 4z
        s.t. 3x+2y+z <= 10
             2x+5y+3z <= 15
             x ≤ 3
             0 ≤ y ≤ 2
             6 ≥ z ≥ -1
             x, y, z >= 0
        variant of previous test with same solution
    """
    model = diff_optimizer(GLPK.Optimizer)
    MOI.set(model, MOI.Silent(), true)
    v = MOI.add_variables(model, 3)

    # define objective
    objective_function = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([-2.0, -3.0, -4.0], v), 0.0)
    MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), objective_function)
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    # set constraints
    MOI.add_constraint(
        model,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([3.0, 2.0, 1.0], v), 0.),
        MOI.LessThan(10.0),
    )
    MOI.add_constraint(
        model,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([2.0, 5.0, 3.0], v), 0.),
        MOI.LessThan(15.0),
    )
    #      -x ≤ 0
    MOI.add_constraint(
        model,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([-1.0, 0.0, 0.0], v), 0.),
        MOI.LessThan(0.0),
    )
    #      -y ≤ 0
    MOI.add_constraint(
        model,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([0.0, -1.0, 0.0], v), 0.),
        MOI.LessThan(0.0),
    )
    #      0 ≤ z
    MOI.add_constraint(
        model,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([0.0, 0.0, 1.0], v), 0.),
        MOI.GreaterThan(0.0),
    )
    #      x ≤ 3
    MOI.add_constraint(
        model,
        MOI.SingleVariable(v[1]),
        MOI.LessThan(3.0),
    )
    #      y ≤ 2
    MOI.add_constraint(
        model,
        MOI.SingleVariable(v[2]),
        MOI.LessThan(2.0),
    )
    #      6 ≥ z
    MOI.add_constraint(
        model,
        MOI.SingleVariable(v[3]),
        MOI.LessThan(6.0),
    )
    #      z ≥ -1
    MOI.add_constraint(
        model,
        MOI.SingleVariable(v[3]),
        MOI.GreaterThan(-1.0),
    )

    MOI.optimize!(model)

    # obtain gradients
    grads = backward(model, ["Q", "q", "G", "h"], ones(3))  # using dl_dz=[1,1,1]

    @test grads[1] ≈ zeros(3,3) atol=ATOL rtol=RTOL
    @test grads[2] ≈ zeros(3) atol=ATOL rtol=RTOL
    @test grads[3] ≈ [0.0 0.0 0.0;
                      0.0 0.0 -5/3;
                      0.0 0.0 5/3;
                      0.0 0.0 -10/3;
                      0.0 0.0 0.0
                      0.0 0.0 0.0
                      0.0 0.0 0.0
                      0.0 0.0 0.0
                      0.0 0.0 0.0
                      ]   atol=ATOL rtol=RTOL
    @test grads[4] ≈ [0.0, 1/3, -1/3, 2/3, 0.0, 0.0, 0.0, 0.0, 0.0]   atol=ATOL rtol=RTOL
end


@testset "Differentiating simple SOCP" begin
    # referred from _soc2test, https://github.com/jump-dev/MathOptInterface.jl/blob/master/src/Test/contconic.jl#L1355
    # find equivalent diffcp python program here: https://github.com/AKS1996/jump-gsoc-2020/blob/master/diffcp_socp_1_py.ipynb

    # Problem SOC2
    # min  x
    # s.t. y ≥ 1/√2
    #      x² + y² ≤ 1
    # in conic form:
    # min  x
    # s.t.  -1/√2 + y ∈ R₊
    #        1 - t ∈ {0}
    #      (t,x,y) ∈ SOC₃

    model = diff_optimizer(SCS.Optimizer)
    MOI.set(model, MOI.Silent(), true)
    x,y,t = MOI.add_variables(model, 3)

    MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, x)], 0.0))
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    ceq  = MOI.add_constraint(model, MOI.VectorAffineFunction([MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(-1.0, t))], [1.0]), MOI.Zeros(1))
    cnon = MOI.add_constraint(model, MOI.VectorAffineFunction([MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, y))], [-1/√2]), MOI.Nonnegatives(1))
    csoc = MOI.add_constraint(model, MOI.VectorAffineFunction(MOI.VectorAffineTerm.([1,2,3], MOI.ScalarAffineTerm.(1.0, [t,x,y])), zeros(3)), MOI.SecondOrderCone(3))

    MOI.optimize!(model)

    x = model.primal_optimal

    cone_types = unique([S for (F, S) in MOI.get(model.optimizer, MOI.ListOfConstraints())])
    conic_form = MatOI.GeometricConicForm{Float64, MatOI.SparseMatrixCSRtoCSC{Float64, Int, MatOI.OneBasedIndexing}, Vector{Float64}}(cone_types)
    index_map = MOI.copy_to(conic_form, model)

    s = DiffOpt.map_rows((ci, r) -> MOI.get(model.optimizer, MOI.ConstraintPrimal(), ci), model.optimizer, conic_form, index_map, DiffOpt.Flattened{Float64}())
    y = DiffOpt.map_rows((ci, r) -> MOI.get(model.optimizer, MOI.ConstraintDual(), ci), model.optimizer, conic_form, index_map, DiffOpt.Flattened{Float64}())

    # these matrices are benchmarked with the output generated by diffcp
    # refer the python file mentioned above to get equivalent python source code
    @test x ≈ [-1/√2; 1/√2; 1.0] atol=ATOL rtol=RTOL
    @test s ≈ [0.0, 0.0, 1.0, -1/√2, 1/√2] atol=ATOL rtol=RTOL
    @test y ≈ [√2, 1.0, √2, 1.0, -1.0] atol=ATOL rtol=RTOL

    dA = zeros(5, 3)
    dA[1:3, :] .= Matrix(1.0I, 3, 3)
    db = zeros(5)
    dc = zeros(3)

    dx, dy, ds = backward(model, dA, db, dc)

    @test dx ≈ [1.12132144; 1/√2; 1/√2] atol=ATOL rtol=RTOL
    @test ds ≈ [0.0; 0.0; -2.92893438e-01;  1.12132144e+00; 7.07106999e-01]  atol=ATOL rtol=RTOL
    @test dy ≈ [2.4142175;   5.00000557;  3.8284315;   √2;   -4.00000495] atol=ATOL rtol=RTOL
end

# refered from _psd0test, https://github.com/jump-dev/MathOptInterface.jl/blob/master/src/Test/contconic.jl#L3919
# find equivalent diffcp program here: https://github.com/AKS1996/jump-gsoc-2020/blob/master/diffcp_sdp_1_py.ipynb

# min X[1,1] + X[2,2]    max y
#     X[2,1] = 1         [0   y/2     [ 1  0
#                         y/2 0    <=   0  1]
#     X >= 0              y free
# Optimal solution:
#     ⎛ 1   1 ⎞
# X = ⎜       ⎟           y = 2
#     ⎝ 1   1 ⎠
function simple_psd(solver)
    model = diff_optimizer(solver)
    MOI.set(model, MOI.Silent(), true)
    X = MOI.add_variables(model, 3)
    vov = MOI.VectorOfVariables(X)
    cX = MOI.add_constraint(
        model,
        MOI.VectorAffineFunction{Float64}(vov),
        MOI.PositiveSemidefiniteConeTriangle(2)
    )

    c  = MOI.add_constraint(
        model,
        MOI.VectorAffineFunction(
            [MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, X[2]))],
            [-1.0]
        ),
        MOI.Zeros(1)
    )

    MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(1.0, [X[1], X[end]]), 0.0))
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    MOI.optimize!(model)

    x = model.primal_optimal

    cone_types = unique([S for (F, S) in MOI.get(model.optimizer, MOI.ListOfConstraints())])
    conic_form = MatOI.GeometricConicForm{Float64, MatOI.SparseMatrixCSRtoCSC{Float64, Int, MatOI.OneBasedIndexing}, Vector{Float64}}(cone_types)
    index_map = MOI.copy_to(conic_form, model)

    s = DiffOpt.map_rows((ci, r) -> MOI.get(model.optimizer, MOI.ConstraintPrimal(), ci), model.optimizer, conic_form, index_map, DiffOpt.Flattened{Float64}())
    y = DiffOpt.map_rows((ci, r) -> MOI.get(model.optimizer, MOI.ConstraintDual(), ci), model.optimizer, conic_form, index_map, DiffOpt.Flattened{Float64}())

    @test x ≈ ones(3) atol=ATOL rtol=RTOL
    @test s ≈ [0.0; ones(3)] atol=ATOL rtol=RTOL
    @test y ≈ [2.0, 1.0, -1.0, 1.0]  atol=ATOL rtol=RTOL

    # test1: changing the constant in `c`, i.e. changing value of X[2]
    dA = zeros(4, 3)
    db = zeros(4)
    db[1] = 1.0
    dc = zeros(3)

    dx, dy, ds = backward(model, dA, db, dc)

    @test dx ≈ -ones(3) atol=ATOL rtol=RTOL  # will change the value of other 2 variables
    @test ds[2:4] ≈ -ones(3)  atol=ATOL rtol=RTOL  # will affect PSD constraint too

    # test2: changing X[1], X[3] but keeping the objective (their sum) same
    dA = zeros(4, 3)
    db = zeros(4)
    dc = zeros(3)
    dc[1] = -1.0
    dc[3] = 1.0

    dx, dy, ds = backward(model, dA, db, dc)

    @test dx ≈ [1.0, 0.0, -1.0] atol=ATOL rtol=RTOL  # note: no effect on X[2]
end

@testset "Differentiating simple PSD program" begin
    simple_psd(SCS.Optimizer)
end


@testset "Differentiating conic with PSD and SOC constraints" begin
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

    model = diff_optimizer(SCS.Optimizer)
    MOI.set(model, MOI.Silent(), true)

    δ = √(1 + (3*√2+2)*√(-116*√2+166) / 14) / 2
    ε = √((1 - 2*(√2-1)*δ^2) / (2-√2))
    y2 = 1 - ε*δ
    y1 = 1 - √2*y2
    obj = y1 + y2/2
    k = -2*δ/ε
    x2 = ((3-2obj)*(2+k^2)-4) / (4*(2+k^2)-4*√2)
    α = √(3-2obj-4x2)/2
    β = k*α

    X = MOI.add_variables(model, 6)
    x = MOI.add_variables(model, 3)

    vov = MOI.VectorOfVariables(X)
    cX = MOI.add_constraint(model, MOI.VectorAffineFunction{Float64}(vov), MOI.PositiveSemidefiniteConeTriangle(3))
    cx = MOI.add_constraint(model, MOI.VectorAffineFunction{Float64}(MOI.VectorOfVariables(x)), MOI.SecondOrderCone(3))

    c1 = MOI.add_constraint(
        model,
        MOI.VectorAffineFunction(
            MOI.VectorAffineTerm.(1:1, MOI.ScalarAffineTerm.([1., 1., 1., 1.], [X[1], X[3], X[end], x[1]])),
            [-1.0]
        ),
        MOI.Zeros(1)
    )
    c2 = MOI.add_constraint(
        model,
        MOI.VectorAffineFunction(
            MOI.VectorAffineTerm.(1:1, MOI.ScalarAffineTerm.([1., 2, 1, 2, 2, 1, 1, 1], [X; x[2]; x[3]])),
            [-0.5]
        ),
        MOI.Zeros(1),
    )

    # this is a useless constraint - refer the tests below
    # even if we comment this, it won't affect the optimal values
    c_extra = MOI.add_constraint(
        model,
        MOI.VectorAffineFunction(
            MOI.VectorAffineTerm.(1:1, MOI.ScalarAffineTerm.(ones(3), x)),
            [100.0]
        ),
        MOI.Nonnegatives(1)
    )

    objXidx = [1:3; 5:6]
    objXcoefs = 2*ones(5)
    MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
    MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([objXcoefs; 1.0], [X[objXidx]; x[1]]), 0.0))
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    MOI.optimize!(model)

    x = model.primal_optimal

    cone_types = unique([S for (F, S) in MOI.get(model.optimizer, MOI.ListOfConstraints())])
    conic_form = MatOI.GeometricConicForm{Float64, MatOI.SparseMatrixCSRtoCSC{Float64, Int, MatOI.OneBasedIndexing}, Vector{Float64}}(cone_types)
    index_map = MOI.copy_to(conic_form, model)

    s = DiffOpt.map_rows((ci, r) -> MOI.get(model.optimizer, MOI.ConstraintPrimal(), ci), model.optimizer, conic_form, index_map, DiffOpt.Flattened{Float64}())
    y = DiffOpt.map_rows((ci, r) -> MOI.get(model.optimizer, MOI.ConstraintDual(), ci), model.optimizer, conic_form, index_map, DiffOpt.Flattened{Float64}())

    @test x ≈ [ 0.21725121; -0.25996907;  0.31108582;  0.21725009; -0.25996907;  0.21725121;
                0.2544097;   0.17989425;  0.17989425] atol=ATOL rtol=RTOL
    @test s ≈ [
        0.0, 0.0, 100.614,
        0.254408, 0.179894, 0.179894,
        0.217251, -0.25997, 0.31109, 0.217251, -0.25997, 0.217251,
    ] atol=ATOL rtol=RTOL   # TODO: it should be 100, not 100.614, its surely a residual error
    @test y ≈ [
        0.544758, 0.321905, 0.0,
        0.455242, -0.321905, -0.321905,
        1.13334, 0.678095, 1.13334, -0.321905, 0.678095, 1.13334,
    ]  atol=ATOL rtol=RTOL

    # test c_extra
    dA = spzeros(12, 9)
    db = spzeros(12)
    db[3] = 1
    dc = spzeros(9)

    dx, dy, ds = backward(model, dA, db, dc)

    # a small change in the constant in c_extra should not affect any other variable or constraint other than c_extra itself
    @test dx ≈ zeros(9) atol=1e-2
    @test dy ≈ zeros(12) atol=0.012
    @test [ds[1:2]; ds[4:end]] ≈ zeros(11) atol=1e-2
    @test ds[3] ≈ 1.0 atol=1e-2   # except c_extra itself
end

@testset "Differentiating conic with PSD and POS constraints" begin
    # refer psdt2test, https://github.com/jump-dev/MathOptInterface.jl/blob/master/src/Test/contconic.jl#L4306
    # find equivalent diffcp program here - https://github.com/AKS1996/jump-gsoc-2020/blob/master/diffcp_sdp_3_py.ipynb

    model = diff_optimizer(SCS.Optimizer)
    MOI.set(model, MOI.Silent(), true)

    x = MOI.add_variables(model, 7)
    @test MOI.get(model, MOI.NumberOfVariables()) == 7

    η = 10.0

    c1  = MOI.add_constraint(
        model,
        MOI.VectorAffineFunction(
            MOI.VectorAffineTerm.(1, MOI.ScalarAffineTerm.(-1.0, x[1:6])),
            [η]
        ),
        MOI.Nonnegatives(1)
    )
    c2 = MOI.add_constraint(model, MOI.VectorAffineFunction(MOI.VectorAffineTerm.(1:6, MOI.ScalarAffineTerm.(1.0, x[1:6])), zeros(6)), MOI.Nonnegatives(6))
    α = 0.8
    δ = 0.9
    c3 = MOI.add_constraint(model, MOI.VectorAffineFunction(MOI.VectorAffineTerm.([fill(1, 7); fill(2, 5);     fill(3, 6)],
                                                            MOI.ScalarAffineTerm.(
                                                            [ δ/2,       α,   δ, δ/4, δ/8,      0.0, -1.0,
                                                                -δ/(2*√2), -δ/4, 0,     -δ/(8*√2), 0.0,
                                                                δ/2,     δ-α,   0,      δ/8,      δ/4, -1.0],
                                                            [x[1:7];     x[1:3]; x[5:6]; x[1:3]; x[5:7]])),
                                                            zeros(3)), MOI.PositiveSemidefiniteConeTriangle(2))
    c4 = MOI.add_constraint(
        model,
        MOI.VectorAffineFunction(
            MOI.VectorAffineTerm.(1, MOI.ScalarAffineTerm.(0.0, [x[1:3]; x[5:6]])),
            [0.0]
        ),
        MOI.Zeros(1)
    )

    MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, x[7])], 0.0))
    MOI.set(model, MOI.ObjectiveSense(), MOI.MAX_SENSE)

    MOI.optimize!(model)

    x = model.primal_optimal

    cone_types = unique([S for (F, S) in MOI.get(model.optimizer, MOI.ListOfConstraints())])
    conic_form = MatOI.GeometricConicForm{Float64, MatOI.SparseMatrixCSRtoCSC{Float64, Int, MatOI.OneBasedIndexing}, Vector{Float64}}(cone_types)
    index_map = MOI.copy_to(conic_form, model)

    s = DiffOpt.map_rows((ci, r) -> MOI.get(model.optimizer, MOI.ConstraintPrimal(), ci), model.optimizer, conic_form, index_map, DiffOpt.Flattened{Float64}())
    y = DiffOpt.map_rows((ci, r) -> MOI.get(model.optimizer, MOI.ConstraintDual(), ci), model.optimizer, conic_form, index_map, DiffOpt.Flattened{Float64}())

    @test x' ≈ [20/3. 0.0 10/3. 0.0 0.0 0.0 1.90192379] atol=ATOL rtol=RTOL
    @test s ≈ [0.0, 0.0, 20/3.0,  0.0,  10/3.0,  0.0,  0.0,  0.0, 4.09807621, -2.12132,  1.09807621] atol=ATOL rtol=RTOL
    @test y ≈ [0.0, 0.19019238, 0., 0.12597667, 0., 0.14264428, 0.14264428, 0.01274047, 0.21132487, 0.408248, 0.78867513] atol=ATOL rtol=RTOL

    dA = ones(11, 7)
    db = ones(11)
    dc = ones(7)

    dx, dy, ds = backward(model, dA, db, dc)

    atol = 0.3
    rtol = 0.01
    
    # compare these with https://github.com/AKS1996/jump-gsoc-2020/blob/master/diffcp_sdp_3_py.ipynb
    # results are not exactly as: 1. there is some residual error   2. diffcp results are SCS specific, hence scaled
    @test dx ≈ [-39.6066, 10.8953, -14.9189, 10.9054, 10.883, 10.9118, -21.7508] atol=atol rtol=rtol
    @test dy ≈ [0.0, -3.56905, 0.0, -0.380035, 0.0, -0.41398, -0.385321, -0.00743119, -0.644986, -0.550542, -2.36765] atol=atol rtol=rtol
    @test ds ≈ [0.0, 0.0, -50.4973, 0.0, -25.8066, 0.0, 0.0, 0.0, -7.96528, -1.62968, -2.18925] atol=atol rtol=rtol

    # TODO: future example, how to differentiate wrt a specific constraint/variable, refer QPLib article for more
    dA = zeros(11, 7)
    dA[3:8, 1:6] = Matrix{Float64}(LinearAlgebra.I, 6, 6)  # differentiating only wrt POS constraint c2
    db = zeros(11)
    dc = zeros(7)

    dx, dy, ds = backward(model, dA, db, dc)

    # note that there's no change in the PSD slack values or dual optimas
    @test dy ≈ [0.0, 0.0, 0.0, 0.125978, 0.0, 0.142644, 0.142641, 0.0127401, 0.0, 0.0, 0.0] atol=atol rtol=RTOL
    @test ds ≈ [0.0, 0.0, -6.66672, 0.0, -3.33336, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] atol=atol rtol=RTOL
end


@testset "Differentiating a simple PSD" begin
    # refer _psd3test, https://github.com/jump-dev/MathOptInterface.jl/blob/master/src/Test/contconic.jl#L4484
    # find equivalent diffcp program here - https://github.com/AKS1996/jump-gsoc-2020/blob/master/diffcp_sdp_0_py.ipynb

    # min x
    # s.t. [x 1 1]
    #      [1 x 1] ⪰ 0
    #      [1 1 x]

    model = DiffOpt.diff_optimizer(SCS.Optimizer)
    MOI.set(model, MOI.Silent(), true)

    x = MOI.add_variable(model)
    fx = MOI.SingleVariable(x)

    func = MOIU.operate(vcat, Float64, fx, one(Float64), fx, one(Float64), one(Float64), fx)

    # do not confuse this constraint with the matrix `c` in the conic form (of the matrices A, b, c)
    c = MOI.add_constraint(model, func, MOI.PositiveSemidefiniteConeTriangle(3))

    # MOI.set(model, MOI.ObjectiveFunction{MOI.SingleVariable}(), MOI.SingleVariable(x))
    MOI.set(
        model,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(1.0, [x]), 0.0)
    )
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    MOI.optimize!(model)

    x = model.primal_optimal

    cone_types = unique([S for (F, S) in MOI.get(model.optimizer, MOI.ListOfConstraints())])
    conic_form = MatOI.GeometricConicForm{Float64, MatOI.SparseMatrixCSRtoCSC{Float64, Int, MatOI.OneBasedIndexing}, Vector{Float64}}(cone_types)
    index_map = MOI.copy_to(conic_form, model)

    s = DiffOpt.map_rows((ci, r) -> MOI.get(model.optimizer, MOI.ConstraintPrimal(), ci), model.optimizer, conic_form, index_map, DiffOpt.Flattened{Float64}())
    y = DiffOpt.map_rows((ci, r) -> MOI.get(model.optimizer, MOI.ConstraintDual(), ci), model.optimizer, conic_form, index_map, DiffOpt.Flattened{Float64}())

    @test x ≈ [1.0] atol=ATOL rtol=RTOL
    @test s ≈ ones(6) atol=ATOL rtol=RTOL
    @test y ≈ [1/3, -1/6, 1/3, -1/6, -1/6, 1/3]  atol=ATOL rtol=RTOL

    # SCS/Mosek specific
    # @test s' ≈ [1.         1.41421356 1.41421356 1.         1.41421356 1.        ] atol=ATOL rtol=RTOL
    # @test y' ≈ [ 0.33333333 -0.23570226 -0.23570226  0.33333333 -0.23570226  0.33333333]  atol=ATOL rtol=RTOL

    dA = zeros(6, 1)
    db = ones(6)
    dc = zeros(1)

    dx, dy, ds = backward(model, dA, db, dc)

    @test dx ≈ [-0.5] atol=ATOL rtol=RTOL
    @test dy ≈ zeros(6) atol=ATOL rtol=RTOL
    @test ds ≈ [0.5, 1.0, 0.5, 1.0, 1.0, 0.5] atol=ATOL rtol=RTOL

    # test 2
    dA = zeros(6, 1)
    db = zeros(6)
    dc = ones(1)

    dx, dy, ds = backward(model, dA, db, dc)

    @test dx ≈ zeros(1) atol=ATOL rtol=RTOL
    @test dy ≈ [0.333333, -0.333333, 0.333333, -0.333333, -0.333333, 0.333333] atol=ATOL rtol=RTOL
    @test ds ≈ zeros(6) atol=ATOL rtol=RTOL
end

@testset "Verifying cache after differentiating a QP" begin
    Q = [
        4.0 1.0
        1.0 2.0
    ]
    q = [1.0, 1.0]
    G = [1.0 1.0]
    h = [-1.0]

    model = diff_optimizer(SCS.Optimizer)
    MOI.set(model, MOI.Silent(), true)
    x = MOI.add_variables(model, 2)

    # define objective
    quad_terms = MOI.ScalarQuadraticTerm{Float64}[]
    for i in 1:2
        for j in i:2 # indexes (i,j), (j,i) will be mirrored. specify only one kind
            push!(
                quad_terms,
                MOI.ScalarQuadraticTerm(Q[i,j], x[i], x[j])
            )
        end
    end
    objective_function = MOI.ScalarQuadraticFunction(
                            MOI.ScalarAffineTerm.(q, x),
                            quad_terms,
                            0.0,
                        )
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(), objective_function)

    # add constraint
    MOI.add_constraint(
        model,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(G[1, :], x), 0.0),
        MOI.LessThan(h[1]),
    )
    MOI.optimize!(model)
    
    @test model.primal_optimal ≈ [-0.25, -0.75] atol=ATOL rtol=RTOL
    @test model.gradient_cache === nothing
    grad_wrt_h = backward(model, ["h"], ones(2))[1]
    @test grad_wrt_h ≈ [1.0] atol=2ATOL rtol=RTOL
    @test model.gradient_cache !== nothing
    grad_wrt_h = backward(model, ["h"], ones(2))[1]
    @test grad_wrt_h ≈ [1.0] atol=2ATOL rtol=RTOL

    # adding two variables invalidates the cache
    y = MOI.add_variables(model, 2)
    MOI.delete(model, y)

    @test model.gradient_cache === nothing
    MOI.optimize!(model)
    grad_wrt_h = backward(model, ["h"], ones(2))[1]
    @test grad_wrt_h ≈ [1.0] atol=2ATOL rtol=RTOL
    @test model.gradient_cache isa DiffOpt.QPCache

    # adding single variable invalidates the cache
    y0 = MOI.add_variable(model)
    @test model.gradient_cache === nothing
    MOI.add_constraint(model, MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, y0)], 0.0), MOI.EqualTo(42.0))
    MOI.optimize!(model)
    @test model.gradient_cache === nothing

    grad_wrt_h = backward(model, ["h"], ones(3))[1]
    @test grad_wrt_h ≈ [1.0] atol=5e-3 rtol=RTOL
    @test model.gradient_cache isa DiffOpt.QPCache

    # adding constraint invalidates the cache
    MOI.add_constraint(
        model,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0, 1.0], x), 0.0),
        MOI.LessThan(0.0),
    )
    @test model.gradient_cache === nothing
    MOI.optimize!(model)
    grad_wrt_h = backward(model, ["h"], ones(3))[1]
    @test grad_wrt_h[1] ≈ 1.0 atol=5e-3 rtol=RTOL
    # second constraint inactive
    @test grad_wrt_h[2] ≈ 0.0 atol=5e-3 rtol=RTOL
    @test model.gradient_cache isa DiffOpt.QPCache
end


@testset "Verifying cache on a PSD" begin
    
    model = DiffOpt.diff_optimizer(SCS.Optimizer)
    MOI.set(model, MOI.Silent(), true)

    x = MOI.add_variable(model)
    fx = MOI.SingleVariable(x)

    func = MOIU.operate(vcat, Float64, fx, 1.0, fx, 1.0, 1.0, fx)

    c = MOI.add_constraint(model, func, MOI.PositiveSemidefiniteConeTriangle(3))

    # MOI.set(model, MOI.ObjectiveFunction{MOI.SingleVariable}(), MOI.SingleVariable(x))
    MOI.set(
        model,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(1.0, [x]), 0.0)
    )
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    @test model.gradient_cache === nothing
    MOI.optimize!(model)
    @test model.gradient_cache === nothing

    x = model.primal_optimal

    cone_types = unique([S for (F, S) in MOI.get(model.optimizer, MOI.ListOfConstraints())])
    conic_form = MatOI.GeometricConicForm{Float64, MatOI.SparseMatrixCSRtoCSC{Float64, Int, MatOI.OneBasedIndexing}, Vector{Float64}}(cone_types)
    index_map = MOI.copy_to(conic_form, model)

    s = DiffOpt.map_rows((ci, r) -> MOI.get(model.optimizer, MOI.ConstraintPrimal(), ci), model.optimizer, conic_form, index_map, DiffOpt.Flattened{Float64}())
    y = DiffOpt.map_rows((ci, r) -> MOI.get(model.optimizer, MOI.ConstraintDual(), ci), model.optimizer, conic_form, index_map, DiffOpt.Flattened{Float64}())

    @test x ≈ [1.0] atol=ATOL rtol=RTOL
    @test s ≈ ones(6) atol=ATOL rtol=RTOL
    @test y ≈ [1/3,  -1/6,  1/3,  -1/6,  -1/6,  1/3]  atol=ATOL rtol=RTOL

    dA = zeros(6, 1)
    db = ones(6)
    dc = zeros(1)

    dx, dy, ds = backward_conic(model, dA, db, dc)

    @test dx ≈ [-0.5] atol=ATOL rtol=RTOL
    @test dy ≈ zeros(6) atol=ATOL rtol=RTOL
    @test ds ≈ [0.5, 1.0, 0.5, 1.0, 1.0, 0.5] atol=ATOL rtol=RTOL

    @test model.gradient_cache isa DiffOpt.ConicCache

    dx2, dy2, ds2 = backward_conic(model, dA, db, dc)
    @test all(
        (dx2, dy2, ds2) .≈ (dx, dy, ds)
    )

    # test 2
    dA = zeros(6, 1)
    db = zeros(6)
    dc = ones(1)

    dx, dy, ds = backward_conic(model, dA, db, dc)

    @test dx ≈ zeros(1) atol=ATOL rtol=RTOL
    @test dy ≈ [0.333333, -0.333333, 0.333333, -0.333333, -0.333333, 0.333333] atol=ATOL rtol=RTOL
    @test ds ≈ zeros(6) atol=ATOL rtol=RTOL
end
