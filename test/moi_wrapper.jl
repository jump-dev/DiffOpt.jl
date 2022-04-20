using Test
import DiffOpt
import MathOptInterface
const MOI = MathOptInterface
const MOIU = MOI.Utilities
import DelimitedFiles
import Ipopt
import HiGHS
import SCS

const VAF = MOI.VectorAffineFunction{Float64}
_vaf(c::Vector{Float64}) = VAF(MOI.ScalarAffineTerm{Float64}[], c)

@testset "MOI Unit" begin
    function test_runtests()
        model = DiffOpt.diff_optimizer(HiGHS.Optimizer)
        # `Variable.ZerosBridge` makes dual needed by some tests fail.
        MOI.Bridges.remove_bridge(model.optimizer.optimizer, MOI.Bridges.Variable.ZerosBridge{Float64})
        MOI.set(model, MOI.Silent(), true)
        config = MOI.Test.Config(exclude = Any[MOI.SolverVersion])
        MOI.Test.runtests(model, config),
        return
    end
    test_runtests()
end

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
end

@testset "Differentiating trivial QP 1" begin
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
end

# @testset "Differentiating a non-convex QP" begin
#     Q = [0.0 0.0; 1.0 2.0]
#     q = [1.0; 1.0]
#     G = [1.0 1.0;]
#     h = [-1.0;]

#     model = DiffOpt.diff_optimizer(Ipopt.Optimizer)
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
#                             quad_terms,
#                             MOI.ScalarAffineTerm.(q, x),
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

    dQ = [-0.12244895  0.01530609 -0.11224488
           0.01530609  0.09183674  0.07653058
          -0.11224488  0.07653058 -0.06122449]
    dq = [-0.2142857,  0.21428567, -0.07142857]

    dG = [0.05102692 0.30612244 0.25510856
          0.06120519 0.36734693 0.30610315]
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
end

# refered from https://github.com/jump-dev/MathOptInterface.jl/blob/master/src/Test/contquadratic.jl#L3
# Find equivalent CVXPYLayers and QPTH code here:
#               https://github.com/AKS1996/jump-gsoc-2020/blob/master/DiffOpt_tests_2_py.ipynb
@testset "Differentiating MOI examples 2" begin
    # non-homogeneous quadratic objective
    #    minimize 2 x^2 + y^2 + xy + x + y
    #       s.t.  x, y >= 0
    #             x + y = 1

    Q = [4 1.0
         1 2]
    q = [1, 1.0]
    G = [-1 0.0
         0 -1]
    h = [0, 0.0]
    A = [1 1.0]
    b = [1.0]

    dQ = [-0.05   -0.05
          -0.05    0.15]
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
        ∇λb = [0.8, -0.8/3],
        ν = [-2.75],
        dνb = zeros(1),
        ∇νb = [-0.7],
    )
end

@testset "Differentiating non trivial convex QP MOI" begin
    nz = 10
    nineq_le = 25
    neq = 10

    # read matrices from files
    names = ["P", "q", "G", "h", "A", "b"]
    matrices = []

    for name in names
        push!(matrices, DelimitedFiles.readdlm(joinpath(dirname(dirname(pathof(DiffOpt))), "test", "data", name * ".txt"), ' ', Float64, '\n'))
    end

    Q, q, G, h, A, b = matrices
    q = vec(q)
    h = vec(h)
    b = vec(b)

    # read gradients from files
    names = ["dP", "dq", "dG", "dh", "dA", "db"]
    grads_actual = []

    for name in names
        push!(grads_actual, DelimitedFiles.readdlm(joinpath(@__DIR__, "data", name * ".txt"), ' ', Float64, '\n'))
    end

    dqb = vec(grads_actual[2])
    dhb = vec(grads_actual[4])
    dbb = vec(grads_actual[6])

    qp_test(
        Ipopt.Optimizer, true, true, true;
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
end

@testset "Differentiating LP; checking gradients for non-active contraints" begin
    # Issue #40 from Gurobi.jl
    # min  x
    # s.t. x >= 0
    #      x >= 3
    nz = 1
    qp_test_with_solutions(
        HiGHS.Optimizer;
        q = ones(nz),
        G = -ones(2, nz),
        h = [0.0, -3.0],
        dzb = ones(nz),
        dqf = ones(nz),
        # Expected solutions
        dGb = [0.0, 3.0],
        dhb = [0.0, -1.0],
    )
end

@testset "Differentiating a simple LP with GreaterThan constraint" begin
    # this is canonically same as above test
    # min  x
    # s.t. x >= 3
    nz = 1
    qp_test_with_solutions(
        Ipopt.Optimizer;
        q = ones(nz),
        G = -ones(1, nz),
        h = [-3.0],
        dzb = ones(nz),
        dqf = ones(nz),
        # Expected solutions
        dGb = [3.0],
        dhb = [-1.0],
    )
end

@testset "Differentiating LP; checking gradients for non-active contraints" begin
    # refered from - https://en.wikipedia.org/wiki/Simplex_algorithm#Example

    # max 2x + 3y + 4z
    # s.t. 3x+2y+z <= 10
    #      2x+5y+3z <= 15
    #      x,y,z >= 0
    nz = 3
    qp_test_with_solutions(
        SCS.Optimizer;
        q = [-2.0, -3.0, -4.0],
        G = [
            3.0  2.0  1.0
            2.0  5.0  3.0
            -1.0  0.0  0.0
            0.0 -1.0  0.0
            0.0  0.0 -1.0
        ],
        h = [10.0, 15.0, 0.0, 0.0, 0.0],
        dzb = ones(nz),
        dqf = ones(nz),
        # Expected solutions
        dqb = zeros(nz),
        dGb = [
            0.0 0.0 0.0
            0.0 0.0 -5/3
            0.0 0.0 5/3
            0.0 0.0 -10/3
            0.0 0.0 0.0
        ],
        dhb = [0.0, 1/3, -1/3, 2/3, 0.0],
    )
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
    nz = 3
    qp_test_with_solutions(
        HiGHS.Optimizer;
        q = [-2.0, -3.0, -4.0],
        G = [
            3.0  2.0  1.0
            2.0  5.0  3.0
           -1.0  0.0  0.0
            0.0 -1.0  0.0
            0.0  0.0 -1.0
            1.0  0.0  0.0
            0.0  1.0  0.0
            0.0  0.0  1.0
        ],
        h = [10.0, 15.0, 0.0, 0.0, 0.0, 3.0, 2.0, 6.0],
        dzb = ones(nz),
        dqf = ones(nz),
        # Expected solutions
        dqb = zeros(nz),
        dGb = [
            0.0 0.0 0.0
            0.0 0.0 -5/3
            0.0 0.0 5/3
            0.0 0.0 -10/3
            0.0 0.0 0.0
            0.0 0.0 0.0
            0.0 0.0 0.0
            0.0 0.0 0.0
        ],
        dhb = [0.0, 1/3, -1/3, 2/3, 0.0, 0.0, 0.0, 0.0],
    )
end

@testset "Differentiating LP with variable bounds 2" begin
    nz = 3
    qp_test_with_solutions(
        HiGHS.Optimizer;
        q = [-2.0, -3.0, -4.0],
        G = [
            3.0  2.0  1.0
            2.0  5.0  3.0
            0.0 -1.0  0.0
            0.0  0.0 -1.0
        ],
        h = [10.0, 15.0, 0.0, 0.0],
        fix_indices = [1],
        fix_values = [0.0],
        dzb = ones(nz),
        dqf = ones(nz),
        # Expected solutions
        dqb = zeros(nz),
        dGb = [0.0 0.0 0.0
               0.0 0.0 -5/3
               0.0 0.0 -10/3
               0.0 0.0 0.0],
        dhb = [0.0, 1/3, 2/3, 0.0],
        dAb = [0.0 0.0 -5/3],
        dbb = [1/3],
    )
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
    nz = 3
    qp_test_with_solutions(
        HiGHS.Optimizer;
        q = [-2.0, -3.0, -4.0],
        G = [
            3.0  2.0  1.0
            2.0  5.0  3.0
           -1.0  0.0  0.0
            0.0 -1.0  0.0
            0.0  0.0 -1.0
        ],
        h = [10.0, 15.0, 0.0, 0.0, 0.0], #5
        ub_indices = [1, 2, 3],
        ub_values = [3.0, 2.0, 6.0],
        lb_indices = [1],
        lb_values = [-1.0],
        dzb = ones(nz),
        dqb = zeros(nz),
        dGb = [0.0 0.0 0.0;
               0.0 0.0 -5/3;
               0.0 0.0 5/3;
               0.0 0.0 -10/3;
               0.0 0.0 0.0
               0.0 0.0 0.0
               0.0 0.0 0.0
               0.0 0.0 0.0
               0.0 0.0 0.0],
        dhb = [0.0, 1/3, -1/3, 2/3, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
end



# TODO: split file here
# above is QP Back
# below is conic forw

function test_simple_socp(eq_vec::Bool)
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

    model = DiffOpt.diff_optimizer(SCS.Optimizer)
    MOI.set(model, MOI.Silent(), true)
    x,y,t = MOI.add_variables(model, 3)

    MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), 1.0x)
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    if eq_vec
        ceq  = MOI.add_constraint(model, MOIU.vectorize([-1.0t + 1.0]), MOI.Zeros(1))
    else
        ceq  = MOI.add_constraint(model, -1.0t, MOI.EqualTo(-1.0))
    end
    cnon = MOI.add_constraint(model, MOIU.vectorize([1.0y - 1/√2]), MOI.Nonnegatives(1))
    csoc = MOI.add_constraint(model, MOIU.vectorize([1.0t, 1.0x, 1.0y]), MOI.SecondOrderCone(3))

    MOI.optimize!(model)

    if eq_vec
        MOI.set(model, DiffOpt.ForwardConstraintFunction(), ceq, MOIU.vectorize([1.0 * x]))
    else
        MOI.set(model, DiffOpt.ForwardConstraintFunction(), ceq, 1.0 * x)
    end
    MOI.set(model, DiffOpt.ForwardConstraintFunction(), cnon, MOIU.vectorize([1.0 * y]))
    MOI.set(model, DiffOpt.ForwardConstraintFunction(), csoc, MOIU.operate(vcat, Float64, 1.0 * t, 0.0, 0.0))

    DiffOpt.forward_differentiate!(model)

    # these matrices are benchmarked with the output generated by diffcp
    # refer the python file mentioned above to get equivalent python source code
    @test model.diff.model.x ≈ [-1/√2; 1/√2; 1.0] atol=ATOL rtol=RTOL
    if eq_vec
        @test model.diff.model.s ≈ [0.0, 0.0, 1.0, -1/√2, 1/√2] atol=ATOL rtol=RTOL
        @test model.diff.model.y ≈ [√2, 1.0, √2, 1.0, -1.0] atol=ATOL rtol=RTOL
    else
        @test model.diff.model.s ≈ [0.0, 1.0, -1/√2, 1/√2, 0.0] atol=ATOL rtol=RTOL
        @test model.diff.model.y ≈ [1.0, √2, 1.0, -1.0, √2] atol=ATOL rtol=RTOL
    end

    dx = [1.12132144; 1/√2; 1/√2]
    for (i, vi) in enumerate([x, y, t])
        @test dx[i] ≈ MOI.get(model, DiffOpt.ForwardVariablePrimal(), vi) atol=ATOL rtol=RTOL
    end
    # @test dx ≈ [1.12132144; 1/√2; 1/√2] atol=ATOL rtol=RTOL
    # @test ds ≈ [0.0; 0.0; -2.92893438e-01;  1.12132144e+00; 7.07106999e-01]  atol=ATOL rtol=RTOL
    # @test dy ≈ [2.4142175; 5.00000557; 3.8284315; √2; -4.00000495] atol=ATOL rtol=RTOL
end

@testset "Differentiating simple SOCP" begin
    for eq_vec in [false, true]
        test_simple_socp(eq_vec)
    end
end

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
function simple_psd(solver)
    model = DiffOpt.diff_optimizer(solver)
    MOI.set(model, MOI.Silent(), true)
    X = MOI.add_variables(model, 3)
    vov = MOI.VectorOfVariables(X)
    cX = MOI.add_constraint(
        model,
        VAF(vov),
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
            1.0 * X[1] + 1.0 * X[end])
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    MOI.optimize!(model)

    x = MOI.get(model, MOI.VariablePrimal(), X)

    cone_types = unique([S for (F, S) in MOI.get(model.optimizer, MOI.ListOfConstraintTypesPresent())])
    conic_form = DiffOpt.GeometricConicForm{Float64}()
    cones = conic_form.constraints.sets
    DiffOpt.set_set_types(cones, cone_types)
    index_map = MOI.copy_to(conic_form, model)

    # s = DiffOpt.map_rows((ci, r) -> MOI.get(model.optimizer, MOI.ConstraintPrimal(), ci), model.optimizer, cones, index_map, DiffOpt.Flattened{Float64}())
    # y = DiffOpt.map_rows((ci, r) -> MOI.get(model.optimizer, MOI.ConstraintDual(), ci), model.optimizer, cones, index_map, DiffOpt.Flattened{Float64}())

    @test x ≈ ones(3) atol=ATOL rtol=RTOL
    # @test s ≈ [0.0; ones(3)] atol=ATOL rtol=RTOL
    # @test y ≈ [2.0, 1.0, -1.0, 1.0]  atol=ATOL rtol=RTOL

    # test1: changing the constant in `c`, i.e. changing value of X[2]
    MOI.set(model, DiffOpt.ForwardConstraintFunction(), c, _vaf([1.0]))

    DiffOpt.forward_differentiate!(model)

    dx = -ones(3)
    for (i, vi) in enumerate(X)
        @test dx[i] ≈ MOI.get(model, DiffOpt.ForwardVariablePrimal(), vi) atol=ATOL rtol=RTOL
    end

    # @test dx ≈ -ones(3) atol=ATOL rtol=RTOL  # will change the value of other 2 variables
    # @test ds[2:4] ≈ -ones(3)  atol=ATOL rtol=RTOL  # will affect PSD constraint too

    # test2: changing X[1], X[3] but keeping the objective (their sum) same
    MOI.set(model, DiffOpt.ForwardConstraintFunction(), c, MOIU.zero_with_output_dimension(MOI.VectorAffineFunction{Float64}, 1))
    MOI.set(model, DiffOpt.ForwardObjective(), -1.0X[1] + 1.0X[3])

    DiffOpt.forward_differentiate!(model)

    # @test dx ≈ [1.0, 0.0, -1.0] atol=ATOL rtol=RTOL  # note: no effect on X[2]
    dx = [1.0, 0.0, -1.0]
    for (i, vi) in enumerate(X)
        @test dx[i] ≈ MOI.get(model, DiffOpt.ForwardVariablePrimal(), vi) atol=ATOL rtol=RTOL
    end
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

    model = DiffOpt.diff_optimizer(SCS.Optimizer)
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
    cX = MOI.add_constraint(model,
        VAF(vov),
        MOI.PositiveSemidefiniteConeTriangle(3))
    cx = MOI.add_constraint(model,
        VAF(MOI.VectorOfVariables(x)),
        MOI.SecondOrderCone(3))

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

    _x = MOI.get(model, MOI.VariablePrimal(), x)
    _X = MOI.get(model, MOI.VariablePrimal(), X)

    cone_types = unique([S for (F, S) in MOI.get(model.optimizer, MOI.ListOfConstraintTypesPresent())])
    conic_form = DiffOpt.GeometricConicForm{Float64}()
    cones = conic_form.constraints.sets
    DiffOpt.set_set_types(cones, cone_types)
    index_map = MOI.copy_to(conic_form, model)

    # s = DiffOpt.map_rows((ci, r) -> MOI.get(model.optimizer, MOI.ConstraintPrimal(), ci), model.optimizer, cones, index_map, DiffOpt.Flattened{Float64}())
    # y = DiffOpt.map_rows((ci, r) -> MOI.get(model.optimizer, MOI.ConstraintDual(), ci), model.optimizer, cones, index_map, DiffOpt.Flattened{Float64}())

    @test _X ≈ [0.21725121; -0.25996907;  0.31108582;  0.21725009; -0.25996907;  0.21725121
        ] atol=ATOL rtol=RTOL
    @test _x ≈ [0.2544097;   0.17989425;  0.17989425
        ] atol=ATOL rtol=RTOL

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
    MOI.set(model, DiffOpt.ForwardConstraintFunction(), c_extra, _vaf([1.0]))

    DiffOpt.forward_differentiate!(model)

    # a small change in the constant in c_extra should not affect any other variable or constraint other than c_extra itself
    for (i, vi) in enumerate(X)
        @test 0.0 ≈ MOI.get(model,
            DiffOpt.ForwardVariablePrimal(), vi) atol=1e-2 rtol=RTOL
    end
    for (i, vi) in enumerate(x)
        @test 0.0 ≈ MOI.get(model,
            DiffOpt.ForwardVariablePrimal(), vi) atol=1e-2 rtol=RTOL
    end
    # @test dx ≈ zeros(9) atol=1e-2
    # @test dy ≈ zeros(12) atol=0.012
    # @test [ds[1:2]; ds[4:end]] ≈ zeros(11) atol=1e-2
    # @test ds[3] ≈ 1.0 atol=1e-2   # except c_extra itself
end

@testset "Differentiating conic with PSD and POS constraints" begin
    # refer psdt2test, https://github.com/jump-dev/MathOptInterface.jl/blob/master/src/Test/contconic.jl#L4306
    # find equivalent diffcp program here - https://github.com/AKS1996/jump-gsoc-2020/blob/master/diffcp_sdp_3_py.ipynb

    model = DiffOpt.diff_optimizer(SCS.Optimizer)
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
    c2 = MOI.add_constraint(
        model,
        MOI.VectorAffineFunction(
            MOI.VectorAffineTerm.(
                1:6, MOI.ScalarAffineTerm.(1.0, x[1:6])), zeros(6)),
            MOI.Nonnegatives(6))
    α = 0.8
    δ = 0.9
    c3 = MOI.add_constraint(
        model,
        MOI.VectorAffineFunction(
            MOI.VectorAffineTerm.(
                [fill(1, 7); fill(2, 5);     fill(3, 6)],
                MOI.ScalarAffineTerm.(
                    [ δ/2,       α,   δ, δ/4, δ/8,      0.0, -1.0,
                        -δ/(2*√2), -δ/4, 0,     -δ/(8*√2), 0.0,
                        δ/2,     δ-α,   0,      δ/8,      δ/4, -1.0],
                    [x[1:7];     x[1:3]; x[5:6]; x[1:3]; x[5:7]])),
                zeros(3)), MOI.PositiveSemidefiniteConeTriangle(2))
    c4 = MOI.add_constraint(
        model,
        MOI.VectorAffineFunction(
            MOI.VectorAffineTerm.(1,
                MOI.ScalarAffineTerm.(0.0, [x[1:3]; x[5:6]])),
            [0.0]
        ),
        MOI.Zeros(1)
    )

    MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, x[7])], 0.0))
    MOI.set(model, MOI.ObjectiveSense(), MOI.MAX_SENSE)

    MOI.optimize!(model)

    # dc = ones(7)
    MOI.set(model, DiffOpt.ForwardObjective(), MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(7), x), 0.0))
    # db = ones(11)
    # dA = ones(11, 7)
    MOI.set(model,
        DiffOpt.ForwardConstraintFunction(), c1, MOIU.vectorize(ones(1, 7) * x + ones(1)))
    MOI.set(model,
        DiffOpt.ForwardConstraintFunction(), c2, MOIU.vectorize(ones(6, 7) * x + ones(6)))
    MOI.set(model,
        DiffOpt.ForwardConstraintFunction(), c3, MOIU.vectorize(ones(3, 7) * x + ones(3)))
    MOI.set(model,
        DiffOpt.ForwardConstraintFunction(), c4, MOIU.vectorize(ones(1, 7) * x + ones(1)))

    DiffOpt.forward_differentiate!(model)

    @test model.diff.model.x ≈ [20/3., 0.0, 10/3., 0.0, 0.0, 0.0, 1.90192379] atol=ATOL rtol=RTOL
    @test model.diff.model.s ≈ [0.0, 0.0, 20/3.0,  0.0,  10/3.0,  0.0,  0.0,  0.0, 4.09807621, -2.12132,  1.09807621] atol=ATOL rtol=RTOL
    @test model.diff.model.y ≈ [0.0, 0.19019238, 0., 0.12597667, 0., 0.14264428, 0.14264428, 0.01274047, 0.21132487, 0.408248, 0.78867513] atol=ATOL rtol=RTOL

    atol = 0.3
    rtol = 0.01

    # compare these with https://github.com/AKS1996/jump-gsoc-2020/blob/master/diffcp_sdp_3_py.ipynb
    # results are not exactly as: 1. there is some residual error   2. diffcp results are SCS specific, hence scaled
    dx = [-39.6066, 10.8953, -14.9189, 10.9054, 10.883, 10.9118, -21.7508]
    for (i, vi) in enumerate(x)
        @test dx[i] ≈ MOI.get(model,
            DiffOpt.ForwardVariablePrimal(), vi) atol=atol rtol=rtol
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
    MOI.set(model,
        DiffOpt.ForwardConstraintFunction(), c1, MOIU.zero_with_output_dimension(VAF, 1))
    MOI.set(model,
        DiffOpt.ForwardConstraintFunction(), c2, MOIU.vectorize(ones(6) .* x[1:6]))
    MOI.set(model,
        DiffOpt.ForwardConstraintFunction(), c3, MOIU.zero_with_output_dimension(VAF, 3))
    MOI.set(model,
        DiffOpt.ForwardConstraintFunction(), c4, MOIU.zero_with_output_dimension(VAF, 1))

    DiffOpt.forward_differentiate!(model)

    # for (i, vi) in enumerate(X)
    #     @test 0.0 ≈ MOI.get(model,
    #         DiffOpt.ForwardVariablePrimal(), vi) atol=1e-2 rtol=RTOL
    # end

    # TODO add a test here, probably on duals

    # # note that there's no change in the PSD slack values or dual optimas
    # @test dy ≈ [0.0, 0.0, 0.0, 0.125978, 0.0, 0.142644, 0.142641, 0.0127401, 0.0, 0.0, 0.0] atol=atol rtol=RTOL
    # @test ds ≈ [0.0, 0.0, -6.66672, 0.0, -3.33336, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] atol=atol rtol=RTOL
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

    func = MOIU.operate(vcat, Float64, x, one(Float64), x, one(Float64), one(Float64), x)

    # do not confuse this constraint with the matrix `c` in the conic form (of the matrices A, b, c)
    c = MOI.add_constraint(model, func, MOI.PositiveSemidefiniteConeTriangle(3))

    MOI.set(
        model,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(1.0, [x]), 0.0)
    )
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    MOI.optimize!(model)

    # SCS/Mosek specific
    # @test s' ≈ [1.         1.41421356 1.41421356 1.         1.41421356 1.        ] atol=ATOL rtol=RTOL
    # @test y' ≈ [ 0.33333333 -0.23570226 -0.23570226  0.33333333 -0.23570226  0.33333333]  atol=ATOL rtol=RTOL

    # dA = zeros(6, 1)
    # db = ones(6)
    MOI.set(model, DiffOpt.ForwardConstraintFunction(), c, _vaf(ones(6)))
    # dc = zeros(1)


    DiffOpt.forward_differentiate!(model)

    @test model.diff.model.x ≈ [1.0] atol=10ATOL rtol=10RTOL
    @test model.diff.model.s ≈ ones(6) atol=ATOL rtol=RTOL
    @test model.diff.model.y ≈ [1/3, -1/6, 1/3, -1/6, -1/6, 1/3]  atol=ATOL rtol=RTOL

    @test -0.5 ≈ MOI.get(model,
        DiffOpt.ForwardVariablePrimal(), x) atol=1e-2 rtol=RTOL

    # @test dx ≈ [-0.5] atol=ATOL rtol=RTOL
    # @test dy ≈ zeros(6) atol=ATOL rtol=RTOL
    # @test ds ≈ [0.5, 1.0, 0.5, 1.0, 1.0, 0.5] atol=ATOL rtol=RTOL

    # test 2
    dA = zeros(6, 1)
    db = zeros(6)
    MOI.set(model, DiffOpt.ForwardConstraintFunction(), c, MOIU.zero_with_output_dimension(VAF, 6))
    MOI.set(model, DiffOpt.ForwardObjective(), 1.0 * x)

    DiffOpt.forward_differentiate!(model)

    @test 0.0 ≈ MOI.get(model,
        DiffOpt.ForwardVariablePrimal(), x) atol=1e-2 rtol=RTOL

    # @test dx ≈ zeros(1) atol=ATOL rtol=RTOL
    # @test dy ≈ [0.333333, -0.333333, 0.333333, -0.333333, -0.333333, 0.333333] atol=ATOL rtol=RTOL
    # @test ds ≈ zeros(6) atol=ATOL rtol=RTOL
end

@testset "Verifying cache after differentiating a QP" begin
    Q = [
        4.0 1.0
        1.0 2.0
    ]
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
            push!(
                quad_terms,
                MOI.ScalarQuadraticTerm(Q[i,j], x[i], x[j])
            )
        end
    end
    objective_function = MOI.ScalarQuadraticFunction(
        quad_terms,
        MOI.ScalarAffineTerm.(q, x),
        0.0,
    )
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(), objective_function)

    # add constraint
    c = MOI.add_constraint(
        model,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(G[1, :], x), 0.0),
        MOI.LessThan(h[1]),
    )
    MOI.optimize!(model)

    x_sol = MOI.get(model, MOI.VariablePrimal(), x)
    @test x_sol ≈ [-0.25, -0.75] atol=ATOL rtol=RTOL

    MOI.set.(model, DiffOpt.ReverseVariablePrimal(), x, ones(2))

    @test model.diff === nothing
    DiffOpt.reverse_differentiate!(model)

    grad_wrt_h = MOI.constant(MOI.get(model, DiffOpt.ReverseConstraintFunction(), c))
    @test grad_wrt_h ≈ -1.0 atol=2ATOL rtol=RTOL

    # adding two variables invalidates the cache
    y = MOI.add_variables(model, 2)
    MOI.delete(model, y)

    @test model.diff === nothing
    MOI.optimize!(model)

    MOI.set.(model, DiffOpt.ReverseVariablePrimal(), x, ones(2))
    DiffOpt.reverse_differentiate!(model)

    grad_wrt_h = MOI.constant(MOI.get(model, DiffOpt.ReverseConstraintFunction(), c))
    @test grad_wrt_h ≈ -1.0 atol=2ATOL rtol=RTOL

    # adding single variable invalidates the cache
    y0 = MOI.add_variable(model)
    @test model.diff === nothing
    MOI.add_constraint(model, MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, y0)], 0.0), MOI.EqualTo(42.0))

    MOI.optimize!(model)
    @test model.diff === nothing

    MOI.set.(model, DiffOpt.ReverseVariablePrimal(), x, ones(2))
    DiffOpt.reverse_differentiate!(model)
    grad_wrt_h = MOI.constant(MOI.get(model, DiffOpt.ReverseConstraintFunction(), c))
    @test grad_wrt_h ≈ -1.0 atol=5e-3 rtol=RTOL
    @test model.diff.model.gradient_cache isa DiffOpt.QPCache

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
    grad_wrt_h = MOI.constant(MOI.get(model, DiffOpt.ReverseConstraintFunction(), c))
    @test grad_wrt_h ≈ -1.0 atol=5e-3 rtol=RTOL
    # second constraint inactive
    grad_wrt_h = MOI.constant(MOI.get(model, DiffOpt.ReverseConstraintFunction(), c2))
    @test grad_wrt_h ≈ 0.0 atol=5e-3 rtol=RTOL
    @test model.diff.model.gradient_cache isa DiffOpt.QPCache
end

@testset "Verifying cache on a PSD" begin

    model = DiffOpt.diff_optimizer(SCS.Optimizer)
    MOI.set(model, MOI.Silent(), true)

    x = MOI.add_variable(model)

    func = MOIU.operate(vcat, Float64, x, 1.0, x, 1.0, 1.0, x)

    c = MOI.add_constraint(model, func, MOI.PositiveSemidefiniteConeTriangle(3))

    MOI.set(
        model,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(1.0, [x]), 0.0)
    )
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    @test model.diff === nothing
    MOI.optimize!(model)
    @test model.diff === nothing

    dA = zeros(6, 1)
    db = ones(6)
    MOI.set(model, DiffOpt.ForwardConstraintFunction(), c, _vaf(ones(6)))
    dc = zeros(1)

    DiffOpt.forward_differentiate!(model)

    @test model.diff.model.x ≈ [1.0] atol=10ATOL rtol=10RTOL
    @test model.diff.model.s ≈ ones(6) atol=ATOL rtol=RTOL
    @test model.diff.model.y ≈ [1/3,  -1/6,  1/3,  -1/6,  -1/6,  1/3]  atol=ATOL rtol=RTOL

    @test -0.5 ≈ MOI.get(model,
    DiffOpt.ForwardVariablePrimal(), x) atol=1e-2 rtol=RTOL

    @test model.diff.model.gradient_cache isa DiffOpt.ConicCache

    DiffOpt.forward_differentiate!(model)

    @test -0.5 ≈ MOI.get(model,
        DiffOpt.ForwardVariablePrimal(), x) atol=1e-2 rtol=RTOL

    # test 2
    MOI.set(model, DiffOpt.ForwardConstraintFunction(), c, _vaf(zeros(6)))
    MOI.set(model, DiffOpt.ForwardObjective(), 1.0 * x)

    DiffOpt.forward_differentiate!(model)

    @test 0.0 ≈ MOI.get(model,
        DiffOpt.ForwardVariablePrimal(), x) atol=1e-2 rtol=RTOL

end
