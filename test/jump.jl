using Test
using JuMP
import DiffOpt
import MathOptInterface as MOI
const MOIU = MOI.Utilities
import LinearAlgebra: dot, ⋅, I
import Ipopt
import Ipopt
import HiGHS
import SCS
import DelimitedFiles

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

    model = JuMP.Model(() -> DiffOpt.diff_optimizer(Ipopt.Optimizer))
    MOI.set(model, MOI.Silent(), true)
    @variable(model, x[1:2])
    @objective(model, Min, dot(Q * x, x) + dot(q, x))
    @constraint(model,
        G * x .<= h,
    )
    optimize!(model)

    @test JuMP.value.(x) ≈ [0.3, 0.7] atol=ATOL rtol=RTOL
end



@testset "Differentiating trivial QP 1" begin
    Q = [
        4.0 1.0
        1.0 2.0
    ]
    q = [1.0, 1.0]
    G = [1.0 1.0]
    h = [-1.0]

    model = JuMP.direct_model(DiffOpt.diff_optimizer(Ipopt.Optimizer))
    MOI.set(model, MOI.Silent(), true)
    x = @variable(model, [1:2])
    @objective(model, Min, dot(Q * x, x) + dot(q, x))
    @constraint(model, ctr_le,
        G * x .<= h,
    )
    optimize!(model)

    @test JuMP.value.(x) ≈ [-0.25, -0.75] atol=ATOL rtol=RTOL

    MOI.set.(model, DiffOpt.ReverseVariablePrimal(), x, 1.0)

    DiffOpt.reverse_differentiate!(model)

    DiffOpt.reverse_differentiate!(model)

    grad_constraint = JuMP.constant(MOI.get(model, DiffOpt.ReverseConstraintFunction(), ctr_le[1]))
    @test grad_constraint ≈ -1.0  atol=ATOL rtol=RTOL

    # Test some overloads from https://github.com/jump-dev/DiffOpt.jl/issues/211
    grad_obj = MOI.get(model, DiffOpt.ReverseObjectiveFunction())
    @test JuMP.coefficient(grad_obj, x[1], x[2]) ≈ 
        DiffOpt.quad_sym_half.(grad_obj, x[1], x[2]) atol=ATOL rtol=RTOL

    @test DiffOpt.quad_sym_half(grad_obj, x[1], x[1]) ≈ 
        2 * JuMP.coefficient(grad_obj, x[1], x[1]) atol=ATOL rtol=RTOL

    # TODO: this simple show fails
    @show ctr_le
end

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

    model = Model(() -> DiffOpt.diff_optimizer(Ipopt.Optimizer))
    MOI.set(model, MOI.Silent(), true)
    @variable(model, x[1:3])
    @objective(model, Min, dot(Q * x, x) + dot(q, x))
    @constraint(model, ctr_le,
        G * x .<= h,
    )
    @constraint(model, ctr_eq,
        A * x .== b,
    )
    optimize!(model)

    @test JuMP.value.(x) ≈ [0.0, 0.5, 0.0] atol=ATOL rtol=RTOL

    MOI.set.(model, DiffOpt.ReverseVariablePrimal(), x, 1.0)

    DiffOpt.reverse_differentiate!(model)

    @test MOIU.isapprox_zero(moi_function(MOI.get(model, DiffOpt.ReverseObjectiveFunction())), ATOL)

    db = [1.0]
    dA = [0.0 -0.5 0.0]
    for (j,jc) in enumerate(ctr_eq)
        grad = MOI.get(model, DiffOpt.ReverseConstraintFunction(), jc)
        @test JuMP.constant(grad) ≈ -db[j] atol=ATOL rtol=RTOL
        for (i,iv) in enumerate(x)
            @test JuMP.coefficient(grad, iv) ≈ dA[j,i] atol=ATOL rtol=RTOL
        end
    end


    dh = zeros(6,1)
    dG = zeros(6,3)
    for (j,jc) in enumerate(ctr_le)
        grad = MOI.get(model, DiffOpt.ReverseConstraintFunction(), jc)
        @test JuMP.constant(grad) ≈ -dh[j] atol=ATOL rtol=RTOL
        for (i,iv) in enumerate(x)
            @test JuMP.coefficient(grad, iv) ≈ dG[j,i] atol=ATOL rtol=RTOL
        end
    end
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

    model = JuMP.direct_model(DiffOpt.diff_optimizer(Ipopt.Optimizer))
    MOI.set(model, MOI.Silent(), true)
    @variables(model, begin
        x
        y
        z
    end)
    @constraint(model, c1,
        x + 2y + 3z >= 4
    )
    @constraint(model, c2,
        x +  y      >= 1
    )
    @objective(model, Min,
        x^2 + x*y + y^2 + y*z + z^2
    )

    optimize!(model)

    z = [x, y, z]

    @test JuMP.value.(z) ≈ [4/7, 3/7, 6/7] atol=ATOL rtol=RTOL

    MOI.set.(model, DiffOpt.ReverseVariablePrimal(), z, 1.0)

    DiffOpt.reverse_differentiate!(model)

    dl_dq = [-0.2142857;  0.21428567; -0.07142857]
    dl_dQ = [-0.12244895  0.01530609 -0.11224488;
              0.01530609  0.09183674  0.07653058;
             -0.11224488  0.07653058 -0.06122449]
    expected = dl_dq' * z + z' * (dl_dQ / 2.0) * z
    @test moi_function(MOI.get(model, DiffOpt.ReverseObjectiveFunction())) ≈ moi_function(expected)  atol=ATOL rtol=RTOL

    dh = [0.35714284; 0.4285714]
    dG = -[0.05102035   0.30612245  0.255102;
           0.06122443   0.36734694  0.3061224]
    for (j,jc) in enumerate([c1, c2])
        grad = MOI.get(model, DiffOpt.ReverseConstraintFunction(), jc)
        @test JuMP.constant(grad) ≈ -dh[j] atol=ATOL rtol=RTOL
        for (i,iv) in enumerate(z)
            @test JuMP.coefficient(grad, iv) ≈ dG[j,i] atol=ATOL rtol=RTOL
        end
    end
end

@testset "Differentiating MOI examples 2 - non trivial backward pass vector" begin
    # non-homogeneous quadratic objective
    #    minimize 2 x^2 + y^2 + xy + x + y
    #       s.t.  x, y >= 0
    #             x + y = 1 (c1)

    model = JuMP.Model(() -> DiffOpt.diff_optimizer(Ipopt.Optimizer))
    MOI.set(model, MOI.Silent(), true)
    @variable(model, x ≥ 0);
    @variable(model, y ≥ 0);
    @constraint(model, c1, x + y == 1)
    @objective(model, Min, 2 * x^2 + y^2 + x * y + x + y)

    optimize!(model)

    z = [x, y]

    @test JuMP.value.(z) ≈ [0.25, 0.75] atol=ATOL rtol=RTOL

    MOI.set.(model, DiffOpt.ReverseVariablePrimal(), z, [1.3, 0.5])

    DiffOpt.reverse_differentiate!(model)

    dl_dq = [-0.2; 0.2]
    dl_dQ = [-0.05   -0.05;
             -0.05    0.15]
    expected = dl_dq' * z + z' * (dl_dQ / 2.0) * z
    @test moi_function(MOI.get(model, DiffOpt.ReverseObjectiveFunction())) ≈ moi_function(expected)  atol=ATOL rtol=RTOL

    func = MOI.get(model, DiffOpt.ReverseConstraintFunction(), c1)
    @test -0.7 ≈ JuMP.constant(func) atol=ATOL rtol=RTOL

    dl_dA = [0.375 -1.075]
    for (j, vi) in enumerate(z)
        @test dl_dA[j] ≈ JuMP.coefficient(func, vi) atol=ATOL rtol=RTOL
    end

    c_le = [LowerBoundRef(x), LowerBoundRef(y)]

    dl_dh = [1e-8; 1e-8]
    dl_dG = [1e-8  1e-8; 1e-8 1e-8]
    for (j,jc) in enumerate(c_le)
        grad = MOI.get(model, DiffOpt.ReverseConstraintFunction(), jc)
        @test JuMP.constant(grad) ≈ dl_dh[j] atol=ATOL rtol=RTOL
        for (i,iv) in enumerate(z)
            @test JuMP.coefficient(grad, iv) ≈ dl_dG[j,i] atol=ATOL rtol=RTOL
        end
    end

end

@testset "Differentiating non trivial convex QP JuMP" begin
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

    model = JuMP.Model(() -> DiffOpt.diff_optimizer(Ipopt.Optimizer))
    MOI.set(model, MOI.Silent(), true)

    @variable(model, x[1:nz])

    @objective(model, Min, dot(Q * x, x) + dot(q, x))
    @constraint(model, c_le, G * x .<= h)
    @constraint(model, c_eq, A * x .== b)

    optimize!(model)

    MOI.set.(model, DiffOpt.ReverseVariablePrimal(), x, 1.0)

    # compute gradients
    DiffOpt.reverse_differentiate!(model)


    # read gradients from files
    param_names = ["dP", "dq", "dG", "dh", "dA", "db"]
    grads_actual = []

    for name in param_names
        push!(grads_actual,
            DelimitedFiles.readdlm(
                joinpath(@__DIR__, "data", "$(name).txt"),
                ' ', Float64, '\n',
            )
        )
    end

    dq = grads_actual[2]
    dh = grads_actual[4]
    db = grads_actual[6]

    grad = MOI.get(model, DiffOpt.ReverseObjectiveFunction())
    @test moi_function(grad) ≈ moi_function(dq ⋅ x) atol=1e-2 rtol=1e-2

    for (i, ci) in enumerate(c_le)
        @test -dh[i] ≈ JuMP.constant(MOI.get(model,
            DiffOpt.ReverseConstraintFunction(), ci)) atol=1e-2 rtol=1e-2
    end

    for (i, ci) in enumerate(c_eq)
        @test -db[i] ≈ JuMP.constant(MOI.get(model,
            DiffOpt.ReverseConstraintFunction(), ci)) atol=1e-2 rtol=1e-2
    end

    # # testing differences
    # for i in 1:size(grads)[1]
    #     @test grads[i] ≈  grads_actual[i] atol=1e-2 rtol=1e-2
    # end
end

@testset "Differentiating LP; checking gradients for non-active contraints" begin
    # Issue #40 from Gurobi.jl
    # min  x
    # s.t. x >= 0
    #      x >= 3

    model = direct_model(DiffOpt.diff_optimizer(HiGHS.Optimizer))
    MOI.set(model, MOI.Silent(), true)

    @variable(model, x[1:1])

    @objective(model, Min, 1.1 * x[1])
    @constraint(model, c1, x[1] ≥ 0)
    @constraint(model, c2, x[1] ≥ 3)

    optimize!(model)

    # obtain gradients
    MOI.set.(model, DiffOpt.ReverseVariablePrimal(), x, 1.0)

    DiffOpt.reverse_differentiate!(model)

    c_le = [c1, c2]

    dh = [0.0, 1.0]
    dG = [0.0, -3.0]
    for (j,jc) in enumerate(c_le)
        grad = MOI.get(model, DiffOpt.ReverseConstraintFunction(), jc)
        @test JuMP.constant(grad) ≈ -dh[j] atol=ATOL rtol=RTOL
        for (i,iv) in enumerate(x)
            @test JuMP.coefficient(grad, iv) ≈ dG[j,i] atol=ATOL rtol=RTOL
        end
    end


    model = direct_model(DiffOpt.diff_optimizer(HiGHS.Optimizer))
    MOI.set(model, MOI.Silent(), true)

    @variable(model, x[1:1])

    @objective(model, Min, x[1])
    @constraint(model, c1, x[1] ≥ 0)
    @constraint(model, c2, x[1] ≥ 3)

    optimize!(model)

    MOI.set.(model, DiffOpt.ReverseVariablePrimal(), x, 1.0)

    DiffOpt.reverse_differentiate!(model)

    c_le = [c1, c2]

    dh = [0.0, 1.0]
    dG = [0.0, -3.0]
    for (j,jc) in enumerate(c_le)
        grad = MOI.get(model, DiffOpt.ReverseConstraintFunction(), jc)
        @test JuMP.constant(grad) ≈ -dh[j] atol=ATOL rtol=RTOL
        for (i,iv) in enumerate(x)
            @test JuMP.coefficient(grad, iv) ≈ dG[j,i] atol=ATOL rtol=RTOL
        end
    end
end


@testset "Differentiating LP; checking gradients for non-active contraints" begin
    # refered from - https://en.wikipedia.org/wiki/Simplex_algorithm#Example

    # max 2x + 3y + 4z
    # s.t. 3x+2y+z <= 10
    #      2x+5y+3z <= 15
    #      x,y,z >= 0

    model = direct_model(DiffOpt.diff_optimizer(SCS.Optimizer))
    MOI.set(model, MOI.Silent(), true)
    @variable(model, v[1:3] ≥ 0)
    @objective(model, Min, dot([-2.0, -3.0, -4.0], v))
    (x, y, z) = v

    @constraint(model, c1, 3x+2y+z <= 10)
    @constraint(model, c2, 2x+5y+3z <= 15)
    optimize!(model)

    MOI.set.(model, DiffOpt.ReverseVariablePrimal(), v, 1.0)

    DiffOpt.reverse_differentiate!(model)

    dQ = zeros(3,3)
    dc = zeros(3)
    dG = [0.0 0.0 0.0;
          0.0 0.0 -5/3;
          0.0 0.0 -5/3;
          0.0 0.0 10/3;
          0.0 0.0 0.0]
    dh = [0.0; 1/3; 1/3; -2/3; 0.0]

    cb = LowerBoundRef.(v)
    cc = [c1, c2]
    ctrs = vcat(cc, cb)#, cc)

    expected = dc' * v + v' * dQ * v
    grad = MOI.get(model, DiffOpt.ReverseObjectiveFunction())
    @test moi_function(grad) ≈ moi_function(expected) atol=ATOL rtol=RTOL

    for (j,jc) in enumerate(ctrs)
        grad = MOI.get(model, DiffOpt.ReverseConstraintFunction(), jc)
        @test JuMP.constant(grad) ≈ -dh[j] atol=ATOL rtol=RTOL
        for (i,iv) in enumerate(v)
            @test JuMP.coefficient(grad, iv) ≈ dG[j,i] atol=ATOL rtol=RTOL
        end
    end
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

    model = direct_model(DiffOpt.diff_optimizer(HiGHS.Optimizer))
    MOI.set(model, MOI.Silent(), true)
    @variable(model, v[1:3] ≥ 0)
    (x, y, z) = v
    @objective(model, Max, 2x + 3y + 4z)
    @constraint(model, c1, 3x+2y+z <= 10)
    @constraint(model, c2, 2x+5y+3z <= 15)
    @constraint(model, c3, x ≤ 3)
    @constraint(model, c4, 1.0 * y ≤ 2)
    @constraint(model, c5, z ≥ 2)

    optimize!(model)

    MOI.set.(model, DiffOpt.ReverseVariablePrimal(), v, 1.0)

    DiffOpt.reverse_differentiate!(model)

    dQ = zeros(3,3)
    dc = zeros(3)
    dG = [0.0 0.0 0.0
          0.0 0.0 -5/3
          0.0 0.0 0.0
          0.0 0.0 0.0
          0.0 0.0 0.0
          0.0 0.0 -5/3
          0.0 0.0 10/3
          0.0 0.0 0.0
          ]
    dh = [0.0, 1/3, 0.0, 0.0, 0.0, 1/3, -2/3, 0.0]

    cb = LowerBoundRef.(v)
    cc = [c1, c2, c3, c4, c5]
    ctrs = vcat(cc, cb)

    expected = dc' * v + v' * dQ * v
    grad = MOI.get(model, DiffOpt.ReverseObjectiveFunction())
    @test moi_function(grad) ≈ moi_function(expected) atol=ATOL rtol=RTOL

    for (j,jc) in enumerate(ctrs)
        grad = MOI.get(model, DiffOpt.ReverseConstraintFunction(), jc)
        @test JuMP.constant(grad) ≈ -dh[j] atol=ATOL rtol=RTOL
        for (i,iv) in enumerate(v)
            @test JuMP.coefficient(grad, iv) ≈ dG[j,i] atol=ATOL rtol=RTOL
        end
    end
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

    model = JuMP.direct_model(DiffOpt.diff_optimizer(SCS.Optimizer))
    MOI.set(model, MOI.Silent(), true)
    @variable(model, x)
    @variable(model, y)
    @variable(model, t)

    vv = [x,y,t]

    @objective(model, Min, x)
    @constraint(model, c1, [-1/√2 + y] in MOI.Nonnegatives(1))
    @constraint(model, c2, [1 - t] in MOI.Zeros(1))
    @constraint(model, c3, [1.0 * t, x, y] in SecondOrderCone())

    optimize!(model)

    v = JuMP.value.(vv)
    # slack variables
    s = collect(Iterators.flatten(JuMP.value.([c1, c2, c3])))
    y = collect(Iterators.flatten(JuMP.dual.([c1, c2, c3])))

    # these matrices are benchmarked with the output generated by diffcp
    # refer the python file mentioned above to get equivalent python source code
    @test v ≈ [-1/√2; 1/√2; 1.0] atol=ATOL rtol=RTOL
    @test s ≈ [0.0, 0.0, 1.0, -1/√2, 1/√2] atol=ATOL rtol=RTOL
    @test y ≈ [1, √2, √2, 1, -1] atol=ATOL rtol=RTOL

    dA = zeros(5, 3)
    dA[1:3, :] .= Matrix(1.0I, 3, 3)
    db = zeros(5)
    dc = zeros(3)

    MOI.set.(model, DiffOpt.ReverseVariablePrimal(), vv, 1.0)

    @test_broken DiffOpt.reverse_differentiate!(model)

    # TODO add tests

end
