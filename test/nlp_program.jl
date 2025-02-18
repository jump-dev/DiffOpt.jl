module TestNLPProgram

using DiffOpt
using JuMP
using Ipopt
using Test
using FiniteDiff
import DelimitedFiles
using SparseArrays

include(joinpath(@__DIR__, "data/nlp_problems.jl"))

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

################################################
#=
# Test JuMP Hessian and Jacobian

From JuMP Tutorial for Querying Hessians:
https://github.com/jump-dev/JuMP.jl/blob/301d46e81cb66c74c6e22cd89fb89ced740f157b/docs/src/tutorials/nonlinear/querying_hessians.jl#L67-L72
=#
################################################

function analytic_hessian(x, σ, μ, p)
    g_1_H = [2.0 0.0; 0.0 0.0]
    g_2_H = p[1] * [2.0 2.0; 2.0 2.0]
    f_H = zeros(2, 2)
    f_H[1, 1] = 2.0 + p[3] * 12.0 * x[1]^2 - p[3] * 4.0 * x[2]
    f_H[1, 2] = f_H[2, 1] = -p[3] * 4.0 * x[1]
    f_H[2, 2] = p[3] * 2.0
    return σ * f_H + μ' * [g_1_H, g_2_H]
end

function analytic_jacobian(x, p)
    g_1_J = [
        2.0 * x[1], # ∂g_1/∂x_1
        0.0,       # ∂g_1/∂x_2
        -1.0,      # ∂g_1/∂p_1 
        0.0,      # ∂g_1/∂p_2
        0.0,      # ∂g_1/∂p_3
    ]
    g_2_J = [
        p[1] * 2.0 * (x[1] + x[2]), # ∂g_2/∂x_1
        2.0 * (x[1] + x[2]),        # ∂g_2/∂x_2
        (x[1] + x[2])^2,            # ∂g_2/∂p_1
        -1.0,                        # ∂g_2/∂p_2
        0.0,                         # ∂g_2/∂p_3
    ]
    return hcat(g_2_J, g_1_J)'[:, :]
end

function _test_create_evaluator(nlp_model)
    @testset "Create Evaluator" begin
        cache = DiffOpt.NonLinearProgram._cache_evaluator!(nlp_model)
        @test cache.evaluator isa MOI.Nonlinear.Evaluator
        @test cache.cons isa Vector{MOI.Nonlinear.ConstraintIndex}
    end
end

function test_compute_optimal_hess_jacobian()
    @testset "Compute Optimal Hessian and Jacobian" begin
        # Model
        model, x, cons, params = create_nonlinear_jump_model()
        # Optimize
        optimize!(model)
        @assert is_solved_and_feasible(model)
        # Create evaluator
        nlp_model = DiffOpt._diff(model.moi_backend.optimizer.model).model
        _test_create_evaluator(nlp_model)
        cons = nlp_model.cache.cons
        y = [
            nlp_model.y[nlp_model.model.nlp_index_2_constraint[row].value]
            for row in cons
        ]
        hessian, jacobian =
            DiffOpt.NonLinearProgram._compute_optimal_hess_jac(nlp_model, cons)
        # Check Hessian
        primal_idx = [i.value for i in nlp_model.cache.primal_vars]
        params_idx = [i.value for i in nlp_model.cache.params]
        @test all(
            isapprox(
                hessian[primal_idx, primal_idx],
                analytic_hessian(
                    nlp_model.x[primal_idx],
                    1.0,
                    -y,
                    nlp_model.x[params_idx],
                );
                atol = 1,
            ),
        )
        # Check Jacobian
        @test all(
            isapprox(
                jacobian[:, [primal_idx; params_idx]],
                analytic_jacobian(
                    nlp_model.x[primal_idx],
                    nlp_model.x[params_idx],
                ),
            ),
        )
    end
end

################################################
#=
# Test Sensitivity through analytical
=#
################################################

function test_analytical_simple(; P = 2) # Number of parameters
    @testset "Bounds Bounds" begin
        m = Model(
            () -> DiffOpt.diff_optimizer(
                Ipopt.Optimizer;
                with_parametric_opt_interface = false,
            ),
        )

        @variable(m, 0 ≤ x[1:P] ≤ 1)
        @variable(m, p[1:P] ∈ Parameter.(0.5))

        @constraint(m, x .≥ p)

        @objective(m, Min, sum(x))

        optimize!(m)
        @assert is_solved_and_feasible(m)

        # Set pertubations
        Δp = [0.1 for _ in 1:P]
        MOI.set.(
            m,
            DiffOpt.ForwardConstraintSet(),
            ParameterRef.(p),
            Parameter.(Δp),
        )

        # Compute derivatives
        DiffOpt.forward_differentiate!(m)

        # Test sensitivities
        @test all(
            isapprox(
                [
                    MOI.get(m, DiffOpt.ForwardVariablePrimal(), x[i]) for
                    i in 1:P
                ],
                [0.1 for _ in 1:P];
                atol = 1e-8,
            ),
        )
    end
    @testset "Bounds as RHS constraints" begin
        m = Model(
            () -> DiffOpt.diff_optimizer(
                Ipopt.Optimizer;
                with_parametric_opt_interface = false,
            ),
        )

        @variable(m, x[1:P])
        @constraint(m, x .≥ 0)
        @constraint(m, x .≤ 1)
        @variable(m, p[1:P] ∈ Parameter.(0.5))

        @constraint(m, x .≥ p)

        @objective(m, Min, sum(x))

        optimize!(m)
        @assert is_solved_and_feasible(m)

        # Set pertubations
        Δp = [0.1 for _ in 1:P]
        MOI.set.(
            m,
            DiffOpt.ForwardConstraintSet(),
            ParameterRef.(p),
            Parameter.(Δp),
        )

        # Compute derivatives
        DiffOpt.forward_differentiate!(m)

        # Test sensitivities
        @test all(
            isapprox(
                [
                    MOI.get(m, DiffOpt.ForwardVariablePrimal(), x[i]) for
                    i in 1:P
                ],
                [0.1 for _ in 1:P];
                atol = 1e-8,
            ),
        )
    end
    @testset "Bounds as Mixed constraints" begin
        m = Model(
            () -> DiffOpt.diff_optimizer(
                Ipopt.Optimizer;
                with_parametric_opt_interface = false,
            ),
        )

        @variable(m, x[1:P])
        @constraint(m, 0 .≤ x)
        @constraint(m, x .≤ 1)
        @variable(m, p[1:P] ∈ Parameter.(0.5))

        @constraint(m, x .≥ p)

        @objective(m, Min, sum(x))

        optimize!(m)
        @assert is_solved_and_feasible(m)

        # Set pertubations
        Δp = [0.1 for _ in 1:P]
        MOI.set.(
            m,
            DiffOpt.ForwardConstraintSet(),
            ParameterRef.(p),
            Parameter.(Δp),
        )

        # Compute derivatives
        DiffOpt.forward_differentiate!(m)

        # Test sensitivities
        @test all(
            isapprox(
                [
                    MOI.get(m, DiffOpt.ForwardVariablePrimal(), x[i]) for
                    i in 1:P
                ],
                [0.1 for _ in 1:P];
                atol = 1e-8,
            ),
        )
    end
    @testset "Bounds as LHS constraints" begin
        m = Model(
            () -> DiffOpt.diff_optimizer(
                Ipopt.Optimizer;
                with_parametric_opt_interface = false,
            ),
        )

        @variable(m, x[1:P])
        @constraint(m, 0 .≤ x)
        @constraint(m, 1 .≥ x)
        @variable(m, p[1:P] ∈ Parameter.(0.5))

        @constraint(m, x .≥ p)

        @objective(m, Min, sum(x))

        optimize!(m)
        @assert is_solved_and_feasible(m)

        # Set pertubations
        Δp = [0.1 for _ in 1:P]
        MOI.set.(
            m,
            DiffOpt.ForwardConstraintSet(),
            ParameterRef.(p),
            Parameter.(Δp),
        )

        # Compute derivatives
        DiffOpt.forward_differentiate!(m)

        # Test sensitivities
        @test all(
            isapprox(
                [
                    MOI.get(m, DiffOpt.ForwardVariablePrimal(), x[i]) for
                    i in 1:P
                ],
                [0.1 for _ in 1:P];
                atol = 1e-8,
            ),
        )
    end
end

# f(x, p) = 0 
# x = g(p)
# ∂x/∂p = ∂g/∂p

DICT_PROBLEMS_Analytical_no_cc = Dict(
    "geq no impact" => (
        p_a = [1.5],
        Δp = [0.2],
        Δx = [0.0],
        Δy = [0.0; 0.0],
        Δvu = [],
        Δvl = [],
        model_generator = create_jump_model_1,
    ),
    "geq impact" => (
        p_a = [2.1],
        Δp = [0.2],
        Δx = [0.2],
        Δy = [0.4; 0.0],
        Δvu = [],
        Δvl = [],
        model_generator = create_jump_model_1,
    ),
    "geq bound impact" => (
        p_a = [2.1],
        Δp = [0.2],
        Δx = [0.2],
        Δy = [0.4],
        Δvu = [],
        Δvl = [0.0],
        model_generator = create_jump_model_2,
    ),
    "leq no impact" => (
        p_a = [-1.5],
        Δp = [-0.2],
        Δx = [0.0],
        Δy = [0.0; 0.0],
        Δvu = [],
        Δvl = [],
        model_generator = create_jump_model_3,
    ),
    "leq impact" => (
        p_a = [-2.1],
        Δp = [-0.2],
        Δx = [-0.2],
        Δy = [0.0; 0.0],
        Δvu = [],
        Δvl = [],
        model_generator = create_jump_model_3,
    ),
    "leq no impact max" => (
        p_a = [2.1],
        Δp = [0.2],
        Δx = [0.0],
        Δy = [0.0; 0.0],
        Δvu = [],
        Δvl = [],
        model_generator = create_jump_model_4,
    ),
    "leq impact max" => (
        p_a = [1.5],
        Δp = [0.2],
        Δx = [0.2],
        Δy = [0.0; 0.0],
        Δvu = [],
        Δvl = [],
        model_generator = create_jump_model_4,
    ),
    "geq no impact max" => (
        p_a = [1.5],
        Δp = [0.2],
        Δx = [0.0],
        Δy = [0.0; 0.0],
        Δvu = [],
        Δvl = [],
        model_generator = create_jump_model_5,
    ),
    "geq impact max" => (
        p_a = [2.1],
        Δp = [0.2],
        Δx = [0.2],
        Δy = [0.0; 0.0],
        Δvu = [],
        Δvl = [],
        model_generator = create_jump_model_5,
    ),
)

function test_compute_derivatives_Analytical(;
    DICT_PROBLEMS = DICT_PROBLEMS_Analytical_no_cc,
)
    @testset "Compute Derivatives Analytical: $problem_name" for (
        problem_name,
        (p_a, Δp, Δx, Δy, Δvu, Δvl, model_generator),
    ) in DICT_PROBLEMS
        # OPT Problem
        model, primal_vars, cons, params = model_generator()
        set_parameter_value.(params, p_a)
        optimize!(model)
        @assert is_solved_and_feasible(model)
        # Set pertubations
        MOI.set.(
            model,
            DiffOpt.ForwardConstraintSet(),
            ParameterRef.(params),
            Parameter.(Δp),
        )
        # Compute derivatives
        DiffOpt.forward_differentiate!(model)
        # Test sensitivities primal_vars
        if !isempty(Δx)
            @test all(
                isapprox.(
                    [
                        MOI.get(model, DiffOpt.ForwardVariablePrimal(), var) for
                        var in primal_vars
                    ],
                    Δx;
                    atol = 1e-4,
                ),
            )
        end
        # Test sensitivities cons
        if !isempty(Δy)
            @test all(
                isapprox.(
                    [
                        MOI.get(model, DiffOpt.ForwardConstraintDual(), con) for
                        con in cons
                    ],
                    Δy;
                    atol = 1e-4,
                ),
            )
        end
        # Test sensitivities dual vars
        if !isempty(Δvu)
            primal_vars_upper = [v for v in primal_vars if has_upper_bound(v)]
            @test all(
                isapprox.(
                    [
                        MOI.get(
                            model,
                            DiffOpt.ForwardConstraintDual(),
                            UpperBoundRef(var),
                        ) for var in primal_vars_upper
                    ],
                    Δvu;
                    atol = 1e-4,
                ),
            )
        end
        if !isempty(Δvl)
            primal_vars_lower = [v for v in primal_vars if has_lower_bound(v)]
            @test all(
                isapprox.(
                    [
                        MOI.get(
                            model,
                            DiffOpt.ForwardConstraintDual(),
                            LowerBoundRef(var),
                        ) for var in primal_vars_lower
                    ],
                    Δvl;
                    atol = 1e-4,
                ),
            )
        end
    end
end

################################################
#=
# Test Sensitivity through finite differences
=#
################################################

function stack_solution(model, p_a, params, primal_vars, cons)
    set_parameter_value.(params, p_a)
    optimize!(model)
    @assert is_solved_and_feasible(model)
    return [value.(primal_vars); dual.(cons)]
end

DICT_PROBLEMS_no_cc = Dict(
    "QP_sIpopt" => (
        p_a = [4.5; 1.0],
        Δp = [0.001; 0.0],
        model_generator = create_nonlinear_jump_model_sipopt,
    ),
    "NLP_1" => (
        p_a = [3.0; 2.0; 200],
        Δp = [0.001; 0.0; 0.0],
        model_generator = create_nonlinear_jump_model_1,
    ),
    "NLP_1_2" => (
        p_a = [3.0; 2.0; 200],
        Δp = [0.0; 0.001; 0.0],
        model_generator = create_nonlinear_jump_model_1,
    ),
    "NLP_1_3" => (
        p_a = [3.0; 2.0; 200],
        Δp = [0.0; 0.0; 0.001],
        model_generator = create_nonlinear_jump_model_1,
    ),
    "NLP_1_4" => (
        p_a = [3.0; 2.0; 200],
        Δp = [0.1; 0.5; 0.5],
        model_generator = create_nonlinear_jump_model_1,
    ),
    "NLP_1_4" => (
        p_a = [3.0; 2.0; 200],
        Δp = [0.5; -0.5; 0.1],
        model_generator = create_nonlinear_jump_model_1,
    ),
    "NLP_2" => (
        p_a = [3.0; 2.0; 10],
        Δp = [0.01; 0.0; 0.0],
        model_generator = create_nonlinear_jump_model_2,
    ),
    "NLP_2_2" => (
        p_a = [3.0; 2.0; 10],
        Δp = [-0.1; 0.0; 0.0],
        model_generator = create_nonlinear_jump_model_2,
    ),
    "NLP_3" => (
        p_a = [3.0; 2.0; 10],
        Δp = [0.001; 0.0; 0.0],
        model_generator = create_nonlinear_jump_model_3,
    ),
    "NLP_3_2" => (
        p_a = [3.0; 2.0; 10],
        Δp = [0.0; 0.001; 0.0],
        model_generator = create_nonlinear_jump_model_3,
    ),
    "NLP_3_3" => (
        p_a = [3.0; 2.0; 10],
        Δp = [0.0; 0.0; 0.001],
        model_generator = create_nonlinear_jump_model_3,
    ),
    "NLP_3_4" => (
        p_a = [3.0; 2.0; 10],
        Δp = [0.5; 0.001; 0.5],
        model_generator = create_nonlinear_jump_model_3,
    ),
    "NLP_3_5" => (
        p_a = [3.0; 2.0; 10],
        Δp = [0.1; 0.3; 0.1],
        model_generator = create_nonlinear_jump_model_3,
    ),
    "NLP_3_6" => (
        p_a = [3.0; 2.0; 10],
        Δp = [0.1; 0.2; -0.5],
        model_generator = create_nonlinear_jump_model_3,
    ),
    "NLP_4" => (
        p_a = [1.0; 2.0; 100],
        Δp = [0.001; 0.0; 0.0],
        model_generator = create_nonlinear_jump_model_4,
    ),
    "NLP_5" => (
        p_a = [1.0; 2.0; 100],
        Δp = [0.0; 0.001; 0.0],
        model_generator = create_nonlinear_jump_model_5,
    ),
    "NLP_6" => (
        p_a = [100.0; 200.0],
        Δp = [0.2; 0.5],
        model_generator = create_nonlinear_jump_model_6,
    ),
)

function test_compute_derivatives_Finite_Diff(;
    DICT_PROBLEMS = DICT_PROBLEMS_no_cc,
)
    @testset "Compute Derivatives FiniteDiff: $problem_name" for (
            problem_name,
            (p_a, Δp, model_generator),
        ) in DICT_PROBLEMS,
        ismin in [true, false]
        # OPT Problem
        model, primal_vars, cons, params = model_generator(; ismin = ismin)
        set_parameter_value.(params, p_a)
        optimize!(model)
        @assert is_solved_and_feasible(model)
        # Set pertubations
        MOI.set.(
            model,
            DiffOpt.ForwardConstraintSet(),
            ParameterRef.(params),
            Parameter.(Δp),
        )
        # Compute derivatives
        DiffOpt.forward_differentiate!(model)
        Δx = [
            MOI.get(model, DiffOpt.ForwardVariablePrimal(), var) for
            var in primal_vars
        ]
        Δy = [
            MOI.get(model, DiffOpt.ForwardConstraintDual(), con) for con in cons
        ]
        # Compute derivatives using finite differences
        ∂s_fd =
            FiniteDiff.finite_difference_jacobian(
                (p) -> stack_solution(model, p, params, primal_vars, cons),
                p_a,
            ) * Δp
        # Test sensitivities primal_vars
        @test all(isapprox.(Δx, ∂s_fd[1:length(primal_vars)]; atol = 1e-4))
        # Test sensitivities cons
        @test all(isapprox.(Δy, ∂s_fd[length(primal_vars)+1:end]; atol = 1e-4))
    end
end

################################################
#=
# Test Sensitivity through Reverse Mode
=#
################################################

# Copied from test/jump.jl and adapated for nlp interface
function test_differentiating_non_trivial_convex_qp_jump()
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
    model = JuMP.Model(() -> DiffOpt.diff_optimizer(Ipopt.Optimizer))
    MOI.set(model, MOI.Silent(), true)
    @variable(model, x[1:nz])
    @variable(model, p_le[1:nineq_le] ∈ MOI.Parameter.(0.0))
    @variable(model, p_eq[1:neq] ∈ MOI.Parameter.(0.0))
    @objective(model, Min, x' * Q * x + q' * x)
    @constraint(model, c_le, G * x .<= h + p_le)
    @constraint(model, c_eq, A * x .== b + p_eq)
    optimize!(model)
    MOI.set.(model, DiffOpt.ReverseVariablePrimal(), x, 1.0)
    # compute gradients
    DiffOpt.reverse_differentiate!(model)
    # read gradients from files
    param_names = ["dP", "dq", "dG", "dh", "dA", "db"]
    grads_actual = []
    for name in param_names
        filename = joinpath(@__DIR__, "data", "$(name).txt")
        push!(
            grads_actual,
            DelimitedFiles.readdlm(filename, ' ', Float64, '\n'),
        )
    end
    dh = grads_actual[4]
    db = grads_actual[6]

    for (i, ci) in enumerate(c_le)
        @test -dh[i] ≈
              -MOI.get(
            model,
            DiffOpt.ReverseConstraintSet(),
            ParameterRef(p_le[i]),
        ).value atol = 1e-2 rtol = 1e-2
    end
    for (i, ci) in enumerate(c_eq)
        @test -db[i] ≈
              -MOI.get(
            model,
            DiffOpt.ReverseConstraintSet(),
            ParameterRef(p_eq[i]),
        ).value atol = 1e-2 rtol = 1e-2
    end

    return
end

################################################
#=
# Test Changing Factorization routine
=#
################################################

function test_changing_factorization()
    P = 2
    m = Model(
        () -> DiffOpt.diff_optimizer(
            Ipopt.Optimizer;
            with_parametric_opt_interface = false,
        ),
    )

    @variable(m, x[1:P])
    @constraint(m, x .≥ 0)
    @constraint(m, x .≤ 1)
    @variable(m, p[1:P] ∈ Parameter.(0.5))

    @constraint(m, x .≥ p)

    @objective(m, Min, sum(x))

    optimize!(m)
    @assert is_solved_and_feasible(m)

    # Set pertubations
    Δp = [0.1 for _ in 1:P]
    MOI.set.(
        m,
        DiffOpt.ForwardConstraintSet(),
        ParameterRef.(p),
        Parameter.(Δp),
    )

    # wrong type
    @test_throws MethodError MOI.set(m, DiffOpt.MFactorization(), 2)

    # correct type but wrong number of arguments
    MOI.set(m, DiffOpt.MFactorization(), SparseArrays.lu)

    @test_throws MethodError DiffOpt.forward_differentiate!(m)

    # correct type and correct number of arguments
    MOI.set(m, DiffOpt.MFactorization(), (M, model) -> SparseArrays.lu(M))

    # Compute derivatives
    DiffOpt.forward_differentiate!(m)

    # Test sensitivities
    @test all(
        isapprox(
            [MOI.get(m, DiffOpt.ForwardVariablePrimal(), x[i]) for i in 1:P],
            [0.1 for _ in 1:P];
            atol = 1e-8,
        ),
    )
end

end # module

TestNLPProgram.runtests()
