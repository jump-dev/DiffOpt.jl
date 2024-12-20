using JuMP
using Ipopt
using Test
using FiniteDiff

include("test/data/nlp_problems.jl")

# Example usage
# using Revise
# using DiffOpt
# using JuMP
# using Ipopt
# using Test

# model, vars, cons, params = create_jump_model_1()
# set_parameter_value.(params, [2.1])
# JuMP.optimize!(model)

# ### Forward differentiation

# # set parameter pertubations
# MOI.set(model, DiffOpt.ForwardParameter(), params[1], 0.2)

# # forward differentiate
# DiffOpt.forward_differentiate!(model)

# # get sensitivities
# MOI.get(model, DiffOpt.ForwardVariablePrimal(), vars[1])

# ### Reverse differentiation

# # set variable pertubations
# MOI.set(model, DiffOpt.ReverseVariablePrimal(), vars[1], 1.0)

# # reverse differentiate
# DiffOpt.reverse_differentiate!(model)

# # get sensitivities
# dp = MOI.get(model, DiffOpt.ReverseParameter(), params[1])

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
        0.0      # ∂g_1/∂p_3
    ]
    g_2_J = [
        p[1] * 2.0 * (x[1] + x[2]), # ∂g_2/∂x_1
        2.0 * (x[1] + x[2]),        # ∂g_2/∂x_2
        (x[1] + x[2])^2,            # ∂g_2/∂p_1
        -1.0,                        # ∂g_2/∂p_2
        0.0                         # ∂g_2/∂p_3
    ]
    return hcat(g_2_J, g_1_J)'[:,:]
end

function test_create_evaluator(nlp_model)
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
        test_create_evaluator(nlp_model)
        cons = nlp_model.cache.cons
        y = [nlp_model.y[nlp_model.model.nlp_index_2_constraint[row].value] for row in cons]
        hessian, jacobian = DiffOpt.NonLinearProgram.compute_optimal_hess_jac(nlp_model, cons)
        # Check Hessian
        primal_idx = [i.value for i in nlp_model.cache.primal_vars]
        params_idx = [i.value for i in nlp_model.cache.params]
        @test all(isapprox(hessian[primal_idx,primal_idx], analytic_hessian(nlp_model.x[primal_idx], 1.0, -y, nlp_model.x[params_idx]); atol = 1))
        # Check Jacobian
        @test all(isapprox(jacobian[:,[primal_idx; params_idx]], analytic_jacobian(nlp_model.x[primal_idx], nlp_model.x[params_idx])))
    end
end

test_compute_optimal_hess_jacobian()

################################################
#=
# Test Sensitivity through analytical
=#
################################################


# f(x, p) = 0 
# x = g(p)
# ∂x/∂p = ∂g/∂p

DICT_PROBLEMS_Analytical_no_cc = Dict(
    "geq no impact" => (p_a=[1.5], Δp=[0.2], Δx=[0.0], Δy=[0.0; 0.0], Δv=[], model_generator=create_jump_model_1),
    "geq impact" => (p_a=[2.1], Δp=[0.2], Δs_a=[0.2; 0.0; 0.2; 0.4; 0.0; 0.4; 0.0], model_generator=create_jump_model_1),
    "geq bound impact" => (p_a=[2.1], Δp=[0.2], Δs_a=[0.2; 0.0; 0.4; 0.0; 0.4], model_generator=create_jump_model_2),
    "leq no impact" => (p_a=[-1.5], Δp=[-0.2], Δs_a=[0.0; 0.2; 0.0; 0.0; 0.0; 0.0; 0.0], model_generator=create_jump_model_3),
    "leq impact" => (p_a=[-2.1], Δp=[-0.2], Δs_a=[-0.2; 0.0; -0.2], model_generator=create_jump_model_3),
    "leq no impact max" => (p_a=[2.1], Δp=[0.2], Δs_a=[0.0; -0.2; 0.0; 0.0; 0.0], model_generator=create_jump_model_4),
    "leq impact max" => (p_a=[1.5], Δp=[0.2], Δs_a=[0.2; 0.0; 0.2], model_generator=create_jump_model_4),
    "geq no impact max" => (p_a=[1.5], Δp=[0.2], Δs_a=[0.0; -0.2; 0.0; 0.0; 0.0], model_generator=create_jump_model_5),
    "geq impact max" => (p_a=[2.1], Δp=[0.2], Δs_a=[0.2; 0.0; 0.2], model_generator=create_jump_model_5),
)

function test_compute_derivatives_Analytical(DICT_PROBLEMS)
    @testset "Compute Derivatives Analytical: $problem_name" for (problem_name, (p_a, Δp, Δx, Δy, Δv, model_generator)) in DICT_PROBLEMS
        # OPT Problem
        model, primal_vars, cons, params = model_generator()
        set_parameter_value.(params, p_a)
        optimize!(model)
        @assert is_solved_and_feasible(model)
        # Set pertubations
        MOI.set.(model, DiffOpt.ForwardParameter(), params, Δp)
        # Compute derivatives
        DiffOpt.forward_differentiate!(model)
        # test sensitivities primal_vars
        @test all(isapprox.([MOI.get(model, DiffOpt.ForwardVariablePrimal(), var) for var in primal_vars], Δx; atol = 1e-4))
        # Check sensitivities
        @test all(isapprox.(Δs[1:length(Δs_a)], Δs_a; atol = 1e-4))
    end
end

################################################
#=
# Test Sensitivity through finite differences
=#
################################################

function eval_model_jump(model, primal_vars, cons, params, p_val)
    set_parameter_value.(params, p_val)
    optimize!(model)
    @assert is_solved_and_feasible(model)
    return value.(primal_vars), dual.(cons), [dual.(LowerBoundRef(v)) for v in primal_vars if has_lower_bound(v)], [dual.(UpperBoundRef(v)) for v in primal_vars if has_upper_bound(v)]
end

function stack_solution(cons, leq_locations, geq_locations, x, _λ, ν_L, ν_U)
    ineq_locations = vcat(geq_locations, leq_locations)
    return Float64[x; value.(get_slack_inequality.(cons[ineq_locations])); _λ; ν_L; _λ[geq_locations]; ν_U; _λ[leq_locations]]
end

function print_wrong_sensitive(Δs, Δs_fd, primal_vars, cons, leq_locations, geq_locations)
    ineq_locations = vcat(geq_locations, leq_locations)
    println("Some sensitivities are not correct: \n")
    # primal vars
    num_primal_vars = length(primal_vars)
    for (i, v) in enumerate(primal_vars)
        if !isapprox(Δs[i], Δs_fd[i]; atol = 1e-6)
            println("Primal var: ", v, " | Δs: ", Δs[i], " | Δs_fd: ", Δs_fd[i])
        end
    end
    # slack vars
    num_slack_vars = length(ineq_locations)
    num_w = num_slack_vars + num_primal_vars
    for (i, c) in enumerate(cons[ineq_locations])
        if !isapprox(Δs[i + num_primal_vars], Δs_fd[i + num_primal_vars] ; atol = 1e-6)
            println("Slack var: ", c, " | Δs: ", Δs[i + num_primal_vars], " | Δs_fd: ", Δs_fd[i + num_primal_vars])
        end
    end
    # dual vars
    num_cons = length(cons)
    for (i, c) in enumerate(cons)
        if !isapprox(Δs[i + num_w], Δs_fd[i + num_w] ; atol = 1e-6)
            println("Dual var: ", c, " | Δs: ", Δs[i + num_w], " | Δs_fd: ", Δs_fd[i + num_w])
        end
    end
    # dual lower bound primal vars
    var_lower = [v for v in primal_vars if has_lower_bound(v)]
    num_lower_bounds = length(var_lower)
    for (i, v) in enumerate(var_lower)
        if !isapprox(Δs[i + num_w + num_cons], Δs_fd[i + num_w + num_cons] ; atol = 1e-6)
            lower_bound_ref = LowerBoundRef(v)
            println("lower bound dual: ", lower_bound_ref, " | Δs: ", Δs[i + num_w + num_cons], " | Δs_fd: ", Δs_fd[i + num_w + num_cons])
        end
    end
    # dual lower bound slack vars
    for (i, c) in enumerate(cons[geq_locations])
        if !isapprox(Δs[i + num_w + num_cons + num_lower_bounds], Δs_fd[i + num_w + num_cons + num_lower_bounds] ; atol = 1e-6)
            println("lower bound slack dual: ", c, " | Δs: ", Δs[i + num_w + num_cons + num_lower_bounds], " | Δs_fd: ", Δs_fd[i + num_w + num_cons + num_lower_bounds])
        end
    end
    for (i, c) in enumerate(cons[leq_locations])
        if !isapprox(Δs[i + num_w + num_cons + num_lower_bounds + length(geq_locations)], Δs_fd[i + num_w + num_cons + num_lower_bounds + length(geq_locations)] ; atol = 1e-6)
            println("upper bound slack dual: ", c, " | Δs: ", Δs[i + num_w + num_cons + num_lower_bounds + length(geq_locations)], " | Δs_fd: ", Δs_fd[i + num_w + num_cons + num_lower_bounds + length(geq_locations)])
        end
    end
    # dual upper bound primal vars
    var_upper = [v for v in primal_vars if has_upper_bound(v)]
    for (i, v) in enumerate(var_upper)
        if !isapprox(Δs[i + num_w + num_cons + num_lower_bounds + num_slack_vars], Δs_fd[i + num_w + num_cons + num_lower_bounds + num_slack_vars] ; atol = 1e-6)
            upper_bound_ref = UpperBoundRef(v)
            println("upper bound dual: ", upper_bound_ref, " | Δs: ", Δs[i + num_w + num_cons + num_lower_bounds + num_slack_vars], " | Δs_fd: ", Δs_fd[i + num_w + num_cons + num_lower_bounds + num_slack_vars])
        end
    end
end

DICT_PROBLEMS_no_cc = Dict(
    "QP_sIpopt" => (p_a=[4.5; 1.0], Δp=[0.001; 0.0], model_generator=create_nonlinear_jump_model_sipopt),
    "NLP_1" => (p_a=[3.0; 2.0; 200], Δp=[0.001; 0.0; 0.0], model_generator=create_nonlinear_jump_model_1),
    "NLP_1_2" => (p_a=[3.0; 2.0; 200], Δp=[0.0; 0.001; 0.0], model_generator=create_nonlinear_jump_model_1),
    "NLP_1_3" => (p_a=[3.0; 2.0; 200], Δp=[0.0; 0.0; 0.001], model_generator=create_nonlinear_jump_model_1),
    "NLP_1_4" => (p_a=[3.0; 2.0; 200], Δp=[0.1; 0.5; 0.5], model_generator=create_nonlinear_jump_model_1),
    "NLP_1_4" => (p_a=[3.0; 2.0; 200], Δp=[0.5; -0.5; 0.1], model_generator=create_nonlinear_jump_model_1),
    "NLP_2" => (p_a=[3.0; 2.0; 10], Δp=[0.01; 0.0; 0.0], model_generator=create_nonlinear_jump_model_2),
    "NLP_2_2" => (p_a=[3.0; 2.0; 10], Δp=[-0.1; 0.0; 0.0], model_generator=create_nonlinear_jump_model_2),
    "NLP_3" => (p_a=[3.0; 2.0; 10], Δp=[0.001; 0.0; 0.0], model_generator=create_nonlinear_jump_model_3),
    "NLP_3_2" => (p_a=[3.0; 2.0; 10], Δp=[0.0; 0.001; 0.0], model_generator=create_nonlinear_jump_model_3),
    "NLP_3_3" => (p_a=[3.0; 2.0; 10], Δp=[0.0; 0.0; 0.001], model_generator=create_nonlinear_jump_model_3),
    "NLP_3_4" => (p_a=[3.0; 2.0; 10], Δp=[0.5; 0.001; 0.5], model_generator=create_nonlinear_jump_model_3),
    "NLP_3_5" => (p_a=[3.0; 2.0; 10], Δp=[0.1; 0.3; 0.1], model_generator=create_nonlinear_jump_model_3),
    "NLP_3_6" => (p_a=[3.0; 2.0; 10], Δp=[0.1; 0.2; -0.5], model_generator=create_nonlinear_jump_model_3),
    "NLP_4" => (p_a=[1.0; 2.0; 100], Δp=[0.001; 0.0; 0.0], model_generator=create_nonlinear_jump_model_4),
    "NLP_5" => (p_a=[1.0; 2.0; 100], Δp=[0.0; 0.001; 0.0], model_generator=create_nonlinear_jump_model_5),
    "NLP_6" => (p_a=[100.0; 200.0], Δp=[0.2; 0.5], model_generator=create_nonlinear_jump_model_6),
)


DICT_PROBLEMS_cc = Dict(
    "QP_JuMP" => (p_a=[1.0; 2.0; 100.0], Δp=[-0.5; 0.5; 0.1], model_generator=create_nonlinear_jump_model),
    "QP_sIpopt2" => (p_a=[5.0; 1.0], Δp=[-0.5; 0.0], model_generator=create_nonlinear_jump_model_sipopt),
)

function test_compute_derivatives_Finite_Diff(DICT_PROBLEMS, iscc=false)
    # @testset "Compute Derivatives: $problem_name" 
    for (problem_name, (p_a, Δp, model_generator)) in DICT_PROBLEMS, ismin in [true, false]
        # OPT Problem
        model, primal_vars, cons, params = model_generator(;ismin=ismin)
        eval_model_jump(model, primal_vars, cons, params, p_a)
        println("$problem_name: ", model)
        # Compute derivatives
        # Δp = [0.001; 0.0; 0.0]
        p_b = p_a .+ Δp
        (Δs, sp_approx), evaluator, cons = compute_sensitivity(model, Δp; primal_vars, params)
        leq_locations, geq_locations = find_inequealities(cons)
        sa = stack_solution(cons, leq_locations, geq_locations, eval_model_jump(model, primal_vars, cons, params, p_a)...)
        # Check derivatives using finite differences
        ∂s_fd = FiniteDiff.finite_difference_jacobian((p) -> stack_solution(cons, leq_locations, geq_locations, eval_model_jump(model, primal_vars, cons, params, p)...), p_a)
        Δs_fd = ∂s_fd * Δp
        # actual solution
        sp = stack_solution(cons, leq_locations, geq_locations, eval_model_jump(model, primal_vars, cons, params, p_b)...)
        # Check sensitivities
        num_important = length(primal_vars) + length(cons)
        test_derivatives = all(isapprox.(Δs, Δs_fd; rtol = 1e-5, atol=1e-6))
        test_approx = all(isapprox.(sp[1:num_important], sp_approx[1:num_important]; rtol = 1e-5, atol=1e-6))
        if test_derivatives || (iscc && test_approx)
            println("All sensitivities are correct")
        elseif iscc && !test_approx
            @show Δp
            println("Fail Approximations")
            print_wrong_sensitive(Δs, sp.-sa, primal_vars, cons, leq_locations, geq_locations)
        else
            @show Δp
            print_wrong_sensitive(Δs, Δs_fd, primal_vars, cons, leq_locations, geq_locations)
        end
        println("--------------------")
    end
end