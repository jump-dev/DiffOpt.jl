using JuMP
using Ipopt

################################################
#=
From JuMP Tutorial for Querying Hessians:
https://github.com/jump-dev/JuMP.jl/blob/301d46e81cb66c74c6e22cd89fb89ced740f157b/docs/src/tutorials/nonlinear/querying_hessians.jl#L67-L72
=#
################################################
function create_nonlinear_jump_model(; ismin = true)
    model = Model(() -> DiffOpt.diff_optimizer(Ipopt.Optimizer))
    set_silent(model)
    @variable(model, p ∈ MOI.Parameter(1.0))
    @variable(model, p2 ∈ MOI.Parameter(2.0))
    @variable(model, p3 ∈ MOI.Parameter(100.0))
    @variable(model, x[i = 1:2], start = -i)
    @constraint(model, g_1, x[1]^2 <= p)
    @constraint(model, g_2, p * (x[1] + x[2])^2 <= p2)
    if ismin
        @objective(model, Min, (1 - x[1])^2 + p3 * (x[2] - x[1]^2)^2)
    else
        @objective(model, Max, -(1 - x[1])^2 - p3 * (x[2] - x[1]^2)^2)
    end

    return model, x, [g_1; g_2], [p; p2; p3]
end

################################################
#=
From sIpopt paper: https://optimization-online.org/2011/04/3008/
=#
################################################

function create_nonlinear_jump_model_sipopt(; ismin = true)
    model = Model(() -> DiffOpt.diff_optimizer(Ipopt.Optimizer))
    set_silent(model)
    @variable(model, p1 ∈ MOI.Parameter(4.5))
    @variable(model, p2 ∈ MOI.Parameter(1.0))
    @variable(model, x[i = 1:3] >= 0, start = -i)
    @constraint(model, g_1, 6 * x[1] + 3 * x[2] + 2 * x[3] - p1 == 0)
    @constraint(model, g_2, p2 * x[1] + x[2] - x[3] - 1 == 0)
    if ismin
        @objective(model, Min, x[1]^2 + x[2]^2 + x[3]^2)
    else
        @objective(model, Max, -x[1]^2 - x[2]^2 - x[3]^2)
    end
    return model, x, [g_1; g_2], [p1; p2]
end

################################################
#=
Simple Problems
=#
################################################

function create_jump_model_1(p_val = [1.5])
    model = Model(() -> DiffOpt.diff_optimizer(Ipopt.Optimizer))
    set_silent(model)

    # Parameters
    @variable(model, p ∈ MOI.Parameter(p_val[1]))

    # Variables
    @variable(model, x)

    # Constraints
    @constraint(model, con1, x >= p)
    @constraint(model, con2, x >= 2)
    @objective(model, Min, x^2)

    return model, [x], [con1; con2], [p]
end

function create_jump_model_2(p_val = [1.5])
    model = Model(() -> DiffOpt.diff_optimizer(Ipopt.Optimizer))
    set_silent(model)

    # Parameters
    @variable(model, p ∈ MOI.Parameter(p_val[1]))

    # Variables
    @variable(model, x >= 2.0)

    # Constraints
    @constraint(model, con1, x >= p)
    @objective(model, Min, x^2)

    return model, [x], [con1], [p]
end

function create_jump_model_3(p_val = [-1.5])
    model = Model(() -> DiffOpt.diff_optimizer(Ipopt.Optimizer))
    set_silent(model)

    # Parameters
    @variable(model, p ∈ MOI.Parameter(p_val[1]))

    # Variables
    @variable(model, x)

    # Constraints
    @constraint(model, con1, x <= p)
    @constraint(model, con2, x <= -2)
    @objective(model, Min, -x)

    return model, [x], [con1; con2], [p]
end

function create_jump_model_4(p_val = [1.5])
    model = Model(() -> DiffOpt.diff_optimizer(Ipopt.Optimizer))
    set_silent(model)

    # Parameters
    @variable(model, p ∈ MOI.Parameter(p_val[1]))

    # Variables
    @variable(model, x)

    # Constraints
    @constraint(model, con1, x <= p)
    @constraint(model, con2, x <= 2)
    @objective(model, Max, x)

    return model, [x], [con1; con2], [p]
end

function create_jump_model_5(p_val = [1.5])
    model = Model(() -> DiffOpt.diff_optimizer(Ipopt.Optimizer))
    set_silent(model)

    # Parameters
    @variable(model, p ∈ MOI.Parameter(p_val[1]))

    # Variables
    @variable(model, x)

    # Constraints
    @constraint(model, con1, x >= p)
    @constraint(model, con2, x >= 2)
    @objective(model, Max, -x)

    return model, [x], [con1; con2], [p]
end

# Softmax model
h(y) = -sum(y .* log.(y))
softmax(x) = exp.(x) / sum(exp.(x))
function create_jump_model_6(p_a = collect(1.0:0.1:2.0))
    model = Model(() -> DiffOpt.diff_optimizer(Ipopt.Optimizer))
    set_silent(model)

    # Parameters
    @variable(model, x[i = 1:length(p_a)] ∈ MOI.Parameter.(p_a))

    # Variables
    @variable(model, y[1:length(p_a)] >= 0.0)

    # Constraints
    @constraint(model, con1, sum(y) == 1)
    @constraint(model, con2[i = 1:length(x)], y[i] <= 1)

    # Objective
    @objective(model, Max, dot(x, y) + h(y))

    return model, y, [con1; con2], x
end

function create_jump_model_7(p_val = [1.5], g = sin)
    model = Model(() -> DiffOpt.diff_optimizer(Ipopt.Optimizer))
    set_silent(model)

    # Parameters
    @variable(model, p ∈ MOI.Parameter(p_val[1]))

    # Variables
    @variable(model, x)

    # Constraints
    @constraint(model, con1, x * g(p) == 1)
    @objective(model, Min, 0)

    return model, [x], [con1], [p]
end

################################################
#=
Non Linear Problems
=#
################################################

function create_nonlinear_jump_model_1(p_val = [1.0; 2.0; 100]; ismin = true)
    model = Model(() -> DiffOpt.diff_optimizer(Ipopt.Optimizer))
    set_silent(model)

    # Parameters
    @variable(model, p[i = 1:3] ∈ MOI.Parameter.(p_val))

    # Variables
    @variable(model, x)
    @variable(model, y)

    # Constraints
    @constraint(model, con1, y >= p[1] * sin(x)) # NLP Constraint
    @constraint(model, con2, x + y == p[1])
    @constraint(model, con3, p[2] * x >= 0.1)
    if ismin
        @objective(model, Min, (1 - x)^2 + p[3] * (y - x^2)^2) # NLP Objective
    else
        @objective(model, Max, -(1 - x)^2 - p[3] * (y - x^2)^2) # NLP Objective
    end

    return model, [x; y], [con1; con2; con3], p
end

function create_nonlinear_jump_model_2(p_val = [3.0; 2.0; 10]; ismin = true)
    model = Model(() -> DiffOpt.diff_optimizer(Ipopt.Optimizer))
    set_silent(model)

    # Parameters
    @variable(model, p[i = 1:3] ∈ MOI.Parameter.(p_val))

    # Variables
    @variable(model, x <= 10)
    @variable(model, y)

    # Constraints
    @constraint(model, con1, y >= p[1] * sin(x)) # NLP Constraint
    @constraint(model, con2, x + y == p[1])
    @constraint(model, con3, p[2] * x >= 0.1)
    if ismin
        @objective(model, Min, (1 - x)^2 + p[3] * (y - x^2)^2) # NLP Objective
    else
        @objective(model, Max, -(1 - x)^2 - p[3] * (y - x^2)^2) # NLP Objective
    end

    return model, [x; y], [con1; con2; con3], p
end

function create_nonlinear_jump_model_3(p_val = [3.0; 2.0; 10]; ismin = true)
    model = Model(() -> DiffOpt.diff_optimizer(Ipopt.Optimizer))
    set_silent(model)

    # Parameters
    @variable(model, p[i = 1:3] ∈ MOI.Parameter.(p_val))

    # Variables
    @variable(model, x <= 10)
    @variable(model, y)

    # Constraints
    @constraint(model, con1, y >= p[1] * sin(x)) # NLP Constraint
    @constraint(model, con2, x + y == p[1])
    @constraint(model, con3, p[2] * x >= 0.1)
    if ismin
        @objective(model, Min, (1 - x)^2 + p[3] * (y - x^2)^2) # NLP Objective
    else
        @objective(model, Max, -(1 - x)^2 - p[3] * (y - x^2)^2) # NLP Objective
    end
    return model, [x; y], [con1; con2; con3], p
end

function create_nonlinear_jump_model_4(p_val = [1.0; 2.0; 100]; ismin = true)
    model = Model(() -> DiffOpt.diff_optimizer(Ipopt.Optimizer))
    set_silent(model)

    # Parameters
    @variable(model, p[i = 1:3] ∈ MOI.Parameter.(p_val))

    # Variables
    @variable(model, x)
    @variable(model, y)

    # Constraints
    @constraint(model, con0, x == p[1] - 0.5)
    @constraint(model, con1, y >= p[1] * sin(x)) # NLP Constraint
    @constraint(model, con2, x + y == p[1])
    @constraint(model, con3, p[2] * x >= 0.1)
    if ismin
        @objective(model, Min, (1 - x)^2 + p[3] * (y - x^2)^2) # NLP Objective
    else
        @objective(model, Max, -(1 - x)^2 - p[3] * (y - x^2)^2) # NLP Objective
    end

    return model, [x; y], [con1; con2; con3], p
end

function create_nonlinear_jump_model_5(p_val = [1.0; 2.0; 100]; ismin = true)
    model = Model(() -> DiffOpt.diff_optimizer(Ipopt.Optimizer))
    set_silent(model)

    # Parameters
    @variable(model, p[i = 1:3] ∈ MOI.Parameter.(p_val))

    # Variables
    @variable(model, x)
    @variable(model, y)

    # Constraints
    fix(x, 0.5)
    con0 = JuMP.FixRef(x)
    @constraint(model, con1, y >= p[1] * sin(x)) # NLP Constraint
    @constraint(model, con2, x + y == p[1])
    @constraint(model, con3, p[2] * x >= 0.1)
    if ismin
        @objective(model, Min, (1 - x)^2 + p[3] * (y - x^2)^2) # NLP Objective
    else
        @objective(model, Max, -(1 - x)^2 - p[3] * (y - x^2)^2) # NLP Objective
    end

    return model, [x; y], [con0; con1; con2; con3], p
end

function create_nonlinear_jump_model_6(p_val = [100.0; 200.0]; ismin = true)
    model = Model(() -> DiffOpt.diff_optimizer(Ipopt.Optimizer))
    set_silent(model)

    # Parameters
    @variable(model, p[i = 1:2] ∈ MOI.Parameter.(p_val))

    # Variables
    @variable(model, x[i = 1:2])
    @variable(model, z) # >= 2.0)
    @variable(model, w) # <= 3.0)
    # @variable(model, f[1:2])

    # Constraints
    @constraint(
        model,
        con1,
        x[2] - 0.0001 * x[1]^2 - 0.2 * z^2 - 0.3 * w^2 >= p[1] + 1
    )
    @constraint(
        model,
        con2,
        x[1] + 0.001 * x[2]^2 + 0.5 * w^2 + 0.4 * z^2 <= 10 * p[1] + 2
    )
    @constraint(model, con3, z^2 + w^2 == 13)
    if ismin
        @objective(model, Min, x[2] - x[1] + z - w)
    else
        @objective(model, Max, -x[2] + x[1] - z + w)
    end

    return model, [x; z; w], [con2; con3], p
end
