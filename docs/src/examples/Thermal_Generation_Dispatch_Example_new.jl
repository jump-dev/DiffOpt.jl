# # Thermal Generation Dispatch Example

#md # [![](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](@__REPO_ROOT_URL__/examples/Thermal_Generation_Dispatch_Example.jl)

# This example illustrates the sensitivity analysis of thermal generation dispatch problem.

# This problem can be described as the choice of thermal generation `g` given a demand `d`, a price for thermal generation `c` and a penalty price `c_{ϕ}` for any demand not attended ϕ.

# ```math
# \begin{split}
# \begin{array} {ll}
# \mbox{minimize} & \sum_{i=1}^{N} c_{i} g_{i} + c_{\phi} \phi \\
# \mbox{s.t.} & g_{i} \ge 0 \quad i=1..N  \\
#             & g_{i} \le G_{i} \quad i=1..N  \\
#             & \sum_{i=1}^{N} g_{i} + \phi = d\\
# \end{array}
# \end{split}
# ```
# where
# - `G_{i}` is the maximum possible generation for a thermal generator `i`

# ## Define and solve the Thermal Dispatch Problem

# First, import the libraries.

using Test
using JuMP
import DiffOpt
import LinearAlgebra: dot
import HiGHS
import MathOptInterface as MOI
import Plots

# Define the model that will be construct given a set of parameters.

function generate_model(
    d_data::Float64;
    g_sup::Vector{Float64},
    c_g::Vector{Float64},
    c_ϕ::Float64,
)
    ## Creation of the Model and Parameters
    model = DiffOpt.quadratic_diff_model(HiGHS.Optimizer)
    set_silent(model)
    I = length(g_sup)

    ## Variables
    @variable(model, g[i in 1:I] >= 0.0)
    @variable(model, ϕ >= 0.0)

    ## Parameters
    @variable(model, d in Parameter(d_data))

    ## Constraints
    @constraint(model, limit_constraints_sup[i in 1:I], g[i] <= g_sup[i])
    @constraint(model, demand_constraint, sum(g) + ϕ == d)

    ## Objectives
    @objective(model, Min, dot(c_g, g) + c_ϕ * ϕ)

    ## Solve the model
    optimize!(model)

    ## Return the solved model
    return model
end

# Define the functions that will get the primal values `g` and `\phi` and sensitivity analysis of the demand `dg/dd` and `d\phi/dd` from a optimized model.

function diff_forward(model::Model, ϵ::Float64 = 1.0)
    ## Initialization of parameters and references to simplify the notation
    vect_ref = [model[:g]; model[:ϕ]]

    ## Get the primal solution of the model
    vect = value.(vect_ref)

    ## Reset the sensitivities of the model
    DiffOpt.empty_input_sensitivities!(model)

    ## Pass the perturbation to the DiffOpt Framework
    DiffOpt.set_forward_parameter(model, model[:d], ϵ)

    ## Compute the derivatives with the Forward Mode
    DiffOpt.forward_differentiate!(model)

    ## Get the derivative of the model
    dvect = DiffOpt.get_forward_variable.(model, vect_ref)

    ## Return the values as a vector
    return [vect; dvect]
end

function diff_reverse(model::Model, ϵ::Float64 = 1.0)
    ## Initialization of parameters and references to simplify the notation
    vect_ref = [model[:g]; model[:ϕ]]

    ## Get the primal solution of the model
    vect = value.(vect_ref)

    ## Set variables needed for the DiffOpt Backward Framework
    dvect = Array{Float64,1}(undef, length(vect_ref))

    ## Loop for each primal variable
    for i in 1:(I+1)
        ## Reset the sensitivities of the model
        DiffOpt.empty_input_sensitivities!(model)

        ## Pass the perturbation to the DiffOpt Framework
        DiffOpt.set_reverse_variable.(model, vect_ref[i], ϵ)

        ## Compute the derivatives with the Forward Mode
        DiffOpt.reverse_differentiate!(model)

        ## Get the derivative of the model
        dvect[i] = DiffOpt.get_reverse_parameter(model, model[:d])
    end

    ## Return the values as a vector
    return [vect; dvect]
end

# Initialize of Parameters

g_sup = [10.0, 20.0, 30.0]
I = length(g_sup)
d = 0.0:0.1:80
d_size = length(d)
c_g = [1.0, 3.0, 5.0]
c_ϕ = 10.0;

# Generate models for each demand `d`
models = generate_model.(d; g_sup = g_sup, c_g = c_g, c_ϕ = c_ϕ);

# Get the results of models with the DiffOpt Forward and Backward context

result_forward = diff_forward.(models)
optimize!.(models)
result_reverse = diff_reverse.(models);

# Organization of results to plot
# Initialize data_results that will contain every result
data_results = Array{Float64,3}(undef, 2, d_size, 2 * (I + 1));

# Populate the data_results array
for k in 1:d_size
    data_results[1, k, :] = result_forward[k]
    data_results[2, k, :] = result_reverse[k]
end

# ## Results with Plot graphs
# ### Results for the forward context
# Result Primal Values:
Plots.plot(
    d,
    data_results[1, :, 1:(I+1)];
    title = "Generation by Demand",
    label = ["Thermal Generation 1" "Thermal Generation 2" "Thermal Generation 3" "Generation Deficit"],
    xlabel = "Demand [unit]",
    ylabel = "Generation [unit]",
)

# Result Sensitivity Analysis:
Plots.plot(
    d,
    data_results[1, :, (I+2):(2*(I+1))];
    title = "Sensitivity of Generation by Demand",
    label = ["T. Gen. 1 Sensitivity" "T. Gen. 2 Sensitivity" "T. Gen. 3 Sensitivity" "Gen. Deficit Sensitivity"],
    xlabel = "Demand [unit]",
    ylabel = "Sensitivity [-]",
)

# ### Results for the reverse context
# Result Primal Values:
Plots.plot(
    d,
    data_results[2, :, 1:(I+1)];
    title = "Generation by Demand",
    label = ["Thermal Generation 1" "Thermal Generation 2" "Thermal Generation 3" "Generation Deficit"],
    xlabel = "Demand [unit]",
    ylabel = "Generation [unit]",
)

# Result Sensitivity Analysis:
Plots.plot(
    d,
    data_results[2, :, (I+2):(2*(I+1))];
    title = "Sensitivity of Generation by Demand",
    label = ["T. Gen. 1 Sensitivity" "T. Gen. 2 Sensitivity" "T. Gen. 3 Sensitivity" "Gen. Deficit Sensitivity"],
    xlabel = "Demand [unit]",
    ylabel = "Sensitivity [-]",
)
