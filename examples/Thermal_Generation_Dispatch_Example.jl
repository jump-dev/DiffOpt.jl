# # Thermal Generation Dispatch Example

#md # [![](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](@__REPO_ROOT_URL__/examples/Thermal_Generation_Dispatch_Example.jl)

# This example illustrates the sensitivity analysis of thermal generation dispatch problem.

# This problem can be described as the choice of thermal generation `g` given a demand `d`, a price for thermal generation `c` and a penalty price `c_{ϕ}` for any demand not attended ϕ.

# ```math
# \begin{split}
# \begin{array} {ll}
# \mbox{minimize} & \sum_{i=1}^{N} c_{i} g_{i} + c_{ϕ} ϕ \\
# \mbox{s.t.} & g_{i} \ge 0 \quad i=1..N  \\
#             & g_{i} \le G_{i} \quad i=1..N  \\
#             & \sum_{i=1}^{N} g_{i} + ϕ = d\\
# \end{array}
# \end{split}
# ```
# where
# - `G_{i}` is the maximum possible generation for a thermal generator `i`

# ## Define and solve the Thermal Dispatch Problem

# First, import the libraries.

using DiffOpt
using GLPK
using MathOptInterface
using JuMP
using Test
using Plots
const MOI = MathOptInterface

# Define the model that will be construct given a set of parameters.

function GenerateModel(d::Float64; g_sup::Array{Float64,1}, c_g::Array{Float64,1}, c_ϕ::Float64)
    ## Creation of the Model and Parameters
    model = Model(() -> diff_optimizer(GLPK.Optimizer))
    I = length(g_sup)

    ## Variables
    @variable(model, g[i in  1:I] >= 0.0)
    @variable(model, ϕ >= 0.0)

    ## Constraints
    @constraint(model, limit_constraints_sup[i in 1:I], g[i] <= g_sup[i])
    @constraint(model, demand_constraint, sum(g) + ϕ == d)

    ## Objectives
    @objective(model, Min, c_g' * g + c_ϕ * ϕ)

    ## Solve the model
    optimize!(model)

    ## Return the solved model
    return model
end

# Define the functions that will get the primal values `g` and `\phi` and sensitivity analysis of the demand `dg/dd` and `d\phi/dd` from a optimized model.

function DiffOptForward(model::Model, ϵ::Float64 = 1.0)
    ## Initialization of parameters and references to simplify the notation
    vectRef = [model[:g]; model[:ϕ]]
    I = length(model[:g])

    ## Get the primal solution of the model
    vect =  MOI.get.(model, MOI.VariablePrimal(), vectRef)
     
    ## Pass the perturbation to the DiffOpt Framework and set the context to Forward
    constraint_equation = convert(MOI.ScalarAffineFunction{Float64}, ϵ)
    MOI.set(model, DiffOpt.ForwardInConstraint(), model[:demand_constraint], constraint_equation)
    DiffOpt.forward(model)
    
    ## Get the derivative of the model
    dvect = MOI.get.(model, DiffOpt.ForwardOutVariablePrimal(), vectRef)
    
    ## Return the values as a vector
    return [vect;dvect]
end

function DiffOptBackward(model::Model, ϵ::Float64 = 1.0)
    ## Initialization of parameters and references to simplify the notation
    vectRef = [model[:g]; model[:ϕ]]
    I = length(model[:g])

    ## Get the primal solution of the model
    vect =  MOI.get.(model, MOI.VariablePrimal(), vectRef)

    ## Set variables needed for the DiffOpt Backward Framework
    dvect = Array{Float64, 1}(undef, I + 1)
    perturbation = zeros(I + 1)

    ## Loop for each primal variable
    for i in 1:I+1
        ## Set the perturbation in the Primal Variables and set the context to Backward
        perturbation[i] = ϵ
        MOI.set.(model, DiffOpt.BackwardInVariablePrimal(), vectRef, perturbation)
        DiffOpt.backward(model)

        ## Get the value of the derivative of the model
        dvect[i] = JuMP.constant(MOI.get(model, DiffOpt.BackwardOutConstraint(), model[:demand_constraint]))
        perturbation[i] = 0.0
    end

    ## Return the values as a vector
    return [vect;dvect]
end

# Initialize of Parameters

g_sup = [10.0, 20.0, 30.0]
I = length(g_sup)
d = 0.0:0.1:80
dSize = length(d)
c_g = [1.0, 3.0, 5.0]
c_ϕ = 10.0

# Generate models for each demand `d`
models = GenerateModel.(d; g_sup = g_sup, c_g = c_g, c_ϕ = c_ϕ)

# Get the results of models with the DiffOpt Forward and Backward context

resultForward = DiffOptForward.(models)

resultBackward = DiffOptBackward.(models)

# Organization of results to plot
# Set dataResults array that will contain every result
dataResults = Array{Float64,3}(undef, 2, dSize, 2*(I+1))

# Populate the dataResults array
for k in 1:dSize
    dataResults[1,k,:] = resultForward[k]
    dataResults[2,k,:] = resultBackward[k]
end

# ## Results with Plot graphs
# ### Results for the forward context
# Result Primal Values:
plot(d,dataResults[1,:,1:I+1])

# Result Sensitivity Analysis:
plot(d,dataResults[1,:,I+2:2*(I+1)])

# ### Results for the backward context
# Result Primal Values:
plot(d,dataResults[2,:,1:I+1])

# Result Sensitivity Analysis:
plot(d,dataResults[2,:,I+2:2*(I+1)])
