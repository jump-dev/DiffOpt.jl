using DiffOpt
using GLPK
using MathOptInterface
using JuMP
using Test
using Plots
const MOI = MathOptInterface

#Example of simple Thermal Generation Dispatch and the derivatives of the generation by the demand (dg/dd)

#Simple Model of Thermal Generation Dispatch 
function GenerateModel(d::Float64; g_sup::Array{Float64,1}, c_g::Array{Float64,1}, c_Df::Float64)
    #Creation of the Model and Parameters
    model = Model(() -> diff_optimizer(GLPK.Optimizer))
    I = length(g_sup)

    #Variables
    @variable(model, g[i in  1:I] >= 0.0)
    @variable(model, Df >= 0.0)

    #Constraints
    @constraint(model, limit_constraints_sup[i in 1:I], g[i] <= g_sup[i])
    @constraint(model, demand_constraint, sum(g) + Df == d)

    #Objectives
    @objective(model, Min, c_g' * g + c_Df * Df)

    #Solve the model
    optimize!(model)

    #Return the solved model
    return model
end

#DiffOpt Forward for the calculation of dg/dd
#It uses a solved (optimized) model created by the function GenerateProblem
function DiffOptForward(model::Model, ϵ::Float64 = 1.0)
    #Initialization of parameters and references to simplify the notation
    vectRef = [model[:g]; model[:Df]]
    I = length(model[:g])

    #Get the primal solution of the model
    vect =  MOI.get.(model, MOI.VariablePrimal(), vectRef)
     
    #Pass the perturbation to the DiffOpt Framework and set the context to Forward
    constraint_equation = convert(MOI.ScalarAffineFunction{Float64}, ϵ)
    MOI.set(model, DiffOpt.ForwardInConstraint(), model[:demand_constraint], constraint_equation)
    DiffOpt.forward(model)
    
    #Get the derivative of the model
    dvect = MOI.get.(model, DiffOpt.ForwardOutVariablePrimal(), vectRef)
    
    #Return the values as a vector
    return [vect;dvect]
end

function DiffOptBackward(model::Model, ϵ::Float64 = 1.0)
    #Initialization of parameters and references to simplify the notation
    vectRef = [model[:g]; model[:Df]]
    I = length(model[:g])

    #Get the primal solution of the model
    vect =  MOI.get.(model, MOI.VariablePrimal(), vectRef)

    #Set variables needed for the DiffOpt Backward Framework
    dvect = Array{Float64, 1}(undef, I + 1)
    perturbation = zeros(I + 1)

    #Loop for each primal variable
    for i in 1:I+1
        #Set the perturbation in the Primal Variables and set the context to Backward
        perturbation[i] = ϵ
        MOI.set.(model, DiffOpt.BackwardInVariablePrimal(), vectRef, perturbation)
        DiffOpt.backward(model)

        #Get the value of the derivative of the model
        dvect[i] = JuMP.constant(MOI.get(model, DiffOpt.BackwardOutConstraint(), model[:demand_constraint]))
        perturbation[i] = 0.0
    end

    #Return the values as a vector
    return [vect;dvect]
end

#Initialization of Parameters
g_sup = [10.0, 20.0, 30.0]
I = length(g_sup)
d = 0.0:0.1:80
dSize = length(d)
c_g = [1.0, 3.0, 5.0]
c_Df = 10.0

#Generate model for each demand d
models = GenerateModel.(d; g_sup = g_sup, c_g = c_g, c_Df = c_Df)

#Get the results of models with the DiffOpt Forward context
resultForward = DiffOptForward.(models)

#Get the results of models with the DiffOpt Backward context
resultBackward = DiffOptBackward.(models)


#Organization of results to plot
#Set dataResults array that will contain every result
dataResults = Array{Float64,3}(undef, 2, dSize, 2*(I+1))

#Populate the dataResults array
for k in 1:dSize
    dataResults[1,k,:] = resultForward[k]
    dataResults[2,k,:] = resultBackward[k]
end

#Plot graphs of the different contexts
plot(d,dataResults[1,:,1:I+1])
savefig("ResultsForward_Primal")
plot(d,dataResults[1,:,I+2:2*(I+1)])
savefig("ResultsForward_Diff")

plot(d,dataResults[2,:,1:I+1])
savefig("ResultsBackward_Primal")
plot(d,dataResults[2,:,I+2:2*(I+1)])
savefig("ResultsBackward_Diff")

#END
