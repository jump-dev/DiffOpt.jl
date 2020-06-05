module DiffOpt

using Random
using LinearAlgebra
using MathOptInterface

const MOI = MathOptInterface;
const MOIU = MathOptInterface.Utilities;

include("./gen_random_problem.jl")
include("./utils.jl")


function DiffModel(_model::MOI.AbstractOptimizer, con_idx)
    
    model = deepcopy(_model)

    # variable indices
    var_idx = MOI.get(model, MOI.ListOfVariableIndices())
    nz      = size(var_idx)[1]
    
    # TODO: both Ipopt, OSQP dont support `ListOfConstraintIndices`
    #       attribute as of now. so passing constraint indices manually
    eq_con_idx   = MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},MOI.EqualTo{Float64}}[]
    ineq_con_idx = MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},MOI.LessThan{Float64}}[]

    # split into equality and inequality constraints
    for con in con_idx
        con_set = MOI.get(model, MOI.ConstraintSet(), con)
        if is_equality(con_set)
            push!(eq_con_idx, con)
        else
            push!(ineq_con_idx, con)
        end
    end

    neq   = size(eq_con_idx)[1]
    nineq = size(ineq_con_idx)[1]
    
    
    # TODO: fix this method
    # Q,p,G,h,A,b = generate_matrices(model, ineq_con_idx, eq_con_idx, var_idx)
    
    z::Array{Float64} = zeros(0) # solution
    λ::Array{Float64} = zeros(0) # lagrangian variables
    ν::Array{Float64} = zeros(0)

    
    function forward()
        # solve the model
        MOI.optimize!(model)
        
        # check status
        @assert MOI.get(model, MOI.TerminationStatus()) in [MOI.LOCALLY_SOLVED, MOI.OPTIMAL]
        
        # get and save the solution
        z = MOI.get(model, MOI.VariablePrimal(), var_idx)
        
        # get and save dual variables
        λ = MOI.get(model, MOI.ConstraintDual(), ineq_con_idx)
    
        if neq > 0
            ν = MOI.get(model, MOI.ConstraintDual(), eq_con_idx)
        end
    
        return z
    end
    
    function backward(params::Array{String}, values::Array{Array{Float64}})
        @assert size(params)[1] == size(values)[1]
        @assert size(params)[1] > 0
        
        # TODO: fix this method
    end
    
    () -> (forward, backward)
end


export DiffModel
export generate_lp, generate_qp

end # module
