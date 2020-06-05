"""
Constructs a Differentiable Optimizer model from a MOI Optimizer.
Supports `forward` and `backward` methods for solving and differentiating the model respectectively.

## Note 
Currently supports differentiating linear and quadratic programs only.
"""
function diff_model(_model::MOI.AbstractOptimizer)
    
    model = deepcopy(_model)

    # variable indices
    var_idx = MOI.get(model, MOI.ListOfVariableIndices())
    nz      = size(var_idx)[1]
    
    eq_con_idx = MOI.get(model, MOI.ListOfConstraintIndices{MOI.ScalarAffineFunction{Float64}, MOI.EqualTo{Float64}}())
    ineq_con_idx = MOI.get(model, MOI.ListOfConstraintIndices{MOI.ScalarAffineFunction{Float64}, MOI.LessThan{Float64}}())

    neq = size(eq_con_idx)[1]
    nineq = size(ineq_con_idx)[1]
    
    
    # TODO: fix this method
    # Q,p,G,h,A,b = generate_matrices(model, ineq_con_idx, eq_con_idx, var_idx)
    
    z = zeros(0) # solution
    λ = zeros(0) # lagrangian variables
    ν = zeros(0)

    
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
