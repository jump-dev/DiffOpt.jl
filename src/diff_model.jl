"""
Constructs a Differentiable Optimizer model from a MOI Optimizer.
Supports `forward` and `backward` methods for solving and differentiating the model respectectively.

## Note 
Currently supports differentiating linear and quadratic programs only.
"""
function diff_model(_model::MOI.AbstractOptimizer)
    
    model = deepcopy(_model)

    Q, q, G, h, A, b, nz, var_idx, nineq, ineq_con_idx, neq, eq_con_idx = get_problem_data(model)
    
    z = zeros(0) # solution
    λ = zeros(0) # lagrangian variables
    ν = zeros(0)

    """
        Solving the convex optimization problem in forward pass
    """
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
    
        return z, λ, ν    
    end
    
    # """
    #     Method to differentiate and obtain gradients/jacobians
    #     of z, λ, ν  with respect to the parameters specified in
    #     in argument
    # """
    # function backward(params)
    #     grads = []
    #     LHS = create_LHS_matrix(z, λ, Q, G, h, A)

    #     # compute the jacobian of (z, λ, ν) with respect to each 
    #     # of the parameters recieved in the method argument
    #     # for instance, to get the jacobians w.r.t vector `b`
    #     # substitute db = I and set all other differential terms
    #     # in the right hand side to zero. For more info refer 
    #     # equation (6) of https://arxiv.org/pdf/1703.00443.pdf
    #     for param in params
    #         if param == "Q"
    #             RHS = create_RHS_matrix(z, ones(nz, nz), zeros(nz, 1),
    #                                     λ, zeros(nineq, nz), zeros(nineq,1),
    #                                     ν, zeros(neq, nz), zeros(neq, 1))
    #             push!(grads, LHS \ RHS)
    #         elseif param == "q"
    #             RHS = create_RHS_matrix(z, zeros(nz, nz), ones(nz, 1),
    #                                     λ, zeros(nineq, nz), zeros(nineq,1),
    #                                     ν, zeros(neq, nz), zeros(neq, 1))
    #             push!(grads, LHS \ RHS)
    #         elseif param == "G"
    #             RHS = create_RHS_matrix(z, zeros(nz, nz), zeros(nz, 1),
    #                                     λ, ones(nineq, nz), zeros(nineq,1),
    #                                     ν, zeros(neq, nz), zeros(neq, 1))
    #             push!(grads, LHS \ RHS)
    #         elseif param == "h"
    #             RHS = create_RHS_matrix(z, zeros(nz, nz), zeros(nz, 1), 
    #                                     λ, zeros(nineq, nz), ones(nineq,1),
    #                                     ν, zeros(neq, nz), zeros(neq, 1))
    #             push!(grads, LHS \ RHS)
    #         elseif param == "A"
    #             RHS = create_RHS_matrix(z, zeros(nz, nz), zeros(nz, 1), 
    #                                     λ, zeros(nineq, nz), zeros(nineq,1),
    #                                     ν, ones(neq, nz), zeros(neq, 1))
    #             push!(grads, LHS \ RHS)
    #         elseif param == "b"
    #             RHS = create_RHS_matrix(z, zeros(nz, nz), zeros(nz, 1), 
    #                                     λ, zeros(nineq, nz), zeros(nineq,1),
    #                                     ν, zeros(neq, nz), ones(neq, 1))
    #             push!(grads, LHS \ RHS)
    #         else
    #             push!(grads, [])
    #         end
    #     end
    #     return grads
    # end

    """
        Method to differentiate optimal solution `z` and return
        product of jacobian matrices (`dz / dQ`, `dz / dq`, etc) with 
        the backward pass vector `dl / dz`

        The method computes the product of 
        1. jacobian of problem solution `z*` with respect to 
           problem parameters `params` recieved as method arguments
        2. a backward pass vector `dl / dz`, where `l` can be a loss function

        Note that does not returns the actual jacobians

        For more info refer eqn(7) and eqn(8) of https://arxiv.org/pdf/1703.00443.pdf
    """
    function backward(params::Array{String}, dl_dz::Array{Float64})
        grads = []
        LHS = create_LHS_matrix(z, λ, Q, G, h, A)
        RHS = [dl_dz'; zeros(neq+nineq,1)]

        partial_grads = -(LHS \ RHS)

        dz = partial_grads[1:nz]
        if nineq > 0
            dλ = partial_grads[nz+1:nz+nineq]
        end
        if neq > 0
            dν = partial_grads[nz+nineq+1:nz+nineq+neq]
        end
        
        for param in params
            if param == "Q"
                push!(grads, 0.5 * (dz * z' + z * dz'))
            elseif param == "q"
                push!(grads, dz)
            elseif param == "G"
                push!(grads, Diagonal(λ) * dλ * z' - λ * dz')
            elseif param == "h"
                push!(grads, -Diagonal(λ) * dλ)
            elseif param == "A"
                push!(grads, dν * z' - ν * dz')
            elseif param == "b"
                push!(grads, -dν)
            else
                push!(grads, [])
            end
        end
        return grads
    end
    
    () -> (forward, backward)
end
