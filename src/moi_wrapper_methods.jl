# Methods in this file should move to their corresponding solver repositories
# Methods created only to make DiffOpt run with different solvers

# TODO: Remove these after https://github.com/jump-dev/Ipopt.jl/pull/204 is merged

using Ipopt

function MOI.get(
    model::Ipopt.Optimizer,
    ::MOI.ListOfConstraintIndices{MOI.ScalarAffineFunction{Float64}, MOI.LessThan{Float64}}
) 
    indices = MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.LessThan{Float64}}[]
    n = size(model.linear_le_constraints)[1]
    for i in 1:n 
        push!(indices, MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.LessThan{Float64}}(i))
    end
    return indices
end


function MOI.get(
    model::Ipopt.Optimizer,
    ::MOI.ListOfConstraintIndices{MOI.ScalarAffineFunction{Float64}, MOI.EqualTo{Float64}}
) 
    indices = MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.EqualTo{Float64}}[]
    n = size(model.linear_eq_constraints)[1]
    for i in 1:n 
        push!(indices, MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.EqualTo{Float64}}(i))
    end
    return indices
end


function MOI.get(
    model::Ipopt.Optimizer,
    ::MOI.ConstraintFunction,
    c::MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.LessThan{Float64}}
) 
    return model.linear_le_constraints[c.value].func
end


function MOI.get(
    model::Ipopt.Optimizer,
    ::MOI.ConstraintSet,
    c::MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.LessThan{Float64}}
) 
    return model.linear_le_constraints[c.value].set
end


function MOI.get(
    model::Ipopt.Optimizer,
    ::MOI.ConstraintFunction,
    c::MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.EqualTo{Float64}}
) 
    return model.linear_eq_constraints[c.value].func
end


function MOI.get(
    model::Ipopt.Optimizer,
    ::MOI.ConstraintSet,
    c::MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.EqualTo{Float64}}
) 
    return model.linear_eq_constraints[c.value].set
end


function MOI.get(
    model::Ipopt.Optimizer,
    ::MOI.ObjectiveFunction
)
    return model.objective
end
 