mutable struct ObjectiveContainer{MOIForm,ArrayForm} <: MOI.ModelLike
    num_variables::Int64
    is_sense_set::Bool
    sense::MOI.OptimizationSense
    is_function_set::Bool
    func::Union{Nothing,ArrayForm}
    function ObjectiveContainer{MOIForm,ArrayForm}() where {MOIForm,ArrayForm}
        o = new{MOIForm,ArrayForm}()
        MOI.empty!(o)
        return o
    end
end

function MOI.empty!(o::ObjectiveContainer)
    o.num_variables = 0
    o.is_sense_set = false
    o.sense = MOI.FEASIBILITY_SENSE
    o.is_function_set = false
    o.func = nothing
    return
end

function MOI.is_empty(o::ObjectiveContainer)
    return !o.is_sense_set && !o.is_function_set
end

# FIXME type piracy
function MOI.add_variable(model::MOI.Utilities.AbstractModel)
    x = MOI.add_variable(model.variables)
    MOI.Utilities._add_variable(model.objective)
    MOI.Utilities._add_variable(model.constraints)
    return x
end
function MOI.Utilities._add_variable(::MOI.Utilities.ObjectiveContainer) end
function MOI.Utilities._add_variable(model::ObjectiveContainer)
    if !model.is_function_set
        error("Adding variables after setting objective is not implemented yet")
    end
    model.num_variables += 1
    return
end

###
### ObjectiveSense
###

MOI.supports(::ObjectiveContainer, ::MOI.ObjectiveSense) = true

MOI.get(o::ObjectiveContainer, ::MOI.ObjectiveSense) = o.sense

function MOI.set(o::ObjectiveContainer, ::MOI.ObjectiveSense, value)
    if value == MOI.FEASIBILITY_SENSE
        MOI.empty!(o)
    end
    o.is_sense_set = true
    o.sense = value
    return
end

###
### ObjectiveFunctionType
###

function MOI.get(
    ::ObjectiveContainer{MOIForm},
    ::MOI.ObjectiveFunctionType,
) where {MOIForm}
    return MOIForm
end

###
### ObjectiveFunction
###

function MOI.supports(
    ::ObjectiveContainer{MOIForm},
    ::MOI.ObjectiveFunction{MOIForm},
) where {MOIForm}
    return true
end

function MOI.get(
    o::ObjectiveContainer{MOIForm},
    ::MOI.ObjectiveFunction{MOIForm},
) where {MOIForm}
    return convert(MOIForm, o.func)
end

function MOI.set(
    o::ObjectiveContainer,
    ::MOI.ObjectiveFunction{MOIForm},
    f::MOIForm,
) where {MOIForm}
    o.is_function_set = true
    o.func = sparse_array_representation(f, o.num_variables)
    return
end

###
### MOI.ListOfModelAttributesSet
###

function MOI.get(o::ObjectiveContainer, ::MOI.ListOfModelAttributesSet)
    ret = MOI.AbstractModelAttribute[]
    if o.is_sense_set
        push!(ret, MOI.ObjectiveSense())
    end
    if o.is_function_set
        F = MOI.get(o, MOI.ObjectiveFunctionType())
        push!(ret, MOI.ObjectiveFunction{MOIForm}())
    end
    return ret
end
