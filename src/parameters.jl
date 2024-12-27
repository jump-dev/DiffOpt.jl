# Copyright (c) 2020: Akshay Sharma and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

# block other methods

MOI.supports(::POI.Optimizer, ::ForwardObjectiveFunction) = false

function MOI.set(
    ::POI.Optimizer,
    ::ForwardObjectiveFunction,
    _,
)
    error(
        "Forward objective function is not supported when " *
        "`with_parametric_opt_interface` is set to `true` in " *
        "`diff_optimizer`." *
        "Use parameters to set the forward sensitivity.",
    )
end

MOI.supports(::POI.Optimizer, ::ForwardConstraintFunction) = false

function MOI.set(
    ::POI.Optimizer,
    ::ForwardConstraintFunction,
    ::MOI.ConstraintIndex,
    _,
)
    error(
        "Forward constraint function is not supported when " *
        "`with_parametric_opt_interface` is set to `true` in " *
        "`diff_optimizer`." *
        "Use parameters to set the forward sensitivity.",
    )
end

MOI.supports(::POI.Optimizer, ::ReverseObjectiveFunction) = false

function MOI.get(
    ::POI.Optimizer,
    ::ReverseObjectiveFunction,
)
    error(
        "Reverse objective function is not supported when " *
        "`with_parametric_opt_interface` is set to `true` in " *
        "`diff_optimizer`." *
        "Use parameters to get the reverse sensitivity.",
    )
end

MOI.supports(::POI.Optimizer, ::ReverseConstraintFunction) = false

function MOI.get(
    ::POI.Optimizer,
    ::ReverseConstraintFunction,
    ::MOI.ConstraintIndex,
)
    error(
        "Reverse constraint function is not supported when " *
        "`with_parametric_opt_interface` is set to `true` in " *
        "`diff_optimizer`." *
        "Use parameters to get the reverse sensitivity.",
    )
end

# functions to be used with ParametricOptInterface.jl

struct ForwardConstraintSet <: MOI.AbstractConstraintAttribute end

struct ReverseConstraintSet <: MOI.AbstractConstraintAttribute end

mutable struct SensitivityData{T}
    parameter_input_forward::Dict{MOI.VariableIndex,T}
    parameter_output_backward::Dict{MOI.VariableIndex,T}
end

function SensitivityData{T}() where {T}
    return SensitivityData{T}(Dict{MOI.VariableIndex,T}(), Dict{MOI.VariableIndex,T}())
end

function _get_sensitivity_data(model::POI.Optimizer{T})::SensitivityData{T} where {T}
    _initialize_sensitivity_data!(model)
    return model.ext[:_sensitivity_data]::SensitivityData{T}
end

function _initialize_sensitivity_data!(model::POI.Optimizer{T}) where {T}
    if !haskey(model.ext, :_sensitivity_data)
        model.ext[:_sensitivity_data] = SensitivityData{T}()
    end
    return
end

const DoubleDictInner = MOI.Utilities.DoubleDicts.DoubleDictInner

# forward mode

function _constraint_set_forward!(
    model::POI.Optimizer{T},
    affine_constraint_cache_dict,
    ::Type{P},
) where {T, P <: POI.ParametricAffineFunction}
    sensitivity_data = _get_sensitivity_data(model)
    for (inner_ci, pf) in affine_constraint_cache_dict
        cte = zero(T)
        terms = MOI.ScalarAffineTerm{T}[]
        sizehint!(terms, 0)
        for term in POI.affine_parameter_terms(pf)
            p = term.variable
            sensitivity = get(sensitivity_data.parameter_input_forward, p, 0.0)
            cte += sensitivity * term.coefficient
        end
        if !iszero(cte)
            MOI.set(
                model.optimizer,
                ForwardConstraintFunction(),
                inner_ci,
                MOI.ScalarAffineFunction{T}(terms, cte),
            )
        end
    end
    return
end

function _constraint_set_forward!(
    model::POI.Optimizer{T},
    vector_affine_constraint_cache_dict,
    ::Type{P},
) where {T, P <: POI.ParametricVectorAffineFunction}
    sensitivity_data = _get_sensitivity_data(model)
    for (inner_ci, pf) in vector_affine_constraint_cache_dict
        cte = zeros(T, length(pf.c))
        terms = MOI.VectorAffineTerm{T}[]
        sizehint!(terms, 0)
        for term in POI.vector_affine_parameter_terms(pf)
            p = term.scalar_term.variable
            sensitivity = get(sensitivity_data.parameter_input_forward, p, 0.0)
            cte[term.output_index] += sensitivity * term.scalar_term.coefficient
        end
        if !iszero(cte)
            MOI.set(
                model.optimizer,
                ForwardConstraintFunction(),
                inner_ci,
                MOI.VectorAffineFunction{T}(terms, cte),
            )
        end
    end
    return
end

function _constraint_set_forward!(
    model::POI.Optimizer{T},
    quadratic_constraint_cache_dict,
    ::Type{P},
) where {T, P <: POI.ParametricQuadraticFunction}
    sensitivity_data = _get_sensitivity_data(model)
    for (inner_ci, pf) in quadratic_constraint_cache_dict
        cte = zero(T)
        terms = MOI.ScalarAffineTerm{T}[]
        for term in POI.affine_parameter_terms(pf)
            p = term.variable
            sensitivity = get(sensitivity_data.parameter_input_forward, p, 0.0)
            cte += sensitivity * term.coefficient
        end
        for term in POI.quadratic_parameter_parameter_terms(pf)
            p_1 = term.variable_1
            p_2 = term.variable_2
            sensitivity_1 = get(sensitivity_data.parameter_input_forward, p_1, 0.0)
            sensitivity_2 = get(sensitivity_data.parameter_input_forward, p_2, 0.0)
            cte +=
                sensitivity_1 * term.coefficient *
                MOI.get(model, MOI.VariablePrimal(), p_2) /
                ifelse(term.variable_1 === term.variable_2, 2, 1)
            cte +=
                sensitivity_2 * term.coefficient *
                MOI.get(model, MOI.VariablePrimal(), p_1) /
                ifelse(term.variable_1 === term.variable_2, 2, 1)
        end
        sizehint!(terms, length(POI.quadratic_parameter_variable_terms(pf)))
        for term in POI.quadratic_parameter_variable_terms(pf)
            p = term.variable_1
            sensitivity = get(sensitivity_data.parameter_input_forward, p, NaN)
            if !isnan(sensitivity)
                push!(
                    terms,
                    MOI.ScalarAffineTerm{T}(
                        sensitivity * term.coefficient,
                        term.variable_2,
                    ),
                )
            end
        end
        if !iszero(cte) || !isempty(terms)
            MOI.set(
                model.optimizer,
                ForwardConstraintFunction(),
                inner_ci,
                MOI.ScalarAffineFunction{T}(terms, cte),
            )
        end
    end
    return
end

function _affine_objective_set_forward!(model::POI.Optimizer{T}) where {T}
    cte = zero(T)
    terms = MOI.ScalarAffineTerm{T}[]
    pf = model.affine_objective_cache
    sizehint!(terms, 0)
    sensitivity_data = _get_sensitivity_data(model)
    for term in POI.affine_parameter_terms(pf)
        p = term.variable
        sensitivity = get(sensitivity_data.parameter_input_forward, p, 0.0)
        cte += sensitivity * term.coefficient
    end
    if !iszero(cte)
        MOI.set(
            model.optimizer,
            ForwardObjectiveFunction(),
            MOI.ScalarAffineFunction{T}(terms, cte),
        )
    end
    return
end

function _quadratic_objective_set_forward!(model::POI.Optimizer{T}) where {T}
    cte = zero(T)
    pf = MOI.get(model, POI.ParametricObjectiveFunction{POI.ParametricQuadraticFunction{T}}())
    sensitivity_data = _get_sensitivity_data(model)
    for term in POI.affine_parameter_terms(pf)
        p = term.variable
        sensitivity = get(sensitivity_data.parameter_input_forward, p, 0.0)
        cte += sensitivity * term.coefficient
    end
    for term in POI.quadratic_parameter_parameter_terms(pf)
        p_1 = term.variable_1
        p_2 = term.variable_2
        sensitivity_1 = get(sensitivity_data.parameter_input_forward, p_1, 0.0)
        sensitivity_2 = get(sensitivity_data.parameter_input_forward, p_2, 0.0)
        cte +=
            sensitivity_1 * term.coefficient *
            MOI.get(model, MOI.VariablePrimal(), p_2) /
            ifelse(term.variable_1 === term.variable_2, 2, 1)
        cte +=
            sensitivity_2 * term.coefficient
            MOI.get(model, MOI.VariablePrimal(), p_1) /
            ifelse(term.variable_1 === term.variable_2, 2, 1)
    end
    terms = MOI.ScalarAffineTerm{T}[]
    sizehint!(terms, length(POI.quadratic_parameter_variable_terms(pf)))
    for term in POI.quadratic_parameter_variable_terms(pf)
        p = term.variable_1
        sensitivity = get(sensitivity_data.parameter_input_forward, p, NaN)
        if !isnan(sensitivity)
            push!(
                terms,
                MOI.ScalarAffineTerm{T}(
                    sensitivity * term.coefficient,
                    term.variable_2,
                ),
            )
        end
    end
    if !iszero(cte) || !isempty(terms)
        MOI.set(
            model.optimizer,
            ForwardObjectiveFunction(),
            MOI.ScalarAffineFunction{T}(terms, cte),
        )
    end
    return
end

function empty_input_sensitivities!(model::POI.Optimizer)
    empty_input_sensitivities!(model.optimizer)
    return
end

function forward_differentiate!(model::POI.Optimizer{T}) where {T}
    empty_input_sensitivities!(model)
    ctr_types = MOI.get(model, POI.ListOfParametricConstraintTypesPresent())
    for (F, S, P) in ctr_types
        dict = MOI.get(model, POI.DictOfParametricConstraintIndicesAndFunctions{F, S, P}())
        _constraint_set_forward!(model, dict, P)
    end
    obj_type = MOI.get(model, POI.ParametricObjectiveType())
    if obj_type <: POI.ParametricAffineFunction
        _affine_objective_set_forward!(model)
    elseif obj_type <: POI.ParametricQuadraticFunction
        _quadratic_objective_set_forward!(model)
    end
    forward_differentiate!(model.optimizer)
    return
end

# function MOI.set(
#     model::POI.Optimizer,
#     ::ForwardParameter,
#     variable::MOI.VariableIndex,
#     value::Number,
# )
#     if _is_variable(model, variable)
#         error("Trying to set a forward parameter sensitivity for a variable")
#     end
#     parameter = variable
#     sensitivity_data = _get_sensitivity_data(model)
#     sensitivity_data.parameter_input_forward[parameter] = value
#     return
# end

function MOI.set(
    model::POI.Optimizer,
    ::ForwardConstraintSet,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,MOI.Parameter{T}},
    value::Number,
) where {T}
    variable = MOI.VariableIndex(ci.value)
    if _is_variable(model, variable)
        error("Trying to set a forward parameter sensitivity for a variable")
    end
    sensitivity_data = _get_sensitivity_data(model)
    sensitivity_data.parameter_input_forward[variable] = value
    return
end

function MOI.get(
    model::POI.Optimizer,
    attr::ForwardVariablePrimal,
    variable::MOI.VariableIndex,
)
    if _is_parameter(model, variable)
        error("Trying to get a forward variable sensitivity for a parameter")
    end
    return MOI.get(model.optimizer, attr, model.variables[variable])
end

# reverse mode

function _constraint_get_reverse!(
    model::POI.Optimizer{T},
    affine_constraint_cache_dict,
    ::Type{P},
) where {T, P <: POI.ParametricAffineFunction}
    sensitivity_data = _get_sensitivity_data(model)
    for (inner_ci, pf) in affine_constraint_cache_dict
        terms = POI.affine_parameter_terms(pf)
        if isempty(terms)
            continue
        end
        grad_pf_cte = MOI.constant(
            MOI.get(
                model.optimizer,
                ReverseConstraintFunction(),
                inner_ci,
            ),
        )
        for term in terms
            p = term.variable
            value = get!(sensitivity_data.parameter_output_backward, p, 0.0)
            sensitivity_data.parameter_output_backward[p] =
                value + term.coefficient * grad_pf_cte
        end
    end
    return
end

function _constraint_get_reverse!(
    model::POI.Optimizer{T},
    vector_affine_constraint_cache_dict,
    ::Type{P},
) where {T, P <: POI.ParametricVectorAffineFunction}
    sensitivity_data = _get_sensitivity_data(model)
    for (inner_ci, pf) in vector_affine_constraint_cache_dict
        terms = POI.vector_affine_parameter_terms(pf)
        if isempty(terms)
            continue
        end
        grad_pf_cte = MOI.constant(
            MOI.get(
                model.optimizer,
                ReverseConstraintFunction(),
                inner_ci,
            ),
        )
        for term in terms
            p = term.scalar_term.variable
            value = get!(sensitivity_data.parameter_output_backward, p, 0.0)
            sensitivity_data.parameter_output_backward[p] =
                value +
                term.scalar_term.coefficient * grad_pf_cte[term.output_index]
        end
    end
    return
end

function _constraint_get_reverse!(
    model::POI.Optimizer{T},
    quadratic_constraint_cache_dict,
    ::Type{P},
) where {T, P <: POI.ParametricQuadraticFunction}
    sensitivity_data = _get_sensitivity_data(model)
    for (inner_ci, pf) in quadratic_constraint_cache_dict
        p_terms = POI.affine_parameter_terms(pf)
        pp_terms = POI.quadratic_parameter_parameter_terms(pf)
        pv_terms = POI.quadratic_parameter_variable_terms(pf)
        if isempty(p_terms) && isempty(pp_terms) && isempty(pv_terms)
            continue
        end
        grad_pf = MOI.get(
            model.optimizer,
            ReverseConstraintFunction(),
            inner_ci,
        )
        grad_pf_cte = MOI.constant(grad_pf)
        for term in p_terms
            p = term.variable
            value = get!(sensitivity_data.parameter_output_backward, p, 0.0)
            sensitivity_data.parameter_output_backward[p] =
                value + term.coefficient * grad_pf_cte
        end
        for term in pp_terms
            p_1 = term.variable_1
            p_2 = term.variable_2
            value_1 = get!(sensitivity_data.parameter_output_backward, p_1, 0.0)
            value_2 = get!(sensitivity_data.parameter_output_backward, p_2, 0.0)
            # TODO: why there is no factor of 2 here????
            # ANS: probably because it was SET
            sensitivity_data.parameter_output_backward[p_1] =
                value_1 +
                term.coefficient * grad_pf_cte *
                MOI.get(model, MOI.VariablePrimal(), p_2) /
                ifelse(term.variable_1 === term.variable_2, 1, 1)
            sensitivity_data.parameter_output_backward[p_2] =
                value_2 +
                term.coefficient * grad_pf_cte *
                MOI.get(model, MOI.VariablePrimal(), p_1) /
                ifelse(term.variable_1 === term.variable_2, 1, 1)
        end
        for term in pv_terms
            p = term.variable_1
            v = term.variable_2 # check if inner or outer (should be inner)
            value = get!(sensitivity_data.parameter_output_backward, p, 0.0)
            sensitivity_data.parameter_output_backward[p] =
                value + term.coefficient * JuMP.coefficient(grad_pf, v) # * fixed value of the parameter ?
        end
    end
    return
end

function _affine_objective_get_reverse!(model::POI.Optimizer{T}) where {T}
    pf = MOI.get(model, POI.ParametricObjectiveFunction{POI.ParametricAffineFunction{T}}())
    terms = POI.affine_parameter_terms(pf)
    if isempty(terms)
        return
    end
    sensitivity_data = _get_sensitivity_data(model)
    grad_pf = MOI.get(model.optimizer, ReverseObjectiveFunction())
    grad_pf_cte = MOI.constant(grad_pf)
    for term in terms
        p = term.variable
        value = get!(sensitivity_data.parameter_output_backward, p, 0.0)
        sensitivity_data.parameter_output_backward[p] =
            value + term.coefficient * grad_pf_cte
    end
    return
end
function _quadratic_objective_get_reverse!(model::POI.Optimizer{T}) where {T}
    pf = MOI.get(model, POI.ParametricObjectiveFunction{POI.ParametricQuadraticFunction{T}}())
    p_terms = POI.affine_parameter_terms(pf)
    pp_terms = POI.quadratic_parameter_parameter_terms(pf)
    pv_terms = POI.quadratic_parameter_variable_terms(pf)
    if isempty(p_terms) && isempty(pp_terms) && isempty(pv_terms)
        return
    end
    sensitivity_data = _get_sensitivity_data(model)
    grad_pf = MOI.get(model.optimizer, ReverseObjectiveFunction())
    grad_pf_cte = MOI.constant(grad_pf)
    for term in p_terms
        p = term.variable
        value = get!(sensitivity_data.parameter_output_backward, p, 0.0)
        sensitivity_data.parameter_output_backward[p] =
            value + term.coefficient * grad_pf_cte
    end
    for term in pp_terms
        p_1 = term.variable_1
        p_2 = term.variable_2
        value_1 = get!(sensitivity_data.parameter_output_backward, p_1, 0.0)
        value_2 = get!(sensitivity_data.parameter_output_backward, p_2, 0.0)
        sensitivity_data.parameter_output_backward[p_1] =
            value_1 +
            term.coefficient * grad_pf_cte *
            MOI.get(model, MOI.VariablePrimal(), p_2) /
            ifelse(term.variable_1 === term.variable_2, 2, 1)
        sensitivity_data.parameter_output_backward[p_2] =
            value_2 +
            term.coefficient * grad_pf_cte *
            MOI.get(model, MOI.VariablePrimal(), p_1) /
            ifelse(term.variable_1 === term.variable_2, 2, 1)
    end
    for term in pv_terms
        p = term.variable_1
        v = term.variable_2 # check if inner or outer (should be inner)
        value = get!(sensitivity_data.parameter_output_backward, p, 0.0)
        sensitivity_data.parameter_output_backward[p] =
            value + term.coefficient * JuMP.coefficient(grad_pf, v) # * fixed value of the parameter ?
    end
    return
end

function reverse_differentiate!(model::POI.Optimizer)
    reverse_differentiate!(model.optimizer)
    sensitivity_data = _get_sensitivity_data(model)
    empty!(sensitivity_data.parameter_output_backward)
    sizehint!(sensitivity_data.parameter_output_backward, length(model.parameters))
    ctr_types = MOI.get(model, POI.ListOfParametricConstraintTypesPresent())
    for (F, S, P) in ctr_types
        dict = MOI.get(model, POI.DictOfParametricConstraintIndicesAndFunctions{F, S, P}())
        _constraint_get_reverse!(model, dict, P)
    end
    obj_type = MOI.get(model, POI.ParametricObjectiveType())
    if obj_type <: POI.ParametricAffineFunction
        _affine_objective_get_reverse!(model)
    elseif obj_type <: POI.ParametricQuadraticFunction
        _quadratic_objective_get_reverse!(model)
    end
    return
end

function _is_parameter(model::POI.Optimizer{T}, variable::MOI.VariableIndex) where {T}
    MOI.is_valid(model, MOI.ConstraintIndex{MOI.VariableIndex,MOI.Parameter{T}}(variable.value))
end

function _is_variable(model::POI.Optimizer{T}, variable::MOI.VariableIndex) where {T}
    MOI.is_valid(model, variable) && !MOI.is_valid(model, MOI.ConstraintIndex{MOI.VariableIndex,MOI.Parameter{T}}(variable.value))
end

function MOI.set(
    model::POI.Optimizer,
    attr::ReverseVariablePrimal,
    variable::MOI.VariableIndex,
    value::Number,
)
    if _is_parameter(model, variable)
        error("Trying to set a backward variable sensitivity for a parameter")
    end
    MOI.set(model.optimizer, attr, variable, value)
    return
end

# MOI.is_set_by_optimize(::ReverseParameter) = true

# function MOI.get(
#     model::POI.Optimizer,
#     ::ReverseParameter,
#     variable::MOI.VariableIndex,
# )
#     if _is_variable(model, variable)
#         error("Trying to get a backward parameter sensitivity for a variable")
#     end
#     p = variable
#     sensitivity_data = _get_sensitivity_data(model)
#     return get(sensitivity_data.parameter_output_backward, p, 0.0)
# end

MOI.is_set_by_optimize(::ReverseConstraintSet) = true

function MOI.get(
    model::POI.Optimizer,
    ::ReverseConstraintSet,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,MOI.Parameter{T}},
) where {T}
    variable = MOI.VariableIndex(ci.value)
    if _is_variable(model, variable)
        error("Trying to get a backward parameter sensitivity for a variable")
    end
    sensitivity_data = _get_sensitivity_data(model)
    return get(sensitivity_data.parameter_output_backward, variable, 0.0)
end

# extras to handle model_dirty

function MOI.get(
    model::JuMP.Model,
    attr::ReverseConstraintSet,
    var_ref::JuMP.ConstraintRef,
)
    JuMP.check_belongs_to_model(var_ref, model)
    return _moi_get_result(JuMP.backend(model), attr, JuMP.index(var_ref))
end

"""
"""
function set_forward_parameter(
    model::JuMP.Model,
    variable::JuMP.VariableRef,
    value::Number,
)
    MOI.set(model, DiffOpt.ForwardConstraintSet(), ParameterRef(variable), value)
end

"""
"""
function get_reverse_parameter(
    model::JuMP.Model,
    variable::JuMP.VariableRef,
)
    return MOI.get(model, DiffOpt.ReverseConstraintSet(), ParameterRef(variable))
end

"""
"""
function set_reverse_variable(
    model::JuMP.Model,
    variable::JuMP.VariableRef,
    value::Number,
)
    MOI.set(model, DiffOpt.ReverseVariablePrimal(), variable, value)
end

"""
"""
function get_forward_variable(
    model::JuMP.Model,
    variable::JuMP.VariableRef,
)
    return MOI.get(model, DiffOpt.ForwardVariablePrimal(), variable)
end
