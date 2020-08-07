# Solving LP primal


```julia
using Random
using GLPK
using MathOptInterface
using Dualization

const MOI  = MathOptInterface
const MOIU = MathOptInterface.Utilities;
```


```julia
D = 10  # variable dimension
N = 20; # no of inequality constraints
```

## create a non-trivial LP problem
$$\text{min } c^Tx$$
$$\text{s.t. } Ax \leq b$$
    $$x \geq 0, x \in R^D$$


```julia
s = rand(N)
s = 2*s.-1
λ = max.(-s, 0)
s = max.(s, 0)
x̂ = rand(D)
A = rand(N, D)
b = A*x̂ .+ s
c = -A'*λ;
```


```julia
# can feed dual problem to optimizer like this:
# model = MOI.instantiate(dual_optimizer(GLPK.Optimizer), with_bridge_type=Float64)

model = GLPK.Optimizer()
x = MOI.add_variables(model, D)

# define objective
objective_function = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(c, x), 0.0)
MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), objective_function)
MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
```


```julia
# will be useful later
constraint_indices = []

# set constraints
for i in 1:N
    push!(constraint_indices, MOI.add_constraint(model,MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(A[i,:], x), 0.),MOI.LessThan(b[i])))
end

for i in 1:D
    push!(constraint_indices, MOI.add_constraint(model,MOI.SingleVariable(x[i]),MOI.GreaterThan(0.)))
end
```


```julia
MOI.optimize!(model)
```


```julia
@assert MOI.get(model, MOI.TerminationStatus()) in [MOI.LOCALLY_SOLVED, MOI.OPTIMAL]
```


```julia
x̄ = MOI.get(model, MOI.VariablePrimal(), x);  # solution
```


```julia
@assert abs(c'x̄ - c'x̂) <= 1e-8   # sanity check
```

## find and solve dual problem 

| primal | dual |
|--------|------|
$$\text{min } c^Tx$$  | $$\text{max } b^Ty$$
$$\text{s.t. } Ax \leq b$$ | $$\text{s.t. } A^Ty \geq c$$
    $$x \geq 0$$ | $$y \leq 0$$
  
- Each primal variable becomes a dual constraint
- Each primal constraint becomes a dual variable


```julia
joint_object    = dualize(model)
dual_model_like = joint_object.dual_model # this is MOI.ModelLike, not an MOI.AbstractOptimizer; can't call optimizer on it
primal_dual_map = joint_object.primal_dual_map;
```


```julia
# copy the dual model objective, constraints, and variables to an optimizer
dual_model = GLPK.Optimizer()
MOI.copy_to(dual_model, dual_model_like)

# solve dual
MOI.optimize!(dual_model);
```


```julia
# NOTE: You can obtain components of the dual model individually by -
# dual_objective = dual_model_like.objective  # b'y
# dual_variable_indices = [primal_dual_map.primal_con_dual_var[x][1] for x in constraint_indices]
# dual_constraint_indices = [primal_dual_map.primal_var_dual_con[i] for i in x];

# ŷ = MOI.get(dm, MOI.VariablePrimal(), dual_variable_indices)
```


```julia
# check if strong duality holds
@assert abs(MOI.get(model, MOI.ObjectiveValue()) - MOI.get(dual_model, MOI.ObjectiveValue())) <= 1e-8
```

## derive and verify KKT conditions

**complimentary slackness**: $$\mu_{i}(A\bar x -b)_i=0,\quad \mu_{j+N} \bar x_j =0 \qquad \text{ where } i=1..N, j = 1..D$$


```julia
is_less_than(set::S) where {S<:MOI.AbstractSet} = false
is_less_than(set::MOI.LessThan{T}) where T = true

map = primal_dual_map.primal_con_dual_var
for con_index in keys(map)
    con_value = MOI.get(model, MOI.ConstraintPrimal(), con_index)
    set = MOI.get(model, MOI.ConstraintSet(), con_index)
    μ         = MOI.get(dual_model, MOI.VariablePrimal(), map[con_index][1])
    
    if is_less_than(set)
        # μ[i]*(Ax - b)[i] = 0
        @assert μ*(con_value - set.upper) < 1e-10
    else
        # μ[j]*x[j] = 0
        @assert μ*(con_value - set.lower) < 1e-10
    end
end
```


```julia

```
