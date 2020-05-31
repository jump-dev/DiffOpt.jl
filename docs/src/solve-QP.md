## solving QP primal


```julia
using Random
using MathOptInterface
using Dualization
using OSQP

const MOI = MathOptInterface
const MOIU = MathOptInterface.Utilities;
```


```julia
n = 20 # variable dimension
m = 15 # no of inequality constraints
p = 15; # no of equality constraints
```

## create a non-trivial QP problem
$$\text{min } \frac{1}{2}x^TQx + q^Tx$$
$$\text{s.t.  }Gx <= h$$
   $$Ax = b$$ 


```julia
x̂ = rand(n)
Q = rand(n, n)
Q = Q'*Q # ensure PSD
q = rand(n)
G = rand(m, n)
h = G*x̂ + rand(m)
A = rand(p, n)
b = A*x̂;
```


```julia
model = MOI.instantiate(OSQP.Optimizer, with_bridge_type=Float64)
x = MOI.add_variables(model, n);
```


```julia
# define objective

quad_terms = MOI.ScalarQuadraticTerm{Float64}[]
for i in 1:n
    for j in i:n # indexes (i,j), (j,i) will be mirrored. specify only one kind
        push!(quad_terms, MOI.ScalarQuadraticTerm(Q[i,j],x[i],x[j]))
    end
end

objective_function = MOI.ScalarQuadraticFunction(MOI.ScalarAffineTerm.(q, x),quad_terms,0.0)
MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(), objective_function)
MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
```


```julia
# maintain constrain to index map - will be useful later
constraint_map = Dict()

# add constraints
for i in 1:m
    ci = MOI.add_constraint(model,MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(G[i,:], x), 0.),MOI.LessThan(h[i]))
    constraint_map[ci] = i
end

for i in 1:p
    ci = MOI.add_constraint(model,MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(A[i,:], x), 0.),MOI.EqualTo(b[i]))
    constraint_map[ci] = i
end
```


```julia
MOI.optimize!(model)
```


```julia
@assert MOI.get(model, MOI.TerminationStatus()) in [MOI.LOCALLY_SOLVED, MOI.OPTIMAL]
```


```julia
x̄ = MOI.get(model, MOI.VariablePrimal(), x);
```


```julia
# objective value (predicted vs actual) sanity check
@assert 0.5*x̄'*Q*x̄ + q'*x̄  <= 0.5*x̂'*Q*x̂ + q'*x̂   
```

## find and solve dual problem 

| primal | dual |
|--------|------|
$$\text{min } \frac{1}{2}x^TQx + q^Tx$$  | $$\text{max } -\frac{1}{2}y^TQ^{-1}y - u^Th - v^Tb$$
$$\text{s.t.  }Gx <= h$$ | $$\text{s.t.  } u \geq 0, u \in R^m, v \in R^n$$
   $$Ax = b$$  |    $$y = q + G^Tu + A^Tv$$
  
- Each primal variable becomes a dual constraint
- Each primal constraint becomes a dual variable


```julia
# NOTE: can't use Ipopt
# Ipopt.Optimizer doesn't supports accessing MOI.ObjectiveFunctionType

joint_object    = dualize(model)
dual_model_like = joint_object.dual_model # this is MOI.ModelLike, not an MOI.AbstractOptimizer; can't call optimizer on it
primal_dual_map = joint_object.primal_dual_map;
```


```julia
# copy the dual model objective, constraints, and variables to an optimizer
dual_model = MOI.instantiate(OSQP.Optimizer, with_bridge_type=Float64)
MOI.copy_to(dual_model, dual_model_like)

# solve dual
MOI.optimize!(dual_model);
```


```julia
# check if strong duality holds
@assert abs(MOI.get(model, MOI.ObjectiveValue()) - MOI.get(dual_model, MOI.ObjectiveValue())) <= 1e-1
```

## derive and verify KKT conditions


```julia
is_equality(set::S) where {S<:MOI.AbstractSet} = false
is_equality(set::MOI.EqualTo{T}) where T = true

map = primal_dual_map.primal_con_dual_var;
```

**complimentary slackness**: $$\mu_{i}(G\bar x -h)_i=0 \qquad \text{ where } i=1..m$$


```julia
for con_index in keys(map)
    # NOTE: OSQP.Optimizer doesn't allows access to MOI.ConstraintPrimal
    #       That's why I defined a custom map 
    
    set = MOI.get(model, MOI.ConstraintSet(), con_index)
    μ   = MOI.get(dual_model, MOI.VariablePrimal(), map[con_index][1])
    
    if !is_equality(set)
        # μ[i]*(Gx - h)[i] = 0
        i = constraint_map[con_index]
        
        # println(μ," - ",G[i,:]'*x̄, " - ",h[i])
        # TODO: assertion fails 
        @assert μ*(G[i,:]'*x̄ - h[i]) < 1e-1  
    end
end
```

**primal feasibility**: 
$$(G\bar x -h)_i=0 \qquad \text{ where } i=1..m$$
$$(A\bar x -b)_j=0 \qquad \text{ where } j=1..p$$

**dual feasibility**: 
$$\mu_i \geq 0 \qquad \text{ where } i=1..m$$


```julia
for con_index in keys(map)
    # NOTE: OSQP.Optimizer doesn't allows access to MOI.ConstraintPrimal
    #       That's why I defined a custom map 
    
    set = MOI.get(model, MOI.ConstraintSet(), con_index)
    μ   = MOI.get(dual_model, MOI.VariablePrimal(), map[con_index][1])
    i = constraint_map[con_index]
    
    if is_equality(set)
        # (Ax - h)[i] = 0
        @assert abs(A[i,:]'*x̄ - b[i]) < 1e-2
    else
        # (Gx - h)[i] = 0
        @assert G[i,:]'*x̄ - h[i] < 1e-2
        
        # μ[i] >= 0
        # TODO: assertion fails 
        @assert μ > -1e-2
    end
end
```


```julia

```
